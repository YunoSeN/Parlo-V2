import { createRobots, createSitemap, publicOrigin } from "./src/lib/publicPages.ts";
import { parseWordLookup } from "./src/lib/wordLookup.ts";
import "dotenv/config";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import { WebSocketServer, WebSocket } from "ws";
import {
  GoogleGenAI,
  type LiveServerMessage,
  Modality,
  ThinkingLevel,
  Type,
} from "@google/genai";
import {
  attachOptionalAuth,
  type AuthRequest,
  requireAuth,
} from "./src/middleware/auth.ts";
import { adminAuth } from "./src/lib/firebase-admin.ts";
import {
  clampString,
  ConcurrencyGuard,
  createRateLimitMiddleware,
  getClientIp,
  isAllowedWebSocketOrigin,
  sanitizeStringArray,
  SlidingWindowLimiter,
} from "./src/middleware/rateLimit.ts";
import { getOrCreateUser, updateUserWeeklyGoal } from "./src/db/users.ts";
import {
  deleteSavedPhraseRecord,
  listUserCustomScenarios,
  listUserPhrases,
  listUserSessions,
  saveSessionDebriefRecord,
  toggleSavedPhraseMastered,
  upsertCustomScenario,
  upsertPracticeSession,
  upsertSavedPhrase,
} from "./src/db/repository.ts";
import type {
  DebriefReport,
  PracticeSessionRecord,
  SavedPhrase,
  Scenario,
} from "./src/types.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const IS_PRODUCTION = process.env.NODE_ENV === "production" || (!process.env.NODE_ENV && process.env.npm_lifecycle_event === "start");

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY is not configured. Please add your key in Settings > Secrets."
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function formatApiError(err: unknown): { status: number; message: string } {
  const raw = err instanceof Error ? err.message : String(err);
  if (raw.includes("API_KEY_INVALID") || raw.includes("PERMISSION_DENIED") || raw.includes("403")) {
    return {
      status: 403,
      message:
        "The Gemini API key was rejected. Please verify your API key in Settings > Secrets.",
    };
  }
  if (raw.includes("RESOURCE_EXHAUSTED") || raw.includes("429")) {
    return {
      status: 429,
      message:
        "Rate limit reached on the Gemini API. Please wait a moment and try again.",
    };
  }
  if (raw.includes("GEMINI_API_KEY is not configured")) {
    return { status: 500, message: raw };
  }
  return { status: 500, message: raw || "Unexpected Gemini API error." };
}

function safeParseModelJson<T>(rawText: string | undefined, fallbackError: string): T {
  const cleaned = (rawText ?? "")
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
  if (!cleaned) {
    throw new Error(fallbackError);
  }
  try {
    return JSON.parse(cleaned) as T;
  } catch {
    throw new Error(fallbackError);
  }
}

interface LiveStartConfig {
  languageName: string;
  languageNative: string;
  dialectLabel?: string;
  dialectDirective?: string;
  level: string;
  voiceName: string;
  scenarioTitle: string;
  scenarioPrompt: string;
  partnerName: string;
  partnerRole: string;
  correctionStyle: "gentle" | "active" | "immersive";
  speakingPace: "slow" | "natural" | "brisk";
  goals: string[];
}

function buildPartnerSystemInstruction(cfg: LiveStartConfig): string {
  const paceDirective =
    cfg.speakingPace === "slow"
      ? "Speak clearly at a relaxed, unhurried pace with clean articulation and short sentences."
      : cfg.speakingPace === "brisk"
      ? "Speak at an authentic, brisk native speed with natural contractions and colloquialisms."
      : "Speak at a natural, comfortable conversational speed suitable for the learner's level.";

  const correctionDirective =
    cfg.correctionStyle === "active"
      ? "When the learner makes a noticeable grammar or word-choice mistake, gently model or briefly note the natural phrasing in the target language, then immediately continue the conversation."
      : cfg.correctionStyle === "gentle"
      ? "Do not interrupt the flow with lectures. If the learner makes a clear mistake, naturally recast the correct form in your reply so they hear how a native speaker says it."
      : "Stay 100% immersed in character. Never break character to correct grammar; focus purely on a realistic, engaging exchange.";

  const goalsList =
    cfg.goals.length > 0
      ? `Conversation milestones to naturally guide the learner toward:\n${cfg.goals
          .map((g, i) => `${i + 1}. ${g}`)
          .join("\n")}`
      : "";

  return [
    `You are ${cfg.partnerName}, a ${cfg.partnerRole}.`,
    `Language: Speak strictly in ${cfg.languageName} (${cfg.languageNative}) unless the learner explicitly asks for brief help in English.`,
    cfg.dialectDirective
      ? `Regional Variety (${cfg.dialectLabel || "Local"}): ${cfg.dialectDirective}`
      : "",
    `Learner CEFR Proficiency: ${cfg.level}. Match your vocabulary and sentence complexity to ${cfg.level} so the learner feels stretched yet supported.`,
    `Scenario: ${cfg.scenarioTitle}. ${cfg.scenarioPrompt}`,
    paceDirective,
    correctionDirective,
    goalsList,
    "Keep each spoken response concise (1 to 3 sentences) and end with a natural prompt or question so the learner speaks often. Never read out stage directions, markdown, or emojis.",
  ]
    .filter(Boolean)
    .join("\n\n");
}

async function startServer() {
  const app = express();
  app.set("trust proxy", 1);
  app.use(express.json({ limit: "512kb" }));

  // Server-side sliding-window rate limiters & WebSocket concurrency guards
  const rateLimiter = new SlidingWindowLimiter();
  const liveConcurrency = new ConcurrencyGuard();

  const coachRateLimit = createRateLimitMiddleware(rateLimiter, {
    windowMs: 60_000,
    maxGuest: 30,
    maxAuthenticated: 90,
    label: "coaching",
  });
  const chatRateLimit = createRateLimitMiddleware(rateLimiter, {
    windowMs: 60_000,
    maxGuest: 20,
    maxAuthenticated: 60,
    label: "tutor chat",
  });
  const accountRateLimit = createRateLimitMiddleware(rateLimiter, {
    windowMs: 60_000,
    maxGuest: 30,
    maxAuthenticated: 120,
    label: "account sync",
  });

  const siteOrigin = publicOrigin(process.env.APP_URL);
  app.get("/robots.txt", (_req, res) => res.type("text/plain").send(createRobots(siteOrigin)));
  app.get("/sitemap.xml", (_req, res) => {
    if (!siteOrigin) return res.status(404).type("text/plain").send("The public site URL is not configured.");
    res.type("application/xml").send(createSitemap(siteOrigin));
  });

  // Health check
  app.get("/api/health", (_req, res) => {
    res.json({ ok: true, hasApiKey: Boolean(process.env.GEMINI_API_KEY) });
  });

  app.use("/api/me", requireAuth, accountRateLimit);
  app.use("/api/coach", attachOptionalAuth, coachRateLimit);
  app.use("/api/chat", attachOptionalAuth, chatRateLimit);

  // --- Authenticated User Data Routes (Cloud SQL PostgreSQL) ---
  app.get("/api/me/state", async (req: AuthRequest, res) => {
    try {
      const uid = req.user!.uid;
      const email = req.user!.email || `${uid}@users.parlo.local`;
      const displayName = req.user!.name || null;
      const user = await getOrCreateUser(uid, email, displayName);
      const [sessions, phrases, customScenarios] = await Promise.all([
        listUserSessions(user.id),
        listUserPhrases(user.id),
        listUserCustomScenarios(user.id),
      ]);
      res.json({
        weeklySessionGoal: user.weeklySessionGoal,
        sessions,
        phrases,
        customScenarios,
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to load saved account data.";
      res.status(500).json({ error: message });
    }
  });

  app.put("/api/me/goal", async (req: AuthRequest, res) => {
    try {
      const { weeklySessionGoal } = (req.body ?? {}) as { weeklySessionGoal?: number };
      if (
        typeof weeklySessionGoal !== "number" ||
        !Number.isFinite(weeklySessionGoal)
      ) {
        return res.status(400).json({ error: "Valid weeklySessionGoal is required." });
      }
      const uid = req.user!.uid;
      const email = req.user!.email || `${uid}@users.parlo.local`;
      await getOrCreateUser(uid, email, req.user!.name || null);
      const updated = await updateUserWeeklyGoal(uid, weeklySessionGoal);
      res.json({ weeklySessionGoal: updated?.weeklySessionGoal ?? weeklySessionGoal });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to save weekly goal.";
      res.status(500).json({ error: message });
    }
  });

  app.post("/api/me/sessions", async (req: AuthRequest, res) => {
    try {
      const rawRecord = req.body as Partial<PracticeSessionRecord>;
      const id = clampString(rawRecord?.id, 128);
      if (!id || !Array.isArray(rawRecord?.transcript) || rawRecord.transcript.length > 400) {
        return res.status(400).json({ error: "Valid session record is required." });
      }
      const record: PracticeSessionRecord = {
        id,
        date: typeof rawRecord.date === "number" && Number.isFinite(rawRecord.date) ? rawRecord.date : Date.now(),
        languageCode: clampString(rawRecord.languageCode, 16, "es"),
        level: (["A2", "B1", "B2", "C1"].includes(String(rawRecord.level)) ? rawRecord.level : "B1") as PracticeSessionRecord["level"],
        scenarioTitle: clampString(rawRecord.scenarioTitle, 160, "Conversation"),
        partnerName: clampString(rawRecord.partnerName, 80, "Partner"),
        durationSeconds: Math.max(1, Math.min(7200, Math.round(Number(rawRecord.durationSeconds) || 1))),
        userTurnsCount: Math.max(0, Math.min(400, Math.round(Number(rawRecord.userTurnsCount) || 0))),
        partnerTurnsCount: Math.max(0, Math.min(400, Math.round(Number(rawRecord.partnerTurnsCount) || 0))),
        goalsCompleted: Math.max(0, Math.min(20, Math.round(Number(rawRecord.goalsCompleted) || 0))),
        goalsTotal: Math.max(0, Math.min(20, Math.round(Number(rawRecord.goalsTotal) || 0))),
        transcript: rawRecord.transcript
          .filter(
            (t): t is { role: "user" | "partner"; text: string } =>
              Boolean(t) &&
              (t.role === "user" || t.role === "partner") &&
              typeof t.text === "string"
          )
          .slice(0, 400)
          .map((t) => ({ role: t.role, text: clampString(t.text, 2000) })),
        debrief: rawRecord.debrief,
      };
      const user = await getOrCreateUser(
        req.user!.uid,
        req.user!.email || `${req.user!.uid}@users.parlo.local`,
        req.user!.name || null
      );
      await upsertPracticeSession(user.id, record);
      res.status(201).json({ ok: true });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to save practice session.";
      res.status(500).json({ error: message });
    }
  });

  app.put(
    "/api/me/sessions/:clientId/debrief",
    async (req: AuthRequest, res) => {
      try {
        const debrief = req.body as DebriefReport;
        if (!debrief?.summary) {
          return res.status(400).json({ error: "Debrief payload is required." });
        }
        const user = await getOrCreateUser(
          req.user!.uid,
          req.user!.email || `${req.user!.uid}@users.parlo.local`,
          req.user!.name || null
        );
        await saveSessionDebriefRecord(
          user.id,
          String(req.params.clientId),
          debrief
        );
        res.json({ ok: true });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Unable to save debrief.";
        res.status(500).json({ error: message });
      }
    }
  );

  app.post("/api/me/phrases", async (req: AuthRequest, res) => {
    try {
      const rawPhrase = req.body as Partial<SavedPhrase>;
      const id = clampString(rawPhrase?.id, 128);
      const phraseText = clampString(rawPhrase?.phrase, 500);
      if (!id || !phraseText) {
        return res.status(400).json({ error: "Valid phrase payload is required." });
      }
      const phrase: SavedPhrase = {
        id,
        phrase: phraseText,
        translation: clampString(rawPhrase?.translation, 500, ""),
        reading: clampString(rawPhrase?.reading, 300) || undefined,
        languageCode: clampString(rawPhrase?.languageCode, 16, "es"),
        scenarioTitle: clampString(rawPhrase?.scenarioTitle, 160, "Personal Entry"),
        note: clampString(rawPhrase?.note, 600) || undefined,
        mastered: Boolean(rawPhrase?.mastered),
        createdAt:
          typeof rawPhrase?.createdAt === "number" && Number.isFinite(rawPhrase.createdAt)
            ? rawPhrase.createdAt
            : Date.now(),
      };
      const user = await getOrCreateUser(
        req.user!.uid,
        req.user!.email || `${req.user!.uid}@users.parlo.local`,
        req.user!.name || null
      );
      await upsertSavedPhrase(user.id, phrase);
      res.status(201).json({ ok: true });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to save phrase.";
      res.status(500).json({ error: message });
    }
  });

  app.patch(
    "/api/me/phrases/:clientId",
    async (req: AuthRequest, res) => {
      try {
        const { mastered } = req.body as { mastered?: boolean };
        if (typeof mastered !== "boolean") {
          return res.status(400).json({ error: "Boolean mastered field is required." });
        }
        const user = await getOrCreateUser(
          req.user!.uid,
          req.user!.email || `${req.user!.uid}@users.parlo.local`,
          req.user!.name || null
        );
        await toggleSavedPhraseMastered(
          user.id,
          String(req.params.clientId),
          mastered
        );
        res.json({ ok: true });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Unable to update phrase.";
        res.status(500).json({ error: message });
      }
    }
  );

  app.delete(
    "/api/me/phrases/:clientId",
    async (req: AuthRequest, res) => {
      try {
        const user = await getOrCreateUser(
          req.user!.uid,
          req.user!.email || `${req.user!.uid}@users.parlo.local`,
          req.user!.name || null
        );
        await deleteSavedPhraseRecord(user.id, String(req.params.clientId));
        res.json({ ok: true });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Unable to delete phrase.";
        res.status(500).json({ error: message });
      }
    }
  );

  app.post("/api/me/scenarios", async (req: AuthRequest, res) => {
    try {
      const rawSc = req.body as Partial<Scenario>;
      const id = clampString(rawSc?.id, 128);
      const title = clampString(rawSc?.title, 140);
      if (!id || !title) {
        return res.status(400).json({ error: "Valid scenario payload is required." });
      }
      const scenario: Scenario = {
        id,
        title,
        city: clampString(rawSc?.city, 80, "Custom"),
        category: clampString(rawSc?.category, 60, "Custom"),
        durationLabel: clampString(rawSc?.durationLabel, 40, "Flexible"),
        defaultLanguageCode: clampString(rawSc?.defaultLanguageCode, 16, "es"),
        recommendedLevel: (["A2", "B1", "B2", "C1"].includes(String(rawSc?.recommendedLevel))
          ? rawSc?.recommendedLevel
          : "B1") as Scenario["recommendedLevel"],
        partnerName: clampString(rawSc?.partnerName, 80, "Partner"),
        partnerRole: clampString(rawSc?.partnerRole, 120, "Conversation Partner"),
        defaultVoice: (["Zephyr", "Kore", "Puck", "Charon", "Fenrir"].includes(String(rawSc?.defaultVoice))
          ? rawSc?.defaultVoice
          : "Zephyr") as Scenario["defaultVoice"],
        fallbackGradient: clampString(
          rawSc?.fallbackGradient,
          120,
          "from-slate-800 via-blue-950 to-slate-900"
        ),
        summary: clampString(rawSc?.summary, 400, ""),
        promptContext: clampString(rawSc?.promptContext, 1200, ""),
        goals: sanitizeStringArray(rawSc?.goals, 8, 180),
        starterPhrases: {},
        isCustom: true,
      };
      const user = await getOrCreateUser(
        req.user!.uid,
        req.user!.email || `${req.user!.uid}@users.parlo.local`,
        req.user!.name || null
      );
      await upsertCustomScenario(user.id, scenario);
      res.status(201).json({ ok: true });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to save custom scenario.";
      res.status(500).json({ error: message });
    }
  });

  // 1. Live Turn Coaching & Grammar Feedback
  app.post("/api/coach/feedback", async (req, res) => {
    try {
      const rawBody = (req.body ?? {}) as {
        utterance?: string;
        languageName?: string;
        dialectLabel?: string;
        dialectDirective?: string;
        level?: string;
        scenarioTitle?: string;
        recentTurns?: { role: string; text: string }[];
        goals?: string[];
      };

      const utterance = clampString(rawBody.utterance, 1200);
      if (!utterance) {
        return res.status(400).json({ error: "Utterance is required." });
      }
      const languageName = clampString(rawBody.languageName, 40, "Spanish");
      const dialectLabel = clampString(rawBody.dialectLabel, 80);
      const dialectDirective = clampString(rawBody.dialectDirective, 260);
      const level = clampString(rawBody.level, 8, "B1");
      const scenarioTitle = clampString(rawBody.scenarioTitle, 140, "Conversation");
      const recentTurns = Array.isArray(rawBody.recentTurns) ? rawBody.recentTurns : [];
      const goals = sanitizeStringArray(rawBody.goals, 8, 180);

      const ai = getGeminiClient();
      const contextText = recentTurns
        .slice(-6)
        .map(
          (t) =>
            `${t.role === "user" ? "Learner" : "Partner"}: ${clampString(t.text, 500)}`
        )
        .join("\n");

      const goalsPrompt =
        goals.length > 0
          ? `Scenario goals (0-indexed):\n${goals
              .map((g, idx) => `[${idx}] ${g}`)
              .join("\n")}`
          : "No scenario goals provided.";

      const dialectPromptLine = dialectLabel
        ? ` (${dialectLabel} regional style${dialectDirective ? `: ${dialectDirective}` : ""})`
        : "";

      const response = await ai.models.generateContent({
        model: "gemini-4-hydrogen",
        contents: `Target language: ${languageName}${dialectPromptLine} (CEFR ${level})
Scenario: ${scenarioTitle}
Recent conversation:
${contextText}

Latest learner utterance to evaluate:
"${utterance}"

${goalsPrompt}

Evaluate the learner's utterance carefully. If it is already natural and accurate, set isNatural to true and return the utterance (or a slightly more idiomatic alternative) in corrected. Keep explanation and tip concise, warm, and actionable in English. Also identify any 0-based goal indices clearly accomplished by the learner so far in the conversation.`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              isNatural: {
                type: Type.BOOLEAN,
                description:
                  "True if the utterance is grammatically sound and natural for the learner's level.",
              },
              corrected: {
                type: Type.STRING,
                description:
                  "The most natural target-language phrasing of what the learner intended to say.",
              },
              explanation: {
                type: Type.STRING,
                description:
                  "Concise 1-2 sentence explanation in English of what worked well or what changed.",
              },
              tip: {
                type: Type.STRING,
                description:
                  "One short practical speaking or nuance tip in English.",
              },
              keyPhrase: {
                type: Type.OBJECT,
                properties: {
                  phrase: {
                    type: Type.STRING,
                    description: "A useful expression in the target language worth saving.",
                  },
                  translation: {
                    type: Type.STRING,
                    description: "Natural English meaning.",
                  },
                  reading: {
                    type: Type.STRING,
                    description:
                      "Romanization or pronunciation guide if non-Latin script, otherwise empty string.",
                  },
                },
                required: ["phrase", "translation"],
              },
              completedGoalIndices: {
                type: Type.ARRAY,
                items: { type: Type.INTEGER },
                description:
                  "0-based indices of goals accomplished by the learner in the conversation.",
              },
            },
            required: [
              "isNatural",
              "corrected",
              "explanation",
              "tip",
              "completedGoalIndices",
            ],
          },
        },
      });

      const parsed = safeParseModelJson<Record<string, unknown>>(
        response.text,
        "Coach feedback returned an invalid format. Please try again."
      );
      res.json(parsed);
    } catch (err) {
      const { status, message } = formatApiError(err);
      res.status(status).json({ error: message });
    }
  });

  // 2. Turn Translation & Breakdown
  app.post("/api/coach/translate", async (req, res) => {
    try {
      const rawBody = (req.body ?? {}) as {
        text?: string;
        languageName?: string;
      };
      const text = clampString(rawBody.text, 1500);
      const languageName = clampString(rawBody.languageName, 40, "Spanish");
      if (!text) {
        return res.status(400).json({ error: "Text is required." });
      }

      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: "gemini-4-hydrogen",
        contents: `Translate this ${languageName} spoken utterance into natural English, include a brief note on any idiomatic phrasing or tone, and provide romanization/pronunciation reading if the script is non-Latin (otherwise leave reading empty).\n\nUtterance: "${text}"`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              english: {
                type: Type.STRING,
                description: "Natural English translation.",
              },
              literalNote: {
                type: Type.STRING,
                description:
                  "Brief 1-sentence note on literal meaning, register, or useful vocabulary.",
              },
              reading: {
                type: Type.STRING,
                description:
                  "Romanization (e.g., romaji, pinyin, revised romanization) if non-Latin script, otherwise empty string.",
              },
            },
            required: ["english", "literalNote"],
          },
        },
      });

      const parsed = safeParseModelJson<Record<string, unknown>>(
        response.text,
        "Translation returned an invalid format. Please try again."
      );
      res.json(parsed);
    } catch (err) {
      const { status, message } = formatApiError(err);
      res.status(status).json({ error: message });
    }
  });

  // 3. Contextual Reply Hints
  app.post("/api/coach/hints", async (req, res) => {
    try {
      const rawBody = (req.body ?? {}) as {
        languageName?: string;
        dialectLabel?: string;
        dialectDirective?: string;
        level?: string;
        scenarioTitle?: string;
        scenarioPrompt?: string;
        recentTurns?: { role: string; text: string }[];
      };
      const languageName = clampString(rawBody.languageName, 40, "Spanish");
      const dialectLabel = clampString(rawBody.dialectLabel, 80);
      const dialectDirective = clampString(rawBody.dialectDirective, 260);
      const level = clampString(rawBody.level, 8, "B1");
      const scenarioTitle = clampString(rawBody.scenarioTitle, 140, "Conversation");
      const scenarioPrompt = clampString(rawBody.scenarioPrompt, 800, "");
      const recentTurns = Array.isArray(rawBody.recentTurns) ? rawBody.recentTurns : [];

      const ai = getGeminiClient();
      const transcriptText =
        recentTurns.length > 0
          ? recentTurns
              .slice(-6)
              .map(
                (t) =>
                  `${t.role === "user" ? "Learner" : "Partner"}: ${clampString(t.text, 500)}`
              )
              .join("\n")
          : "The conversation is just starting.";

      const dialectCue = dialectLabel
        ? ` (${dialectLabel} variety${dialectDirective ? `: ${dialectDirective}` : ""})`
        : "";

      const response = await ai.models.generateContent({
        model: "gemini-4-hydrogen",
        contents: `Generate 3 natural, distinct reply options the learner can say next in ${languageName}${dialectCue} (CEFR ${level}).
Scenario: ${scenarioTitle} — ${scenarioPrompt}
Conversation so far:
${transcriptText}

Make each option fit a distinct register or direction (e.g., Polite, Casual, Curious follow-up). Include romanization/reading only if ${languageName} uses a non-Latin script.`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              suggestions: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    text: {
                      type: Type.STRING,
                      description: "Natural reply in the target language.",
                    },
                    reading: {
                      type: Type.STRING,
                      description:
                        "Pronunciation/romanization if non-Latin script, else empty string.",
                    },
                    translation: {
                      type: Type.STRING,
                      description: "Natural English translation.",
                    },
                    register: {
                      type: Type.STRING,
                      description: "Short 1-2 word tone label like Polite, Warm, Direct, Curious.",
                    },
                  },
                  required: ["text", "translation", "register"],
                },
              },
            },
            required: ["suggestions"],
          },
        },
      });

      const parsed = safeParseModelJson<Record<string, unknown>>(
        response.text,
        "Reply suggestions returned an invalid format. Please try again."
      );
      res.json(parsed);
    } catch (err) {
      const { status, message } = formatApiError(err);
      res.status(status).json({ error: message });
    }
  });

  // 4. Post-Session Debrief Report
  app.post("/api/coach/debrief", async (req, res) => {
    try {
      const rawBody = (req.body ?? {}) as {
        languageName?: string;
        level?: string;
        scenarioTitle?: string;
        transcript?: { role: string; text: string }[];
      };
      const languageName = clampString(rawBody.languageName, 40, "Spanish");
      const level = clampString(rawBody.level, 8, "B1");
      const scenarioTitle = clampString(rawBody.scenarioTitle, 140, "Conversation");
      const transcript = Array.isArray(rawBody.transcript) ? rawBody.transcript.slice(0, 120) : [];

      if (!transcript.length) {
        return res
          .status(400)
          .json({ error: "A conversation transcript is required for review." });
      }

      const ai = getGeminiClient();
      const formatted = transcript
        .map(
          (t) =>
            `${t.role === "user" ? "Learner" : "Partner"}: ${clampString(t.text, 800)}`
        )
        .join("\n");

      const response = await ai.models.generateContent({
        model: "gemini-4-hydrogen",
        contents: `Review this completed ${languageName} (CEFR ${level}) speaking practice session in the scenario "${scenarioTitle}".

Transcript:
${formatted}

Provide a clear, encouraging, concrete coaching debrief in English with specific examples drawn strictly from the transcript above. Do not invent arbitrary numeric scores.`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              summary: {
                type: Type.STRING,
                description: "2-3 sentence overview of how the conversation went.",
              },
              strengths: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: "2-3 specific things the learner did well.",
              },
              focusAreas: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    original: {
                      type: Type.STRING,
                      description: "What the learner actually said.",
                    },
                    improved: {
                      type: Type.STRING,
                      description: "More natural target-language phrasing.",
                    },
                    note: {
                      type: Type.STRING,
                      description: "Short explanation why.",
                    },
                  },
                  required: ["original", "improved", "note"],
                },
              },
              vocabularyHighlights: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    term: { type: Type.STRING },
                    translation: { type: Type.STRING },
                    context: { type: Type.STRING },
                  },
                  required: ["term", "translation", "context"],
                },
              },
              nextStep: {
                type: Type.STRING,
                description: "One concrete practice focus for the next session.",
              },
            },
            required: [
              "summary",
              "strengths",
              "focusAreas",
              "vocabularyHighlights",
              "nextStep",
            ],
          },
        },
      });

      const parsed = safeParseModelJson<Record<string, unknown>>(
        response.text,
        "Debrief report returned an invalid format. Please try again."
      );
      res.json(parsed);
    } catch (err) {
      const { status, message } = formatApiError(err);
      res.status(status).json({ error: message });
    }
  });

  // 5. Click-Any-Word Cultural Dictionary Breakdown
  app.post("/api/coach/word", async (req, res) => {
    try {
      const rawBody = (req.body ?? {}) as {
        word?: string;
        sentenceContext?: string;
        languageName?: string;
        dialectLabel?: string;
        level?: string;
      };

      const trimmedWord = clampString(rawBody.word, 80);
      if (!trimmedWord) {
        return res.status(400).json({ error: "A word is required for lookup." });
      }
      const sentenceContext = clampString(rawBody.sentenceContext, 500, trimmedWord);
      const languageName = clampString(rawBody.languageName, 40, "Spanish");
      const dialectLabel = clampString(rawBody.dialectLabel, 80);
      const level = clampString(rawBody.level, 8, "B1");

      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: "gemini-4-hydrogen",
        contents: `Analyze the target word "${trimmedWord}" in ${languageName}${dialectLabel ? ` (${dialectLabel} regional usage)` : ""} for a CEFR ${level} learner.
Sentence context: "${sentenceContext}"

Return its dictionary base/root form, part of speech, pronunciation/romanization reading (if non-Latin script or tricky stress, otherwise empty string), crisp contextual English meaning, an optional short usageNote explaining the grammar or idiom, one short natural example sentence in ${languageName}, and an English translation of that example sentence. Include a dialectNote only if a specific, well-established regional or register distinction matters in this sentence. Otherwise return an empty string. Omit speculative origins or generalizations about speakers.`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              word: { type: Type.STRING },
              baseForm: { type: Type.STRING },
              partOfSpeech: { type: Type.STRING },
              reading: { type: Type.STRING },
              meaning: { type: Type.STRING },
              usageNote: { type: Type.STRING },
              dialectNote: { type: Type.STRING },
              example: { type: Type.STRING },
              exampleTranslation: { type: Type.STRING },
            },
            required: [
              "word",
              "baseForm",
              "partOfSpeech",
              "meaning",
              "usageNote",
              "dialectNote",
              "example",
              "exampleTranslation",
            ],
          },
        },
      });

      const parsed = safeParseModelJson<Record<string, unknown>>(
        response.text,
        "Word breakdown returned an invalid format. Please try again."
      );
      res.json(parseWordLookup(parsed));
    } catch (err) {
      const { status, message } = formatApiError(err);
      res.status(status).json({ error: message });
    }
  });

  // -------------------------------------------------------------------------
  // Multi-turn Gemini Language Tutor Chatbot (with role-based system instructions)
  // -------------------------------------------------------------------------
  app.post("/api/chat", async (req, res) => {
    try {
      const {
        history = [],
        message,
        roleId = "conversation",
        speedMode = "balanced",
        languageName = "Spanish",
        languageNative = "Español",
        dialectLabel = "",
        dialectDirective = "",
        level = "B1",
        cultureAtelier = "Madrid Atelier",
        scenarioTitle = "Everyday Conversation",
      } = (req.body ?? {}) as {
        history?: { role: "user" | "model"; text: string }[];
        message?: string;
        roleId?: "conversation" | "grammar" | "culture" | "phrases";
        speedMode?: "balanced" | "fast";
        languageName?: string;
        languageNative?: string;
        dialectLabel?: string;
        dialectDirective?: string;
        level?: string;
        cultureAtelier?: string;
        scenarioTitle?: string;
      };

      const trimmedMessage = clampString(message, 2000);
      if (!trimmedMessage) {
        return res.status(400).json({ error: "A message is required." });
      }

      const safeLanguageName = clampString(languageName, 40, "Spanish");
      const safeLanguageNative = clampString(languageNative, 40, "Español");
      const safeDialectLabel = clampString(dialectLabel, 80);
      const safeDialectDirective = clampString(dialectDirective, 260);
      const safeLevel = clampString(level, 8, "B1");
      const safeAtelier = clampString(cultureAtelier, 80, "Madrid Atelier");
      const safeScenarioTitle = clampString(scenarioTitle, 140, "Everyday Conversation");

      const dialectSuffix = safeDialectDirective
        ? `\nRegional style (${safeDialectLabel || "Local"}): ${safeDialectDirective}`
        : "";

      const roleInstructions: Record<string, string> = {
        conversation: `You are Parlo's warm, articulate native-speaking conversation partner for ${safeLanguageName} (${safeLanguageNative}), grounded in the "${safeAtelier}" aesthetic and the scenario "${safeScenarioTitle}".${dialectSuffix}
Respond primarily in natural ${safeLanguageName} calibrated to CEFR ${safeLevel}. Keep turns concise (2-4 sentences) and end with a natural follow-up question to keep the dialogue flowing. If the learner writes in English or seems stuck, gently bridge back into ${safeLanguageName} with a brief English gloss in parentheses only where genuinely helpful. For Japanese, Mandarin, or Korean, include a brief pronunciation reading in parentheses for uncommon words.`,
        grammar: `You are Parlo's precision ${safeLanguageName} (${safeLanguageNative}) grammar and phrasing coach for a CEFR ${safeLevel} learner practicing "${safeScenarioTitle}".${dialectSuffix}
Explain nuances clearly in concise English while quoting authentic ${safeLanguageName} examples. When the learner shares a sentence, highlight: (1) a more natural native phrasing, (2) why it sounds better (register, idiom, or tense), and (3) one quick parallel example they can reuse right away. Keep answers scannable and encouraging.`,
        culture: `You are Parlo's cultural etiquette and local life concierge for ${safeLanguageName} (${safeLanguageNative}) — "${safeAtelier}".${dialectSuffix}
Share vivid, practical insights on social etiquette, politeness registers, regional expressions, cafe/table customs, and unspoken cultural cues connected to "${safeScenarioTitle}" and CEFR ${safeLevel} communication. Always include 1-2 authentic ${safeLanguageName} expressions with English translations so the learner can sound effortlessly respectful and local.`,
        phrases: `You are Parlo's phrase architect and pronunciation mentor for ${safeLanguageName} (${safeLanguageNative}) at CEFR ${safeLevel}.${dialectSuffix}
Give ready-to-speak phrases tailored to "${safeScenarioTitle}" or whatever the learner asks for. For each phrase, provide: the exact ${safeLanguageName} expression, pronunciation/reading guidance (especially for Japanese, Mandarin, Korean, or tricky liaisons), a natural English translation, and whether it fits casual or polite settings. Keep responses crisp and practical.`,
      };

      const systemInstruction =
        roleInstructions[roleId] ?? roleInstructions.conversation;

      const sanitizedHistory = Array.isArray(history)
        ? history
            .filter(
              (item) =>
                item &&
                (item.role === "user" || item.role === "model") &&
                typeof item.text === "string" &&
                item.text.trim().length > 0
            )
            .slice(-24)
            .map((item) => ({
              role: item.role,
              parts: [{ text: clampString(item.text, 2000) }],
            }))
        : [];

      const model =
        speedMode === "fast" ? "gemini-3.1-flash-lite" : "gemini-4-hydrogen";

      const ai = getGeminiClient();
      const chat = ai.chats.create({
        model,
        history: sanitizedHistory,
        config: {
          systemInstruction,
          ...(model === "gemini-4-hydrogen"
            ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
            : {}),
        },
      });

      const response = await chat.sendMessage({ message: trimmedMessage });
      const reply = (response.text ?? "").trim();

      if (!reply) {
        return res
          .status(502)
          .json({ error: "Gemini returned an empty response. Please retry." });
      }

      res.json({ reply, model });
    } catch (err) {
      const { status, message } = formatApiError(err);
      res.status(status).json({ error: message });
    }
  });

  // Frontend mounting (Vite middleware in dev, dist in prod)
  if (!IS_PRODUCTION) {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("/", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  const httpServer = http.createServer(app);
  const wss = new WebSocketServer({
    noServer: true,
    maxPayload: 256 * 1024, // 256 KB hard cap per WebSocket frame
  });

  const LIVE_MAX_SESSION_MS = 15 * 60 * 1000; // 15-minute session cap
  const LIVE_IDLE_TIMEOUT_MS = 3 * 60 * 1000; // 3-minute idle timeout
  const LIVE_MAX_CONCURRENT_PER_CLIENT = 2;

  httpServer.on("upgrade", (request, socket, head) => {
    const url = new URL(request.url ?? "/", `http://${request.headers.host || "localhost"}`);
    if (url.pathname !== "/api/live") {
      socket.destroy();
      return;
    }

    if (!isAllowedWebSocketOrigin(request.headers.origin, request.headers.host)) {
      socket.write("HTTP/1.1 403 Forbidden\r\n\r\n");
      socket.destroy();
      return;
    }

    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit("connection", ws, request);
    });
  });

  wss.on("connection", (clientWs: WebSocket, request: http.IncomingMessage) => {
    const clientIp = getClientIp(request);
    let concurrencyKey = `live:ip:${clientIp}`;
    if (!liveConcurrency.acquire(concurrencyKey, LIVE_MAX_CONCURRENT_PER_CLIENT)) {
      if (clientWs.readyState === WebSocket.OPEN) {
        clientWs.send(
          JSON.stringify({
            type: "error",
            message:
              "You already have active voice sessions open. Close another tab or end your current session first.",
          })
        );
      }
      clientWs.close(1008, "Concurrent live session limit reached");
      return;
    }

    let liveSession: Awaited<ReturnType<GoogleGenAI["live"]["connect"]>> | null = null;
    let isClosing = false;
    let releasedConcurrency = false;
    let idleTimer: ReturnType<typeof setTimeout> | null = null;
    let maxDurationTimer: ReturnType<typeof setTimeout> | null = null;
    let audioFramesInWindow = 0;
    let audioWindowStart = Date.now();

    const releaseResources = () => {
      if (idleTimer) {
        clearTimeout(idleTimer);
        idleTimer = null;
      }
      if (maxDurationTimer) {
        clearTimeout(maxDurationTimer);
        maxDurationTimer = null;
      }
      if (!releasedConcurrency) {
        releasedConcurrency = true;
        liveConcurrency.release(concurrencyKey);
      }
    };

    const safeSend = (payload: Record<string, unknown>) => {
      if (clientWs.readyState === WebSocket.OPEN) {
        clientWs.send(JSON.stringify(payload));
      }
    };

    const closeLiveSession = () => {
      if (liveSession) {
        try {
          liveSession.close();
        } catch {
          // ignore close errors
        }
        liveSession = null;
      }
    };

    const resetIdleTimer = () => {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        isClosing = true;
        closeLiveSession();
        safeSend({
          type: "error",
          message:
            "Voice session timed out after 3 minutes of inactivity. Your transcript has been preserved.",
        });
        try {
          clientWs.close(1000, "Idle timeout");
        } catch {
          // ignore
        }
      }, LIVE_IDLE_TIMEOUT_MS);
    };

    resetIdleTimer();

    clientWs.on("message", async (raw) => {
      let msg: {
        type?: string;
        authToken?: string;
        config?: LiveStartConfig;
        data?: string;
        text?: string;
      };
      try {
        msg = JSON.parse(raw.toString());
      } catch {
        return;
      }

      if (msg.type === "start" && msg.config) {
        isClosing = false;
        closeLiveSession();

        // Optional Firebase token in encrypted WS start payload (never in URL query string)
        let verifiedUid: string | null = null;
        if (typeof msg.authToken === "string" && msg.authToken.trim().length > 0) {
          try {
            const decoded = await adminAuth.verifyIdToken(msg.authToken.trim());
            verifiedUid = decoded.uid;
          } catch {
            // Fall back to guest IP quota
          }
        }

        const startLimitKey = verifiedUid
          ? `live-start:uid:${verifiedUid}`
          : `live-start:ip:${clientIp}`;
        const startLimit = rateLimiter.check(
          startLimitKey,
          verifiedUid ? 30 : 12,
          10 * 60_000
        );
        if (!startLimit.allowed) {
          safeSend({
            type: "error",
            message: verifiedUid
              ? `Voice session start limit reached. Please wait ${startLimit.retryAfterSeconds}s before starting again.`
              : `Guest voice preview limit reached. Wait ${startLimit.retryAfterSeconds}s or sign in with Google for higher limits.`,
          });
          return;
        }

        const sanitizedConfig: LiveStartConfig = {
          languageName: clampString(msg.config.languageName, 40, "Spanish"),
          languageNative: clampString(msg.config.languageNative, 40, "Español"),
          dialectLabel: clampString(msg.config.dialectLabel, 80) || undefined,
          dialectDirective:
            clampString(msg.config.dialectDirective, 260) || undefined,
          level: ["A2", "B1", "B2", "C1"].includes(msg.config.level)
            ? msg.config.level
            : "B1",
          voiceName: msg.config.voiceName,
          scenarioTitle: clampString(
            msg.config.scenarioTitle,
            140,
            "Everyday Conversation"
          ),
          scenarioPrompt: clampString(msg.config.scenarioPrompt, 1200, ""),
          partnerName: clampString(msg.config.partnerName, 60, "Partner"),
          partnerRole: clampString(
            msg.config.partnerRole,
            100,
            "Conversation Partner"
          ),
          correctionStyle: ["gentle", "active", "immersive"].includes(
            msg.config.correctionStyle
          )
            ? msg.config.correctionStyle
            : "gentle",
          speakingPace: ["slow", "natural", "brisk"].includes(
            msg.config.speakingPace
          )
            ? msg.config.speakingPace
            : "natural",
          goals: sanitizeStringArray(msg.config.goals, 8, 180),
        };

        try {
          const ai = getGeminiClient();
          const systemInstruction = buildPartnerSystemInstruction(sanitizedConfig);
          const allowedVoices = new Set([
            "Zephyr",
            "Kore",
            "Puck",
            "Charon",
            "Fenrir",
          ]);
          const voiceName = allowedVoices.has(sanitizedConfig.voiceName)
            ? sanitizedConfig.voiceName
            : "Zephyr";

          const session = await ai.live.connect({
            model: "gemini-3.8-live",
            config: {
              responseModalities: [Modality.AUDIO],
              speechConfig: {
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName },
                },
              },
              systemInstruction,
              inputAudioTranscription: {},
              outputAudioTranscription: {},
            },
            callbacks: {
              onopen: () => {
                safeSend({ type: "ready" });
              },
              onmessage: (message: LiveServerMessage) => {
                resetIdleTimer();
                const content = message.serverContent;
                if (!content) return;

                if (content.interrupted) {
                  safeSend({ type: "interrupted" });
                }

                if (content.inputTranscription?.text) {
                  safeSend({
                    type: "inputTranscript",
                    text: content.inputTranscription.text,
                  });
                }

                if (content.outputTranscription?.text) {
                  safeSend({
                    type: "outputTranscript",
                    text: content.outputTranscription.text,
                  });
                }

                for (const part of content.modelTurn?.parts ?? []) {
                  if (part.inlineData?.data) {
                    safeSend({
                      type: "audio",
                      data: part.inlineData.data,
                    });
                  }
                }

                if (content.turnComplete) {
                  safeSend({ type: "turnComplete" });
                }
              },
              onerror: (e: ErrorEvent) => {
                safeSend({
                  type: "error",
                  message:
                    e?.message ||
                    "Connection to the Gemini Live voice session encountered an error.",
                });
              },
              onclose: (e: CloseEvent) => {
                liveSession = null;
                if (!isClosing) {
                  safeSend({
                    type: "closed",
                    reason: e?.reason || "",
                  });
                }
              },
            },
          });

          if (isClosing || clientWs.readyState !== WebSocket.OPEN) {
            session.close();
            return;
          }

          liveSession = session;
          resetIdleTimer();
          if (maxDurationTimer) clearTimeout(maxDurationTimer);
          maxDurationTimer = setTimeout(() => {
            isClosing = true;
            closeLiveSession();
            safeSend({
              type: "error",
              message:
                "The 15-minute session has ended. You can start a new conversation.",
            });
            try {
              clientWs.close(1000, "Max session duration reached");
            } catch {
              // ignore
            }
          }, LIVE_MAX_SESSION_MS);

          // Emit ready right after connect resolves in case onopen fired before handler attachment
          safeSend({ type: "ready" });
        } catch (err) {
          const { message } = formatApiError(err);
          safeSend({ type: "error", message });
        }
        return;
      }

      if (msg.type === "audio" && typeof msg.data === "string" && liveSession) {
        // Bound frame size and rate (~35 chunks/sec max; normal 4096@16kHz is ~4/sec)
        if (msg.data.length > 65_536) return;
        const now = Date.now();
        if (now - audioWindowStart > 1000) {
          audioWindowStart = now;
          audioFramesInWindow = 0;
        }
        audioFramesInWindow += 1;
        if (audioFramesInWindow > 35) return;

        resetIdleTimer();
        try {
          liveSession.sendRealtimeInput({
            audio: {
              data: msg.data,
              mimeType: "audio/pcm;rate=16000",
            },
          });
        } catch {
          // ignore transient send errors on closing socket
        }
        return;
      }

      if (msg.type === "text" && typeof msg.text === "string" && liveSession) {
        const cleanedText = clampString(msg.text, 1200);
        if (!cleanedText) return;
        resetIdleTimer();
        try {
          liveSession.sendRealtimeInput({ text: cleanedText });
        } catch (err) {
          const { message } = formatApiError(err);
          safeSend({ type: "error", message });
        }
        return;
      }

      if (msg.type === "promptGreeting" && liveSession) {
        resetIdleTimer();
        try {
          liveSession.sendRealtimeInput({
            text: "Greet the learner warmly in character in the target language to open our conversation.",
          });
        } catch {
          // ignore
        }
        return;
      }

      if (msg.type === "stop") {
        isClosing = true;
        closeLiveSession();
        safeSend({ type: "closed", reason: "Session ended by user." });
      }
    });

    clientWs.on("close", () => {
      isClosing = true;
      closeLiveSession();
      releaseResources();
    });

    clientWs.on("error", () => {
      isClosing = true;
      closeLiveSession();
      releaseResources();
    });
  });

  httpServer.listen(PORT, process.env.HOST || "0.0.0.0", () => {
    console.log(`Parlo server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
