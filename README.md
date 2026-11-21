# Parlo

Parlo is a conversational language partner for Spanish, French, Japanese, German, Italian, Mandarin, Korean and Portuguese. In Practice, choose a language, start a voice session or try an expression. Open **Why this wording?** for context, hide the answer to practise recall, and save expressions to **Words** for spaced review. In **Words → Flashcards**, recall the expression from its meaning or choose the original expression-first view. Interactive rooms are available under **3D Rooms**.

## Run locally

Use Node.js 24. The production entry point and tests were exercised with Node 24.19.0. The supplied Bun lockfile records dependency resolution. In a fresh checkout, run `bun install --frozen-lockfile` with a compatible Bun version, or use `npm install` if using npm. A fresh network install was outside this review. In the supplied Windows working folder, the dependencies are already available through a local directory junction.

```sh
npm run dev
```

Open `http://localhost:3000`. Guest phrase practice and review use browser storage. The French invitation guide at `/learn/french/ca-me-ferait-plaisir.html` also works with JavaScript disabled. Its practice link opens the matching phrase in the Studio.

## Optional services

- Set `GEMINI_API_KEY` in the server process environment or a local `.env` to use Gemini Live and model coaching. Leaving it unset lets you use the authored practice material. Browser audio playback depends on the voices installed on the device.
- Firebase client settings are in `firebase-applet-config.json`. Account sync uses Firebase-authenticated routes and the existing PostgreSQL schema in `src/db/schema.ts`. The database pool reads `SQL_HOST`, `SQL_USER`, `SQL_PASSWORD` and `SQL_DB_NAME`. Guest practice requires none of these. The review used local data and simulated service responses.
- Set `APP_URL` to the public HTTP(S) origin when building or serving a published copy. This supplies absolute URLs for the sitemap. Use `HOST` and `PORT` to choose the server bind address and port. If an existing reverse proxy sets forwarding headers, `TRUSTED_PROXY_IPS` accepts its comma-separated literal IP addresses. By default guest limits use the direct connection address.

## Build and verify

```sh
npm run lint
npm test
npm run build
npm start
```

For browser checks, run the built server locally, make Playwright and Chromium available, then run `node scripts/check-practice.mjs`. Set `PARLO_TEST_URL` for a different loopback port, `PARLO_PLAYWRIGHT_PATH` to an existing Playwright entry module, or `PARLO_BROWSER` to a Chromium executable. `PARLO_SCREENSHOTS` enables image capture. `PARLO_WEBGL_SMOKE=1` also exercises the three rooms through Chromium's software WebGL2 renderer. Model and chat responses in the browser check are local fixtures.

The [product and engineering review](REVIEW.md) records the changes, measurements, release gates and next experiment. The [research ledger](RESEARCH.md) links the sources and methods. Original assessment and design references are retained under `.aistudio/references/`.
