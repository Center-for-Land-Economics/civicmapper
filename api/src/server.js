import express from "express";
import cors from "cors";
import https from "https";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import { randomUUID } from "crypto";
import { fileURLToPath } from "url";

// ---------------------------------------------------------------------------
// Structured logger
// ---------------------------------------------------------------------------
// Emits newline-delimited JSON to stdout. Azure Container Apps captures stdout
// and makes it queryable in Azure Monitor / Log Analytics.
// Usage: log("info", "event_name", req, { extra: "fields" })
//        log("error", "event_name", null, { error: err.message })

function log(level, event, req, extra = {}) {
  const entry = {
    level,
    event,
    ts: new Date().toISOString(),
    ...(req ? { requestId: req.requestId, ip: req.ip, method: req.method } : {}),
    ...extra
  };
  // Use console.error for error/warn so they appear in stderr streams too
  if (level === "error" || level === "warn") {
    console.error(JSON.stringify(entry));
  } else {
    console.log(JSON.stringify(entry));
  }
}

function classifyErrorFromStatus(statusCode) {
  if (statusCode >= 500) return "server_error";
  if (statusCode === 401) return "unauthorized";
  if (statusCode === 403) return "forbidden";
  if (statusCode === 404) return "not_found";
  if (statusCode === 408) return "timeout";
  if (statusCode === 429) return "rate_limited";
  if (statusCode >= 400) return "client_error";
  return null;
}

function tagRequestOutcome(res, errorClass, extra = {}) {
  res.locals.errorClass = errorClass;
  Object.assign(res.locals, extra);
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DATA_PROXY_BASE_URL = normalizeBaseUrl(
  process.env.DATA_PROXY_BASE_URL ||
    process.env.BLOB_STORAGE_BASE ||
    "https://landeconomics.blob.core.windows.net/parquets-dev"
);

// Optional SAS token appended to every blob request (enables private containers).
// Set BLOB_SAS_TOKEN in the Container App env — do NOT include the leading "?".
const BLOB_SAS_TOKEN = (process.env.BLOB_SAS_TOKEN || "").trim();

const rawAllowed = (process.env.ALLOWED_ORIGINS || "")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);

const defaultAllowed = [
  "https://civicmapper.org",
  "https://www.civicmapper.org",
  "https://dev.civicmapper.org",
  "https://api.civicmapper.org",
  "https://api.dev.civicmapper.org",
  "http://localhost:5173",
  "http://127.0.0.1:5173",
  "http://localhost:4173",
  "http://127.0.0.1:4173"
];

const allowAnyOrigin = rawAllowed.includes("*");
const allowedOrigins = allowAnyOrigin ? [] : rawAllowed.length ? rawAllowed : defaultAllowed;
const normalizedAllowed = new Set(allowedOrigins.map(normalizeOrigin));
console.log(
  "[CORS] Allowed origins:",
  allowAnyOrigin ? "*" : Array.from(normalizedAllowed).join(", ") || "(none)"
);

const envLabel = deriveEnvLabel();
let httpsRequestImpl = https.request.bind(https);

const dataProxyAgent = new https.Agent({
  keepAlive: true,
  keepAliveMsecs: 1000,
  maxSockets: 64,
  maxFreeSockets: 16
});

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

const app = express();

// Trust the first proxy hop (Azure Container Apps / Front Door).
// Required for express-rate-limit to read the real client IP from
// X-Forwarded-For without throwing a validation error.
app.set("trust proxy", 1);

// Security headers via helmet. CSP is intentionally disabled here — the static
// frontend sets its own headers via staticwebapp.config.json; the API only
// serves JSON/binary responses.
app.use(helmet({ contentSecurityPolicy: false }));

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin || allowAnyOrigin) return callback(null, true);
      if (normalizedAllowed.has(normalizeOrigin(origin))) {
        return callback(null, true);
      }
      log("warn", "cors_blocked", null, { origin });
      return callback(new Error("CORS"));
    },
    methods: ["GET", "HEAD", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization", "X-Requested-With", "Range"],
    exposedHeaders: ["Content-Range", "Content-Length", "Content-Type"],
    optionsSuccessStatus: 204
  })
);

app.use(express.json({ limit: "256kb" }));
app.use(express.urlencoded({ extended: false }));

// Assign a unique ID to every request for log correlation.
app.use((req, _res, next) => {
  req.requestId = randomUUID();
  next();
});

app.use((req, res, next) => {
  const startedAt = Date.now();
  let settled = false;

  const flushLog = (phase) => {
    if (settled) return;
    settled = true;

    const status = res.statusCode;
    const timedOut = res.locals.routeTimedOut === true;
    const errorClass = res.locals.errorClass || (timedOut ? "timeout" : classifyErrorFromStatus(status));
    const level = phase === "aborted" || status >= 500 ? "error" : status >= 400 ? "warn" : "info";

    log(level, "http_request", req, {
      route: req.route?.path || req.path,
      status,
      duration_ms: Date.now() - startedAt,
      timed_out: timedOut,
      error_class: errorClass,
      phase
    });
  };

  res.on("finish", () => flushLog("finish"));
  res.on("close", () => {
    if (!res.writableEnded) flushLog("aborted");
  });

  next();
});

// ---------------------------------------------------------------------------
// Rate limiters
// ---------------------------------------------------------------------------

// Data/parquet proxy.
//
// The old comment here said "a typical parquet load makes 2–10 range requests"
// and capped IPs at 60/15 min. That was wrong for PMTiles cities: NYC (204 MB),
// Houston (233 MB), Albuquerque (90 MB), Portland (97 MB), Baltimore (55 MB),
// Denver (70 MB) all serve via PMTiles range requests that scale with the
// user's panning + zooming — easily 30–80 requests per session per city. After
// 2–3 cities a real user blew the 60-req cap and got 429s back; subsequent
// pan/zoom on EVERY already-loaded city also 429'd, so the whole app appeared
// broken once they hit it.
//
// The app is fully public (no login), so every request counts against the
// per-IP cap. Sized for real usage: a long multi-city session can make
// 500–1000+ range requests, and office/campus NATs share one IP across many
// users. 4000/15 min keeps crawlers bounded without 429ing real explorers.
const dataLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 4000,
  standardHeaders: true,
  legacyHeaders: false,
  message: { ok: false, error: "Too many requests, please try again later." }
});

const telemetryLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 120,
  standardHeaders: true,
  legacyHeaders: false,
  message: { ok: false, error: "Too many telemetry events, please try again later." }
});

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

app.get("/healthz", (_req, res) => res.json({ ok: true }));

app.get("/api/hello", (_req, res) =>
  res.json({ message: "hello", env: process.env.NODE_ENV || "dev" })
);

app.post(["/telemetry", "/api/telemetry"], telemetryLimiter, (req, res) => {
  const body = typeof req.body === "object" && req.body ? req.body : {};
  const status = Number.isFinite(Number(body.status)) ? Number(body.status) : null;
  const durationMs = Number.isFinite(Number(body.duration_ms)) ? Number(body.duration_ms) : null;

  log("info", "browser_rum", req, {
    rum_name: typeof body.name === "string" ? body.name.slice(0, 120) : "unknown",
    rum_page: typeof body.path === "string" ? body.path : req.path,
    rum_env: typeof body.env === "string" ? body.env : envLabel || null,
    rum_context: typeof body.context === "string" ? body.context : null,
    rum_duration_ms: durationMs,
    rum_status: status,
    rum_error: typeof body.error === "string" ? body.error.slice(0, 500) : null,
    rum_message: typeof body.message === "string" ? body.message.slice(0, 500) : null,
    rum_auth_mode: typeof body.auth_mode === "string" ? body.auth_mode : null
  });

  res.status(202).json({ ok: true });
});

const dataRoutes = ["/data/:filename", "/api/data/:filename"];

for (const route of dataRoutes) {
  app.options(route, (_req, res) => res.sendStatus(204));
  app.get(route, dataLimiter, handleDatasetProxy);
  app.head(route, dataLimiter, handleDatasetProxy);
}

// Parking lot datasets live in a parking/ subfolder in blob storage.
// These must be registered before the generic :filename routes would 404 them.
const parkingRoutes = ["/data/parking/:filename", "/api/data/parking/:filename"];

for (const route of parkingRoutes) {
  app.options(route, (_req, res) => res.sendStatus(204));
  app.get(route, dataLimiter, (req, res) => handleDatasetProxy(req, res, "parking/"));
  app.head(route, dataLimiter, (req, res) => handleDatasetProxy(req, res, "parking/"));
}

// Staging slots — reviewer preview before promotion to dev.
// Staging ID is an opaque 8-char hex hash; only people with the ID can access files.
// Route: /data/staging/:stagingId/:filename  (and /api/data/staging/...)
// Maps to blob: staging/{stagingId}/{filename}
const stagingRoutes = ["/data/staging/:stagingId/:filename", "/api/data/staging/:stagingId/:filename"];

function handleStagingProxy(req, res) {
  const { stagingId, filename } = req.params;
  if (!/^[0-9a-f]{8}$/i.test(stagingId)) {
    log("warn", "staging_invalid_id", req, { stagingId });
    return res.status(400).json({ ok: false, error: "Invalid staging ID" });
  }
  handleDatasetProxy(req, res, `staging/${stagingId}/`);
}

for (const route of stagingRoutes) {
  app.options(route, (_req, res) => res.sendStatus(204));
  app.get(route, dataLimiter, handleStagingProxy);
  app.head(route, dataLimiter, handleStagingProxy);
}

app.use((err, _req, res, next) => {
  if (err && err.message === "CORS") {
    return res.status(403).json({ ok: false, error: "CORS" });
  }
  return next(err);
});

// Export app for testing. Only bind to a port when run directly.
export {
  app,
  setHttpsRequestForTesting
};
const isMain = fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  app.listen(process.env.PORT || 8080, () => log("info", "server_start", null, { port: process.env.PORT || 8080 }));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function normalizeOrigin(origin = "") {
  return origin.replace(/\/+$/, "");
}

function normalizeBaseUrl(url = "") {
  return url.replace(/\/+$/, "");
}

function deriveEnvLabel() {
  const label =
    (process.env.APP_ENV_LABEL || process.env.APP_ENV || "").trim() ||
    (process.env.NODE_ENV === "production"
      ? "Prod"
      : process.env.NODE_ENV === "development"
        ? "Dev"
        : (process.env.NODE_ENV || "").trim());
  return label;
}

function setHttpsRequestForTesting(fn) {
  httpsRequestImpl = fn || https.request.bind(https);
}

function handleDatasetProxy(req, res, subfolder = "") {
  // Guard: Express passes (req, res, next) to route handlers, so subfolder may
  // receive the next() function when the handler is registered directly without
  // a wrapper (e.g. app.get(route, dataLimiter, handleDatasetProxy)).
  // Treat any non-string value as the empty string (top-level files).
  if (typeof subfolder !== "string") subfolder = "";

  const t0 = Date.now();

  try {
    const filename = (req.params?.filename || "").trim();
    if (!filename) {
      tagRequestOutcome(res, "missing_filename");
      res.status(400).json({ ok: false, error: "Filename required" });
      return;
    }

    if (!isSafeFilename(filename)) {
      log("warn", "data_invalid_filename", req, { filename });
      tagRequestOutcome(res, "invalid_filename");
      res.status(400).json({ ok: false, error: "Invalid filename" });
      return;
    }

    if (!DATA_PROXY_BASE_URL) {
      tagRequestOutcome(res, "proxy_not_configured");
      res.status(500).json({ ok: false, error: "DATA_PROXY_BASE_URL not configured" });
      return;
    }

    // Subfolder is validated at the route level (only "parking/" is currently used).
    const blobPath = subfolder ? `${subfolder}${filename}` : filename;

    // Append SAS token when configured — this enables the backing Azure Blob
    // containers to be set to Private while still being accessible via the proxy.
    const blobUrl = new URL(
      `${DATA_PROXY_BASE_URL}/${blobPath}${BLOB_SAS_TOKEN ? "?" + BLOB_SAS_TOKEN : ""}`
    );

    const headers = {};
    const rangeHeader = req.headers?.range;
    const requestedRange = typeof rangeHeader === "string" && rangeHeader ? rangeHeader : null;
    if (typeof rangeHeader === "string" && rangeHeader) {
      headers.Range = rangeHeader;
    }

    let bytesStreamed = 0;
    const proxyRequest = httpsRequestImpl(
      {
        method: req.method,
        protocol: blobUrl.protocol,
        hostname: blobUrl.hostname,
        port: blobUrl.port,
        path: `${blobUrl.pathname}${blobUrl.search}`,
        headers,
        agent: dataProxyAgent
      },
      (proxyResponse) => {
        const status = proxyResponse.statusCode ?? 502;
        const contentLengthHeader = proxyResponse.headers?.["content-length"];
        const contentLength = Array.isArray(contentLengthHeader)
          ? Number(contentLengthHeader[0])
          : Number(contentLengthHeader);
        res.status(status);

        copyHeader(proxyResponse.headers, res, "content-type", "Content-Type");
        copyHeader(proxyResponse.headers, res, "content-length", "Content-Length");
        copyHeader(proxyResponse.headers, res, "content-range", "Content-Range");
        copyHeader(proxyResponse.headers, res, "accept-ranges", "Accept-Ranges");
        copyHeader(proxyResponse.headers, res, "etag", "ETag");
        copyHeader(proxyResponse.headers, res, "last-modified", "Last-Modified");
        copyHeader(proxyResponse.headers, res, "cache-control", "Cache-Control");

        if (req.method === "HEAD") {
          proxyResponse.resume();
          proxyResponse.on("end", () => {
            log("info", "data_request", req, {
              filename: blobPath,
              status,
              duration_ms: Date.now() - t0,
              range_requested: !!requestedRange,
              content_length: Number.isFinite(contentLength) ? contentLength : null,
              bytes_streamed: 0
            });
            res.end();
          });
          proxyResponse.on("error", () => res.end());
          return;
        }

        proxyResponse.on("data", (chunk) => {
          bytesStreamed += Buffer.isBuffer(chunk) ? chunk.length : Buffer.byteLength(String(chunk));
        });

        proxyResponse.on("error", (err) => {
          log("error", "data_stream_error", req, {
            filename: blobPath,
            error: err.message,
            duration_ms: Date.now() - t0,
            status,
            range_requested: !!requestedRange,
            content_length: Number.isFinite(contentLength) ? contentLength : null,
            bytes_streamed: bytesStreamed
          });
          if (!res.headersSent) {
            tagRequestOutcome(res, "data_stream_error");
            res.status(502).json({ ok: false, error: "Failed to fetch dataset" });
          } else {
            res.destroy(err);
          }
        });

        proxyResponse.on("end", () => {
          log("info", "data_request", req, {
            filename: blobPath,
            status,
            duration_ms: Date.now() - t0,
            range_requested: !!requestedRange,
            content_length: Number.isFinite(contentLength) ? contentLength : null,
            bytes_streamed: bytesStreamed
          });
        });

        proxyResponse.pipe(res);
      }
    );

    // Destroy the socket if blob storage doesn't respond within 15 s.
    // Without this, hung outbound connections accumulate and eventually
    // exhaust the Container App's socket resources, causing all subsequent
    // range requests to hang indefinitely (manifests as 504 from Front Door).
    proxyRequest.setTimeout(15000, () => {
      log("error", "data_upstream_timeout", req, {
        filename: blobPath,
        duration_ms: Date.now() - t0,
        range_requested: !!requestedRange,
        bytes_streamed: bytesStreamed
      });
      proxyRequest.destroy();
      if (!res.headersSent) {
        tagRequestOutcome(res, "upstream_timeout", { routeTimedOut: true });
        res.status(504).json({ ok: false, error: "Upstream timeout" });
      }
    });

    proxyRequest.on("error", (error) => {
      log("error", "data_proxy_error", req, {
        filename: blobPath,
        error: error.message,
        duration_ms: Date.now() - t0,
        range_requested: !!requestedRange,
        bytes_streamed: bytesStreamed
      });
      if (!res.headersSent) {
        tagRequestOutcome(res, "data_proxy_error");
        res.status(502).json({ ok: false, error: "Failed to fetch dataset" });
      } else {
        res.destroy(error);
      }
    });

    proxyRequest.end();
  } catch (error) {
    log("error", "data_proxy_unexpected", req, { error: error.message, duration_ms: Date.now() - t0 });
    tagRequestOutcome(res, "data_proxy_unexpected");
    res.status(500).json({ ok: false, error: "Dataset proxy error" });
  }
}

function isSafeFilename(name) {
  return /^[A-Za-z0-9_.\-]+$/.test(name);
}

function copyHeader(source, res, key, targetKey) {
  const value = source?.[key];
  if (value !== undefined) {
    res.setHeader(targetKey, value);
  }
}
