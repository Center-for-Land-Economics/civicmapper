// Parking-label campaigns — LOCAL DEVELOPMENT ONLY.
//
// Serves the campaigns written by data/scripts/label_campaign.py (data/labels/<campaign>/) to the
// dev-only labelling page (viz/label.html) and appends submitted labels to
// data/labels/<campaign>/labels.jsonl. There is no account system: a labeller types a name.
//
// Two independent gates keep this off the public API:
//   1. the router is only mounted when NODE_ENV !== "production" (or LABELS_ENABLED=1);
//   2. every request must arrive on a loopback socket with no X-Forwarded-For header, i.e. from
//      this machine directly or through the Vite dev proxy — never through Azure ingress.
// Sync to the parking-labels-dev Azure container with data/scripts/label_sync.py.

import express from "express";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
export const LABELS_DIR = process.env.LABELS_DIR || path.join(ROOT, "data", "labels");

// "cutout" marks a hole (landscaped island, kiosk): exports subtract it from the parking shapes it
// overlaps. Labels keep the raw shapes as drawn.
export const CLASSES = ["paved", "unpaved", "structure", "unsure", "cutout"];
export const OUTCOMES = ["labeled", "no_parking", "cant_tell", "skip"];
const FLAGS = ["shadows", "trees", "construction", "imagery_stale", "clouds", "misregistered"];
const LEASE_MS = 10 * 60 * 1000;
const SAM_URL = process.env.SAM_URL || "http://127.0.0.1:8765/segment";
const CAMPAIGN_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/i;

export function labelsEnabled(env = process.env) {
  return env.LABELS_ENABLED === "1" || env.NODE_ENV !== "production";
}

function isLoopback(addr = "") {
  return addr === "127.0.0.1" || addr === "::1" || addr === "::ffff:127.0.0.1";
}

export function localOnly(req, res, next) {
  if (!isLoopback(req.socket?.remoteAddress) || req.headers["x-forwarded-for"]) {
    return res.status(404).json({ ok: false, error: "Not found" });
  }
  return next();
}

// ---------------------------------------------------------------------------
// Campaign files
// ---------------------------------------------------------------------------

const cache = new Map(); // `${campaign}/${file}` -> { mtimeMs, data }

function campaignDir(name) {
  if (!CAMPAIGN_RE.test(name)) return null;
  const dir = path.join(LABELS_DIR, name);
  return fs.existsSync(path.join(dir, "campaign.json")) ? dir : null;
}

function readJson(dir, file) {
  const p = path.join(dir, file);
  if (!fs.existsSync(p)) return null;
  const key = p;
  const { mtimeMs } = fs.statSync(p);
  const hit = cache.get(key);
  if (hit && hit.mtimeMs === mtimeMs) return hit.data;
  const data = JSON.parse(fs.readFileSync(p, "utf8"));
  cache.set(key, { mtimeMs, data });
  return data;
}

export function readLabels(dir) {
  const p = path.join(dir, "labels.jsonl");
  if (!fs.existsSync(p)) return [];
  return fs.readFileSync(p, "utf8").split("\n").filter(Boolean).flatMap((line) => {
    try { return [JSON.parse(line)]; } catch { return []; }
  });
}

/** Latest submission per (square, labeller); skips kept separately. */
function latestBySquare(labels) {
  const out = new Map(); // square -> Map(labeler -> label)
  for (const l of labels) {
    if (!out.has(l.square_id)) out.set(l.square_id, new Map());
    out.get(l.square_id).set(l.labeler.toLowerCase(), l);
  }
  return out;
}

function needed(sq) {
  return sq.properties.double ? 2 : 1;
}

// Held-out test squares: the same stable hash split as data/scripts/label_finetune.py (25%).
// The label page never pre-fills them with model guesses, so accuracy is measured on squares drawn
// from scratch. Targeted squares (picked where the model looks wrong) are never held out.
export const HOLDOUT_SHARE = 0.25;
export function isHoldout(props) {
  if (props.targeted) return false;
  const h = parseInt(crypto.createHash("sha1").update(props.id).digest("hex").slice(0, 8), 16);
  return h / 0xffffffff < HOLDOUT_SHARE;
}

function squareStatus(sq, perSquare) {
  const by = perSquare.get(sq.properties.id) || new Map();
  const done = [...by.values()].filter((l) => l.outcome !== "skip");
  return {
    labels: done.length,
    labelers: done.map((l) => l.labeler),
    skips: [...by.values()].filter((l) => l.outcome === "skip").length,
    complete: done.length >= needed(sq),
    holdout: isHoldout(sq.properties)
  };
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

function validRing(ring) {
  return Array.isArray(ring) && ring.length >= 4 && ring.length <= 2000 &&
    ring.every((c) => Array.isArray(c) && c.length >= 2 &&
      Number.isFinite(c[0]) && Number.isFinite(c[1]) &&
      Math.abs(c[0]) <= 180 && Math.abs(c[1]) <= 90);
}

// Paint grid (label page "Paint" tool): one cell per NAIP pixel across the square, row-major from
// the top-left corner; values 0 none, 1 paved, 2 unpaved, 3 structure, 4 unsure, 5 cut-out;
// rle = "value:count,…". Painted cells override polygons at export; painted cut-outs clear them.
const MAX_PAINT_CELLS = 2000 * 2000;
function validatePaint(p, errors) {
  if (p === undefined || p === null) return null;
  const ok = p && typeof p === "object" && Number.isInteger(p.width) && Number.isInteger(p.height) &&
    p.width > 0 && p.height > 0 && p.width * p.height <= MAX_PAINT_CELLS &&
    Number.isFinite(p.cell_m) && p.cell_m > 0 && p.cell_m <= 10 &&
    Array.isArray(p.corners) && p.corners.length === 4 &&
    p.corners.every((c) => Array.isArray(c) && c.length === 2 && c.every(Number.isFinite)) &&
    typeof p.rle === "string" && /^[0-5]:\d+(,[0-5]:\d+)*$/.test(p.rle);
  if (!ok) { errors.push("paint"); return null; }
  let total = 0, classes = false;
  for (const run of p.rle.split(",")) {
    const [v, n] = run.split(":").map(Number);
    total += n;
    if (v >= 1 && v <= 4 && n > 0) classes = true;
  }
  if (total !== p.width * p.height) { errors.push("paint: run lengths do not match the grid"); return null; }
  return { classes, value: { width: p.width, height: p.height, cell_m: p.cell_m, corners: p.corners, rle: p.rle } };
}

export function validateLabel(body, squareIds) {
  const errors = [];
  const b = body && typeof body === "object" ? body : {};
  const labeler = typeof b.labeler === "string" ? b.labeler.trim() : "";
  if (!labeler || labeler.length > 60 || !/^[\p{L}\p{N} ._'-]+$/u.test(labeler)) errors.push("labeler");
  if (!squareIds.has(b.square_id)) errors.push("square_id");
  if (!OUTCOMES.includes(b.outcome)) errors.push("outcome");
  if (!["blind", "correction", "model"].includes(b.mode)) errors.push("mode");
  const feats = Array.isArray(b.polygons) ? b.polygons : [];
  if (feats.length > 500) errors.push("polygons: too many");
  const polygons = [];
  for (const f of feats) {
    const g = f?.geometry;
    const cls = f?.properties?.class;
    if (!g || g.type !== "Polygon" || !Array.isArray(g.coordinates) || !g.coordinates.every(validRing) ||
        !CLASSES.includes(cls)) {
      errors.push("polygons: invalid feature");
      break;
    }
    polygons.push({ type: "Feature", geometry: { type: "Polygon", coordinates: g.coordinates },
      properties: { class: cls, ...(f.properties.from_prefill ? { from_prefill: true } : {}) } });
  }
  const paint = validatePaint(b.paint, errors);
  if (b.outcome === "labeled" && !polygons.some((p) => p.properties.class !== "cutout") && !paint?.classes) {
    errors.push("labeled needs at least one polygon or painted cell that is not a cut-out");
  }
  if (b.outcome !== "labeled" && paint) errors.push(`${b.outcome} takes no paint`);
  if (b.outcome !== "labeled" && polygons.length > 0) errors.push(`${b.outcome} takes no polygons`);
  const flags = Array.isArray(b.flags) ? b.flags.filter((f) => FLAGS.includes(f)) : [];
  const note = typeof b.note === "string" ? b.note.slice(0, 1000) : "";
  const seconds = Number.isFinite(b.seconds) ? Math.max(0, Math.min(86400, Math.round(b.seconds))) : null;
  const naipYear = Number.isInteger(b.naip_year) ? b.naip_year : null;
  const imagery = typeof b.imagery === "string" ? b.imagery.slice(0, 40) : null;
  if (errors.length) return { errors };
  return {
    label: {
      square_id: b.square_id, labeler, outcome: b.outcome, mode: b.mode, polygons, flags, note,
      seconds, naip_year: naipYear, imagery,
      ...(paint ? { paint: paint.value } : {})
    }
  };
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

export function labelsRouter() {
  const router = express.Router();
  const leases = new Map(); // `${campaign}/${square}` -> { labeler, until }
  router.use(localOnly);

  router.get("/campaigns", (_req, res) => {
    if (!fs.existsSync(LABELS_DIR)) return res.json({ ok: true, campaigns: [] });
    const campaigns = fs.readdirSync(LABELS_DIR).flatMap((name) => {
      const dir = campaignDir(name);
      if (!dir) return [];
      const meta = readJson(dir, "campaign.json");
      const squares = readJson(dir, "squares.geojson")?.features || [];
      const per = latestBySquare(readLabels(dir));
      const st = squares.map((s) => squareStatus(s, per));
      return [{ ...meta, progress: {
        squares: squares.length,
        complete: st.filter((s) => s.complete).length,
        labels: st.reduce((n, s) => n + s.labels, 0),
        needed: squares.reduce((n, s) => n + needed(s), 0)
      } }];
    });
    res.json({ ok: true, campaigns });
  });

  router.get("/:campaign/squares", (req, res) => {
    const dir = campaignDir(req.params.campaign);
    if (!dir) return res.status(404).json({ ok: false, error: "No such campaign" });
    const fc = readJson(dir, "squares.geojson");
    const per = latestBySquare(readLabels(dir));
    res.json({ ...fc, features: fc.features.map((f) => ({
      ...f, properties: { ...f.properties, ...squareStatus(f, per) } })) });
  });

  // Context layers, optionally just one square's features.
  router.get("/:campaign/layer/:layer", (req, res) => {
    const dir = campaignDir(req.params.campaign);
    if (!dir || !["prefill", "parcels", "buildings", "roads", "model_prefill"].includes(req.params.layer)) {
      return res.status(404).json({ ok: false, error: "Not found" });
    }
    const fc = readJson(dir, `${req.params.layer}.geojson`) || { type: "FeatureCollection", features: [] };
    const sq = typeof req.query.square === "string" ? req.query.square : null;
    res.json(sq ? { ...fc, features: fc.features.filter((f) => f.properties.square_id === sq) } : fc);
  });

  // Next square for a labeller: incomplete, not already done or skipped by them, not leased by
  // someone else. A square needing a second label is offered before an untouched one so double
  // labels finish early enough to measure agreement.
  router.get("/:campaign/next", (req, res) => {
    const dir = campaignDir(req.params.campaign);
    if (!dir) return res.status(404).json({ ok: false, error: "No such campaign" });
    const who = typeof req.query.labeler === "string" ? req.query.labeler.trim().toLowerCase() : "";
    if (!who) return res.status(400).json({ ok: false, error: "labeler required" });
    const after = typeof req.query.after === "string" ? req.query.after : null;
    const squares = readJson(dir, "squares.geojson").features;
    const per = latestBySquare(readLabels(dir));
    const now = Date.now();
    const open = squares.filter((s) => {
      const id = s.properties.id;
      const mine = per.get(id)?.has(who);
      const lease = leases.get(`${req.params.campaign}/${id}`);
      const leasedByOther = lease && lease.until > now && lease.labeler !== who;
      return !mine && !leasedByOther && !squareStatus(s, per).complete && id !== after;
    });
    const started = open.filter((s) => squareStatus(s, per).labels > 0);
    const pick = started[0] || open[0];
    if (!pick) return res.json({ ok: true, square: null, remaining: 0 });
    leases.set(`${req.params.campaign}/${pick.properties.id}`, { labeler: who, until: now + LEASE_MS });
    res.json({ ok: true, square: { ...pick, properties: { ...pick.properties, ...squareStatus(pick, per) } },
      remaining: open.length });
  });

  router.get("/:campaign/square/:id", (req, res) => {
    const dir = campaignDir(req.params.campaign);
    if (!dir) return res.status(404).json({ ok: false, error: "No such campaign" });
    const sq = readJson(dir, "squares.geojson").features.find((f) => f.properties.id === req.params.id);
    if (!sq) return res.status(404).json({ ok: false, error: "No such square" });
    const per = latestBySquare(readLabels(dir));
    res.json({ ok: true, square: { ...sq, properties: { ...sq.properties, ...squareStatus(sq, per) } } });
  });

  router.get("/:campaign/labels", (req, res) => {
    const dir = campaignDir(req.params.campaign);
    if (!dir) return res.status(404).json({ ok: false, error: "No such campaign" });
    let labels = readLabels(dir);
    if (typeof req.query.square === "string") labels = labels.filter((l) => l.square_id === req.query.square);
    if (typeof req.query.labeler === "string") {
      const who = req.query.labeler.trim().toLowerCase();
      labels = labels.filter((l) => l.labeler.toLowerCase() === who);
    }
    res.json({ ok: true, labels });
  });

  // Segment tool: forward clicks to the local SAM 2.1 helper (data/scripts/sam_server.py).
  router.post("/:campaign/sam", async (req, res) => {
    const dir = campaignDir(req.params.campaign);
    if (!dir) return res.status(404).json({ ok: false, error: "No such campaign" });
    const b = req.body || {};
    const pt = (p) => Array.isArray(p) && p.length === 3 && Number.isFinite(p[0]) && Number.isFinite(p[1]) &&
      Math.abs(p[0]) <= 180 && Math.abs(p[1]) <= 90 && typeof p[2] === "boolean";
    const points = Array.isArray(b.points) ? b.points : [];
    const box = Array.isArray(b.box) && b.box.length === 4 && b.box.every(Number.isFinite) ? b.box : null;
    if (typeof b.square_id !== "string" || points.length > 30 || !points.every(pt) || (!points.length && !box)) {
      return res.status(400).json({ ok: false, error: "square_id plus 1-30 points [lng, lat, include] or a box" });
    }
    try {
      const r = await fetch(SAM_URL, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ campaign: req.params.campaign, square_id: b.square_id, points, box }),
        signal: AbortSignal.timeout(120000)
      });
      return res.status(r.status).json(await r.json());
    } catch {
      return res.status(503).json({ ok: false, error: "Segment server not running: start it with " +
        "python data/scripts/sam_server.py (GPU environment)" });
    }
  });

  router.post("/:campaign/labels", (req, res) => {
    const dir = campaignDir(req.params.campaign);
    if (!dir) return res.status(404).json({ ok: false, error: "No such campaign" });
    const squares = readJson(dir, "squares.geojson").features;
    const ids = new Set(squares.map((s) => s.properties.id));
    const { label, errors } = validateLabel(req.body, ids);
    if (errors) return res.status(400).json({ ok: false, errors });
    const record = { id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
      campaign: req.params.campaign, submitted: new Date().toISOString(), ...label };
    fs.appendFileSync(path.join(dir, "labels.jsonl"), JSON.stringify(record) + "\n");
    leases.delete(`${req.params.campaign}/${label.square_id}`);
    res.json({ ok: true, id: record.id });
  });

  return router;
}
