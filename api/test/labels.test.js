/**
 * Dev-only parking-label routes (src/labels.js).
 * Run: node --test test/labels.test.js
 */
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "labels-test-"));
process.env.LABELS_DIR = dir;
const camp = path.join(dir, "demo");
fs.mkdirSync(camp);
fs.writeFileSync(path.join(camp, "campaign.json"), JSON.stringify({ campaign: "demo", city: "x", n: 2 }));
const sq = (id, double) => ({
  type: "Feature",
  geometry: { type: "Polygon", coordinates: [[[0, 0], [0, 0.002], [0.002, 0.002], [0.002, 0], [0, 0]]] },
  properties: { id, stratum: "commercial", double, naip_year: 2024, naip_items: ["a"], imagery_date: "2024-06-01" }
});
fs.writeFileSync(path.join(camp, "squares.geojson"),
  JSON.stringify({ type: "FeatureCollection", features: [sq("demo-0001", true), sq("demo-0002", false)] }));

const { app } = await import("../src/server.js");
const { labelsEnabled, validateLabel } = await import("../src/labels.js");

let server, base;
before(() => new Promise((resolve) => {
  server = http.createServer(app);
  server.listen(0, "127.0.0.1", () => { base = `http://127.0.0.1:${server.address().port}`; resolve(); });
}));
after(() => { server.close(); fs.rmSync(dir, { recursive: true, force: true }); });

const poly = { type: "Feature", properties: { class: "paved" },
  geometry: { type: "Polygon", coordinates: [[[0, 0], [0, 0.001], [0.001, 0.001], [0, 0]]] } };
const post = (body) => fetch(`${base}/api/labels/demo/labels`, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

test("disabled in production unless LABELS_ENABLED=1", () => {
  assert.equal(labelsEnabled({ NODE_ENV: "production" }), false);
  assert.equal(labelsEnabled({ NODE_ENV: "production", LABELS_ENABLED: "1" }), true);
  assert.equal(labelsEnabled({}), true);
});

test("refuses forwarded (non-local) requests", async () => {
  const r = await fetch(`${base}/api/labels/campaigns`, { headers: { "X-Forwarded-For": "203.0.113.5" } });
  assert.equal(r.status, 404);
});

test("lists campaigns with progress", async () => {
  const js = await (await fetch(`${base}/api/labels/campaigns`)).json();
  assert.equal(js.campaigns[0].campaign, "demo");
  assert.deepEqual(js.campaigns[0].progress, { squares: 2, complete: 0, labels: 0, needed: 3 });
});

test("rejects bad campaign names and invalid labels", async () => {
  assert.equal((await fetch(`${base}/api/labels/..%2F..%2Fetc/squares`)).status, 404);
  assert.equal((await post({ labeler: "A", square_id: "nope", outcome: "labeled", mode: "blind", polygons: [poly] })).status, 400);
  assert.equal((await post({ labeler: "A", square_id: "demo-0001", outcome: "labeled", mode: "blind", polygons: [] })).status, 400);
  const bad = { ...poly, properties: { class: "lawn" } };
  assert.ok(validateLabel({ labeler: "A", square_id: "demo-0001", outcome: "labeled", mode: "blind", polygons: [bad] },
    new Set(["demo-0001"])).errors);
});

test("double squares need two labellers; next skips what you already did", async () => {
  let next = await (await fetch(`${base}/api/labels/demo/next?labeler=A`)).json();
  assert.equal(next.square.properties.id, "demo-0001");
  assert.equal((await post({ labeler: "A", square_id: "demo-0001", outcome: "labeled", mode: "blind",
    polygons: [poly], seconds: 30, flags: ["shadows", "bogus"] })).status, 200);
  next = await (await fetch(`${base}/api/labels/demo/next?labeler=A`)).json();
  assert.equal(next.square.properties.id, "demo-0002");
  next = await (await fetch(`${base}/api/labels/demo/next?labeler=B`)).json();
  assert.equal(next.square.properties.id, "demo-0001", "a started double square is offered first");
  assert.equal((await post({ labeler: "B", square_id: "demo-0001", outcome: "no_parking", mode: "blind",
    polygons: [] })).status, 200);
  const js = await (await fetch(`${base}/api/labels/campaigns`)).json();
  assert.equal(js.campaigns[0].progress.complete, 1);
  const lines = fs.readFileSync(path.join(camp, "labels.jsonl"), "utf8").trim().split("\n").map(JSON.parse);
  assert.equal(lines.length, 2);
  assert.deepEqual(lines[0].flags, ["shadows"]);
});
