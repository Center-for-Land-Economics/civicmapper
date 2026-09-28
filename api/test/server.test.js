/**
 * API integration tests using Node's built-in test runner.
 * Run: node --test test/server.test.js
 */

import { test, before, after, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import {
  app,
  setHttpsRequestForTesting
} from "../src/server.js";

let server;
let base;

before(() => {
  return new Promise((resolve) => {
    server = http.createServer(app);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      base = `http://127.0.0.1:${port}`;
      resolve();
    });
  });
});

after(() => {
  return new Promise((resolve) => server.close(resolve));
});

beforeEach(() => {
  setHttpsRequestForTesting(null);
});

afterEach(() => {
  setHttpsRequestForTesting(null);
});

function createMockHttpsRequest({ statusCode = 200, headers = {}, body = "" } = {}) {
  const calls = [];
  const fn = (options, callback) => {
    calls.push(options);
    const request = new EventEmitter();
    request.setTimeout = (_ms, handler) => {
      request._timeoutHandler = handler;
      return request;
    };
    request.write = () => {};
    request.destroy = (error) => {
      if (error) request.emit("error", error);
    };
    request.end = () => {
      const response = new PassThrough();
      response.statusCode = statusCode;
      response.headers = headers;
      callback(response);
      response.end(body);
    };
    return request;
  };
  return { fn, calls };
}

async function captureStructuredLogs(run) {
  const logs = [];
  const originalLog = console.log;
  const originalError = console.error;
  console.log = (...args) => logs.push(...args);
  console.error = (...args) => logs.push(...args);
  try {
    return await run(logs);
  } finally {
    console.log = originalLog;
    console.error = originalError;
  }
}

function findStructuredEvent(logs, eventName) {
  for (const entry of logs) {
    if (typeof entry !== "string") continue;
    try {
      const parsed = JSON.parse(entry);
      if (parsed?.event === eventName) return parsed;
    } catch {}
  }
  return null;
}

// ---------------------------------------------------------------------------
// /healthz
// ---------------------------------------------------------------------------

test("GET /healthz returns { ok: true }", async () => {
  const resp = await fetch(`${base}/healthz`);
  assert.equal(resp.status, 200);
  const body = await resp.json();
  assert.equal(body.ok, true);
});

test("GET /api/hello returns message field", async () => {
  const resp = await fetch(`${base}/api/hello`);
  assert.equal(resp.status, 200);
  const body = await resp.json();
  assert.ok(typeof body.message === "string");
});

test("GET /healthz allows the fixed dev app origin", async () => {
  const resp = await fetch(`${base}/healthz`, {
    headers: { Origin: "https://dev.civicmapper.org" }
  });
  assert.equal(resp.status, 200);
  assert.equal(resp.headers.get("access-control-allow-origin"), "https://dev.civicmapper.org");
});

test("GET /healthz blocks unknown origins", async () => {
  const resp = await fetch(`${base}/healthz`, {
    headers: { Origin: "https://evil.example.com" }
  });
  assert.equal(resp.status, 403);
  const body = await resp.json();
  assert.equal(body.error, "CORS");
});

// ---------------------------------------------------------------------------
// /api/data — public proxy
// ---------------------------------------------------------------------------

test("GET /api/data/file.parquet requires no auth", async () => {
  const mock = createMockHttpsRequest({
    statusCode: 200,
    headers: { "content-type": "application/octet-stream", "content-length": "4" },
    body: "test"
  });
  setHttpsRequestForTesting(mock.fn);

  const resp = await fetch(`${base}/api/data/southbend-in-parcels.parquet`);
  assert.equal(resp.status, 200);
  assert.equal(await resp.text(), "test");
});

test("GET /api/data/file.parquet reuses a keep-alive agent and logs byte metrics", async () => {
  const mock = createMockHttpsRequest({
    statusCode: 206,
    headers: {
      "content-type": "application/octet-stream",
      "content-length": "4",
      "content-range": "bytes 0-3/4",
      "accept-ranges": "bytes"
    },
    body: "test"
  });
  setHttpsRequestForTesting(mock.fn);

  const dataRequestLog = await captureStructuredLogs(async (logs) => {
    const resp = await fetch(`${base}/api/data/southbend-in-parcels.parquet`, {
      headers: { Range: "bytes=0-3" }
    });
    assert.equal(resp.status, 206);
    assert.equal(await resp.text(), "test");
    return findStructuredEvent(logs, "data_request");
  });

  assert.equal(mock.calls.length, 1);
  assert.equal(mock.calls[0].headers.Range, "bytes=0-3");
  assert.equal(mock.calls[0].agent?.keepAlive, true);
  assert.equal(dataRequestLog?.status, 206);
  assert.equal(dataRequestLog?.range_requested, true);
  assert.equal(dataRequestLog?.content_length, 4);
  assert.equal(dataRequestLog?.bytes_streamed, 4);
});

test("HEAD /api/data/file.parquet logs zero streamed bytes", async () => {
  const mock = createMockHttpsRequest({
    statusCode: 200,
    headers: {
      "content-type": "application/octet-stream",
      "content-length": "123",
      "accept-ranges": "bytes"
    }
  });
  setHttpsRequestForTesting(mock.fn);

  const dataRequestLog = await captureStructuredLogs(async (logs) => {
    const resp = await fetch(`${base}/api/data/southbend-in-parcels.parquet`, {
      method: "HEAD"
    });
    assert.equal(resp.status, 200);
    return findStructuredEvent(logs, "data_request");
  });

  assert.equal(mock.calls.length, 1);
  assert.equal(mock.calls[0].method, "HEAD");
  assert.equal(dataRequestLog?.content_length, 123);
  assert.equal(dataRequestLog?.bytes_streamed, 0);
  assert.equal(dataRequestLog?.range_requested, false);
});

test("POST /api/telemetry accepts browser rum payloads", async () => {
  const resp = await fetch(`${base}/api/telemetry`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: "page_ready",
      path: "/app.html",
      duration_ms: 123
    })
  });

  assert.equal(resp.status, 202);
  const body = await resp.json();
  assert.equal(body.ok, true);
});

// ---------------------------------------------------------------------------
// /api/data — filename validation
// ---------------------------------------------------------------------------

test("GET /api/data/ with path traversal attempt returns 400", async () => {
  const resp = await fetch(`${base}/api/data/..%2Fsecret`);
  // Express decodes %2F, so this routes to /api/data/../secret which Express normalizes
  // Either 400 or 404 is acceptable — must not be 200
  assert.ok(resp.status === 400 || resp.status === 404, `Expected 400/404, got ${resp.status}`);
});

// ---------------------------------------------------------------------------
// /data — browser caching (Cache-Control, revalidation, ETag exposure)
// ---------------------------------------------------------------------------

test("GET /data/<file>?v=<token> is cacheable for an hour and exposes ETag", async () => {
  const mock = createMockHttpsRequest({
    statusCode: 206,
    headers: {
      "content-type": "application/octet-stream",
      "content-length": "4",
      "content-range": "bytes 0-3/4",
      etag: '"0x8DE"'
    },
    body: "test"
  });
  setHttpsRequestForTesting(mock.fn);

  const resp = await fetch(`${base}/data/houston-tx-parcels.pmtiles?v=2026-06-24`, {
    headers: { Range: "bytes=0-3", Origin: "https://civicmapper.org" }
  });
  assert.equal(resp.status, 206);
  assert.equal(resp.headers.get("cache-control"), "public, max-age=3600");
  assert.equal(resp.headers.get("etag"), '"0x8DE"');
  assert.match(resp.headers.get("access-control-expose-headers") || "", /ETag/);
  await resp.text();
  // The version token is only a browser cache key — it is not sent on to blob storage.
  assert.equal(mock.calls[0].path, "/parquets-dev/houston-tx-parcels.pmtiles");
});

test("GET /data/<file> without a version token must revalidate", async () => {
  const mock = createMockHttpsRequest({
    statusCode: 200,
    headers: { "content-type": "application/octet-stream", "content-length": "4" },
    body: "test"
  });
  setHttpsRequestForTesting(mock.fn);

  const resp = await fetch(`${base}/data/southbend-in-parcels.parquet`);
  assert.equal(resp.status, 200);
  assert.equal(resp.headers.get("cache-control"), "public, no-cache");
  await resp.text();
});

test("GET /data/<file> forwards revalidation headers and passes a 304 through", async () => {
  const mock = createMockHttpsRequest({ statusCode: 304, headers: { etag: '"0x8DE"' } });
  setHttpsRequestForTesting(mock.fn);

  const resp = await fetch(`${base}/data/southbend-in-parcels.parquet`, {
    headers: {
      "If-None-Match": '"0x8DE"',
      "If-Modified-Since": "Sat, 03 Jan 2026 05:22:51 GMT",
      Range: "bytes=0-3",
      "If-Range": '"0x8DE"'
    }
  });
  assert.equal(resp.status, 304);
  assert.equal(resp.headers.get("cache-control"), "public, no-cache");
  assert.equal(mock.calls[0].headers["If-None-Match"], '"0x8DE"');
  assert.equal(mock.calls[0].headers["If-Modified-Since"], "Sat, 03 Jan 2026 05:22:51 GMT");
  assert.equal(mock.calls[0].headers["If-Range"], '"0x8DE"');
  assert.equal(mock.calls[0].headers.Range, "bytes=0-3");
});

test("GET /data/<file> keeps an upstream Cache-Control and adds none to errors", async () => {
  const withUpstream = createMockHttpsRequest({
    statusCode: 200,
    headers: { "content-length": "4", "cache-control": "max-age=60" },
    body: "test"
  });
  setHttpsRequestForTesting(withUpstream.fn);
  const resp = await fetch(`${base}/data/southbend-in-parcels.parquet?v=x`);
  assert.equal(resp.headers.get("cache-control"), "max-age=60");
  await resp.text();

  const notFound = createMockHttpsRequest({ statusCode: 404, headers: { "content-length": "0" } });
  setHttpsRequestForTesting(notFound.fn);
  const missing = await fetch(`${base}/data/missing.parquet?v=x`);
  assert.equal(missing.status, 404);
  assert.equal(missing.headers.get("cache-control"), null);
  await missing.text();
});
