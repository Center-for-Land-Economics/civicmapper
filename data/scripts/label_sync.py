#!/usr/bin/env python3
"""
Sync parking-label campaigns between data/labels/ and the Azure container parking-labels-dev.

labels.jsonl is merged, not overwritten: records from both sides are unioned by their id, so
several machines (volunteers each running the dev tool) can label the same campaign and sync in
any order. Campaign files (campaign.json, squares/prefill/parcels.geojson) are uploaded when they
differ and downloaded when missing locally.

Usage:
  python data/scripts/label_sync.py                 # sync every local and remote campaign
  python data/scripts/label_sync.py pilot-plano     # one campaign
  python data/scripts/label_sync.py --pull          # download only (never upload)
  python data/scripts/label_sync.py --dry-run

Credentials: AZURE_LABELS_CONNECTION_STRING in data/.env, a connection string whose SAS is scoped to
the parking-labels-dev container with read, add, create, write and list permissions. Falls back to
AZURE_STORAGE_CONNECTION_STRING when that is an account-level key (the parquets-dev SAS is scoped
to its own container and cannot reach this one).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labels"
ENV_FILE = ROOT / "data" / ".env"
CAMPAIGN_FILES = ["campaign.json", "squares.geojson", "prefill.geojson", "parcels.geojson"]
CONTAINER = os.environ.get("LABELS_CONTAINER", "parking-labels-dev")


def load_env(path: Path) -> None:
    if not path.exists():
        sys.exit(f"Missing {path}\nCreate it with: AZURE_STORAGE_CONNECTION_STRING=...")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def parse_jsonl(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  skipping malformed line: {line[:80]}")
    return out


def merge_labels(local: list[dict], remote: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in local + remote:
        if r.get("id") in seen:
            continue
        seen.add(r.get("id"))
        out.append(r)
    return sorted(out, key=lambda r: (r.get("submitted", ""), r.get("id", "")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaigns", nargs="*")
    ap.add_argument("--pull", action="store_true", help="download only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    load_env(ENV_FILE)
    conn = os.environ.get("AZURE_LABELS_CONNECTION_STRING") or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        sys.exit("AZURE_LABELS_CONNECTION_STRING not found in data/.env")
    try:
        from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError
        from azure.storage.blob import BlobServiceClient, ContentSettings
    except ImportError:
        sys.exit("Missing dependency. Run:  python -m pip install azure-storage-blob")
    cc = BlobServiceClient.from_connection_string(conn, retry_total=8).get_container_client(CONTAINER)

    try:
        remote_names = {b.name for b in cc.list_blobs()}
    except ClientAuthenticationError:
        sys.exit(f"Azure refused access to container {CONTAINER!r}. Add AZURE_LABELS_CONNECTION_STRING to "
                 "data/.env: a connection string with a SAS for that container (permissions: read, add, "
                 "create, write, list).")
    remote_campaigns = {n.split("/", 1)[0] for n in remote_names if "/" in n}
    local_campaigns = {p.name for p in LABELS.glob("*") if (p / "campaign.json").exists()} if LABELS.exists() else set()
    todo = a.campaigns or sorted(local_campaigns | remote_campaigns)

    def download(name: str) -> bytes | None:
        try:
            return cc.download_blob(name).readall()
        except ResourceNotFoundError:
            return None

    def upload(name: str, data: bytes, ctype: str) -> None:
        if a.dry_run or a.pull:
            print(f"  would upload {name} ({len(data):,} bytes)" if a.dry_run else f"  (pull only) {name} differs")
            return
        cc.upload_blob(name, data, overwrite=True, content_settings=ContentSettings(content_type=ctype))
        print(f"  uploaded {name} ({len(data):,} bytes)")

    for c in todo:
        print(f"== {c}")
        d = LABELS / c
        if not a.dry_run:
            d.mkdir(parents=True, exist_ok=True)
        for f in CAMPAIGN_FILES:
            name, p = f"{c}/{f}", d / f
            if p.exists():
                data = p.read_bytes()
                if name in remote_names:
                    rem = download(name)
                    if rem is not None and hashlib.sha256(rem).digest() == hashlib.sha256(data).digest():
                        continue
                upload(name, data, "application/json" if f.endswith(".json") else "application/geo+json")
            elif name in remote_names:
                if a.dry_run:
                    print(f"  would download {name}")
                else:
                    p.write_bytes(download(name) or b"")
                    print(f"  downloaded {name}")
        # labels: union by id
        name, p = f"{c}/labels.jsonl", d / "labels.jsonl"
        local = parse_jsonl(p.read_text(encoding="utf-8")) if p.exists() else []
        rem_bytes = download(name) if name in remote_names else None
        remote = parse_jsonl(rem_bytes.decode("utf-8")) if rem_bytes else []
        merged = merge_labels(local, remote)
        body = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in merged)
        new_local = len(merged) - len(local)
        new_remote = len(merged) - len(remote)
        print(f"  labels: local {len(local)}, remote {len(remote)} -> {len(merged)} "
              f"(+{new_local} to local, +{new_remote} to Azure)")
        if new_local and not a.dry_run:
            # Rewrite atomically so a running label API never reads a half-written file.
            tmp = p.with_suffix(".jsonl.tmp")
            tmp.write_text(body, encoding="utf-8")
            tmp.replace(p)
        if new_remote:
            upload(name, body.encode("utf-8"), "application/x-ndjson")
    return 0


if __name__ == "__main__":
    sys.exit(main())
