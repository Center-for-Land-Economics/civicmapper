"""Promote data artifacts from parquets-dev -> parquets-prod (server-side copy).

Mirrors the dev container into prod for every blob that is MISSING or a different
SIZE in prod (so it's idempotent and only does the work that's needed). This is the
quick "promote everything we forgot" tool — it copies whatever is in dev regardless
of per-city filename quirks (e.g. ibx -> nyc-ibx-parcels, stpaul parking naming),
so there's nothing to keep in sync by hand.

Source is the PUBLIC parquets-dev URL (anonymous read), so only prod WRITE is needed.
Reads AZURE_STORAGE_CONNECTION_STRING from the env var or data/.env (must have write
on parquets-prod — the same credential you used for `upload_city_dev.py --container parquets-prod`).

Usage:
  python data/promote_to_prod.py                 # sync ALL missing/changed blobs
  python data/promote_to_prod.py dallas sanantonio austin   # only blobs matching these names
  python data/promote_to_prod.py --dry-run       # show what WOULD copy, don't copy
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

from azure.storage.blob import BlobServiceClient

DEV_CONTAINER = "parquets-dev"
PROD_CONTAINER = "parquets-prod"
DEV_PUBLIC_BASE = "https://landeconomics.blob.core.windows.net/parquets-dev"


def load_conn() -> str:
    c = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if c:
        return c.strip()
    envf = Path(__file__).with_name(".env")
    if envf.exists():
        m = re.search(r"AZURE_STORAGE_CONNECTION_STRING=(.+)", envf.read_text())
        if m:
            return m.group(1).strip()
    sys.exit("AZURE_STORAGE_CONNECTION_STRING not found (env var or data/.env)")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    filters = [a.lower() for a in args]

    svc = BlobServiceClient.from_connection_string(load_conn())
    dev = svc.get_container_client(DEV_CONTAINER)
    prod = svc.get_container_client(PROD_CONTAINER)

    dev_blobs = {b.name: b.size for b in dev.list_blobs() if not b.name.startswith("staging/")}
    prod_sizes = {b.name: b.size for b in prod.list_blobs()}

    names = sorted(dev_blobs)
    if filters:
        names = [n for n in names if any(f in n.lower() for f in filters)]

    todo = [n for n in names if prod_sizes.get(n) != dev_blobs[n]]
    skip = len(names) - len(todo)
    print(f"dev blobs: {len(dev_blobs)} | matched: {len(names)} | to copy: {len(todo)} | up-to-date: {skip}")
    if dry:
        for n in todo:
            print(f"  WOULD COPY {n}  ({dev_blobs[n]/1e6:.1f} MB)")
        return 0

    failed = []
    for i, name in enumerate(todo, 1):
        src = f"{DEV_PUBLIC_BASE}/{name}"
        dst = prod.get_blob_client(name)
        print(f"[{i}/{len(todo)}] copy {name} ({dev_blobs[name]/1e6:.1f} MB) ...", flush=True)
        dst.start_copy_from_url(src)  # server-side; dev is public so no source SAS needed
        status = "pending"
        for _ in range(600):
            status = dst.get_blob_properties().copy.status
            if status in ("success", "failed", "aborted"):
                break
            time.sleep(1)
        if status != "success":
            print(f"    !! {status}")
            failed.append(name)

    print(f"\nDone. copied={len(todo)-len(failed)} failed={len(failed)}")
    if failed:
        print("FAILED:", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
