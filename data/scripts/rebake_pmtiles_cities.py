#!/usr/bin/env python3
"""Rebake + reupload the PMTiles cities so their tiles carry `land_area_acres`.

Why: the land tab computes per-sqft HEIGHT client-side as value/(land_area_acres*43560) on the
parcel layer AND the low-zoom H3 hexes. Cities baked before that schema lack `land_area_acres` on
the hex layer (and older ETLs lack it on parcels too) → they render FLAT. parquet_to_pmtiles.py now
derives `land_area_acres` from geometry when missing, so a plain rebake fixes every city. Houston is
already correct and is excluded by default.

For each city this:
  1. runs  parquet_to_pmtiles.py --city <city> --upload --overwrite  (+ --drop-remnants for cities
     with hideRemnants:true in cities.ts, + --wsl on Windows) → rebuilds & uploads .pmtiles + metadata
  2. on success, bumps that city's `pmtilesVersion` in viz/src/cities.ts so the deployed proxy/CDN
     refetches the new tiles.

Reads AZURE_STORAGE_CONNECTION_STRING from data/.env (same as upload_city_dev.py). Continue-on-error;
re-run with --only <list> to retry failures.

Usage:
  python data/scripts/rebake_pmtiles_cities.py                 # all 21 stale cities
  python data/scripts/rebake_pmtiles_cities.py --only chicago,denver
  python data/scripts/rebake_pmtiles_cities.py --dry-run
  python data/scripts/rebake_pmtiles_cities.py --version 2026-06-18-landacres
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

# Never crash on a ✓/✗/— glyph if the console codepage isn't UTF-8 (Windows cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:
    pass

REPO = Path(__file__).resolve().parents[2]
BAKE = REPO / "data" / "scripts" / "parquet_to_pmtiles.py"
UPLOAD_CITY = REPO / "data" / "upload_city_dev.py"
CITIES_DIR = REPO / "viz" / "src" / "cities"
ENV_FILE = REPO / "data" / ".env"

# Every PMTiles city except Houston (already on the current schema). Verified 2026-06-17.
DEFAULT_CITIES = [
    "albuquerque", "austin", "baltimore", "bcs", "bellingham", "chicago", "cincinnati",
    "cleveland", "dallas", "denver", "detroit", "fortcollins", "morgantown", "nyc",
    "portland", "pueblo", "rochester", "sanantonio", "spokane", "stpaul", "syracuse",
]


def find_parquet(city: str) -> Path | None:
    """The city's source parcel parquet under data/jurisidictions/data/<city>/, by glob (robust to
    parquet_registry's stale legacy_filename values). Returns None if absent (e.g. nyc)."""
    d = REPO / "data" / "jurisidictions" / "data" / city
    cands = sorted(d.glob("*-parcels.parquet"))
    return cands[0] if cands else None


def find_tiles(city: str) -> tuple[Path | None, Path | None]:
    """The city's already-baked .pmtiles + metadata.json under data/jurisidictions/data/<city>/."""
    d = REPO / "data" / "jurisidictions" / "data" / city
    pm = sorted(d.glob("*-parcels.pmtiles"))
    md = sorted(d.glob("*-parcels-metadata.json"))
    return (pm[0] if pm else None, md[0] if md else None)


def upload_tiles(city: str, env: dict) -> bool:
    """Push the already-baked .pmtiles + metadata directly to the dev blob — no rebake, no
    parquet_registry dependency (so it works for ad-hoc cities like `harris`). Uses the same
    robust chunked/retrying BlobServiceClient config as parquet_to_pmtiles' uploader."""
    pm, md = find_tiles(city)
    if pm is None or md is None:
        print(f"  ✗ {city}: missing baked tiles "
              f"(pmtiles={pm is not None}, metadata={md is not None}) under "
              f"data/jurisidictions/data/{city}/")
        return False
    conn = env.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        print("  ✗ AZURE_STORAGE_CONNECTION_STRING not set (data/.env)")
        return False
    container = env.get("AZURE_DEV_CONTAINER", "parquets-dev")
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        print("  ✗ azure-storage-blob not installed")
        return False
    MB = 1024 * 1024
    svc = BlobServiceClient.from_connection_string(
        conn, max_single_put_size=4 * MB, max_block_size=4 * MB,
        connection_timeout=300, read_timeout=600, retry_total=8,
    )
    cc = svc.get_container_client(container)
    try:
        for f in (pm, md):
            print(f"  uploading {f.name} ({f.stat().st_size / MB:.1f} MB) -> {container} ...")
            with f.open("rb") as h:
                cc.get_blob_client(f.name).upload_blob(h, overwrite=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {city}: upload error: {e}")
        return False


def load_env_into(env: dict) -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def city_json_path(city: str) -> Path:
    return CITIES_DIR / f"{city}.json"


def remnant_cities(cities: list[str]) -> set[str]:
    out = set()
    for c in cities:
        p = city_json_path(c)
        if p.exists() and json.loads(p.read_text(encoding="utf-8")).get("hideRemnants") is True:
            out.add(c)
    return out


def bump_version(city: str, version: str) -> str:
    """Set pmtilesVersion in viz/src/cities/<city>.json. Returns a status string."""
    p = city_json_path(city)
    if not p.exists():
        return "city-json-not-found"
    d = json.loads(p.read_text(encoding="utf-8"))
    if "pmtilesFilename" not in d:
        return "not-a-pmtiles-city"
    d["pmtilesVersion"] = version
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=f"{datetime.date.today().isoformat()}-landacres",
                    help="New pmtilesVersion stamp written per city on success.")
    ap.add_argument("--only", help="Comma-separated subset of cities to process.")
    ap.add_argument("--skip", help="Comma-separated cities to exclude.")
    ap.add_argument("--include-houston", action="store_true", help="Also rebake Houston (already current).")
    ap.add_argument("--wsl", action=argparse.BooleanOptionalAction, default=(platform.system() == "Windows"),
                    help="Pass --wsl to the bake (tippecanoe/pmtiles in WSL). Default: on for Windows.")
    ap.add_argument("--no-version-bump", action="store_true", help="Don't edit cities.ts.")
    ap.add_argument("--upload-only", action="store_true",
                    help="Skip baking; upload each city's ALREADY-BAKED local .pmtiles + metadata via "
                         "upload_city_dev.py (robust chunked client), then bump version. Use for cities "
                         "whose bake succeeded but the upload failed — don't use on cities that didn't bake.")
    ap.add_argument("--no-upload", action="store_true",
                    help="Bake only, don't pass --upload to parquet_to_pmtiles. Use when the blob SAS is "
                         "expired/unavailable: bake the fleet now, then push with --upload-only later.")
    ap.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    args = ap.parse_args()

    cities = [c.strip() for c in args.only.split(",")] if args.only else list(DEFAULT_CITIES)
    if args.include_houston and "houston" not in cities:
        cities.append("houston")
    if args.skip:
        skip = {c.strip() for c in args.skip.split(",")}
        cities = [c for c in cities if c not in skip]

    remnants = remnant_cities(cities)
    env = dict(os.environ)
    load_env_into(env)
    # Force UTF-8 in the child bake: parquet_to_pmtiles.py prints ✅/✗ glyphs, which crash on a
    # cp1252 stdout (e.g. when this script runs as a background task whose output file is cp1252).
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if not args.dry_run and not env.get("AZURE_STORAGE_CONNECTION_STRING"):
        print("WARNING: AZURE_STORAGE_CONNECTION_STRING not set (data/.env / env). --upload will fail.")

    mode = "Uploading (no rebake)" if args.upload_only else "Rebaking"
    print(f"{mode} {len(cities)} cities -> pmtilesVersion '{args.version}' "
          f"(wsl={args.wsl}, drop-remnants for: {sorted(remnants) or 'none'})\n")

    ok, failed = [], []
    for i, city in enumerate(cities, 1):
        if args.upload_only:
            # Push the ALREADY-BAKED local .pmtiles + metadata directly (no rebake, no
            # parquet_registry dependency — works for ad-hoc cities like `harris`).
            pm, md = find_tiles(city)
            print(f"[{i}/{len(cities)}] {city}\n  upload-only: {pm} + {md}")
            if args.dry_run:
                continue
            success = upload_tiles(city, env)
        else:
            # Pass the parquet explicitly (--file): parquet_to_pmtiles --city resolution relies on
            # parquet_registry.legacy_filename, which is stale/wrong for several cities (e.g. it
            # expects 'bellingham.parquet' but the file is 'bellingham-wa-parcels.parquet'). Globbing
            # the real file is robust regardless. --city is still passed for correct output naming.
            pq = find_parquet(city)
            if pq is None:
                print(f"[{i}/{len(cities)}] {city}\n  ✗ no source parquet under "
                      f"data/jurisidictions/data/{city}/ — skipping")
                failed.append(city)
                continue
            cmd = [sys.executable, str(BAKE), "--city", city, "--file", str(pq), "--overwrite"]
            if not args.no_upload:
                cmd.append("--upload")
            if city in remnants:
                cmd.append("--drop-remnants")
            if args.wsl:
                cmd.append("--wsl")
            print(f"[{i}/{len(cities)}] {city}\n  $ {' '.join(cmd)}")
            if args.dry_run:
                continue
            success = subprocess.run(cmd, cwd=REPO, env=env).returncode == 0
        if not success:
            verb = "upload" if args.upload_only else "bake/upload"
            print(f"  ✗ {city}: {verb} failed — skipping version bump")
            failed.append(city)
            continue
        did = "uploaded" if args.upload_only else "rebaked + uploaded"
        if not args.no_version_bump:
            status = bump_version(city, args.version)  # persists incrementally per city
            if status == "ok":
                print(f"  ✓ {city}: {did}, pmtilesVersion -> {args.version}")
            else:
                print(f"  ✓ {city}: {did}, but version bump skipped ({status})")
        else:
            print(f"  ✓ {city}: {did}")
        ok.append(city)

    print(f"\nDone. ok={len(ok)} failed={len(failed)}")
    if failed:
        print("Failed (retry with --only " + ",".join(failed) + "):", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
