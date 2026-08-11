"""
Metadata-only percentile refresh for already-baked PMTiles cities.

Adds/updates each city's robust percentiles (p1 / p99 / p999) + quantile breaks in its
`<slug>-parcels-metadata.json` WITHOUT re-baking the tiles. This is cheap and correct because the
percentiles are derived from the SOURCE PARQUET, not the tiles — so patching the JSON is all that's
needed to give an existing PMTiles city the new p99.9 colour-domain top (frontend reads
percentiles[field].p999; older tiles without it fall back to max). The computation is imported from
parquet_to_pmtiles.compute_render_percentiles — the single shared source of truth with the full bake.

Run:
  python data/scripts/refresh_metadata_percentiles.py                  # every city with a local metadata JSON
  python data/scripts/refresh_metadata_percentiles.py --city austin    # specific city(ies)
  python data/scripts/refresh_metadata_percentiles.py --city austin --upload   # + push JSON to the dev blob

IMPORTANT after uploading: bump that city's `pmtilesVersion` in viz/src/cities.ts. The frontend
appends it as a cache-bust to the metadata URL, so without a bump the CDN keeps serving the stale JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import geopandas as gpd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))              # data/scripts  → parquet_to_pmtiles
sys.path.insert(0, str(_HERE.parent))       # data/         → parquet_registry
from parquet_to_pmtiles import compute_render_percentiles, upload_file  # noqa: E402  (shared logic)
from parquet_registry import CITY_PARQUETS, list_cities                  # noqa: E402

DATA_DIR = _HERE.parent / "jurisidictions" / "data"


def paths_for(city_key: str) -> tuple[Path, Path]:
    """(parquet_path, metadata_json_path) for a city, matching the bake's on-disk naming."""
    meta = CITY_PARQUETS[city_key]
    cdir = DATA_DIR / meta.city
    ppath = cdir / meta.canonical_filename
    if not ppath.exists():
        legacy = cdir / meta.legacy_filename
        if legacy.exists():
            ppath = legacy
    mpath = cdir / meta.canonical_filename.replace(".parquet", "-metadata.json")
    return ppath, mpath


def refresh_city(city_key: str, do_upload: bool, container: str, conn: str | None) -> bool:
    ppath, mpath = paths_for(city_key)
    if not mpath.exists():
        print(f"  ⏭  {city_key}: no metadata JSON ({mpath.name}) — skip (not a baked PMTiles city here, "
              f"or download/bake it first)")
        return False
    if not ppath.exists():
        print(f"  ⚠  {city_key}: metadata present but source parquet missing ({ppath}) — skip")
        return False

    meta = json.loads(mpath.read_text(encoding="utf-8"))
    gdf = gpd.read_parquet(ppath)
    rows = compute_render_percentiles(gdf)
    if not rows:
        print(f"  ⚠  {city_key}: no render fields present in parquet — skip")
        return False

    meta.setdefault("percentiles", {})
    meta.setdefault("quantileBreaks", {})
    for field, pct, breaks in rows:
        meta["percentiles"][field] = pct
        meta["quantileBreaks"][field] = breaks

    backup = mpath.with_name(mpath.name + ".prep999")
    if not backup.exists():
        shutil.copy2(mpath, backup)
    mpath.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  ✅ {city_key}: patched {len(rows)} field(s) → {mpath.name}")
    for field, pct, _ in rows:
        print(f"       {field}: p99={pct['p99']:.1f}  p99.9={pct['p999']:.1f}")

    if do_upload:
        if not conn:
            print(f"     ↪ upload skipped (no AZURE_STORAGE_CONNECTION_STRING). Set it, or run "
                  f"`python data/upload_city_dev.py {city_key}` to push the patched metadata via the SAS uploader.")
        else:
            from azure.storage.blob import BlobServiceClient
            svc = BlobServiceClient.from_connection_string(conn, connection_timeout=120, read_timeout=300, retry_total=6)
            cc = svc.get_container_client(container)
            upload_file(cc, mpath, mpath.name, overwrite=True)
    return True


def _load_conn_string() -> str | None:
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn:
        return conn
    envf = _HERE.parent / ".env"
    if envf.exists():
        for line in envf.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("AZURE_STORAGE_CONNECTION_STRING="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", action="append", help="City key(s). Default: every city with a local metadata JSON.")
    ap.add_argument("--upload", action="store_true", help="Also upload the patched JSON (needs AZURE_STORAGE_CONNECTION_STRING).")
    ap.add_argument("--container", default="parquets-dev", help="Blob container (default: parquets-dev).")
    args = ap.parse_args()

    conn = _load_conn_string() if args.upload else None
    targets = args.city or list_cities()
    print(f"Refreshing metadata percentiles for {len(targets)} candidate city(ies) "
          f"(patch only{' + upload' if args.upload else ''})…")

    patched = 0
    for c in targets:
        if c not in CITY_PARQUETS:
            print(f"  ⏭  {c}: unknown city key")
            continue
        if refresh_city(c, args.upload, args.container, conn):
            patched += 1

    print(f"\nDone: {patched} city(ies) patched.")
    if patched and not args.upload:
        print("Next: upload the patched *-parcels-metadata.json (rerun with --upload, or via "
              "upload_city_dev.py) and BUMP that city's pmtilesVersion in viz/src/cities.ts "
              "(cache-bust) so the CDN serves the fresh metadata.")


if __name__ == "__main__":
    main()
