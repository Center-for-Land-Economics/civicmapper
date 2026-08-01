#!/usr/bin/env python3
"""Patch citywide value/acre totals (metadata['cityTotals']) into PMTiles metadata JSONs.

The Land Value blurb ("$Y of land value over X acres in <city>") needs totals the browser can't
compute for PMTiles cities (parcels aren't in memory). The bake now emits cityTotals itself
(parquet_to_pmtiles.add_region_value_totals), but every city baked before that only gets the blurb
when it has region groups WITH per-region totals. This script backfills cityTotals into an existing
metadata JSON from the local parcel parquet — no tile re-bake needed; re-upload only the (small)
metadata JSON afterwards:

    python data/scripts/patch_metadata_city_totals.py austin
    python data/scripts/patch_metadata_city_totals.py --all          # every city with both files local
    python data/scripts/patch_metadata_city_totals.py --parquet X.parquet --metadata Y.json

Then:  python data/upload_city_dev.py <city>

Logic mirrors the bake exactly:
    land  <- current_full_land_value, else REALLANDVA
    impr  <- improvement_value, else REALIMPROV, else 0 (combined/land-only cities)
    total <- full_market_value, else land + impr
    acres <- projected geometry area (equal-area EPSG:6933) — NOT the land_area_acres column,
             which carries corrupt outliers (see land-area-acres notes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from parquet_registry import list_cities, resolve_city  # noqa: E402

SQM_PER_ACRE = 4046.8564224


def log(msg: str) -> None:
    print(msg, flush=True)


def compute_city_totals(gdf: gpd.GeoDataFrame) -> dict | None:
    """Citywide {acres, land, impr, total}; None if no value columns (blurb stays hidden)."""
    if "current_full_land_value" in gdf.columns:
        land = pd.to_numeric(gdf["current_full_land_value"], errors="coerce")
    elif "REALLANDVA" in gdf.columns:
        land = pd.to_numeric(gdf["REALLANDVA"], errors="coerce")
    else:
        return None
    land = land.fillna(0.0)
    if "improvement_value" in gdf.columns:
        impr = pd.to_numeric(gdf["improvement_value"], errors="coerce").fillna(0.0)
    elif "REALIMPROV" in gdf.columns:
        impr = pd.to_numeric(gdf["REALIMPROV"], errors="coerce").fillna(0.0)
    else:
        impr = pd.Series(0.0, index=gdf.index)
    total = pd.to_numeric(gdf["full_market_value"], errors="coerce").fillna(0.0) \
        if "full_market_value" in gdf.columns else (land + impr)
    acres = gdf.geometry.to_crs(6933).area / SQM_PER_ACRE
    return {"acres": round(float(acres.sum()), 1), "land": round(float(land.sum())),
            "impr": round(float(impr.sum())), "total": round(float(total.sum()))}


def patch_one(parquet: Path, metadata: Path) -> bool:
    log(f"Reading {parquet} ...")
    gdf = gpd.read_parquet(parquet)
    totals = compute_city_totals(gdf)
    if totals is None:
        log("  no value columns (REALLANDVA/current_full_land_value) — skipped")
        return False
    meta = json.loads(metadata.read_text())
    meta["cityTotals"] = totals
    metadata.write_text(json.dumps(meta))
    log(f"  cityTotals: ${totals['total']/1e9:.1f}B total (${totals['land']/1e9:.1f}B land) "
        f"over {totals['acres']:,.0f} acres ({len(gdf):,} parcels)")
    log(f"  wrote {metadata}")
    return True


def city_paths(key: str) -> tuple[Path, Path]:
    """Local parquet + metadata paths per the upload_city_dev.py conventions."""
    meta = resolve_city(key)
    base = REPO_ROOT / "data" / "jurisidictions" / "data"
    juris = next((base / n for n in (key, meta.city) if (base / n).exists()), base / key)
    stem = f"{meta.city}-{meta.state}-parcels"
    return juris / meta.canonical_filename, juris / f"{stem}-metadata.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("city", nargs="?", help="City key from parquet_registry (e.g. austin).")
    ap.add_argument("--all", action="store_true", help="Patch every registry city with both files local.")
    ap.add_argument("--parquet", type=Path, help="Explicit parquet path (overrides city lookup).")
    ap.add_argument("--metadata", type=Path, help="Explicit metadata JSON path (overrides city lookup).")
    args = ap.parse_args()

    if args.parquet or args.metadata:
        if not (args.parquet and args.metadata):
            raise SystemExit("--parquet and --metadata must be given together")
        if not args.parquet.exists() or not args.metadata.exists():
            raise SystemExit("parquet or metadata path missing")
        patch_one(args.parquet, args.metadata)
        return

    keys = list_cities() if args.all else ([args.city] if args.city else [])
    if not keys:
        raise SystemExit("Give a city key, --all, or --parquet/--metadata. Known: " + ", ".join(list_cities()))
    patched, skipped = [], []
    for key in keys:
        pq, md = city_paths(key)
        if not pq.exists() or not md.exists():
            skipped.append(key)
            if not args.all:
                raise SystemExit(f"{key}: missing {pq if not pq.exists() else md}")
            continue
        log(f"\n=== {key} ===")
        if patch_one(pq, md):
            patched.append(key)
    if args.all:
        log(f"\nPatched: {', '.join(patched) or '(none)'}")
        log(f"Skipped (files not local): {', '.join(skipped) or '(none)'}")
    if patched:
        log("\nNext: re-upload each city's metadata JSON — python data/upload_city_dev.py <city>")


if __name__ == "__main__":
    main()
