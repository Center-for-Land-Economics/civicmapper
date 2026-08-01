#!/usr/bin/env python3
"""Patch per-region value/acre totals into the Houston PMTiles metadata.

The land-value tab's region widget can show/hide regions, and we want the headline blurb
("$Y of land value over X acres in Z") to update live as the selection changes. Houston is served
as PMTiles, so the parcels aren't in the browser's memory — only the metadata JSON is. That metadata
already carries per-region parcel *counts* (groups[field].counts) but not value/acre *sums*.

This reads the local parcel parquet, groups by each region field, sums land/improvement/total value
and acres, and writes them into metadata.groups[field].totals as:

    groups[<field>].totals = { <region name>: { "acres": .., "land": .., "impr": .., "total": .. } }

keyed by the SAME region names as groups[field].counts. No tile re-bake needed — re-upload only the
(small) metadata JSON afterwards.

Value sources (Houston / HCAD):
    land  <- current_full_land_value     impr  <- improvement_value
    total <- full_market_value

Acres come from the PROJECTED GEOMETRY, not the stored `land_area_acres` column — that column has
corrupt outliers (e.g. one parcel reading 3.4M acres, > all of Harris County), and the project
already treats geometry area as the source of truth for area (see parking_lot_extraction.py /
flag_remnants.py). Geometry is projected to UTM 15N (EPSG:32615, meters) for Houston.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_PARQUET = HERE / "data" / "houston" / "houston-tx-parcels.parquet"
DEFAULT_METADATA = REPO_ROOT / "viz" / "public" / "houston-tx-parcels-metadata.json"

FIELDS = ["jurisdiction", "council_district", "super_neighborhood", "civic_club"]
LAND_COL, IMPR_COL, TOTAL_COL = "current_full_land_value", "improvement_value", "full_market_value"
AREA_CRS = "EPSG:32615"          # UTM zone 15N (meters) — covers Houston
SQM_PER_ACRE = 4046.8564224


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = ap.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"Missing parquet: {args.parquet}")
    if not args.metadata.exists():
        raise SystemExit(f"Missing metadata: {args.metadata}")

    log(f"Reading {args.parquet} ...")
    df = gpd.read_parquet(args.parquet)
    log(f"  {len(df):,} parcels, crs={df.crs}")
    for c in (LAND_COL, IMPR_COL, TOTAL_COL):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # Acres from projected geometry (robust to the corrupt land_area_acres column).
    log(f"Computing acres from geometry ({AREA_CRS}) ...")
    df["acres"] = df.geometry.to_crs(AREA_CRS).area / SQM_PER_ACRE
    df["acres"] = pd.to_numeric(df["acres"], errors="coerce").fillna(0.0)
    log(f"  total acres = {df['acres'].sum():,.0f} (median parcel {df['acres'].median():.3f} ac)")
    ACRES_COL = "acres"

    meta = json.loads(args.metadata.read_text())
    groups = meta.setdefault("groups", {})

    for field in FIELDS:
        if field not in df.columns:
            log(f"  skip {field}: not in parquet")
            continue
        g = df.groupby(df[field].astype(str))
        agg = g.agg(acres=(ACRES_COL, "sum"), land=(LAND_COL, "sum"),
                    impr=(IMPR_COL, "sum"), total=(TOTAL_COL, "sum"))
        totals = {
            name: {
                "acres": round(float(r.acres), 1),
                "land": round(float(r.land)),
                "impr": round(float(r.impr)),
                "total": round(float(r.total)),
            }
            for name, r in agg.iterrows()
        }
        groups.setdefault(field, {})["totals"] = totals
        top = sorted(totals.items(), key=lambda kv: kv[1]["total"], reverse=True)[:3]
        log(f"  {field}: {len(totals)} regions; top by total value: "
            + ", ".join(f"{n} (${t['total']/1e9:.1f}B, {t['acres']:,.0f}ac)" for n, t in top))

    args.metadata.write_text(json.dumps(meta))
    log(f"\nWrote {args.metadata}")
    log("Next: re-upload houston-tx-parcels-metadata.json to the parquets-dev blob (parking/parcel upload flow).")


if __name__ == "__main__":
    main()
