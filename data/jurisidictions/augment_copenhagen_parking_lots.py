"""
Mirror the Parking tab into the Underused tab for Copenhagen (as done for Tallinn).

Two augmentations, run AFTER the parking pipeline (re-upload both parquets afterward):

1. PARCELS: tag every parcel hosting surface parking as `property_land_use_refined =
   "Parking Lot"` and carry the per-parcel surface-parking FOOTPRINT value + area onto the
   parcel parquet (parking_footprint_land_value / parking_footprint_area_sqft). Surface parking
   is a sub-parcel thing — the Parking tab sums the land value UNDER each lot (footprint × ppsf),
   so the Underused tab sums that same footprint value for the "Parking Lot" category and the two
   totals agree exactly (no double-counting developed parcels with an ancillary lot).
   "Parking Lot" wins over "Vacant"/"Underdeveloped".

2. PARKING: tag each parking feature with its parcel's `district` (parish/sogn), so the Parking
   tab's region panel groups by the same regions as the Value/Underused tabs. (Tallinn spatial-
   joined lot centroids to a boundary overlay; Copenhagen has no overlay geojson yet, but every
   lot already carries a parcel_id from the pipeline's spatial join, so we map parcel_id → district.)

Idempotent (reverts prior "Parking Lot" tags first). Backs up to *.preparking.parquet.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_PARCELS = REPO_ROOT / "data" / "jurisidictions" / "data" / "copenhagen" / "copenhagen-hovedstaden-dk-parcels.parquet"
DEFAULT_PARKING = REPO_ROOT / "data" / "parking" / "copenhagen" / "copenhagen-hovedstaden-dk-parking-lots.parquet"


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parcels", type=Path, default=DEFAULT_PARCELS)
    ap.add_argument("--parking", type=Path, default=DEFAULT_PARKING)
    args = ap.parse_args()
    for p in (args.parcels, args.parking):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    g = gpd.read_parquet(args.parcels)
    pk = gpd.read_parquet(args.parking)  # GeoDataFrame — so re-writing preserves the GeoParquet
                                         # "geo" metadata (pd.read/write would strip it → the
                                         # frontend then rejects the file as "missing geo metadata")

    # ── 2 (do first: needs parcel district before we touch anything) ──────────────
    #     Tag parking features with their parcel's district for the Parking-tab regions.
    dmap = dict(zip(g["parcel_id"].astype(str), g["district"]))
    pk_backup = args.parking.with_suffix(".preregions.parquet")
    if not pk_backup.exists():
        shutil.copy2(args.parking, pk_backup)
        print(f"Backed up parking -> {pk_backup}", flush=True)
    pk["district"] = pk["parcel_id"].astype(str).map(dmap)
    pk.to_parquet(args.parking, index=False)
    print(f"Tagged district on {pk['district'].notna().sum():,}/{len(pk):,} parking features.", flush=True)

    # ── 1: footprint value/area carry-over → parcels ──────────────────────────────
    surf = pk[pk["parking_type"] == "surface"].copy()
    eff = num(surf, "effective_surface_land_value")
    surf["fval"] = eff.where(eff.notna(), num(surf, "estimated_parking_land_value")).fillna(0.0)
    a_eff = num(surf, "surface_area_sqft")
    surf["farea"] = a_eff.where(a_eff.notna(), num(surf, "parking_area_sqft")).fillna(0.0)
    agg = surf.groupby("parcel_id").agg(fval=("fval", "sum"), farea=("farea", "sum")).reset_index()
    agg["parcel_id"] = agg["parcel_id"].astype(str)
    print(f"Surface parking on {len(agg):,} parcels; footprint value {agg['fval'].sum()/1e6:,.1f}M kr, "
          f"area {agg['farea'].sum()/43560:,.0f} ac", flush=True)

    backup = args.parcels.with_suffix(".preparking.parquet")
    if not backup.exists():
        shutil.copy2(args.parcels, backup)
        print(f"Backed up parcels -> {backup}", flush=True)

    # Idempotent: revert prior "Parking Lot" tags before re-tagging.
    g.loc[g["property_land_use_refined"] == "Parking Lot", "property_land_use_refined"] = None
    pid = g["parcel_id"].astype(str)
    full_fval = pid.map(dict(zip(agg["parcel_id"], agg["fval"]))).fillna(0.0)
    full_farea = pid.map(dict(zip(agg["parcel_id"], agg["farea"]))).fillna(0.0)
    # BROADCAST TRAP: the footprint value is aggregated per BFE, but 435 BFEs span multiple
    # jordstykke polygons. Mapping the full per-BFE value onto every polygon row would inflate
    # the Underused-tab sum N×. Area-allocate across the BFE's polygons (as done for land value
    # in run_copenhagen.allocate_values) so summing rows recovers the true per-BFE footprint.
    # allocate by physical geometry footprint (geom_area_sqm), NOT the assessed land_area_sqm
    share_col = "geom_area_sqm" if "geom_area_sqm" in g.columns else "land_area_sqm"
    bfe_area = g.groupby("parcel_id")[share_col].transform("sum")
    share = (pd.to_numeric(g[share_col], errors="coerce") / bfe_area).where(bfe_area > 0, 0.0).fillna(0.0)
    g["parking_footprint_land_value"] = full_fval * share
    g["parking_footprint_area_sqft"] = full_farea * share
    is_lot = g["parking_footprint_land_value"] > 0
    g.loc[is_lot, "property_land_use_refined"] = "Parking Lot"
    print(f"Tagged {int(is_lot.sum()):,} parcels as Parking Lot.")
    print("refined:", g["property_land_use_refined"].value_counts(dropna=False).to_dict())

    g.to_parquet(args.parcels, index=False)
    print(f"Wrote {args.parcels}", flush=True)


if __name__ == "__main__":
    main()
