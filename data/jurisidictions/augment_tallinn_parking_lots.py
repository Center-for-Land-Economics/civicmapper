"""
Tag Tallinn PARCELS that host surface parking as `property_land_use_refined = "Parking Lot"`
and carry the per-parcel surface-parking FOOTPRINT value + area onto the parcel parquet, so the
Underused tab's "Parking Lot" line reports the SAME footprint-based number the Parking tab does
(they then agree automatically).

Why footprint (not full parcel) value: surface parking is a sub-parcel thing. The Parking tab
sums the land value *under each parking polygon* (footprint × ppsf). Summing whole-parcel values
in the Underused tab would double-count developed parcels that merely have an ancillary lot. So
we carry the footprint value per parcel and the Underused tab sums THAT for the Parking Lot
category — matching the Parking tab's surface total exactly and covering every surface lot.

To match the Parking-tab headline field-for-field, we aggregate per parcel from SURFACE lots:
  parking_footprint_land_value  <- sum(effective_surface_land_value | estimated_parking_land_value)
  parking_footprint_area_sqft   <- sum(surface_area_sqft | parking_area_sqft)
(these are the exact fields parking.ts sums for "X acres worth €Y sits under surface parking").

Every parcel with any surface parking is tagged "Parking Lot" ("Parking Lot" wins over "Vacant").
Reads the joined parking parquet; rewrites the parcel parquet in place (backup *.preparking.parquet).
Run AFTER the parking pipeline; re-upload the parcel parquet afterward.
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
DEFAULT_PARCELS = REPO_ROOT / "data" / "jurisidictions" / "data" / "tallinn" / "tallinn-harju-ee-parcels.parquet"
DEFAULT_PARKING = REPO_ROOT / "data" / "parking" / "tallinn" / "tallinn-harju-ee-parking-lots.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parcels", type=Path, default=DEFAULT_PARCELS)
    ap.add_argument("--parking", type=Path, default=DEFAULT_PARKING)
    args = ap.parse_args()

    for p in (args.parcels, args.parking):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    pk = pd.read_parquet(args.parking)
    surf = pk[pk["parking_type"] == "surface"].copy()

    def num(col):
        return pd.to_numeric(surf[col], errors="coerce") if col in surf else pd.Series(np.nan, index=surf.index)

    # Match the Parking-tab headline fields exactly (effective_* preferred, fall back to raw).
    eff = num("effective_surface_land_value")
    surf["fval"] = eff.where(eff.notna(), num("estimated_parking_land_value")).fillna(0.0)
    a_eff = num("surface_area_sqft")
    surf["farea"] = a_eff.where(a_eff.notna(), num("parking_area_sqft")).fillna(0.0)

    agg = surf.groupby("parcel_id").agg(fval=("fval", "sum"), farea=("farea", "sum")).reset_index()
    agg["parcel_id"] = agg["parcel_id"].astype(str)
    print(f"Surface parking on {len(agg):,} parcels; "
          f"footprint value EUR {agg['fval'].sum()/1e6:,.1f}M, area {agg['farea'].sum()/43560:,.0f} ac", flush=True)

    g = gpd.read_parquet(args.parcels)
    backup = args.parcels.with_suffix(".preparking.parquet")
    if not backup.exists():
        shutil.copy2(args.parcels, backup)
        print(f"Backed up -> {backup}", flush=True)

    # Start the refined field from its non-parking base (Vacant/None) so re-runs are idempotent:
    # anything previously set to "Parking Lot" reverts, then we re-tag from the current analysis.
    g.loc[g["property_land_use_refined"] == "Parking Lot", "property_land_use_refined"] = None

    fv = dict(zip(agg["parcel_id"], agg["fval"]))
    fa = dict(zip(agg["parcel_id"], agg["farea"]))
    pid = g["parcel_id"].astype(str)
    g["parking_footprint_land_value"] = pid.map(fv).fillna(0.0)
    g["parking_footprint_area_sqft"] = pid.map(fa).fillna(0.0)

    is_lot = g["parking_footprint_land_value"] > 0
    g.loc[is_lot, "property_land_use_refined"] = "Parking Lot"   # wins over "Vacant"
    print(f"\nTagged {int(is_lot.sum()):,} parcels as Parking Lot (any surface parking).")
    print("refined:", g["property_land_use_refined"].value_counts(dropna=False).to_dict())

    g.to_parquet(args.parcels, index=False)
    print(f"Wrote {args.parcels}", flush=True)


if __name__ == "__main__":
    main()
