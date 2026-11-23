#!/usr/bin/env python3
"""
Build the Allegheny County, PA canonical parcel parquet.

This is the whole county, not the City of Pittsburgh. Pittsburgh is one
municipality inside it (tax wards 101-132). East Pittsburgh, the other
boroughs, and the townships stay in.

The geometry layer is already countywide:
  https://services5.arcgis.com/eYH3NsaQKd0zpREV/ArcGIS/rest/services/Allegheny_County_Parcels/FeatureServer/0
Values come from the same WPRDC assessment extract as Pittsburgh, joined on
PARID = PIN. There is no city clip. Taxability, condo merging, and the
coincident-card collapse are the same rules as run_pittsburgh.py.

Outputs:
  data/jurisidictions/data/allegheny/allegheny-pa-parcels.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_pittsburgh as pg  # noqa: E402

DATA_DIR = HERE / "data" / "allegheny"
GEOM_CACHE = DATA_DIR / "allegheny-pa-geometry.parquet"
# Reuse the county assessment download the Pittsburgh ETL already cached.
SHARED_ASSESS = HERE / "data" / "pittsburgh" / "allegheny-assessments.csv"


def fetch_county_geometry() -> gpd.GeoDataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if GEOM_CACHE.exists():
        pg.log(f"Using cached county geometry: {GEOM_CACHE.name}")
        geometry = gpd.read_parquet(GEOM_CACHE)
    else:
        geometry = pg._pull_where("1=1", "Allegheny County")
        geometry.to_parquet(GEOM_CACHE, index=False)
        pg.log(f"Cached {len(geometry):,} county parcels")
    geometry["MUNICODE"] = pd.to_numeric(geometry["MUNICODE"], errors="coerce")
    if not (500_000 <= len(geometry) <= 700_000):
        raise RuntimeError(
            f"Expected the full Allegheny County layer (~585k parcels), got {len(geometry):,}."
        )
    return geometry


def main() -> None:
    pg.log("Allegheny County: keeping the full countywide parcel layer. No Pittsburgh clip.")
    geometry = fetch_county_geometry()
    pg.ASSESS_CACHE = SHARED_ASSESS if SHARED_ASSESS.exists() else DATA_DIR / "allegheny-assessments.csv"
    assess = pg.fetch_assessments()
    pins = set(geometry["PIN"])
    roll = assess[assess["PARID"].isin(pins)].copy()
    pg.log(f"Assessments matched to county parcels: {len(roll):,} of {len(geometry):,} geometries")
    joined = geometry.merge(roll, left_on="PIN", right_on="PARID", how="inner", suffixes=("", "_roll"))
    if "MUNICODE_roll" in joined.columns:
        joined = joined.drop(columns=["MUNICODE_roll"])
    pg.log(f"Joined rows: {len(joined):,}")
    pg.log(f"Municipalities (MUNICODE): {joined['MUNICODE'].nunique(dropna=True):,}")
    wards = pd.to_numeric(joined["MUNICODE"], errors="coerce").between(101, 132)
    pg.log(f"  City of Pittsburgh wards 101-132: {int(wards.sum()):,}")
    pg.log(f"  rest of Allegheny County: {int((~wards).sum()):,}")
    joined = pg.collapse_duplicate_pins(joined)
    pg.export_parcels(joined, DATA_DIR, "allegheny-pa", place_column="municipality")


if __name__ == "__main__":
    main()
