#!/usr/bin/env python3
"""
Build Irving, TX's canonical parcel parquet: the whole city, within city limits (176 km2, all in
Dallas County).

Source: Dallas Central Appraisal District's 2026 certified data files (DCAD2026_CURRENT:
certified values + supplements, files of 2026-11-09) joined on GIS_PARCEL_ID to DCAD's parcel
polygons (PARCEL_GEOM, 2026-11); see dcad_county.py. Manual downloads from dallascad.org (not
reachable from the build host), shared by every Dallas County city in
data/jurisidictions/data/dcad/. City of Irving accounts: CITY_JURIS_DESC == IRVING.
City limits: Census TIGER/Line 2025 place 4837000.
"""
from pathlib import Path

import geopandas as gpd

from dcad_county import dcad_link, dcad_rows
from tx_common import log
from tx_pipeline import process

CITY, STATE = "irving", "tx"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / CITY
TIGER = ROOT / "data" / "jurisidictions" / "data" / "elpaso" / "tl_2025_48_place.zip"

PUBLIC = ["CITY OF IRVING", "IRVING CITY OF", "IRVING INDEPENDENT SCHOOL", "IRVING ISD",
          "CARROLLTON FARMERS BRANCH", "COPPELL INDEPENDENT", "DALLAS FORT WORTH INTERNATIONAL AIRPORT",
          "DFW INTERNATIONAL AIRPORT", "DALLAS FT WORTH INTL AIRPORT", "DALLAS COUNTY OF",
          "COUNTY OF DALLAS", "DALLAS AREA RAPID TRANSIT", "DALLAS COLLEGE", "DALLAS CO COMMUNITY COLLEGE",
          "NORTH TEXAS TOLLWAY AUTHORITY", "TRINITY RIVER AUTHORITY", "IRVING HOUSING",
          "TEXAS DEPARTMENT OF TRANSPORTATION", "STATE OF TEXAS", "UNITED STATES OF AMERICA",
          "CITY OF DALLAS", "DALLAS CITY OF"]


def load_boundary():
    cache = DATA_DIR / f"{CITY}-{STATE}-boundary.parquet"
    if not cache.exists():
        g = gpd.read_file(f"zip://{TIGER}")
        g[g["GEOID"] == "4837000"].to_crs(4326).to_parquet(cache)
    return gpd.read_parquet(cache).union_all()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary()
    log("=== Dallas County (DCAD) ===")
    raw, off_roll, coded, exempt_fn = dcad_rows(boundary, DATA_DIR, city_juris="IRVING")
    process(raw, off_roll, boundary, city=CITY, state=STATE, data_dir=DATA_DIR,
            public_names=PUBLIC, exempt_fn=exempt_fn,
            link_fn=lambda county, acct: dcad_link(acct), coded=coded)


if __name__ == "__main__":
    main()
