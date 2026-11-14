#!/usr/bin/env python3
"""
Build Plano, TX's canonical parcel parquet: the whole city, within city limits (180 km2 in Collin
County, 6.7 km2 in Denton County).

Sources:
- Collin County: Collin CAD's public parcel layer (CCAD Parcel Feature Set), 2026 certified values
  from its prior-year fields (the layer has rolled to the 2027 roll year); see collin_county.py.
  City of Plano: entityCityCode CPL.
- Denton County: Denton CAD's 2026 certified export + parcel GIS, when placed in
  data/jurisidictions/data/denton-cad/ (not reachable from the build host). Until then the
  Denton County strip on the west edge is left out and logged.
City limits: Census TIGER/Line 2025 place 4858016.
"""
from pathlib import Path

import geopandas as gpd

from collin_county import ccad_link, ccad_rows
from tx_common import log
from tx_pipeline import process

CITY, STATE = "plano", "tx"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / CITY
TIGER = ROOT / "data" / "jurisidictions" / "data" / "elpaso" / "tl_2025_48_place.zip"

PUBLIC = ["CITY OF PLANO", "PLANO CITY OF", "PLANO INDEPENDENT SCHOOL", "PLANO ISD",
          "FRISCO INDEPENDENT SCHOOL", "LEWISVILLE INDEPENDENT", "COLLIN COUNTY OF", "COUNTY OF COLLIN",
          "COLLIN COLLEGE", "COLLIN COUNTY COMMUNITY COLLEGE", "DALLAS AREA RAPID TRANSIT",
          "NORTH TEXAS MUNICIPAL WATER", "NORTH TEXAS TOLLWAY AUTHORITY", "PLANO HOUSING",
          "TEXAS DEPARTMENT OF TRANSPORTATION", "STATE OF TEXAS", "UNITED STATES OF AMERICA"]


def load_boundary():
    cache = DATA_DIR / f"{CITY}-{STATE}-boundary.parquet"
    if not cache.exists():
        g = gpd.read_file(f"zip://{TIGER}")
        g[g["GEOID"] == "4858016"].to_crs(4326).to_parquet(cache)
    return gpd.read_parquet(cache).union_all()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary()
    log("=== Collin County (CCAD) ===")
    raw, off_roll, coded, exempt_fn = ccad_rows(boundary, DATA_DIR, city_code="CPL")
    log("=== Denton County: no Denton CAD data yet; the city's Denton strip is left out ===")
    process(raw, off_roll, boundary, city=CITY, state=STATE, data_dir=DATA_DIR,
            public_names=PUBLIC, exempt_fn=exempt_fn,
            link_fn=lambda county, acct: ccad_link(acct), coded=coded)


if __name__ == "__main__":
    main()
