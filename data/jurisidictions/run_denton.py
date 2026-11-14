#!/usr/bin/env python3
"""
Build Denton, TX's canonical parcel parquet: the whole city, within city limits
(255 km2, all in Denton County).

Sources:
- Denton County: Denton CAD's 2026 certified export joined on prop_id to its certified parcel
  polygons (Parcels_Certified_2026). See denton_county.py. Entity C05.
City limits: Census TIGER/Line 2025 place 4819972. Shared runner: tx_multicounty.py.
"""
from denton_county import denton_link, denton_rows
from tx_multicounty import build_city

CITY, STATE = "denton", "tx"
PUBLIC = ["CITY OF DENTON", "DENTON CITY OF", "DENTON INDEPENDENT SCHOOL", "DENTON ISD",
          "ARGYLE INDEPENDENT", "SANGER INDEPENDENT", "KRUM INDEPENDENT", "DENTON COUNTY OF",
          "COUNTY OF DENTON", "UNIVERSITY OF NORTH TEXAS", "TEXAS WOMANS UNIVERSITY", "TEXAS WOMAN'S UNIVERSITY",
          "DENTON COUNTY TRANSPORTATION AUTHORITY", "DENTON HOUSING", "UPPER TRINITY REGIONAL WATER", "TEXAS DEPARTMENT OF TRANSPORTATION",
          "STATE OF TEXAS", "UNITED STATES OF AMERICA", "NORTH TEXAS TOLLWAY AUTHORITY", "NORTH TEXAS MUNICIPAL WATER"]

if __name__ == "__main__":
    build_city(CITY, STATE, "4819972", [
        ("Denton", lambda b, d: denton_rows(b, d, city=CITY, state=STATE, city_entity="C05"),
         denton_link),
    ], PUBLIC)
