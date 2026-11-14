#!/usr/bin/env python3
"""
Build Plano, TX's canonical parcel parquet: the whole city, within city limits
(180 km2 in Collin County, 6.7 km2 (the west-edge strip) in Denton County).

Sources:
- Collin County: Collin CAD's public parcel layer (CCAD Parcel Feature Set); 2026 certified values
  from its prior-year fields (the layer has rolled to 2027). See collin_county.py. City code CPL.
- Denton County: Denton CAD's 2026 certified export joined on prop_id to its certified parcel
  polygons (Parcels_Certified_2026). See denton_county.py. Entity C29.
City limits: Census TIGER/Line 2025 place 4858016. Shared runner: tx_multicounty.py.
"""
from collin_county import ccad_link, ccad_rows
from denton_county import denton_link, denton_rows
from tx_multicounty import build_city

CITY, STATE = "plano", "tx"
PUBLIC = ["CITY OF PLANO", "PLANO CITY OF", "PLANO INDEPENDENT SCHOOL", "PLANO ISD",
          "FRISCO INDEPENDENT SCHOOL", "LEWISVILLE INDEPENDENT", "COLLIN COUNTY OF", "COUNTY OF COLLIN",
          "COLLIN COLLEGE", "COLLIN COUNTY COMMUNITY COLLEGE", "DALLAS AREA RAPID TRANSIT", "PLANO HOUSING",
          "TEXAS DEPARTMENT OF TRANSPORTATION", "STATE OF TEXAS", "UNITED STATES OF AMERICA", "NORTH TEXAS TOLLWAY AUTHORITY",
          "NORTH TEXAS MUNICIPAL WATER"]

if __name__ == "__main__":
    build_city(CITY, STATE, "4858016", [
        ("Collin", lambda b, d: ccad_rows(b, d, city_code="CPL"), ccad_link),
        ("Denton", lambda b, d: denton_rows(b, d, city=CITY, state=STATE, city_entity="C29"),
         denton_link),
    ], PUBLIC)
