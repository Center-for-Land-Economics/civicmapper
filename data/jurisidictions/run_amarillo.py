#!/usr/bin/env python3
"""
Build Amarillo, TX's canonical parcel parquet: the whole city, within city limits (187 km2 in
Potter County, 93 km2 in Randall County).

Source: Potter-Randall Appraisal District's 2026 certified exports (one per county) joined on
prop_id to PRAD's 2026 certified GIS parcels; see prad_county.py. City of Amarillo entities:
FP (Potter), FR (Randall). City limits: Census TIGER/Line 2025 place 4803000.
Shared runner: tx_multicounty.py.
"""
from prad_county import prad_link, prad_rows
from tx_multicounty import build_city

CITY, STATE = "amarillo", "tx"
PUBLIC = ["CITY OF AMARILLO", "AMARILLO CITY OF", "AMARILLO INDEPENDENT SCHOOL", "AMARILLO ISD",
          "CANYON INDEPENDENT SCHOOL", "CANYON ISD", "RIVER ROAD INDEPENDENT", "HIGHLAND PARK INDEPENDENT",
          "BUSHLAND INDEPENDENT", "POTTER COUNTY OF", "COUNTY OF POTTER", "RANDALL COUNTY OF",
          "COUNTY OF RANDALL", "AMARILLO COLLEGE", "AMARILLO JUNIOR COLLEGE", "AMARILLO HOSPITAL DISTRICT",
          "AMARILLO HOUSING AUTHORITY", "TEXAS TECH UNIVERSITY", "WEST TEXAS A&M", "TEXAS DEPARTMENT OF TRANSPORTATION",
          "STATE OF TEXAS", "UNITED STATES OF AMERICA", "AMARILLO ECONOMIC DEVELOPMENT"]

if __name__ == "__main__":
    build_city(CITY, STATE, "4803000", [
        ("Potter", lambda b, d: prad_rows(b, d, county="Potter", city=CITY, state=STATE, city_entity="FP"),
         prad_link),
        ("Randall", lambda b, d: prad_rows(b, d, county="Randall", city=CITY, state=STATE, city_entity="FR"),
         prad_link),
    ], PUBLIC)
