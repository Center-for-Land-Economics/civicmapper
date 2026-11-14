#!/usr/bin/env python3
"""
Build Laredo, TX's canonical parcel parquet: the whole city, within city limits (286 km2, all in
Webb County).

Source: Webb CAD's 2026 certified export joined on geo_id to the City of Laredo's public parcel
layer; see webb_county.py. City of Laredo entity: C1. City limits: Census TIGER/Line 2025 place
4841464. Shared runner: tx_multicounty.py.
"""
from tx_multicounty import build_city
from webb_county import webb_link, webb_rows

CITY, STATE = "laredo", "tx"
PUBLIC = ["CITY OF LAREDO", "LAREDO CITY OF", "LAREDO INDEPENDENT SCHOOL", "LAREDO ISD",
          "UNITED INDEPENDENT SCHOOL", "UNITED ISD", "WEBB COUNTY OF", "COUNTY OF WEBB",
          "LAREDO COLLEGE", "LAREDO COMMUNITY COLLEGE", "TEXAS A&M INTERNATIONAL", "TEXAS A & M INTERNATIONAL",
          "LAREDO HOUSING AUTHORITY", "HOUSING AUTHORITY OF THE CITY OF LAREDO", "WEBB COUNTY APPRAISAL",
          "TEXAS DEPARTMENT OF TRANSPORTATION", "STATE OF TEXAS", "UNITED STATES OF AMERICA",
          "LAREDO MUNICIPAL HOUSING"]

if __name__ == "__main__":
    build_city(CITY, STATE, "4841464", [
        ("Webb", lambda b, d: webb_rows(b, d, city=CITY, state=STATE, city_entity="C1"), webb_link),
    ], PUBLIC)
