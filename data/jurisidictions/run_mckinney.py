#!/usr/bin/env python3
"""
Build McKinney, TX's canonical parcel parquet: the whole city, within city limits
(180 km2, all in Collin County).

Sources:
- Collin County: Collin CAD's public parcel layer (CCAD Parcel Feature Set); 2026 certified values
  from its prior-year fields (the layer has rolled to 2027). See collin_county.py. City code CMC.
City limits: Census TIGER/Line 2025 place 4845744. Shared runner: tx_multicounty.py.
"""
from collin_county import ccad_link, ccad_rows
from tx_multicounty import build_city

CITY, STATE = "mckinney", "tx"
PUBLIC = ["CITY OF MCKINNEY", "MCKINNEY CITY OF", "MCKINNEY INDEPENDENT SCHOOL", "MCKINNEY ISD",
          "PROSPER INDEPENDENT", "FRISCO INDEPENDENT SCHOOL", "LOVEJOY INDEPENDENT", "MELISSA INDEPENDENT",
          "COLLIN COUNTY OF", "COUNTY OF COLLIN", "COLLIN COLLEGE", "COLLIN COUNTY COMMUNITY COLLEGE",
          "MCKINNEY HOUSING", "MCKINNEY ECONOMIC DEVELOPMENT", "MCKINNEY COMMUNITY DEVELOPMENT", "TEXAS DEPARTMENT OF TRANSPORTATION",
          "STATE OF TEXAS", "UNITED STATES OF AMERICA", "NORTH TEXAS TOLLWAY AUTHORITY", "NORTH TEXAS MUNICIPAL WATER"]

if __name__ == "__main__":
    build_city(CITY, STATE, "4845744", [
        ("Collin", lambda b, d: ccad_rows(b, d, city_code="CMC"), ccad_link),
    ], PUBLIC)
