#!/usr/bin/env python3
"""
Build Dallas, TX's canonical parcel parquet: the whole city, within city limits, on 2026 CERTIFIED
values (replaces the 2026-06 build on DCAD's proposed-value files; see git history).

Sources, one per county, combined in one tx_pipeline run (shared runner tx_multicounty.py):
- Dallas County (895 km2): Dallas Central Appraisal District's 2026 certified data files
  (DCAD2026_CURRENT: certified values + supplements, files of 2026-11-09) joined on
  GIS_PARCEL_ID to DCAD's parcel polygons (PARCEL_GEOM, 2026-11); see dcad_county.py. Shared
  downloads in data/jurisidictions/data/dcad/. City of Dallas: CITY_JURIS_DESC == DALLAS.
- Collin County (31 km2): Collin CAD's public parcel layer, 2026 certified values from its
  prior-year fields; see collin_county.py. City code CDA.
- Denton County (6 km2): Denton CAD's 2026 certified export + certified parcel polygons; see
  denton_county.py. Entity C20.
Not included: the Rockwall County part (53 km2, almost all Lake Ray Hubbard) and the Kaufman
County part (8 km2); no Rockwall / Kaufman CAD data on hand.
City limits: Census TIGER/Line 2025 place 4819000.
Links: DCAD's account page by division; Collin CAD eSearch; Denton CAD property page.
"""
from collin_county import ccad_link, ccad_rows
from dcad_county import dcad_link, dcad_rows
from denton_county import denton_link, denton_rows
from tx_multicounty import build_city

CITY, STATE = "dallas", "tx"
PUBLIC = ["CITY OF DALLAS", "DALLAS CITY OF", "DALLAS COUNTY OF", "COUNTY OF DALLAS",
          "DALLAS INDEPENDENT SCHOOL", "DALLAS ISD", "RICHARDSON INDEPENDENT", "PLANO INDEPENDENT",
          "CARROLLTON FARMERS BRANCH", "DUNCANVILLE INDEPENDENT", "WILMER HUTCHINS", "MESQUITE INDEPENDENT",
          "GARLAND INDEPENDENT", "DALLAS COLLEGE", "DALLAS CO COMMUNITY COLLEGE", "DALLAS COUNTY COMMUNITY COLLEGE",
          "PARKLAND HEALTH", "PARKLAND MEMORIAL", "PARKLAND HOSPITAL", "DALLAS COUNTY HOSPITAL", "DALLAS AREA RAPID TRANSIT", "DALLAS HOUSING AUTHORITY",
          "HOUSING AUTHORITY OF THE CITY OF DALLAS", "UNIVERSITY OF TEXAS", "UNIV OF TEXAS", "UNIVERSITY OF NORTH TEXAS",
          "DALLAS FORT WORTH INTERNATIONAL AIRPORT", "DFW INTERNATIONAL AIRPORT", "NORTH TEXAS TOLLWAY AUTHORITY",
          "TRINITY RIVER AUTHORITY", "TEXAS DEPARTMENT OF TRANSPORTATION", "STATE OF TEXAS",
          "UNITED STATES OF AMERICA", "COLLIN COUNTY OF", "DENTON COUNTY OF"]

if __name__ == "__main__":
    build_city(CITY, STATE, "4819000", [
        ("Dallas", lambda b, d: dcad_rows(b, d, city_juris="DALLAS"), dcad_link),
        ("Collin", lambda b, d: ccad_rows(b, d, city_code="CDA"), ccad_link),
        ("Denton", lambda b, d: denton_rows(b, d, city=CITY, state=STATE, city_entity="C20"), denton_link),
    ], PUBLIC)
