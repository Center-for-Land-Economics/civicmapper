#!/usr/bin/env python3
"""
Build Brownsville, TX's canonical parcel parquet: the whole city, within city limits.

Brownsville lies entirely in Cameron County. Both sources are the Cameron Appraisal District's
own downloads (www.cameroncad.org/Pages/Data), fetched automatically:
- Values: the CERTIFIED 2026 appraisal export (PACS Legacy 8.0.33, exported 2026-07-23). The
  district's page states appraisal year 2026 is certified and 2027 is a work in progress, so the
  City of Brownsville's GIS copy (pyear 2027, Sept 2026) is NOT used for values.
- Geometry: the district's GIS parcels (parcels_public.zip), joined on PROP_ID.
City limits: Census TIGER/Line 2025 place 4810768 (Brownsville city), cross-checked with the
roll's City of Brownsville taxing entity (CBR).

The pipeline is shared: pacs_city.py (roll specifics) + tx_pipeline.py (everything after).
Link: Cameron CAD's True Prodigy portal, property-detail/<prop_id>/<year>, pinned to 2026 (the
certified year these values come from). Pattern confirmed working by the user (prop 79665).
"""
from pacs_city import PacsCity, build

CONFIG = PacsCity(
    city="brownsville", state="tx", county="Cameron",
    gis_url="https://www.cameroncad.org/cad/exports/GIS/parcels_public.zip",
    export_url=("https://www.cameroncad.org/cad/exports/certified/2026/"
                "cameron-2026-certified-appraisal-export-20260723.zip"),
    tiger_geoid="4810768",
    city_entity="CBR",
    link=lambda prop_id: f"https://cameron.prodigycad.com/property-detail/{prop_id}/2026",
    gis_values={"land": ["lndHS", "lndNHS", "agMkt"], "impr": ["impHS", "impNHS"], "market": "market"},
    public_names=["CITY OF BROWNSVILLE", "BROWNSVILLE CITY OF", "BROWNSVILLE PUBLIC UTILITY",
                  "BROWNSVILLE PUB UTIL", "BROWNSVILLE NAVIGATION DISTRICT", "PORT OF BROWNSVILLE",
                  "CAMERON COUNTY OF", "COUNTY OF CAMERON", "BROWNSVILLE INDEPENDENT SCHOOL", "BROWNSVILLE ISD",
                  "TEXAS SOUTHMOST COLLEGE", "UNIVERSITY OF TEXAS RIO GRANDE",
                  "BROWNSVILLE HOUSING AUTHORITY", "BROWNSVILLE COMMUNITY IMPROVEMENT CORP"],
)

if __name__ == "__main__":
    build(CONFIG)
