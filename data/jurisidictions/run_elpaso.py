#!/usr/bin/env python3
"""
Build El Paso, TX's canonical parcel parquet: the whole city, within city limits.

Sources:
- Values: El Paso Central Appraisal District's 2026 certified export ("As of Supplement",
  exported 2026-10-13; PACS layout 8.0.34, positions identical to 8.0.33 for every field used).
  epcad.org sits behind a browser challenge, so the zip is downloaded by hand from
  https://epcad.org/OpenGovernment and placed at
    data/jurisidictions/data/elpaso/elpaso-tx-certified-export.zip
- Geometry: the district's own web-map service (ElPasoCADWebService, hosted by BIS Consulting),
  joined on prop_id (217,960 of 218,260 polygons in and around the city match). The district's
  GIS shapefile download carries no values, so it is not needed.
City limits: Census TIGER/Line 2025 place 4824000, cross-checked with the City of El Paso taxing
entity (CEP).

Local state codes are read through the district's state-code table: C2 = COLONIA lots
(residential), C10 = commercial vacant lot, X21-X23 = ranch improvements on ag land (statewide
D2, not exempt), XV-C / XV-R = totally exempt.

Pipeline shared with Brownsville: pacs_city.py + tx_pipeline.py.
Link: El Paso CAD's True Prodigy portal, property-detail/<prop_id> (same platform and pattern
as Travis CAD's; not verified from the build host).
"""
from pacs_city import PacsCity, build

CONFIG = PacsCity(
    city="elpaso", state="tx", county="El Paso",
    gis_url=None,
    gis_layer=("https://services2.arcgis.com/fKvlzLJczghwPYHS/arcgis/rest/services/"
               "ElPasoCADWebService/FeatureServer/0"),
    export_url="https://epcad.org/OpenGovernment",   # manual download (browser challenge)
    tiger_geoid="4824000",
    city_entity="CEP",
    link=lambda prop_id: f"https://elpaso.prodigycad.com/property-detail/{prop_id}",
    gis_values={"land": ["land_val"], "impr": ["imprv_val"], "market": "market"},
    public_names=["CITY OF EL PASO", "EL PASO CITY OF", "EL PASO COUNTY OF", "COUNTY OF EL PASO",
                  "EL PASO WATER", "PUBLIC SERVICE BOARD", "EL PASO INDEPENDENT SCHOOL",
                  "YSLETA INDEPENDENT SCHOOL", "SOCORRO INDEPENDENT SCHOOL", "EL PASO ISD",
                  "UNIVERSITY OF TEXAS AT EL PASO", "EL PASO COMMUNITY COLLEGE",
                  "HOUSING AUTHORITY OF THE CITY OF EL PASO", "UNIVERSITY MEDICAL CENTER",
                  "EL PASO COUNTY HOSPITAL DISTRICT", "FORT BLISS", "UNITED STATES OF AMERICA"],
)

if __name__ == "__main__":
    build(CONFIG)
