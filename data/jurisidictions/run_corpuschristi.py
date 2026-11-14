#!/usr/bin/env python3
"""
Build Corpus Christi, TX's canonical parcel parquet: the whole city, within city limits.

Sources:
- Values: Nueces Appraisal District's 2026 certified appraisal export (PACS fixed-width).
  nuecescad.net is not reachable from the build host, so the zip is downloaded by hand and
  placed at
    data/jurisidictions/data/corpuschristi/corpuschristi-tx-certified-export.zip
- Geometry: the district's own web-map service (NuecesCADWebService, BIS-hosted), joined on
  prop_id. Its values are the 2027 work-in-progress roll (owner_tax_yr 2027), so they are used
  only to fill placeholder records, never as the values.
City limits: Census TIGER/Line 2025 place 4817000, cross-checked with the roll's City of
Corpus Christi taxing entity (C03).

Corpus Christi also reaches into San Patricio County (47 parcels, ~$75M land, mostly
north-shore industrial land and Port of Corpus Christi tracts), Kleberg County (4 tracts,
3 public) and Aransas County (water only). As of 2026-11 these are not included; Nueces
holds ~120k of the city's parcels.

Pipeline shared with Brownsville and El Paso: pacs_city.py + tx_pipeline.py.
"""
from pacs_city import PacsCity, build

CONFIG = PacsCity(
    city="corpuschristi", state="tx", county="Nueces",
    gis_url=None,
    gis_layer=("https://services6.arcgis.com/j94FvPaik4etwHFk/arcgis/rest/services/"
               "NuecesCADWebService/FeatureServer/0"),
    export_url="https://nuecescad.net/",   # manual download (not reachable from the build host)
    tiger_geoid="4817000",
    city_entity="C03",
    # Placeholder until a working per-parcel URL is confirmed.
    link=lambda prop_id: f"https://nuecescad.net/?prop_id={prop_id}",
    gis_values={"land": ["land_val"], "impr": ["imprv_val"], "market": "market"},
    public_names=["CITY OF CORPUS CHRISTI", "CORPUS CHRISTI CITY OF", "NUECES COUNTY",
                  "COUNTY OF NUECES", "PORT OF CORPUS CHRISTI", "PORT OF CORPUS CHRISTI AUTHORITY",
                  "CORPUS CHRISTI INDEPENDENT SCHOOL", "CORPUS CHRISTI ISD",
                  "FLOUR BLUFF INDEPENDENT SCHOOL", "TULOSO MIDWAY INDEPENDENT SCHOOL",
                  "CALALLEN INDEPENDENT SCHOOL", "WEST OAKS INDEPENDENT SCHOOL",
                  "LONDON INDEPENDENT SCHOOL", "DEL MAR COLLEGE", "TEXAS A&M UNIVERSITY",
                  "TEXAS A & M UNIVERSITY", "CORPUS CHRISTI HOUSING AUTHORITY",
                  "HOUSING AUTHORITY OF THE CITY OF CORPUS CHRISTI", "NUECES COUNTY HOSPITAL",
                  "STATE OF TEXAS", "UNITED STATES OF AMERICA", "UNITED STATES GOVERNMENT"],
)

if __name__ == "__main__":
    build(CONFIG)
