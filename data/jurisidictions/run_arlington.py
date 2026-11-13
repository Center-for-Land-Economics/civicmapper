#!/usr/bin/env python3
"""
Build Arlington, TX's canonical parcel parquet from the Tarrant Appraisal District (TAD) roll.

Fully automated, one source (no manual download):
- Parcels (geometry + LAND_VAL / IMPR_VAL / MARKET_VALUE + SPTB state class + LAND_ACRE +
  TAX_YEAR): the City of Fort Worth's public copy of the TAD parcel layer, refreshed nightly
  from TAD for every Tarrant County parcel (TAX_YEAR 2026 as of 2026-11).
    https://services5.arcgis.com/3ddLCBXe1bRt7mzj/arcgis/rest/services/Parcels_Public_Vview/FeatureServer/0
  Pulled with the city boundary's envelope as a spatial pre-filter and cached on first run.
  Rows missing a land/improvement split are filled from TAD's own 2026 PropertyData export
  (see tad_city.py). Arlington is entirely in Tarrant County.
- City limits: City of Arlington "Municipal Boundary" FeatureServer, MUNICIPALNAME='ARLINGTON'
  (centroid-within clip). Cross-checked against TAD's own taxing-unit code (CITY='24').
    https://services.arcgis.com/jXi5GuMZwfCYtZP9/arcgis/rest/services/Municipal_Boundary/FeatureServer/0

Outputs:
- data/jurisidictions/data/arlington/arlington-tx-parcels.parquet (+ dated snapshot)

Notes:
- TAD's EXEMPTION_CODE is 'YES' for ANY exemption, including the ordinary homestead
  exemption (61% of Arlington parcels), so it is NOT a full-exemption flag, and TAD almost never
  uses state class X. Totally exempt = class X, OR a named public owner, OR $0 City of Arlington
  taxable value on an appraised non-residential parcel (city, state, ISDs, UTA, churches,
  charities, cemeteries). Homes with $0 taxable (100% disabled-veteran homesteads) stay in
  unless the owner is institutional.
- LAND_SQFT is empty in this layer. The $/sqft denominator is the appraiser's LAND_ACRE when it
  is within 25% of the geodesic polygon area, else the polygon area: on ~6.7k parcels LAND_ACRE
  holds square feet, tiny fractions or a partial area. Emits land_area_acres, area_source,
  likely_remnant.
- Stacked records: TAD maps every condo unit onto the whole complex polygon (each owning
  "% OF COMMON AREA"; The Cloisters is 491 units on one 27.6-acre polygon), carries one record
  per undivided-interest owner with that owner's SHARE (5120 Yucca Ct: $41,250 + $13,750 land),
  and splits lots into accounts at school/TIF boundaries. All are partial values of one
  footprint, so they are SUMMED onto it and its denominator is the footprint area (a condo
  unit's LAND_ACRE is its share or 0). The surviving row keeps the highest-land-value record's
  class. One account split across several polygons instead keeps its values once (asserted
  identical across pieces) with geometry unioned (skill §2/§2a).
- Condo units mapped as tiny per-unit polygons (no stacked record, no common-area parcel) are
  merged per development onto a nominal-value common-area parcel when one contains them
  (Pebblebrook Village), else onto the convex hull of the units minus other parcels' land
  (skill §6b fallback). Fee-simple townhomes keep their own small lots.
- ~3,100 polygons inside the city have no roll record (no value/class/year) and are dropped.
- Values are TAD's 2026 appraisal (LAND_VAL + IMPR_VAL = MARKET_VALUE).

The pipeline itself is shared with every Tarrant city: see tad_city.py.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import requests

from tad_city import TadCity, build

BOUNDARY_QUERY = ("https://services.arcgis.com/jXi5GuMZwfCYtZP9/arcgis/rest/services/"
                  "Municipal_Boundary/FeatureServer/0/query")


def load_boundary(data_dir: Path):
    """City of Arlington Municipal Boundary (MUNICIPALNAME='ARLINGTON'), cached."""
    cache = data_dir / "arlington-tx-boundary.parquet"
    if cache.exists():
        b = gpd.read_parquet(cache)
    else:
        r = requests.get(BOUNDARY_QUERY, params={"where": "MUNICIPALNAME='ARLINGTON'",
                         "outFields": "*", "outSR": 4326, "f": "geojson"}, timeout=120)
        r.raise_for_status()
        b = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        b.to_parquet(cache)
    return b.to_crs(4326).union_all()


CONFIG = TadCity(
    city="arlington", state="tx",
    load_boundary=load_boundary,
    city_codes={"Tarrant": {"24"}},          # TAD taxing-unit code for the City of Arlington
    public_names=["CITY OF ARLINGTON", "ARLINGTON CITY OF", "ARLINGTON HOUSING AUTH"],
)

if __name__ == "__main__":
    build(CONFIG)
