#!/usr/bin/env python3
"""
Build Fort Worth, TX's canonical parcel parquet: the whole city, within city limits.

Fort Worth spans Tarrant, Denton, Parker, Wise and Johnson Counties. All come from one automated
source, the City of Fort Worth's public parcel layer (Parcels_Public_Vview): Tarrant rows are
TAD's roll refreshed nightly; the Denton and Parker parts of the city are loaded from those
appraisal districts with 2026 values (asserted). ~12.7k Tarrant accounts (4% of the city,
$5.7B of market value, much of downtown) carry only a market total in the layer; their land /
improvement split, land-use code and acreage come from TAD's own 2026 PropertyData export.

City limits: Census TIGER/Line 2025 place 4827000 (Fort Worth city), i.e. the city-reported
boundary as of 2025-01-01, newer than the city's hosted copy (TIGER 2024; +6.3 km2 of
annexations). Cross-checked against the taxing-unit codes in the roll (Tarrant 26/026,
Denton C36, Parker CFW).

Notes:
- Parker County rows are labelled "Fort WorthETJ" and mostly lie outside the city; the TIGER
  clip keeps only those inside. Their CITY_TAXABLE_VALUE / APPRAISED_VALUE are not real city
  figures, so the $0-taxable exemption rule does not apply there (class X + public owners do).
- Denton County rows carry no LAND_ACRE, so their land area is the polygon area.
- Links: Tarrant parcels link to TAD's property page by account; Denton and Parker parcels link
  to the district's property search (per-parcel URL patterns not verified).

The pipeline itself is shared with every Tarrant city: see tad_city.py.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import geopandas as gpd

from tad_city import TadCity, build

TIGER_PLACE_URL = "https://www2.census.gov/geo/tiger/TIGER2025/PLACE/tl_2025_48_place.zip"
GEOID = "4827000"   # Fort Worth city


def load_boundary(data_dir: Path):
    """Census TIGER/Line 2025 place boundary for Fort Worth city, cached."""
    cache = data_dir / "fortworth-tx-boundary.parquet"
    if not cache.exists():
        z = data_dir / "tl_2025_48_place.zip"
        if not z.exists():
            urllib.request.urlretrieve(TIGER_PLACE_URL, z)
        g = gpd.read_file(f"zip://{z}")
        g[g["GEOID"] == GEOID].to_crs(4326).to_parquet(cache)
    return gpd.read_parquet(cache).union_all()


CONFIG = TadCity(
    city="fortworth", state="tx",
    load_boundary=load_boundary,
    counties=("Tarrant", "Denton", "Parker", "Wise", "Johnson"),
    city_codes={"Tarrant": {"26", "026"}, "Denton": {"C36"}, "Parker": {"CFW"}},
    public_names=["CITY OF FORT WORTH", "FORT WORTH CITY OF", "FORT WORTH HOUSING",
                  "HOUSING AUTHORITY OF FTW", "FORT WORTH TRANSPORTATION", "TRINITY METRO",
                  "TARRANT COUNTY OF"],
)

if __name__ == "__main__":
    build(CONFIG)
