#!/usr/bin/env python3
"""
Build Lubbock, TX's canonical parcel parquet: the whole city, within city limits (374 km2, all in
Lubbock County). PROVISIONAL: Lubbock Appraisal District's own export and GIS are not reachable,
so this uses TxGIO StratMap 2026 land parcels for Lubbock County (stratmap26-landparcels_48303,
collected from Lubbock CAD 2026-07-01; manual download into data/jurisidictions/data/lubbock-stratmap/).
StratMap's Lubbock edition has land and market value but no state class, exemption flag or city
code: improvement = market - land, the property class comes from the City of Lubbock's zoning
(NewZoning_UDC), and exemptions from owner names. See stratmap_county.py.
City limits: Census TIGER/Line 2025 place 4845000. Shared runner: tx_multicounty.py.
"""
from pathlib import Path

import geopandas as gpd

from stratmap_county import stratmap_rows
from tx_common import fetch_arcgis_geojson
from tx_multicounty import build_city

CITY, STATE = "lubbock", "tx"
SRC_DIR = Path(__file__).resolve().parent / "data" / "lubbock-stratmap"
ZONING = "https://services2.arcgis.com/eYXun6c1pgy8Qpta/arcgis/rest/services/NewZoning_UDC/FeatureServer/0"


def lubbock_link(prop_id: str) -> str:
    # Lubbock CAD's property page by quick-ref id (R...); pattern confirmed by the user (R64540).
    return f"https://lubbockcad.org/Property-Detail/PropertyQuickRefID/{prop_id}"


def zoning_class(district: str) -> str:
    d = (district or "").lower()
    if "single family" in d or "medium density" in d:
        return "A1"
    if "high density" in d:
        return "B1"
    if "industrial" in d:
        return "F2"
    if any(k in d for k in ("commercial", "office", "broadway", "arts", "civic", "depot", "general")):
        return "F1"
    return ""


def rows(boundary, data_dir):
    z = fetch_arcgis_geojson(ZONING, cache=SRC_DIR / "lubbock-zoning-udc.parquet", id_field="OBJECTID_1")
    z = z.rename(columns={"ZoningDist": "zone"})
    z["zone"] = z["zone"].fillna("").astype(str)
    return stratmap_rows(boundary, data_dir, zip_path=SRC_DIR / "stratmap26-landparcels_48303_lp.zip",
                         layer_path="fgdb/stratmap-2026-land-parcels-lubbock_48303.gdb", county="Lubbock",
                         zoning=gpd.GeoDataFrame(z, geometry="geometry", crs=4326), zoning_class=zoning_class,
                         cache_name="lubbock-tx-stratmap-geometry.parquet")


PUBLIC = ["CITY OF LUBBOCK", "LUBBOCK CITY OF", "LUBBOCK COUNTY", "COUNTY OF LUBBOCK",
          "LUBBOCK INDEPENDENT SCHOOL", "LUBBOCK ISD", "FRENSHIP INDEPENDENT", "FRENSHIP ISD",
          "LUBBOCK-COOPER", "LUBBOCK COOPER", "ROOSEVELT INDEPENDENT", "IDALOU INDEPENDENT",
          "TEXAS TECH UNIVERSITY", "TEXAS TECH UNIV", "BOARD OF REGENTS", "SOUTH PLAINS COLLEGE",
          "LUBBOCK CHRISTIAN UNIVERSITY", "UNIVERSITY MEDICAL CENTER", "LUBBOCK COUNTY HOSPITAL",
          "LUBBOCK HOUSING AUTHORITY", "URBAN RENEWAL", "HOUSING AUTHORITY OF THE CITY OF LUBBOCK", "LUBBOCK RESERVOIR",
          "STATE OF TEXAS", "TEXAS DEPARTMENT OF TRANSPORTATION", "UNITED STATES OF AMERICA",
          "CHURCH", "BAPTIST", "METHODIST", "CATHOLIC DIOCESE", "DIOCESE OF LUBBOCK", "LUTHERAN",
          "PRESBYTERIAN", "EPISCOPAL", "CONGREGATION", "ASSEMBLY OF GOD", "MINISTRIES", "CEMETERY"]

if __name__ == "__main__":
    # No appraiser acreage in StratMap's Lubbock edition: hide vacant lot-value fragments up to
    # 2,500 sqft (WATTS MCGEE L 1: $70,000 on a 1,228 sqft polygon).
    build_city(CITY, STATE, "4845000", [("Lubbock", rows, lubbock_link)], PUBLIC, strip_max_sqft=2500)
