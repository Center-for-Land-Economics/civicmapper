#!/usr/bin/env python3
"""
Run Albuquerque parcel ETL end-to-end using Bernalillo County assessor parcels
clipped to the official Albuquerque jurisdiction boundary.

Sources:
- Albuquerque open data portal / AGIS parcel context:
  https://data.cabq.gov/
- Bernalillo County assessor map service:
  https://assessormap.bernco.gov/server/rest/services/GIS/ASROnline_Public_Map/MapServer

Outputs:
- data/jurisidictions/data/albuquerque/albuquerque-nm-parcels.parquet
- data/jurisidictions/data/albuquerque/albuquerque-nm-parcels_YYYY_MM_DD.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Geod
from shapely.geometry import shape
from shapely.ops import unary_union

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.parcel_calculations import add_improvement_ratio_fields


ASSESSOR_BASE = "https://assessormap.bernco.gov/server/rest/services/GIS/ASROnline_Public_Map/MapServer"
JURISDICTION_LAYER = 29
PARCEL_LAYER = 22
JURISDICTION_NAME = "ALBUQUERQUE"
SOURCE_SRID = 2903
OUTPUT_DIR = Path("data/jurisidictions/data/albuquerque")
RAW_FIELDS = [
    "OBJECTID",
    "UPC",
    "OWNER",
    "OWNCODE",
    "OWNADD",
    "OWNCITY",
    "OWNSTATE",
    "SITUSADD",
    "SITUSCITY",
    "SITUSSTATE",
    "TAXDIST",
    "LEGALDESC",
    "DOCNUM",
    "ROLLTYPE",
    "VALCLASS",
    "PROPCLASS",
    "LANDVALUE",
    "AGVALUE",
    "IMPTVALUE",
    "TOTVALUE",
    "LANDTXBLE",
    "IMPTTXBLE",
    "TOTTXBLE",
    "HOHEXEMP",
    "VETEXEMP",
    "OTHEREXEMP",
    "TOTALEXEMP",
    "ACREAGE",
    "LUC",
    "LUC_MSG",
    "PID",
    "TID",
]
GOVERNMENT_OWNER_PATTERNS = [
    "CITY OF ALBUQUERQUE",
    "COUNTY OF BERNALILLO",
    "BERNALILLO COUNTY",
    "STATE OF NEW MEXICO",
    "UNIVERSITY OF NEW MEXICO",
    "ALBUQUERQUE PUBLIC SCHOOLS",
    "BOARD OF EDUCATION",
    "MIDDLE RIO GRANDE",
    "UNITED STATES",
    "US GOV",
    "HOUSING AUTHORITY",
    "PUBLIC SCHOOL",
    "CHURCH",
]
RESIDENTIAL_HINTS = [
    "RESIDENTIAL",
    "DWLG",
    "TOWNHOUSE",
    "CONDO",
    "APARTMENT",
    "MULTI-FAMILY",
    "MOBILE HOME",
    "DUPLEX",
    "TRIPLEX",
    "FOURPLEX",
]
COMMERCIAL_HINTS = [
    "COMMERCIAL",
    "OFFICE",
    "RETAIL",
    "HOTEL",
    "MOTEL",
    "RESTAURANT",
    "BANK",
    "SHOPPING",
]
INDUSTRIAL_HINTS = [
    "INDUSTRIAL",
    "WAREHOUSE",
    "MANUFACTUR",
    "DISTRIBUTION",
]
AG_OPEN_HINTS = [
    "AGRIC",
    "FARM",
    "RANCH",
    "GRAZING",
    "OPEN SPACE",
]
EXEMPT_LUC_PATTERNS = [
    "VACANT EXEMPT",
    "TAX EXEMPT PARK",
    "PUBLIC/GOVERNMENTAL",
    "RELIGIOUS",
    "SCHOOL",
    "PARKS/RECREATION",
    "EASEMENTS",
    "ROAD EASEMENT/COMMON",
    "CONDO COMMON AREA",
    "TOWNHOME COMMON AREA",
    "COMMON AREA VAC RES",
    "VAC COMMERCIAL COMMON AREA",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Albuquerque canonical parcel parquet.")
    parser.add_argument("--use-cache", action="store_true", help="Reuse cached raw parquet if present.")
    parser.add_argument("--skip-upload", action="store_true", help="Do not upload the canonical parquet.")
    return parser.parse_args()


def post_json(url: str, data: dict, *, timeout: int = 180) -> dict:
    response = requests.post(url, data=data, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS error from {url}: {payload['error']}")
    return payload


def fetch_boundary() -> tuple[gpd.GeoDataFrame, dict]:
    payload = post_json(
        f"{ASSESSOR_BASE}/{JURISDICTION_LAYER}/query",
        {
            "f": "geojson",
            "where": f"JURISDICTIONNAME='{JURISDICTION_NAME}'",
            "outFields": "JURISDICTIONNAME",
            "returnGeometry": "true",
            "outSR": 4326,
        },
    )
    boundary = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    if boundary.empty:
        raise RuntimeError("Failed to fetch Albuquerque jurisdiction boundary.")

    payload_sr = post_json(
        f"{ASSESSOR_BASE}/{JURISDICTION_LAYER}/query",
        {
            "f": "json",
            "where": f"JURISDICTIONNAME='{JURISDICTION_NAME}'",
            "outFields": "JURISDICTIONNAME",
            "returnGeometry": "true",
            "outSR": SOURCE_SRID,
        },
    )
    return boundary, payload_sr["features"][0]["geometry"]


def fetch_object_ids(boundary_geometry: dict) -> list[int]:
    payload = post_json(
        f"{ASSESSOR_BASE}/{PARCEL_LAYER}/query",
        {
            "f": "json",
            "geometry": json.dumps(boundary_geometry),
            "geometryType": "esriGeometryPolygon",
            "inSR": SOURCE_SRID,
            "spatialRel": "esriSpatialRelIntersects",
            "where": "1=1",
            "returnIdsOnly": "true",
        },
    )
    object_ids = sorted(payload.get("objectIds", []))
    if not object_ids:
        raise RuntimeError("No assessor parcel object IDs returned for Albuquerque.")
    return object_ids


def fetch_parcel_chunk(object_ids: list[int]) -> gpd.GeoDataFrame:
    payload = post_json(
        f"{ASSESSOR_BASE}/{PARCEL_LAYER}/query",
        {
            "f": "geojson",
            "objectIds": ",".join(str(v) for v in object_ids),
            "outFields": ",".join(RAW_FIELDS),
            "returnGeometry": "true",
            "outSR": 4326,
        },
    )
    gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    for field in RAW_FIELDS:
        if field not in gdf.columns:
            gdf[field] = np.nan
    return gdf


def download_raw(boundary_geometry: dict, raw_path: Path) -> gpd.GeoDataFrame:
    object_ids = fetch_object_ids(boundary_geometry)
    print(f"Found {len(object_ids):,} parcel IDs for Albuquerque")
    chunk_size = 800
    frames: list[gpd.GeoDataFrame] = []
    for idx in range(0, len(object_ids), chunk_size):
        chunk = object_ids[idx : idx + chunk_size]
        frame = fetch_parcel_chunk(chunk)
        frames.append(frame)
        fetched = min(idx + len(chunk), len(object_ids))
        print(f"  fetched {fetched:,}/{len(object_ids):,}")
        time.sleep(0.05)

    raw = pd.concat(frames, ignore_index=True)
    raw_gdf = gpd.GeoDataFrame(raw, geometry="geometry", crs="EPSG:4326")
    raw_gdf.to_parquet(raw_path, index=False)
    return raw_gdf


def geodesic_area_sqft(geom) -> float:
    geod = Geod(ellps="WGS84")
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type == "Polygon":
        lon, lat = geom.exterior.coords.xy
        area_m2, _ = geod.polygon_area_perimeter(lon, lat)
        hole_area = 0.0
        for ring in geom.interiors:
            lon_h, lat_h = ring.coords.xy
            part_area, _ = geod.polygon_area_perimeter(lon_h, lat_h)
            hole_area += abs(part_area)
        return max(abs(area_m2) - hole_area, 0.0) * 10.763910416709722
    if geom.geom_type == "MultiPolygon":
        return sum(geodesic_area_sqft(part) for part in geom.geoms)
    return np.nan


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).strip().upper().split())


def classify_original(row: pd.Series) -> str:
    luc_msg = clean_text(row.get("LUC_MSG"))
    return luc_msg.title() if luc_msg else "Other"


def classify_refined(row: pd.Series) -> str | None:
    category = str(row.get("property_land_use_category") or "")
    category_upper = category.upper()
    land_value = pd.to_numeric(row.get("current_full_land_value"), errors="coerce")
    improvement_value = pd.to_numeric(row.get("improvement_value"), errors="coerce")
    total = (0 if pd.isna(land_value) else land_value) + (0 if pd.isna(improvement_value) else improvement_value)

    if "VACANT" in category_upper:
        return "Vacant"
    if "PARKING" in category_upper and "PARKS/" not in category_upper and "MOBILE HOME PARK" not in category_upper:
        return "Parking Lot"
    if total > 0 and (0 if pd.isna(improvement_value) else improvement_value) <= 0:
        return "Vacant"
    if total > 0 and (0 if pd.isna(improvement_value) else improvement_value) < 0.5 * total:
        return "Underdeveloped"
    return None


def collapse_duplicates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    dupes = gdf.duplicated(subset=["UPC"], keep=False).sum()
    print(f"Duplicate rows by UPC: {dupes:,}")
    if dupes == 0:
        return gdf

    numeric_cols = [
        col
        for col in [
            "LANDVALUE",
            "AGVALUE",
            "IMPTVALUE",
            "TOTVALUE",
            "LANDTXBLE",
            "IMPTTXBLE",
            "TOTTXBLE",
            "HOHEXEMP",
            "VETEXEMP",
            "OTHEREXEMP",
            "TOTALEXEMP",
            "ACREAGE",
        ]
        if col in gdf.columns
    ]
    categorical_cols = [c for c in gdf.columns if c not in set(numeric_cols + ["UPC", "geometry"])]
    agg = {c: "sum" for c in numeric_cols}
    agg.update({c: "first" for c in categorical_cols})
    collapsed = gdf.groupby("UPC", dropna=False).agg(agg).reset_index()
    geometries = gdf.groupby("UPC", dropna=False)["geometry"].apply(
        lambda parts: unary_union([geom for geom in parts if geom is not None and not geom.is_empty])
    )
    collapsed["geometry"] = geometries.values
    return gpd.GeoDataFrame(collapsed, geometry="geometry", crs=gdf.crs)


def build_export(raw_gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    boundary_geom = boundary.geometry.union_all()

    gdf = raw_gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0))
    gdf = gdf[gdf["geometry"].notna()].copy()
    gdf = gdf[gdf.geometry.intersects(boundary_geom)].copy()
    print(f"After jurisdiction clip: {len(gdf):,}")

    gdf = collapse_duplicates(gdf)
    print(f"After duplicate collapse: {len(gdf):,}")

    for col in ["LANDVALUE", "IMPTVALUE", "TOTVALUE", "TOTALEXEMP"]:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
    gdf["total_value_calc"] = gdf["LANDVALUE"].fillna(0) + gdf["IMPTVALUE"].fillna(0)

    owner_text = (
        gdf[["OWNER", "OWNCODE", "OWNCITY"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
        .str.strip()
    )
    government_regex = "|".join(GOVERNMENT_OWNER_PATTERNS)
    exempt_by_owner = owner_text.str.contains(government_regex, regex=True, na=False)
    exempt_by_value = (gdf["total_value_calc"] > 0) & (
        gdf["TOTALEXEMP"].fillna(0) >= 0.995 * gdf["total_value_calc"]
    )
    exempt_by_class = (
        gdf["PROPCLASS"].fillna("").astype(str).str.upper().str.contains("EXEMPT|GOV", regex=True)
        | gdf["VALCLASS"].fillna("").astype(str).str.upper().str.contains("EXEMPT|GOV", regex=True)
        | gdf["LUC_MSG"].fillna("").astype(str).str.upper().str.contains("EXEMPT", regex=True)
    )
    exempt_by_luc = gdf["LUC_MSG"].fillna("").astype(str).str.upper().isin(EXEMPT_LUC_PATTERNS)
    gdf["exemption_flag"] = (exempt_by_owner | exempt_by_value | exempt_by_class | exempt_by_luc).astype(int)
    before = len(gdf)
    gdf = gdf[gdf["exemption_flag"] == 0].copy()
    print(f"Removed {before - len(gdf):,} exempt parcels -> {len(gdf):,}")

    gdf["property_land_use_category"] = gdf.apply(classify_original, axis=1)
    gdf["current_full_land_value"] = pd.to_numeric(gdf["LANDVALUE"], errors="coerce")
    gdf["improvement_value"] = pd.to_numeric(gdf["IMPTVALUE"], errors="coerce")
    gdf["full_market_value"] = gdf["current_full_land_value"].fillna(0) + gdf["improvement_value"].fillna(0)
    gdf["REALLANDVA"] = gdf["current_full_land_value"]
    gdf["REALIMPROV"] = gdf["improvement_value"]
    gdf["property_land_use_refined"] = gdf.apply(classify_refined, axis=1)
    gdf["area_sqft"] = gdf["geometry"].apply(geodesic_area_sqft)
    gdf.loc[gdf["area_sqft"] <= 0, "area_sqft"] = np.nan
    gdf["land_value_per_sqft"] = gdf["current_full_land_value"] / gdf["area_sqft"]
    gdf["improvement_value_per_sqft"] = gdf["improvement_value"] / gdf["area_sqft"]
    gdf["full_market_value_per_sqft"] = gdf["full_market_value"] / gdf["area_sqft"]
    gdf["REALLANDVA_per_sqft"] = gdf["land_value_per_sqft"]
    gdf["REALIMPROV_per_sqft"] = gdf["improvement_value_per_sqft"]
    gdf["link"] = np.nan

    gdf = add_improvement_ratio_fields(
        gdf,
        land_col="current_full_land_value",
        improvement_col="improvement_value",
    )

    export_columns = [
        "geometry",
        "UPC",
        "SITUSADD",
        "SITUSCITY",
        "TAXDIST",
        "LUC",
        "LUC_MSG",
        "PROPERTY_CLASS",
        "exemption_flag",
        "property_land_use_category",
        "property_land_use_refined",
        "REALLANDVA",
        "current_full_land_value",
        "REALLANDVA_per_sqft",
        "land_value_per_sqft",
        "REALIMPROV",
        "improvement_value",
        "REALIMPROV_per_sqft",
        "improvement_value_per_sqft",
        "full_market_value",
        "full_market_value_per_sqft",
        "TLLDIMPROV",
        "IMPR_LAND_RATIO",
        "IMPR_LAND_PCT",
        "IMPR_PCT_TOTAL",
        "area_sqft",
        "link",
    ]

    gdf["PROPERTY_CLASS"] = gdf["PROPCLASS"]
    export = gdf[export_columns].copy()
    export = gpd.GeoDataFrame(export, geometry="geometry", crs="EPSG:4326")
    export["geometry"] = export["geometry"].apply(lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0))
    return export


def maybe_upload(local_path: Path) -> None:
    from azure.storage.blob import BlobServiceClient

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        print("AZURE_STORAGE_CONNECTION_STRING not set; skipping upload")
        return

    container = os.getenv("AZURE_DEV_CONTAINER", "parquets-dev")
    blob_name = local_path.name
    client = BlobServiceClient.from_connection_string(conn_str).get_container_client(container)
    with open(local_path, "rb") as fh:
        client.upload_blob(name=blob_name, data=fh, overwrite=True)
    print(f"Uploaded {blob_name} -> {container}/{blob_name}")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime("%Y_%m_%d")
    raw_path = OUTPUT_DIR / f"albuquerque_raw_{today_str}.parquet"
    canonical_path = OUTPUT_DIR / "albuquerque-nm-parcels.parquet"
    dated_path = OUTPUT_DIR / f"albuquerque-nm-parcels_{today_str}.parquet"

    boundary, boundary_sr = fetch_boundary()
    print("Fetched official Albuquerque jurisdiction boundary")

    if args.use_cache and raw_path.exists():
        print(f"Loading cached raw parquet: {raw_path}")
        raw_gdf = gpd.read_parquet(raw_path)
    else:
        raw_gdf = download_raw(boundary_sr, raw_path)

    print(f"Loaded {len(raw_gdf):,} raw assessor parcel rows")
    export = build_export(raw_gdf, boundary)

    export.to_parquet(canonical_path, index=False)
    export.to_parquet(dated_path, index=False)
    print(f"Saved canonical parquet: {canonical_path}")
    print(f"Saved dated parquet: {dated_path}")
    print("\nOriginal category counts:")
    print(export["property_land_use_category"].value_counts(dropna=False).head(25).to_string())
    print("\nRefined category counts:")
    print(export["property_land_use_refined"].value_counts(dropna=False).to_string())

    if not args.skip_upload:
        maybe_upload(canonical_path)


if __name__ == "__main__":
    main()
