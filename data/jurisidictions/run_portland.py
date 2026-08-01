#!/usr/bin/env python3
"""
Build Portland's canonical parcel parquet from official Multnomah County services.

Sources:
- Taxlots with valuation fields:
  https://www3.multco.us/gisagspublic/rest/services/DART/Taxlots_Orion_Public/MapServer/0
- City boundaries:
  https://www3.multco.us/gisagspublic/rest/services/DART/LevyCode/MapServer/4

Outputs:
- data/jurisidictions/data/portland/portland-or-parcels.parquet
- data/jurisidictions/data/portland/portland-or-parcels_YYYY_MM_DD.parquet
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Geod
from shapely.ops import unary_union

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.parcel_calculations import add_improvement_ratio_fields


TAXLOTS_QUERY_URL = "https://www3.multco.us/gisagspublic/rest/services/DART/Taxlots_Orion_Public/MapServer/0/query"
BOUNDARY_QUERY_URL = "https://www3.multco.us/gisagspublic/rest/services/DART/LevyCode/MapServer/4/query"
CITY_NAME = "CITY OF PORTLAND"
PAGE_SIZE = 2000
OUTPUT_DIR = Path("data/jurisidictions/data/portland")
RAW_FIELDS = [
    "OBJECTID_1",
    "MAPTAXLOT",
    "PROPID",
    "ALTACCTNUM",
    "NAME",
    "NAME2",
    "ADDR1",
    "ADDR2",
    "CITY",
    "STATE",
    "ZIP",
    "SITUSADDR",
    "SITUSCITY",
    "SITUSSTATE",
    "SITUSZIP",
    "MAPID",
    "LEGAL",
    "TRACTLOT",
    "LOC_CODE",
    "ACCOUNT_STATUS",
    "LEVYCODE",
    "PROPCLASS",
    "PROP_CODE",
    "SALE_PRICE",
    "EXEMPTION",
    "ZONING",
    "SIZEACRES",
    "SIZESQFT",
    "IMPTYPE",
    "ACTYEARBUILT",
    "MAINAREA",
    "UNITS",
    "MAIN_SQFT",
    "ROLLYEAR",
    "ROLLLAND",
    "ROLLIMP",
    "ROLLM50",
    "PROPERTY_TYPE",
    "APPRAISER",
    "ROLLMAV",
]
EXCLUDED_PROPERTY_TYPES = {
    "BILLBOARD",
    "INDUSTRIAL M&E",
    "INDUSTRIAL LEASEHOLD IMPROVEMENTS",
    "INDUSTRIAL LOCAL",
    "INDUSTRIAL STATE",
}
GOVERNMENT_OWNER_PATTERNS = [
    "CITY OF PORTLAND",
    "PORT OF PORTLAND",
    "MULTNOMAH COUNTY",
    "STATE OF OREGON",
    "OREGON STATE OF",
    "TRI-COUNTY METRO TRANS DIST",
    "TRIMET",
    "METRO REGIONAL GOVERNMENT",
    "UNITED STATES",
    "US GOV",
    "US POSTAL SERVICE",
    "PORTLAND PUBLIC SCHOOLS",
    "SCHOOL DISTRICT",
    "OREGON HEALTH & SCIENCE UNIVERSITY",
    "HOUSING AUTHORITY",
    "UNIVERSITY",
    "PARKS AND RECREATION",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Portland canonical parcel parquet.")
    parser.add_argument("--use-cache", action="store_true", help="Reuse the latest cached raw parquet if present.")
    parser.add_argument("--skip-upload", action="store_true", help="Do not upload the canonical parquet.")
    return parser.parse_args()


def request_json(url: str, params: dict, *, timeout: int = 240) -> dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS error from {url}: {payload['error']}")
    return payload


def fetch_count(url: str, where: str = "1=1") -> int:
    payload = request_json(
        url,
        {
            "f": "json",
            "where": where,
            "returnCountOnly": "true",
        },
        timeout=120,
    )
    return int(payload["count"])


def fetch_page(url: str, fields: list[str], offset: int, *, where: str = "1=1") -> gpd.GeoDataFrame:
    payload = request_json(
        url,
        {
            "f": "geojson",
            "where": where,
            "outFields": ",".join(fields),
            "returnGeometry": "true",
            "outSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "orderByFields": fields[0],
        },
    )
    gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    for field in fields:
        if field not in gdf.columns:
            gdf[field] = np.nan
    return gdf


def latest_cached_raw(raw_glob: str) -> Path | None:
    files = sorted(Path().glob(raw_glob))
    return files[-1] if files else None


def download_layer(url: str, fields: list[str], raw_path: Path, *, where: str = "1=1", label: str) -> gpd.GeoDataFrame:
    total = fetch_count(url, where=where)
    print(f"Found {total:,} {label} records")
    frames: list[gpd.GeoDataFrame] = []
    offset = 0
    while offset < total:
        frame = fetch_page(url, fields, offset, where=where)
        if frame.empty:
            break
        frames.append(frame)
        offset += len(frame)
        print(f"  fetched {offset:,}/{total:,}")
        time.sleep(0.03)

    raw = pd.concat(frames, ignore_index=True)
    raw_gdf = gpd.GeoDataFrame(raw, geometry="geometry", crs="EPSG:4326")
    raw_gdf.to_parquet(raw_path, index=False)
    print(f"Saved raw cache: {raw_path}")
    return raw_gdf


def fetch_portland_boundary() -> gpd.GeoDataFrame:
    payload = request_json(
        BOUNDARY_QUERY_URL,
        {
            "f": "geojson",
            "where": f"Tax_Dist_Desc='{CITY_NAME}'",
            "outFields": "Tax_Dist_Desc",
            "returnGeometry": "true",
            "outSR": 4326,
        },
    )
    boundary = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    if boundary.empty:
        raise RuntimeError("Failed to fetch Portland municipal boundary.")
    return boundary


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


def normalize_parcel_id(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return re.sub(r"-A\d+$", "", text)


def collapse_duplicate_parcels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    duplicate_count = gdf.duplicated(subset=["parcel_id"], keep=False).sum()
    print(f"Duplicate parcel_id rows: {duplicate_count:,}")
    if duplicate_count == 0:
        return gdf

    numeric_cols = [
        col
        for col in [
            "ROLLLAND",
            "ROLLIMP",
            "ROLLM50",
            "ROLLMAV",
            "SALE_PRICE",
            "SIZEACRES",
            "SIZESQFT",
            "UNITS",
            "MAINAREA",
            "MAIN_SQFT",
        ]
        if col in gdf.columns
    ]
    categorical_cols = [c for c in gdf.columns if c not in set(numeric_cols + ["parcel_id", "geometry"])]
    agg = {c: "sum" for c in numeric_cols}
    agg.update({c: "first" for c in categorical_cols})
    collapsed = gdf.groupby("parcel_id", dropna=False).agg(agg).reset_index()
    geometries = gdf.groupby("parcel_id", dropna=False)["geometry"].apply(
        lambda parts: unary_union([geom for geom in parts if geom is not None and not geom.is_empty])
    )
    collapsed["geometry"] = geometries.values
    result = gpd.GeoDataFrame(collapsed, geometry="geometry", crs=gdf.crs)
    print(f"Rows after duplicate collapse: {len(result):,}")
    return result


def classify_original(row: pd.Series) -> str:
    property_type = clean_text(row.get("PROPERTY_TYPE"))
    prop_code = clean_text(row.get("PROP_CODE"))
    if property_type:
        if prop_code:
            return f"{property_type.title()} ({prop_code})"
        return property_type.title()
    if prop_code:
        return f"Property Code {prop_code}"
    return "Other"


def classify_refined(row: pd.Series) -> str | None:
    property_type = clean_text(row.get("PROPERTY_TYPE"))
    prop_code = clean_text(row.get("PROP_CODE"))
    propclass = clean_text(row.get("PROPCLASS"))
    category = clean_text(row.get("property_land_use_category"))
    text_blob = " ".join(
        [
            clean_text(row.get("NAME")),
            clean_text(row.get("LEGAL")),
            clean_text(row.get("SITUSADDR")),
            category,
        ]
    )

    land = float(pd.to_numeric(row.get("current_full_land_value"), errors="coerce") or 0.0)
    improvement = float(pd.to_numeric(row.get("improvement_value"), errors="coerce") or 0.0)
    total = land + improvement
    residential_like = (
        "RESIDENTIAL" in property_type
        or prop_code in {"W", "B"}
        or propclass.startswith("1")
    )
    commercial_like = any(
        token in property_type for token in ["COMMERCIAL", "INDUSTRIAL"]
    ) or any(token in category for token in ["COMMERCIAL", "INDUSTRIAL"]) or prop_code in {"CC", "EA", "IG", "IL", "IS"}

    if ("PARKING" in text_blob or "PARKADE" in text_blob) and not residential_like:
        return "Parking Lot"
    if "GARAGE" in text_blob and commercial_like and not residential_like:
        return "Parking Lot"
    if improvement <= 0 and land > 0:
        return "Vacant"
    if total <= 0 or residential_like:
        return None
    if total > 0 and commercial_like and improvement < 0.5 * total:
        return "Underdeveloped"
    return None


def build_export(raw_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    boundary_union = unary_union(boundary_gdf.geometry)

    gdf = raw_gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0))
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    representative_points = gdf.geometry.representative_point()
    gdf = gdf[representative_points.within(boundary_union)].copy()
    print(f"After Portland boundary clip: {len(gdf):,}")

    situs_city = gdf["SITUSCITY"].fillna("").astype(str).str.strip().str.upper()
    non_portland_situs_mask = situs_city.ne("") & situs_city.ne("PORTLAND")
    dropped_non_portland = int(non_portland_situs_mask.sum())
    if dropped_non_portland:
        print(f"Dropping {dropped_non_portland:,} parcels with non-Portland SITUSCITY values")
        gdf = gdf[~non_portland_situs_mask].copy()
        situs_city = gdf["SITUSCITY"].fillna("").astype(str).str.strip().str.upper()

    null_situs_count = int(situs_city.eq("").sum())
    print(f"Parcels with blank SITUSCITY retained after boundary clip: {null_situs_count:,}")

    gdf["PROPERTY_TYPE"] = gdf["PROPERTY_TYPE"].fillna("").astype(str)
    excluded_mask = gdf["PROPERTY_TYPE"].str.upper().isin(EXCLUDED_PROPERTY_TYPES) | gdf["PROP_CODE"].fillna("").astype(str).str.upper().eq("IS")
    excluded_count = int(excluded_mask.sum())
    if excluded_count:
        print(f"Dropping {excluded_count:,} machinery/billboard account rows")
        gdf = gdf[~excluded_mask].copy()

    gdf["parcel_id"] = gdf["MAPTAXLOT"].apply(normalize_parcel_id)
    gdf = gdf[gdf["parcel_id"].ne("")].copy()
    gdf = collapse_duplicate_parcels(gdf)

    for col in ["ROLLLAND", "ROLLIMP", "ROLLM50", "ROLLMAV"]:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0)

    owner_text = (
        gdf[["NAME", "NAME2", "CITY"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
        .str.strip()
    )
    government_regex = "|".join(re.escape(pattern) for pattern in GOVERNMENT_OWNER_PATTERNS)
    exempt_by_owner = owner_text.str.contains(government_regex, regex=True, na=False)
    exempt_by_field = gdf["EXEMPTION"].fillna("").astype(str).str.strip().ne("")
    gdf["exemption_flag"] = (exempt_by_owner | exempt_by_field).astype(int)

    before = len(gdf)
    gdf = gdf[gdf["exemption_flag"] == 0].copy()
    print(f"Removed {before - len(gdf):,} exempt parcels -> {len(gdf):,}")

    gdf["property_land_use_category"] = gdf.apply(classify_original, axis=1)
    gdf["current_full_land_value"] = gdf["ROLLLAND"].clip(lower=0)
    gdf["improvement_value"] = gdf["ROLLIMP"].clip(lower=0)
    gdf["full_market_value"] = gdf["current_full_land_value"] + gdf["improvement_value"]
    gdf["property_land_use_refined"] = gdf.apply(classify_refined, axis=1)
    gdf["area_sqft"] = gdf.geometry.apply(geodesic_area_sqft)
    gdf.loc[gdf["area_sqft"] <= 0, "area_sqft"] = np.nan
    gdf["land_value_per_sqft"] = gdf["current_full_land_value"] / gdf["area_sqft"]
    gdf["improvement_value_per_sqft"] = gdf["improvement_value"] / gdf["area_sqft"]
    gdf["full_market_value_per_sqft"] = gdf["full_market_value"] / gdf["area_sqft"]
    gdf["REALLANDVA"] = gdf["current_full_land_value"]
    gdf["REALIMPROV"] = gdf["improvement_value"]
    gdf["REALLANDVA_per_sqft"] = gdf["land_value_per_sqft"]
    gdf["REALIMPROV_per_sqft"] = gdf["improvement_value_per_sqft"]
    gdf["link"] = np.where(
        gdf["PROPID"].fillna("").astype(str).str.strip().ne(""),
        "https://taxgraph.multco.us/property/" + gdf["PROPID"].fillna("").astype(str).str.strip(),
        np.nan,
    )

    gdf = add_improvement_ratio_fields(
        gdf,
        land_col="current_full_land_value",
        improvement_col="improvement_value",
    )

    export_columns = [
        "geometry",
        "parcel_id",
        "MAPTAXLOT",
        "PROPID",
        "NAME",
        "SITUSADDR",
        "SITUSCITY",
        "PROPERTY_TYPE",
        "PROPCLASS",
        "PROP_CODE",
        "EXEMPTION",
        "property_land_use_category",
        "property_land_use_refined",
        "exemption_flag",
        "current_full_land_value",
        "improvement_value",
        "full_market_value",
        "REALLANDVA",
        "REALIMPROV",
        "area_sqft",
        "land_value_per_sqft",
        "improvement_value_per_sqft",
        "full_market_value_per_sqft",
        "REALLANDVA_per_sqft",
        "REALIMPROV_per_sqft",
        "TLLDIMPROV",
        "IMPR_LAND_RATIO",
        "IMPR_LAND_PCT",
        "IMPR_PCT_TOTAL",
        "link",
    ]
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
    today = datetime.now().strftime("%Y_%m_%d")
    raw_glob = "data/jurisidictions/data/portland/portland-or-raw_*.parquet"
    raw_path = OUTPUT_DIR / f"portland-or-raw_{today}.parquet"
    output_path = OUTPUT_DIR / "portland-or-parcels.parquet"
    dated_path = OUTPUT_DIR / f"portland-or-parcels_{today}.parquet"

    boundary = fetch_portland_boundary()
    print(f"Fetched Portland boundary fragments: {len(boundary):,}")

    raw_gdf: gpd.GeoDataFrame
    if args.use_cache:
        cached = latest_cached_raw(raw_glob)
        if cached and cached.exists():
            print(f"Using cached raw parcels: {cached}")
            raw_gdf = gpd.read_parquet(cached)
        else:
            raw_gdf = download_layer(TAXLOTS_QUERY_URL, RAW_FIELDS, raw_path, label="Multnomah County taxlot")
    else:
        raw_gdf = download_layer(TAXLOTS_QUERY_URL, RAW_FIELDS, raw_path, label="Multnomah County taxlot")

    export = build_export(raw_gdf, boundary)
    export.to_parquet(output_path, index=False)
    export.to_parquet(dated_path, index=False)

    print(f"Saved canonical parquet: {output_path}")
    print(f"Saved dated snapshot : {dated_path}")
    print(f"Final parcel count   : {len(export):,}")
    print("\nRefined category counts:")
    print(export["property_land_use_refined"].value_counts(dropna=False).to_string())
    print("\nOriginal category counts:")
    print(export["property_land_use_category"].value_counts(dropna=False).head(25).to_string())

    if not args.skip_upload:
        maybe_upload(output_path)


if __name__ == "__main__":
    main()
