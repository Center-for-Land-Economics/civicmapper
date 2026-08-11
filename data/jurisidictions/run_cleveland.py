#!/usr/bin/env python3
"""
Build Cleveland's canonical parcel parquet from the live Cuyahoga County MyPlace layer.

Source:
- https://gis.cuyahogacounty.us/server/rest/services/MyPLACE/Parcels_WMA_GJOIN_WGS84/MapServer/2

Outputs:
- data/jurisidictions/data/cleveland/cleveland-oh-parcels.parquet
- data/jurisidictions/data/cleveland/cleveland-oh-parcels_YYYY_MM_DD.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
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


QUERY_URL = "https://gis.cuyahogacounty.us/server/rest/services/MyPLACE/Parcels_WMA_GJOIN_WGS84/MapServer/2/query"
CITY_WHERE = "UPPER(par_city) = 'CLEVELAND'"
PAGE_SIZE = 1000
OUTPUT_DIR = Path("data/jurisidictions/data/cleveland")
RAW_FIELDS = [
    "objectid",
    "parcelpin",
    "PARCEL_PK",
    "parcel_owner",
    "mail_name",
    "par_addr_all",
    "par_city",
    "tax_luc",
    "tax_luc_description",
    "ext_luc",
    "ext_luc_description",
    "property_class",
    "tax_district",
    "tax_abatement",
    "condo_complex_id",
    "certified_tax_land",
    "certified_tax_building",
    "certified_tax_total",
    "certified_exempt_land",
    "certified_exempt_building",
    "certified_exempt_total",
    "gross_certified_land",
    "gross_certified_building",
    "gross_certified_total",
    "total_square_ft",
    "total_acreage",
]
EXEMPT_CLASS_CODES = {"E", "CE", "RE", "IE"}
RESIDENTIAL_HINTS = [
    "1-FAMILY",
    "2-FAMILY",
    "3-FAMILY",
    "4- 6 UNIT",
    "APARTMENT",
    "APTS",
    "CONDO",
    "TOWNHOUSE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cleveland canonical parcel parquet.")
    parser.add_argument("--use-cache", action="store_true", help="Reuse the latest cached raw parquet if present.")
    parser.add_argument("--skip-upload", action="store_true", help="Do not upload the canonical parquet.")
    return parser.parse_args()


def request_json(params: dict, *, timeout: int = 240) -> dict:
    response = requests.get(QUERY_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS error: {payload['error']}")
    return payload


def fetch_count() -> int:
    payload = request_json(
        {
            "f": "json",
            "where": CITY_WHERE,
            "returnCountOnly": "true",
        },
        timeout=120,
    )
    return int(payload["count"])


def fetch_page(offset: int) -> gpd.GeoDataFrame:
    payload = request_json(
        {
            "f": "geojson",
            "where": CITY_WHERE,
            "outFields": ",".join(RAW_FIELDS),
            "returnGeometry": "true",
            "outSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
        }
    )
    gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    for field in RAW_FIELDS:
        if field not in gdf.columns:
            gdf[field] = np.nan
    return gdf


def latest_cached_raw(raw_glob: str) -> Path | None:
    files = sorted(glob.glob(raw_glob))
    return Path(files[-1]) if files else None


def download_raw(raw_path: Path) -> gpd.GeoDataFrame:
    total = fetch_count()
    print(f"Found {total:,} Cleveland parcel records")
    frames: list[gpd.GeoDataFrame] = []
    offset = 0
    while offset < total:
        frame = fetch_page(offset)
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


def collapse_duplicate_parcels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    duplicate_count = gdf.duplicated(subset=["parcelpin"], keep=False).sum()
    print(f"Duplicate parcelpin rows: {duplicate_count:,}")
    if duplicate_count == 0:
        return gdf

    numeric_candidates = [
        "certified_tax_land",
        "certified_tax_building",
        "certified_tax_total",
        "certified_exempt_land",
        "certified_exempt_building",
        "certified_exempt_total",
        "gross_certified_land",
        "gross_certified_building",
        "gross_certified_total",
        "total_square_ft",
        "total_acreage",
    ]
    numeric_cols = [c for c in numeric_candidates if c in gdf.columns]
    categorical_cols = [c for c in gdf.columns if c not in set(numeric_cols + ["geometry", "parcelpin"])]

    agg = {c: "sum" for c in numeric_cols}
    agg.update({c: "first" for c in categorical_cols})
    collapsed = gdf.groupby("parcelpin", dropna=False).agg(agg).reset_index()
    geom_union = gdf.groupby("parcelpin", dropna=False)["geometry"].apply(
        lambda geoms: unary_union([geom for geom in geoms if geom is not None])
        if any(geom is not None for geom in geoms)
        else None
    )
    collapsed["geometry"] = geom_union.values
    result = gpd.GeoDataFrame(collapsed, geometry="geometry", crs=gdf.crs)
    print(f"Rows after duplicate collapse: {len(result):,}")
    return result


def original_category(row: pd.Series) -> str:
    desc = str(row.get("tax_luc_description") or "").strip()
    if desc:
        return desc.title()

    property_class = str(row.get("property_class") or "").strip().upper()
    return {
        "R": "Residential",
        "C": "Commercial",
        "I": "Industrial",
        "P": "Public / Transportation",
        "H": "Hospitality / Apartments",
        "A": "Agricultural",
        "LW": "Listed With Other Parcel",
        "E": "Exempt",
        "CE": "Commercial Exempt",
        "RE": "Residential Exempt",
        "IE": "Industrial Exempt",
    }.get(property_class, "Other")


def refined_category(row: pd.Series) -> str | None:
    desc = str(row.get("tax_luc_description") or "").upper()
    if "VAC" in desc or "VACANT" in desc:
        return "Vacant"
    if "PARKING" in desc:
        return "Parking Lot"

    land = float(pd.to_numeric(row.get("current_full_land_value"), errors="coerce") or 0.0)
    improvement = float(pd.to_numeric(row.get("improvement_value"), errors="coerce") or 0.0)
    total = land + improvement
    if total <= 0:
        return None

    if any(hint in desc for hint in RESIDENTIAL_HINTS) and improvement > 0:
        return None

    if improvement < 0.5 * total:
        return "Underdeveloped"
    return None


def build_export(raw_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = raw_gdf.copy()
    gdf = gdf[gdf["par_city"].fillna("").str.upper() == "CLEVELAND"].copy()
    gdf = collapse_duplicate_parcels(gdf)

    for col in [
        "certified_tax_land",
        "certified_tax_building",
        "certified_tax_total",
        "certified_exempt_total",
    ]:
        gdf[col] = pd.to_numeric(gdf.get(col), errors="coerce")

    exempt_class = gdf["property_class"].fillna("").str.upper().isin(EXEMPT_CLASS_CODES)
    exempt_value = gdf["certified_exempt_total"].fillna(0) > 0
    gdf["exemption_flag"] = (exempt_class | exempt_value).astype(int)
    before = len(gdf)
    gdf = gdf[gdf["exemption_flag"] == 0].copy()
    print(f"Removed {before - len(gdf):,} exempt parcels -> {len(gdf):,} remaining")

    gdf["property_land_use_category"] = gdf.apply(original_category, axis=1)
    gdf["current_full_land_value"] = gdf["certified_tax_land"].fillna(0)
    gdf["improvement_value"] = gdf["certified_tax_building"].fillna(0)
    gdf["full_market_value"] = gdf["current_full_land_value"] + gdf["improvement_value"]
    gdf["property_land_use_refined"] = gdf.apply(refined_category, axis=1)

    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0))
    gdf["area_sqft"] = gdf["geometry"].apply(geodesic_area_sqft)
    gdf.loc[gdf["area_sqft"] < 1, "area_sqft"] = np.nan

    gdf["land_value_per_sqft"] = gdf["current_full_land_value"] / gdf["area_sqft"]
    gdf["improvement_value_per_sqft"] = gdf["improvement_value"] / gdf["area_sqft"]
    gdf["full_market_value_per_sqft"] = gdf["full_market_value"] / gdf["area_sqft"]

    gdf["REALLANDVA"] = gdf["current_full_land_value"]
    gdf["REALIMPROV"] = gdf["improvement_value"]
    gdf["REALLANDVA_per_sqft"] = gdf["land_value_per_sqft"]
    gdf["REALIMPROV_per_sqft"] = gdf["improvement_value_per_sqft"]

    gdf = add_improvement_ratio_fields(
        gdf,
        land_col="REALLANDVA",
        improvement_col="REALIMPROV",
    )
    gdf["TLLDIMPROV_per_sqft"] = gdf["TLLDIMPROV"] / gdf["area_sqft"]

    gdf["link"] = np.nan

    columns = [
        "geometry",
        "parcelpin",
        "par_addr_all",
        "parcel_owner",
        "mail_name",
        "tax_luc",
        "tax_luc_description",
        "ext_luc",
        "ext_luc_description",
        "property_class",
        "tax_district",
        "tax_abatement",
        "condo_complex_id",
        "exemption_flag",
        "property_land_use_category",
        "property_land_use_refined",
        "current_full_land_value",
        "improvement_value",
        "full_market_value",
        "land_value_per_sqft",
        "improvement_value_per_sqft",
        "full_market_value_per_sqft",
        "REALLANDVA",
        "REALIMPROV",
        "REALLANDVA_per_sqft",
        "REALIMPROV_per_sqft",
        "TLLDIMPROV",
        "TLLDIMPROV_per_sqft",
        "IMPR_LAND_RATIO",
        "IMPR_LAND_PCT",
        "IMPR_PCT_TOTAL",
        "area_sqft",
        "link",
    ]
    export = gdf[columns].copy()
    return gpd.GeoDataFrame(export, geometry="geometry", crs="EPSG:4326")


def maybe_upload(parquet_path: Path) -> None:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("AZURE_STORAGE_CONNECTION_STRING not set; skipping upload")
        return

    from azure.storage.blob import BlobServiceClient

    container = os.getenv("AZURE_DEV_CONTAINER", "parquets-dev")
    blob_name = parquet_path.name
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container)
    with open(parquet_path, "rb") as handle:
        container_client.upload_blob(blob_name, handle, overwrite=True)
    print(f"Uploaded to {container}/{blob_name}")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y_%m_%d")
    raw_path = OUTPUT_DIR / f"cleveland_raw_{today_str}.parquet"
    canonical_path = OUTPUT_DIR / "cleveland-oh-parcels.parquet"
    dated_path = OUTPUT_DIR / f"cleveland-oh-parcels_{today_str}.parquet"

    if args.use_cache:
        cached = latest_cached_raw(str(OUTPUT_DIR / "cleveland_raw_*.parquet"))
        if not cached:
            raise FileNotFoundError("No cached Cleveland raw parquet found. Run without --use-cache first.")
        print(f"Loading cached raw parquet: {cached}")
        raw_gdf = gpd.read_parquet(cached)
    else:
        raw_gdf = download_raw(raw_path)

    export = build_export(raw_gdf)
    export.to_parquet(canonical_path, index=False)
    export.to_parquet(dated_path, index=False)

    print(f"Saved canonical parquet: {canonical_path}")
    print(f"Saved dated parquet: {dated_path}")
    print(f"Exported rows: {len(export):,}")
    print("Refined category counts:")
    print(export["property_land_use_refined"].value_counts(dropna=False).to_string())

    if not args.skip_upload:
        maybe_upload(canonical_path)


if __name__ == "__main__":
    main()
