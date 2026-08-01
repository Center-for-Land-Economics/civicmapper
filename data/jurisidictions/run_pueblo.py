#!/usr/bin/env python3
"""
Build Pueblo's canonical parcel parquet from live Pueblo County services.

Sources:
- Parcels:
  https://maps.co.pueblo.co.us/outside/rest/services/Landbase/PuebloCounty_Parcels/MapServer/1
- Municipal boundaries:
  https://maps.co.pueblo.co.us/outside/rest/services/Landbase/PuebloCounty_MunicipalCountyBoundaries/MapServer/0
- City zoning:
  https://maps.co.pueblo.co.us/outside/rest/services/Landbase/PuebloCounty_ZoningCountyCity/MapServer/0

Outputs:
- data/jurisidictions/data/pueblo/pueblo-co-parcels.parquet
- data/jurisidictions/data/pueblo/pueblo-co-parcels_YYYY_MM_DD.parquet
"""

from __future__ import annotations

import argparse
import glob
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


PARCEL_QUERY_URL = (
    "https://maps.co.pueblo.co.us/outside/rest/services/"
    "Landbase/PuebloCounty_Parcels/MapServer/1/query"
)
BOUNDARY_QUERY_URL = (
    "https://maps.co.pueblo.co.us/outside/rest/services/"
    "Landbase/PuebloCounty_MunicipalCountyBoundaries/MapServer/0/query"
)
ZONING_QUERY_URL = (
    "https://maps.co.pueblo.co.us/outside/rest/services/"
    "Landbase/PuebloCounty_ZoningCountyCity/MapServer/0/query"
)
CITY_NAME = "PUEBLO"
PAGE_SIZE = 1000
OUTPUT_DIR = Path("data/jurisidictions/data/pueblo")
RAW_FIELDS = [
    "OBJECTID",
    "PAR_NUM",
    "PAR_TXT",
    "Owner",
    "TaxExempt",
    "Neighborhood",
    "Subdivision",
    "LandActualValue",
    "ImprovementsActualValue",
    "AssessorURL",
]
ZONING_FIELDS = ["OBJECTID", "ZONE_CODE", "MAP_AMENDMENT", "ZoningURL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Pueblo canonical parcel parquet.")
    parser.add_argument("--use-cache", action="store_true", help="Reuse the latest cached raw parquet if present.")
    parser.add_argument("--skip-upload", action="store_true", help="Accepted for CLI compatibility; upload is handled separately.")
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
    files = sorted(glob.glob(raw_glob))
    return Path(files[-1]) if files else None


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


def fetch_pueblo_boundary() -> gpd.GeoDataFrame:
    payload = request_json(
        BOUNDARY_QUERY_URL,
        {
            "f": "geojson",
            "where": f"UPPER(City_Name)='{CITY_NAME}'",
            "outFields": "City_Name",
            "returnGeometry": "true",
            "outSR": 4326,
        },
    )
    boundary = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    if boundary.empty:
        raise RuntimeError("Failed to fetch Pueblo municipal boundary.")
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


def collapse_duplicate_parcels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    duplicate_count = gdf.duplicated(subset=["PAR_TXT"], keep=False).sum()
    print(f"Duplicate PAR_TXT rows: {duplicate_count:,}")
    if duplicate_count == 0:
        return gdf

    numeric_cols = ["LandActualValue", "ImprovementsActualValue"]
    categorical_cols = [c for c in gdf.columns if c not in set(numeric_cols + ["geometry", "PAR_TXT"])]
    agg = {c: "sum" for c in numeric_cols}
    agg.update({c: "first" for c in categorical_cols})
    collapsed = gdf.groupby("PAR_TXT", dropna=False).agg(agg).reset_index()
    geom_union = gdf.groupby("PAR_TXT", dropna=False)["geometry"].apply(
        lambda geoms: unary_union([geom for geom in geoms if geom is not None])
        if any(geom is not None for geom in geoms)
        else None
    )
    collapsed["geometry"] = geom_union.values
    result = gpd.GeoDataFrame(collapsed, geometry="geometry", crs=gdf.crs)
    print(f"Rows after duplicate collapse: {len(result):,}")
    return result


def classify_original(zone_code: str, exempt_flag: int, improvement_value: float) -> str:
    zone = (zone_code or "").strip().upper()
    if zone:
        return zone
    if exempt_flag:
        return "EXEMPT"
    if improvement_value <= 0:
        return "UNCLASSIFIED VACANT"
    return "UNCLASSIFIED"


def classify_refined(row: pd.Series) -> str | None:
    zone = str(row.get("zone_code") or "").strip().upper()
    land = float(pd.to_numeric(row.get("current_full_land_value"), errors="coerce") or 0.0)
    improvement = float(pd.to_numeric(row.get("improvement_value"), errors="coerce") or 0.0)
    total = land + improvement

    if improvement <= 0 and land > 0:
        return "Vacant"

    commercial_like = zone.startswith(("B", "I"))
    if commercial_like and total > 0 and improvement < 0.5 * total:
        return "Underdeveloped"

    return None


def build_export(raw_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame, zoning_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = raw_gdf.copy()
    gdf = gdf[gdf.geometry.notna()].copy()

    boundary_union = unary_union(boundary_gdf.geometry)
    gdf = gdf[gdf.geometry.intersects(boundary_union)].copy()
    gdf = collapse_duplicate_parcels(gdf)

    for col in ["LandActualValue", "ImprovementsActualValue"]:
        gdf[col] = pd.to_numeric(gdf.get(col), errors="coerce").fillna(0)

    gdf["exemption_flag"] = (
        gdf["TaxExempt"].fillna("").astype(str).str.strip().str.upper().eq("Y").astype(int)
    )

    zone_subset = zoning_gdf[["zone_code", "zoning_url", "geometry"]].copy()
    centroids = gdf.geometry.representative_point()
    centroids_gdf = gpd.GeoDataFrame(gdf[["PAR_TXT"]].copy(), geometry=centroids, crs=gdf.crs)
    zone_join = gpd.sjoin(centroids_gdf, zone_subset, how="left", predicate="within")
    zone_join = zone_join.groupby("PAR_TXT", dropna=False).first().reset_index()
    gdf = gdf.merge(zone_join[["PAR_TXT", "zone_code", "zoning_url"]], on="PAR_TXT", how="left")

    def clean_text(series: pd.Series) -> pd.Series:
        text = series.fillna("").astype(str).str.strip()
        return text.mask(text.eq(""), np.nan)

    gdf["owner"] = clean_text(gdf["Owner"])
    gdf["neighborhood"] = clean_text(gdf["Neighborhood"])
    gdf["subdivision"] = clean_text(gdf["Subdivision"])
    gdf["assessor_url"] = clean_text(gdf["AssessorURL"])

    gdf["property_land_use_category"] = [
        classify_original(zone, exempt, improvement)
        for zone, exempt, improvement in zip(
            gdf["zone_code"].fillna(""),
            gdf["exemption_flag"],
            gdf["ImprovementsActualValue"],
        )
    ]

    gdf["current_full_land_value"] = gdf["LandActualValue"].clip(lower=0)
    gdf["improvement_value"] = gdf["ImprovementsActualValue"].clip(lower=0)
    gdf["full_market_value"] = gdf["current_full_land_value"] + gdf["improvement_value"]
    gdf["property_land_use_refined"] = gdf.apply(classify_refined, axis=1)
    gdf["area_sqft"] = gdf.geometry.apply(geodesic_area_sqft)
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

    gdf = gdf[gdf["exemption_flag"] == 0].copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    export_columns = [
        "geometry",
        "PAR_TXT",
        "zone_code",
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
        "owner",
        "neighborhood",
        "subdivision",
        "assessor_url",
        "zoning_url",
    ]
    export = gdf[export_columns].copy()
    export = export.rename(columns={"PAR_TXT": "parcel_id"})
    return export


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y_%m_%d")
    raw_glob = str(OUTPUT_DIR / "pueblo-co-raw_*.parquet")
    raw_path = OUTPUT_DIR / f"pueblo-co-raw_{today}.parquet"
    zoning_path = OUTPUT_DIR / f"pueblo-co-zoning_{today}.parquet"
    output_path = OUTPUT_DIR / "pueblo-co-parcels.parquet"
    dated_path = OUTPUT_DIR / f"pueblo-co-parcels_{today}.parquet"

    boundary = fetch_pueblo_boundary()
    print(f"Fetched Pueblo boundary fragments: {len(boundary):,}")

    raw_gdf: gpd.GeoDataFrame
    if args.use_cache:
        cached = latest_cached_raw(raw_glob)
        if cached and cached.exists():
            print(f"Using cached raw parcels: {cached}")
            raw_gdf = gpd.read_parquet(cached)
        else:
            raw_gdf = download_layer(PARCEL_QUERY_URL, RAW_FIELDS, raw_path, label="Pueblo County parcel")
    else:
        raw_gdf = download_layer(PARCEL_QUERY_URL, RAW_FIELDS, raw_path, label="Pueblo County parcel")

    zoning_gdf = download_layer(ZONING_QUERY_URL, ZONING_FIELDS, zoning_path, label="Pueblo zoning")
    zoning_gdf = zoning_gdf.rename(columns={"ZONE_CODE": "zone_code", "ZoningURL": "zoning_url"})
    zoning_gdf = zoning_gdf[zoning_gdf.geometry.notna()].copy()

    export = build_export(raw_gdf, boundary, zoning_gdf)
    export.to_parquet(output_path, index=False)
    export.to_parquet(dated_path, index=False)

    print(f"Saved canonical parquet: {output_path}")
    print(f"Saved dated snapshot : {dated_path}")
    print(f"Final parcel count   : {len(export):,}")
    print("\nRefined category counts:")
    print(export["property_land_use_refined"].value_counts(dropna=False).to_string())
    print("\nOriginal category counts:")
    print(export["property_land_use_category"].value_counts(dropna=False).head(20).to_string())


if __name__ == "__main__":
    main()
