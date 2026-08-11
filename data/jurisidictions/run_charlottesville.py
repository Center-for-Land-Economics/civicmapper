#!/usr/bin/env python3
"""
Build Charlottesville's canonical parcel parquet from the city's official ArcGIS services.

Sources:
- Parcel geometry:
  https://gisweb.charlottesville.org/arcgis/rest/services/OpenData_1/MapServer/43
- Parcel area/details:
  https://gisweb.charlottesville.org/arcgis/rest/services/OpenData_1/MapServer/72
- Parcel owner points:
  https://gisweb.charlottesville.org/arcgis/rest/services/OpenData_1/MapServer/74
- Base active parcel table:
  https://gisweb.charlottesville.org/arcgis/rest/services/OpenData_2/MapServer/8
- Assessment history:
  https://gisweb.charlottesville.org/arcgis/rest/services/OpenData_2/MapServer/2
- Residential details:
  https://gisweb.charlottesville.org/arcgis/rest/services/OpenData_2/MapServer/17
- Commercial details:
  https://gisweb.charlottesville.org/arcgis/rest/services/OpenData_2/MapServer/19

Outputs:
- data/jurisidictions/data/charlottesville/charlottesville-va-parcels.parquet
- data/jurisidictions/data/charlottesville/charlottesville-va-parcels_YYYY_MM_DD.parquet
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Geod

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.parcel_calculations import add_improvement_ratio_fields


SERVICE_1 = "https://gisweb.charlottesville.org/arcgis/rest/services/OpenData_1/MapServer"
SERVICE_2 = "https://gisweb.charlottesville.org/arcgis/rest/services/OpenData_2/MapServer"

GEOMETRY_URL = f"{SERVICE_1}/43/query"
PARCEL_DETAILS_URL = f"{SERVICE_1}/72/query"
OWNER_POINTS_URL = f"{SERVICE_1}/74/query"
BASE_ACTIVE_URL = f"{SERVICE_2}/8/query"
ASSESSMENTS_URL = f"{SERVICE_2}/2/query"
RESIDENTIAL_URL = f"{SERVICE_2}/17/query"
COMMERCIAL_URL = f"{SERVICE_2}/19/query"

OUTPUT_DIR = Path("data/jurisidictions/data/charlottesville")
PAGE_SIZE = 1000
GEOD = Geod(ellps="WGS84")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Charlottesville canonical parcel parquet.")
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Accepted for CLI compatibility; upload is handled separately.",
    )
    return parser.parse_args()


def request_json(url: str, params: dict, *, timeout: int = 180) -> dict:
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


def fetch_max_string_stat(url: str, field_name: str) -> str:
    payload = request_json(
        url,
        {
            "f": "json",
            "where": "1=1",
            "returnGeometry": "false",
            "outStatistics": (
                f'[{{"statisticType":"max","onStatisticField":"{field_name}",'
                f'"outStatisticFieldName":"max_value"}}]'
            ),
        },
        timeout=120,
    )
    features = payload.get("features", [])
    if not features:
        raise RuntimeError(f"Could not resolve max statistic for {field_name} from {url}")
    value = features[0]["attributes"].get("max_value")
    if value is None:
        raise RuntimeError(f"Statistic response for {field_name} was empty from {url}")
    return str(value)


def fetch_geo_layer(url: str, out_fields: list[str], *, where: str = "1=1") -> gpd.GeoDataFrame:
    total = fetch_count(url, where)
    frames: list[gpd.GeoDataFrame] = []
    offset = 0
    print(f"Fetching geometry layer ({total:,} rows)")
    while offset < total:
        payload = request_json(
            url,
            {
                "f": "geojson",
                "where": where,
                "outFields": ",".join(out_fields),
                "returnGeometry": "true",
                "outSR": 4326,
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "orderByFields": "OBJECTID",
            },
        )
        frame = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
        if frame.empty:
            break
        frames.append(frame)
        offset += len(frame)
        print(f"  fetched {offset:,}/{total:,}")
        time.sleep(0.03)

    result = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")


def fetch_table(url: str, out_fields: list[str], *, where: str = "1=1", order_by: str) -> pd.DataFrame:
    total = fetch_count(url, where)
    frames: list[pd.DataFrame] = []
    offset = 0
    print(f"Fetching table {url} ({total:,} rows)")
    while offset < total:
        payload = request_json(
            url,
            {
                "f": "json",
                "where": where,
                "outFields": ",".join(out_fields),
                "returnGeometry": "false",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "orderByFields": order_by,
            },
        )
        rows = [feature["attributes"] for feature in payload.get("features", [])]
        frame = pd.DataFrame(rows)
        if frame.empty:
            break
        frames.append(frame)
        offset += len(frame)
        print(f"  fetched {offset:,}/{total:,}")
        time.sleep(0.03)

    if not frames:
        return pd.DataFrame(columns=out_fields)
    return pd.concat(frames, ignore_index=True)


def clean_text(value: object) -> object:
    if value is None:
        return np.nan
    text = str(value).strip()
    return text if text else np.nan


def first_non_null(series: pd.Series) -> object:
    for value in series:
        cleaned = clean_text(value)
        if not pd.isna(cleaned):
            return cleaned
    return np.nan


def combine_unique(series: pd.Series) -> object:
    values: list[str] = []
    seen: set[str] = set()
    for value in series:
        cleaned = clean_text(value)
        if pd.isna(cleaned):
            continue
        cleaned_str = str(cleaned)
        if cleaned_str not in seen:
            seen.add(cleaned_str)
            values.append(cleaned_str)
    if not values:
        return np.nan
    return " | ".join(values)


def geodesic_area_sqft(geom) -> float:
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type == "Polygon":
        lon, lat = geom.exterior.coords.xy
        area_m2, _ = GEOD.polygon_area_perimeter(lon, lat)
        holes_m2 = 0.0
        for ring in geom.interiors:
            lon_h, lat_h = ring.coords.xy
            hole_area, _ = GEOD.polygon_area_perimeter(lon_h, lat_h)
            holes_m2 += abs(hole_area)
        return max(abs(area_m2) - holes_m2, 0.0) * 10.763910416709722
    if geom.geom_type == "MultiPolygon":
        return sum(geodesic_area_sqft(part) for part in geom.geoms)
    return np.nan


def aggregate_by_parcel(df: pd.DataFrame, aggregations: dict[str, object]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["parcel_number", *aggregations.keys()])
    grouped = df.groupby("parcel_number", dropna=False).agg(aggregations).reset_index()
    return grouped


def pick_original_category(use_code: object, state_code: object, zoning: object) -> object:
    for value in (use_code, state_code, zoning):
        cleaned = clean_text(value)
        if not pd.isna(cleaned):
            return cleaned
    return np.nan


def classify_refined(row: pd.Series) -> str | None:
    use_code = str(row.get("use_code") or "").lower()
    state_code = str(row.get("state_code") or "").lower()
    land = float(pd.to_numeric(row.get("current_full_land_value"), errors="coerce") or 0.0)
    improvement = float(pd.to_numeric(row.get("improvement_value"), errors="coerce") or 0.0)
    total = land + improvement

    if "parking lot" in use_code or "parking structure" in use_code:
        return "Parking Lot"

    if "vacant" in use_code or "490 land" in use_code:
        return "Vacant"

    if improvement <= 0 and land > 0:
        return "Vacant"

    if total <= 0:
        return None

    residential_tokens = (
        "single family",
        "duplex",
        "triplex",
        "quadplex",
        "condominium",
        "apartments",
    )
    if any(token in use_code for token in residential_tokens):
        return None
    if "residential" in state_code and "multi-family" not in state_code:
        return None

    if improvement < 0.5 * total:
        return "Underdeveloped"

    return None


def build_export() -> gpd.GeoDataFrame:
    geometries = fetch_geo_layer(GEOMETRY_URL, ["OBJECTID", "GPIN"])
    geometries = geometries.rename(columns={"GPIN": "gpin"})
    geometries["gpin"] = pd.to_numeric(geometries["gpin"], errors="coerce")
    geometries = geometries[geometries["gpin"].notna()].copy()
    geometries = geometries.drop_duplicates(subset=["gpin"]).copy()
    geometries["geometry"] = geometries["geometry"].apply(
        lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0)
    )

    parcel_details = fetch_table(
        PARCEL_DETAILS_URL,
        [
            "GeoParcelIdentificationNumber",
            "ParcelNumber",
            "OwnerName",
            "Assessment",
            "LotSquareFeet",
            "Zoning",
            "StreetNumber",
            "StreetName",
            "Unit",
            "LegalDescription",
        ],
        order_by="OBJECTID",
    )
    parcel_details = parcel_details.rename(
        columns={
            "GeoParcelIdentificationNumber": "gpin_details",
            "ParcelNumber": "parcel_number",
            "OwnerName": "owner_details",
            "Assessment": "current_assessed_value",
            "LotSquareFeet": "lot_sqft_source",
            "Zoning": "zoning_details",
            "StreetNumber": "street_number_details",
            "StreetName": "street_name_details",
            "Unit": "unit_details",
            "LegalDescription": "legal_description_details",
        }
    )
    parcel_details["parcel_number"] = parcel_details["parcel_number"].map(clean_text)
    parcel_details["gpin_details"] = pd.to_numeric(parcel_details["gpin_details"], errors="coerce")
    parcel_details = parcel_details[parcel_details["parcel_number"].notna()].copy()
    parcel_details = aggregate_by_parcel(
        parcel_details,
        {
            "gpin_details": first_non_null,
            "owner_details": first_non_null,
            "current_assessed_value": "max",
            "lot_sqft_source": "max",
            "zoning_details": first_non_null,
            "street_number_details": first_non_null,
            "street_name_details": first_non_null,
            "unit_details": first_non_null,
            "legal_description_details": first_non_null,
        },
    )

    owner_points = fetch_table(
        OWNER_POINTS_URL,
        [
            "GeoParcelIdentificationNumber",
            "ParcelNumber",
            "UseCode",
            "OwnerName",
            "OwnerAddress",
            "OwnerCityState",
            "OwnerZipCode",
            "CurrentAssessedValue",
        ],
        order_by="OBJECTID",
    )
    owner_points = owner_points.rename(
        columns={
            "GeoParcelIdentificationNumber": "gpin_owner",
            "ParcelNumber": "parcel_number",
            "UseCode": "use_code_numeric",
            "OwnerName": "owner_points_name",
            "OwnerAddress": "owner_address",
            "OwnerCityState": "owner_city_state",
            "OwnerZipCode": "owner_zip_code",
            "CurrentAssessedValue": "current_assessed_value_owner_points",
        }
    )
    owner_points["parcel_number"] = owner_points["parcel_number"].map(clean_text)
    owner_points["gpin_owner"] = pd.to_numeric(owner_points["gpin_owner"], errors="coerce")
    owner_points = owner_points[owner_points["parcel_number"].notna()].copy()
    owner_points = aggregate_by_parcel(
        owner_points,
        {
            "gpin_owner": first_non_null,
            "use_code_numeric": first_non_null,
            "owner_points_name": first_non_null,
            "owner_address": first_non_null,
            "owner_city_state": first_non_null,
            "owner_zip_code": first_non_null,
            "current_assessed_value_owner_points": "max",
        },
    )

    base_active = fetch_table(
        BASE_ACTIVE_URL,
        [
            "ParcelNumber",
            "GPIN",
            "TaxType",
            "StateCode",
            "Zone",
            "Acreage",
            "Legal",
            "StreetNumber",
            "StreetName",
            "Unit",
        ],
        order_by="RecordID_Int",
    )
    base_active = base_active.rename(
        columns={
            "ParcelNumber": "parcel_number",
            "GPIN": "gpin_base",
            "TaxType": "tax_type",
            "StateCode": "state_code",
            "Zone": "zone_base",
            "Acreage": "acreage",
            "Legal": "legal_description_base",
            "StreetNumber": "street_number_base",
            "StreetName": "street_name_base",
            "Unit": "unit_base",
        }
    )
    base_active["parcel_number"] = base_active["parcel_number"].map(clean_text)
    base_active["gpin_base"] = pd.to_numeric(base_active["gpin_base"], errors="coerce")
    base_active = base_active[base_active["parcel_number"].notna()].copy()
    base_active = aggregate_by_parcel(
        base_active,
        {
            "gpin_base": first_non_null,
            "tax_type": first_non_null,
            "state_code": first_non_null,
            "zone_base": first_non_null,
            "acreage": "max",
            "legal_description_base": first_non_null,
            "street_number_base": first_non_null,
            "street_name_base": first_non_null,
            "unit_base": first_non_null,
        },
    )

    assessments = fetch_table(
        ASSESSMENTS_URL,
        [
            "RecordID_Int",
            "ParcelNumber",
            "TaxYear",
            "LandValue",
            "ImprovementValue",
            "TotalValue",
        ],
        where=f"TaxYear = '{fetch_max_string_stat(ASSESSMENTS_URL, 'TaxYear')}'",
        order_by="RecordID_Int",
    )
    assessments = assessments.rename(
        columns={
            "ParcelNumber": "parcel_number",
            "TaxYear": "tax_year",
            "LandValue": "current_full_land_value",
            "ImprovementValue": "improvement_value",
            "TotalValue": "full_market_value",
        }
    )
    assessments["parcel_number"] = assessments["parcel_number"].map(clean_text)
    assessments["tax_year_num"] = pd.to_numeric(assessments["tax_year"], errors="coerce")
    assessments = assessments.sort_values(["parcel_number", "tax_year_num", "RecordID_Int"])
    assessments = assessments.groupby("parcel_number", dropna=False).tail(1).copy()

    residential = fetch_table(
        RESIDENTIAL_URL,
        [
            "ParcelNumber",
            "UseCode",
            "YearBuilt",
            "SquareFootageFinishedLiving",
        ],
        order_by="RecordID_Int",
    )
    residential = residential.rename(
        columns={
            "ParcelNumber": "parcel_number",
            "UseCode": "use_code_residential",
            "YearBuilt": "year_built_residential",
            "SquareFootageFinishedLiving": "building_sqft_residential",
        }
    )
    residential["parcel_number"] = residential["parcel_number"].map(clean_text)
    residential["building_sqft_residential"] = pd.to_numeric(
        residential["building_sqft_residential"],
        errors="coerce",
    )
    residential = residential[residential["parcel_number"].notna()].copy()
    residential = aggregate_by_parcel(
        residential,
        {
            "use_code_residential": combine_unique,
            "year_built_residential": first_non_null,
            "building_sqft_residential": "max",
        },
    )

    commercial = fetch_table(
        COMMERCIAL_URL,
        [
            "ParcelNumber",
            "UseCode",
            "YearBuilt",
            "GrossArea",
        ],
        order_by="RecordID_Int",
    )
    commercial = commercial.rename(
        columns={
            "ParcelNumber": "parcel_number",
            "UseCode": "use_code_commercial",
            "YearBuilt": "year_built_commercial",
            "GrossArea": "building_sqft_commercial",
        }
    )
    commercial["parcel_number"] = commercial["parcel_number"].map(clean_text)
    commercial["building_sqft_commercial"] = pd.to_numeric(
        commercial["building_sqft_commercial"],
        errors="coerce",
    )
    commercial = commercial[commercial["parcel_number"].notna()].copy()
    commercial = aggregate_by_parcel(
        commercial,
        {
            "use_code_commercial": combine_unique,
            "year_built_commercial": first_non_null,
            "building_sqft_commercial": "max",
        },
    )

    export = parcel_details.merge(owner_points, on="parcel_number", how="outer")
    export = export.merge(base_active, on="parcel_number", how="left")
    export = export.merge(assessments, on="parcel_number", how="left")
    export = export.merge(residential, on="parcel_number", how="left")
    export = export.merge(commercial, on="parcel_number", how="left")

    export["gpin"] = (
        pd.to_numeric(export["gpin_details"], errors="coerce")
        .combine_first(pd.to_numeric(export["gpin_owner"], errors="coerce"))
        .combine_first(pd.to_numeric(export["gpin_base"], errors="coerce"))
    )
    export = export.merge(geometries[["gpin", "geometry"]], on="gpin", how="left")
    export = gpd.GeoDataFrame(export, geometry="geometry", crs="EPSG:4326")
    export = export[export.geometry.notna() & ~export.geometry.is_empty].copy()

    export["owner"] = export["owner_details"].combine_first(export["owner_points_name"])
    export["legal_description"] = export["legal_description_details"].combine_first(export["legal_description_base"])
    export["zoning"] = export["zoning_details"].combine_first(export["zone_base"])
    export["use_code"] = export["use_code_residential"].combine_first(export["use_code_commercial"])
    export["use_code"] = export["use_code"].combine_first(export["use_code_numeric"])

    export["address"] = (
        export["street_number_details"].fillna(export["street_number_base"]).fillna("").astype(str).str.strip()
        + " "
        + export["street_name_details"].fillna(export["street_name_base"]).fillna("").astype(str).str.strip()
        + " "
        + export["unit_details"].fillna(export["unit_base"]).fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    export.loc[export["address"].eq(""), "address"] = np.nan

    export["parcel_id"] = export["parcel_number"]
    export["exemption_flag"] = export["tax_type"].fillna("").astype(str).str.strip().ne("Taxable").astype(int)
    export["property_land_use_category"] = [
        pick_original_category(use_code, state_code, zoning)
        for use_code, state_code, zoning in zip(
            export["use_code"],
            export["state_code"],
            export["zoning"],
        )
    ]

    for column in [
        "current_assessed_value",
        "current_assessed_value_owner_points",
        "lot_sqft_source",
        "acreage",
        "current_full_land_value",
        "improvement_value",
        "full_market_value",
        "building_sqft_residential",
        "building_sqft_commercial",
    ]:
        if column in export.columns:
            export[column] = pd.to_numeric(export[column], errors="coerce")

    export["current_full_land_value"] = export["current_full_land_value"].clip(lower=0).fillna(0)
    export["improvement_value"] = export["improvement_value"].clip(lower=0).fillna(0)
    total_from_parts = export["current_full_land_value"] + export["improvement_value"]
    export["full_market_value"] = export["full_market_value"].clip(lower=0)
    export["full_market_value"] = export["full_market_value"].where(export["full_market_value"] > 0, total_from_parts)
    export["full_market_value"] = export["full_market_value"].fillna(total_from_parts)

    export["property_land_use_refined"] = export.apply(classify_refined, axis=1)
    export["area_sqft"] = export.geometry.apply(geodesic_area_sqft)
    export.loc[export["area_sqft"] < 1, "area_sqft"] = np.nan
    export["land_value_per_sqft"] = export["current_full_land_value"] / export["area_sqft"]
    export["improvement_value_per_sqft"] = export["improvement_value"] / export["area_sqft"]
    export["full_market_value_per_sqft"] = export["full_market_value"] / export["area_sqft"]
    export["REALLANDVA"] = export["current_full_land_value"]
    export["REALIMPROV"] = export["improvement_value"]
    export["REALLANDVA_per_sqft"] = export["land_value_per_sqft"]
    export["REALIMPROV_per_sqft"] = export["improvement_value_per_sqft"]

    export = add_improvement_ratio_fields(
        export,
        land_col="REALLANDVA",
        improvement_col="REALIMPROV",
    )

    export = export[export["exemption_flag"] == 0].copy()
    export = export.drop_duplicates(subset=["parcel_number"]).copy()
    export = export.sort_values("parcel_number").reset_index(drop=True)

    export_columns = [
        "geometry",
        "parcel_id",
        "parcel_number",
        "gpin",
        "property_land_use_category",
        "property_land_use_refined",
        "use_code",
        "state_code",
        "tax_type",
        "zoning",
        "owner",
        "address",
        "legal_description",
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
    ]
    return export[export_columns].copy()


def main() -> None:
    parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y_%m_%d")
    output_path = OUTPUT_DIR / "charlottesville-va-parcels.parquet"
    dated_path = OUTPUT_DIR / f"charlottesville-va-parcels_{today}.parquet"

    export = build_export()
    export.to_parquet(output_path, index=False)
    export.to_parquet(dated_path, index=False)

    print(f"Saved canonical parquet: {output_path}")
    print(f"Saved dated snapshot : {dated_path}")
    print(f"Final parcel count   : {len(export):,}")
    print("\nRefined category counts:")
    print(export["property_land_use_refined"].value_counts(dropna=False).to_string())
    print("\nOriginal category counts:")
    print(export["property_land_use_category"].value_counts(dropna=False).head(25).to_string())


if __name__ == "__main__":
    main()
