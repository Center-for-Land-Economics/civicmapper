#!/usr/bin/env python3
"""
Build Denver's canonical parcel parquet from the live Denver open data parcels layer.

Source:
- https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/ArcGIS/rest/services/ODC_PROP_PARCELS_A/FeatureServer/245

Outputs:
- data/jurisidictions/data/denver/denver-co-parcels.parquet
- data/jurisidictions/data/denver/denver-co-parcels_YYYY_MM_DD.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.cloud_utils import ensure_geodataframe, get_feature_data_with_geometry
from data.parcel_calculations import add_improvement_ratio_fields


BASE_URL = "https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/ArcGIS/rest/services"
DATASET_NAME = "ODC_PROP_PARCELS_A"
LAYER_ID = 245
OUTPUT_DIR = Path("data/jurisidictions/data/denver")
CITY_FILTER = "DENVER"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Denver canonical parcel parquet.")
    parser.add_argument("--use-cache", action="store_true", help="Reuse the latest cached raw parquet if present.")
    parser.add_argument("--skip-upload", action="store_true", help="Do not upload the canonical parquet.")
    parser.add_argument("--skip-pmtiles", action="store_true", help="Do not build/upload PMTiles artifacts.")
    return parser.parse_args()


def latest_cached_raw() -> Path | None:
    files = sorted(glob.glob(str(OUTPUT_DIR / "denver_parcels_*.parquet")))
    return Path(files[-1]) if files else None


def download_raw(raw_path: Path) -> gpd.GeoDataFrame:
    print("Downloading Denver parcels from ArcGIS FeatureServer...")
    gdf = get_feature_data_with_geometry(
        DATASET_NAME,
        BASE_URL,
        layer_id=LAYER_ID,
        paginate=True,
        out_epsg=4326,
        verbose=True,
    )
    if gdf is None or gdf.empty:
        raise RuntimeError("Denver parcel download returned no features.")
    gdf.to_parquet(raw_path, index=False)
    print(f"Saved raw cache: {raw_path}")
    return ensure_geodataframe(gdf)


def collapse_duplicate_geometries(geoms: pd.Series):
    items = [geom for geom in geoms if geom is not None]
    if not items:
        return None

    remaining = list(items)
    merged = []
    while remaining:
        ref = remaining.pop(0)
        group = [ref]
        rest = []
        for geom in remaining:
            if ref.intersects(geom) or ref.touches(geom) or ref.equals(geom):
                group.append(geom)
            else:
                rest.append(geom)
        merged.append(unary_union(group))
        remaining = rest

    if len(merged) == 1:
        return merged[0]

    polygons = []
    for geom in merged:
        if geom.geom_type == "Polygon":
            polygons.append(geom)
        elif geom.geom_type == "MultiPolygon":
            polygons.extend(geom.geoms)
        else:
            polygons.append(geom)
    return MultiPolygon(polygons)


def collapse_duplicate_parcels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "NUMS_CONCAT" not in gdf.columns:
        print("NUMS_CONCAT not found; skipping duplicate collapse.")
        return gdf

    duplicate_count = gdf.duplicated(subset=["NUMS_CONCAT"], keep=False).sum()
    print(f"Duplicate NUMS_CONCAT rows: {duplicate_count:,}")
    if duplicate_count == 0:
        return gdf

    numeric_sum_cols = [
        col
        for col in gdf.columns
        if col.isupper() and col != "NUMS_CONCAT" and pd.api.types.is_numeric_dtype(gdf[col])
    ]
    categorical_cols = [
        col for col in gdf.columns if col not in set(numeric_sum_cols + ["geometry", "NUMS_CONCAT"])
    ]

    agg = {col: "sum" for col in numeric_sum_cols}
    agg.update({col: "first" for col in categorical_cols})
    agg["geometry"] = collapse_duplicate_geometries

    collapsed = gdf.groupby("NUMS_CONCAT", dropna=False).agg(agg).reset_index()
    result = gpd.GeoDataFrame(collapsed, geometry="geometry", crs=gdf.crs)
    print(f"Rows after duplicate collapse: {len(result):,}")
    return result


def categorize_property_type(d_class_cn: object) -> str:
    category_mapping = {
        "Single Family": [
            "SFR Grade C", "SFR Grade B", "SFR Grade A", "SFR Grade D or E", "SFR Grade X",
            "SFR Grade B w/RK", "SFR Grade C, D, or E, w/RK", "SFR Grade A or X, w/RK",
        ],
        "Condo/Townhouse/Rowhouse": [
            "RESIDENTIAL-CONDOMINIUM", "RESIDENTIAL CONDOMINIUM", "COMMERCIAL-CONDOMINIUM",
            "RESIDENTIAL-ROWHOUSE",
        ],
        "Small Multi-Family (2-4 units)": [
            "RESIDENTIAL-DUPLEX", "RESIDENTIAL-TRIPLEX", "RESIDENTIAL-4 TO 8 UNITS",
        ],
        "Large Multi-Family (5+ units)": [
            "RESIDENTIAL-APARTMENT", "RESIDENTIAL-MULTI UNIT APTS", "RESIDENTIAL-SENIOR CITIZEN APT",
        ],
        "Other Residential": [
            "RESIDENTIAL", "RESIDENTIAL LAND CONTIGUOUS", "RESIDENTIAL  LAND FOR LAND/ IMPS PARCEL",
            "RESIDENTIAL-MISC IMPS", "RESIDENTIAL GRACE YEAR", "RESIDENTIAL-BOARDING HOME",
            "RESIDENTIAL-NURSING FACILITY",
        ],
        "Vacant Land": [
            "VACANT LAND", "VACANT LAND /GENERAL COMMON ELEMENTS", "VACANT LAND W/MINOR STRUCTURE",
            "GENERAL COMMON ELEMENTS", "RESIDENTIAL LAND CONTIGUOUS",
        ],
        "Mobile Home": [
            "MH / Minor Structures", "MOBILE HOME LAND", "Mobile Home Park",
        ],
        "Agricultural/Open": [
            "DRY FARM LAND", "CUR - USE - AG", "Agricultural", "Agricultural Not Classified", "GOLF COURSE",
        ],
        "Commercial": [
            "COMMERCIAL-RETAIL", "COMMERCIAL-OFFICE", "COMMERCIAL-MISC IMPS", "COMMERCIAL-RESTAURANT",
            "COMMERCIAL-SHOPPING CENTER", "COMMERCIAL-MEDICAL OFFICE", "COMMERCIAL-CONDOMINIUM",
            "COMMERCIAL-HOTEL", "COMMERCIAL-PARKING GARAGE", "COMMERCIAL-MOTEL", "COMMERCIAL",
            "COMMERCIAL-THEATER", "COMMERCIAL-FINANCIAL OFFICE", "RETAIL W/MIXED USE",
            "RESTAURANT W/MIXED USE", "HOTEL W/MIXED USE", "WAREHOUSE W/MIXED USE",
            "OFFICE W/MIXED USE", "MOTEL W/MIXED USE", "SHOPPING CENTER W/MIXED USE",
            "FINANCIAL OFFICE W/MIXED USE", "OTHER COMMERCIAL P.I.",
        ],
        "Industrial/Manufacturing": [
            "INDUSTRIAL-WAREHOUSE", "INDUSTRIAL-AUTO SERVICE GARAGE", "INDUSTRIAL-CHURCH",
            "INDUSTRIAL-SCHOOL", "INDUSTRIAL-AUTO DEALER", "INDUSTRIAL-FACTORY",
            "INDUSTRIAL-CONV STORE W/PUMPS", "INDUSTRIAL-MISC RECREATION", "INDUSTRIAL-SERVICE STATION",
            "INDUSTRIAL-PRESCHOOL", "INDUSTRIAL-MEETING HALL", "INDUSTRIAL-CAR WASH",
            "INDUSTRIAL-VETERINARY", "INDUSTRIAL-FOOD PROCESSING", "INDUSTRIAL-CONVERTED CHURCH",
            "INDUSTRIAL-CEMETERY BLDG", "INDUSTRIAL-SHIPPING TERMINAL", "INDUSTRIAL-MORTUARY",
            "INDUSTRIAL-HEALTH CLUB", "INDUSTRIAL-PRINTING PLANT", "INDUSTRIAL-MEAT PACKING",
            "INDUSTRIAL-DRY CLEANING", "INDUSTRIAL-GREENHOUSE", "INDUSTRIAL-GRAIN ELEVATOR",
            "INDUSTRIAL-CITY CLUB", "INDUSTRIAL-BOWLING ALLEY", "INDUSTRIAL-BRICK PLANT",
            "FACTORY W/MIXED USE",
        ],
        "Civic/Institutional": [
            "SPECIAL PURPOSE", "FIRE STATION", "POLICE-FIRE STATION", "COUNTY JAIL", "STOCK SHOW",
            "DENVER PARK", "STADIUM", "STADIUM-MIXED USE", "AIRPORT P.I. RETAIL", "AIRPORT P.I. OTHER",
            "SCHOOL", "SOCIAL/RECREATION W/MIXED USE", "MEDICAL OFFICE W/MIXED USE",
            "ENTERTAINMENT P.I.", "PUBLIC ASSEMBLY", "OTHER CULTURAL", "PARK", "RECREATION P.I.",
        ],
        "Transportation/Utilities": ["TRANSPORTATION", "UTILITIES", "COMMUNICATION"],
        "Mixed Use": [
            "RETAIL W/MIXED USE", "OFFICE W/MIXED USE", "RESTAURANT W/MIXED USE", "HOTEL W/MIXED USE",
            "WAREHOUSE W/MIXED USE", "MOTEL W/MIXED USE", "SHOPPING CENTER W/MIXED USE",
            "FINANCIAL OFFICE W/MIXED USE", "MISC IMPROVEMENTS W/MIXED USE",
            "FOOD PROCESSING W/MIXED USE", "VETERINARY W/MIXED USE", "THEATER W/MIXED USE",
        ],
        "Possessory Interest": ["POSSESSORY INTEREST"],
        "Unknown": [None, "None"],
    }

    for category, descriptions in category_mapping.items():
        if d_class_cn in descriptions:
            return category

    text = str(d_class_cn).upper() if d_class_cn is not None else ""
    if "VACANT" in text or "LAND" in text:
        return "Vacant Land"
    if "RESIDENTIAL" in text:
        if "CONDOMINIUM" in text or "ROWHOUSE" in text:
            return "Condo/Townhouse/Rowhouse"
        if "DUPLEX" in text or "TRIPLEX" in text or "4 TO 8" in text:
            return "Small Multi-Family (2-4 units)"
        if "APARTMENT" in text or "MULTI UNIT" in text or "SENIOR CITIZEN" in text:
            return "Large Multi-Family (5+ units)"
        return "Other Residential"
    if "COMMERCIAL" in text:
        return "Commercial"
    if "INDUSTRIAL" in text:
        return "Industrial/Manufacturing"
    if "MOBILE HOME" in text or "MH" in text:
        return "Mobile Home"
    if "AGRICULTURAL" in text or "FARM" in text:
        return "Agricultural/Open"
    if "PARK" in text or "GOLF" in text:
        return "Civic/Institutional"
    if "SCHOOL" in text or "JAIL" in text or "HOSPITAL" in text or "POLICE" in text or "FIRE" in text:
        return "Civic/Institutional"
    if "MIXED USE" in text:
        return "Mixed Use"
    return "Other"


def build_export(raw_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = ensure_geodataframe(raw_gdf).copy()
    city_mask = gdf["SITUS_CITY"].fillna("").astype(str).str.upper().eq(CITY_FILTER)
    gdf = gdf[city_mask].copy()
    print(f"Rows after Denver filter: {len(gdf):,}")
    gdf = collapse_duplicate_parcels(gdf)

    if "D_CLASS_CN" in gdf.columns:
        gdf["PROPERTY_CATEGORY"] = gdf["D_CLASS_CN"].apply(categorize_property_type)
    else:
        gdf["PROPERTY_CATEGORY"] = "Other"

    if "full_exmp" not in gdf.columns:
        if {"EXEMPT_AMT_LOCAL", "ASSESSED_TOTAL_VALUE_LOCAL"}.issubset(gdf.columns):
            assessed_total = pd.to_numeric(gdf["ASSESSED_TOTAL_VALUE_LOCAL"], errors="coerce").fillna(0)
            exempt_amt = pd.to_numeric(gdf["EXEMPT_AMT_LOCAL"], errors="coerce").fillna(0)
            ratio = exempt_amt / assessed_total.replace(0, np.nan)
            gdf["full_exmp"] = ((ratio >= 0.995) | ((assessed_total <= 0) & (exempt_amt > 0))).astype(int)
        elif "TAXABLE_AMT_LOCAL" in gdf.columns:
            taxable = pd.to_numeric(gdf["TAXABLE_AMT_LOCAL"], errors="coerce").fillna(0)
            gdf["full_exmp"] = (taxable <= 0).astype(int)
        else:
            raise ValueError("Denver export needs taxable or exempt value fields.")

    gdf = gdf[gdf["full_exmp"] == 0].copy()
    gdf["exemption_flag"] = 0

    if "APPRAISED_LAND_VALUE" in gdf.columns:
        gdf["land_value"] = pd.to_numeric(gdf["APPRAISED_LAND_VALUE"], errors="coerce")
    elif "ASSESSED_LAND_VALUE_LOCAL" in gdf.columns:
        gdf["land_value"] = pd.to_numeric(gdf["ASSESSED_LAND_VALUE_LOCAL"], errors="coerce")
    else:
        raise ValueError("Denver export needs a land value field.")

    if "APPRAISED_IMP_VALUE" in gdf.columns:
        gdf["improvement_value"] = pd.to_numeric(gdf["APPRAISED_IMP_VALUE"], errors="coerce").fillna(0)
    elif "ASSESSED_BLDG_VALUE_LOCAL" in gdf.columns:
        gdf["improvement_value"] = pd.to_numeric(gdf["ASSESSED_BLDG_VALUE_LOCAL"], errors="coerce").fillna(0)
    else:
        gdf["improvement_value"] = 0

    gdf["property_land_use_category"] = gdf["PROPERTY_CATEGORY"]

    def categorize_property_refined(row: pd.Series) -> str | None:
        category = str(row.get("PROPERTY_CATEGORY") or "")
        land = float(pd.to_numeric(row.get("land_value"), errors="coerce") or 0.0)
        improvement = float(pd.to_numeric(row.get("improvement_value"), errors="coerce") or 0.0)
        total = land + improvement
        if "Vacant" in category:
            return "Vacant"
        if "Parking" in category:
            return "Parking Lot"
        if total > 0 and improvement < 0.5 * total:
            return "Underdeveloped"
        return None

    gdf["property_land_use_refined"] = gdf.apply(categorize_property_refined, axis=1)

    if "Shape__Area" in gdf.columns:
        gdf["area_sqft"] = pd.to_numeric(gdf["Shape__Area"], errors="coerce")
    else:
        projected = gdf.to_crs(3857) if gdf.crs and gdf.crs.to_epsg() != 3857 else gdf
        gdf["area_sqft"] = projected.geometry.area * 10.763910416709722
    gdf["area_sqft"] = gdf["area_sqft"].replace(0, np.nan)

    if "APPRAISED_TOTAL_VALUE" in gdf.columns:
        gdf["full_market_value"] = pd.to_numeric(gdf["APPRAISED_TOTAL_VALUE"], errors="coerce")
    elif "ASSESSED_TOTAL_VALUE_LOCAL" in gdf.columns:
        gdf["full_market_value"] = pd.to_numeric(gdf["ASSESSED_TOTAL_VALUE_LOCAL"], errors="coerce")
    else:
        gdf["full_market_value"] = gdf["land_value"].fillna(0) + gdf["improvement_value"].fillna(0)

    gdf["full_market_value_per_sqft"] = gdf["full_market_value"] / gdf["area_sqft"]
    gdf["land_value_per_sqft"] = gdf["land_value"] / gdf["area_sqft"]
    gdf["improvement_value_per_sqft"] = gdf["improvement_value"] / gdf["area_sqft"]

    gdf = add_improvement_ratio_fields(gdf, land_col="land_value", improvement_col="improvement_value")
    gdf["link"] = gdf.get("link", np.nan)

    columns_to_export = [
        "geometry",
        "exemption_flag",
        "property_land_use_category",
        "property_land_use_refined",
        "full_market_value",
        "full_market_value_per_sqft",
        "land_value",
        "land_value_per_sqft",
        "improvement_value",
        "improvement_value_per_sqft",
        "TLLDIMPROV",
        "IMPR_LAND_RATIO",
        "IMPR_LAND_PCT",
        "IMPR_PCT_TOTAL",
        "link",
    ]
    for col in columns_to_export:
        if col not in gdf.columns:
            gdf[col] = np.nan

    export_final = gdf[columns_to_export].rename(columns={"land_value": "current_full_land_value"}).copy()
    export_final["geometry"] = export_final["geometry"].apply(
        lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0)
    )
    export_final = gpd.GeoDataFrame(export_final, geometry="geometry", crs=gdf.crs)
    if export_final.crs is None or export_final.crs.to_epsg() != 4326:
        export_final = export_final.to_crs("EPSG:4326")
    return export_final


def maybe_upload_parquet(local_path: Path) -> None:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("AZURE_STORAGE_CONNECTION_STRING not set; skipping parquet upload.")
        return

    from azure.storage.blob import BlobServiceClient

    container = os.getenv("AZURE_DEV_CONTAINER", "parquets-dev")
    blob_name = local_path.name
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container)
    with open(local_path, "rb") as handle:
        container_client.upload_blob(name=blob_name, data=handle, overwrite=True)
    print(f"Uploaded {local_path} -> {container}/{blob_name}")


def maybe_build_pmtiles(skip_upload: bool) -> None:
    script_path = Path("data/scripts/parquet_to_pmtiles.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find {script_path}")

    cmd = [sys.executable, str(script_path), "--city", "denver", "--overwrite"]
    if not skip_upload and os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
        cmd.append("--upload")

    print(f"Running PMTiles conversion: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y_%m_%d")
    raw_path = OUTPUT_DIR / f"denver_parcels_{today_str}.parquet"

    if args.use_cache:
        cached = latest_cached_raw()
        if cached is None:
            print("No cached Denver raw parquet found; downloading fresh data.")
            raw_gdf = download_raw(raw_path)
        else:
            print(f"Loading cached raw parquet: {cached}")
            raw_gdf = ensure_geodataframe(gpd.read_parquet(cached))
    else:
        raw_gdf = download_raw(raw_path)

    export_final = build_export(raw_gdf)
    canonical_path = OUTPUT_DIR / "denver-co-parcels.parquet"
    dated_path = OUTPUT_DIR / f"denver-co-parcels_{today_str}.parquet"
    export_final.to_parquet(canonical_path, index=False)
    export_final.to_parquet(dated_path, index=False)

    print(f"Saved canonical parquet: {canonical_path}")
    print(f"Saved dated parquet: {dated_path}")
    print(f"Total rows: {len(export_final):,}")
    print("Refined category counts:")
    print(export_final["property_land_use_refined"].value_counts(dropna=False).to_string())

    if not args.skip_upload:
        maybe_upload_parquet(canonical_path)
    else:
        print("Skipping parquet upload.")

    if not args.skip_pmtiles:
        maybe_build_pmtiles(skip_upload=args.skip_upload)
    else:
        print("Skipping PMTiles conversion.")


if __name__ == "__main__":
    main()
