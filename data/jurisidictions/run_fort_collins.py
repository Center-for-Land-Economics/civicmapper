#!/usr/bin/env python3
"""
Build Fort Collins' canonical parcel parquet from official Larimer County sources.

Sources:
- Parcel geometry:
  https://maps1.larimer.org/arcgis/rest/services/MapServices/Parcels/MapServer/3
- Assessor public data center:
  https://www.larimer.gov/assessor/publicdata

Outputs:
- data/jurisidictions/data/fortcollins/fortcollins-co-parcels.parquet
- data/jurisidictions/data/fortcollins/fortcollins-co-parcels_YYYY_MM_DD.parquet
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from os.path import commonprefix

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Geod
from shapely.ops import unary_union

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.parcel_calculations import add_improvement_ratio_fields


QUERY_URL = "https://maps1.larimer.org/arcgis/rest/services/MapServices/Parcels/MapServer/3/query"
PLATTED_LOTS_URL = "https://maps1.larimer.org/arcgis/rest/services/MapServices/Parcels/MapServer/4/query"
CITY_WHERE = "UPPER(IS_INCORPORATED_NAME) = 'FORT COLLINS'"
PAGE_SIZE = 1000
OBJECT_ID_CHUNK_SIZE = 100
OUTPUT_DIR = Path("data/jurisidictions/data/fortcollins")
CONDO_COMMON_AREA_TOLERANCE_METERS = 2.0

ACCOUNT_URL = "https://storage.googleapis.com/lc-public/asr/assessor-public-account.csv"
OWNER_URL = "https://storage.googleapis.com/lc-public/asr/assessor-public-owner.csv"
VALUE_URL = "https://storage.googleapis.com/lc-public/asr/assessor-public-value-detail.csv"
IMPROVEMENT_URL = "https://storage.googleapis.com/lc-public/asr/assessor-public-improvement.csv"

RAW_FIELDS = [
    "PARCELNUM",
    "LOCADDRESS",
    "NAME",
    "NAME1",
    "ACCTTYPE",
    "LOCCITY",
    "SCHEDNUM",
]

PLATTED_LOT_FIELDS = [
    "OBJECTID",
    "LABEL",
    "SUBNUM",
    "SUBNAME",
    "RUNDATE",
]

ACCOUNT_COLUMNS = [
    "PARCELNO",
    "ACCOUNTNO",
    "SCHEDULENUM",
    "ACCTTYPE",
    "APPRAISALTYPE",
    "SITUSADDRESS",
    "SITUSCITY",
    "SUBDIVISIONNAME",
    "BUILDINGCOUNT",
    "LANDGROSSACRES",
    "LANDGROSSSF",
    "TAXYEAR",
]

OWNER_COLUMNS = [
    "PARCELNO",
    "ACCOUNTNO",
    "SCHEDULENUM",
    "NAME1",
    "NAME2",
    "MAILADDRESS1",
    "MAILADDRESS2",
    "MAILCITY",
    "MAILSTATE",
    "MAILZIPCODE",
    "TAXYEAR",
]

VALUE_COLUMNS = [
    "PARCELNO",
    "ACCOUNTNO",
    "SCHEDULENUM",
    "VALUETYPE",
    "ABSTRACTTYPE",
    "ABSTRACTCODE",
    "ABSTRACTDESCRIPTION",
    "CLASSIFICATIONID",
    "CLASSIFICATIONDESCRIPTION",
    "LANDACRES",
    "LANDSF",
    "LANDUNITCOUNT",
    "ACTUALVALUE",
    "TAXYEAR",
]

IMPROVEMENT_COLUMNS = [
    "PARCELNO",
    "ACCOUNTNO",
    "SCHEDULENUM",
    "IMPACTUALVALUE",
    "IMPNO",
    "PROPERTYTYPE",
    "OCCDESCRIPTION",
    "SF",
    "CONDOIMPSF",
    "CLASSDESCRIPTION",
    "BLTASYEARBUILT",
    "TAXYEAR",
]

EXEMPT_ACCOUNT_TYPES = {"EXEMPT", "NON TAXABLE"}
EXEMPT_ABSTRACT_HINTS = [
    "POLITICAL",
    "CHURCH",
    "COLLEGE",
    "SCHOOL",
    "EXEMPT",
    "HOUSING AUTHORITY",
]
GOVERNMENT_OWNER_PATTERNS = [
    "CITY OF FORT COLLINS",
    "FORT COLLINS MUNICIPAL",
    "COUNTY OF LARIMER",
    "STATE OF COLORADO",
    "COLORADO STATE UNIVERSITY",
    "POUDRE SCHOOL DISTRICT",
    "HOUSING AUTHORITY OF THE CITY OF FORT COLLINS",
    "UNITED STATES OF AMERICA",
    "UNITED STATES ",
]
RESIDENTIAL_HINTS = [
    "SINGLE FAMILY",
    "CONDO",
    "TOWNHOUSE",
    "DUPLEX",
    "TRIPLEX",
    "APARTMENT",
    "PATIO HOME",
    "MOBILE HOME",
    "MH IN PARK",
    "RESIDENCE",
    "RESIDENTIAL",
]
VACANT_HINTS = [
    "VACANT",
    "UNIMP",
    "LAND ONLY",
    "NOBLD",
]
PARKING_HINTS = [
    "PARKING LOT",
    "PARKING STRUCTURE",
    "PARKING GARAGE",
]
NON_PARKING_EXCLUSIONS = [
    "MH IN PARK",
    "MH PARK",
    "MOBILE HOME PARK",
]
COMMON_PLAT_LABEL_PATTERN = re.compile(r"\bGCE\b|COMMON|TR\b|TRACT|OPEN SPACE|LCE", re.IGNORECASE)
CONDO_PARENT_MIN_RATIO = 1.05
ASSOCIATION_PARENT_PATTERN = re.compile(
    r"ASSOCIATION|ASSN|OWNERS|OWNER'?S|HOA|COMMON AREA|COMMON ELEMENT|MASTER ASSOCIATION",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Fort Collins canonical parcel parquet.")
    parser.add_argument("--use-cache", action="store_true", help="Reuse the latest cached raw geometry parquet if present.")
    return parser.parse_args()


def request_json(params: dict, *, timeout: int = 240, url: str = QUERY_URL) -> dict:
    response = requests.post(url, data=params, timeout=timeout)
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


def fetch_object_ids() -> list[int]:
    payload = request_json(
        {
            "f": "json",
            "where": CITY_WHERE,
            "returnIdsOnly": "true",
        },
        timeout=120,
    )
    object_ids = sorted(payload.get("objectIds", []))
    if not object_ids:
        raise RuntimeError("No parcel object IDs returned for Fort Collins.")
    return object_ids


def fetch_object_ids_for_geometry(
    url: str,
    bounds: tuple[float, float, float, float],
) -> list[int]:
    xmin, ymin, xmax, ymax = bounds
    payload = request_json(
        {
            "f": "json",
            "where": "1=1",
            "geometry": json.dumps(
                {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "spatialReference": {"wkid": 4326},
                }
            ),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "returnIdsOnly": "true",
        },
        timeout=120,
        url=url,
    )
    return sorted(payload.get("objectIds", []))


def fetch_chunk(object_ids: list[int]) -> gpd.GeoDataFrame:
    payload = request_json(
        {
            "f": "geojson",
            "objectIds": ",".join(str(v) for v in object_ids),
            "outFields": ",".join(RAW_FIELDS),
            "returnGeometry": "true",
            "outSR": 4326,
        }
    )
    gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    for field in RAW_FIELDS:
        if field not in gdf.columns:
            gdf[field] = np.nan
    return gdf


def fetch_platted_lot_chunk(object_ids: list[int]) -> gpd.GeoDataFrame:
    payload = request_json(
        {
            "f": "geojson",
            "objectIds": ",".join(str(v) for v in object_ids),
            "outFields": ",".join(PLATTED_LOT_FIELDS),
            "returnGeometry": "true",
            "outSR": 4326,
        },
        url=PLATTED_LOTS_URL,
    )
    gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    for field in PLATTED_LOT_FIELDS:
        if field not in gdf.columns:
            gdf[field] = np.nan
    return gdf


def latest_cached_raw(raw_glob: str) -> Path | None:
    files = sorted(glob.glob(raw_glob))
    return Path(files[-1]) if files else None


def download_raw(raw_path: Path) -> gpd.GeoDataFrame:
    object_ids = fetch_object_ids()
    total = len(object_ids)
    print(f"Found {total:,} Fort Collins parcel records")
    frames: list[gpd.GeoDataFrame] = []
    for idx in range(0, total, OBJECT_ID_CHUNK_SIZE):
        chunk = object_ids[idx : idx + OBJECT_ID_CHUNK_SIZE]
        frame = fetch_chunk(chunk)
        if frame.empty:
            break
        frames.append(frame)
        fetched = min(idx + len(chunk), total)
        print(f"  fetched {fetched:,}/{total:,}")
        time.sleep(0.03)

    raw = pd.concat(frames, ignore_index=True)
    raw_gdf = gpd.GeoDataFrame(raw, geometry="geometry", crs="EPSG:4326")
    raw_gdf.to_parquet(raw_path, index=False)
    print(f"Saved raw cache: {raw_path}")
    return raw_gdf


def download_platted_lots(raw_gdf: gpd.GeoDataFrame, lots_path: Path) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = raw_gdf.total_bounds
    object_ids = fetch_object_ids_for_geometry(PLATTED_LOTS_URL, (xmin, ymin, xmax, ymax))
    if not object_ids:
        raise RuntimeError("No platted lot object IDs returned for Fort Collins extent.")

    total = len(object_ids)
    print(f"Found {total:,} platted lot records")
    frames: list[gpd.GeoDataFrame] = []
    for idx in range(0, total, OBJECT_ID_CHUNK_SIZE):
        chunk = object_ids[idx : idx + OBJECT_ID_CHUNK_SIZE]
        frame = fetch_platted_lot_chunk(chunk)
        if frame.empty:
            break
        frames.append(frame)
        fetched = min(idx + len(chunk), total)
        print(f"  fetched {fetched:,}/{total:,}")
        time.sleep(0.03)

    lots = pd.concat(frames, ignore_index=True)
    lots_gdf = gpd.GeoDataFrame(lots, geometry="geometry", crs="EPSG:4326")
    lots_gdf.to_parquet(lots_path, index=False)
    print(f"Saved platted lots cache: {lots_path}")
    return lots_gdf


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


def collapse_duplicate_schedules(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    duplicate_count = gdf.duplicated(subset=["SCHEDNUM"], keep=False).sum()
    print(f"Duplicate SCHEDNUM rows: {duplicate_count:,}")
    if duplicate_count == 0:
        return gdf

    numeric_cols: list[str] = []
    categorical_cols = [c for c in gdf.columns if c not in set(numeric_cols + ["geometry", "SCHEDNUM"])]

    agg = {c: "sum" for c in numeric_cols}
    agg.update({c: "first" for c in categorical_cols})
    collapsed = gdf.groupby("SCHEDNUM", dropna=False).agg(agg).reset_index()

    def cleaned_union(geoms: pd.Series):
        cleaned = []
        for geom in geoms:
            if geom is None:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom is None or geom.is_empty:
                continue
            cleaned.append(geom)
        if not cleaned:
            return None
        return unary_union(cleaned)

    geom_union = gdf.groupby("SCHEDNUM", dropna=False)["geometry"].apply(
        cleaned_union
    )
    collapsed["geometry"] = geom_union.values
    result = gpd.GeoDataFrame(collapsed, geometry="geometry", crs=gdf.crs)
    print(f"Rows after duplicate collapse: {len(result):,}")
    return result


def load_filtered_csv(url: str, schednums: set[str], usecols: list[str], *, chunksize: int = 50000) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(url, dtype=str, usecols=usecols, chunksize=chunksize):
        sub = chunk[chunk["SCHEDULENUM"].isin(schednums)].copy()
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def pick_primary_text(df: pd.DataFrame, key_col: str, text_col: str, weight_col: str) -> pd.DataFrame:
    work = df[[key_col, text_col, weight_col]].copy()
    work[text_col] = work[text_col].fillna("").astype(str).str.strip()
    work = work[work[text_col] != ""]
    if work.empty:
        return pd.DataFrame(columns=[key_col, text_col])
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0)
    work = work.sort_values([key_col, weight_col, text_col], ascending=[True, False, True])
    return work.drop_duplicates(subset=[key_col], keep="first")[[key_col, text_col]]


def first_latest(df: pd.DataFrame, key_col: str, taxyear_col: str = "TAXYEAR") -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work[taxyear_col] = pd.to_numeric(work[taxyear_col], errors="coerce")
    work = work.sort_values([key_col, taxyear_col], ascending=[True, False])
    return work.drop_duplicates(subset=[key_col], keep="first")


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def group_label(values: pd.Series) -> str | None:
    unique = [collapse_whitespace(str(v)) for v in values.dropna().astype(str) if str(v).strip()]
    if not unique:
        return None
    unique = list(dict.fromkeys(unique))
    if len(unique) == 1:
        return unique[0]

    prefix = commonprefix(unique).rstrip()
    if prefix:
        if " " in prefix:
            prefix = prefix[: prefix.rfind(" ")].strip()
        if len(prefix) >= 8:
            return prefix

    return f"{unique[0]} (+{len(unique) - 1} more)"


def joined_unique(values: pd.Series, limit: int = 8) -> str | None:
    unique = [collapse_whitespace(str(v)) for v in values.dropna().astype(str) if str(v).strip()]
    unique = list(dict.fromkeys(unique))
    if not unique:
        return None
    return " | ".join(unique[:limit])


def most_common_value(values: pd.Series) -> str | None:
    cleaned = [collapse_whitespace(str(v)) for v in values.dropna().astype(str) if str(v).strip()]
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    top_count = counts.iloc[0]
    top_values = counts[counts == top_count].index.tolist()
    if len(top_values) == 1:
        return top_values[0]
    return sorted(top_values)[0]


def first_plus_more(values: pd.Series) -> str | None:
    unique = [collapse_whitespace(str(v)) for v in values.dropna().astype(str) if str(v).strip()]
    unique = list(dict.fromkeys(unique))
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return f"{unique[0]} (+{len(unique) - 1} more)"


def base_identifier(value: object) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    return text.split(" (+", 1)[0].strip()


def collapsed_identifier(values: pd.Series, total_count: int | None = None) -> str | None:
    base_values = [base_identifier(v) for v in values.dropna()]
    unique = [v for v in dict.fromkeys(base_values) if v]
    if not unique:
        return None
    count = total_count if total_count is not None else len(unique)
    if count <= 1:
        return unique[0]
    return f"{unique[0]} (+{count - 1} more)"


def subdivision_label(values: pd.Series) -> str | None:
    unique = [collapse_whitespace(str(v)) for v in values.dropna().astype(str) if str(v).strip()]
    unique = list(dict.fromkeys(unique))
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    prefix = commonprefix(unique).rstrip(" ,-/")
    if len(prefix) >= 8:
        return prefix
    return first_plus_more(pd.Series(unique))


SUBDIVISION_STOPWORDS = {
    "AND",
    "AT",
    "COND",
    "CONDO",
    "CONDOS",
    "CONDOMINIUM",
    "CONDOMINIUMS",
    "UNIT",
    "UNITS",
    "BUILDING",
    "BUILDINGS",
    "BLDG",
    "BLD",
    "BLDGS",
    "FTC",
    "PUD",
    "PLAT",
}

SUBDIVISION_BREAKWORDS = {
    "SUPP",
    "SUP",
    "FIRST",
    "SECOND",
    "THIRD",
    "FOURTH",
    "FIFTH",
    "SIXTH",
    "SEVENTH",
    "EIGHTH",
    "NINTH",
    "TENTH",
    "ELEVENTH",
    "TWELFTH",
    "THIRTEENTH",
    "FOURTEENTH",
    "FIFTEENTH",
    "SIXTEENTH",
    "SEVENTEENTH",
    "EIGHTEENTH",
    "NINETEENTH",
    "TWENTIETH",
    "BLOCK",
    "BLK",
    "BUILDING",
    "BUILDINGS",
    "BLDG",
    "BLD",
    "BLDGS",
    "UNIT",
    "UNITS",
    "PH",
    "PHASE",
    "CARPORT",
    "SPACE",
    "AMENDMENT",
    "AMENDED",
    "AMD",
    "AMND",
    "AMNDMNT",
    "CORRECTED",
    "COMMON",
    "COMMONS",
    "COM",
    "THRU",
}


def subdivision_root(value: object) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    tokens = [token for token in re.findall(r"[A-Z0-9]+", text.upper()) if token not in SUBDIVISION_STOPWORDS]
    if not tokens:
        return None
    return tokens[0]


def subdivision_family(value: object) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    tokens = []
    for token in re.findall(r"[A-Z0-9]+", text.upper()):
        if token in SUBDIVISION_BREAKWORDS:
            break
        if token in SUBDIVISION_STOPWORDS:
            continue
        if len(token) == 1 and token.isalpha():
            continue
        if re.fullmatch(r"\d+(?:ST|ND|RD|TH)?", token):
            continue
        if token.isdigit() and len(token) >= 4:
            continue
        tokens.append(token)
    if not tokens:
        return None
    return " ".join(tokens[:3])


def normalized_base_address(value: object) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    text = text.upper()
    if text == "NONE" or text.startswith("NONE "):
        return None
    text = re.sub(r"\s+\(\+\d+\s+MORE\)$", "", text)
    if text == "NONE":
        return None
    text = re.sub(r"\s+(APT|UNIT|STE|SUITE|BLDG|BUILDING|#)\s*.*$", "", text)
    text = re.sub(r"\s+[A-Z]\d+[A-Z0-9-]*$", "", text)
    text = re.sub(r"\s+\d+[A-Z]?$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def merged_condo_category(categories: pd.Series) -> str:
    values = [str(v).lower() for v in categories.dropna()]
    has_commercial = any("commercial condo" in v for v in values)
    has_residential = any("res condo" in v or "res mu condo" in v for v in values)
    has_town = any("town" in v for v in values)
    if has_commercial and has_residential:
        return "Mixed-Use Condo"
    if has_commercial:
        return "Commercial Condo"
    if has_town and not has_residential:
        return "Townhome Condo"
    return "Residential Condo"


def condo_group_kind(category: object, account_type: object) -> str:
    text = " ".join(filter(None, [normalize_text(category), normalize_text(account_type)])).upper()
    if "COMMERCIAL" in text or text.startswith("COM "):
        return "commercial"
    if "TOWN" in text:
        return "townhome"
    return "residential"


def connected_components(indexes: list[int], adjacency: dict[int, set[int]]) -> list[list[int]]:
    remaining = set(indexes)
    components: list[list[int]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    return components


def is_common_plat_label(value: object) -> bool:
    text = normalize_text(value)
    if not text:
        return False
    return bool(COMMON_PLAT_LABEL_PATTERN.search(text))


def is_raw_association_parent(row: pd.Series) -> bool:
    name_text = " ".join(
        part for part in [normalize_text(row.get("NAME")), normalize_text(row.get("NAME1"))] if part
    )
    accttype = normalize_text(row.get("ACCTTYPE")) or ""
    if not name_text and not accttype:
        return False
    if ASSOCIATION_PARENT_PATTERN.search(name_text):
        return True
    accttype = accttype.upper()
    return "NON TAX" in accttype or "EXEMPT" in accttype


def build_condo_ground_parent_candidates(
    raw_gdf: gpd.GeoDataFrame,
    lots_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    raw_parent = raw_gdf.copy()
    raw_parent["SCHEDNUM"] = raw_parent["SCHEDNUM"].fillna("").astype(str).str.strip()
    raw_parent["is_blank_parent"] = raw_parent["SCHEDNUM"].eq("")
    raw_parent["is_assoc_parent"] = raw_parent.apply(is_raw_association_parent, axis=1)
    raw_parent = raw_parent[raw_parent["is_blank_parent"] | raw_parent["is_assoc_parent"]].copy()
    raw_parent = raw_parent[raw_parent["PARCELNUM"].notna()].copy()
    if not raw_parent.empty:
        raw_parent["geometry"] = raw_parent["geometry"].apply(
            lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0)
        )
        raw_parent = raw_parent[raw_parent["geometry"].notna() & ~raw_parent["geometry"].is_empty].copy()
        raw_parent["parent_area_rank"] = raw_parent["geometry"].apply(geodesic_area_sqft)
        raw_parent["parent_priority"] = np.where(raw_parent["is_assoc_parent"], 1, 0)
        raw_parent = raw_parent.sort_values(
            ["PARCELNUM", "parent_priority", "parent_area_rank"],
            ascending=[True, False, False],
        ).drop_duplicates(subset=["PARCELNUM"], keep="first")
        raw_parent["parent_source"] = "raw"
        raw_parent["parent_key"] = raw_parent["PARCELNUM"].astype(str)
        raw_parent["parent_family"] = np.nan

    common_lots = lots_gdf.copy()
    common_lots["geometry"] = common_lots["geometry"].apply(
        lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0)
    )
    common_lots = common_lots[common_lots["geometry"].notna() & ~common_lots["geometry"].is_empty].copy()
    common_lots["label_norm"] = common_lots["LABEL"].fillna("").astype(str).str.strip()
    common_lots = common_lots[common_lots["label_norm"].apply(is_common_plat_label)].copy()
    if not common_lots.empty:
        common_lots["parent_source"] = "plat_common"
        common_lots["parent_key"] = common_lots["OBJECTID"].astype(int).astype(str)
        common_lots["parent_family"] = common_lots["SUBNAME"].apply(subdivision_family)

    raw_parent = gpd.GeoDataFrame(raw_parent, geometry="geometry", crs=raw_gdf.crs)
    common_lots = gpd.GeoDataFrame(common_lots, geometry="geometry", crs=lots_gdf.crs)
    return raw_parent, common_lots


def fill_geometry_holes(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return type(geom)(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        parts = [type(part)(part.exterior) for part in geom.geoms if part is not None and not part.is_empty]
        if not parts:
            return geom
        return unary_union(parts)
    return geom


def subdivision_token_set(value: object) -> set[str]:
    text = normalize_text(value)
    if not text:
        return set()
    tokens: list[str] = []
    for token in re.findall(r"[A-Z0-9]+", text.upper()):
        if token in SUBDIVISION_BREAKWORDS:
            break
        if token in SUBDIVISION_STOPWORDS:
            continue
        if len(token) == 1 and token.isalpha():
            continue
        if re.fullmatch(r"\d+(?:ST|ND|RD|TH)?", token):
            continue
        if token.isdigit() and len(token) >= 4:
            continue
        tokens.append(token)
    return set(tokens)


def collapse_ground_parent_campuses(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()
    work["campus_family"] = work["subdivision_name"].apply(subdivision_family)
    work["base_address"] = work["situs_address"].apply(normalized_base_address)
    work["land_gross_sqft_num"] = pd.to_numeric(work["land_gross_sqft"], errors="coerce").fillna(0)

    campus = work[
        work["account_type"].fillna("").isin(["Merged Condo Ground Parcel", "Merged Condo Common Plat"])
        & work["campus_family"].notna()
        & work["base_address"].notna()
    ].copy()
    other = work.drop(index=campus.index, errors="ignore").copy()
    if campus.empty:
        return gdf

    candidate_groups: list[pd.DataFrame] = []
    for (_, _), group in campus.groupby(["campus_family", "base_address"], dropna=False):
        if len(group) < 2:
            continue
        total_sched = int(pd.to_numeric(group["merged_schedule_count"], errors="coerce").fillna(1).sum())
        if total_sched < 4:
            continue
        candidate_groups.append(group)

    if not candidate_groups:
        return gdf

    assigned: set[int] = set()
    collapsed_rows: list[dict] = []

    for group in sorted(
        candidate_groups,
        key=lambda grp: (
            -int(pd.to_numeric(grp["merged_schedule_count"], errors="coerce").fillna(1).sum()),
            -float(grp["land_gross_sqft_num"].sum()),
        ),
    ):
        remaining = [idx for idx in group.index if idx not in assigned]
        if len(remaining) < 2:
            continue

        subgroup = campus.loc[remaining].copy()
        geometry = fill_geometry_holes(unary_union(subgroup["geometry"].tolist()))
        area_sqft = geodesic_area_sqft(geometry)
        if not pd.notna(area_sqft) or area_sqft <= 0:
            area_sqft = float(subgroup["land_gross_sqft_num"].sum())
        if not pd.notna(area_sqft) or area_sqft <= 0:
            continue

        current_full_land_value = pd.to_numeric(subgroup["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(subgroup["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(subgroup["improvement_sqft"], errors="coerce").fillna(0).sum()
        merged_unit_count = int(pd.to_numeric(subgroup["merged_unit_count"], errors="coerce").fillna(1).sum())
        merged_parcel_count = int(pd.to_numeric(subgroup["merged_parcel_count"], errors="coerce").fillna(1).sum())
        merged_schedule_count = int(pd.to_numeric(subgroup["merged_schedule_count"], errors="coerce").fillna(1).sum())

        row = subgroup.iloc[0].copy()
        row["geometry"] = geometry
        row["parcel_number"] = collapsed_identifier(subgroup["parcel_number"], merged_parcel_count)
        row["schedule_number"] = collapsed_identifier(subgroup["schedule_number"], merged_schedule_count)
        row["account_number"] = collapsed_identifier(subgroup["account_number"], merged_schedule_count)
        row["owner_name"] = "Multiple Owners" if subgroup["owner_name"].dropna().nunique() > 1 else group_label(subgroup["owner_name"])
        row["situs_address"] = group_label(subgroup["base_address"]) or group_label(subgroup["situs_address"]) or row["situs_address"]
        row["situs_city"] = group_label(subgroup["situs_city"]) or row["situs_city"]
        row["subdivision_name"] = subdivision_label(subgroup["subdivision_name"]) or row["subdivision_name"]
        row["account_type"] = "Merged Condo Campus"
        row["classification_description"] = group_label(subgroup["classification_description"]) or row["classification_description"]
        row["source_abstract_description"] = joined_unique(subgroup["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(subgroup["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(subgroup["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(subgroup["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = area_sqft
        row["area_sqft"] = area_sqft
        row["property_land_use_category"] = merged_condo_category(subgroup["property_land_use_category"])
        row["property_land_use_refined"] = np.nan
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / area_sqft if area_sqft > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / area_sqft if area_sqft > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / area_sqft if area_sqft > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = merged_unit_count
        row["merged_parcel_count"] = merged_parcel_count
        row["merged_schedule_count"] = merged_schedule_count
        row["merged_source_categories"] = joined_unique(subgroup["property_land_use_category"], limit=12)
        collapsed_rows.append(row.to_dict())
        assigned.update(remaining)

    if not collapsed_rows:
        return gdf

    remaining = campus.drop(index=list(assigned), errors="ignore").drop(
        columns=["campus_family", "base_address", "land_gross_sqft_num"],
        errors="ignore",
    )
    other = other.drop(columns=["campus_family", "base_address", "land_gross_sqft_num"], errors="ignore")
    combined = pd.concat(
        [other, remaining, gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)],
        ignore_index=True,
        sort=False,
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(f"Collapsed {len(assigned):,} ground-parent condo rows into {len(collapsed_rows):,} campus groups")
    return combined


def collapse_named_parent_complexes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()
    work["token_set"] = work["subdivision_name"].apply(subdivision_token_set)
    work["area_sqft_num"] = pd.to_numeric(work["area_sqft"], errors="coerce").fillna(0)
    work["merged_schedule_count_num"] = pd.to_numeric(work["merged_schedule_count"], errors="coerce").fillna(1)
    work["name_text"] = work["subdivision_name"].fillna("").astype(str).str.upper()

    child_mask = (
        work["token_set"].apply(bool)
        & (
            work["property_land_use_category"].fillna("").str.contains("condo|multi-unit", case=False, na=False)
            | work["account_type"].fillna("").isin(["Platted Parcel Merge", "Multiple Unit"])
            | work["name_text"].str.contains("CONDO|PIER", case=False, na=False)
        )
        & work["area_sqft_num"].between(1, 15000)
    )
    parent_mask = (
        work["token_set"].apply(bool)
        & work["area_sqft_num"].ge(20000)
        & ~child_mask
        & (
            work["account_type"].fillna("").str.contains("Commercial|Residential|Platted", case=False, na=False)
            | work["property_land_use_category"].fillna("").str.contains("special purpose|commercial|res", case=False, na=False)
            | work["name_text"].str.contains("CONDO|PIER", case=False, na=False)
        )
    )

    children = work[child_mask].copy()
    parents = work[parent_mask].copy()
    other = work[~child_mask & ~parent_mask].copy()
    if children.empty or parents.empty:
        return gdf

    child_3857 = children[["geometry", "token_set", "area_sqft_num", "merged_schedule_count_num"]].copy().to_crs(3857)
    parent_3857 = parents[["geometry", "token_set", "area_sqft_num"]].copy().to_crs(3857)
    joined = gpd.sjoin(
        child_3857,
        parent_3857,
        how="inner",
        predicate="intersects",
        lsuffix="child",
        rsuffix="parent",
    )
    if joined.empty:
        return gdf

    joined = joined[
        joined.apply(
            lambda row: bool(row["token_set_child"] & row["token_set_parent"])
            and float(row["area_sqft_num_parent"]) >= float(row["area_sqft_num_child"]) * 3.0,
            axis=1,
        )
    ].copy()
    if joined.empty:
        return gdf

    candidates: list[dict[str, object]] = []
    right_index_col = "index_parent" if "index_parent" in joined.columns else "index_right"
    for parent_idx, group in joined.groupby(right_index_col, dropna=False):
        child_indexes = sorted(group.index.unique().tolist())
        total_sched = int(pd.to_numeric(group["merged_schedule_count_num"], errors="coerce").fillna(1).sum())
        if total_sched < 6 or len(child_indexes) < 3:
            continue
        candidates.append(
            {
                "parent_idx": int(parent_idx),
                "child_indexes": child_indexes,
                "total_sched": total_sched,
                "parent_area_sqft": float(group["area_sqft_num_parent"].iloc[0]),
            }
        )

    if not candidates:
        return gdf

    assigned_children: set[int] = set()
    assigned_parents: set[int] = set()
    collapsed_rows: list[dict] = []

    for candidate in sorted(candidates, key=lambda item: (-item["total_sched"], -item["parent_area_sqft"])):
        parent_idx = candidate["parent_idx"]
        if parent_idx in assigned_parents:
            continue
        remaining_children = [idx for idx in candidate["child_indexes"] if idx not in assigned_children]
        if len(remaining_children) < 3:
            continue

        parent_row = parents.loc[parent_idx]
        child_group = children.loc[remaining_children].copy()
        total_sched = int(pd.to_numeric(child_group["merged_schedule_count"], errors="coerce").fillna(1).sum())
        if total_sched < 6:
            continue

        all_rows = pd.concat([child_group, parents.loc[[parent_idx]]], ignore_index=False)
        geometry = fill_geometry_holes(unary_union(all_rows["geometry"].tolist()))
        area_sqft = geodesic_area_sqft(geometry)
        if not pd.notna(area_sqft) or area_sqft <= 0:
            area_sqft = float(pd.to_numeric(parent_row["area_sqft"], errors="coerce") or 0)
        if not pd.notna(area_sqft) or area_sqft <= 0:
            continue

        current_full_land_value = pd.to_numeric(all_rows["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(all_rows["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(all_rows["improvement_sqft"], errors="coerce").fillna(0).sum()
        merged_unit_count = int(pd.to_numeric(child_group["merged_unit_count"], errors="coerce").fillna(1).sum())
        merged_parcel_count = int(pd.to_numeric(all_rows["merged_parcel_count"], errors="coerce").fillna(1).sum())
        merged_schedule_count = int(pd.to_numeric(all_rows["merged_schedule_count"], errors="coerce").fillna(1).sum())

        row = parent_row.copy()
        row["geometry"] = geometry
        row["parcel_number"] = parent_row["parcel_number"]
        row["schedule_number"] = collapsed_identifier(all_rows["schedule_number"], merged_schedule_count)
        row["account_number"] = collapsed_identifier(all_rows["account_number"], merged_schedule_count)
        row["owner_name"] = "Multiple Owners" if all_rows["owner_name"].dropna().nunique() > 1 else group_label(all_rows["owner_name"])
        row["situs_address"] = group_label(all_rows["situs_address"]) or parent_row["situs_address"]
        row["situs_city"] = group_label(all_rows["situs_city"]) or parent_row["situs_city"]
        row["subdivision_name"] = subdivision_label(all_rows["subdivision_name"]) or parent_row["subdivision_name"]
        row["account_type"] = "Merged Named Complex Parent"
        row["classification_description"] = group_label(all_rows["classification_description"]) or parent_row["classification_description"]
        row["source_abstract_description"] = joined_unique(all_rows["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(all_rows["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(all_rows["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(all_rows["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = area_sqft
        row["area_sqft"] = area_sqft
        row["property_land_use_category"] = most_common_value(child_group["property_land_use_category"]) or parent_row["property_land_use_category"]
        row["property_land_use_refined"] = np.nan
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / area_sqft if area_sqft > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / area_sqft if area_sqft > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / area_sqft if area_sqft > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = merged_unit_count
        row["merged_parcel_count"] = merged_parcel_count
        row["merged_schedule_count"] = merged_schedule_count
        row["merged_source_categories"] = joined_unique(all_rows["property_land_use_category"], limit=12)
        collapsed_rows.append(row.to_dict())
        assigned_children.update(remaining_children)
        assigned_parents.add(parent_idx)

    if not collapsed_rows:
        return gdf

    children_remaining = children.drop(index=list(assigned_children), errors="ignore")
    parents_remaining = parents.drop(index=list(assigned_parents), errors="ignore")
    combined = pd.concat(
        [other, children_remaining, parents_remaining, gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)],
        ignore_index=True,
        sort=False,
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(
        f"Collapsed {len(assigned_children):,} child rows into {len(collapsed_rows):,} named parent-complex groups"
    )
    return combined.drop(columns=["token_set", "area_sqft_num", "merged_schedule_count_num", "name_text"], errors="ignore")


def absorb_remaining_rows_into_named_parents(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()
    work["token_set"] = work["subdivision_name"].apply(subdivision_token_set)
    work["area_sqft_num"] = pd.to_numeric(work["area_sqft"], errors="coerce").fillna(0)

    parents = work[work["account_type"].fillna("").eq("Merged Named Complex Parent")].copy()
    others = work[~work["account_type"].fillna("").eq("Merged Named Complex Parent")].copy()
    if parents.empty or others.empty:
        return gdf

    child_mask = (
        others["token_set"].apply(bool)
        & others["area_sqft_num"].between(1, 12000)
        & (
            others["property_land_use_category"].fillna("").str.contains("condo|multi-unit", case=False, na=False)
            | others["account_type"].fillna("").isin(["Platted Parcel Merge", "Multiple Unit"])
        )
    )
    children = others[child_mask].copy()
    non_children = others.drop(index=children.index, errors="ignore").copy()
    if children.empty:
        return gdf

    assigned_children: set[int] = set()
    assigned_parents: set[int] = set()
    collapsed_rows: list[dict] = []

    parents_3857 = parents.to_crs(3857)
    children_3857 = children.to_crs(3857)

    for parent_idx, parent_row_3857 in parents_3857.iterrows():
        parent_tokens = parents.at[parent_idx, "token_set"]
        candidate_mask = children.index.to_series().apply(lambda idx: idx not in assigned_children)
        candidate_idxs = children[candidate_mask].index.tolist()
        if not candidate_idxs:
            continue
        token_idxs = [
            idx for idx in candidate_idxs
            if bool(children.at[idx, "token_set"] & parent_tokens)
        ]
        if not token_idxs:
            continue
        child_geoms_3857 = children_3857.loc[token_idxs]
        spatial_idxs = child_geoms_3857[
            child_geoms_3857["geometry"].intersects(parent_row_3857.geometry)
        ].index.tolist()
        if not spatial_idxs:
            continue

        parent_row = parents.loc[int(parent_idx)]
        child_group = children.loc[spatial_idxs].copy()
        all_rows = pd.concat([child_group, parents.loc[[int(parent_idx)]]], ignore_index=False)
        geometry = fill_geometry_holes(unary_union(all_rows["geometry"].tolist()))
        area_sqft = geodesic_area_sqft(geometry)
        if not pd.notna(area_sqft) or area_sqft <= 0:
            area_sqft = float(pd.to_numeric(parent_row["area_sqft"], errors="coerce") or 0)
        if not pd.notna(area_sqft) or area_sqft <= 0:
            continue

        current_full_land_value = pd.to_numeric(all_rows["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(all_rows["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(all_rows["improvement_sqft"], errors="coerce").fillna(0).sum()
        merged_unit_count = int(pd.to_numeric(all_rows["merged_unit_count"], errors="coerce").fillna(1).sum())
        merged_parcel_count = int(pd.to_numeric(all_rows["merged_parcel_count"], errors="coerce").fillna(1).sum())
        merged_schedule_count = int(pd.to_numeric(all_rows["merged_schedule_count"], errors="coerce").fillna(1).sum())

        row = parent_row.copy()
        row["geometry"] = geometry
        row["schedule_number"] = collapsed_identifier(all_rows["schedule_number"], merged_schedule_count)
        row["account_number"] = collapsed_identifier(all_rows["account_number"], merged_schedule_count)
        row["owner_name"] = "Multiple Owners" if all_rows["owner_name"].dropna().nunique() > 1 else group_label(all_rows["owner_name"])
        row["situs_address"] = group_label(all_rows["situs_address"]) or parent_row["situs_address"]
        row["situs_city"] = group_label(all_rows["situs_city"]) or parent_row["situs_city"]
        row["subdivision_name"] = subdivision_label(all_rows["subdivision_name"]) or parent_row["subdivision_name"]
        row["classification_description"] = group_label(all_rows["classification_description"]) or parent_row["classification_description"]
        row["source_abstract_description"] = joined_unique(all_rows["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(all_rows["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(all_rows["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(all_rows["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = area_sqft
        row["area_sqft"] = area_sqft
        row["property_land_use_category"] = most_common_value(all_rows["property_land_use_category"]) or parent_row["property_land_use_category"]
        row["property_land_use_refined"] = np.nan
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / area_sqft if area_sqft > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / area_sqft if area_sqft > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / area_sqft if area_sqft > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = merged_unit_count
        row["merged_parcel_count"] = merged_parcel_count
        row["merged_schedule_count"] = merged_schedule_count
        row["merged_source_categories"] = joined_unique(all_rows["property_land_use_category"], limit=12)
        collapsed_rows.append(row.to_dict())
        assigned_children.update(spatial_idxs)
        assigned_parents.add(int(parent_idx))

    if not collapsed_rows:
        return gdf

    remaining_children = children.drop(index=list(assigned_children), errors="ignore")
    parents_remaining = parents.drop(index=list(assigned_parents), errors="ignore")
    combined = pd.concat(
        [non_children, remaining_children, gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)],
        ignore_index=True,
        sort=False,
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(f"Absorbed {len(assigned_children):,} remaining child rows into {len(collapsed_rows):,} named parents")
    return combined.drop(columns=["token_set", "area_sqft_num"], errors="ignore")


def absorb_support_rows_into_family_anchors(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()
    area_series = work["area_sqft"] if "area_sqft" in work.columns else work["land_gross_sqft"]
    work["family"] = work["subdivision_name"].apply(subdivision_family)
    work["token_set"] = work["subdivision_name"].apply(subdivision_token_set)
    work["area_sqft_num"] = pd.to_numeric(area_series, errors="coerce").fillna(0)
    work["merged_schedule_count_num"] = pd.to_numeric(work["merged_schedule_count"], errors="coerce").fillna(1)
    work["owner_text"] = work["owner_name"].fillna("").astype(str)

    anchor_mask = (
        work["token_set"].apply(bool)
        & (
            work["account_type"].fillna("").isin(
                [
                    "Merged Condo Ground Parcel",
                    "Merged Condo Common Plat",
                    "Merged Condo Campus",
                    "Merged Named Complex Parent",
                ]
            )
            | (
                work["property_land_use_category"].fillna("").str.contains("condo|multi-unit", case=False, na=False)
                & work["merged_schedule_count_num"].ge(4)
            )
        )
    )
    support_mask = (
        work["token_set"].apply(bool)
        & ~anchor_mask
        & (
            work["owner_text"].str.contains(ASSOCIATION_PARENT_PATTERN, na=False)
            | work["account_type"].fillna("").isin(["Platted Parcel Merge", "Multiple Unit"])
            | work["property_land_use_category"].fillna("").str.contains(
                "unimp|vacant|special purpose|condo|multi-unit",
                case=False,
                na=False,
            )
        )
        & work["area_sqft_num"].between(1, 80000)
    )

    anchors = work[anchor_mask].copy()
    supports = work[support_mask].copy()
    other = work[~anchor_mask & ~support_mask].copy()
    if anchors.empty or supports.empty:
        return gdf

    anchors_3857 = anchors.to_crs(3857)
    supports_3857 = supports.to_crs(3857)

    assigned_supports: set[int] = set()
    assigned_anchors: set[int] = set()
    collapsed_rows: list[dict] = []

    for anchor_idx, anchor_row_3857 in anchors_3857.sort_values(
        ["merged_schedule_count_num", "area_sqft_num"],
        ascending=[False, False],
    ).iterrows():
        anchor_tokens = anchors.at[anchor_idx, "token_set"]
        family = anchors.at[anchor_idx, "family"]
        candidate_idxs = [
            idx
            for idx in supports.index
            if idx not in assigned_supports
            and bool(supports.at[idx, "token_set"] & anchor_tokens)
            and (
                not family
                or supports.at[idx, "family"] == family
                or bool(supports.at[idx, "token_set"] & anchor_tokens)
            )
        ]
        if not candidate_idxs:
            continue

        support_group_3857 = supports_3857.loc[candidate_idxs]
        support_group_3857 = support_group_3857[
            support_group_3857["geometry"].intersects(anchor_row_3857.geometry)
            | support_group_3857["geometry"].distance(anchor_row_3857.geometry).le(12)
        ].copy()
        if support_group_3857.empty:
            continue

        support_group = supports.loc[support_group_3857.index].copy()
        all_rows = pd.concat([anchors.loc[[anchor_idx]], support_group], ignore_index=False)
        geometry = fill_geometry_holes(unary_union(all_rows["geometry"].tolist()))
        area_sqft = geodesic_area_sqft(geometry)
        if not pd.notna(area_sqft) or area_sqft <= 0:
            area_sqft = float(pd.to_numeric(all_rows["land_gross_sqft"], errors="coerce").fillna(0).sum())
        if not pd.notna(area_sqft) or area_sqft <= 0:
            continue

        current_full_land_value = pd.to_numeric(all_rows["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(all_rows["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(all_rows["improvement_sqft"], errors="coerce").fillna(0).sum()
        merged_unit_count = int(pd.to_numeric(all_rows["merged_unit_count"], errors="coerce").fillna(1).sum())
        merged_parcel_count = int(pd.to_numeric(all_rows["merged_parcel_count"], errors="coerce").fillna(1).sum())
        merged_schedule_count = int(pd.to_numeric(all_rows["merged_schedule_count"], errors="coerce").fillna(1).sum())

        row = anchors.loc[anchor_idx].copy()
        row["geometry"] = geometry
        row["parcel_number"] = collapsed_identifier(all_rows["parcel_number"], merged_parcel_count)
        row["schedule_number"] = collapsed_identifier(all_rows["schedule_number"], merged_schedule_count)
        row["account_number"] = collapsed_identifier(all_rows["account_number"], merged_schedule_count)
        row["owner_name"] = "Multiple Owners" if all_rows["owner_name"].dropna().nunique() > 1 else group_label(all_rows["owner_name"])
        row["situs_address"] = group_label(all_rows["situs_address"]) or row["situs_address"]
        row["situs_city"] = group_label(all_rows["situs_city"]) or row["situs_city"]
        row["subdivision_name"] = subdivision_label(all_rows["subdivision_name"]) or row["subdivision_name"]
        row["classification_description"] = group_label(all_rows["classification_description"]) or row["classification_description"]
        row["source_abstract_description"] = joined_unique(all_rows["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(all_rows["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(all_rows["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(all_rows["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = area_sqft
        row["area_sqft"] = area_sqft
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / area_sqft if area_sqft > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / area_sqft if area_sqft > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / area_sqft if area_sqft > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = merged_unit_count
        row["merged_parcel_count"] = merged_parcel_count
        row["merged_schedule_count"] = merged_schedule_count
        row["merged_source_categories"] = joined_unique(all_rows["property_land_use_category"], limit=12)
        collapsed_rows.append(row.to_dict())
        assigned_supports.update(support_group.index.tolist())
        assigned_anchors.add(anchor_idx)

    if not collapsed_rows:
        return gdf

    anchors_remaining = anchors.drop(index=list(assigned_anchors), errors="ignore")
    supports_remaining = supports.drop(index=list(assigned_supports), errors="ignore")
    combined = pd.concat(
        [other, anchors_remaining, supports_remaining, gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)],
        ignore_index=True,
        sort=False,
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(f"Absorbed {len(assigned_supports):,} support rows into {len(collapsed_rows):,} family anchors")
    return combined.drop(
        columns=["family", "token_set", "area_sqft_num", "merged_schedule_count_num", "owner_text"],
        errors="ignore",
    )


def collapse_exact_family_anchor_clusters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()
    area_series = work["area_sqft"] if "area_sqft" in work.columns else work["land_gross_sqft"]
    work["family"] = work["subdivision_name"].apply(subdivision_family)
    work["area_sqft_num"] = pd.to_numeric(area_series, errors="coerce").fillna(0)

    anchor_mask = (
        work["family"].notna()
        & work["account_type"].fillna("").isin(
            [
                "Merged Condo Ground Parcel",
                "Merged Condo Common Plat",
                "Merged Condo Campus",
                "Merged Named Complex Parent",
            ]
        )
    )
    anchors = work[anchor_mask].copy()
    other = work[~anchor_mask].copy()
    if anchors.empty:
        return gdf

    anchors_3857 = anchors.to_crs(3857)
    assigned: set[int] = set()
    collapsed_rows: list[dict] = []

    for family, family_group in anchors.groupby("family", dropna=False):
        if not family or len(family_group) < 2:
            continue
        idxs = family_group.index.tolist()
        adjacency: dict[int, set[int]] = {idx: set() for idx in idxs}
        for i, idx1 in enumerate(idxs):
            geom1 = anchors_3857.at[idx1, "geometry"]
            for idx2 in idxs[i + 1 :]:
                geom2 = anchors_3857.at[idx2, "geometry"]
                if geom1.intersects(geom2) or geom1.distance(geom2) <= 12:
                    adjacency[idx1].add(idx2)
                    adjacency[idx2].add(idx1)

        for component in connected_components(idxs, adjacency):
            remaining = [idx for idx in component if idx not in assigned]
            if len(remaining) < 2:
                continue
            group = anchors.loc[remaining].copy()
            geometry = fill_geometry_holes(unary_union(group["geometry"].tolist()))
            area_sqft = geodesic_area_sqft(geometry)
            if not pd.notna(area_sqft) or area_sqft <= 0:
                area_sqft = float(pd.to_numeric(group["land_gross_sqft"], errors="coerce").fillna(0).sum())
            if not pd.notna(area_sqft) or area_sqft <= 0:
                continue

            current_full_land_value = pd.to_numeric(group["current_full_land_value"], errors="coerce").fillna(0).sum()
            improvement_value = pd.to_numeric(group["improvement_value"], errors="coerce").fillna(0).sum()
            full_market_value = current_full_land_value + improvement_value
            improvement_sqft = pd.to_numeric(group["improvement_sqft"], errors="coerce").fillna(0).sum()
            merged_unit_count = int(pd.to_numeric(group["merged_unit_count"], errors="coerce").fillna(1).sum())
            merged_parcel_count = int(pd.to_numeric(group["merged_parcel_count"], errors="coerce").fillna(1).sum())
            merged_schedule_count = int(pd.to_numeric(group["merged_schedule_count"], errors="coerce").fillna(1).sum())

            row = group.iloc[0].copy()
            row["geometry"] = geometry
            row["parcel_number"] = collapsed_identifier(group["parcel_number"], merged_parcel_count)
            row["schedule_number"] = collapsed_identifier(group["schedule_number"], merged_schedule_count)
            row["account_number"] = collapsed_identifier(group["account_number"], merged_schedule_count)
            row["owner_name"] = "Multiple Owners" if group["owner_name"].dropna().nunique() > 1 else group_label(group["owner_name"])
            row["situs_address"] = group_label(group["situs_address"]) or row["situs_address"]
            row["situs_city"] = group_label(group["situs_city"]) or row["situs_city"]
            row["subdivision_name"] = subdivision_label(group["subdivision_name"]) or row["subdivision_name"]
            row["classification_description"] = group_label(group["classification_description"]) or row["classification_description"]
            row["source_abstract_description"] = joined_unique(group["source_abstract_description"])
            row["source_occupancy_description"] = joined_unique(group["source_occupancy_description"])
            row["building_count"] = pd.to_numeric(group["building_count"], errors="coerce").fillna(0).sum()
            year_built = pd.to_numeric(group["year_built"], errors="coerce")
            row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
            row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
            row["land_gross_sqft"] = area_sqft
            row["area_sqft"] = area_sqft
            row["current_full_land_value"] = current_full_land_value
            row["improvement_value"] = improvement_value
            row["full_market_value"] = full_market_value
            row["full_market_value_per_sqft"] = full_market_value / area_sqft if area_sqft > 0 else np.nan
            row["land_value_per_sqft"] = current_full_land_value / area_sqft if area_sqft > 0 else np.nan
            row["improvement_value_per_sqft"] = improvement_value / area_sqft if area_sqft > 0 else np.nan
            row["link"] = np.nan
            row["merged_unit_count"] = merged_unit_count
            row["merged_parcel_count"] = merged_parcel_count
            row["merged_schedule_count"] = merged_schedule_count
            row["merged_source_categories"] = joined_unique(group["property_land_use_category"], limit=12)
            collapsed_rows.append(row.to_dict())
            assigned.update(remaining)

    if not collapsed_rows:
        return gdf

    anchors_remaining = anchors.drop(index=list(assigned), errors="ignore")
    combined = pd.concat(
        [other, anchors_remaining, gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)],
        ignore_index=True,
        sort=False,
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(f"Collapsed {len(assigned):,} anchor rows into {len(collapsed_rows):,} exact-family clusters")
    return combined.drop(columns=["family", "area_sqft_num"], errors="ignore")


def collapse_condos_to_ground_parents(
    gdf: gpd.GeoDataFrame,
    raw_gdf: gpd.GeoDataFrame,
    lots_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    condo_mask = gdf["property_land_use_category"].fillna("").str.contains("condo|town", case=False, na=False)
    condo = gdf[condo_mask].copy()
    other = gdf[~condo_mask].copy()
    if condo.empty:
        return gdf

    condo["condo_family"] = condo["subdivision_name"].apply(subdivision_family)
    condo = condo[condo["condo_family"].notna()].copy()
    if condo.empty:
        return gdf

    raw_parent, common_lots = build_condo_ground_parent_candidates(raw_gdf, lots_gdf)
    if raw_parent.empty and common_lots.empty:
        return gdf

    condo_3857 = condo[
        [
            "geometry",
            "condo_family",
            "parcel_number",
            "merged_schedule_count",
            "merged_parcel_count",
            "merged_unit_count",
            "land_gross_sqft",
        ]
    ].copy().to_crs(3857)
    condo_3857["geometry"] = condo_3857["geometry"].buffer(CONDO_COMMON_AREA_TOLERANCE_METERS)
    condo_3857["land_gross_sqft_num"] = pd.to_numeric(condo["land_gross_sqft"], errors="coerce").fillna(0).values
    condo_3857["merged_schedule_weight"] = pd.to_numeric(condo["merged_schedule_count"], errors="coerce").fillna(1).values

    candidate_groups: list[dict[str, object]] = []

    if not raw_parent.empty:
        raw_3857 = raw_parent[["parent_key", "geometry"]].copy().to_crs(3857)
        raw_3857["parent_area_sqft"] = raw_3857["geometry"].area * 10.76391041671
        raw_join = gpd.sjoin(
            condo_3857[["geometry", "condo_family", "land_gross_sqft_num", "merged_schedule_weight"]],
            raw_3857[["parent_key", "geometry", "parent_area_sqft"]],
            how="inner",
            predicate="intersects",
        )
        if not raw_join.empty:
            raw_join = raw_join[
                raw_join["parent_area_sqft"] >= raw_join["land_gross_sqft_num"].clip(lower=1) * CONDO_PARENT_MIN_RATIO
            ].copy()
            for (parent_key, family_key), group in raw_join.groupby(["parent_key", "condo_family"], dropna=False):
                condo_indexes = sorted(group.index.unique().tolist())
                total_schedule_weight = int(pd.to_numeric(group["merged_schedule_weight"], errors="coerce").fillna(1).sum())
                if total_schedule_weight < 2:
                    continue
                candidate_groups.append(
                    {
                        "source_priority": 0,
                        "parent_source": "raw",
                        "parent_key": str(parent_key),
                        "condo_family": family_key,
                        "condo_indexes": condo_indexes,
                        "schedule_weight": total_schedule_weight,
                        "parent_area_sqft": float(group["parent_area_sqft"].iloc[0]),
                    }
                )

    if not common_lots.empty:
        lot_3857 = common_lots[["parent_key", "parent_family", "label_norm", "geometry"]].copy().to_crs(3857)
        lot_3857["parent_area_sqft"] = lot_3857["geometry"].area * 10.76391041671
        lot_join = gpd.sjoin(
            condo_3857[["geometry", "condo_family", "land_gross_sqft_num", "merged_schedule_weight"]],
            lot_3857[["parent_key", "parent_family", "label_norm", "geometry", "parent_area_sqft"]],
            how="inner",
            predicate="intersects",
        )
        if not lot_join.empty:
            lot_join = lot_join[
                lot_join["parent_area_sqft"] >= lot_join["land_gross_sqft_num"].clip(lower=1) * CONDO_PARENT_MIN_RATIO
            ].copy()
            lot_join = lot_join[
                lot_join["parent_family"].isna() | lot_join["parent_family"].eq(lot_join["condo_family"])
            ].copy()
            for (parent_key, family_key), group in lot_join.groupby(["parent_key", "condo_family"], dropna=False):
                condo_indexes = sorted(group.index.unique().tolist())
                total_schedule_weight = int(pd.to_numeric(group["merged_schedule_weight"], errors="coerce").fillna(1).sum())
                if total_schedule_weight < 2:
                    continue
                candidate_groups.append(
                    {
                        "source_priority": 1,
                        "parent_source": "plat_common",
                        "parent_key": str(parent_key),
                        "condo_family": family_key,
                        "condo_indexes": condo_indexes,
                        "schedule_weight": total_schedule_weight,
                        "parent_area_sqft": float(group["parent_area_sqft"].iloc[0]),
                    }
                )

    if not candidate_groups:
        return gdf

    raw_lookup = raw_parent.set_index("parent_key") if not raw_parent.empty else None
    raw_parent_3857 = raw_parent.to_crs(3857) if not raw_parent.empty else None
    lot_lookup = common_lots.set_index("parent_key") if not common_lots.empty else None

    assigned_condo: set[int] = set()
    absorbed_other_parcels: set[str] = set()
    collapsed_rows: list[dict] = []

    for candidate in sorted(
        candidate_groups,
        key=lambda item: (
            item["source_priority"],
            -int(item.get("schedule_weight", len(item["condo_indexes"]))),
            -float(item["parent_area_sqft"]),
        ),
    ):
        remaining = [idx for idx in candidate["condo_indexes"] if idx not in assigned_condo]
        remaining_schedule_weight = int(
            pd.to_numeric(condo.loc[remaining, "merged_schedule_count"], errors="coerce").fillna(1).sum()
        )
        if remaining_schedule_weight < 2:
            continue

        group = condo.loc[remaining].copy()
        if candidate["parent_source"] == "raw":
            if raw_lookup is None or raw_parent_3857 is None or candidate["parent_key"] not in raw_lookup.index:
                continue
            group_geom_3857 = unary_union(condo_3857.loc[remaining, "geometry"].tolist())
            raw_matches_3857 = raw_parent_3857[raw_parent_3857["geometry"].intersects(group_geom_3857)].copy()
            if raw_matches_3857.empty:
                raw_matches = raw_parent[raw_parent["parent_key"].eq(candidate["parent_key"])].copy()
            else:
                raw_matches = raw_parent.loc[raw_matches_3857.index].copy()
                max_child_area = float(
                    pd.to_numeric(group["land_gross_sqft"], errors="coerce").fillna(0).clip(lower=1).max()
                )
                raw_matches["parent_area_sqft"] = raw_matches["geometry"].apply(geodesic_area_sqft)
                raw_matches = raw_matches[
                    raw_matches["parent_area_sqft"] >= max_child_area * CONDO_PARENT_MIN_RATIO
                ].copy()
                if raw_matches.empty:
                    continue
            raw_matches = raw_matches.sort_values(
                ["is_assoc_parent", "parent_area_rank"],
                ascending=[False, False],
            )
            parent_row = raw_matches.iloc[0]
            parent_geometry = unary_union(raw_matches["geometry"].tolist())
            parent_parcel_number = collapsed_identifier(raw_matches["PARCELNUM"], len(raw_matches))
            absorbed_other_parcels.update(
                {
                    str(parcel).strip()
                    for parcel in raw_matches["PARCELNUM"].dropna().astype(str)
                    if str(parcel).strip()
                }
            )
        else:
            if lot_lookup is None or candidate["parent_key"] not in lot_lookup.index:
                continue
            parent_row = lot_lookup.loc[candidate["parent_key"]]
            parent_geometry = parent_row["geometry"]
            parent_parcel_number = None

        parent_area_sqft = geodesic_area_sqft(parent_geometry)
        if not pd.notna(parent_area_sqft) or parent_area_sqft <= 0:
            parent_area_sqft = float(candidate["parent_area_sqft"])
        if not pd.notna(parent_area_sqft) or parent_area_sqft <= 0:
            continue

        merged_geometry = unary_union([parent_geometry] + group["geometry"].tolist())
        if merged_geometry is not None and not merged_geometry.is_empty and not merged_geometry.is_valid:
            merged_geometry = merged_geometry.buffer(0)
        merged_geometry = fill_geometry_holes(merged_geometry)

        current_full_land_value = pd.to_numeric(group["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(group["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(group["improvement_sqft"], errors="coerce").fillna(0).sum()
        merged_unit_count = int(pd.to_numeric(group["merged_unit_count"], errors="coerce").fillna(1).sum())
        merged_parcel_count = int(pd.to_numeric(group["merged_parcel_count"], errors="coerce").fillna(1).sum())
        merged_schedule_count = int(pd.to_numeric(group["merged_schedule_count"], errors="coerce").fillna(1).sum())

        row = group.iloc[0].copy()
        row["geometry"] = merged_geometry
        row["parcel_number"] = parent_parcel_number or collapsed_identifier(group["parcel_number"], merged_parcel_count)
        row["schedule_number"] = collapsed_identifier(group["schedule_number"], merged_schedule_count)
        row["account_number"] = collapsed_identifier(group["account_number"], merged_schedule_count)
        row["owner_name"] = "Multiple Owners" if group["owner_name"].dropna().nunique() > 1 else group_label(group["owner_name"])
        row["situs_address"] = group_label(group["situs_address"]) or row["situs_address"]
        row["situs_city"] = group_label(group["situs_city"]) or row["situs_city"]
        row["subdivision_name"] = subdivision_label(group["subdivision_name"]) or row["subdivision_name"]
        row["account_type"] = "Merged Condo Ground Parcel" if candidate["parent_source"] == "raw" else "Merged Condo Common Plat"
        row["classification_description"] = group_label(group["classification_description"]) or row["classification_description"]
        row["source_abstract_description"] = joined_unique(group["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(group["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(group["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(group["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = parent_area_sqft
        row["area_sqft"] = parent_area_sqft
        row["property_land_use_category"] = merged_condo_category(group["property_land_use_category"])
        row["property_land_use_refined"] = np.nan
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / parent_area_sqft if parent_area_sqft > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / parent_area_sqft if parent_area_sqft > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / parent_area_sqft if parent_area_sqft > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = merged_unit_count
        row["merged_parcel_count"] = merged_parcel_count
        row["merged_schedule_count"] = merged_schedule_count
        row["merged_source_categories"] = joined_unique(group["property_land_use_category"], limit=12)
        collapsed_rows.append(row.to_dict())
        assigned_condo.update(remaining)

    if not collapsed_rows:
        return gdf

    condo_remaining = condo.drop(index=list(assigned_condo), errors="ignore")
    if absorbed_other_parcels:
        other = other[
            ~other["parcel_number"].fillna("").astype(str).isin(absorbed_other_parcels)
        ].copy()
    combined = pd.concat(
        [other, condo_remaining, gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)],
        ignore_index=True,
        sort=False,
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(
        f"Collapsed {len(assigned_condo):,} condo rows into {len(collapsed_rows):,} ground-parent groups"
    )
    return combined.drop(columns=["condo_family"], errors="ignore")


def collapse_condos_to_platted_lots(
    gdf: gpd.GeoDataFrame,
    lots_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    condo_mask = (
        gdf["property_land_use_category"].fillna("").str.contains("condo|town", case=False, na=False)
        & gdf["exemption_flag"].fillna(0).eq(0)
    )
    condo = gdf[condo_mask].copy()
    other = gdf[~condo_mask].copy()
    if condo.empty or lots_gdf.empty:
        return gdf

    condo["condo_family"] = condo["subdivision_name"].apply(subdivision_family)
    condo = condo[condo["condo_family"].notna()].copy()
    if condo.empty:
        return gdf

    lots = lots_gdf.copy()
    lots["geometry"] = lots["geometry"].apply(
        lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0)
    )
    lots["lot_family"] = lots["SUBNAME"].apply(subdivision_family)
    lots = lots[lots["lot_family"].notna()].copy()
    if lots.empty:
        return gdf

    condo_3857 = condo[["geometry", "condo_family"]].copy().to_crs(3857)
    condo_3857["geometry"] = condo_3857["geometry"].buffer(CONDO_COMMON_AREA_TOLERANCE_METERS)
    lots_3857 = lots[["OBJECTID", "LABEL", "SUBNUM", "SUBNAME", "lot_family", "geometry"]].copy().to_crs(3857)

    joined = gpd.sjoin(
        condo_3857,
        lots_3857,
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return gdf

    lots_lookup = lots.set_index("OBJECTID")
    lots_lookup_3857 = lots_3857.set_index("OBJECTID")
    collapsed_rows: list[dict] = []
    assigned_condo: set[int] = set()

    for family_key, family_group in sorted(
        condo.groupby("condo_family", dropna=False),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        if not family_key:
            continue
        remaining_condo = [idx for idx in family_group.index if idx not in assigned_condo]
        if len(remaining_condo) < 2:
            continue

        family_join = joined[
            joined.index.isin(remaining_condo) & joined["lot_family"].eq(family_key)
        ].copy()
        if family_join.empty:
            continue

        family_lot_ids = sorted(family_join["OBJECTID"].dropna().astype(int).unique().tolist())
        family_lots = lots_lookup.loc[family_lot_ids].copy()
        family_lots_3857 = lots_lookup_3857.loc[family_lot_ids].copy()
        if family_lots.empty or family_lots_3857.empty:
            continue

        component_condo_idx = [idx for idx in remaining_condo if idx in family_join.index.unique()]
        if len(component_condo_idx) < 2:
            continue

        group = condo.loc[component_condo_idx].copy()
        cleaned_lot_geoms = [
            geom for geom in family_lots["geometry"].tolist()
            if geom is not None and not geom.is_empty
        ]
        geometry = unary_union(cleaned_lot_geoms)
        if geometry is not None and not geometry.is_empty and not geometry.is_valid:
            geometry = geometry.buffer(0)
        area_sqft = geodesic_area_sqft(geometry)
        if not pd.notna(area_sqft) or area_sqft <= 0:
            area_sqft = float(family_lots_3857["geometry"].area.sum() * 10.76391041671)

        current_full_land_value = pd.to_numeric(group["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(group["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(group["improvement_sqft"], errors="coerce").fillna(0).sum()
        merged_unit_count = int(pd.to_numeric(group["merged_unit_count"], errors="coerce").fillna(1).sum())
        merged_schedule_count = int(pd.to_numeric(group["merged_schedule_count"], errors="coerce").fillna(1).sum())
        merged_parcel_count = int(pd.to_numeric(group["merged_parcel_count"], errors="coerce").fillna(1).sum())

        row = group.iloc[0].copy()
        row["geometry"] = geometry
        row["parcel_number"] = collapsed_identifier(group["parcel_number"], merged_parcel_count)
        row["schedule_number"] = collapsed_identifier(group["schedule_number"], merged_schedule_count)
        row["account_number"] = collapsed_identifier(group["account_number"], merged_schedule_count)
        row["owner_name"] = "Multiple Owners" if group["owner_name"].dropna().nunique() > 1 else group_label(group["owner_name"])
        row["situs_address"] = group_label(group["situs_address"]) or row["situs_address"]
        row["situs_city"] = group_label(group["situs_city"]) or row["situs_city"]
        row["subdivision_name"] = subdivision_label(group["subdivision_name"]) or group_label(family_lots["SUBNAME"]) or row["subdivision_name"]
        row["account_type"] = "Merged Condo Platted Lot"
        row["classification_description"] = group_label(group["classification_description"]) or row["classification_description"]
        row["source_abstract_description"] = joined_unique(group["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(group["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(group["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(group["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = area_sqft
        row["area_sqft"] = area_sqft
        row["property_land_use_category"] = merged_condo_category(group["property_land_use_category"])
        row["property_land_use_refined"] = np.nan
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = merged_unit_count
        row["merged_parcel_count"] = merged_parcel_count
        row["merged_schedule_count"] = merged_schedule_count
        row["merged_source_categories"] = joined_unique(group["property_land_use_category"], limit=12)
        collapsed_rows.append(row.to_dict())
        assigned_condo.update(component_condo_idx)

    if not collapsed_rows:
        return gdf

    condo_remaining = condo.drop(index=list(assigned_condo), errors="ignore")
    combined = pd.concat(
        [other, condo_remaining, gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)],
        ignore_index=True,
        sort=False,
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(
        f"Collapsed {len(assigned_condo):,} condo rows into {len(collapsed_rows):,} platted-lot condo groups"
    )
    return combined.drop(columns=["condo_family"], errors="ignore")


def collapse_tax_parcels_to_platted_lots(
    gdf: gpd.GeoDataFrame,
    lots_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    if gdf.empty or lots_gdf.empty:
        return gdf, {
            "assigned_tax_parcels": float(len(gdf)),
            "platted_parcels_with_tax_data": 0.0,
            "platted_parcels_with_multi_tax_parcels": 0.0,
            "pct_multi_tax_parcel_platted": 0.0,
        }

    work = gdf.copy()
    lots = lots_gdf.copy()
    lots["geometry"] = lots["geometry"].apply(
        lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0)
    )
    lots = lots[lots["geometry"].notna() & ~lots["geometry"].is_empty].copy()
    if lots.empty:
        return gdf, {
            "assigned_tax_parcels": float(len(gdf)),
            "platted_parcels_with_tax_data": 0.0,
            "platted_parcels_with_multi_tax_parcels": 0.0,
            "pct_multi_tax_parcel_platted": 0.0,
        }

    work["source_row_id"] = work.index.astype(int)
    work_3857 = work[["source_row_id", "geometry"]].copy().to_crs(3857)
    lots_3857 = lots[["OBJECTID", "LABEL", "SUBNUM", "SUBNAME", "geometry"]].copy().to_crs(3857)

    joined = gpd.sjoin(
        work_3857,
        lots_3857,
        how="inner",
        predicate="intersects",
    )

    assignments_parts: list[pd.DataFrame] = []
    if not joined.empty:
        right_geoms = lots_3857["geometry"]
        joined = joined.reset_index().rename(columns={"index": "left_index"})
        joined["intersection_area_m2"] = [
            geom.intersection(right_geoms.loc[idx_right]).area
            if geom is not None and idx_right in right_geoms.index
            else 0.0
            for geom, idx_right in zip(joined["geometry"], joined["index_right"])
        ]
        joined = joined.sort_values(
            ["source_row_id", "intersection_area_m2", "OBJECTID"],
            ascending=[True, False, True],
        )
        best = joined.drop_duplicates(subset=["source_row_id"], keep="first").copy()
        assignments_parts.append(best[["source_row_id", "OBJECTID"]])

    assigned_ids = (
        set(assignments_parts[0]["source_row_id"].tolist())
        if assignments_parts
        else set()
    )
    unmatched = work[~work["source_row_id"].isin(assigned_ids)].copy()
    if not unmatched.empty:
        unmatched_points = unmatched[["source_row_id", "geometry"]].copy()
        unmatched_points["geometry"] = unmatched_points["geometry"].representative_point()
        unmatched_points = unmatched_points.to_crs(3857)
        nearest = gpd.sjoin_nearest(
            unmatched_points,
            lots_3857[["OBJECTID", "geometry"]],
            how="left",
            distance_col="nearest_distance_m",
        )
        nearest = nearest.dropna(subset=["OBJECTID"]).copy()
        nearest["OBJECTID"] = nearest["OBJECTID"].astype(int)
        assignments_parts.append(nearest[["source_row_id", "OBJECTID"]])

    if not assignments_parts:
        return gdf, {
            "assigned_tax_parcels": 0.0,
            "platted_parcels_with_tax_data": 0.0,
            "platted_parcels_with_multi_tax_parcels": 0.0,
            "pct_multi_tax_parcel_platted": 0.0,
        }

    assignments = pd.concat(assignments_parts, ignore_index=True).drop_duplicates(subset=["source_row_id"], keep="first")
    assigned = work.merge(assignments, on="source_row_id", how="inner")
    if assigned.empty:
        return gdf, {
            "assigned_tax_parcels": 0.0,
            "platted_parcels_with_tax_data": 0.0,
            "platted_parcels_with_multi_tax_parcels": 0.0,
            "pct_multi_tax_parcel_platted": 0.0,
        }

    lots_lookup = lots.set_index("OBJECTID")
    collapsed_rows: list[dict] = []

    for object_id, group in assigned.groupby("OBJECTID", dropna=False):
        if pd.isna(object_id) or object_id not in lots_lookup.index:
            continue
        lot_row = lots_lookup.loc[object_id]
        geometry = lot_row["geometry"]
        area_sqft = geodesic_area_sqft(geometry)
        if not pd.notna(area_sqft) or area_sqft <= 0:
            continue

        merged_schedule_count = int(pd.to_numeric(group["merged_schedule_count"], errors="coerce").fillna(1).sum())
        merged_parcel_count = int(pd.to_numeric(group["merged_parcel_count"], errors="coerce").fillna(1).sum())
        merged_unit_count = int(pd.to_numeric(group["merged_unit_count"], errors="coerce").fillna(1).sum())
        current_full_land_value = pd.to_numeric(group["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(group["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(group["improvement_sqft"], errors="coerce").fillna(0).sum()

        row = group.iloc[0].copy()
        row["geometry"] = geometry
        row["parcel_number"] = collapsed_identifier(group["parcel_number"], merged_parcel_count)
        row["schedule_number"] = collapsed_identifier(group["schedule_number"], merged_schedule_count)
        row["account_number"] = collapsed_identifier(group["account_number"], merged_schedule_count)
        row["owner_name"] = group_label(group["owner_name"])
        row["situs_address"] = group_label(group["situs_address"])
        row["situs_city"] = most_common_value(group["situs_city"]) or group_label(group["situs_city"])
        row["subdivision_name"] = normalize_text(lot_row.get("SUBNAME")) or subdivision_label(group["subdivision_name"])
        row["account_type"] = (
            "Platted Parcel Merge" if merged_schedule_count > 1 else most_common_value(group["account_type"])
        )
        row["classification_description"] = most_common_value(group["classification_description"]) or group_label(group["classification_description"])
        row["source_abstract_description"] = joined_unique(group["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(group["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(group["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(group["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = area_sqft
        row["area_sqft"] = area_sqft
        row["property_land_use_category"] = most_common_value(group["property_land_use_category"])
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / area_sqft if area_sqft > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / area_sqft if area_sqft > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / area_sqft if area_sqft > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = merged_unit_count
        row["merged_parcel_count"] = merged_parcel_count
        row["merged_schedule_count"] = merged_schedule_count
        row["merged_source_categories"] = joined_unique(group["property_land_use_category"], limit=12)
        row["platted_object_id"] = int(object_id)
        row["platted_label"] = normalize_text(lot_row.get("LABEL"))
        row["platted_subnum"] = normalize_text(lot_row.get("SUBNUM"))
        collapsed_rows.append(row.to_dict())

    collapsed = gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)
    populated_platted = len(collapsed)
    multi_tax = int((collapsed["merged_parcel_count"].fillna(1) > 1).sum()) if populated_platted else 0
    stats = {
        "assigned_tax_parcels": float(len(assigned)),
        "platted_parcels_with_tax_data": float(populated_platted),
        "platted_parcels_with_multi_tax_parcels": float(multi_tax),
        "pct_multi_tax_parcel_platted": (multi_tax / populated_platted * 100.0) if populated_platted else 0.0,
    }
    print(
        "Collapsed "
        f"{len(assigned):,} tax parcels onto {populated_platted:,} platted parcels; "
        f"{multi_tax:,} platted parcels ({stats['pct_multi_tax_parcel_platted']:.2f}%) "
        "have more than one tax parcel assigned"
    )
    return collapsed, stats


def collapse_shared_geometry_condos(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()
    work["geom_key"] = work["geometry"].apply(lambda geom: geom.wkb_hex if geom is not None else None)
    condo_mask = work["property_land_use_category"].fillna("").str.contains("condo|town", case=False, na=False)
    condo = work[condo_mask].copy()
    other = work[~condo_mask].copy()

    duplicate_geom_keys = set(condo["geom_key"].value_counts()[lambda s: s > 1].index)
    if not duplicate_geom_keys:
        return gdf

    condo_single = condo[~condo["geom_key"].isin(duplicate_geom_keys)].copy()
    condo_multi = condo[condo["geom_key"].isin(duplicate_geom_keys)].copy()

    collapsed_rows: list[dict] = []
    for _, group in condo_multi.groupby("geom_key", dropna=False):
        geometry = group["geometry"].iloc[0]
        area_sqft = pd.to_numeric(group["area_sqft"], errors="coerce").dropna()
        merged_area_sqft = float(area_sqft.iloc[0]) if not area_sqft.empty else geodesic_area_sqft(geometry)

        current_full_land_value = pd.to_numeric(group["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(group["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(group["improvement_sqft"], errors="coerce").fillna(0).sum()

        merged_parcel_count = int(group["parcel_number"].dropna().nunique())
        merged_schedule_count = int(group["schedule_number"].dropna().nunique())
        merged_unit_count = int(len(group))

        row = group.iloc[0].copy()
        row["geometry"] = geometry
        row["parcel_number"] = first_plus_more(group["parcel_number"]) or row["parcel_number"]
        row["schedule_number"] = first_plus_more(group["schedule_number"])
        row["account_number"] = first_plus_more(group["account_number"])
        row["owner_name"] = "Multiple Owners" if group["owner_name"].dropna().nunique() > 1 else group_label(group["owner_name"])
        row["situs_address"] = group_label(group["situs_address"]) or row["situs_address"]
        row["situs_city"] = group_label(group["situs_city"]) or row["situs_city"]
        row["subdivision_name"] = subdivision_label(group["subdivision_name"]) or row["subdivision_name"]
        row["account_type"] = "Merged Condo Units"
        row["classification_description"] = group_label(group["classification_description"]) or row["classification_description"]
        row["source_abstract_description"] = joined_unique(group["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(group["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(group["building_count"], errors="coerce").max()
        year_built = pd.to_numeric(group["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = pd.to_numeric(group["land_gross_sqft"], errors="coerce").max()
        row["area_sqft"] = merged_area_sqft
        row["property_land_use_category"] = merged_condo_category(group["property_land_use_category"])
        row["property_land_use_refined"] = np.nan
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / merged_area_sqft if merged_area_sqft and merged_area_sqft > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / merged_area_sqft if merged_area_sqft and merged_area_sqft > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / merged_area_sqft if merged_area_sqft and merged_area_sqft > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = merged_unit_count
        row["merged_parcel_count"] = merged_parcel_count
        row["merged_schedule_count"] = merged_schedule_count
        row["merged_source_categories"] = joined_unique(group["property_land_use_category"], limit=12)
        collapsed_rows.append(row.to_dict())

    collapsed = gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)
    combined = pd.concat([other, condo_single, collapsed], ignore_index=True, sort=False)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(
        f"Collapsed {len(condo_multi):,} condo-like unit rows into {len(collapsed):,} shared-footprint records"
    )
    return combined.drop(columns=["geom_key"], errors="ignore")


def collapse_condo_site_parcels(
    gdf: gpd.GeoDataFrame,
    raw_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    condo_mask = gdf["property_land_use_category"].fillna("").str.contains("condo|town", case=False, na=False)
    condo = gdf[condo_mask].copy()
    other = gdf[~condo_mask].copy()
    if condo.empty:
        return gdf

    raw_site = raw_gdf.copy()
    raw_site["SCHEDNUM"] = raw_site["SCHEDNUM"].fillna("").astype(str).str.strip()
    raw_site = raw_site[raw_site["SCHEDNUM"] == ""].copy()
    if raw_site.empty:
        return gdf

    raw_site["geom_key"] = raw_site["geometry"].apply(lambda geom: geom.wkb_hex if geom is not None else None)
    raw_site = raw_site.drop_duplicates(subset=["PARCELNUM", "geom_key"]).copy()
    raw_site = raw_site[raw_site["PARCELNUM"].notna()].copy()
    if raw_site.empty:
        return gdf
    raw_site["site_area_rank"] = raw_site["geometry"].apply(geodesic_area_sqft)
    raw_site = raw_site.sort_values(["PARCELNUM", "site_area_rank"], ascending=[True, False]).drop_duplicates(
        subset=["PARCELNUM"],
        keep="first",
    )

    condo_3857 = condo.to_crs(3857)
    raw_site_3857 = raw_site[["PARCELNUM", "geometry"]].copy().to_crs(3857)
    joined = gpd.sjoin(
        condo_3857[["geometry"]],
        raw_site_3857,
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return gdf

    candidate_groups: dict[str, pd.Index] = {}
    for parcelnum, idxs in joined.groupby("PARCELNUM").groups.items():
        group = condo.loc[idxs].copy()
        if len(group) < 2:
            continue
        land_gross = pd.to_numeric(group["land_gross_sqft"], errors="coerce").fillna(0)
        if not land_gross.le(1).all():
            continue
        candidate_groups[str(parcelnum)] = pd.Index(group.index)

    if not candidate_groups:
        return gdf

    site_meta = raw_site.set_index(raw_site["PARCELNUM"].astype(str))
    candidate_items = sorted(candidate_groups.items(), key=lambda item: len(item[1]), reverse=True)
    assigned: set[int] = set()
    collapsed_rows: list[dict] = []

    for parcelnum, idxs in candidate_items:
        remaining = [idx for idx in idxs if idx not in assigned]
        if len(remaining) < 2:
            continue
        group = condo.loc[remaining].copy()
        site_geometry = site_meta.loc[parcelnum, "geometry"]
        site_area_sqft = geodesic_area_sqft(site_geometry)

        current_full_land_value = pd.to_numeric(group["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(group["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(group["improvement_sqft"], errors="coerce").fillna(0).sum()
        merged_unit_count = int(pd.to_numeric(group["merged_unit_count"], errors="coerce").fillna(1).sum())
        merged_parcel_count = int(pd.to_numeric(group["merged_parcel_count"], errors="coerce").fillna(1).sum())
        merged_schedule_count = int(pd.to_numeric(group["merged_schedule_count"], errors="coerce").fillna(1).sum())

        row = group.iloc[0].copy()
        row["geometry"] = site_geometry
        row["parcel_number"] = parcelnum
        row["schedule_number"] = collapsed_identifier(group["schedule_number"], merged_schedule_count)
        row["account_number"] = collapsed_identifier(group["account_number"], merged_schedule_count)
        row["owner_name"] = "Multiple Owners" if group["owner_name"].dropna().nunique() > 1 else group_label(group["owner_name"])
        row["situs_address"] = group_label(group["situs_address"]) or row["situs_address"]
        row["situs_city"] = group_label(group["situs_city"]) or row["situs_city"]
        row["subdivision_name"] = subdivision_label(group["subdivision_name"]) or row["subdivision_name"]
        row["account_type"] = "Merged Condo Site"
        row["classification_description"] = group_label(group["classification_description"]) or row["classification_description"]
        row["source_abstract_description"] = joined_unique(group["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(group["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(group["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(group["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = site_area_sqft
        row["area_sqft"] = site_area_sqft
        row["property_land_use_category"] = merged_condo_category(group["property_land_use_category"])
        row["property_land_use_refined"] = np.nan
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / site_area_sqft if site_area_sqft and site_area_sqft > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / site_area_sqft if site_area_sqft and site_area_sqft > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / site_area_sqft if site_area_sqft and site_area_sqft > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = merged_unit_count
        row["merged_parcel_count"] = merged_parcel_count
        row["merged_schedule_count"] = merged_schedule_count
        row["merged_source_categories"] = joined_unique(group["property_land_use_category"], limit=12)
        collapsed_rows.append(row.to_dict())
        assigned.update(remaining)

    if not collapsed_rows:
        return gdf

    condo_remaining = condo.drop(index=list(assigned), errors="ignore")
    collapsed = gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)
    combined = pd.concat([other, condo_remaining, collapsed], ignore_index=True, sort=False)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(
        f"Collapsed {len(assigned):,} condo site/building rows into {len(collapsed):,} shared site parcels"
    )
    return combined


def collapse_condo_parent_parcels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    site = gdf[gdf["account_type"].fillna("").eq("Merged Condo Site")].copy()
    other = gdf[~gdf["account_type"].fillna("").eq("Merged Condo Site")].copy()
    if site.empty:
        return gdf

    parent = other[
        other["property_land_use_category"].fillna("").str.contains("unimp|vacant", case=False, na=False)
        & pd.to_numeric(other["improvement_value"], errors="coerce").fillna(0).le(0)
    ].copy()
    if parent.empty:
        return gdf

    site["subdivision_root"] = site["subdivision_name"].apply(subdivision_root)
    parent["subdivision_root"] = parent["subdivision_name"].apply(subdivision_root)
    parent = parent[parent["subdivision_root"].notna()].copy()
    if parent.empty:
        return gdf

    site_3857 = site.to_crs(3857)
    parent_3857 = parent.to_crs(3857)
    joined = gpd.sjoin(
        site_3857[["subdivision_root", "geometry"]],
        parent_3857[["parcel_number", "subdivision_root", "geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="site",
        rsuffix="parent",
    )
    if joined.empty:
        return gdf

    candidate_groups: dict[str, pd.Index] = {}
    for parcelnum, idxs in joined.groupby("parcel_number").groups.items():
        group = site.loc[idxs].copy()
        parent_root = parent.loc[parent["parcel_number"] == parcelnum, "subdivision_root"].iloc[0]
        if not parent_root:
            continue
        group = group[group["subdivision_root"] == parent_root].copy()
        if len(group) < 2:
            continue
        candidate_groups[str(parcelnum)] = pd.Index(group.index)

    if not candidate_groups:
        return gdf

    candidate_items = sorted(
        candidate_groups.items(),
        key=lambda item: (
            len(item[1]),
            float(parent.loc[parent["parcel_number"] == item[0], "area_sqft"].fillna(0).iloc[0]),
        ),
        reverse=True,
    )

    assigned_sites: set[int] = set()
    assigned_parents: set[int] = set()
    collapsed_rows: list[dict] = []

    for parcelnum, idxs in candidate_items:
        parent_row = parent[parent["parcel_number"] == parcelnum].iloc[0]
        if parent_row.name in assigned_parents:
            continue
        remaining = [idx for idx in idxs if idx not in assigned_sites]
        if len(remaining) < 2:
            continue
        group = site.loc[remaining].copy()
        total_schedule_count = 1 + int(pd.to_numeric(group["merged_schedule_count"], errors="coerce").fillna(1).sum())
        total_parcel_count = 1 + int(pd.to_numeric(group["merged_parcel_count"], errors="coerce").fillna(1).sum())
        total_unit_count = int(pd.to_numeric(group["merged_unit_count"], errors="coerce").fillna(1).sum())
        group_plus_parent = pd.concat([group, parent.loc[[parent_row.name]]], ignore_index=False)

        current_full_land_value = pd.to_numeric(group_plus_parent["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(group_plus_parent["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(group_plus_parent["improvement_sqft"], errors="coerce").fillna(0).sum()

        row = parent_row.copy()
        row["geometry"] = parent_row["geometry"]
        row["parcel_number"] = parent_row["parcel_number"]
        row["schedule_number"] = collapsed_identifier(group_plus_parent["schedule_number"], total_schedule_count)
        row["account_number"] = collapsed_identifier(group_plus_parent["account_number"], total_schedule_count)
        row["owner_name"] = (
            "Multiple Owners"
            if group_plus_parent["owner_name"].dropna().nunique() > 1
            else group_label(group_plus_parent["owner_name"])
        )
        row["situs_address"] = group_label(group["situs_address"]) or parent_row["situs_address"]
        row["situs_city"] = group_label(group_plus_parent["situs_city"]) or parent_row["situs_city"]
        row["subdivision_name"] = subdivision_label(group_plus_parent["subdivision_name"]) or parent_row["subdivision_name"]
        row["account_type"] = "Merged Condo Parent Parcel"
        row["classification_description"] = group_label(group_plus_parent["classification_description"]) or parent_row["classification_description"]
        row["source_abstract_description"] = joined_unique(group_plus_parent["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(group_plus_parent["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(group_plus_parent["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(group_plus_parent["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = parent_row["area_sqft"]
        row["area_sqft"] = parent_row["area_sqft"]
        row["property_land_use_category"] = merged_condo_category(group["property_land_use_category"])
        row["property_land_use_refined"] = np.nan
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / row["area_sqft"] if row["area_sqft"] and row["area_sqft"] > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / row["area_sqft"] if row["area_sqft"] and row["area_sqft"] > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / row["area_sqft"] if row["area_sqft"] and row["area_sqft"] > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = total_unit_count
        row["merged_parcel_count"] = total_parcel_count
        row["merged_schedule_count"] = total_schedule_count
        row["merged_source_categories"] = joined_unique(group_plus_parent["property_land_use_category"], limit=12)
        collapsed_rows.append(row.to_dict())
        assigned_sites.update(remaining)
        assigned_parents.add(parent_row.name)

    if not collapsed_rows:
        return gdf

    site_remaining = site.drop(index=list(assigned_sites), errors="ignore")
    other_remaining = other.drop(index=list(assigned_parents), errors="ignore")
    collapsed = gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)
    combined = pd.concat([other_remaining, site_remaining, collapsed], ignore_index=True, sort=False)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(
        f"Collapsed {len(assigned_sites):,} condo site rows into {len(collapsed):,} scheduled parent parcels"
    )
    return combined


def collapse_condo_address_campuses(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    condo_mask = gdf["property_land_use_category"].fillna("").str.contains("condo|town", case=False, na=False)
    condo = gdf[condo_mask].copy()
    other = gdf[~condo_mask].copy()
    if condo.empty:
        return gdf

    condo["land_gross_sqft_num"] = pd.to_numeric(condo["land_gross_sqft"], errors="coerce").fillna(0)
    condo["subdivision_root"] = condo["subdivision_name"].apply(subdivision_root)
    condo["base_address"] = condo["situs_address"].apply(normalized_base_address)
    remainder = condo[
        ~condo["account_type"].fillna("").isin(["Merged Condo Site", "Merged Condo Parent Parcel"])
        & condo["land_gross_sqft_num"].le(1)
        & condo["subdivision_root"].notna()
        & condo["base_address"].notna()
    ].copy()
    if remainder.empty:
        return gdf

    candidate_groups: dict[tuple[str, str], pd.Index] = {}
    for key, group in remainder.groupby(["base_address", "subdivision_root"], dropna=False):
        if len(group) < 4:
            continue
        candidate_groups[key] = pd.Index(group.index)

    if not candidate_groups:
        return gdf

    candidate_items = sorted(candidate_groups.items(), key=lambda item: len(item[1]), reverse=True)
    assigned: set[int] = set()
    collapsed_rows: list[dict] = []

    for (_, _), idxs in candidate_items:
        remaining = [idx for idx in idxs if idx not in assigned]
        if len(remaining) < 4:
            continue
        group = remainder.loc[remaining].copy()
        geometry = unary_union(group["geometry"].tolist())
        area_sqft = geodesic_area_sqft(geometry)

        current_full_land_value = pd.to_numeric(group["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(group["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(group["improvement_sqft"], errors="coerce").fillna(0).sum()
        merged_unit_count = int(pd.to_numeric(group["merged_unit_count"], errors="coerce").fillna(1).sum())
        merged_parcel_count = int(pd.to_numeric(group["merged_parcel_count"], errors="coerce").fillna(1).sum())
        merged_schedule_count = int(pd.to_numeric(group["merged_schedule_count"], errors="coerce").fillna(1).sum())

        row = group.iloc[0].copy()
        row["geometry"] = geometry
        row["parcel_number"] = collapsed_identifier(group["parcel_number"], merged_parcel_count)
        row["schedule_number"] = collapsed_identifier(group["schedule_number"], merged_schedule_count)
        row["account_number"] = collapsed_identifier(group["account_number"], merged_schedule_count)
        row["owner_name"] = "Multiple Owners" if group["owner_name"].dropna().nunique() > 1 else group_label(group["owner_name"])
        row["situs_address"] = group_label(group["base_address"]) or group_label(group["situs_address"]) or row["situs_address"]
        row["situs_city"] = group_label(group["situs_city"]) or row["situs_city"]
        row["subdivision_name"] = subdivision_label(group["subdivision_name"]) or row["subdivision_name"]
        row["account_type"] = "Merged Condo Campus"
        row["classification_description"] = group_label(group["classification_description"]) or row["classification_description"]
        row["source_abstract_description"] = joined_unique(group["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(group["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(group["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(group["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = area_sqft
        row["area_sqft"] = area_sqft
        row["property_land_use_category"] = merged_condo_category(group["property_land_use_category"])
        row["property_land_use_refined"] = np.nan
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = merged_unit_count
        row["merged_parcel_count"] = merged_parcel_count
        row["merged_schedule_count"] = merged_schedule_count
        row["merged_source_categories"] = joined_unique(group["property_land_use_category"], limit=12)
        collapsed_rows.append(row.to_dict())
        assigned.update(remaining)

    if not collapsed_rows:
        return gdf

    condo_remaining = condo.drop(index=list(assigned), errors="ignore").drop(
        columns=["land_gross_sqft_num", "subdivision_root", "base_address"],
        errors="ignore",
    )
    other = other.drop(columns=["land_gross_sqft_num", "subdivision_root", "base_address"], errors="ignore")
    collapsed = gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)
    combined = pd.concat([other, condo_remaining, collapsed], ignore_index=True, sort=False)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(
        f"Collapsed {len(assigned):,} zero-land condo rows into {len(collapsed):,} address-based campus records"
    )
    return combined


def collapse_condo_common_area_parcels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()
    work["subdivision_family"] = work["subdivision_name"].apply(subdivision_family)
    work["area_sqft_num"] = pd.to_numeric(work["area_sqft"], errors="coerce").fillna(0)
    work["improvement_value_num"] = pd.to_numeric(work["improvement_value"], errors="coerce").fillna(0)

    condo_mask = (
        work["exemption_flag"].fillna(0).eq(0)
        & work["property_land_use_category"].fillna("").str.contains("condo|town", case=False, na=False)
        & work["subdivision_family"].notna()
        & pd.to_numeric(work["merged_schedule_count"], errors="coerce").fillna(1).le(1)
    )
    common_area_mask = (
        work["exemption_flag"].fillna(0).eq(1)
        & work["property_land_use_category"].fillna("").str.contains("non taxable", case=False, na=False)
        & work["improvement_value_num"].le(0)
    )

    condo = work[condo_mask].copy()
    common_area = work[common_area_mask].copy()
    other = work[~common_area_mask & ~condo_mask].copy()
    if condo.empty or common_area.empty:
        return gdf

    condo_3857 = condo[["geometry", "subdivision_family", "area_sqft_num"]].copy().to_crs(3857)
    condo_3857["geometry"] = condo_3857["geometry"].buffer(CONDO_COMMON_AREA_TOLERANCE_METERS)
    common_area_3857 = common_area[["geometry", "area_sqft_num"]].copy().to_crs(3857)
    joined = gpd.sjoin(
        condo_3857,
        common_area_3857,
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return gdf

    assigned_condo: set[int] = set()
    assigned_common: set[int] = set()
    collapsed_rows: list[dict] = []

    for family_key, family_group in sorted(
        condo.groupby("subdivision_family", dropna=False),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        if not family_key or len(family_group) < 2:
            continue

        remaining_condo = [idx for idx in family_group.index if idx not in assigned_condo]
        if len(remaining_condo) < 2:
            continue

        family_join = joined.loc[joined.index.isin(remaining_condo)].copy()
        if family_join.empty:
            continue

        condo_group = condo.loc[remaining_condo].copy()
        median_condo_area = float(
            condo_group["area_sqft_num"].replace(0, np.nan).dropna().median()
            if condo_group["area_sqft_num"].replace(0, np.nan).notna().any()
            else 0.0
        )
        min_parent_area = max(10000.0, median_condo_area * 3.0)
        min_touch_count = max(4, min(24, math.ceil(len(condo_group) * 0.15)))

        parent_counts = family_join.groupby("index_right").size().sort_values(ascending=False)
        eligible_parent_ids = [
            idx
            for idx, touch_count in parent_counts.items()
            if idx not in assigned_common
            and touch_count >= min_touch_count
            and float(common_area.at[idx, "area_sqft_num"]) >= min_parent_area
        ]
        if not eligible_parent_ids:
            continue

        parent_group = common_area.loc[eligible_parent_ids].copy()
        all_rows = pd.concat([condo_group, parent_group], ignore_index=False)
        geometry = unary_union(all_rows["geometry"].tolist())
        area_sqft = geodesic_area_sqft(geometry)

        condo_schedule_count = int(pd.to_numeric(condo_group["merged_schedule_count"], errors="coerce").fillna(1).sum())
        condo_parcel_count = int(pd.to_numeric(condo_group["merged_parcel_count"], errors="coerce").fillna(1).sum())
        condo_unit_count = int(pd.to_numeric(condo_group["merged_unit_count"], errors="coerce").fillna(1).sum())
        parent_schedule_count = int(parent_group["schedule_number"].notna().sum())
        parent_parcel_count = int(parent_group["parcel_number"].notna().sum())

        current_full_land_value = pd.to_numeric(all_rows["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(all_rows["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(all_rows["improvement_sqft"], errors="coerce").fillna(0).sum()

        seed = condo_group.iloc[0].copy()
        seed["parcel_number"] = collapsed_identifier(all_rows["parcel_number"], condo_parcel_count + parent_parcel_count)
        seed["schedule_number"] = collapsed_identifier(all_rows["schedule_number"], condo_schedule_count + parent_schedule_count)
        seed["account_number"] = collapsed_identifier(all_rows["account_number"], condo_schedule_count + parent_schedule_count)
        seed["geometry"] = geometry
        seed["owner_name"] = "Multiple Owners" if condo_group["owner_name"].dropna().nunique() > 1 else group_label(condo_group["owner_name"])
        seed["situs_address"] = group_label(condo_group["situs_address"]) or seed["situs_address"]
        seed["situs_city"] = group_label(condo_group["situs_city"]) or seed["situs_city"]
        seed["subdivision_name"] = subdivision_label(condo_group["subdivision_name"]) or seed["subdivision_name"]
        seed["account_type"] = "Merged Condo Common Area"
        seed["classification_description"] = group_label(condo_group["classification_description"]) or seed["classification_description"]
        seed["source_abstract_description"] = joined_unique(all_rows["source_abstract_description"])
        seed["source_occupancy_description"] = joined_unique(all_rows["source_occupancy_description"])
        seed["building_count"] = pd.to_numeric(condo_group["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(condo_group["year_built"], errors="coerce")
        seed["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        seed["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        seed["land_gross_sqft"] = area_sqft
        seed["area_sqft"] = area_sqft
        seed["property_land_use_category"] = merged_condo_category(condo_group["property_land_use_category"])
        seed["property_land_use_refined"] = np.nan
        seed["current_full_land_value"] = current_full_land_value
        seed["improvement_value"] = improvement_value
        seed["full_market_value"] = full_market_value
        seed["full_market_value_per_sqft"] = full_market_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        seed["land_value_per_sqft"] = current_full_land_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        seed["improvement_value_per_sqft"] = improvement_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        seed["exemption_flag"] = 0
        seed["link"] = np.nan
        seed["merged_unit_count"] = condo_unit_count
        seed["merged_parcel_count"] = condo_parcel_count + parent_parcel_count
        seed["merged_schedule_count"] = condo_schedule_count + parent_schedule_count
        seed["merged_source_categories"] = joined_unique(all_rows["property_land_use_category"], limit=12)
        collapsed_rows.append(seed.to_dict())
        assigned_condo.update(condo_group.index.tolist())
        assigned_common.update(parent_group.index.tolist())

    if not collapsed_rows:
        return gdf

    condo_remaining = condo.drop(index=list(assigned_condo), errors="ignore")
    common_remaining = common_area.drop(index=list(assigned_common), errors="ignore")
    combined = pd.concat(
        [other, condo_remaining, common_remaining, gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)],
        ignore_index=True,
        sort=False,
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(
        f"Collapsed {len(assigned_condo):,} condo rows with {len(assigned_common):,} non-taxable common parcels into {len(collapsed_rows):,} condo common-area groups"
    )
    return combined.drop(columns=["subdivision_family", "area_sqft_num", "improvement_value_num"], errors="ignore")


def collapse_condos_data_first(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()
    work["subdivision_family"] = work["subdivision_name"].apply(subdivision_family)
    work["land_gross_sqft_num"] = pd.to_numeric(work["land_gross_sqft"], errors="coerce").fillna(0)
    work["land_value_key"] = pd.to_numeric(work["current_full_land_value"], errors="coerce").fillna(0).round(0)
    work["improvement_value_num"] = pd.to_numeric(work["improvement_value"], errors="coerce").fillna(0)
    work["condo_group_kind"] = work.apply(
        lambda row: condo_group_kind(row.get("property_land_use_category"), row.get("account_type")),
        axis=1,
    )

    condo_mask = (
        work["property_land_use_category"].fillna("").str.contains("condo|town", case=False, na=False)
        & pd.to_numeric(work["merged_schedule_count"], errors="coerce").fillna(1).le(1)
    )
    condo = work[condo_mask].copy()
    other = work[~condo_mask].copy()
    if condo.empty:
        return gdf

    condo = condo[condo["subdivision_family"].notna()].copy()
    if condo.empty:
        return gdf

    parent_candidates = other[
        other["subdivision_family"].notna()
        & other["property_land_use_category"].fillna("").str.contains("unimp|vacant", case=False, na=False)
        & other["improvement_value_num"].le(0)
    ].copy()

    assigned_condo: set[int] = set()
    assigned_parent: set[int] = set()
    collapsed_rows: list[dict] = []

    def build_collapsed_row(group: pd.DataFrame, parent_group: pd.DataFrame) -> dict:
        all_rows = pd.concat([group, parent_group], ignore_index=False)
        geometry = unary_union(all_rows["geometry"].tolist())
        area_sqft = geodesic_area_sqft(geometry)

        condo_schedule_count = int(pd.to_numeric(group["merged_schedule_count"], errors="coerce").fillna(1).sum())
        condo_parcel_count = int(pd.to_numeric(group["merged_parcel_count"], errors="coerce").fillna(1).sum())
        condo_unit_count = int(pd.to_numeric(group["merged_unit_count"], errors="coerce").fillna(1).sum())
        parent_schedule_count = int(parent_group["schedule_number"].notna().sum())
        parent_parcel_count = int(parent_group["parcel_number"].notna().sum())

        current_full_land_value = pd.to_numeric(all_rows["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(all_rows["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(all_rows["improvement_sqft"], errors="coerce").fillna(0).sum()

        seed = parent_group.iloc[0].copy() if not parent_group.empty else group.iloc[0].copy()
        if len(parent_group) == 1:
            seed["parcel_number"] = parent_group["parcel_number"].iloc[0]
        else:
            seed["parcel_number"] = collapsed_identifier(all_rows["parcel_number"], condo_parcel_count + parent_parcel_count)
        seed["schedule_number"] = collapsed_identifier(all_rows["schedule_number"], condo_schedule_count + parent_schedule_count)
        seed["account_number"] = collapsed_identifier(all_rows["account_number"], condo_schedule_count + parent_schedule_count)
        seed["geometry"] = geometry
        seed["owner_name"] = "Multiple Owners" if all_rows["owner_name"].dropna().nunique() > 1 else group_label(all_rows["owner_name"])
        seed["situs_address"] = group_label(group["situs_address"]) or seed["situs_address"]
        seed["situs_city"] = group_label(all_rows["situs_city"]) or seed["situs_city"]
        seed["subdivision_name"] = subdivision_label(all_rows["subdivision_name"]) or seed["subdivision_name"]
        if not parent_group.empty:
            seed["account_type"] = "Merged Condo Parent Parcel"
        elif len(group) >= 4:
            seed["account_type"] = "Merged Condo Campus"
        else:
            seed["account_type"] = "Merged Condo Units"
        seed["classification_description"] = group_label(all_rows["classification_description"]) or seed["classification_description"]
        seed["source_abstract_description"] = joined_unique(all_rows["source_abstract_description"])
        seed["source_occupancy_description"] = joined_unique(all_rows["source_occupancy_description"])
        seed["building_count"] = pd.to_numeric(all_rows["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(all_rows["year_built"], errors="coerce")
        seed["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        seed["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        seed["land_gross_sqft"] = area_sqft
        seed["area_sqft"] = area_sqft
        seed["property_land_use_category"] = merged_condo_category(group["property_land_use_category"])
        seed["property_land_use_refined"] = np.nan
        seed["current_full_land_value"] = current_full_land_value
        seed["improvement_value"] = improvement_value
        seed["full_market_value"] = full_market_value
        seed["full_market_value_per_sqft"] = full_market_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        seed["land_value_per_sqft"] = current_full_land_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        seed["improvement_value_per_sqft"] = improvement_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        seed["link"] = np.nan
        seed["merged_unit_count"] = condo_unit_count
        seed["merged_parcel_count"] = condo_parcel_count + parent_parcel_count
        seed["merged_schedule_count"] = condo_schedule_count + parent_schedule_count
        seed["merged_source_categories"] = joined_unique(all_rows["property_land_use_category"], limit=12)
        return seed.to_dict()

    family_groups = sorted(
        condo.groupby(["subdivision_family"], dropna=False),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for family_key, family_group in family_groups:
        remaining = [idx for idx in family_group.index if idx not in assigned_condo]
        if len(remaining) < 2:
            continue

        group = condo.loc[remaining].copy()
        zero_land_share = float(group["land_gross_sqft_num"].le(1).mean())
        unique_land_values = int(group["land_value_key"].nunique())

        parent_group = parent_candidates[parent_candidates["subdivision_family"] == family_key].copy()
        parent_group = parent_group.loc[[idx for idx in parent_group.index if idx not in assigned_parent]].copy()

        if len(parent_group) > 1:
            parent_group = parent_group.sort_values(
                ["land_gross_sqft_num", "current_full_land_value"],
                ascending=[False, False],
            ).head(1)

        if not parent_group.empty:
            collapsed_rows.append(build_collapsed_row(group, parent_group))
            assigned_condo.update(group.index.tolist())
            assigned_parent.update(parent_group.index.tolist())
            continue

        collapse_whole_family = zero_land_share >= 0.8 and (
            len(group) >= 4 or unique_land_values <= 3
        )
        if collapse_whole_family:
            collapsed_rows.append(build_collapsed_row(group, group.iloc[0:0].copy()))
            assigned_condo.update(group.index.tolist())

    if not collapsed_rows:
        return gdf

    condo_remaining = condo.drop(index=list(assigned_condo), errors="ignore")
    other_remaining = other.drop(index=list(assigned_parent), errors="ignore")
    combined = pd.concat([other_remaining, condo_remaining, gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)], ignore_index=True, sort=False)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(
        f"Collapsed {len(assigned_condo):,} condo rows and {len(assigned_parent):,} parent parcels into {len(collapsed_rows):,} data-first condo groups"
    )
    return combined.drop(
        columns=["subdivision_family", "land_gross_sqft_num", "land_value_key", "improvement_value_num", "condo_group_kind"],
        errors="ignore",
    )


def collapse_condo_complex_parcels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()
    condo_mask = work["property_land_use_category"].fillna("").str.contains("condo|town", case=False, na=False)
    condo = work[condo_mask].copy()
    other = work[~condo_mask].copy()
    if condo.empty:
        return gdf

    condo["subdivision_family"] = condo["subdivision_name"].apply(subdivision_family)
    candidate_groups: list[pd.DataFrame] = []
    for _, group in condo.groupby(["subdivision_family"], dropna=False):
        if len(group) < 2:
            continue
        family = group["subdivision_family"].iloc[0]
        if not family:
            continue
        total_sched = int(pd.to_numeric(group["merged_schedule_count"], errors="coerce").fillna(1).sum())
        if total_sched < 4 and len(group) < 4:
            continue
        candidate_groups.append(group)

    if not candidate_groups:
        return gdf

    assigned: set[int] = set()
    collapsed_rows: list[dict] = []

    for group in sorted(candidate_groups, key=lambda grp: pd.to_numeric(grp["merged_schedule_count"], errors="coerce").fillna(1).sum(), reverse=True):
        remaining = [idx for idx in group.index if idx not in assigned]
        if len(remaining) < 2:
            continue
        subgroup = condo.loc[remaining].copy()
        total_sched = int(pd.to_numeric(subgroup["merged_schedule_count"], errors="coerce").fillna(1).sum())
        total_parcels = int(pd.to_numeric(subgroup["merged_parcel_count"], errors="coerce").fillna(1).sum())
        total_units = int(pd.to_numeric(subgroup["merged_unit_count"], errors="coerce").fillna(1).sum())
        geometry = unary_union(subgroup["geometry"].tolist())
        area_sqft = geodesic_area_sqft(geometry)

        current_full_land_value = pd.to_numeric(subgroup["current_full_land_value"], errors="coerce").fillna(0).sum()
        improvement_value = pd.to_numeric(subgroup["improvement_value"], errors="coerce").fillna(0).sum()
        full_market_value = current_full_land_value + improvement_value
        improvement_sqft = pd.to_numeric(subgroup["improvement_sqft"], errors="coerce").fillna(0).sum()

        row = subgroup.iloc[0].copy()
        row["geometry"] = geometry
        row["parcel_number"] = collapsed_identifier(subgroup["parcel_number"], total_parcels)
        row["schedule_number"] = collapsed_identifier(subgroup["schedule_number"], total_sched)
        row["account_number"] = collapsed_identifier(subgroup["account_number"], total_sched)
        row["owner_name"] = "Multiple Owners" if subgroup["owner_name"].dropna().nunique() > 1 else group_label(subgroup["owner_name"])
        row["situs_address"] = group_label(subgroup["situs_address"]) or row["situs_address"]
        row["situs_city"] = group_label(subgroup["situs_city"]) or row["situs_city"]
        row["subdivision_name"] = subdivision_label(subgroup["subdivision_name"]) or row["subdivision_name"]
        row["account_type"] = "Merged Condo Complex"
        row["classification_description"] = group_label(subgroup["classification_description"]) or row["classification_description"]
        row["source_abstract_description"] = joined_unique(subgroup["source_abstract_description"])
        row["source_occupancy_description"] = joined_unique(subgroup["source_occupancy_description"])
        row["building_count"] = pd.to_numeric(subgroup["building_count"], errors="coerce").fillna(0).sum()
        year_built = pd.to_numeric(subgroup["year_built"], errors="coerce")
        row["year_built"] = year_built[year_built > 0].min() if (year_built > 0).any() else np.nan
        row["improvement_sqft"] = improvement_sqft if improvement_sqft > 0 else np.nan
        row["land_gross_sqft"] = area_sqft
        row["area_sqft"] = area_sqft
        row["property_land_use_category"] = merged_condo_category(subgroup["property_land_use_category"])
        row["property_land_use_refined"] = np.nan
        row["current_full_land_value"] = current_full_land_value
        row["improvement_value"] = improvement_value
        row["full_market_value"] = full_market_value
        row["full_market_value_per_sqft"] = full_market_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        row["land_value_per_sqft"] = current_full_land_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        row["improvement_value_per_sqft"] = improvement_value / area_sqft if area_sqft and area_sqft > 0 else np.nan
        row["link"] = np.nan
        row["merged_unit_count"] = total_units
        row["merged_parcel_count"] = total_parcels
        row["merged_schedule_count"] = total_sched
        row["merged_source_categories"] = joined_unique(subgroup["property_land_use_category"], limit=12)
        collapsed_rows.append(row.to_dict())
        assigned.update(remaining)

    if not collapsed_rows:
        return gdf

    condo_remaining = condo.drop(index=list(assigned), errors="ignore").drop(
        columns=["subdivision_family"],
        errors="ignore",
    )
    other = other.drop(columns=["subdivision_family"], errors="ignore")
    collapsed = gpd.GeoDataFrame(collapsed_rows, geometry="geometry", crs=gdf.crs)
    combined = pd.concat([other, condo_remaining, collapsed], ignore_index=True, sort=False)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    print(
        f"Collapsed {len(assigned):,} condo parcel records into {len(collapsed):,} condo complex records"
    )
    return combined


def original_category(row: pd.Series) -> str:
    for field in ["source_abstract_description", "source_occupancy_description", "account_type"]:
        value = normalize_text(row.get(field))
        if value:
            return value
    return "Other"


def is_exempt(row: pd.Series) -> int:
    account_type = (normalize_text(row.get("account_type")) or "").upper()
    if account_type in EXEMPT_ACCOUNT_TYPES:
        return 1

    desc = " ".join(
        filter(
            None,
            [
                normalize_text(row.get("source_abstract_description")),
                normalize_text(row.get("classification_description")),
                normalize_text(row.get("source_occupancy_description")),
            ],
        )
    ).upper()
    if any(hint in desc for hint in EXEMPT_ABSTRACT_HINTS):
        return 1

    owner = (normalize_text(row.get("owner_name")) or "").upper()
    if any(pattern in owner for pattern in GOVERNMENT_OWNER_PATTERNS):
        return 1

    return 0


def refined_category(row: pd.Series) -> str | None:
    text = " ".join(
        filter(
            None,
            [
                normalize_text(row.get("property_land_use_category")),
                normalize_text(row.get("source_occupancy_description")),
                normalize_text(row.get("account_type")),
            ],
        )
    ).upper()

    if any(hint in text for hint in VACANT_HINTS):
        return "Vacant"

    if any(hint in text for hint in PARKING_HINTS) and not any(exclusion in text for exclusion in NON_PARKING_EXCLUSIONS):
        return "Parking Lot"

    land = float(pd.to_numeric(row.get("current_full_land_value"), errors="coerce") or 0.0)
    improvement = float(pd.to_numeric(row.get("improvement_value"), errors="coerce") or 0.0)
    total = land + improvement
    if total <= 0:
        return None

    if any(hint in text for hint in RESIDENTIAL_HINTS) and improvement > 0:
        return None

    if improvement < 0.5 * total:
        return "Underdeveloped"
    return None


def build_export(raw_gdf: gpd.GeoDataFrame, lots_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = raw_gdf.copy()
    gdf["SCHEDNUM"] = gdf["SCHEDNUM"].astype(str).str.strip()
    gdf = gdf[gdf["SCHEDNUM"].notna() & (gdf["SCHEDNUM"] != "")].copy()
    gdf = collapse_duplicate_schedules(gdf)

    schednums = set(gdf["SCHEDNUM"].astype(str))
    print(f"Joining assessor CSVs for {len(schednums):,} schedules...")

    account_df = load_filtered_csv(ACCOUNT_URL, schednums, ACCOUNT_COLUMNS)
    account_df = account_df[account_df["APPRAISALTYPE"].fillna("").str.upper() == "REAL"].copy()
    account_df = first_latest(account_df, "SCHEDULENUM")
    account_df = account_df.rename(columns={"SCHEDULENUM": "SCHEDNUM", "ACCTTYPE": "ACCOUNT_ACCTTYPE"})

    owner_df = load_filtered_csv(OWNER_URL, schednums, OWNER_COLUMNS)
    owner_df = first_latest(owner_df, "SCHEDULENUM")
    owner_df = owner_df.rename(columns={"SCHEDULENUM": "SCHEDNUM"})
    owner_df["owner_name"] = (
        owner_df[["NAME1", "NAME2"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", np.nan)
    )

    value_df = load_filtered_csv(VALUE_URL, schednums, VALUE_COLUMNS)
    value_df["ACTUALVALUE"] = pd.to_numeric(value_df["ACTUALVALUE"], errors="coerce")
    value_df["LANDSF"] = pd.to_numeric(value_df["LANDSF"], errors="coerce")
    value_df["LANDACRES"] = pd.to_numeric(value_df["LANDACRES"], errors="coerce")

    land_df = (
        value_df[value_df["ABSTRACTTYPE"].fillna("").str.upper() == "L"]
        .groupby("SCHEDULENUM", dropna=False)
        .agg(
            current_full_land_value=("ACTUALVALUE", "sum"),
            value_land_sqft=("LANDSF", "max"),
            value_land_acres=("LANDACRES", "max"),
        )
        .reset_index()
    )
    land_df = land_df.rename(columns={"SCHEDULENUM": "SCHEDNUM"})
    improvement_value_df = (
        value_df[value_df["ABSTRACTTYPE"].fillna("").str.upper() == "I"]
        .groupby("SCHEDULENUM", dropna=False)
        .agg(improvement_value=("ACTUALVALUE", "sum"))
        .reset_index()
    )
    improvement_value_df = improvement_value_df.rename(columns={"SCHEDULENUM": "SCHEDNUM"})
    primary_abstract_df = pick_primary_text(
        value_df, "SCHEDULENUM", "ABSTRACTDESCRIPTION", "ACTUALVALUE"
    ).rename(columns={"SCHEDULENUM": "SCHEDNUM", "ABSTRACTDESCRIPTION": "source_abstract_description"})
    primary_class_df = pick_primary_text(
        value_df, "SCHEDULENUM", "CLASSIFICATIONDESCRIPTION", "ACTUALVALUE"
    ).rename(columns={"SCHEDULENUM": "SCHEDNUM", "CLASSIFICATIONDESCRIPTION": "classification_description"})

    improvement_df = load_filtered_csv(IMPROVEMENT_URL, schednums, IMPROVEMENT_COLUMNS)
    improvement_df["IMPACTUALVALUE"] = pd.to_numeric(improvement_df["IMPACTUALVALUE"], errors="coerce")
    improvement_df["SF"] = pd.to_numeric(improvement_df["SF"], errors="coerce")
    improvement_df["CONDOIMPSF"] = pd.to_numeric(improvement_df["CONDOIMPSF"], errors="coerce")
    improvement_df["BLTASYEARBUILT"] = pd.to_numeric(improvement_df["BLTASYEARBUILT"], errors="coerce")
    improvement_df["improvement_sqft_component"] = improvement_df[["SF", "CONDOIMPSF"]].max(axis=1)

    improvement_sqft_df = (
        improvement_df.groupby("SCHEDULENUM", dropna=False)
        .agg(
            improvement_sqft=("improvement_sqft_component", "sum"),
            year_built=("BLTASYEARBUILT", lambda s: s[s > 0].min() if (s > 0).any() else np.nan),
        )
        .reset_index()
    )
    improvement_sqft_df = improvement_sqft_df.rename(columns={"SCHEDULENUM": "SCHEDNUM"})
    primary_occ_df = pick_primary_text(
        improvement_df, "SCHEDULENUM", "OCCDESCRIPTION", "IMPACTUALVALUE"
    ).rename(columns={"SCHEDULENUM": "SCHEDNUM", "OCCDESCRIPTION": "source_occupancy_description"})

    gdf = gdf.merge(
        account_df[
            [
                "SCHEDNUM",
                "PARCELNO",
                "ACCOUNTNO",
                "ACCOUNT_ACCTTYPE",
                "SITUSADDRESS",
                "SITUSCITY",
                "SUBDIVISIONNAME",
                "BUILDINGCOUNT",
                "LANDGROSSACRES",
                "LANDGROSSSF",
            ]
        ],
        on="SCHEDNUM",
        how="left",
    )
    gdf = gdf.merge(
        owner_df[
            [
                "SCHEDNUM",
                "owner_name",
                "MAILADDRESS1",
                "MAILADDRESS2",
                "MAILCITY",
                "MAILSTATE",
                "MAILZIPCODE",
            ]
        ],
        on="SCHEDNUM",
        how="left",
    )
    for df_part in [
        land_df,
        improvement_value_df,
        primary_abstract_df,
        primary_class_df,
        improvement_sqft_df,
        primary_occ_df,
    ]:
        gdf = gdf.merge(df_part, on="SCHEDNUM", how="left")

    gdf["parcel_number"] = gdf["PARCELNO"].fillna(gdf["PARCELNUM"]).astype(str).str.strip()
    gdf["schedule_number"] = gdf["SCHEDNUM"].astype(str).str.strip()
    gdf["account_number"] = gdf["ACCOUNTNO"].astype(str).replace("nan", "").str.strip()
    gdf["account_type"] = gdf["ACCOUNT_ACCTTYPE"].fillna(gdf["ACCTTYPE"]).astype(str).str.strip().replace("nan", np.nan)
    gdf["situs_address"] = gdf["SITUSADDRESS"].fillna(gdf["LOCADDRESS"]).astype(str).str.strip().replace("nan", np.nan)
    gdf["situs_city"] = gdf["SITUSCITY"].fillna(gdf["LOCCITY"]).astype(str).str.strip().replace("nan", np.nan)
    gdf["subdivision_name"] = gdf["SUBDIVISIONNAME"].astype(str).str.strip().replace("nan", np.nan)
    gdf["land_gross_sqft"] = pd.to_numeric(
        gdf["LANDGROSSSF"].fillna(gdf["value_land_sqft"]),
        errors="coerce",
    )
    gdf["building_count"] = pd.to_numeric(gdf["BUILDINGCOUNT"], errors="coerce")
    gdf["owner_name"] = gdf["owner_name"].fillna(
        gdf[["NAME", "NAME1"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", np.nan)
    )

    gdf["property_land_use_category"] = gdf.apply(original_category, axis=1)
    gdf["exemption_flag"] = gdf.apply(is_exempt, axis=1)

    gdf["current_full_land_value"] = pd.to_numeric(gdf["current_full_land_value"], errors="coerce")
    gdf["improvement_value"] = pd.to_numeric(gdf["improvement_value"], errors="coerce")
    gdf["full_market_value"] = gdf["current_full_land_value"].fillna(0) + gdf["improvement_value"].fillna(0)

    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0))
    print("Computing geodesic parcel areas...")
    gdf["area_sqft"] = gdf["geometry"].apply(geodesic_area_sqft)
    gdf.loc[gdf["area_sqft"] < 1, "area_sqft"] = np.nan

    gdf["merged_unit_count"] = 1
    gdf["merged_parcel_count"] = 1
    gdf["merged_schedule_count"] = 1
    gdf["merged_source_categories"] = np.nan

    before = len(gdf)
    gdf = gdf[gdf["exemption_flag"] == 0].copy()
    print(f"Removed {before - len(gdf):,} exempt parcels -> {len(gdf):,} remaining")

    gdf, platted_stats = collapse_tax_parcels_to_platted_lots(gdf, lots_gdf)
    gdf = collapse_condos_to_ground_parents(gdf, raw_gdf, lots_gdf)
    gdf = collapse_ground_parent_campuses(gdf)
    gdf = collapse_named_parent_complexes(gdf)
    gdf = absorb_remaining_rows_into_named_parents(gdf)
    gdf = absorb_support_rows_into_family_anchors(gdf)
    gdf = collapse_exact_family_anchor_clusters(gdf)

    gdf["land_value_per_sqft"] = gdf["current_full_land_value"] / gdf["area_sqft"]
    gdf["improvement_value_per_sqft"] = gdf["improvement_value"] / gdf["area_sqft"]
    gdf["full_market_value_per_sqft"] = gdf["full_market_value"] / gdf["area_sqft"]

    gdf = add_improvement_ratio_fields(
        gdf,
        land_col="current_full_land_value",
        improvement_col="improvement_value",
    )
    gdf["property_land_use_refined"] = gdf.apply(refined_category, axis=1)
    gdf["link"] = np.where(
        gdf["merged_schedule_count"].fillna(1) > 1,
        np.nan,
        "https://www.larimer.gov/assessor/search#/property/" + gdf["schedule_number"].fillna(""),
    )

    columns_to_export = [
        "geometry",
        "parcel_number",
        "schedule_number",
        "account_number",
        "owner_name",
        "situs_address",
        "situs_city",
        "subdivision_name",
        "account_type",
        "classification_description",
        "source_abstract_description",
        "source_occupancy_description",
        "building_count",
        "year_built",
        "improvement_sqft",
        "land_gross_sqft",
        "merged_unit_count",
        "merged_parcel_count",
        "merged_schedule_count",
        "merged_source_categories",
        "exemption_flag",
        "property_land_use_category",
        "property_land_use_refined",
        "full_market_value",
        "full_market_value_per_sqft",
        "current_full_land_value",
        "land_value_per_sqft",
        "improvement_value",
        "improvement_value_per_sqft",
        "TLLDIMPROV",
        "IMPR_LAND_RATIO",
        "IMPR_LAND_PCT",
        "IMPR_PCT_TOTAL",
        "link",
    ]
    for column in columns_to_export:
        if column not in gdf.columns:
            gdf[column] = np.nan

    export = gdf[columns_to_export].copy()
    export = gpd.GeoDataFrame(export, geometry="geometry", crs="EPSG:4326")
    print("\nProperty category counts:")
    print(export["property_land_use_category"].value_counts(dropna=False).head(20).to_string())
    print("\nRefined category counts:")
    print(export["property_land_use_refined"].value_counts(dropna=False).to_string())
    print(
        "\nPlatted parcel merge stats:\n"
        f"  assigned_tax_parcels={int(platted_stats['assigned_tax_parcels']):,}\n"
        f"  populated_platted_parcels={int(platted_stats['platted_parcels_with_tax_data']):,}\n"
        f"  platted_parcels_with_multi_tax_parcels={int(platted_stats['platted_parcels_with_multi_tax_parcels']):,}\n"
        f"  pct_multi_tax_parcel_platted={platted_stats['pct_multi_tax_parcel_platted']:.2f}%"
    )
    return export


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime("%Y_%m_%d")
    raw_glob = str(OUTPUT_DIR / "fortcollins-co-raw_*.parquet")
    raw_path = OUTPUT_DIR / f"fortcollins-co-raw_{today_str}.parquet"
    lots_glob = str(OUTPUT_DIR / "fortcollins-co-platted-lots_*.parquet")
    lots_path = OUTPUT_DIR / f"fortcollins-co-platted-lots_{today_str}.parquet"

    if args.use_cache:
        cached = latest_cached_raw(raw_glob)
        if cached is None:
            raise FileNotFoundError(f"No cached raw Fort Collins parquet found under {OUTPUT_DIR}")
        print(f"Loading cached raw parquet: {cached}")
        raw_gdf = gpd.read_parquet(cached)
        cached_lots = latest_cached_raw(lots_glob)
        if cached_lots is None:
            print("No cached platted lots parquet found; downloading fresh platted lots...")
            lots_gdf = download_platted_lots(raw_gdf, lots_path)
        else:
            print(f"Loading cached platted lots parquet: {cached_lots}")
            lots_gdf = gpd.read_parquet(cached_lots)
    else:
        raw_gdf = download_raw(raw_path)
        lots_gdf = download_platted_lots(raw_gdf, lots_path)

    export = build_export(raw_gdf, lots_gdf)

    canonical_path = OUTPUT_DIR / "fortcollins-co-parcels.parquet"
    dated_path = OUTPUT_DIR / f"fortcollins-co-parcels_{today_str}.parquet"
    export.to_parquet(canonical_path, index=False)
    export.to_parquet(dated_path, index=False)
    print(f"\nSaved canonical parquet: {canonical_path}")
    print(f"Saved dated parquet:     {dated_path}")
    print(f"Total rows: {len(export):,}")


if __name__ == "__main__":
    main()
