"""Fetch + cache helpers for run_lincoln.py (kept separate so exploration can import them).

Two sources, joined on the 13-digit Lancaster County parcel id:

1. GEOMETRY + county attributes — the City of Lincoln / Lancaster County Assessor's own
   TaxParcels layer: https://gis.lincoln.ne.gov/public/rest/services/Assessor/TaxParcels/FeatureServer/0
   Carries the current total assessed value (CNTASSDVAL), class (CLASSCD), prime use (PRIMEUSE)
   and the condo flags (HighRise / FloorOrder / UnitNumber) — but LNDVALUE is EMPTY on every
   row (0 of 122,699), and so are the taxable-value fields.

2. LAND / IMPROVEMENT split — the State of Nebraska's statewide parcel layer, compiled by the
   OCIO from each county's CAMA export:
   https://giscat.ne.gov/enterprise/rest/services/StatewideParcelsExternal/FeatureServer/0
   County_ID '109' = Lancaster (121,664 rows, CAMAUpdate 2026-01-01). Land_Value /
   Improvements_Value / Other_Impr_Value / Ag_Land_Value / Total_Assessed_Value.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}
COUNTY_URL = ("https://gis.lincoln.ne.gov/public/rest/services/Assessor/TaxParcels/"
              "FeatureServer/0/query")
STATE_URL = ("https://giscat.ne.gov/enterprise/rest/services/StatewideParcelsExternal/"
             "FeatureServer/0/query")
CITY_LIMITS_URL = ("https://gis.lincoln.ne.gov/public/rest/services/CityCounty/CityLimits/"
                   "MapServer/5/query")
LANCASTER = "109"

# Owner names / mailing address are deliberately NOT requested (issue #12).
COUNTY_FIELDS = (
    "OBJECTID,PARCELID,LOWPARCELID,StatedArea,StatedAreaUnit,GIS_AREA,HighRise,FloorOrder,"
    "UnitNumber,NGHBRHDCD,CLASSCD,CLASSDSCRP,SITEADDRESS,PRPRTYDSCRP,CNVYNAME,CVTTXCD,"
    "FLOORCOUNT,BLDGAREA,RESFLRAREA,RESYRBLT,RESSTRTYP,STRCLASS,CNTASSDVAL,PRIMEUSE,"
    "VALMETHOD,GREENBELT"
)
STATE_FIELDS = (
    "OBJECTID,Parcel_ID,State_PID,Acres_Deeded,GIS_Acres,Subdivision,ASSRLOC,"
    "Total_Assessed_Value,Land_Value,Improvements_Value,Other_Impr_Value,Ag_Land_Value,"
    "Classification_Code,Property_Parcel_Status,Ownership_Type,TAXDIST,CAMAUpdate,"
    "Neighborhood,ImpSF,BuildingYear"
)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _get(url, params, timeout=300, method="get"):
    for attempt in range(6):
        try:
            if method == "post":
                r = requests.post(url, data=params, headers=HEADERS, timeout=timeout)
            else:
                r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            body = json.loads(r.content)
            # ArcGIS returns HTTP 200 + {"error": ...} on overload — parsing that as "zero
            # features" would silently truncate the pull.
            if isinstance(body, dict) and "error" in body:
                raise RuntimeError(str(body["error"])[:200])
            return body
        except Exception as e:  # noqa: BLE001
            log(f"  retry {attempt + 1}: {type(e).__name__}: {str(e)[:160]}")
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"request failed: {url} {params.get('resultOffset')}")


def fetch_boundary(cache: Path) -> gpd.GeoDataFrame:
    if not cache.exists():
        r = requests.get(CITY_LIMITS_URL, params={
            "where": "CITY='Lincoln'", "outFields": "CITY,YEAR,ACRES,SQMI",
            "returnGeometry": "true", "outSR": 4326, "f": "geojson"}, headers=HEADERS, timeout=120)
        r.raise_for_status()
        cache.write_bytes(r.content)
    b = gpd.read_file(cache).to_crs("EPSG:4326")
    if not len(b):
        raise RuntimeError("Lincoln city limits came back empty")
    return b


def fetch_county_parcels(cache: Path, bounds, page: int = 2000) -> gpd.GeoDataFrame:
    """County TaxParcels in the city's envelope (geometry + attributes), cached."""
    if cache.exists():
        log(f"Using cached county parcels: {cache.name}")
        return gpd.read_parquet(cache)
    minx, miny, maxx, maxy = bounds
    gp = {"geometry": json.dumps({"xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
                                  "spatialReference": {"wkid": 4326}}),
          "geometryType": "esriGeometryEnvelope", "inSR": 4326,
          "spatialRel": "esriSpatialRelIntersects"}
    total = _get(COUNTY_URL, {**gp, "where": "1=1", "returnCountOnly": "true", "f": "json"})["count"]
    log(f"Pulling {total:,} county parcels in the Lincoln envelope...")
    pages, off = [], 0
    while off < total:
        body = _get(COUNTY_URL, {**gp, "where": "1=1", "outFields": COUNTY_FIELDS,
                                 "returnGeometry": "true", "resultOffset": off,
                                 "resultRecordCount": page, "outSR": 4326,
                                 "orderByFields": "OBJECTID", "f": "geojson"})
        feats = body.get("features", [])
        if not feats:
            break
        pages.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326"))
        off += len(feats)
        if off % 20000 < page:
            log(f"  fetched {off:,}/{total:,}")
    g = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    # The layer is live-edited; offset paging can repeat a row across a page boundary.
    g = g.drop_duplicates("OBJECTID").reset_index(drop=True)
    if len(g) < total * 0.999:
        raise RuntimeError(f"county pull short: {len(g):,} of {total:,}")
    g.to_parquet(cache, index=False)
    log(f"  cached -> {cache.name} ({len(g):,} rows)")
    return g


def fetch_state_values(cache: Path, page: int = 2000) -> pd.DataFrame:
    """Statewide-parcel CAMA attributes for Lancaster County (no geometry), cached."""
    if cache.exists():
        log(f"Using cached state values: {cache.name}")
        return pd.read_parquet(cache)
    where = f"County_ID='{LANCASTER}'"
    total = _get(STATE_URL, {"where": where, "returnCountOnly": "true", "f": "json"})["count"]
    log(f"Pulling {total:,} Lancaster rows from the NE statewide parcel layer (attributes only)...")
    rows, off = [], 0
    while off < total:
        body = _get(STATE_URL, {"where": where, "outFields": STATE_FIELDS,
                                "returnGeometry": "false", "resultOffset": off,
                                "resultRecordCount": page, "orderByFields": "OBJECTID",
                                "f": "json"})
        feats = body.get("features", [])
        if not feats:
            break
        rows.extend(f["attributes"] for f in feats)
        off += len(feats)
        if off % 20000 < page:
            log(f"  fetched {off:,}/{total:,}")
    df = pd.DataFrame(rows)
    if len(df) != total:
        raise RuntimeError(f"state pull short: {len(df):,} of {total:,}")
    df.to_parquet(cache, index=False)
    log(f"  cached -> {cache.name} ({len(df):,} rows)")
    return df


# Institutional owners whose property is exempt in Nebraska but which the Lancaster roll still
# appraises at full value and the open data carries no tax-status flag for (see run_lincoln.py).
# Matched on the owner name AT FETCH TIME; only the matched keyword is cached — the owner name
# itself never touches disk (issue #12).
INSTITUTIONAL_OWNER_KEYWORDS = (
    "BOARD OF REGENTS", "UNIVERSITY", "COLLEGE", "SCHOOL DIST", "PUBLIC SCHOOLS",
    "CITY OF LINCOLN", "LINCOLN CITY OF", "LANCASTER COUNTY", "COUNTY OF LANCASTER",
    "STATE OF NEBRASKA", "NEBRASKA STATE OF", "UNITED STATES", " USA ", "HOUSING AUTHORITY",
    "AIRPORT AUTHORITY", "JOINT PUBLIC AGENCY", "PUBLIC BUILDING COMMISSION",
    "NATURAL RESOURCES DIST", "RAILROAD", "RAILWAY", "BNSF", "UNION PACIFIC",
    "CHURCH", "CONGREGATION", "DIOCESE", "MEDICAL CENTER", "HOSPITAL", "COMM HEALTH",
    "MADONNA", "TABITHA",
)
# A for-profit company can carry one of those words (NEBRASKA HEART HOSPITAL LLC is taxable).
FOR_PROFIT_MARKERS = (" LLC", " L L C", " INC", " LP", " LTD", " CORP")


def institutional_keyword(name) -> str | None:
    if not isinstance(name, str):
        return None
    n = " " + name.upper().strip() + " "
    for kw in INSTITUTIONAL_OWNER_KEYWORDS:
        if kw in n:
            if kw in ("MEDICAL CENTER", "HOSPITAL", "COMM HEALTH", "MADONNA", "TABITHA",
                      "COLLEGE", "UNIVERSITY") and any(f in n for f in FOR_PROFIT_MARKERS):
                return None
            return kw
    return None


def fetch_owner_flags(cache: Path, bounds, page: int = 5000) -> pd.DataFrame:
    """PARCELID -> matched institutional-owner keyword (or None). Names are not cached."""
    if cache.exists():
        log(f"Using cached owner flags: {cache.name}")
        return pd.read_parquet(cache)
    minx, miny, maxx, maxy = bounds
    gp = {"geometry": json.dumps({"xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
                                  "spatialReference": {"wkid": 4326}}),
          "geometryType": "esriGeometryEnvelope", "inSR": 4326,
          "spatialRel": "esriSpatialRelIntersects"}
    rows, off = [], 0
    while True:
        body = _get(COUNTY_URL, {**gp, "where": "1=1", "outFields": "OBJECTID,PARCELID,OWNERNME1",
                                 "returnGeometry": "false", "resultOffset": off,
                                 "resultRecordCount": page, "orderByFields": "OBJECTID", "f": "json"})
        feats = body.get("features", [])
        if not feats:
            break
        for f in feats:
            a = f["attributes"]
            rows.append({"PARCELID": a["PARCELID"],
                         "inst_owner_kw": institutional_keyword(a.get("OWNERNME1"))})
        off += len(feats)
    df = pd.DataFrame(rows).drop_duplicates("PARCELID")
    df.to_parquet(cache, index=False)
    log(f"  cached owner flags -> {cache.name} ({len(df):,} rows, "
        f"{int(df['inst_owner_kw'].notna().sum()):,} institutional)")
    return df
