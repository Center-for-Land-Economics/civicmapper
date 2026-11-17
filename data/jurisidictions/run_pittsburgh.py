#!/usr/bin/env python3
"""
Build the City of Pittsburgh, PA canonical parcel parquet.

Pittsburgh is a city inside Allegheny County. It is not a consolidated city-county
and it is not the same thing as the postal city "PITTSBURGH" (dozens of boroughs
and townships use that mailing city) or as East Pittsburgh borough (MUNICODE 822).

SOURCES
  Geometry (countywide, no dollar values):
    https://services5.arcgis.com/eYH3NsaQKd0zpREV/ArcGIS/rest/services/Allegheny_County_Parcels/FeatureServer/0
    One polygon per PIN. Fields used: PIN, MUNICODE, MAPBLOCKLO, CALCACREAG.
    ~585k parcels countywide. The county's own tax districts for the City of
    Pittsburgh are ward codes 101-132. The municipal-boundary layer uses 100 for
    the city as a whole. East Pittsburgh is 822 and is not this city.

  Values, use, and tax status (countywide assessment extract, no owner names):
    https://data.wprdc.org/dataset/property-assessments
    assessments.csv, joined on PARID = PIN.
    Shipped land and building are FAIRMARKETLAND and FAIRMARKETBUILDING.
    FAIRMARKET* is the 2012 base-year assessed value before any exemption or
    preferential reduction. Homestead, Clean & Green, and abatement are not
    applied to FAIRMARKET* at all. Those reductions exist only on COUNTY*
    (homestead, Clean & Green, and abatement) and LOCAL* (Clean & Green).
    This ETL does not read COUNTY* or LOCAL*. The county help file says the
    base-year figures are not current market values.
    Base year: 2012. https://realestate.alleghenycounty.us/help.html

  City boundary (authoritative clip):
    https://services1.arcgis.com/vdNDkVykv9vEWFX4/arcgis/rest/services/AlleghenyCountyMunicipalBoundaries/FeatureServer/0
    NAME='PITTSBURGH' AND TYPE='CITY' (LABEL Pittsburgh, boundary MUNICODE 100).

CLIP
  The geometry feed is countywide. This script does not trust PROPERTYCITY and
  does not stop at the ward code. It keeps a parcel only when its representative
  point (a point guaranteed to lie inside the polygon) falls inside the official
  city boundary. Ward codes 101-132 are the candidate index; parcels inside the
  boundary with any other MUNICODE are pulled and kept, and ward parcels whose
  representative point falls outside the boundary are dropped. Both disagreements
  are logged. In the November 2026 extract that edge set was 78 parcels from
  Reserve, Swissvale, Penn Hills, Greentree, West Homestead, Wilkinsburg,
  Baldwin, and Rosslyn Farms. East Pittsburgh (822) was not among them.

EXEMPTION
  The county states taxability. TAXCODE T is taxable. TAXCODE E is exempt.
  TAXCODE P is PURTA, and TAXSUBCODE says whether that PURTA parcel is taxable
  (T) or exempt (E). exemption_flag follows those fields only. CLASS is a use
  class, not a tax status, so a CLASS U utility the county marks taxable stays
  in. HOMESTEADFLAG is a partial owner-occupant reduction, not a full exemption,
  and it is not exported.

CONDOS
  Assessments are one row per PARID (no duplicate keys). Condo units are separate
  PINs. Residential condominiums (use 50), condo garage units (57), and commercial
  condo units (550) almost all carry $0 land and $0 lot area on a stub polygon.
  The development's land is the parent lot: MAPBLOCKLO with the unit suffix
  removed (1-C-1662-42 -> 1-C-1662), usually use 118 CONDOMINIUM COMMON PROPERTY
  or a $0 master lot, and the county records $0 land there too. There is no land
  figure to allocate (unlike a roll that puts the land on the master). Units are
  merged down onto that parent footprint and their building values are summed, so
  the map shows the development instead of a stack of pencils. The land value
  stays the assessor's $0. Condos are a few percent of taxable land value, so
  this does not impute a land value (see the Boston note in the add-city skill).

Outputs:
  data/jurisidictions/data/pittsburgh/pittsburgh-pa-parcels.parquet
  data/jurisidictions/data/pittsburgh/pittsburgh-pa-parcels_YYYY_MM_DD.parquet
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import (  # noqa: E402
    add_improvement_ratio_fields,
    check_area_agreement,
    classify_property_refined,
    gis_area_sqft,
)

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "pittsburgh"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "pittsburgh-pa-geometry.parquet"
EXTRA_CACHE = DATA_DIR / "pittsburgh-pa-boundary-extras.parquet"
BOUNDARY_CACHE = DATA_DIR / "pittsburgh-boundary.geojson"
ASSESS_CACHE = DATA_DIR / "allegheny-assessments.csv"

PARCELS_URL = (
    "https://services5.arcgis.com/eYH3NsaQKd0zpREV/ArcGIS/rest/services/"
    "Allegheny_County_Parcels/FeatureServer/0/query"
)
BOUNDARY_URL = (
    "https://services1.arcgis.com/vdNDkVykv9vEWFX4/arcgis/rest/services/"
    "AlleghenyCountyMunicipalBoundaries/FeatureServer/0/query"
)
ASSESS_URL = (
    "https://data.wprdc.org/dataset/2b3df818-601e-4f06-b150-643557229491/resource/"
    "9a1c60bd-f9f7-4aba-aeb7-af8c3aaa44e5/download/assessments.csv"
)
# Ward tax districts of the City of Pittsburgh. Boundary layer uses 100 for the
# city polygon. 822 is East Pittsburgh borough and is intentionally absent.
WARD_WHERE = "MUNICODE >= 101 AND MUNICODE <= 132"
GEOM_FIELDS = "PIN,MUNICODE,MAPBLOCKLO,CALCACREAG"
ASSESS_COLS = [
    "PARID", "PROPERTYHOUSENUM", "PROPERTYFRACTION", "PROPERTYADDRESS", "PROPERTYCITY",
    "PROPERTYUNIT", "PROPERTYZIP", "MUNICODE", "MUNIDESC", "CLASS", "USECODE", "USEDESC",
    "TAXCODE", "TAXDESC", "TAXSUBCODE", "LOTAREA", "HOMESTEADFLAG",
    "FAIRMARKETBUILDING", "FAIRMARKETLAND", "FAIRMARKETTOTAL", "FINISHEDLIVINGAREA",
    "YEARBLT", "TAXYEAR", "ASOFDATE",
]
HEADERS = {"User-Agent": "Mozilla/5.0 CivicMapper-ETL", "Accept": "application/json"}
PAGE = 2000
SQFT_PER_ACRE = 43560.0
UTM = "EPSG:32617"  # Pittsburgh is in UTM zone 17N
VALUE_BASE_YEAR = 2012
PORTAL = "https://realestate.alleghenycounty.us/GeneralInfo?ID="

# Unit records that carry building value on a stub and no land of their own.
UNIT_USES = {"50", "56", "57", "550"}
# Common-area / greenbelt land a unit can fall back onto.
COMMON_USES = {"118", "55", "556"}
STUB_SQFT = 1500.0

LINK_OK_PREFIX = "https://realestate.alleghenycounty.us/"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _polygonal(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [p for p in geom.geoms if p.geom_type in ("Polygon", "MultiPolygon") and not p.is_empty]
        if not polys:
            return None
        return polys[0] if len(polys) == 1 else unary_union(polys)
    return None


def _get_json(url: str, params: dict, timeout: int = 180) -> dict:
    last = None
    for attempt in range(5):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            if "error" in body:
                raise RuntimeError(str(body["error"])[:300])
            return body
        except Exception as exc:  # noqa: BLE001
            last = exc
            log(f"  retry {attempt + 1}: {type(exc).__name__}: {exc}")
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"request failed: {last}")


def fetch_boundary() -> gpd.GeoDataFrame:
    if not BOUNDARY_CACHE.exists():
        body = _get_json(BOUNDARY_URL, {
            "where": "NAME='PITTSBURGH' AND TYPE='CITY'",
            "outFields": "NAME,TYPE,LABEL,MUNICODE,FIPS",
            "returnGeometry": "true",
            "outSR": 4326,
            "f": "geojson",
        })
        BOUNDARY_CACHE.write_text(json.dumps(body))
    boundary = gpd.read_file(BOUNDARY_CACHE).to_crs("EPSG:4326")
    if len(boundary) != 1:
        raise RuntimeError(f"Expected one Pittsburgh city boundary, got {len(boundary)}")
    log(f"City boundary: {boundary.iloc[0]['LABEL']} "
        f"(boundary MUNICODE {boundary.iloc[0]['MUNICODE']}, "
        f"FIPS {boundary.iloc[0]['FIPS']})")
    return boundary


def _pull_where(where: str, label: str) -> gpd.GeoDataFrame:
    total = _get_json(PARCELS_URL, {
        "where": where, "returnCountOnly": "true", "f": "json",
    })["count"]
    log(f"Pulling {total:,} {label} parcels...")
    frames = []
    offset = 0
    while offset < total:
        body = _get_json(PARCELS_URL, {
            "where": where,
            "outFields": GEOM_FIELDS,
            "returnGeometry": "true",
            "resultOffset": offset,
            "resultRecordCount": PAGE,
            "orderByFields": "FID",
            "outSR": 4326,
            "f": "geojson",
        })
        features = body.get("features") or []
        if not features:
            raise RuntimeError(f"{label} page at offset {offset} was empty ({offset}/{total})")
        frames.append(gpd.GeoDataFrame.from_features(features, crs="EPSG:4326"))
        offset += len(features)
        if offset % 20000 < PAGE or offset >= total:
            log(f"  {label}: {offset:,}/{total:,}")
    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    out["PIN"] = out["PIN"].astype(str).str.strip()
    before = len(out)
    out = out.drop_duplicates(subset=["PIN", "geometry"])
    if len(out) != before:
        log(f"  dropped {before - len(out):,} exact duplicate page rows")
    return out


def fetch_geometry(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ward candidates plus any non-ward parcel that intersects the city polygon."""
    if GEOM_CACHE.exists():
        log(f"Using cached ward geometry: {GEOM_CACHE.name}")
        wards = gpd.read_parquet(GEOM_CACHE)
    else:
        wards = _pull_where(WARD_WHERE, "ward 101-132")
        wards.to_parquet(GEOM_CACHE, index=False)
        log(f"Cached {len(wards):,} ward parcels")

    if EXTRA_CACHE.exists():
        log(f"Using cached boundary extras: {EXTRA_CACHE.name}")
        extras = gpd.read_parquet(EXTRA_CACHE)
    else:
        extras = _fetch_boundary_extras(boundary)
        extras.to_parquet(EXTRA_CACHE, index=False)

    if len(extras):
        both = pd.concat([wards, extras], ignore_index=True)
        both = gpd.GeoDataFrame(both, geometry="geometry", crs="EPSG:4326")
    else:
        both = wards
    both["MUNICODE"] = pd.to_numeric(both["MUNICODE"], errors="coerce")
    return both


def _fetch_boundary_extras(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Parcels that intersect the city polygon but are not coded to a city ward."""
    geom = boundary.geometry.iloc[0]
    if geom.geom_type == "MultiPolygon":
        rings = []
        for poly in geom.geoms:
            rings.append([[x, y] for x, y in poly.exterior.coords])
            for hole in poly.interiors:
                rings.append([[x, y] for x, y in hole.coords])
    else:
        rings = [[[x, y] for x, y in geom.exterior.coords]]
        for hole in geom.interiors:
            rings.append([[x, y] for x, y in hole.coords])
    payload = {
        "geometry": json.dumps({"rings": rings, "spatialReference": {"wkid": 4326}}),
        "geometryType": "esriGeometryPolygon",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "where": f"NOT ({WARD_WHERE})",
        "returnCountOnly": "true",
        "f": "json",
    }
    response = requests.post(PARCELS_URL, data=payload, headers=HEADERS, timeout=180)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(f"boundary-extras count failed: {body['error']}")
    total = int(body["count"])
    log(f"Non-ward parcels intersecting the city polygon: {total:,}")
    if total == 0:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if total > 20000:
        raise RuntimeError(
            f"Expected a thin edge set of non-ward parcels, got {total:,}. "
            "Refusing to treat that as the city."
        )
    frames = []
    offset = 0
    while offset < total:
        page = dict(payload)
        page.pop("returnCountOnly")
        page.update({
            "outFields": GEOM_FIELDS,
            "returnGeometry": "true",
            "resultOffset": offset,
            "resultRecordCount": PAGE,
            "orderByFields": "FID",
            "outSR": 4326,
            "f": "geojson",
        })
        last = None
        got = None
        for attempt in range(5):
            try:
                resp = requests.post(PARCELS_URL, data=page, headers=HEADERS, timeout=300)
                resp.raise_for_status()
                parsed = resp.json()
                if "error" in parsed:
                    raise RuntimeError(str(parsed["error"])[:300])
                got = parsed
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                log(f"  extras retry {attempt + 1} @ {offset}: {exc}")
                time.sleep(4 * (attempt + 1))
        if got is None:
            raise RuntimeError(f"extras page failed at {offset}: {last}")
        features = got.get("features") or []
        if not features:
            raise RuntimeError(f"extras page at offset {offset} was empty")
        frames.append(gpd.GeoDataFrame.from_features(features, crs="EPSG:4326"))
        offset += len(features)
        log(f"  extras: {offset:,}/{total:,}")
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


def fetch_assessments() -> pd.DataFrame:
    if not ASSESS_CACHE.exists() or ASSESS_CACHE.stat().st_size < 100_000_000:
        log("Downloading Allegheny County assessments (countywide CSV)...")
        with requests.get(ASSESS_URL, headers=HEADERS, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            tmp = ASSESS_CACHE.with_suffix(".csv.part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
            tmp.replace(ASSESS_CACHE)
        log(f"Saved {ASSESS_CACHE.name} ({ASSESS_CACHE.stat().st_size / 1e6:.0f} MB)")
    else:
        log(f"Using cached assessments: {ASSESS_CACHE.name}")
    frame = pd.read_csv(ASSESS_CACHE, usecols=ASSESS_COLS, dtype=str, low_memory=False)
    frame["PARID"] = frame["PARID"].str.strip()
    for col in ("FAIRMARKETLAND", "FAIRMARKETBUILDING", "FAIRMARKETTOTAL",
                "LOTAREA", "FINISHEDLIVINGAREA", "MUNICODE", "YEARBLT", "TAXYEAR"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    # The extract zero-pads use codes to 3 characters ("050" = condominium).
    frame["USECODE"] = (
        frame["USECODE"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.lstrip("0")
    )
    frame.loc[frame["USECODE"].eq(""), "USECODE"] = "0"
    frame["CLASS"] = frame["CLASS"].fillna("").astype(str).str.strip().str.upper()
    frame["TAXCODE"] = frame["TAXCODE"].fillna("").astype(str).str.strip().str.upper()
    frame["TAXSUBCODE"] = frame["TAXSUBCODE"].fillna("").astype(str).str.strip().str.upper()
    frame["USEDESC"] = frame["USEDESC"].fillna("").astype(str).str.strip()
    log(f"Assessment rows: {len(frame):,}  "
        f"tax year mode {frame['TAXYEAR'].mode().iloc[0] if frame['TAXYEAR'].notna().any() else 'n/a'}  "
        f"as-of {frame['ASOFDATE'].dropna().mode().iloc[0] if frame['ASOFDATE'].notna().any() else 'n/a'}")
    return frame


def clip_to_city(parcels: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    city = unary_union(boundary.geometry)
    parcels = parcels[parcels.geometry.notna() & ~parcels.geometry.is_empty].copy()
    parcels["geometry"] = parcels.geometry.apply(_polygonal)
    parcels = parcels[parcels.geometry.notna()].copy()
    points = gpd.GeoDataFrame(geometry=parcels.geometry.representative_point(), crs=parcels.crs)
    city_frame = gpd.GeoDataFrame(geometry=[city], crs="EPSG:4326")
    inside_idx = points.sjoin(city_frame, predicate="within", how="inner").index
    inside = parcels.index.isin(inside_idx)
    ward = parcels["MUNICODE"].between(101, 132)
    log("Clip (representative point within the City of Pittsburgh boundary):")
    log(f"  candidates: {len(parcels):,}")
    log(f"  inside boundary and ward 101-132: {int((inside & ward).sum()):,}")
    log(f"  inside boundary, NOT a city ward (kept; the boundary wins): {int((inside & ~ward).sum()):,}")
    log(f"  ward 101-132 but outside the boundary (dropped): {int((~inside & ward).sum()):,}")
    if int((inside & ~ward).sum()):
        stray = parcels.loc[inside & ~ward, "MUNICODE"].value_counts().head(8)
        log(f"  non-ward municodes inside the city: {stray.to_dict()}")
    kept = parcels.loc[inside].copy()
    if len(kept) < 50000 or len(kept) > 250000:
        raise RuntimeError(f"City clip produced {len(kept):,} parcels; expected on the order of 140k.")
    return kept


def collapse_duplicate_pins(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One geometry footprint per PIN. Values are account-level, so they must not be summed."""
    dup_keys = gdf.loc[gdf["PIN"].duplicated(keep=False), "PIN"]
    log(f"PINs with more than one geometry row: {dup_keys.nunique():,}")
    if dup_keys.empty:
        return gdf
    value_cols = ["FAIRMARKETLAND", "FAIRMARKETBUILDING", "FAIRMARKETTOTAL", "LOTAREA", "FINISHEDLIVINGAREA"]
    multi = gdf[gdf["PIN"].isin(set(dup_keys))]
    for col in value_cols:
        if col not in multi.columns:
            continue
        varied = multi.groupby("PIN")[col].nunique(dropna=False)
        n_varied = int(varied.gt(1).sum())
        if n_varied:
            raise RuntimeError(
                f"{n_varied} duplicate PINs disagree on {col}. "
                "That is distinct value records, not a broadcast; refusing to take first."
            )
    agg = {c: "first" for c in gdf.columns if c not in {"PIN", "geometry"}}
    collapsed = gdf.groupby("PIN", dropna=False).agg(agg).reset_index()
    geoms = gdf.groupby("PIN", dropna=False)["geometry"].apply(
        lambda series: unary_union([g for g in series if g is not None and not g.is_empty])
    )
    collapsed = collapsed.merge(geoms.rename("geometry"), left_on="PIN", right_index=True, how="left")
    out = gpd.GeoDataFrame(collapsed, geometry="geometry", crs="EPSG:4326")
    if out["PIN"].duplicated().any():
        raise RuntimeError("PIN collapse left duplicate keys")
    log(f"After PIN union: {len(out):,}")
    return out


def _situs(frame: pd.DataFrame) -> pd.Series:
    def one(row) -> str:
        bits = []
        number = str(row["PROPERTYHOUSENUM"] or "").strip()
        if number.endswith(".0"):
            number = number[:-2]
        if number and number not in {"0", "nan", "None"}:
            bits.append(number)
        frac = str(row["PROPERTYFRACTION"] or "").strip()
        if frac and frac.lower() not in {"nan", "none"}:
            bits.append(frac)
        street = str(row["PROPERTYADDRESS"] or "").strip()
        if street and street.lower() not in {"nan", "none"}:
            bits.append(street)
        unit = str(row["PROPERTYUNIT"] or "").strip()
        line = " ".join(bits)
        if unit and unit.lower() not in {"nan", "none"}:
            line = f"{line} #{unit}".strip()
        city = str(row["PROPERTYCITY"] or "").strip()
        zip_code = str(row["PROPERTYZIP"] or "").strip()
        if zip_code.endswith(".0"):
            zip_code = zip_code[:-2]
        tail = " ".join(p for p in (city, "PA", zip_code) if p and p.lower() not in {"nan", "none"})
        if line and tail:
            return f"{line}, {tail}"
        return line or tail

    return frame.apply(one, axis=1)


def categorize(usecode: str, usedesc: str, cls: str, land: float, improvement: float) -> str:
    code = str(usecode or "").strip()
    desc = str(usedesc or "").upper()
    total = (land or 0) + (improvement or 0)
    if code in {"50", "55", "56", "57", "118", "450", "550", "556"} or "CONDO" in desc:
        return "Condominium"
    if "VACANT" in desc or code in {"100", "110", "111", "130", "300", "400"}:
        if cls == "C" or code == "400":
            return "Vacant Commercial"
        if cls == "I" or code == "300":
            return "Vacant Industrial"
        if cls == "F" or code == "110":
            return "Vacant Agricultural"
        return "Vacant Residential"
    if code == "777" or "PARKING LOT" in desc:
        return "Parking Lot"
    if code == "455" or "COMMERCIAL GARAGE" in desc:
        return "Parking Garage"
    if code == "456" or ("PARKING" in desc and "GARAGE" in desc):
        # The county mixes surface lots and garages in use 456. A garage carries
        # the building; a surface lot does not.
        if total > 0 and (improvement or 0) > (land or 0):
            return "Parking Garage"
        return "Parking Lot"
    if code == "10":
        return "Single Family"
    if code == "20":
        return "Two Family"
    if code == "30":
        return "Three Family"
    if code == "40":
        return "Four Family"
    if code == "70":
        return "Rowhouse"
    if code == "60":
        return "Townhouse"
    if code in {"80", "90", "415"} or "MOBILE HOME" in desc:
        return "Mobile Home"
    if code in {"401", "402", "403", "557", "558", "559"} or "APART" in desc:
        return "Multi-Family"
    if cls == "I":
        return "Industrial"
    if cls == "C":
        return "Commercial"
    if cls == "F":
        return "Agricultural"
    if cls == "U":
        return "Utility"
    if cls == "G":
        return "Government"
    if cls == "R":
        return "Residential Other"
    return "Other"


def _dev_key(mapblock: str) -> str:
    """Parent lot id. Allegheny mapblocks are map-block-lot, then a unit suffix.

    1-G-224-1001 and 1-C-1662-42 both roll up to the first three segments.
    """
    parts = [p for p in str(mapblock or "").split("-") if p]
    if len(parts) >= 4:
        return "-".join(parts[:3])
    return ""


def _filled(geom):
    """Drop holes so a unit drawn inside a punched-out courtyard still matches."""
    from shapely.geometry import MultiPolygon, Polygon
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return unary_union([Polygon(p.exterior) for p in geom.geoms if not p.is_empty])
    return geom


def merge_condos(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge $0-land condo stubs onto the parent lot, or a hull of the stubs.

    Unit polygons are ~300 sqft placeholders. The real land is the map-block-lot
    polygon (MAPBLOCKLO with the unit suffix removed) when the county drew one,
    and otherwise the cluster of stubs that share that lot id. The county records
    no land value on either, so building value is summed and land stays $0.
    """
    work = gdf.to_crs(UTM).copy()
    work["geometry"] = work.geometry.apply(_polygonal)
    work = work[work.geometry.notna()].copy()
    work["MAPBLOCKLO"] = work["MAPBLOCKLO"].fillna("").astype(str).str.strip().str.upper()
    work["sqft"] = work.geometry.area * 10.763910416709722
    land = pd.to_numeric(work["FAIRMARKETLAND"], errors="coerce").fillna(0.0)
    lot = pd.to_numeric(work["LOTAREA"], errors="coerce").fillna(0.0)
    work["_dev"] = work["MAPBLOCKLO"].map(_dev_key)
    known = set(work.loc[work["MAPBLOCKLO"].ne(""), "MAPBLOCKLO"])
    # A stub is a condo card on a tiny footprint. Land may be $0 (the usual case)
    # or a real unit share; either way it is summed onto the lot, not left as a pencil.
    is_stub = (
        work["USECODE"].isin(UNIT_USES)
        & (lot.le(0) | work["sqft"].lt(STUB_SQFT))
        & work["exemption_flag"].eq(0)
    )
    # The lot polygon itself is not a stub, even when a unit suffix names it.
    is_stub = is_stub & ~work["MAPBLOCKLO"].isin(set(work.loc[is_stub, "_dev"]))
    has_lot = is_stub & work["_dev"].isin(known) & work["_dev"].ne(work["MAPBLOCKLO"])
    log(f"Condo stubs on a drawn parent lot: {int(has_lot.sum()):,} "
        f"across {work.loc[has_lot, '_dev'].nunique():,} lots")

    def _sum_onto(parents: gpd.GeoDataFrame, units: gpd.GeoDataFrame, key: str) -> gpd.GeoDataFrame:
        summed = units.groupby(key).agg(
            unit_land=("FAIRMARKETLAND", "sum"),
            unit_bldg=("FAIRMARKETBUILDING", "sum"),
            unit_living=("FINISHEDLIVINGAREA", "sum"),
            n_units=("PIN", "size"),
        )
        parents = parents.copy()
        parents["FAIRMARKETLAND"] = pd.to_numeric(parents["FAIRMARKETLAND"], errors="coerce").fillna(0) + parents[key].map(summed["unit_land"]).fillna(0)
        parents["FAIRMARKETBUILDING"] = pd.to_numeric(parents["FAIRMARKETBUILDING"], errors="coerce").fillna(0) + parents[key].map(summed["unit_bldg"]).fillna(0)
        parents["FAIRMARKETTOTAL"] = parents["FAIRMARKETLAND"] + parents["FAIRMARKETBUILDING"]
        parents["FINISHEDLIVINGAREA"] = pd.to_numeric(parents["FINISHEDLIVINGAREA"], errors="coerce").fillna(0) + parents[key].map(summed["unit_living"]).fillna(0)
        parents["assessor_records"] = parents[key].map(summed["n_units"]).fillna(0).astype(int) + 1
        revived = parents["exemption_flag"].eq(1)
        if int(revived.sum()):
            log(f"  exempt lots kept because taxable condo units sit on them: {int(revived.sum())}")
            parents.loc[revived, "exemption_flag"] = 0
        parents["property_hint"] = "Condominium"
        parents["USEDESC"] = parents["assessor_records"].map(lambda n: f"CONDOMINIUM ({int(n) - 1} units)")
        return parents

    unit_geoms = work.loc[has_lot].groupby("_dev")["geometry"].apply(
        lambda series: unary_union([g for g in series if g is not None])
    )
    parents = work[work["MAPBLOCKLO"].isin(set(work.loc[has_lot, "_dev"]))].copy()
    # One polygon per lot id: the largest, if the id was duplicated.
    parents = parents.sort_values("sqft", ascending=False).drop_duplicates("MAPBLOCKLO")
    parents["_dev"] = parents["MAPBLOCKLO"]
    parents = _sum_onto(parents, work.loc[has_lot], "_dev")
    merged_geoms = []
    for dev, geom in zip(parents["_dev"], parents.geometry):
        extra = unit_geoms.get(dev)
        merged_geoms.append(unary_union([geom, extra]) if extra is not None else geom)
    parents["geometry"] = merged_geoms

    consumed = set(work.loc[has_lot, "PIN"]) | set(parents["PIN"])
    rest = work[~work["PIN"].isin(consumed)].copy()

    orphan = (
        rest["USECODE"].isin(UNIT_USES)
        & rest["exemption_flag"].eq(0)
        & rest["_dev"].ne("")
        & (
            pd.to_numeric(rest["LOTAREA"], errors="coerce").fillna(0).le(0)
            | rest["sqft"].lt(STUB_SQFT)
        )
    )
    counts = rest.loc[orphan, "_dev"].value_counts()
    hull_keys = set(counts[counts >= 2].index)
    hull_rows = []
    hull_pins = set()
    if hull_keys:
        for dev, group in rest.loc[orphan & rest["_dev"].isin(hull_keys)].groupby("_dev"):
            hull = unary_union([g for g in group.geometry if g is not None]).convex_hull
            if hull is None or hull.is_empty or hull.geom_type != "Polygon":
                hull = unary_union([g for g in group.geometry if g is not None]).buffer(1.0)
            label = pd.to_numeric(group["FAIRMARKETBUILDING"], errors="coerce").fillna(0).idxmax()
            base = group.loc[label]
            if isinstance(base, pd.DataFrame):
                base = base.iloc[0]
            record = base.to_dict()
            record["geometry"] = _polygonal(hull)
            record["FAIRMARKETLAND"] = float(pd.to_numeric(group["FAIRMARKETLAND"], errors="coerce").fillna(0).sum())
            record["FAIRMARKETBUILDING"] = float(pd.to_numeric(group["FAIRMARKETBUILDING"], errors="coerce").fillna(0).sum())
            record["FAIRMARKETTOTAL"] = record["FAIRMARKETLAND"] + record["FAIRMARKETBUILDING"]
            record["FINISHEDLIVINGAREA"] = float(pd.to_numeric(group["FINISHEDLIVINGAREA"], errors="coerce").fillna(0).sum())
            record["assessor_records"] = int(len(group))
            record["property_hint"] = "Condominium"
            record["USEDESC"] = f"CONDOMINIUM ({len(group)} units)"
            record["exemption_flag"] = 0
            hull_rows.append(record)
            hull_pins.update(group["PIN"])
        log(f"Condo stubs with no drawn lot, hulled by mapblock: {len(hull_pins):,} units -> {len(hull_rows):,} developments")
    else:
        log("Condo stubs with no drawn lot: none")

    rest = rest[~rest["PIN"].isin(hull_pins)].copy()
    if hull_rows:
        rest = pd.concat([rest, gpd.GeoDataFrame(hull_rows, geometry="geometry", crs=UTM)], ignore_index=True)

    # Stubs that still have no lot: drop them onto a common-area polygon that
    # contains them once its unit-shaped holes are filled.
    rest_lot = pd.to_numeric(rest["LOTAREA"], errors="coerce").fillna(0)
    rest_sqft = rest.geometry.area * 10.763910416709722
    hint = rest["property_hint"] if "property_hint" in rest.columns else ""
    still = (
        rest["USECODE"].isin(UNIT_USES)
        & rest["exemption_flag"].eq(0)
        & (rest_lot.le(0) | rest_sqft.lt(STUB_SQFT))
        & ~pd.Series(hint, index=rest.index).fillna("").eq("Condominium")
    )
    commons = rest[rest["USECODE"].isin(COMMON_USES)].copy()
    matched_pins = set()
    if int(still.sum()) and len(commons):
        commons["geometry"] = commons.geometry.apply(_filled)
        points = gpd.GeoDataFrame(
            {"unit_pin": rest.loc[still, "PIN"].to_numpy()},
            geometry=list(rest.loc[still].geometry.representative_point()),
            crs=UTM,
        )
        lots = commons[["PIN", "geometry"]].rename(columns={"PIN": "parent_pin"})
        hit = points.sjoin(lots, predicate="within", how="left")
        hit = hit[hit["parent_pin"].notna()].copy()
        if len(hit):
            area = commons.set_index("PIN").geometry.area
            hit["_area"] = hit["parent_pin"].map(area)
            hit = hit.sort_values("_area").drop_duplicates("unit_pin")
            link = hit.set_index("unit_pin")["parent_pin"]
            stub_rows = rest[rest["PIN"].isin(set(link.index))].copy()
            stub_rows["_dev"] = stub_rows["PIN"].map(link)
            targets = rest[rest["PIN"].isin(set(link.values))].copy()
            targets["_dev"] = targets["PIN"]
            targets = _sum_onto(targets, stub_rows, "_dev")
            extra = stub_rows.groupby("_dev")["geometry"].apply(
                lambda series: unary_union([g for g in series if g is not None])
            )
            targets["geometry"] = [
                unary_union([geom, extra.get(pin)]) if extra.get(pin) is not None else geom
                for pin, geom in zip(targets["PIN"], targets.geometry)
            ]
            rest = rest[~rest["PIN"].isin(set(link.index)) & ~rest["PIN"].isin(set(targets["PIN"]))].copy()
            rest = pd.concat([rest, targets], ignore_index=True)
            matched_pins = set(link.index)
        log(f"Condo stubs matched spatially to common area: {len(matched_pins):,}")
    elif int(still.sum()):
        log(f"Condo stubs left unmatched: {int(still.sum()):,}")

    out = pd.concat([
        parents.drop(columns=["_dev"], errors="ignore"),
        rest.drop(columns=["_dev"], errors="ignore"),
    ], ignore_index=True)
    if "property_hint" not in out.columns:
        out["property_hint"] = ""
    out["property_hint"] = out["property_hint"].fillna("")
    if "assessor_records" not in out.columns:
        out["assessor_records"] = 1
    out["assessor_records"] = out["assessor_records"].fillna(1).astype(int)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=UTM)
    out["geometry"] = out.geometry.apply(_polygonal)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].to_crs("EPSG:4326")
    if out["PIN"].duplicated().any():
        raise RuntimeError("condo merge left duplicate PINs")
    log(f"After condo merge: {len(out):,} parcels (started from {len(gdf):,})")
    return out



def collapse_coincident_cards(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Collapse land/building cards the county drew on one copied footprint.

    Allegheny issues a second PIN (a land card, a building card, an air-rights
    card) on the same polygon. Values that differ are separate records and are
    summed. Values that are identical on every card are one assessment copied
    onto each card, so they are kept once.
    """
    work = gdf.to_crs(UTM).copy()
    sqft = (work.geometry.area * 10.763910416709722).round(0)
    point = work.geometry.representative_point()
    key = point.x.round(1).astype(str) + "," + point.y.round(1).astype(str) + "|" + sqft.astype("Int64").astype(str)
    counts = key.value_counts()
    stacked = key.isin(set(counts[counts > 1].index))
    log(f"Coincident footprints: {int(counts[counts > 1].sum()):,} rows in {int((counts > 1).sum()):,} stacks")
    if not stacked.any():
        return gdf

    def _prefer(group: pd.DataFrame) -> pd.Series:
        building = pd.to_numeric(group["FAIRMARKETBUILDING"], errors="coerce").fillna(0)
        land_val = pd.to_numeric(group["FAIRMARKETLAND"], errors="coerce").fillna(0)
        vacant = group["USEDESC"].fillna("").str.upper().str.contains("VACANT")
        score = building + land_val.where(~vacant, 0) * 0.001
        label = score.idxmax()
        row = group.loc[label]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    kept_rows = []
    drop_pins = set()
    for _, group in work.loc[stacked].groupby(key[stacked], sort=False):
        land_vals = pd.to_numeric(group["FAIRMARKETLAND"], errors="coerce").fillna(0).round(0)
        bldg_vals = pd.to_numeric(group["FAIRMARKETBUILDING"], errors="coerce").fillna(0).round(0)
        base = _prefer(group).to_dict()
        same = int(land_vals.nunique()) <= 1 and int(bldg_vals.nunique()) <= 1
        if same:
            base["FAIRMARKETLAND"] = float(land_vals.iloc[0])
            base["FAIRMARKETBUILDING"] = float(bldg_vals.iloc[0])
        else:
            base["FAIRMARKETLAND"] = float(land_vals.sum())
            base["FAIRMARKETBUILDING"] = float(bldg_vals.sum())
            base["FINISHEDLIVINGAREA"] = float(pd.to_numeric(group["FINISHEDLIVINGAREA"], errors="coerce").fillna(0).sum())
        base["FAIRMARKETTOTAL"] = base["FAIRMARKETLAND"] + base["FAIRMARKETBUILDING"]
        base["assessor_records"] = int(pd.to_numeric(group.get("assessor_records", 1), errors="coerce").fillna(1).sum())
        base["exemption_flag"] = 0 if (group["exemption_flag"] == 0).any() else 1
        base["geometry"] = unary_union([g for g in group.geometry if g is not None and not g.is_empty])
        kept_rows.append(base)
        drop_pins.update(group["PIN"])

    rest = work[~work["PIN"].isin(drop_pins)].copy()
    folded = gpd.GeoDataFrame(kept_rows, geometry="geometry", crs=UTM)
    out = pd.concat([rest, folded], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=UTM).to_crs("EPSG:4326")
    out["geometry"] = out.geometry.apply(_polygonal)
    out = out[out.geometry.notna()].copy()
    if out["PIN"].duplicated().any():
        raise RuntimeError("coincident-card collapse left duplicate PINs")
    log(f"After coincident-card collapse: {len(out):,} parcels")
    return out


def main() -> None:
    boundary = fetch_boundary()
    geometry = fetch_geometry(boundary)
    clipped = clip_to_city(geometry, boundary)
    assess = fetch_assessments()
    city_pins = set(clipped["PIN"])
    roll = assess[assess["PARID"].isin(city_pins)].copy()
    log(f"Assessments matched to clipped parcels: {len(roll):,} of {len(clipped):,} geometries")
    missing_geom = assess["MUNICODE"].between(101, 132) & ~assess["PARID"].isin(city_pins)
    log(f"Ward 101-132 assessment rows with no in-city geometry: {int(missing_geom.sum()):,}")
    if int(missing_geom.sum()):
        log(f"  those uses: {assess.loc[missing_geom, 'USEDESC'].value_counts().head(6).to_dict()}")

    joined = clipped.merge(roll, left_on="PIN", right_on="PARID", how="inner", suffixes=("", "_roll"))
    if "MUNICODE_roll" in joined.columns:
        joined = joined.drop(columns=["MUNICODE_roll"])
    log(f"Joined rows: {len(joined):,}")
    joined = collapse_duplicate_pins(joined)

    # Exemption follows the county's stated tax status. CLASS is land use, not
    # taxability, so a taxable utility stays. Homestead, Clean & Green, and
    # abatement are not consulted here. They are not in FAIRMARKET*, which is
    # the value this file ships. They are applied only on COUNTY* and LOCAL*,
    # and those columns are not read.
    tax = joined["TAXCODE"].fillna("").astype(str).str.strip().str.upper()
    sub = joined["TAXSUBCODE"].fillna("").astype(str).str.strip().str.upper()
    joined["exemption_flag"] = (tax.eq("E") | (tax.eq("P") & sub.eq("E"))).astype(int)
    log("Exemption flag before the drop:")
    log(f"  flag 1: {int(joined['exemption_flag'].sum()):,}  "
        f"flag 0: {int((joined['exemption_flag'] == 0).sum()):,}")
    log(f"  by TAXCODE x CLASS (flagged rows):\n"
        f"{joined.loc[joined['exemption_flag'].eq(1)].groupby(['TAXCODE', 'CLASS']).size().to_string()}")
    homestead = joined["HOMESTEADFLAG"].fillna("").astype(str).str.upper().str.contains("HOM")
    log(f"  homestead (partial, kept, not flagged): {int((homestead & joined['exemption_flag'].eq(0)).sum()):,}")

    joined = merge_condos(joined)
    joined = collapse_coincident_cards(joined)
    before = len(joined)
    shipped = joined[joined["exemption_flag"].eq(0)].copy()
    if shipped["exemption_flag"].ne(0).any():
        raise RuntimeError("exempt parcels remained in the shipped set")

    shipped["land_value"] = pd.to_numeric(shipped["FAIRMARKETLAND"], errors="coerce").fillna(0.0)
    shipped["improvement_value"] = pd.to_numeric(shipped["FAIRMARKETBUILDING"], errors="coerce").fillna(0.0)
    shipped["full_market_value"] = shipped["land_value"] + shipped["improvement_value"]
    hint = shipped.get("property_hint", pd.Series("", index=shipped.index)).fillna("")
    shipped["property_land_use_category"] = [
        "Condominium" if h == "Condominium" else categorize(code, desc, cls, land, impr)
        for h, code, desc, cls, land, impr in zip(
            hint, shipped["USECODE"], shipped["USEDESC"], shipped["CLASS"],
            shipped["land_value"], shipped["improvement_value"],
        )
    ]
    log("Original categories:")
    log(shipped["property_land_use_category"].value_counts(dropna=False).to_string())

    log("Computing geodesic areas...")
    shipped["geom_area_sqft"] = shipped.geometry.apply(gis_area_sqft)
    shipped.loc[shipped["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
    source_area = pd.to_numeric(shipped["LOTAREA"], errors="coerce")
    # Condo parents carry the common lot's stated area; unit rows were removed.
    check_area_agreement(shipped["geom_area_sqft"], source_area, label="LOTAREA", log=log)
    calc_acres = pd.to_numeric(shipped["CALCACREAG"], errors="coerce") * SQFT_PER_ACRE
    check_area_agreement(shipped["geom_area_sqft"], calc_acres, label="CALCACREAG", log=log)
    shipped["land_area_sqft"] = shipped["geom_area_sqft"]
    shipped["area_source"] = "gis"
    shipped["land_area_acres"] = shipped["land_area_sqft"] / SQFT_PER_ACRE
    shipped["likely_remnant"] = (shipped["land_area_sqft"] < 500).fillna(False).astype(int)

    den = shipped["land_area_sqft"].replace(0, np.nan)
    shipped["land_value_per_sqft"] = shipped["land_value"] / den
    shipped["improvement_value_per_sqft"] = shipped["improvement_value"] / den
    shipped["full_market_value_per_sqft"] = shipped["full_market_value"] / den
    shipped = add_improvement_ratio_fields(
        shipped, land_col="land_value", improvement_col="improvement_value"
    )
    shipped["bld_ar"] = pd.to_numeric(shipped["FINISHEDLIVINGAREA"], errors="coerce").fillna(0)
    shipped["property_land_use_refined"] = classify_property_refined(
        shipped,
        sf_cutoff=0.67,
        other_cutoff=0.50,
        exclude_categories=("Utility", "Government", "Agricultural", "Condominium"),
        category_col="property_land_use_category",
        land_col="land_value",
        improvement_col="improvement_value",
        bld_ar_col="bld_ar",
        fetch_footprints=False,
    )
    log("Refined categories:")
    log(shipped["property_land_use_refined"].value_counts(dropna=False).to_string())

    shipped["address"] = _situs(shipped)
    shipped["ward"] = shipped["MUNIDESC"].fillna("").astype(str).str.strip()
    shipped["use_description"] = shipped["USEDESC"].fillna("").astype(str).str.strip()
    shipped["link"] = PORTAL + shipped["PIN"].astype(str)
    shipped["value_base_year"] = VALUE_BASE_YEAR
    shipped["tax_year"] = pd.to_numeric(shipped["TAXYEAR"], errors="coerce")
    if "assessor_records" not in shipped.columns:
        shipped["assessor_records"] = 1

    columns = [
        "geometry", "exemption_flag", "likely_remnant",
        "property_land_use_category", "property_land_use_refined",
        "full_market_value", "full_market_value_per_sqft",
        "land_value", "land_value_per_sqft",
        "improvement_value", "improvement_value_per_sqft",
        "TLLDIMPROV", "IMPR_LAND_RATIO", "IMPR_LAND_PCT", "IMPR_PCT_TOTAL",
        "land_area_acres", "area_source", "link",
        "PIN", "address", "ward", "use_description", "USECODE",
        "assessor_records", "value_base_year", "tax_year",
    ]
    final = shipped[columns].rename(columns={
        "land_value": "current_full_land_value",
        "PIN": "pin",
        "USECODE": "use_code",
    })
    final["geometry"] = final.geometry.apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
    final["geometry"] = final.geometry.apply(_polygonal)
    final = final[final.geometry.notna()].copy()
    final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")
    if final["pin"].duplicated().any():
        raise RuntimeError("final parquet has duplicate pins")
    if not str(final.crs).endswith("4326"):
        raise RuntimeError(f"CRS is {final.crs}, expected EPSG:4326")

    out = DATA_DIR / "pittsburgh-pa-parcels.parquet"
    stamped = DATA_DIR / f"pittsburgh-pa-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet"
    final.to_parquet(out, index=False)
    final.to_parquet(stamped, index=False)
    log(f"SAVED {out} | rows {len(final):,}")
    log(f"land ${final['current_full_land_value'].sum():,.0f}  "
        f"building ${final['improvement_value'].sum():,.0f}  "
        f"total ${final['full_market_value'].sum():,.0f}")
    log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")
    log(f"bounds: {[round(v, 4) for v in final.total_bounds]}")

    log("--- condo/stub smoke alarms (skill section 6a) ---")
    area = shipped["geom_area_sqft"]
    ppsf = pd.to_numeric(final["land_value_per_sqft"], errors="coerce")
    log(f"  footprint sqft p1/p5/p10: {[round(area.quantile(q)) for q in (0.01, 0.05, 0.10)]}")
    log(f"  sub-500 / sub-1000 sqft footprints: {int((area < 500).sum()):,} / {int((area < 1000).sum()):,}")
    log(f"  land $/sqft p50/p95/p99/max: ${ppsf.median():,.2f} / ${ppsf.quantile(.95):,.2f} / "
        f"${ppsf.quantile(.99):,.2f} / ${ppsf.max():,.2f}")
    points = final.geometry.representative_point()
    stacks = (points.x.round(5).astype(str) + "," + points.y.round(5).astype(str)).value_counts()
    log(f"  stacked clusters: {int((stacks > 1).sum())}, max stack: {int(stacks.max())}")
    holes = final.geometry.apply(lambda g: 0 if g is None else sum(
        len(p.interiors) for p in (g.geoms if getattr(g, "geom_type", "") == "MultiPolygon" else [g])
        if getattr(p, "interiors", None) is not None
    ))
    log(f"  parcels with interior rings: {int((holes > 0).sum()):,}")
    log(f"  zero land value: {int((final['current_full_land_value'].fillna(0) <= 0).sum()):,}")
    center = boundary.geometry.iloc[0].representative_point()
    log(f"  boundary representative point (map center candidate): {center.x:.5f}, {center.y:.5f}")
    if not final["link"].str.startswith(LINK_OK_PREFIX).all():
        raise RuntimeError("a parcel link is missing the assessor portal prefix")


if __name__ == "__main__":
    main()
