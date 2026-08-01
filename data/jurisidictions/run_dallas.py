#!/usr/bin/env python3
"""
Build Dallas's canonical parcel parquet by joining two free DCAD sources.

Sources:
- Parcel geometry + class + owner + situs + area + exempt flag + detail link:
  City of Dallas "DallasTaxParcels" ArcGIS FeatureServer (aggregates the 5 county
  appraisal districts; we keep City-of-Dallas / Dallas-County parcels).
    https://gis.dallascityhall.com/arcgis/rest/services/Basemap/DallasTaxParcels/FeatureServer/0
  Pulled + cached on first run (paginated GeoJSON); reused after.
- Appraisal values (land / improvement / total): Dallas Central Appraisal District
  (DCAD) appraisal-roll export, comma-delimited. The single file we need is
  ACCOUNT_APPRL_YEAR (account number + IMPR_VAL + LAND_VAL + TOT_VAL + DIVISION_CD).
  DCAD publishes these as free bulk ZIPs with no stated usage restrictions
  (checked 2026-07: dallascad.org has no terms-of-use page; data is provided
  as-is without warranty). This script still expects a manual download — go to:
    https://www.dallascad.org/DataProducts.aspx
    -> "Data Files with Proposed Values" (current year, Comma Delimited)
  and place the ZIP at:
    data/jurisidictions/data/dallas/dcad-appraisal-export.zip
  then re-run; the script picks it up from there.
  (Courtesy note: DCAD's robots.txt disallows crawling per-account /Acct pages —
  always use the bulk ZIPs, never scrape account pages.)
  (The values are the only thing the GIS layer is missing.)

Outputs:
- data/jurisidictions/data/dallas/dallas-tx-parcels.parquet
- data/jurisidictions/data/dallas/dallas-tx-parcels_YYYY_MM_DD.parquet

Notes:
- City of Dallas spans 5 counties, but DCAD values only cover Dallas County. We keep
  the Dallas-County portion of the city (~280k parcels; the small slivers in
  Collin/Denton/Rockwall/Kaufman lack DCAD values and are dropped). Documented filter.
- $/sqft denominator is the assessor's reported parcel area (AREA_FEET from the GIS
  layer, which equals the DCAD parcel area), falling back to the geodesic polygon
  area when AREA_FEET is missing. Emits QC columns land_area_acres, area_source, and
  likely_remnant (tiny <500 sqft fractional remnants whose $/sqft is meaningless).
- City limits via the authoritative City of Dallas boundary FeatureServer (centroid
  within), not osmnx geocoding.
- SPTBCODE is the DCAD legacy state class (A11/B11/C12/F10...) -> PTAD categories
  (see data/jurisidictions Dallas notes). classify_property_refined(fetch_footprints
  =False): Dallas hides the Vacant & Underdeveloped tab (hideUnderutilized), so the
  Overture footprint cross-check is skipped for speed; flip to True if ever enabled.
- Upload + PMTiles are separate steps (Dallas is large -> PMTiles):
    python data/scripts/parquet_to_pmtiles.py --city dallas --h3 --wsl --upload
"""
from __future__ import annotations

import io
import os
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from datetime import datetime
from pathlib import Path
from shapely.ops import unary_union
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "dallas"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "dallas-tx-geometry.parquet"


def find_zip():
    """The DCAD export; accept the documented name or any DCAD*.zip dropped in the dir."""
    named = DATA_DIR / "dcad-appraisal-export.zip"
    if named.exists():
        return named
    cands = sorted(DATA_DIR.glob("*.zip")) + sorted(DATA_DIR.glob("*.ZIP"))
    return cands[0] if cands else named


ZIP = find_zip()

PARCELS_URL = ("https://gis.dallascityhall.com/arcgis/rest/services/Basemap/"
               "DallasTaxParcels/FeatureServer/0/query")
BOUNDARY_URL = ("https://services5.arcgis.com/74bZbbuf05Ctvbzv/arcgis/rest/services/"
                "City_of_Dallas_Boundary/FeatureServer/0/query")
# Server-side pre-filter: City of Dallas parcels that live in Dallas County (the set
# DCAD has values for). The authoritative boundary clip below refines this.
PARCEL_WHERE = "CITY='DALLAS' AND COUNTY='DALLAS COUNTY'"
GEOM_FIELDS = ("GIS_ACCT,SPTBCODE,PROP_CL,AREA_FEET,CITY,COUNTY,"
               "TAXPANAME1,TOTEXEMPT,Website")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
geod = Geod(ellps="WGS84")

# DCAD ACCOUNT_APPRL_YEAR column order (used only if the CSV ships without a header row).
APPRL_COLS = [
    "ACCOUNT_NUM", "APPRAISAL_YR", "IMPR_VAL", "LAND_VAL", "LAND_AG_EXEMPT", "AG_USE_VAL",
    "TOT_VAL", "HMSTD_CAP_VAL", "REVAL_YR", "PREV_REVAL_YR", "PREV_MKT_VAL", "TOT_CONTRIB_AMT",
    "TAXPAYER_REP", "CITY_JURIS_DESC", "COUNTY_JURIS_DESC", "ISD_JURIS_DESC",
    "HOSPITAL_JURIS_DESC", "COLLEGE_JURIS_DESC", "SPECIAL_DIST_JURIS_DESC",
    "CITY_SPLIT_PCT", "COUNTY_SPLIT_PCT", "ISD_SPLIT_PCT", "HOSPITAL_SPLIT_PCT",
    "COLLEGE_SPLIT_PCT", "SPECIAL_DIST_SPLIT_PCT", "CITY_TAXABLE_VAL", "COUNTY_TAXABLE_VAL",
    "ISD_TAXABLE_VAL", "HOSPITAL_TAXABLE_VAL", "COLLEGE_TAXABLE_VAL", "SPECIAL_DIST_TAXABLE_VAL",
    "CITY_CEILING_VALUE", "COUNTY_CEILING_VALUE", "ISD_CEILING_VALUE", "HOSPITAL_CEILING_VALUE",
    "COLLEGE_CEILING_VALUE", "SPECIAL_DIST_CEILING_VALUE", "VID_IND", "GIS_PARCEL_ID",
    "APPRAISAL_METH_CD", "RENDITION_PENALTY", "DIVISION_CD", "EXTRNL_CNTY_ACCT",
    "EXTRNL_CITY_ACCT", "P_BUS_TYP_CD", "BLDG_CLASS_CD", "SPTD_CODE",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def norm_acct(s):
    return s.astype(str).str.strip().str.upper()


# ── 1. Parcel geometry from the City of Dallas FeatureServer (cached) ─────────
def fetch_parcels():
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    total = requests.get(PARCELS_URL, params={"where": PARCEL_WHERE,
                         "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} parcels from DallasTaxParcels (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        r = requests.get(PARCELS_URL, params={
            "where": PARCEL_WHERE, "outFields": GEOM_FIELDS, "returnGeometry": "true",
            "resultOffset": off, "resultRecordCount": PAGE, "outSR": 4326, "f": "geojson",
        }, headers=HEADERS, timeout=180)
        r.raise_for_status()
        gdf = gpd.read_file(io.BytesIO(r.content))
        if not len(gdf):
            break
        pages.append(gdf)
        off += len(gdf)
        if off % 20000 < PAGE:
            log(f"  fetched {off:,}/{total:,}")
        if len(gdf) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


geom = fetch_parcels()
geom["acct"] = norm_acct(geom["GIS_ACCT"])
geom = geom.dropna(subset=["acct"])
geom = geom[geom["acct"].str.len() > 0]

# ── 2. DCAD appraisal values (land / improvement / total) ────────────────────
if not ZIP.exists():
    raise SystemExit(
        f"\nMissing DCAD appraisal export: {ZIP}\n"
        "Download the current-year comma-delimited 'Data Files with Proposed Values' "
        "ZIP from https://www.dallascad.org/DataProducts.aspx and place it there, "
        "then re-run.\n")

log(f"Reading ACCOUNT_APPRL_YEAR from {ZIP.name}...")
NEED = ["ACCOUNT_NUM", "LAND_VAL", "IMPR_VAL", "TOT_VAL", "DIVISION_CD"]
with zipfile.ZipFile(ZIP) as zf:
    member = next((n for n in zf.namelist()
                   if "ACCOUNT_APPRL_YEAR" in os.path.basename(n).upper()), None)
    if member is None:
        raise SystemExit("ACCOUNT_APPRL_YEAR file not found inside the DCAD zip. "
                         f"Members: {zf.namelist()[:20]}")
    # Peek the header to decide header-aware vs positional read (file is ~330MB).
    with zf.open(member) as fh:
        first = io.TextIOWrapper(fh, encoding="latin-1").readline()
    has_header = "ACCOUNT_NUM" in first.upper()
    read_kw = dict(dtype=str, sep=",", encoding="latin-1", on_bad_lines="skip", usecols=NEED)
    if not has_header:
        read_kw.update(header=None, names=APPRL_COLS)
    with zf.open(member) as fh:
        appr = pd.read_csv(fh, **read_kw)
    # TOTAL_EXEMPTION: authoritative list of fully-exempt accounts.
    tot_member = next((n for n in zf.namelist()
                       if "TOTAL_EXEMPTION" in os.path.basename(n).upper()), None)
    exempt_accts = set()
    if tot_member:
        with zf.open(tot_member) as fh:
            te = pd.read_csv(fh, dtype=str, sep=",", encoding="latin-1",
                             on_bad_lines="skip", usecols=["ACCOUNT_NUM"])
        exempt_accts = set(norm_acct(te["ACCOUNT_NUM"]))
        log(f"  {len(exempt_accts):,} totally-exempt accounts (TOTAL_EXEMPTION)")
appr.columns = [c.upper().strip() for c in appr.columns]
appr["acct"] = norm_acct(appr["ACCOUNT_NUM"])
for c in ["LAND_VAL", "IMPR_VAL", "TOT_VAL"]:
    appr[c] = pd.to_numeric(appr.get(c), errors="coerce").fillna(0)
appr["DIVISION_CD"] = appr.get("DIVISION_CD", "").astype(str).str.strip().str.upper()
appr = appr.drop_duplicates("acct")
vals = appr[["acct", "LAND_VAL", "IMPR_VAL", "TOT_VAL", "DIVISION_CD"]]
log(f"  {len(vals):,} appraisal rows")

parcel = geom.merge(vals, on="acct", how="left")
matched = int(parcel["TOT_VAL"].notna().sum())
log(f"Joined {len(parcel):,} parcels | matched values {matched:,} "
    f"({100*matched/max(len(parcel),1):.1f}%)")
parcel["land_val"] = parcel["LAND_VAL"]
parcel["bld_val"] = parcel["IMPR_VAL"]
parcel["tot_appr_val"] = parcel["TOT_VAL"].where(parcel["TOT_VAL"] > 0,
                                                 parcel["LAND_VAL"] + parcel["IMPR_VAL"])
parcel["state_class"] = parcel["SPTBCODE"].astype(str).str.strip().str.upper()
parcel["mailto"] = parcel["TAXPANAME1"]

# ── 3. Authoritative city-limits filter (centroid within City of Dallas) ─────
if parcel.crs is None:
    parcel = parcel.set_crs("EPSG:4326")
elif parcel.crs.to_epsg() != 4326:
    parcel = parcel.to_crs("EPSG:4326")
rb = requests.get(BOUNDARY_URL, params={"where": "1=1", "outFields": "*",
                  "outSR": 4326, "f": "geojson"}, headers=HEADERS, timeout=120)
rb.raise_for_status()
boundary = unary_union(list(gpd.read_file(io.BytesIO(rb.content)).to_crs("EPSG:4326").geometry))
parcel["geometry"] = parcel["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
valid = parcel["geometry"].notnull() & parcel["geometry"].apply(lambda x: getattr(x, "is_valid", False))
cent = parcel.loc[valid, "geometry"].to_crs(3857).centroid.to_crs(4326)
inside = valid.copy()
inside[valid] = cent.within(boundary)
parcel = parcel[inside].copy()
log(f"City-limits filter -> {len(parcel):,}")

# ── 4. dedup, categorize, exempt, refined ────────────────────────────────────
# A single DCAD account is often split into many GIS polygons (corridors, multi-part
# parcels). The value join broadcasts the account's value onto every one of those
# polygons, so values are account-level and must be taken ONCE (first) — summing them
# multiplies by the polygon count (e.g. the UP railroad: 52 segments -> 52x $82.7M).
# Area, by contrast, IS per-polygon and must be summed to recover the whole parcel.
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
if ndup:
    value_cols = [c for c in ["tot_appr_val", "land_val", "bld_val"] if c in parcel.columns]
    area_cols = [c for c in ["AREA_FEET"] if c in parcel.columns]
    first_cols = [c for c in parcel.columns
                  if c not in set(value_cols + area_cols + ["geometry", "acct"])]
    agg = {c: "first" for c in value_cols}
    agg.update({c: "sum" for c in area_cols})
    agg.update({c: "first" for c in first_cols})
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs=parcel.crs)
log(f"After dedup -> {len(parcel):,}")


def categorize(v):
    """DCAD legacy state class (SPTBCODE) -> coarse property category.
    Mirrors the DCAD->PTAD crosswalk (A=SF, B=MF, C=vacant, D/E=ag/rural,
    F10=commercial, F20=industrial, G=mineral, J=utility, L/M/N/O/S=personal/inventory)."""
    raw = str(v or "").strip().upper()
    if not raw or raw == "NAN":
        return "Other"
    if raw.startswith("X"):
        return "Exempt"
    if raw.startswith("A"):
        return "Single Family"
    if raw.startswith("B"):
        return "Multifamily"
    if raw.startswith("C"):
        return "Vacant Residential"
    if raw.startswith("D") or raw.startswith("E"):
        return "Agricultural / Rural"
    if raw.startswith("F"):
        return "Industrial" if raw.startswith("F2") else "Commercial"
    if raw.startswith("G"):
        return "Mineral / Oil & Gas"
    if raw.startswith("J") or raw.startswith("U"):
        return "Utility"
    if raw[0] in ("L", "M", "N", "O", "S"):
        return "Personal Property / Inventory"
    return "Other"


parcel["PROPERTY_CATEGORY"] = parcel["state_class"].apply(categorize)
ex = parcel.copy()
ebs = ex["PROPERTY_CATEGORY"].isin(["Exempt"])
# GIS TOTEXEMPT carries a totally-exempt indicator/value when populated.
etot = ex.get("TOTEXEMPT", pd.Series("", index=ex.index)).astype(str).str.strip()
etot = etot.ne("") & etot.str.upper().ne("NAN") & etot.ne("0")
# BPP (business personal property) has no land footprint to map.
ebpp = ex.get("DIVISION_CD", pd.Series("", index=ex.index)).astype(str).str.upper().eq("BPP")
KW = ["CITY OF DALLAS", "DALLAS COUNTY", "STATE OF TEXAS", "DALLAS ISD", "DALLAS IND SCH",
      "DALLAS COLLEGE", "DALLAS CO COMMUNITY COLLEGE", "DALLAS COUNTY COMMUNITY",
      "PARKLAND", "DALLAS AREA RAPID TRANSIT", "DART", "DALLAS HOUSING",
      "HOUSING AUTHORITY", "UNITED STATES", "US GOVT", "U.S. GOVERNMENT",
      "DALLAS WATER", "UNIVERSITY OF TEXAS", "UNIV OF TEXAS"]
eown = ex["mailto"].astype(str).str.upper().str.contains("|".join(KW), na=False)
# Authoritative fully-exempt accounts from DCAD TOTAL_EXEMPTION.
ete = ex["acct"].isin(exempt_accts) if exempt_accts else pd.Series(False, index=ex.index)
ex["exemption_flag"] = (ebs | etot | ebpp | eown | ete).astype(int)
ex = ex[ex["exemption_flag"] == 0].copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex = ex[~ex["property_land_use_category"].isin(
    {"Mineral / Oil & Gas", "Personal Property / Inventory"})].copy()
ex["land_value"] = pd.to_numeric(ex.get("land_val", np.nan), errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex.get("bld_val", np.nan), errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(ex, fetch_footprints=False)
log(f"After exempt/refine -> {len(ex):,}")

# ── 5. Canonical fields — reported AREA_FEET denominator (GIS-area fallback) ──
def gis_area_sqft(geom):
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type == "Polygon":
        lon, lat = geom.exterior.coords.xy
        a, _ = geod.polygon_area_perimeter(lon, lat)
        return abs(a) * 10.763910416709722
    if geom.geom_type == "MultiPolygon":
        return sum(gis_area_sqft(p) for p in geom.geoms)
    return np.nan


ex["geometry"] = ex["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
log("Computing GIS areas...")
ex["gis_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["gis_area_sqft"] < 1, "gis_area_sqft"] = np.nan
ex["reported_sqft"] = pd.to_numeric(ex.get("AREA_FEET", np.nan), errors="coerce")
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

use_reported = ex["reported_sqft"] > 0
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["gis_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = pd.to_numeric(ex.get("tot_appr_val", np.nan), errors="coerce")
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

if "Website" in ex.columns and ex["Website"].notna().any():
    ex["link"] = ex["Website"].astype(str)
else:
    ex["link"] = "https://www.dallascad.org/AcctDetail.aspx?ID=" + ex["acct"].astype(str)

# ── 6. Export ────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant"]
for c in COLUMNS:
    if c not in ex.columns:
        ex[c] = np.nan
final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final["geometry"] = final["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
final = gpd.GeoDataFrame(final, geometry="geometry", crs=ex.crs)
if final.crs is None or final.crs.to_epsg() != 4326:
    final = final.to_crs("EPSG:4326")
out = DATA_DIR / "dallas-tx-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"dallas-tx-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"land_value_per_sqft: p50=${final['land_value_per_sqft'].median():.0f} "
    f"p99=${final['land_value_per_sqft'].quantile(.99):.0f} "
    f"p999=${final['land_value_per_sqft'].quantile(.999):.0f} "
    f"max=${final['land_value_per_sqft'].max():.0f}")
log("DONE")
