#!/usr/bin/env python3
"""
Build the combined DMV (DC / Maryland / Virginia) canonical parcel parquet.

The "DMV" is a SINGLE CivicMapper city (`dmv`) that stitches together many independent
jurisdictions across three states. Each parcel carries a `jurisdiction` column so the
frontend can offer a Houston-style region toggle that switches by jurisdiction
(CityDef.jurisdictionGroups -> field "jurisdiction"; the PMTiles bake auto-encodes it via
H3_CATEGORICAL_FIELDS). A WMATA Metrorail overlay is layered on top in the frontend
(viz/public/dmv-metro-*.geojson) — not part of this ETL.

Each jurisdiction is a self-contained *adapter* (fetch -> normalize -> canonical columns +
`jurisdiction`), cached to data/dmv/raw/<key>.parquet so re-runs skip the multi-minute pull.
The adapters concatenate into one parquet; per-jurisdiction quirks (schema, condo stacking,
exemption logic) stay inside each adapter.

Sources (all public ArcGIS REST, no token; verified reachable 2026-07-06):
  DC              maps2.dcgis.dc.gov  Property_and_Land/MapServer/40 (Owner Polygons; inline values)
  Montgomery MD   mdgeodata.md.gov    PlanningCadastre/MD_ParcelBoundaries/MapServer/0  JURSCODE='MONT'
  Prince Georges  mdgeodata.md.gov    PlanningCadastre/MD_ParcelBoundaries/MapServer/0  JURSCODE='PRIN'
  Fairfax Co VA   fairfaxcounty.gov   GIS/ParcelPlusAssessedValues/MapServer/0
  Arlington VA    (see va adapters)
  Alexandria VA   (see va adapters)
  Loudoun VA      (see va adapters)
  Prince William  (see va adapters)

Usage:
    python data/jurisidictions/run_dmv.py                 # all jurisdictions (cached where present)
    python data/jurisidictions/run_dmv.py --only dc mont  # subset by key
    python data/jurisidictions/run_dmv.py --scrape        # force fresh pulls
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.ops import unary_union
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "dmv"
RAW_DIR = DATA_DIR / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
}
SQFT_PER_ACRE = 43560.0
SQM_TO_SQFT = 10.763910416709722
GEOD = Geod(ellps="WGS84")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared ArcGIS paginated GeoJSON fetch (retry per page; don't stop on a short
# page unless it's shorter than the page size — the Seattle/DC 1000-cap trap).
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_page(query_url: str, where: str, out_fields: str, order_by: str,
                off: int, count: int, timeout: int, retries: int = 5) -> gpd.GeoDataFrame | None:
    """One page (offset..offset+count). Returns a GeoDataFrame, or None if all retries fail."""
    for attempt in range(retries):
        try:
            r = requests.get(query_url, params={
                "where": where, "outFields": out_fields, "returnGeometry": "true",
                "resultOffset": off, "resultRecordCount": count, "outSR": 4326,
                "orderByFields": order_by, "f": "geojson",
            }, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return gpd.read_file(io.BytesIO(r.content))
        except Exception as e:  # noqa: BLE001
            log(f"      retry {attempt+1} @off {off} n={count}: {type(e).__name__}: {str(e)[:90]}")
            time.sleep(3 * (attempt + 1))
    return None


def fetch_arcgis(query_url: str, out_fields: str, where: str = "1=1",
                 page: int = 1000, order_by: str = "OBJECTID",
                 timeout: int = 240) -> gpd.GeoDataFrame:
    total = requests.get(query_url, params={"where": where, "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=120).json().get("count")
    log(f"    total={total if total is not None else '?'} where={where!r}")
    pages, off = [], 0
    while True:
        gdf = _fetch_page(query_url, where, out_fields, order_by, off, page, timeout)
        if gdf is None:
            # A whole page keeps 500ing (often a server timeout on a large offset). Fall back to
            # smaller sub-chunks across the same window before giving up on the page.
            log(f"      page @off {off} failed; falling back to 200-row sub-chunks")
            subs, sub_off = [], off
            while sub_off < off + page:
                sub = _fetch_page(query_url, where, out_fields, order_by, sub_off, 200, timeout, retries=6)
                if sub is None:
                    raise RuntimeError(f"fetch failed at offset {sub_off} for {query_url}")
                if len(sub) == 0:
                    break
                subs.append(sub)
                sub_off += len(sub)
                if len(sub) < 200:
                    break
            gdf = gpd.GeoDataFrame(pd.concat(subs, ignore_index=True), crs="EPSG:4326") if subs \
                else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        n = len(gdf)
        if n == 0:
            break
        pages.append(gdf)
        off += n
        if total and off % (page * 20) < page:
            log(f"      fetched {off:,}/{total:,}")
        if n < page:
            break
    if not pages:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    g = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    log(f"    pulled {len(g):,} features")
    return g


def cached_fetch(key: str, fetch_fn, force: bool) -> gpd.GeoDataFrame:
    raw = RAW_DIR / f"{key}.parquet"
    if raw.exists() and not force:
        g = gpd.read_parquet(raw)
        log(f"  [{key}] cached raw: {len(g):,} rows")
        return g
    log(f"  [{key}] fetching fresh...")
    g = fetch_fn()
    g.to_parquet(raw, index=False)
    log(f"  [{key}] cached raw -> {raw.name} ({len(g):,} rows)")
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Geometry / area helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_valid(gs: gpd.GeoSeries) -> gpd.GeoSeries:
    return gs.apply(lambda g: g if (g is None or g.is_valid) else g.buffer(0))


def geodesic_area_sqft(geom) -> float:
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type == "Polygon":
        lon, lat = geom.exterior.coords.xy
        a, _ = GEOD.polygon_area_perimeter(lon, lat)
        return abs(a) * SQM_TO_SQFT
    if geom.geom_type == "MultiPolygon":
        return sum(geodesic_area_sqft(p) for p in geom.geoms)
    return np.nan


def collapse_stacked_footprints(parcel: gpd.GeoDataFrame, sum_cols: list[str]) -> gpd.GeoDataFrame:
    """Condo/units stacked on one footprint (>1 distinct row at the same representative point):
    SUM the value + area columns onto one unioned polygon. Matches run_newportnews.py."""
    rp = parcel.geometry.representative_point()
    parcel = parcel.copy()
    parcel["_rpkey"] = rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)
    vc = parcel["_rpkey"].value_counts()
    stacked = vc[vc > 1].index
    if len(stacked) == 0:
        return parcel.drop(columns=["_rpkey"])
    log(f"    stacked footprints: {len(stacked):,}; max stack {int(vc.max())}; "
        f"parcels involved {int(vc[vc > 1].sum()):,}")
    is_st = parcel["_rpkey"].isin(stacked)
    single = parcel[~is_st].copy()
    multi = parcel[is_st].copy()
    sum_cols = [c for c in sum_cols if c in multi.columns]
    first_cols = [c for c in multi.columns if c not in (["geometry", "_rpkey"] + sum_cols)]
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "first" for c in first_cols})
    coll = multi.groupby("_rpkey", dropna=False).agg(agg).reset_index()
    gu = multi.groupby("_rpkey", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]))
    coll["geometry"] = gu.values
    coll = gpd.GeoDataFrame(coll, geometry="geometry", crs=parcel.crs)
    out = gpd.GeoDataFrame(pd.concat([single, coll], ignore_index=True),
                           geometry="geometry", crs=parcel.crs)
    return out.drop(columns=["_rpkey"], errors="ignore")


def refined_category(cat: str, land: float, impr: float) -> str | None:
    c = str(cat or "")
    total = (land or 0) + (impr or 0)
    if "Vacant" in c:
        return "Vacant"
    if "Parking" in c:
        return "Parking Lot"
    if total > 0 and (impr or 0) == 0:
        return "Vacant"
    if total > 0 and (impr or 0) < 0.5 * total:
        return "Underdeveloped"
    return None


CANON_COLUMNS = [
    "geometry", "jurisdiction", "exemption_flag",
    "property_land_use_category", "property_land_use_refined",
    "full_market_value", "full_market_value_per_sqft",
    "current_full_land_value", "land_value_per_sqft",
    "improvement_value", "improvement_value_per_sqft",
    "TLLDIMPROV", "IMPR_LAND_RATIO", "IMPR_LAND_PCT", "IMPR_PCT_TOTAL",
    "link", "land_area_acres", "likely_remnant",
]


def finalize(gdf: gpd.GeoDataFrame, jurisdiction: str) -> gpd.GeoDataFrame:
    """Given an adapter frame with columns: geometry, land_value, improvement_value,
    [full_market_value], property_land_use_category, exemption_flag, [link],
    [land_area_sqft] — compute per-sqft / ratio / refined / remnant and return CANON_COLUMNS.
    Exempt parcels (exemption_flag==1) are dropped here. Area falls back to geodesic if the
    adapter didn't supply land_area_sqft."""
    g = gdf.copy()
    g["jurisdiction"] = jurisdiction
    g["geometry"] = make_valid(g.geometry)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()

    # drop exempt
    before = len(g)
    if "exemption_flag" not in g.columns:
        g["exemption_flag"] = 0
    g["exemption_flag"] = pd.to_numeric(g["exemption_flag"], errors="coerce").fillna(0).astype(int)
    g = g[g["exemption_flag"] == 0].copy()

    land = pd.to_numeric(g.get("land_value"), errors="coerce")
    impr = pd.to_numeric(g.get("improvement_value"), errors="coerce")
    if "full_market_value" in g.columns:
        total = pd.to_numeric(g["full_market_value"], errors="coerce")
        total = total.fillna(land.fillna(0) + impr.fillna(0))
    else:
        total = land.fillna(0) + impr.fillna(0)
    g["land_value"] = land
    g["improvement_value"] = impr
    g["full_market_value"] = total

    # area — prefer the adapter's stated land area; fall back to geodesic polygon area.
    # (Build geodesic area as a plain float Series; GeoSeries.apply can return geometry dtype.)
    geom_area = pd.Series([geodesic_area_sqft(x) for x in g.geometry], index=g.index, dtype="float64")
    if "land_area_sqft" in g.columns:
        stated = pd.to_numeric(g["land_area_sqft"], errors="coerce")
        area = stated.where(stated >= 1, geom_area)
    else:
        area = geom_area
    area = area.where(area >= 1, np.nan)
    g["land_area_sqft"] = area
    g["land_area_acres"] = area / SQFT_PER_ACRE
    g["likely_remnant"] = (area < 500).astype(int)

    den = area.replace(0, np.nan)
    g["full_market_value_per_sqft"] = g["full_market_value"] / den
    g["land_value_per_sqft"] = g["land_value"] / den
    g["improvement_value_per_sqft"] = g["improvement_value"] / den
    g = add_improvement_ratio_fields(g, land_col="land_value", improvement_col="improvement_value")

    if "property_land_use_category" not in g.columns:
        g["property_land_use_category"] = "Other"
    g["property_land_use_refined"] = [
        refined_category(c, l, i) for c, l, i in
        zip(g["property_land_use_category"], g["land_value"].fillna(0), g["improvement_value"].fillna(0))
    ]
    if "link" not in g.columns:
        g["link"] = np.nan

    g = g.rename(columns={"land_value": "current_full_land_value"})
    for c in CANON_COLUMNS:
        if c not in g.columns:
            g[c] = np.nan
    out = gpd.GeoDataFrame(g[CANON_COLUMNS], geometry="geometry", crs=g.crs)
    if out.crs is None or out.crs.to_epsg() != 4326:
        out = out.to_crs("EPSG:4326")
    log(f"    [{jurisdiction}] finalized {len(out):,} (dropped {before - len(g):,} exempt); "
        f"land $/sqft p50={out['land_value_per_sqft'].median():.0f} "
        f"p99={out['land_value_per_sqft'].quantile(.99):.0f}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# ADAPTERS — one per jurisdiction. Each returns an adapter-frame (see finalize()).
# ═════════════════════════════════════════════════════════════════════════════

# ── Washington, DC ────────────────────────────────────────────────────────────
DC_URL = ("https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
          "Property_and_Land/MapServer/40/query")
DC_FIELDS = ("SSL,PROPTYPE,USECODE,CLASSTYPE,OWNERNAME,LANDAREA,"
             "NEWLAND,NEWIMPR,NEWTOTAL,PREMISEADD")


def dc_categorize(proptype: str) -> str:
    p = str(proptype or "").upper()
    if not p or p == "NONE":
        return "Other"
    if "VACANT" in p:
        return "Vacant Land"
    if "PARKING" in p:
        return "Parking"
    if "SINGLE FAMILY" in p or "ROW" in p or "SEMI-DETACHED" in p or "TOWN" in p:
        return "Single Family"
    if "CONDOMINIUM" in p or "CONDO" in p:
        return "Condominium"
    if "FLAT" in p or "APARTMENT" in p or "MULTIFAMILY" in p or "WALK UP" in p:
        return "Multifamily"
    if "COMMERCIAL" in p or "OFFICE" in p or "RETAIL" in p or "HOTEL" in p or "STORE" in p:
        return "Commercial"
    if "INDUSTRIAL" in p or "WAREHOUSE" in p:
        return "Industrial"
    if "EXEMPT" in p or "PUBLIC" in p:
        return "Exempt / Governmental"
    return "Other"


def adapter_dc(force: bool) -> gpd.GeoDataFrame:
    g = cached_fetch("dc", lambda: fetch_arcgis(DC_URL, DC_FIELDS, page=1000), force)
    for c in ["NEWLAND", "NEWIMPR", "NEWTOTAL", "LANDAREA"]:
        g[c] = pd.to_numeric(g.get(c), errors="coerce")
    g["property_land_use_category"] = g["PROPTYPE"].apply(dc_categorize)
    # Exempt: PROPTYPE-based flag; the zero-value drop below removes federal/ROW/reservation land.
    g["exemption_flag"] = (g["property_land_use_category"] == "Exempt / Governmental").astype(int)
    g["land_value"] = g["NEWLAND"]
    g["improvement_value"] = g["NEWIMPR"]
    g["full_market_value"] = g["NEWTOTAL"]
    g["land_area_sqft"] = g["LANDAREA"].where(g["LANDAREA"] >= 1, np.nan)
    ssl = g["SSL"].astype(str).str.strip()
    g["link"] = "https://taxpayerservicecenter.com/RP_Detail.jsp?ssl=" + ssl.str.replace(" ", "+")
    # drop rows with no value at all (federal/ROW) — keep only assessed parcels
    g = g[(g["NEWLAND"].fillna(0) > 0) | (g["NEWIMPR"].fillna(0) > 0)].copy()
    return finalize(g, "Washington, DC")


# ── Maryland (statewide SDAT parcels, filtered per county) ────────────────────
MD_URL = ("https://mdgeodata.md.gov/imap/rest/services/PlanningCadastre/"
          "MD_ParcelBoundaries/MapServer/0/query")
MD_FIELDS = "ACCTID,NFMLNDVL,NFMIMPVL,NFMTTLVL,LU,DESCLU,EXCLASS,DESCEXCL,ACRES,SDATWEBADR,JURSCODE"


def md_categorize(desclu: str) -> str:
    d = str(desclu or "").strip().upper()
    if not d or d == "NAN" or d == "NONE":
        return "Other"
    if "EXEMPT" in d:
        return "Exempt / Governmental"
    if "VACANT" in d:
        return "Vacant Land"
    if "AGRICULT" in d:
        return "Agricultural"
    if "APARTMENT" in d:
        return "Multifamily"
    if "CONDO" in d:
        return "Condominium"
    if "COUNTRY CLUB" in d:
        return "Other"
    if "COMMERCIAL" in d:
        return "Commercial"
    if "INDUSTRIAL" in d:
        return "Industrial"
    if "RESIDENTIAL" in d:
        return "Residential"
    return "Other"


def _adapter_md_county(key: str, jurscode: str, label: str, force: bool) -> gpd.GeoDataFrame:
    g = cached_fetch(key, lambda: fetch_arcgis(MD_URL, MD_FIELDS, where=f"JURSCODE='{jurscode}'",
                                               page=1000, order_by="ACCTID"), force)
    for c in ["NFMLNDVL", "NFMIMPVL", "NFMTTLVL", "ACRES"]:
        g[c] = pd.to_numeric(g.get(c), errors="coerce")
    # keep only parcels with assessment
    has = g["NFMTTLVL"].notna() | g["DESCLU"].notna()
    g = g[has].copy()
    # condo-cure: collapse duplicate ACCTID (multi-polygon parcels), values first
    if "ACCTID" in g.columns and g.duplicated(subset=["ACCTID"], keep=False).any():
        num = [c for c in ["NFMLNDVL", "NFMIMPVL", "NFMTTLVL", "ACRES"] if c in g.columns]
        cat = [c for c in g.columns if c not in set(num + ["geometry", "ACCTID"])]
        agg = {c: "first" for c in cat}
        agg.update({c: "first" for c in num})  # account-level values broadcast; take first not sum
        coll = g.groupby("ACCTID", dropna=False).agg(agg).reset_index()
        gu = g.groupby("ACCTID", dropna=False)["geometry"].apply(
            lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
        coll["geometry"] = gu.values
        g = gpd.GeoDataFrame(coll, geometry="geometry", crs=g.crs)
        log(f"    [{key}] after ACCTID dedup: {len(g):,}")
    g["property_land_use_category"] = g["DESCLU"].apply(md_categorize)
    exclass = g.get("EXCLASS")
    exclass = exclass.fillna("").astype(str).str.strip() if exclass is not None else pd.Series("", index=g.index)
    g["exemption_flag"] = ((g["property_land_use_category"] == "Exempt / Governmental")
                           | (exclass.ne("") & exclass.str.upper().ne("NAN"))).astype(int)
    g["land_value"] = g["NFMLNDVL"]
    g["improvement_value"] = g["NFMIMPVL"]
    g["full_market_value"] = g["NFMTTLVL"]
    g["land_area_sqft"] = (g["ACRES"] * SQFT_PER_ACRE).where(g["ACRES"] > 0, np.nan)
    link = g.get("SDATWEBADR")
    g["link"] = link.astype(str).str.strip().replace({"nan": np.nan, "": np.nan}) if link is not None else np.nan
    return finalize(g, label)


def adapter_montgomery(force: bool) -> gpd.GeoDataFrame:
    return _adapter_md_county("mont", "MONT", "Montgomery County, MD", force)


def adapter_princegeorges(force: bool) -> gpd.GeoDataFrame:
    return _adapter_md_county("prin", "PRIN", "Prince George's County, MD", force)


# ── Fairfax County, VA ────────────────────────────────────────────────────────
FFX_URL = ("https://www.fairfaxcounty.gov/mercator/rest/services/GIS/"
           "ParcelPlusAssessedValues/MapServer/0/query")
FFX_FIELDS = "PIN,LUC,APRLAND,APRBLDG,APRTOT,FLAG4_DESC"


def ffx_categorize(luc: str) -> str:
    c = str(luc or "").strip()
    if not c:
        return "Other"
    # Fairfax LUC bands (no server domain; 0xx=residential dominates ~94%).
    if c in ("041", "042", "043", "045", "047"):
        return "Condominium"
    if c.startswith("0"):   # 011 SFD, 021/03x townhouse/attached, 09x etc. -> residential
        return "Single Family"
    if c.startswith("1"):   # 1xx multifamily / apartments
        return "Multifamily"
    if c.startswith(("2", "3", "4", "5")):  # commercial / industrial
        return "Commercial"
    if c.startswith("9"):   # 9xx common-area / exempt / residual
        return "Exempt / Governmental"
    return "Other"          # 6xx-8xx misc


def adapter_fairfax(force: bool) -> gpd.GeoDataFrame:
    g = cached_fetch("ffx", lambda: fetch_arcgis(FFX_URL, FFX_FIELDS, page=2000), force)
    for c in ["APRLAND", "APRBLDG", "APRTOT"]:
        g[c] = pd.to_numeric(g.get(c), errors="coerce")
    g["property_land_use_category"] = g["LUC"].apply(ffx_categorize)
    # Exemption is authoritative: FLAG4_DESC is exactly "Tax Exempt" vs "No Exemption".
    # (Do NOT substring-match "EXEMPT" — it also hits "No Exemption".) Zero-value common
    # areas are removed by the APRTOT>0 filter below.
    desc = g.get("FLAG4_DESC").fillna("").astype(str) if "FLAG4_DESC" in g.columns else pd.Series("", index=g.index)
    g["exemption_flag"] = (desc.str.strip().str.upper() == "TAX EXEMPT").astype(int)
    g["land_value"] = g["APRLAND"]
    g["improvement_value"] = g["APRBLDG"]
    g["full_market_value"] = g["APRTOT"]
    pin = g["PIN"].astype(str).str.strip()
    g["link"] = "https://icare.fairfaxcounty.gov/ffxcare/search/commonsearch.aspx?mode=realprop&pin=" + pin.str.replace(" ", "+")
    g = g[(g["APRTOT"].fillna(0) > 0)].copy()
    # CONDO STACKING: units share identical geometry each with full value -> merge stacked
    # footprints, SUMMING land+bldg (and total), one polygon out. (Newport News pattern.)
    g = collapse_stacked_footprints(g, sum_cols=["land_value", "improvement_value", "full_market_value"])
    return finalize(g, "Fairfax County, VA")


# ── Arlington County, VA ──────────────────────────────────────────────────────
ARL_URL = ("https://arlgis.arlingtonva.us/arcgis/rest/services/StaffMap/"
           "Property_Map_public/MapServer/3/query")
ARL_FIELDS = "RPCMSTR,LRSN,LAND,IMPROVEMENT,TOTAL,PROPERTY_CLASS_DESC,LOTSIZE,tax_exemption_type_dsc"


def va_class_categorize(desc: str) -> str:
    d = str(desc or "").upper()
    if not d or d == "NAN" or d == "NONE":
        return "Other"
    if "VACANT" in d:
        return "Vacant Land"
    if "PARKING" in d:
        return "Parking"
    if "SINGLE FAMILY" in d or "TOWNHOUSE" in d or "TOWN HOUSE" in d or "DUPLEX" in d:
        return "Single Family"
    if "CONDO" in d:
        return "Condominium"
    if "APARTMENT" in d or "MULTI" in d or "GARDEN" in d:
        return "Multifamily"
    if "COMMERCIAL" in d or "OFFICE" in d or "RETAIL" in d or "HOTEL" in d or "STORE" in d:
        return "Commercial"
    if "INDUSTRIAL" in d or "WAREHOUSE" in d:
        return "Industrial"
    if "EXEMPT" in d or "PUBLIC" in d:
        return "Exempt / Governmental"
    return "Other"


def adapter_arlington(force: bool) -> gpd.GeoDataFrame:
    g = cached_fetch("arl", lambda: fetch_arcgis(ARL_URL, ARL_FIELDS, page=2000), force)
    for c in ["LAND", "IMPROVEMENT", "TOTAL", "LOTSIZE"]:
        g[c] = pd.to_numeric(g.get(c), errors="coerce")
    g["property_land_use_category"] = g["PROPERTY_CLASS_DESC"].apply(va_class_categorize)
    exd = g.get("tax_exemption_type_dsc")
    exd = exd.fillna("").astype(str).str.strip() if exd is not None else pd.Series("", index=g.index)
    g["exemption_flag"] = ((g["property_land_use_category"] == "Exempt / Governmental")
                           | (exd.ne("") & exd.str.upper().ne("NAN"))).astype(int)
    g["land_value"] = g["LAND"]
    g["improvement_value"] = g["IMPROVEMENT"]
    g["full_market_value"] = g["TOTAL"]
    g["land_area_sqft"] = g["LOTSIZE"].where(g["LOTSIZE"] >= 1, np.nan)
    lrsn = g["LRSN"].astype(str).str.strip()
    g["link"] = "https://propertysearch.arlingtonva.us/Home/Assessments?lrsn=" + lrsn
    g = g[(g["TOTAL"].fillna(0) > 0)].copy()
    # Arlington maps condos individually — collapse stacked footprints (sum values + area).
    g = collapse_stacked_footprints(g, sum_cols=["land_value", "improvement_value", "full_market_value"])
    return finalize(g, "Arlington County, VA")


# ── City of Fairfax, VA (independent city; own AGOL org, NOT Fairfax County) ──
FCITY_URL = ("https://services2.arcgis.com/DANcyjLcCCpGk8Ri/arcgis/rest/services/"
             "GISCAMA/FeatureServer/0/query")
FCITY_FIELDS = "ParcelID,CurrentLan,CurrentBui,CurrentTot,ELU,TotalLand"


def adapter_fairfaxcity(force: bool) -> gpd.GeoDataFrame:
    g = cached_fetch("fairfaxcity", lambda: fetch_arcgis(FCITY_URL, FCITY_FIELDS, page=2000), force)
    for c in ["CurrentLan", "CurrentBui", "CurrentTot", "TotalLand"]:
        g[c] = pd.to_numeric(g.get(c), errors="coerce")
    g["property_land_use_category"] = g["ELU"].apply(va_class_categorize)
    g["exemption_flag"] = (g["property_land_use_category"] == "Exempt / Governmental").astype(int)
    g["land_value"] = g["CurrentLan"]
    g["improvement_value"] = g["CurrentBui"]
    g["full_market_value"] = g["CurrentTot"]
    g["land_area_sqft"] = (g["TotalLand"] * SQFT_PER_ACRE).where(g["TotalLand"] > 0, np.nan)  # TotalLand = acres
    g["link"] = np.nan
    g = g[(g["CurrentTot"].fillna(0) > 0)].copy()
    g = collapse_stacked_footprints(g, sum_cols=["land_value", "improvement_value", "full_market_value"])
    return finalize(g, "City of Fairfax, VA")


# ═════════════════════════════════════════════════════════════════════════════
# BLOCKED (values not in open GIS — need out-of-band CAMA extract / FOIA):
#   Alexandria, Falls Church, Loudoun, Prince William. Geometry+owner+landuse are
#   available but land/improvement dollar values live behind per-parcel WAF'd
#   lookup apps only. Not shippable for a land-value map without the value join.
# ═════════════════════════════════════════════════════════════════════════════

ADAPTERS: dict[str, callable] = {
    "dc": adapter_dc,
    "mont": adapter_montgomery,
    "prin": adapter_princegeorges,
    "ffx": adapter_fairfax,
    "arl": adapter_arlington,
    "fairfaxcity": adapter_fairfaxcity,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of jurisdiction keys")
    ap.add_argument("--scrape", action="store_true", help="force fresh pulls (ignore raw cache)")
    args = ap.parse_args()

    keys = args.only if args.only else list(ADAPTERS.keys())
    unknown = [k for k in keys if k not in ADAPTERS]
    if unknown:
        raise SystemExit(f"Unknown keys {unknown}; available: {list(ADAPTERS)}")

    frames, failed = [], []
    for k in keys:
        log(f"=== {k} ===")
        try:
            frames.append(ADAPTERS[k](args.scrape))
        except Exception as e:  # noqa: BLE001
            # A single jurisdiction's transient source failure must NOT lose the others' work.
            log(f"  !! {k} FAILED (skipping): {type(e).__name__}: {e}")
            failed.append(k)
    if not frames:
        raise SystemExit(f"All jurisdictions failed: {failed}")
    if failed:
        log(f"⚠️  FAILED jurisdictions (re-run to include): {failed}")
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                geometry="geometry", crs="EPSG:4326")
    out = DATA_DIR / "dmv-dc-parcels.parquet"
    combined.to_parquet(out, index=False)
    log(f"SAVED {out} | {len(combined):,} parcels")
    log("Per-jurisdiction counts:")
    print(combined["jurisdiction"].value_counts().to_string())
    log("Refined categories:")
    print(combined["property_land_use_refined"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
