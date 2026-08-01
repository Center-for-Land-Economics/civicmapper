#!/usr/bin/env python3
"""
Build the City of Seattle, WA canonical parcel parquet.

Seattle is in King County. King County GIS publishes a single one-stop countywide
parcel polygon layer ("Parcels for King County with Address, Property and Ownership
Information") that carries geometry + assessor land/improvement value + present-use
code + lot size + a tax-value-reason flag, so this is fully automated (no manual
appraisal-roll join). The layer is countywide (~636k parcels), so we clip to the
City of Seattle municipal boundary (centroid-within). The parcel layer also carries a
geographic CTYNAME='Seattle' (~189k), but per playbook §4 we clip on the authoritative
boundary polygon rather than a situs-city field.

Sources (public, no token):
- Parcels (geometry + values + use + tax-value reason), King County GIS:
  https://services.arcgis.com/Ej0PsM5Aw677QF1W/arcgis/rest/services/PARCEL_ADDRESS_PUB_AREA_3069/FeatureServer/0
  Fields used: PIN, PROPTYPE, PREUSE_CODE, PREUSE_DESC, APPRLNDVAL, APPR_IMPR,
  LOTSQFT, KCA_ACRES, TAXVAL_RSN, CTYNAME.
- City of Seattle boundary, King County "Cities and unincorporated areas" polygon:
  https://gismaps.kingcounty.gov/arcgis/rest/services/Administration/KingCo_AdministrativeAreas/MapServer/2
  (NAME='Seattle')

Outputs:
- data/jurisidictions/data/seattle/seattle-wa-parcels.parquet
- data/jurisidictions/data/seattle/seattle-wa-parcels_YYYY_MM_DD.parquet

Notes:
- ~189k Seattle parcels -> large city -> PMTiles + H3 hexes (parquet_to_pmtiles.py).
- Condos: King County maps each condominium at the COMPLEX level (one footprint per
  building, LOTSQFT == the complex footprint, complex-total values), NOT as per-unit
  stubs stacked on a point. Verified ~0 stacked clusters -> NO condo stub-merge needed
  (unlike Thurston/Olympia). Townhouse-plat parcels are real fee-simple lots. The
  smoke-alarm diagnostics at the end confirm no pencils leaked in.
- Classification keys off PREUSE_DESC (Single Family / Townhouse / Duplex / Apartment /
  Condominium / Vacant(*) / Office / Retail / Warehouse / Industrial / Parking(*) ...).
- Exempt: TAXVAL_RSN in ('EX','OP') is the authoritative flag (~6.6k). Additionally drop
  inherently public / non-market PREUSE (Right of Way/Utility, Utility Public, Park Public,
  Reserve/Wilderness, Easement, Tideland, Water) which leak un-flagged. KEEP churches,
  private schools, nonprofits, group homes unless EX-flagged (taxable on otherwise normal
  parcels).
- $/sqft denominator is LOTSQFT (already sqft), fallback KCA_ACRES->sqft, fallback geodesic.
- link: King County Assessor eReal Property detail (?ParcelNbr=<PIN>).
"""
from __future__ import annotations

import io
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from datetime import datetime
from pathlib import Path
from shapely.ops import unary_union
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))          # so `import data.scripts.*` resolves
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "seattle"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# v2 cache: adds taxpayer + property-name fields (for the split-development merge below),
# so it is a distinct file from the original value-only geometry cache.
GEOM_CACHE = DATA_DIR / "seattle-wa-geometry-v2.parquet"

PARCELS_URL = ("https://services.arcgis.com/Ej0PsM5Aw677QF1W/arcgis/rest/services/"
               "PARCEL_ADDRESS_PUB_AREA_3069/FeatureServer/0/query")
BOUNDARY_URL = ("https://gismaps.kingcounty.gov/arcgis/rest/services/Administration/"
                "KingCo_AdministrativeAreas/MapServer/2/query")
OUT_FIELDS = ("PIN,PROPTYPE,PREUSE_CODE,PREUSE_DESC,APPRLNDVAL,APPR_IMPR,"
              "LOTSQFT,KCA_ACRES,TAXVAL_RSN,CTYNAME,PROP_NAME,KCTP_ATTN,KCTP_ADDR")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 1000   # King County FeatureServer maxRecordCount is 1000
geod = Geod(ellps="WGS84")

# Tax-value-reason codes that mark exempt / state-operating (utility) property to drop.
# Tax-relief reasons (FS senior, HI home-improvement, HP historic, CU current-use, NP, DP,
# MX) sit on otherwise-normal parcels and are intentionally NOT dropped.
EXEMPT_RSN = {"EX", "OP"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_boundary():
    """City of Seattle municipal boundary as a single (multi)polygon, EPSG:4326."""
    r = requests.get(BOUNDARY_URL, params={
        "where": "NAME='Seattle'", "outFields": "NAME", "returnGeometry": "true",
        "outSR": 4326, "f": "geojson"}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    bnd = gpd.read_file(io.BytesIO(r.content))
    if bnd.crs is None:
        bnd = bnd.set_crs(4326)
    elif bnd.crs.to_epsg() != 4326:
        bnd = bnd.to_crs(4326)
    poly = unary_union([g.buffer(0) for g in bnd.geometry if g is not None])
    log(f"Seattle boundary: {len(bnd)} feature(s); bbox {[round(x,5) for x in bnd.total_bounds]}")
    return poly, list(bnd.total_bounds)


def fetch_parcels(bbox):
    """Pull King County parcels intersecting the Seattle bbox (envelope filter)."""
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    minx, miny, maxx, maxy = bbox
    spatial = {
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
    }
    total = requests.get(PARCELS_URL, params={**spatial, "where": "1=1",
                         "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} King County parcels in Seattle bbox (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(5):
            try:
                r = requests.get(PARCELS_URL, params={
                    **spatial, "where": "1=1", "outFields": OUT_FIELDS,
                    "returnGeometry": "true", "resultOffset": off,
                    "resultRecordCount": PAGE, "outSR": 4326,
                    "orderByFields": "OBJECTID", "f": "geojson",
                }, headers=HEADERS, timeout=240)
                r.raise_for_status()
                gdf = gpd.read_file(io.BytesIO(r.content))
                break
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt+1} @off {off}: {type(e).__name__}: {e}")
                time.sleep(5 * (attempt + 1))
        if gdf is None:
            raise RuntimeError(f"Parcel pull failed at offset {off}")
        if not len(gdf):
            break
        pages.append(gdf)
        off += len(gdf)
        if off % 20000 < PAGE:
            log(f"  fetched {off:,}/{total:,}")
        # NOTE: the server caps each page at maxRecordCount (1000) regardless of the
        # requested PAGE, so a short page is NOT the last page — keep going until
        # `off >= total` or a page returns 0 rows (handled above).
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


boundary_poly, bbox = fetch_boundary()
geom = fetch_parcels(bbox)

# ── clip to City of Seattle (centroid within municipal boundary) ──────────────
if geom.crs is None:
    geom = geom.set_crs("EPSG:4326")
elif geom.crs.to_epsg() != 4326:
    geom = geom.to_crs("EPSG:4326")
geom["geometry"] = geom["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
geom = geom[geom["geometry"].notnull() & geom["geometry"].apply(
    lambda x: getattr(x, "is_valid", False) and not x.is_empty)].copy()
rep = geom.geometry.representative_point()
inside = rep.within(boundary_poly)
log(f"Centroid-in-Seattle clip: {int(inside.sum()):,} of {len(geom):,} bbox parcels")
parcel = geom[inside.values].copy()

# ── field cleanup ─────────────────────────────────────────────────────────────
parcel["acct"] = parcel["PIN"].astype(str).str.strip()
parcel = parcel[parcel["acct"].ne("") & parcel["acct"].ne("None") & parcel["acct"].ne("nan")]
for c in ["APPRLNDVAL", "APPR_IMPR", "LOTSQFT", "KCA_ACRES"]:
    parcel[c] = pd.to_numeric(parcel[c], errors="coerce")
for c in ["PROPTYPE", "PREUSE_DESC", "TAXVAL_RSN", "PROP_NAME", "KCTP_ATTN", "KCTP_ADDR"]:
    if c in parcel.columns:
        parcel[c] = parcel[c].astype(str).str.strip()
parcel["PREUSE_DESC"] = parcel["PREUSE_DESC"].replace({"nan": "", "None": ""})

parcel = parcel.rename(columns={"APPRLNDVAL": "land_val", "APPR_IMPR": "bld_val"})
parcel["tot_appr_val"] = parcel["land_val"].fillna(0) + parcel["bld_val"].fillna(0)
# reported lot area (sqft): LOTSQFT preferred, fall back to KCA_ACRES converted to sqft
acres_sqft = parcel["KCA_ACRES"] * SQFT_PER_ACRE
parcel["stated_sqft"] = parcel["LOTSQFT"].where(parcel["LOTSQFT"] > 0, acres_sqft)

# ── dedup multi-polygon parcels (values first — broadcast, never sum; geometry unioned) ──
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
log(f"Rows sharing a PIN (multi-polygon): {ndup:,}")
if ndup:
    first_cols = [c for c in parcel.columns if c not in ("geometry", "acct")]
    agg = {c: "first" for c in first_cols}
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
log(f"After PIN dedup -> {len(parcel):,}")

# ── exemption flag + classification ──────────────────────────────────────────
PUBLIC_PREUSE = (  # inherently public / non-market land that leaks past the EX flag
    "Right of Way", "Utility, Public", "Park, Public", "Reserve/Wilderness",
    "Easement", "Tideland", "River/Creek", "Water Body", "Open Space",
)


def is_public(desc):
    d = str(desc or "")
    return any(k in d for k in PUBLIC_PREUSE)


parcel["exemption_flag"] = (
    parcel["TAXVAL_RSN"].str.upper().isin(EXEMPT_RSN)
    | parcel["PREUSE_DESC"].apply(is_public)
).astype(int)


def categorize(desc):
    d = str(desc or "").strip()
    dl = d.lower()
    if not d:
        return "Other"
    if "vacant" in dl:
        return "Vacant Land"
    if "parking" in dl:
        return "Parking"
    if "condominium" in dl:
        return "Condominium"
    if "mobile home" in dl or "mhc" in dl:
        return "Mobile Home"
    if ("apartment" in dl or "duplex" in dl or "triplex" in dl or "plex" in dl
            or "rooming house" in dl or "fraternity" in dl or "retirement" in dl):
        return "Multifamily"
    if "single family" in dl or "townhouse" in dl or "residential" in dl:
        return "Single Family"
    if ("warehouse" in dl or "industrial" in dl or "manufacturing" in dl
            or "mining" in dl or "terminal" in dl):
        return "Industrial"
    if any(k in dl for k in (
            "office", "retail", "store", "restaurant", "lounge", "tavern", "hotel",
            "motel", "bank", "medical", "dental", "service", "commercial", "shopping",
            "car wash", "gas station", "auto", "bowling", "theater", "health club",
            "mortuary", "daycare", "marina", "conv store", "hospital", "nursing",
            "club", "art gallery", "governmental", "school", "church", "religious",
            "post office", "fire", "museum", "library")):
        return "Commercial"
    return "Other"


parcel["PROPERTY_CATEGORY"] = [categorize(d) for d in parcel["PREUSE_DESC"]]


def merge_split_developments(gdf, utm_epsg=32610, min_group_impr=100_000,
                             dominant_share=0.55, buffer_m=0.5):
    """Merge contiguous, same-taxpayer parcels where King County concentrated a
    multi-parcel development's ENTIRE improvement value onto one parcel (a "minor")
    and left its siblings at improvement=0.

    King County assesses multi-parcel economic units (apartment complexes, Amazon
    towers, hospitals, campuses) by dumping all the improvement value on a single
    minor — the sibling land/parking parcels carry $0 improvement (their PROP_NAME
    literally reads e.g. "GREEN LAKE VILLAGE (imp on Minor 1710)"). Left un-merged
    this renders as a lone "purple tower" of improvement value beside a ring of
    near-zero-improvement parcels (reported by a user, Green Lake Village, 2026-07).

    Fix: group parcels by same taxpayer (KCTP_ATTN+KCTP_ADDR) AND spatial contiguity
    (touching, small buffer to bridge topology gaps), then — only for groups that show
    the concentration signature (a dominant improvement parcel + >=1 near-zero-improvement
    sibling that owns land) — union the group into ONE polygon and SUM land+improvement.
    The lump then spreads over the true footprint (e.g. Building Cure $20.5k->$10.3k/sqft).

    Distinct same-owner developments on separate blocks (Green Lake Village vs The Eddy
    vs The Teel — all Wallace Properties) stay separate because they don't touch. Normal
    same-owner multi-building holdings (no zeroed siblings) are left untouched. Runs on
    the post-exempt taxable set, so public land (parks, Port, City Light) is already gone.
    """
    import re
    from shapely.strtree import STRtree

    def _own(attn, addr):
        a = re.sub(r"[^A-Z0-9 ]", " ", str(attn or "").upper())
        a = re.sub(r"\b(STE|SUITE|UNIT|APT|FL|FLOOR)\b", " ", a)
        a = re.sub(r"\s+", " ", a).strip()
        if not a:
            return ""   # no taxpayer name -> never group (individually-owned lots)
        b = re.sub(r"[^A-Z0-9 ]", " ", str(addr or "").upper())
        return a + " | " + re.sub(r"\s+", " ", b).strip()

    g = gdf.reset_index(drop=True).copy()
    g["_own"] = [_own(a, d) for a, d in zip(g.get("KCTP_ATTN"), g.get("KCTP_ADDR"))]
    g["_impr"] = pd.to_numeric(g["improvement_value"], errors="coerce").fillna(0.0)
    g["_land"] = pd.to_numeric(g["land_value"], errors="coerce").fillna(0.0)

    cand = g[g["_own"].ne("") & g.geometry.notna()]
    if len(cand) < 2:
        return gdf
    gp = cand.to_crs(utm_epsg)
    idx = list(cand.index)
    geoms = [geom.buffer(buffer_m) for geom in gp.geometry]
    tree = STRtree(geoms)
    owns = cand["_own"].values
    parent = list(range(len(idx)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, geom in enumerate(geoms):
        for j in tree.query(geom):
            j = int(j)
            if j <= i or owns[i] != owns[j] or not geom.intersects(geoms[j]):
                continue
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[ra] = rb

    comp = {}
    for k in range(len(idx)):
        comp.setdefault(find(k), []).append(idx[k])

    drop_ix, new_rows, n_merged = [], [], 0
    for members in comp.values():
        if len(members) < 2:
            continue
        sub = g.loc[members]
        tot_impr = float(sub["_impr"].sum())
        max_impr = float(sub["_impr"].max())
        near0 = sub["_impr"] <= max(1000.0, 0.02 * max_impr)
        if not (tot_impr > min_group_impr and near0.sum() >= 1
                and float(sub.loc[near0, "_land"].sum()) > 0
                and max_impr / max(tot_impr, 1.0) >= dominant_share):
            continue
        dom = sub.loc[(sub["_impr"] + sub["_land"]).idxmax()].copy()  # keeps PIN/link/category
        dom["geometry"] = unary_union([x for x in sub.geometry if x is not None])
        dom["land_value"] = float(sub["_land"].sum())
        dom["improvement_value"] = float(sub["_impr"].sum())
        # CRITICAL: the $/sqft denominator must be the WHOLE development's lot area, not just
        # the dominant minor's — otherwise the summed value over one small lot re-creates the
        # very tower we're removing. Sum the members' stated lot area (0 -> geometry fallback).
        if "stated_sqft" in sub.columns:
            dom["stated_sqft"] = float(pd.to_numeric(sub["stated_sqft"], errors="coerce").sum())
        for _c in ("LOTSQFT", "KCA_ACRES"):
            if _c in sub.columns:
                dom[_c] = float(pd.to_numeric(sub[_c], errors="coerce").sum())
        drop_ix.extend(members)
        new_rows.append(dom)
        n_merged += 1

    log(f"Split-development merge: {n_merged} groups absorbing {len(drop_ix)} parcels")
    if not new_rows:
        return gdf
    out = pd.concat([g.drop(index=drop_ix),
                     gpd.GeoDataFrame(new_rows, crs=g.crs)], ignore_index=True)
    return (gpd.GeoDataFrame(out, geometry="geometry", crs=g.crs)
            .drop(columns=["_own", "_impr", "_land"], errors="ignore"))


ex = parcel[parcel["exemption_flag"] == 0].copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")

# Merge King County's split multi-parcel developments BEFORE classify/area/per-sqft so all
# downstream fields (ratios, $/sqft, remnant flag, refined class) compute on the merged unit.
ex = merge_split_developments(ex)
ex["tot_appr_val"] = ex["land_value"].fillna(0) + ex["improvement_value"].fillna(0)

ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other",),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    fetch_footprints=False)
log(f"After exempt filter -> {len(ex):,} (dropped {int(parcel['exemption_flag'].sum()):,} exempt)")


# ── canonical fields — LOTSQFT denominator, geodesic fallback ─────────────────
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
ex["geom_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
ex["reported_sqft"] = pd.to_numeric(ex.get("stated_sqft", np.nan), errors="coerce")
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

use_reported = ex["reported_sqft"] > 0
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["geom_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = pd.to_numeric(ex.get("tot_appr_val", np.nan), errors="coerce")
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

# King County Assessor eReal Property detail page (ParcelNbr = 10-digit PIN).
ex["link"] = ("https://blue.kingcounty.com/Assessor/eRealProperty/Detail.aspx?ParcelNbr="
              + ex["acct"].astype(str))

# ── export ────────────────────────────────────────────────────────────────────
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
out = DATA_DIR / "seattle-wa-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"seattle-wa-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")

# ── smoke alarms (skill §6a): confirm no unit-stub pencils leaked in ──────────
gp = final.to_crs(32610)
a = gp.geometry.area * 10.7639
lv = pd.to_numeric(final["current_full_land_value"], errors="coerce") / (
    final["land_area_acres"] * SQFT_PER_ACRE).replace(0, np.nan)
holes = final.geometry.apply(lambda g: 0 if g is None else
    sum(len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
log(f"SMOKE footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01,.05,.10)]}")
log(f"SMOKE sub-500 / sub-1000 sqft footprints: {int((a<500).sum())} / {int((a<1000).sum())}")
log(f"SMOKE land $/sqft p50/p99/max: {round(lv.median())} / {round(lv.quantile(.99))} / {round(lv.max())}")
log(f"SMOKE parcels with holes: {int((holes>0).sum())}")
log("DONE")
