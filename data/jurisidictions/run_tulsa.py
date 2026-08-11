#!/usr/bin/env python3
"""
Build the City of Tulsa, OK canonical parcel parquet.

Fully automated from the INCOG-hosted Tulsa County Assessor parcel layer (geometry +
assessor land/improvement/total values + post-cap taxable value + use/type/account-type
classification + homestead flag + area, all in one layer), filtered to SiteCity='TULSA'.

Source (Indian Nations Council of Governments — Tulsa-area regional planning agency that
publishes the Tulsa County Assessor parcel base):
- Parcels_TulsaCo FeatureServer/0 (PACS-style assessor schema):
  https://map11.incog.org/arcgis11wa/rest/services/Parcels_TulsaCo/FeatureServer/0
  Pulled (where SiteCity='TULSA', ~157k parcels) + cached on first run; reused after.

Outputs:
- data/jurisidictions/data/tulsa/tulsa-ok-parcels.parquet
- data/jurisidictions/data/tulsa/tulsa-ok-parcels_YYYY_MM_DD.parquet

Notes:
- Jurisdiction filter is the assessor's SiteCity (the property's situs city), not the owner
  mailing City — SiteCity='TULSA' is the City of Tulsa proper. (Per the data owner's guidance.)
- Oklahoma has no Texas-SPTB state class; category is derived from AcctType + PropertyType +
  UseCode (PARKING use code -> Parking Lot; Mfg/Warehouse/Flex/Storage/Hangar -> Industrial).
- Exempt parcels are caught by AcctType ('Exempt', 'Exempt Com', 'Exempt Res', 'Exempt Ag').
  'Partial Exempt' is kept (it carries taxable value). HomesteadExemption is a residential
  homestead exemption, NOT a full exemption -> those parcels are kept.
- A single account (AccountNo) split into multiple GIS polygons keeps account-level values and
  reported area ONCE (first); geometry is unioned. (No summing — the Dallas N× value bug.)
- Condos: per-unit accounts can stack on one footprint. The diagnostic below reports stacking;
  units sharing a footprint (>1 distinct account at one representative point) are collapsed into
  a single footprint with SUMMED values (different units = different value) so the H3 hex
  $/sqft isn't deflated by N-times-counted overlapping area.
- $/sqft denominator is the assessor's reported GrossSF (fallback GrossAcre, then geodesic
  polygon area). Emits land_area_acres, area_source, likely_remnant (<500 sqft).

PMTiles bake (Tulsa is ~150k parcels -> PMTiles + H3 hexes):
    python data/scripts/parquet_to_pmtiles.py --city tulsa \
      --file data/jurisidictions/data/tulsa/tulsa-ok-parcels.parquet --upload --overwrite
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
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "tulsa"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "tulsa-ok-geometry.parquet"

PARCELS_URL = ("https://map11.incog.org/arcgis11wa/rest/services/"
               "Parcels_TulsaCo/FeatureServer/0/query")
WHERE = "SiteCity='TULSA'"
OUT_FIELDS = ("AccountNo,ACCT_NUM,AcctType,PropertyType,UseCode,OccCode,HomesteadExemption,"
              "Owner,Name1,TotalLandValue,TotalImpValue,TotalAcctValue,TaxableValue,"
              "GrossSF,GrossAcre,ImpSFTotal,SiteCity,PropertyAddress")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
geod = Geod(ellps="WGS84")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── 1. Parcels (geometry + values + class + area), cached ─────────────────────
def fetch_parcels():
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    total = requests.get(PARCELS_URL, params={"where": WHERE, "returnCountOnly": "true",
                         "f": "json"}, headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} City of Tulsa parcels (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(5):
            try:
                r = requests.get(PARCELS_URL, params={
                    "where": WHERE, "outFields": OUT_FIELDS, "returnGeometry": "true",
                    "resultOffset": off, "resultRecordCount": PAGE, "outSR": 4326,
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
        if len(gdf) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


geom = fetch_parcels()
geom["acct"] = geom["AccountNo"].astype(str).str.strip()
geom = geom[geom["acct"].ne("") & geom["acct"].ne("None") & geom["acct"].ne("nan")]
for c in ["TotalLandValue", "TotalImpValue", "TotalAcctValue", "TaxableValue",
          "GrossSF", "GrossAcre", "ImpSFTotal"]:
    geom[c] = pd.to_numeric(geom[c], errors="coerce")

parcel = geom.rename(columns={"TotalLandValue": "land_val", "TotalImpValue": "bld_val",
                              "GrossSF": "rep_sqft", "ImpSFTotal": "imp_sf"})
parcel["tot_appr_val"] = parcel["TotalAcctValue"].where(
    parcel["TotalAcctValue"] > 0, parcel["land_val"].fillna(0) + parcel["bld_val"].fillna(0))
for c in ["AcctType", "PropertyType", "UseCode"]:
    parcel[c] = parcel[c].astype(str).str.strip()

if parcel.crs is None:
    parcel = parcel.set_crs("EPSG:4326")
elif parcel.crs.to_epsg() != 4326:
    parcel = parcel.to_crs("EPSG:4326")
parcel["geometry"] = parcel["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
parcel = parcel[parcel["geometry"].notnull() & parcel["geometry"].apply(
    lambda x: getattr(x, "is_valid", False) and not x.is_empty)].copy()
log(f"Valid-geometry parcels (SiteCity='TULSA') -> {len(parcel):,}")

# ── 2. dedup multi-polygon accounts (account-level values + area first; geom unioned) ──
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
log(f"Rows sharing an AccountNo (multi-polygon accounts): {ndup:,}")
if ndup:
    first_cols = [c for c in parcel.columns if c not in ("geometry", "acct")]
    agg = {c: "first" for c in first_cols}
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
log(f"After account dedup -> {len(parcel):,} unique accounts")

# ── 2b. Condo diagnostic + same-footprint collapse ───────────────────────────
# Per-unit condo accounts can sit stacked on one footprint. Detect groups of >1 DISTINCT
# account sharing a representative point; collapse them into one footprint with SUMMED values
# (distinct units = distinct value, unlike the account-split case above which is `first`).
rp = parcel.geometry.representative_point()
parcel["_rpkey"] = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str))
vc = parcel["_rpkey"].value_counts()
stacked_keys = vc[vc > 1].index
log(f"Stacked footprints (>1 account at one point): {len(stacked_keys):,}; "
    f"max stack: {int(vc.max())}; parcels involved: {int(vc[vc > 1].sum()):,}")
if len(stacked_keys):
    is_stacked = parcel["_rpkey"].isin(stacked_keys)
    single = parcel[~is_stacked].copy()
    single["_collapsed"] = 0
    multi = parcel[is_stacked].copy()
    # Condo units are PER-UNIT shares of one shared parcel, so sum the values across the stack.
    # rep_sqft (reported land area) is ALSO summed — but it is unreliable as the land denominator:
    # most Tulsa condos report per-unit GrossSF shares that sum to the footprint, but some report a
    # tiny nominal GrossSF with the real footprint only in geometry. The shared-footprint land area
    # is reconciled later as max(summed GrossSF, union-polygon geodesic area) for `_collapsed` rows.
    sum_cols = ["land_val", "bld_val", "tot_appr_val", "TotalAcctValue", "imp_sf", "rep_sqft"]
    first_cols = [c for c in multi.columns if c not in (["geometry", "_rpkey"] + sum_cols)]
    agg = {c: "sum" for c in sum_cols if c in multi.columns}
    agg.update({c: "first" for c in first_cols})
    coll = multi.groupby("_rpkey", dropna=False).agg(agg).reset_index()
    gu = multi.groupby("_rpkey", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]))
    coll["geometry"] = gu.values
    coll["_collapsed"] = 1
    coll = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
    parcel = pd.concat([single, coll], ignore_index=True)
    parcel = gpd.GeoDataFrame(parcel, geometry="geometry", crs="EPSG:4326")
else:
    parcel["_collapsed"] = 0
parcel = parcel.drop(columns=["_rpkey"], errors="ignore")
log(f"After condo footprint collapse -> {len(parcel):,}")

# ── 3. Exemption flag (AcctType) + classification ────────────────────────────
EXEMPT_TYPES = {"Exempt", "Exempt Com", "Exempt Res", "Exempt Ag"}
parcel["exemption_flag"] = parcel["AcctType"].isin(EXEMPT_TYPES).astype(int)

IND_USE = {"MFG", "WAREHOUSE", "FLEX", "STORAGE", "STORAGE/WH", "SELF STG", "HANGAR"}
MF_USE = {"APARTMENTS", "MULTI UNIT", "TOWNHOME", "MHP", "LIHTC", "LIHTC-S"}


def categorize(at, pt, uc):
    at = str(at or "").strip()
    pt = str(pt or "").strip()
    uc = str(uc or "").strip().upper()
    if uc == "PARKING":
        return "Parking"
    if (at.startswith("Commercial") or at in ("Comm Res", "Comm Ag")
            or pt in ("Commercial", "Industrial")):
        if pt == "Industrial" or uc in IND_USE:
            return "Industrial"
        return "Commercial"
    if pt == "Condo":
        return "Condominium"
    if pt in ("Duplex", "Triplex", "Multiple Unit", "Townhouse") or uc in MF_USE:
        return "Multifamily"
    if pt == "Mobile Home":
        return "Mobile Home"
    if at == "Agricultural" or pt == "Agricultural" or uc == "AGRI":
        return "Agricultural / Rural"
    if pt == "Residential" or at == "Residential":
        return "Single Family"
    return "Other"


parcel["PROPERTY_CATEGORY"] = [categorize(a, p, u) for a, p, u in
                               zip(parcel["AcctType"], parcel["PropertyType"], parcel["UseCode"])]

ex = parcel[parcel["exemption_flag"] == 0].copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Agricultural / Rural", "Other"),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    bld_ar_col="imp_sf", fetch_footprints=False)
log(f"After exempt filter -> {len(ex):,} (dropped {int(parcel['exemption_flag'].sum()):,} exempt)")

# ── 4. Canonical fields — reported GrossSF denominator (geodesic fallback) ───
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
rep = pd.to_numeric(ex.get("rep_sqft", np.nan), errors="coerce")
rep = rep.where(rep > 0, pd.to_numeric(ex.get("GrossAcre", np.nan), errors="coerce") * SQFT_PER_ACRE)
ex["reported_sqft"] = rep
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

# Collapsed condo stacks: the land denominator must be the SHARED complex footprint, not a single
# unit's reported area (summing unit land values while dividing by one unit's area inflated $/sqft
# by the unit count — the R70985830752319 / 7141 S Quincy case). Use the larger of the summed
# per-unit GrossSF and the union polygon's geodesic area: that equals the real footprint whether the
# source reports per-unit area shares (sum≈polygon) or a nominal per-unit area (polygon is truth);
# degenerate sub-500-sqft sliver footprints fall through to likely_remnant and are dropped.
col = ex["_collapsed"] == 1
ex.loc[col, "reported_sqft"] = np.maximum(
    pd.to_numeric(ex.loc[col, "rep_sqft"], errors="coerce").fillna(0.0),
    pd.to_numeric(ex.loc[col, "geom_area_sqft"], errors="coerce").fillna(0.0))
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

# Tulsa County Assessor detail page (AccountNo like "R99201920105170").
ex["link"] = "https://assessor.tulsacounty.org/Property/Info?accountNo=" + ex["acct"].astype(str)

# ── 5. Export ────────────────────────────────────────────────────────────────
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
out = DATA_DIR / "tulsa-ok-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"tulsa-ok-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")
log(f"land_value_per_sqft: p50=${final['land_value_per_sqft'].median():.0f} "
    f"p99=${final['land_value_per_sqft'].quantile(.99):.0f} "
    f"p999=${final['land_value_per_sqft'].quantile(.999):.0f} "
    f"max=${final['land_value_per_sqft'].max():.0f}")
log("DONE")
