#!/usr/bin/env python3
"""
Build the City of Richmond, VA canonical parcel parquet.

Richmond is a Virginia INDEPENDENT CITY (FIPS 51760) — not Richmond County, VA (51159,
a rural Northern Neck county), and not Richmond CA / Richmond KY. The city's own parcel
layer is already city-only, so no county clip is needed.

Source (City of Richmond AGOL org `k3vhq11XkBNeeOfM`, public, no token):
- Parcels/FeatureServer/0:
  https://services1.arcgis.com/k3vhq11XkBNeeOfM/arcgis/rest/services/Parcels/FeatureServer/0
  ~76.9k parcels, assessment date 2026-01-01 (current roll).

  NOTE: do NOT use `City_of_Richmond_Parcels` (CORparcels) from the same org — despite the
  name it holds only ~1,340 CITY-OWNED parcels at a 2022 assessment date. `Parcels` is the
  full roll. (Same schema, so the mistake is silent — this comment is the guardrail.)

Outputs:
- data/jurisidictions/data/richmond/richmond-va-parcels.parquet
- data/jurisidictions/data/richmond/richmond-va-parcels_YYYY_MM_DD.parquet

Notes:
- ~77k parcels -> PMTiles + H3 hexes (over the ~100k rule of thumb's "when in doubt" line,
  and well above the 54k Newport News browser-path city). Bake with --drop-remnants.
- Values: LandValue (land), DwellingValue (improvement), TotalValue (total, used directly
  rather than recomputed so the assessor's own total is preserved).
- Land area: LandSqFt is already SQUARE FEET (unlike Newport News' STATEDAREA, which is
  acres). Geodesic polygon area is the fallback.
- Exempt: TaxExemptCode non-null/non-blank -> exemption_flag=1, excluded (4,423 parcels).
- Condos: Richmond maps ~4.6k condo-class parcels ('R Condo Residential ...', 'B Commercial
  Condo', 'R Condo Common Area ...'). Per-unit records stack on one footprint, so the
  same-footprint collapse (SUM values + SUM stated area, union geometry) from
  run_newportnews.py / run_tulsa.py is applied.
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

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "richmond"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "richmond-va-geometry.parquet"

PARCELS_URL = ("https://services1.arcgis.com/k3vhq11XkBNeeOfM/arcgis/rest/services/"
               "Parcels/FeatureServer/0/query")
WHERE = "1=1"
OUT_FIELDS = ("PIN,OwnerName,LandValue,DwellingValue,TotalValue,LandSqFt,TaxExemptCode,"
              "PropertyClassID,PropertyClass,LandUse,AssessmentDate")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
geod = Geod(ellps="WGS84")

# Richmond's public property search redirects to a DataScout portal with no stable per-PIN
# deep link, so every parcel points at the city's search landing page.
SEARCH_URL = "https://apps.richmondgov.com/applications/PropertySearch/"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_parcels():
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    total = requests.get(PARCELS_URL, params={"where": WHERE, "returnCountOnly": "true",
                         "f": "json"}, headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} Richmond parcels (paginated GeoJSON)...")
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
        if off % 10000 < PAGE:
            log(f"  fetched {off:,}/{total:,}")
        if len(gdf) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


geom = fetch_parcels()
geom["acct"] = geom["PIN"].astype(str).str.strip()
geom = geom[geom["acct"].ne("") & geom["acct"].ne("None") & geom["acct"].ne("nan")]
for c in ["LandValue", "DwellingValue", "TotalValue", "LandSqFt"]:
    geom[c] = pd.to_numeric(geom[c], errors="coerce")
for c in ["PropertyClass", "LandUse", "TaxExemptCode", "OwnerName"]:
    geom[c] = geom[c].astype(str).str.strip()

parcel = geom.rename(columns={"LandValue": "land_val", "DwellingValue": "bld_val",
                              "TotalValue": "tot_appr_val"})
# LandSqFt is already square feet (NOT acres — differs from run_newportnews.py's STATEDAREA).
parcel["stated_sqft"] = pd.to_numeric(parcel["LandSqFt"], errors="coerce")

if parcel.crs is None:
    parcel = parcel.set_crs("EPSG:4326")
elif parcel.crs.to_epsg() != 4326:
    parcel = parcel.to_crs("EPSG:4326")
parcel["geometry"] = parcel["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
parcel = parcel[parcel["geometry"].notnull() & parcel["geometry"].apply(
    lambda x: getattr(x, "is_valid", False) and not x.is_empty)].copy()
log(f"Valid-geometry parcels -> {len(parcel):,}")

# ── dedup multi-polygon parcels (values FIRST, geometry unioned) ──────────────
# One account split across many GIS polygons: a sum here would multiply the account's
# value by its polygon count (the Dallas railroad bug). Values are account-level -> first.
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
log(f"Rows sharing a PIN (multi-polygon): {ndup:,}")
if ndup:
    first_cols = [c for c in parcel.columns if c not in ("geometry", "acct")]
    agg = {c: "first" for c in first_cols}
    agg["stated_sqft"] = "sum"   # per-polygon GIS area sums; values stay `first`
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
log(f"After PIN dedup -> {len(parcel):,}")

# ── exemption flag, applied PER RECORD BEFORE the condo collapse ─────────────
# Richmond carries an authoritative assessor exemption code; any non-blank value = exempt.
# This MUST precede the same-footprint collapse: a condo stack (up to 217 units here) can mix
# exempt and taxable units, and collapsing first with `first` for the code would apply one
# unit's status to the whole stack — dropping ~$0.5B of taxable land value or smuggling exempt
# value in. Richmond's stacks are unit records sharing the ground polygon, not units sitting on
# a separate parent ground parcel, so there is no common-area parent to preserve through the
# filter (the playbook's Fort Collins trap does not apply to this feed).
_tec = parcel["TaxExemptCode"].replace({"None": "", "nan": "", "NaN": ""}).fillna("")
parcel["exemption_flag"] = _tec.str.strip().ne("").astype(int)
n_exempt = int(parcel["exemption_flag"].sum())
parcel = parcel[parcel["exemption_flag"] == 0].copy()
log(f"After exempt filter -> {len(parcel):,} (dropped {n_exempt:,} exempt records)")

# ── condo same-footprint collapse (>1 distinct account at one point -> SUM) ───
rp = parcel.geometry.representative_point()
parcel["_rpkey"] = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str))
vc = parcel["_rpkey"].value_counts()
stacked_keys = vc[vc > 1].index
log(f"Stacked footprints: {len(stacked_keys):,}; max stack: {int(vc.max())}; "
    f"parcels involved: {int(vc[vc > 1].sum()):,}")
if len(stacked_keys):
    is_stacked = parcel["_rpkey"].isin(stacked_keys)
    single = parcel[~is_stacked].copy()
    single["_collapsed"] = 0
    multi = parcel[is_stacked].copy()
    # Condo units are PER-UNIT shares of one shared parcel: sum values AND per-unit stated
    # area across the stack, then reconcile the land denominator against the union polygon.
    sum_cols = ["land_val", "bld_val", "tot_appr_val", "stated_sqft"]
    first_cols = [c for c in multi.columns if c not in (["geometry", "_rpkey"] + sum_cols)]
    agg = {c: "sum" for c in sum_cols if c in multi.columns}
    agg.update({c: "first" for c in first_cols})
    coll = multi.groupby("_rpkey", dropna=False).agg(agg).reset_index()
    gu = multi.groupby("_rpkey", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]))
    coll["geometry"] = gu.values
    coll["_collapsed"] = 1
    coll = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
    parcel = gpd.GeoDataFrame(pd.concat([single, coll], ignore_index=True),
                              geometry="geometry", crs="EPSG:4326")
else:
    parcel["_collapsed"] = 0
parcel = parcel.drop(columns=["_rpkey"], errors="ignore")
log(f"After condo footprint collapse -> {len(parcel):,}")

# ── classification ───────────────────────────────────────────────────────────
# Surface parking classes (unambiguously paved, at-grade). Structured decks/garages are
# improved buildings and are deliberately NOT called Parking here.
SURFACE_PARKING = ("paved surface parking", "industrial paved parking", "paved parking",
                   "apartment parking lot/deck")
CONDO_TOKENS = ("condo",)


def categorize(cls, use):
    c = str(cls or "").strip()
    cl = c.lower()
    u = str(use or "").strip()

    if any(t in cl for t in SURFACE_PARKING):
        return "Parking"
    if "vacant" in cl or u == "Vacant":
        return "Vacant Land"
    if any(t in cl for t in CONDO_TOKENS):
        return "Condominium"
    if u == "Single Family":
        return "Single Family"
    if u in ("Multi-Family", "Duplex (2 Family)"):
        return "Multifamily"
    if u in ("Commercial", "Office"):
        return "Commercial"
    if u == "Industrial":
        return "Industrial"
    if u == "Mixed-Use":
        return "Mixed Use"
    if cl.startswith("r mobile home") or "mobile home" in cl:
        return "Mobile Home"
    return "Other"


parcel["PROPERTY_CATEGORY"] = [categorize(c, u) for c, u in
                               zip(parcel["PropertyClass"], parcel["LandUse"])]

ex = parcel.copy()  # exempt records already removed above, pre-collapse
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other",),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    fetch_footprints=False)
log(f"Classified {len(ex):,} taxable parcels")


# ── canonical fields — LandSqFt denominator, geodesic fallback ───────────────
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

# Collapsed condo stacks: the land denominator is the SHARED complex footprint, not one
# unit's share. Take the larger of summed per-unit stated area and the union polygon area.
col = ex["_collapsed"] == 1
ex.loc[col, "reported_sqft"] = np.maximum(
    pd.to_numeric(ex.loc[col, "reported_sqft"], errors="coerce").fillna(0.0),
    pd.to_numeric(ex.loc[col, "geom_area_sqft"], errors="coerce").fillna(0.0))
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

# Reconcile reported area against the mapped polygon. Richmond's LandSqFt agrees with the
# polygon for ~93% of parcels (ratio p25=0.98, p50=1.00, p75=1.00), but ~5.5k condo / LIHTC /
# apartment / utility records carry a PER-UNIT SHARE (often 0-5 sqft) against the real shared
# footprint. Trusting those produced $1M-$4M/sqft parcels (smoke alarms #2/#4). The polygon is
# the authoritative footprint, so reported area is used only when it is in a sane band around
# it; otherwise fall back to geodesic polygon area.
ratio = ex["reported_sqft"] / ex["geom_area_sqft"].replace(0, np.nan)
use_reported = (ex["reported_sqft"] > 0) & ratio.between(0.5, 2.0)
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["geom_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
log(f"Area denominator: reported={int(use_reported.sum()):,} "
    f"gis-fallback={int((~use_reported).sum()):,} "
    f"(of which stub/share reported area: {int(((ex['reported_sqft'] > 0) & ~ratio.between(0.5, 2.0)).sum()):,})")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = pd.to_numeric(ex.get("tot_appr_val", np.nan), errors="coerce")
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

ex["link"] = SEARCH_URL

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
out = DATA_DIR / "richmond-va-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"richmond-va-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")
log(f"land_value_per_sqft: p50=${final['land_value_per_sqft'].median():.0f} "
    f"p99=${final['land_value_per_sqft'].quantile(.99):.0f} "
    f"max=${final['land_value_per_sqft'].max():.0f}")
log(f"TOTAL land value: ${final['current_full_land_value'].sum():,.0f} over "
    f"{final['land_area_acres'].sum():,.0f} acres")
log("DONE")
