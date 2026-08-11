#!/usr/bin/env python3
"""
Build the City of Vancouver, WA canonical parcel parquet.

Vancouver is in Clark County. Clark County GIS publishes a single one-stop countywide
"Taxlots" polygon layer that carries geometry + assessor market land/building/total value
+ a coarse property-use class + assessor lot size + a tax-status code + the municipal
jurisdiction the parcel sits in (JurisDesc), so this is fully automated (no manual
appraisal-roll join). We clip to the City of Vancouver by the assessor's own jurisdiction
tag JurisDesc='Vancouver' (~59.4k) — this is the taxing/municipal jurisdiction of the
parcel itself (distinct from the owner MailCity and the situs SitusCity fields), so it is
authoritative for city limits, not a situs/mailing heuristic (playbook §4).

Source (public, no token):
- Taxlots (geometry + values + use class + tax status + jurisdiction), Clark County GIS:
  https://gis.clark.wa.gov/arcgisfedpw/rest/services/ClarkView_Public/Taxlots/MapServer/0
  Fields used: Prop_id, PropertyUseClass, AcctTypeDesc, TaxStat, MktLandVal, MktBldgVal,
  MktTotVal, AssrSqFt, AssrAc, GISSqft, JurisDesc.

Outputs:
- data/jurisidictions/data/vancouver/vancouver-wa-parcels.parquet
- data/jurisidictions/data/vancouver/vancouver-wa-parcels_YYYY_MM_DD.parquet

Notes:
- ~59k Vancouver parcels -> PMTiles + H3 hexes (parquet_to_pmtiles.py). Big enough that the
  raw-GeoParquet browser path would look sparse when zoomed out (the Olympia lesson).
- PropertyUseClass is COARSE (8 values: Commercial, Industrial, Multifamily, Condos,
  Mobile Home, Platted/Rural Residential, New Construction). There is no Vacant/Parking/
  Office class, so vacant + underdeveloped + parking are derived downstream by
  classify_property_refined from the land/improvement value ratios (improvement_value==0
  -> Vacant, high land-share -> Underdeveloped), same as elsewhere.
- Exempt: TaxStat in ('EX','DOR') is dropped (EX = exempt; DOR = state-assessed / dept. of
  revenue operating, analogous to King County 'OP'). Tax-RELIEF codes on otherwise-normal
  taxable parcels are KEPT: 'SNR/DSBL' (senior/disabled), 'MFTE-08/10/12' (multi-family tax
  exemption — a temporary improvement abatement; the land is still taxable). 'U500' (2
  parcels) is left taxable (immaterial).
- Condos: 'Condos' is a PropertyUseClass. Whether Clark County maps them at the complex
  level (like King County -> no merge) or as per-unit stubs (like Thurston/Olympia -> merge)
  is verified by the smoke-alarm diagnostics at the end; add the run_olympia.py merge-down
  block only if pencils leak in.
- $/sqft denominator: AssrSqFt (assessor sqft) preferred, fallback AssrAc->sqft, fallback
  GISSqft, fallback geodesic.
- link: Clark County property information portal (Prop_id) — deep-link param unverified.
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

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "vancouver"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "vancouver-wa-geometry.parquet"

PARCELS_URL = ("https://gis.clark.wa.gov/arcgisfedpw/rest/services/ClarkView_Public/"
               "Taxlots/MapServer/0/query")
OUT_FIELDS = ("Prop_id,PropertyUseClass,AcctTypeDesc,TaxStat,MktLandVal,MktBldgVal,"
              "MktTotVal,AssrSqFt,AssrAc,GISSqft,JurisDesc")
WHERE = "JurisDesc='Vancouver'"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000   # Clark County Taxlots maxRecordCount is 2000
geod = Geod(ellps="WGS84")

# TaxStat codes that mark exempt / state-assessed property to drop. Tax-relief codes
# (SNR/DSBL senior-disabled, MFTE-* multifamily improvement abatement) sit on otherwise
# normal taxable parcels and are intentionally NOT dropped.
EXEMPT_TAXSTAT = {"EX", "DOR"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_parcels():
    """Pull all Taxlots with JurisDesc='Vancouver' (city limits), paginated GeoJSON."""
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    total = requests.get(PARCELS_URL, params={"where": WHERE, "returnCountOnly": "true",
                         "f": "json"}, headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} City of Vancouver taxlots (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(5):
            try:
                r = requests.get(PARCELS_URL, params={
                    "where": WHERE, "outFields": OUT_FIELDS,
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
        # NOTE: the server caps each page at maxRecordCount regardless of the requested
        # PAGE, so a short page is NOT necessarily the last — keep going until off >= total
        # or a page returns 0 rows (handled above).
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


geom = fetch_parcels()

# ── validate + normalize geometry ─────────────────────────────────────────────
if geom.crs is None:
    geom = geom.set_crs("EPSG:4326")
elif geom.crs.to_epsg() != 4326:
    geom = geom.to_crs("EPSG:4326")
geom["geometry"] = geom["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
parcel = geom[geom["geometry"].notnull() & geom["geometry"].apply(
    lambda x: getattr(x, "is_valid", False) and not x.is_empty)].copy()
log(f"Valid geometries: {len(parcel):,} of {len(geom):,}")

# ── field cleanup ─────────────────────────────────────────────────────────────
parcel["acct"] = parcel["Prop_id"].astype(str).str.strip()
parcel = parcel[parcel["acct"].ne("") & parcel["acct"].ne("None") & parcel["acct"].ne("nan")]
for c in ["MktLandVal", "MktBldgVal", "MktTotVal", "AssrSqFt", "AssrAc", "GISSqft"]:
    parcel[c] = pd.to_numeric(parcel[c], errors="coerce")
for c in ["PropertyUseClass", "AcctTypeDesc", "TaxStat"]:
    parcel[c] = parcel[c].astype(str).str.strip().replace({"nan": "", "None": ""})

parcel = parcel.rename(columns={"MktLandVal": "land_val", "MktBldgVal": "bld_val"})
parcel["tot_appr_val"] = parcel["MktTotVal"].where(
    parcel["MktTotVal"] > 0, parcel["land_val"].fillna(0) + parcel["bld_val"].fillna(0))
# reported lot area (sqft): AssrSqFt preferred, fall back to AssrAc converted, then GISSqft
acres_sqft = parcel["AssrAc"] * SQFT_PER_ACRE
parcel["stated_sqft"] = parcel["AssrSqFt"].where(parcel["AssrSqFt"] > 0, acres_sqft)
parcel["stated_sqft"] = parcel["stated_sqft"].where(parcel["stated_sqft"] > 0, parcel["GISSqft"])

# ── dedup multi-polygon parcels (values first — broadcast, never sum; geometry unioned) ──
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
log(f"Rows sharing a Prop_id (multi-polygon): {ndup:,}")
if ndup:
    first_cols = [c for c in parcel.columns if c not in ("geometry", "acct")]
    agg = {c: "first" for c in first_cols}
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
log(f"After Prop_id dedup -> {len(parcel):,}")

# ── exemption flag + classification ──────────────────────────────────────────
parcel["exemption_flag"] = parcel["TaxStat"].str.upper().isin(EXEMPT_TAXSTAT).astype(int)


def categorize(use_class):
    d = str(use_class or "").strip().lower()
    if not d:
        return "Other"
    if "condo" in d:
        return "Condominium"
    if "mobile home" in d:
        return "Mobile Home"
    if "multifamily" in d or "multi-family" in d:
        return "Multifamily"
    if "industrial" in d:
        return "Industrial"
    if "commercial" in d:
        return "Commercial"
    if "residential" in d:          # Platted Residential / Rural Residential
        return "Single Family"
    return "Other"                  # New Construction + anything unmapped


parcel["PROPERTY_CATEGORY"] = [categorize(d) for d in parcel["PropertyUseClass"]]

ex = parcel[parcel["exemption_flag"] == 0].copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other",),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    fetch_footprints=False)
log(f"After exempt filter -> {len(ex):,} (dropped {int(parcel['exemption_flag'].sum()):,} exempt)")


# ── canonical fields — AssrSqFt denominator, geodesic fallback ────────────────
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

# Clark County property information portal (Property ID). Deep-link param unverified.
ex["link"] = ("https://gis.clark.wa.gov/gishome/property/?p=" + ex["acct"].astype(str))

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
out = DATA_DIR / "vancouver-wa-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"vancouver-wa-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
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
