#!/usr/bin/env python3
"""
Build Flint, MI's canonical parcel parquet from the City of Flint's own parcel layer.

Source (one-stop: geometry + assessor values, published by the City of Flint GIS org on
ArcGIS Online, services5.arcgis.com/lqqWNtSxx8Akj04A):

  "Main_COF_Parcel"  Main_COF_Parcel_view/FeatureServer/0   (~54.9k polygons)

It is ALREADY City-of-Flint-only (Flint is its own assessing unit inside Genesee County;
Prop_City is FLINT on every row, and Flint Township / Burton / Mt. Morris Twp are not in
it), so NO boundary clip is needed (cf. Detroit / Baltimore city-only sources).

Fields used (owner / taxpayer-name fields are never requested):
  PIDText / PIDdash  parcel number (one row per PID; asserted below)
  Land_Value         assessor's land value on the TRUE CASH VALUE (100% market) basis.
                     Verified: on vacant parcels (class x02) HomeSEV ~= 0.5 x Land_Value.
  HomeSEV            State Equalized Value (Michigan assesses at 50% of true cash value,
                     Const. art. IX §3 / MCL 211.27a). Despite the name it is populated for
                     every class, not just homesteads.
  Resb_Value         RESIDENTIAL building value only (0 on commercial/industrial parcels).
  Exempt / Exempt_Cod  tax status; 'TAXABLE' or 'EXEMPT (<MCL section>)'.
  Prop_Class         Michigan statewide class: 2xx commercial, 3xx industrial, 4xx residential;
                     x01 improved, x02 vacant. (No x07 condo classes occur in Flint.)
  Inv22              the City's 2022 parcel-condition windshield survey (Good / Fair / Poor /
                     Sub-Standard / Vacant Lot / Parking Lot).

Values (all on the true-cash-value basis, so land and improvement add up to market value):
  land_value         = Land_Value
  est. TCV           = 2 x HomeSEV   (fallback when SEV is 0: Land_Value + Resb_Value)
  improvement_value  = max(est. TCV - land_value, 0)
  full_market_value  = land_value + improvement_value
Taxable Value (the Prop-A-capped base) is NOT used: it is not published here and it is a
capped tax base, not a market value.

Exemptions: every non-'TAXABLE' parcel is dropped (Detroit precedent). The big one is
MCL 211.7gg — property held by a land bank fast-track authority, i.e. the Genesee County
Land Bank (~12k parcels). Its land/acreage totals are logged before the drop.

Flint (~38k shipped parcels) is under the ~100k PMTiles threshold -> browser GeoParquet path.

Outputs:
- data/jurisidictions/data/flint/flint-mi-parcels.parquet
- data/jurisidictions/data/flint/flint-mi-parcels_YYYY_MM_DD.parquet
"""
from __future__ import annotations

import io
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
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined, gis_area_sqft  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "flint"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "flint-mi-geometry.parquet"

URL = ("https://services5.arcgis.com/lqqWNtSxx8Akj04A/arcgis/rest/services/"
       "Main_COF_Parcel_view/FeatureServer/0/query")
FIELDS = ("PIDText,PIDdash,Full_Prop,Exempt,Exempt_Cod,HomeSEV,Land_Value,Resb_Value,"
          "Resb_Style,Dwelling_U,LandNetAcr,Use_Type,Prop_Class,ECF_Desc,Zoning,LandUse,"
          "Inv22,Ward,Year_Built")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh) Chrome/124 Safari/537.36"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch():
    total = requests.get(URL, params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=120).json()["count"]
    log(f"  pulling {total:,} parcels (geojson, {PAGE}/page)...")
    pages, off = [], 0
    while off < total:
        params = {"where": "1=1", "outFields": FIELDS, "returnGeometry": "true",
                  "outSR": 4326, "f": "geojson", "orderByFields": "FID",
                  "resultOffset": off, "resultRecordCount": PAGE}
        for attempt in range(4):
            try:
                r = requests.get(URL, params=params, headers=HEADERS, timeout=180)
                r.raise_for_status()
                g = gpd.read_file(io.BytesIO(r.content))
                break
            except Exception as e:  # transient AGOL hiccup
                if attempt == 3:
                    raise
                log(f"    retry page@{off} ({e})")
                time.sleep(2 * (attempt + 1))
        if len(g) == 0:
            break
        pages.append(g)
        off += len(g)
    gdf = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    assert len(gdf) == total, f"fetched {len(gdf)} of {total}"
    return gdf


# ── 1. Parcels (cached) ──────────────────────────────────────────────────────
if GEOM_CACHE.exists():
    log(f"Using cached parcels: {GEOM_CACHE.name}")
    raw = gpd.read_parquet(GEOM_CACHE)
else:
    log("Fetching City of Flint parcels (Main_COF_Parcel)...")
    raw = fetch()
    raw.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached -> {GEOM_CACHE.name} ({len(raw):,} rows)")

raw["pid"] = raw["PIDText"].astype(str).str.strip()
raw = raw[raw["pid"].ne("") & raw["pid"].ne("None") & raw.geometry.notna()].copy()
for c in ["Land_Value", "HomeSEV", "Resb_Value", "Prop_Class", "Exempt", "Ward"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
log(f"Parcels with a PID + geometry: {len(raw):,} | bounds {[round(v, 4) for v in raw.total_bounds]}")

# ── 2. Duplicate-PID preflight (skill §2a) ───────────────────────────────────
d = raw[raw["pid"].duplicated(keep=False)]
log(f"PIDs with >1 geometry row: {d['pid'].nunique():,}")
if len(d):
    # One PID = one assessor record; extra rows are pieces of the same parcel. Values are
    # broadcast (first), geometry unioned — never summed, never keep-largest.
    nuniq = d.groupby("pid")["Land_Value"].nunique().gt(1).sum()
    log(f"  of which with differing Land_Value: {nuniq}")
    geom_u = raw.groupby("pid")["geometry"].apply(lambda gs: unary_union(list(gs)))
    raw = raw.drop_duplicates("pid").set_index("pid")
    raw["geometry"] = geom_u
    raw = gpd.GeoDataFrame(raw.reset_index(), geometry="geometry", crs="EPSG:4326")
assert raw["pid"].duplicated().sum() == 0

# ── 3. Values (true-cash-value basis) ────────────────────────────────────────
land = raw["Land_Value"].fillna(0).astype(float)
sev = raw["HomeSEV"].fillna(0).astype(float)
resb = raw["Resb_Value"].fillna(0).astype(float)
est_tcv = np.where(sev > 0, 2.0 * sev, land + resb)
raw["land_value"] = land
raw["improvement_value"] = np.clip(est_tcv - land, 0, None)
raw["full_market_value"] = raw["land_value"] + raw["improvement_value"]
raw["sev"] = sev

# ── 4. Categorize (Michigan property class + local hints) ────────────────────
MF_STYLES = ("TWO UNIT", "THREE UNIT", "FOUR UNIT")


def categorize(pc, style, ecf, inv):
    pc = int(pc) if pd.notna(pc) else 0
    style = str(style or "").upper()
    ecf = str(ecf or "").upper()
    inv = str(inv or "")
    if inv == "Parking Lot" and pc in (201, 202, 301, 302):
        return "Parking"
    if pc == 402:
        return "Vacant Residential"
    if pc == 202:
        return "Vacant Commercial"
    if pc == 302:
        return "Vacant Industrial"
    if pc == 401:
        return "Multifamily" if any(s in style for s in MF_STYLES) else "Single Family"
    if pc == 201:
        return "Multifamily" if "APARTMENT" in ecf else "Commercial"
    if pc == 301:
        return "Industrial"
    return "Other"


raw["property_land_use_category"] = [categorize(*t) for t in zip(
    raw["Prop_Class"], raw["Resb_Style"], raw["ECF_Desc"], raw["Inv22"])]

# ── 5. Exemptions — anything not TAXABLE is dropped ──────────────────────────
code = raw["Exempt_Cod"].fillna("").astype(str).str.strip().str.upper()
raw["exemption_flag"] = (code.ne("TAXABLE") | raw["Exempt"].fillna(0).ne(0)).astype(int)
raw["gis_area_sqft"] = raw["geometry"].apply(gis_area_sqft)
_ex = raw[raw["exemption_flag"] == 1]
log(f"Exempt/non-taxable: {len(_ex):,} of {len(raw):,} "
    f"({_ex['gis_area_sqft'].sum() / SQFT_PER_ACRE:,.0f} acres, land ${_ex['land_value'].sum():,.0f})")
for c, g in _ex.groupby(code[_ex.index]):
    log(f"    {c:22s} {len(g):6,} parcels  {g['gis_area_sqft'].sum() / SQFT_PER_ACRE:8,.0f} ac  "
        f"land ${g['land_value'].sum():>13,.0f}")
lb = _ex[code[_ex.index].eq("EXEMPT (211.7GG)")]
log(f"  Land bank (MCL 211.7gg): {len(lb):,} parcels, {lb['gis_area_sqft'].sum() / SQFT_PER_ACRE:,.0f} acres, "
    f"land ${lb['land_value'].sum():,.0f}; {int(lb['Prop_Class'].isin([202, 302, 402]).sum()):,} "
    f"vacant-class, survey 'Vacant Lot' {int((lb['Inv22'] == 'Vacant Lot').sum()):,}")
parcel = raw[raw["exemption_flag"] == 0].copy()
log(f"After exempt filter -> {len(parcel):,}")

# ── 6. Canonical fields ──────────────────────────────────────────────────────
parcel["geometry"] = parcel["geometry"].apply(lambda x: x if x.is_valid else x.buffer(0))
parcel["gis_area_sqft"] = parcel["geometry"].apply(gis_area_sqft)
parcel.loc[parcel["gis_area_sqft"] < 1, "gis_area_sqft"] = np.nan
_src = pd.to_numeric(parcel["LandNetAcr"], errors="coerce") * SQFT_PER_ACRE
_agree = (parcel["gis_area_sqft"] / _src.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
log(f"Area cross-check vs LandNetAcr — median ratio {_agree.median():.4f} (expect ~1.0), "
    f"rows off by >10%: {int((_agree.sub(1).abs() > 0.10).sum()):,}")
# LandNetAcr has garbage outliers (max 576,734 ac), so the polygon is the denominator.
parcel["land_area_sqft"] = parcel["gis_area_sqft"]
parcel["area_source"] = "gis"
parcel["land_area_acres"] = parcel["land_area_sqft"] / SQFT_PER_ACRE
parcel["likely_remnant"] = (parcel["land_area_sqft"] < 500).astype(int)

den = parcel["land_area_sqft"].replace(0, np.nan)
parcel["full_market_value_per_sqft"] = parcel["full_market_value"] / den
parcel["land_value_per_sqft"] = parcel["land_value"] / den
parcel["improvement_value_per_sqft"] = parcel["improvement_value"] / den
parcel = add_improvement_ratio_fields(parcel, land_col="land_value", improvement_col="improvement_value")

parcel["property_land_use_refined"] = classify_property_refined(
    parcel, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other",),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    state_class_col="__none__", fetch_footprints=False)

# No stable public per-parcel deep link (the City's assessing search is session-based);
# the popup omits a null link.
parcel["link"] = np.nan
parcel["parcel_number"] = parcel["PIDdash"].astype(str)
parcel["address"] = parcel["Full_Prop"].astype(str).str.replace(r",\s*FLINT, MI, \d+$", "", regex=True)
parcel["survey_condition_2022"] = parcel["Inv22"].fillna("").replace("", np.nan)
parcel["state_equalized_value"] = parcel["sev"]
parcel["ward"] = parcel["Ward"]

# ── 7. Export ────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant", "parcel_number", "address", "state_equalized_value",
           "survey_condition_2022", "ward"]
final = parcel[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")
assert final["parcel_number"].duplicated().sum() == 0
out = DATA_DIR / "flint-mi-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"flint-mi-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet",
                 index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"land ${final.current_full_land_value.sum():,.0f} over "
    f"{final.land_area_acres.sum():,.0f} acres | impr ${final.improvement_value.sum():,.0f} | "
    f"total ${final.full_market_value.sum():,.0f}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")

# ── §6a smoke alarms: condo / unit stubs ──────────────────────────────────────
log("--- condo/stub smoke alarms (skill §6a) ---")
a = parcel["gis_area_sqft"]
lv = pd.to_numeric(final["land_value_per_sqft"], errors="coerce")
log(f"  footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
log(f"  sub-500 / sub-1000 sqft footprints: {int((a < 500).sum()):,} / {int((a < 1000).sum()):,}")
log(f"  land $/sqft p50/p95/p99/max: ${lv.median():,.2f} / ${lv.quantile(.95):,.2f} / "
    f"${lv.quantile(.99):,.2f} / ${lv.max():,.2f}")
rp = final.geometry.representative_point()
vc = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)).value_counts()
log(f"  stacked clusters: {int((vc > 1).sum())}, max stack: {int(vc.max())}")
holes = final.geometry.apply(lambda g: 0 if g is None else sum(
    len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
log(f"  parcels with interior rings (holes): {int((holes > 0).sum()):,}")
log(f"  zero land value (renders as gp-error): "
    f"{int((final['current_full_land_value'].fillna(0) <= 0).sum()):,}")
log(f"  bounds: {[round(v, 4) for v in final.total_bounds]}")
log("DONE")
