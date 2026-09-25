#!/usr/bin/env python3
"""
Build the City of Providence, RI canonical parcel parquet.

Rhode Island has no county government: the City of Providence is the assessing and taxing
jurisdiction, and the city's own parcel layer is already city-only (MuniName = Providence),
so no county clip is needed.

Sources (public, no token):
- Geometry + land/building split — City of Providence AGOL org `wv9mHoqblhTsnqdG` (PVDGIS),
  `Parcel_Zoning_FL/FeatureServer/0` ("Parcels with CAMA"):
  https://services6.arcgis.com/wv9mHoqblhTsnqdG/arcgis/rest/services/Parcel_Zoning_FL/FeatureServer/0
  44,346 polygons, TaxRollYear = 2025 = ASSESSED 2024-12-31 (the 2024 revaluation /
  statistical update), BILLED FY2026 (July 2025 - June 2026). Layer last edited 2026-09-11.

  VINTAGE — verified 2026-09-25 from the billed taxes, not from dataset labels. The roll's
  implied rates (total_taxes / net assessment) are exactly the FY2026 ordinance rates in RI
  Division of Municipal Finance's "FY 2026 Tax Rates by Class of Property, Assessment Date
  December 31, 2024": 8.40 (1-family owner-occupied), 7.55 (2-5 family OO), 14.60 (1-family
  non-OO), 14.00 (2-5 family non-OO), 26.00 (6-10 units), 28.50 (11+), 29.20 (commercial) —
  e.g. 39 Pratt St 933,500 x 8.40/1000 = $7,841.40 billed. The FY2025 table (assessed
  2023-12-31) has Providence at a single 18.35 residential / 35.10 commercial rate, and those
  rates are what the Socrata "2024" roll bills. DMF convention: roll year N = assessed
  12/31/(N-1) = FY(N+1) bills. The Socrata description of the "2025" roll ("calendar year 2024
  ... billed in FY2025") is off by one fiscal year; trust the rates.

  NOTE — the same org publishes FOUR same-schema "Parcels with CAMA" layers. Only this one is
  current. Verified 2026-09-25 by TaxRollYear and by matching sample parcels to the Socrata
  rolls (059-0291-0000 / 009-0559-0000 / 001-0007-0000):
    Parcel_Zoning_FL/0      TaxRollYear 2025  $35.8B  edited 2026-09-11  <- USE
                            (319,500 / 933,500 / 507,400 = Socrata "2025" roll; assessed
                            2024-12-31, FY2026 bills)
    Parcels_with_CAMA/0     TaxRollYear 2024  $27.6B  edited 2025-05-30  (230,100 / 750,000 /
                            396,900 = Socrata "2024" roll; assessed 2023-12-31, FY2025 bills
                            at 18.35/35.10 — pre-revaluation)
    Parcel_Boundaries_v1/0  TaxRollYear 2024  $27.7B  edited 2024-09-12  (same 2024 roll)
    Parcels_wCAMA/0         TaxRollYear 2023  $27.4B  edited 2024-09-16  (226,900 for
                            059-0291-0000 = Socrata "2023" roll; assessed 2022-12-31)
  The names suggest the opposite; the silent ~30% under-valuation is why this comment exists.

- Exemption status — City of Providence Socrata "2025 Property Tax Roll" (6ub4-iebe) — the
  same roll as the layer above (assessed 2024-12-31, FY2026 bills):
  https://data.providenceri.gov/Finance/2025-Property-Tax-Roll/6ub4-iebe
  Total assessment only (no land/building split) but carries the levy code and exempt amount.
  Joined on PROPID == tax_map; where joined, AssessedValueTotal == total_assmt on 100% of rows.

Outputs:
- data/jurisidictions/data/providence/providence-ri-parcels.parquet
- data/jurisidictions/data/providence/providence-ri-parcels_YYYY_MM_DD.parquet

Source traps handled:
- DUPLICATE POLYGONS: 169 PROPIDs appear 2-3x with identical geometry and identical values.
  Deduped by PROPID with values `first` (skill §2 — never summed).
- EXEMPTION: the GIS layer has no exemption field. levy_code_1 == 'E01' (Non Residential
  Exempt; exempt amount == assessment on every E01 row) -> exemption_flag = 1, excluded,
  applied PER RECORD BEFORE the condo collapse. Rows the roll does not cover fall back to the
  exempt use codes 70-84. Partial abatements are NOT exemptions and are kept taxable: 8LAW
  (use code 12, RIGL 44-5-13.11 "8% law" rent-based assessment) and TSA (use code 83,
  RIGL 44-3-9 tax stabilization agreements) carry full market land/building assessments.
- CONDOS CARRY $0 LAND: all 4,128 condo units (use 23/24) have AssessedValueLand = 0 with the
  whole unit value in AssessedValueBuildings — the assessor does not split condo land. Units are
  mapped as N identical copies of the building-lot polygon (max stack 504), so they are
  collapsed by footprint (values SUMMED across distinct units, geometry unioned) into one
  parcel per lot. The merged lot keeps the assessor's $0 land value; it is NOT imputed from
  neighbours (Seattle/Gwinnett precedent: publish the assessor's number and note it). These
  lots appear in the gp-error ($0 land) layer.
- Utility (use 10) parcels excluded (state-assessed/utility, per the add-city skill).
- ~315 roll records ($0.43B assessed) have no polygon in the GIS layer and are not mapped.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined, gis_area_sqft  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "providence"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "providence-ri-geometry.parquet"
ROLL_CACHE = DATA_DIR / "providence-ri-taxroll-2025.parquet"

PARCELS_URL = ("https://services6.arcgis.com/wv9mHoqblhTsnqdG/arcgis/rest/services/"
               "Parcel_Zoning_FL/FeatureServer/0/query")
ROLL_URL = "https://data.providenceri.gov/resource/6ub4-iebe.json"
OUT_FIELDS = ("OBJECTID,PROPID,POLY_TYPE,UnitNum,ParcAddress,AssessedValueTotal,AssessedValueLand,"
              "AssessedValueBuildings,TaxRollYear,MuniUseCode,MuniUseCodeDesc,AreaSF,NumUnits,ZONING")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
EXPECTED_ROLL_YEAR = "2025"

# Providence's online assessor search has no stable per-parcel deep link reachable from here,
# so every parcel points at the Tax Assessor's landing page (Richmond precedent).
SEARCH_URL = "https://www.providenceri.gov/tax-assessor/"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_parcels():
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    total = requests.get(PARCELS_URL, params={"where": "1=1", "returnCountOnly": "true",
                         "f": "json"}, headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} Providence parcels (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(5):
            try:
                r = requests.get(PARCELS_URL, params={
                    "where": "1=1", "outFields": OUT_FIELDS, "returnGeometry": "true",
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
        if len(gdf) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


def fetch_roll():
    if ROLL_CACHE.exists():
        log(f"Using cached tax roll: {ROLL_CACHE.name}")
        return pd.read_parquet(ROLL_CACHE)
    rows, off = [], 0
    while True:
        r = requests.get(ROLL_URL, params={
            "$limit": 50000, "$offset": off, "$order": ":id",
            "$select": "p_id,tax_map,plat,lot,unit,class,short_desc,levy_code_1,short_desc_1,"
                       "formated_address,total_assmt,total_exempt,total_taxes"},
            headers=HEADERS, timeout=240)
        r.raise_for_status()
        batch = r.json()
        rows += batch
        off += len(batch)
        if len(batch) < 50000:
            break
    roll = pd.DataFrame(rows)
    roll.to_parquet(ROLL_CACHE, index=False)
    log(f"  cached tax roll -> {ROLL_CACHE.name} ({len(roll):,} rows)")
    return roll


geom = fetch_parcels()
log(f"Raw polygons: {len(geom):,}")
yrs = geom["TaxRollYear"].dropna().astype(str).value_counts()
log(f"TaxRollYear: {yrs.to_dict()}")
if yrs.index[0] != EXPECTED_ROLL_YEAR:
    raise SystemExit(f"Expected TaxRollYear {EXPECTED_ROLL_YEAR}; layer is {yrs.index[0]} — "
                     "wrong/stale layer? (see the NOTE in the header)")

geom["acct"] = geom["PROPID"].fillna("").astype(str).str.strip()
for c in ["AssessedValueLand", "AssessedValueBuildings", "AssessedValueTotal", "AreaSF", "NumUnits"]:
    geom[c] = pd.to_numeric(geom[c], errors="coerce")
geom["use"] = geom["MuniUseCode"].fillna("").astype(str).str.strip()
geom["use_desc"] = geom["MuniUseCodeDesc"].fillna("").astype(str).str.strip()

parcel = geom[geom["POLY_TYPE"].eq("PARCEL") & geom["acct"].ne("")].copy()
log(f"POLY_TYPE=PARCEL with a PROPID -> {len(parcel):,} (dropped water/ROW/unkeyed polygons)")
parcel = parcel.rename(columns={"AssessedValueLand": "land_val", "AssessedValueBuildings": "bld_val",
                                "AssessedValueTotal": "tot_appr_val"})
parcel["stated_sqft"] = parcel["AreaSF"]
parcel["geometry"] = parcel["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
parcel = parcel[parcel["geometry"].notnull() & parcel["geometry"].apply(
    lambda x: getattr(x, "is_valid", False) and not x.is_empty)].copy()

# ── dedup repeated PROPIDs (values FIRST, geometry unioned) ──────────────────
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
log(f"Rows sharing a PROPID: {ndup:,}")
if ndup:
    first_cols = [c for c in parcel.columns if c not in ("geometry", "acct")]
    agg = {c: "first" for c in first_cols}
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]))
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
log(f"After PROPID dedup -> {len(parcel):,}")

# ── join the roll-year-2025 tax roll (assessed 2024-12-31, FY2026 bills) ──
roll = fetch_roll().drop_duplicates("p_id").drop_duplicates("tax_map")
roll["total_assmt"] = pd.to_numeric(roll["total_assmt"], errors="coerce")
roll["total_exempt"] = pd.to_numeric(roll["total_exempt"], errors="coerce")
parcel = parcel.merge(roll[["tax_map", "levy_code_1", "class", "total_assmt", "total_exempt"]],
                      left_on="acct", right_on="tax_map", how="left")
parcel = gpd.GeoDataFrame(parcel, geometry="geometry", crs="EPSG:4326")
matched = parcel["tax_map"].notna()
log(f"Roll join: {int(matched.sum()):,} matched, {int((~matched).sum()):,} unmatched; "
    f"total agreement on matched = {(parcel.loc[matched, 'tot_appr_val'] == parcel.loc[matched, 'total_assmt']).mean():.4f}")

# ── exemption flag, PER RECORD, BEFORE the condo collapse ────────────────────
EXEMPT_USE = {"70", "71", "72", "73", "74", "75", "76", "78", "79", "80", "82", "84"}
levy = parcel["levy_code_1"].fillna("").astype(str)
parcel["exemption_flag"] = np.where(matched, levy.eq("E01"), parcel["use"].isin(EXEMPT_USE)).astype(int)
n_exempt = int(parcel["exemption_flag"].sum())
ex_land = parcel.loc[parcel["exemption_flag"] == 1, "land_val"].sum()
parcel = parcel[parcel["exemption_flag"] == 0].copy()
log(f"After exempt filter -> {len(parcel):,} (dropped {n_exempt:,} exempt, ${ex_land/1e9:.2f}B land)")

n_util = int(parcel["use"].eq("10").sum())
parcel = parcel[~parcel["use"].eq("10")].copy()
zero = parcel["tot_appr_val"].fillna(0) <= 0
parcel = parcel[~zero].copy()
log(f"Dropped {n_util} utility and {int(zero.sum())} zero-value records -> {len(parcel):,}")


# ── classification (per record, so condo stacks can vote) ────────────────────
def categorize(use: str) -> str:
    return {
        "01": "Single Family",
        "02": "Multifamily",           # 2-5 family
        "03": "Multifamily",           # apartments 6+
        "04": "Mixed Use",             # "Combination" residential + commercial
        "05": "Commercial",
        "06": "Commercial",
        "07": "Industrial",
        "12": "Other Improved",        # 8LAW records (RIGL 44-5-13.11)
        "13": "Vacant Land",           # residential vacant
        "14": "Vacant Land",           # commercial/industrial vacant (incl. unimproved lots)
        "23": "Condominium",
        "24": "Condominium",           # commercial condo
        "33": "Other",                 # farm/forest
        "83": "Tax-Stabilized (TSA)",  # RIGL 44-3-9 stabilization agreements
    }.get(use, "Other")


# ~270 valued GIS records have a blank MuniUseCode; the roll's `class` carries the same code
# family ("1", "2", "03-610", "04-05U", "23", ...), so normalise it to the two-digit use code.
def roll_class_to_use(c) -> str:
    c = str(c or "").strip()
    if not c or c == "nan":
        return ""
    head = c.split("-")[0]
    return head.zfill(2) if head.isdigit() else ""


blank = parcel["use"].eq("")
parcel.loc[blank, "use"] = parcel.loc[blank, "class"].map(roll_class_to_use)
log(f"Blank use code filled from roll class: {int((blank & parcel['use'].ne('')).sum()):,}")
parcel["PROPERTY_CATEGORY"] = parcel["use"].map(categorize)
log(f"record categories: {parcel['PROPERTY_CATEGORY'].value_counts().to_dict()}")

# ── condo same-footprint collapse (distinct accounts on one polygon -> SUM) ──
rp = parcel.to_crs(32619).geometry.representative_point()
parcel["_rpkey"] = rp.x.round(0).astype(str) + "," + rp.y.round(0).astype(str)
vc = parcel["_rpkey"].value_counts()
stacked = vc[vc > 1].index
log(f"Stacked footprints: {len(stacked):,}; max stack: {int(vc.max())}; "
    f"records involved: {int(vc[vc > 1].sum()):,}")
is_st = parcel["_rpkey"].isin(stacked)
single = parcel[~is_st].copy()
single["_collapsed"] = 0
multi = parcel[is_st].copy()
if len(multi):
    # dominant category by total value within the stack (a condo lot with one commercial unit
    # stays a Condominium)
    dom = (multi.groupby(["_rpkey", "PROPERTY_CATEGORY"])["tot_appr_val"].sum()
           .reset_index().sort_values("tot_appr_val", ascending=False)
           .drop_duplicates("_rpkey").set_index("_rpkey")["PROPERTY_CATEGORY"])
    sum_cols = ["land_val", "bld_val", "tot_appr_val"]
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "first" for c in multi.columns
                if c not in sum_cols + ["geometry", "_rpkey"]})
    coll = multi.groupby("_rpkey").agg(agg).reset_index()
    coll["PROPERTY_CATEGORY"] = coll["_rpkey"].map(dom)
    coll["geometry"] = multi.groupby("_rpkey")["geometry"].apply(
        lambda gs: unary_union(list(gs))).reindex(coll["_rpkey"]).values
    coll["stated_sqft"] = np.nan   # per-unit AreaSF is 0 on condos; use the polygon
    coll["_collapsed"] = 1
    parcel = gpd.GeoDataFrame(pd.concat([single, gpd.GeoDataFrame(coll, geometry="geometry",
                              crs="EPSG:4326")], ignore_index=True), geometry="geometry", crs="EPSG:4326")
else:
    parcel = single
parcel = parcel.drop(columns=["_rpkey"], errors="ignore")
log(f"After condo footprint collapse -> {len(parcel):,}")

ex = parcel.copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")
# Providence's roll has no parking use code (surface lots sit in 13/14 "Vacant Land" or 06
# "Commercial II"), so no parcel is forced into 'Parking Lot' here; parking comes from the
# separate parking dataset.
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other",),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    fetch_footprints=False)

# ── canonical fields — AreaSF denominator, geodesic fallback ─────────────────
log("Computing GIS areas...")
ex["geom_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
ex["reported_sqft"] = pd.to_numeric(ex["stated_sqft"], errors="coerce")
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan
ratio = ex["reported_sqft"] / ex["geom_area_sqft"].replace(0, np.nan)
use_reported = (ex["reported_sqft"] > 0) & ratio.between(0.5, 2.0)
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["geom_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
log(f"Area denominator: reported={int(use_reported.sum()):,} gis-fallback={int((~use_reported).sum()):,} "
    f"(reported outside 0.5-2.0x polygon: {int(((ex['reported_sqft'] > 0) & ~ratio.between(0.5, 2.0)).sum()):,})")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = pd.to_numeric(ex["tot_appr_val"], errors="coerce")
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")
ex["link"] = SEARCH_URL

# ── smoke alarms (skill §6a) ─────────────────────────────────────────────────
a = ex["geom_area_sqft"]
lv = ex["land_value_per_sqft"]
log(f"footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
log(f"sub-500 / sub-1000 sqft valued: {int(((a < 500) & (ex['land_value'] > 0)).sum())} / "
    f"{int(((a < 1000) & (ex['land_value'] > 0)).sum())}")
log(f"land $/sqft p50/p99/max: {lv.median():.0f} / {lv.quantile(.99):.0f} / {lv.max():.0f}")
rp2 = ex.to_crs(32619).geometry.representative_point()
vc2 = (rp2.x.round(0).astype(str) + "," + rp2.y.round(0).astype(str)).value_counts()
log(f"residual stacked clusters: {int((vc2 > 1).sum())}")
log(f"$0-land parcels shipped: {int((ex['land_value'] <= 0).sum())} "
    f"(condo lots: {int(((ex['land_value'] <= 0) & (ex['property_land_use_category'] == 'Condominium')).sum())})")

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
final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")
out = DATA_DIR / "providence-ri-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"providence-ri-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")
log(f"TOTAL land value: ${final['current_full_land_value'].sum():,.0f} over "
    f"{final['land_area_acres'].sum():,.0f} acres; total market ${final['full_market_value'].sum():,.0f}")
log("DONE")
