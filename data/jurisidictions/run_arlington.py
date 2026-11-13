#!/usr/bin/env python3
"""
Build Arlington, TX's canonical parcel parquet from the Tarrant Appraisal District (TAD) roll.

Fully automated, one source (no manual download):
- Parcels (geometry + LAND_VAL / IMPR_VAL / MARKET_VALUE + SPTB state class + LAND_ACRE +
  TAX_YEAR): the City of Fort Worth's public copy of the TAD parcel layer, refreshed nightly
  from TAD for every Tarrant County parcel (TAX_YEAR 2026 as of 2026-11).
    https://services5.arcgis.com/3ddLCBXe1bRt7mzj/arcgis/rest/services/Parcels_Public_Vview/FeatureServer/0
  Pulled with the city boundary's envelope as a spatial pre-filter and cached on first run.
  (Its non-Tarrant rows - Denton/Johnson/Parker/Wise parts of Fort Worth - are 2023
  certified data; Arlington is entirely in Tarrant County, so none are used here.)
- City limits: City of Arlington "Municipal Boundary" FeatureServer, MUNICIPALNAME='ARLINGTON'
  (centroid-within clip). Cross-checked against TAD's own taxing-unit code (CITY='24').
    https://services.arcgis.com/jXi5GuMZwfCYtZP9/arcgis/rest/services/Municipal_Boundary/FeatureServer/0

Outputs:
- data/jurisidictions/data/arlington/arlington-tx-parcels.parquet (+ dated snapshot)

Notes:
- TAD's EXEMPTION_CODE is 'YES' for ANY exemption, including the ordinary homestead
  exemption (61% of Arlington parcels), so it is NOT a full-exemption flag, and TAD almost never
  uses state class X. Totally exempt = class X, OR a named public owner, OR $0 City of Arlington
  taxable value on an appraised non-residential parcel (city, state, ISDs, UTA, churches,
  charities, cemeteries). Homes with $0 taxable (100% disabled-veteran homesteads) stay in
  unless the owner is institutional.
- LAND_SQFT is empty in this layer. The $/sqft denominator is the appraiser's LAND_ACRE when it
  is within 25% of the geodesic polygon area, else the polygon area: on ~6.7k parcels LAND_ACRE
  holds square feet, tiny fractions or a partial area. Emits land_area_acres, area_source,
  likely_remnant.
- Stacked records: TAD maps every condo unit onto the whole complex polygon (each owning
  "% OF COMMON AREA"; The Cloisters is 491 units on one 27.6-acre polygon), carries one record
  per undivided-interest owner with that owner's SHARE (5120 Yucca Ct: $41,250 + $13,750 land),
  and splits lots into accounts at school/TIF boundaries. All are partial values of one
  footprint, so they are SUMMED onto it and its denominator is the footprint area (a condo
  unit's LAND_ACRE is its share or 0). The surviving row keeps the highest-land-value record's
  class. One account split across several polygons instead keeps its values once (asserted
  identical across pieces) with geometry unioned (skill §2/§2a).
- Condo units mapped as tiny per-unit polygons (no stacked record, no common-area parcel) are
  merged per development onto a nominal-value common-area parcel when one contains them
  (Pebblebrook Village), else onto the convex hull of the units minus other parcels' land
  (skill §6b fallback). Fee-simple townhomes keep their own small lots.
- ~3,100 polygons inside the city have no roll record (no value/class/year) and are dropped.
- Values are TAD's 2026 appraisal (LAND_VAL + IMPR_VAL = MARKET_VALUE).
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
sys.path.append(str(Path(__file__).resolve().parent))
from parcel_calculations import (add_improvement_ratio_fields, classify_property_refined,  # noqa: E402
                                 gis_area_sqft, land_value_singularities)
from tx_common import (CONDO_EXCLUDE_RE, CONDO_LEGAL_RE, NON_LAND_CATEGORIES, SQFT_PER_ACRE, categorize_sptb,  # noqa: E402
                       clip_centroid_within, dedup_accounts, fetch_arcgis_geojson, fix_geometry,
                       log, merge_condo_stubs, merge_stacked_footprints, owner_is_institutional, road_row_polygons,
                       owner_is_public)

CITY, STATE = "arlington", "tx"
DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / CITY
DATA_DIR.mkdir(parents=True, exist_ok=True)

PARCELS_LAYER = ("https://services5.arcgis.com/3ddLCBXe1bRt7mzj/arcgis/rest/services/"
                 "Parcels_Public_Vview/FeatureServer/0")
BOUNDARY_QUERY = ("https://services.arcgis.com/jXi5GuMZwfCYtZP9/arcgis/rest/services/"
                  "Municipal_Boundary/FeatureServer/0/query")
TAD_CITY_CODE = "24"   # TAD taxing-unit code for the City of Arlington
EXPECTED_TAX_YEAR = "2026"

# Arlington-specific public owners, on top of tx_common.PUBLIC_OWNER_KEYWORDS.
# Specific names only: "TARRANT COUNTY" alone also matches private owners
# (TARRANT COUNTY PROPERTIES LLC, SCHNEIDER TARRANT COUNTY LLC, TARRANT COUNTY FARM BUREAU).
ARLINGTON_PUBLIC = ["CITY OF ARLINGTON", "ARLINGTON CITY OF", "TARRANT COUNTY OF",
                    "TARRANT COUNTY COLLEGE", "TARRANT COUNTY JUNIOR COLLEGE",
                    "TARRANT COUNTY HOSPITAL", "MHMR OF TARRANT", "TARRANT REGIONAL WATER",
                    "TRINITY RIVER AUTHORITY", "ARLINGTON HOUSING AUTH", "TEXAS STATE OF",
                    "MUNICIPAL MANAGEMENT DISTRICT", "UNITED STATES POSTAL"]


# ── 1. Boundary + parcels ─────────────────────────────────────────────────────
def load_boundary():
    cache = DATA_DIR / f"{CITY}-{STATE}-boundary.parquet"
    if cache.exists():
        b = gpd.read_parquet(cache)
    else:
        r = requests.get(BOUNDARY_QUERY, params={"where": "MUNICIPALNAME='ARLINGTON'",
                         "outFields": "*", "outSR": 4326, "f": "geojson"}, timeout=120)
        r.raise_for_status()
        b = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        b.to_parquet(cache)
    return b.to_crs(4326).union_all()


boundary = load_boundary()
x0, y0, x1, y1 = boundary.bounds
raw = fetch_arcgis_geojson(PARCELS_LAYER,
                           geometry=dict(xmin=x0 - 0.002, ymin=y0 - 0.002,
                                         xmax=x1 + 0.002, ymax=y1 + 0.002),
                           cache=DATA_DIR / f"{CITY}-{STATE}-geometry.parquet")
log(f"Raw TAD rows in envelope: {len(raw):,}")

raw = raw[raw["COUNTYNAME"].astype(str).str.strip().eq("Tarrant")].copy()
years = raw["TAX_YEAR"].astype(str).value_counts().to_dict()
log(f"TAX_YEAR: {years}")
assert set(years) == {EXPECTED_TAX_YEAR}, f"unexpected tax years {years}"
# Polygons whose account has no roll record carry no value, class or year (the layer's
# roll join came up empty). They cannot be mapped as land value; kept aside for the condo
# diagnostics below, then dropped.
no_roll = raw["TAX_YEAR"].isna()
off_roll = raw[no_roll].copy()
raw = raw[~no_roll].copy()
log(f"Dropped {int(no_roll.sum()):,} polygons with no roll record")

raw["acct"] = raw["ACCOUNT"].astype(str).str.strip()
raw = raw[raw["acct"].ne("") & raw["acct"].ne("None")]
for c in ["LAND_VAL", "IMPR_VAL", "MARKET_VALUE", "LAND_ACRE"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw = fix_geometry(raw)

# ── 2. City limits (centroid within), cross-checked with TAD's taxing unit ───
parcel = clip_centroid_within(raw, boundary)
code24 = raw["CITY"].fillna("").astype(str).str.split(",").apply(lambda xs: TAD_CITY_CODE in [x.strip() for x in xs])
inside_ids, coded_ids = set(parcel["acct"]), set(raw.loc[code24, "acct"])
log(f"City-limits clip -> {len(parcel):,} rows ({parcel['acct'].nunique():,} accounts); "
    f"TAD CITY=24 accounts: {len(coded_ids):,}")
log(f"  in boundary but not coded 24: {len(inside_ids - coded_ids):,} | "
    f"coded 24 but outside boundary: {len(coded_ids - inside_ids):,}")

# ── 3. Per-record classification + exemptions ────────────────────────────────
for c in ["APPRAISED_VALUE", "CITY_TAXABLE_VALUE"]:
    parcel[c] = pd.to_numeric(parcel[c], errors="coerce")
parcel["state_class"] = parcel["STATE_USE_CODE"].fillna("").astype(str).str.strip().str.upper()
parcel["PROPERTY_CATEGORY"] = parcel["state_class"].apply(categorize_sptb)
log(f"state_class: {parcel['state_class'].value_counts().head(25).to_dict()}")
# Totally exempt property. TAD rarely codes it as state class X (10 parcels citywide); it keeps
# the use code and zeroes the taxable value. The City of Arlington's taxable value being $0 on
# an appraised parcel is therefore the exemption signal (763 of 768 city-owned parcels).
# Homes with $0 taxable are mostly 100% disabled-veteran homesteads - personal exemptions on
# private homes - so for residential classes it also needs an institutional owner.
by_class = parcel["PROPERTY_CATEGORY"].eq("Exempt")
by_owner = owner_is_public(parcel["OWNER_NAME"], ARLINGTON_PUBLIC)
taxable0 = parcel["CITY_TAXABLE_VALUE"].eq(0) & parcel["APPRAISED_VALUE"].gt(0)
residential = parcel["state_class"].str.match(r"^[AB]")
by_value = taxable0 & (~residential | owner_is_institutional(parcel["OWNER_NAME"]))
exempt = by_class | by_owner | by_value
log(f"Exempt: state class X {int(by_class.sum()):,} | public owner {int(by_owner.sum()):,} | "
    f"$0 city taxable {int(by_value.sum()):,} | total {int(exempt.sum()):,}")
log("  public-owner matches without $0 taxable (audit): "
    f"{parcel.loc[by_owner & ~taxable0, 'OWNER_NAME'].value_counts().head(25).to_dict()}")
log(f"  $0-taxable homes kept as private (personal exemptions): {int((taxable0 & residential & ~exempt).sum()):,}")
parcel["exemption_flag"] = exempt.astype(int)
parcel = parcel[parcel["exemption_flag"].eq(0)].copy()
dropped = parcel["PROPERTY_CATEGORY"].isin(NON_LAND_CATEGORIES)
log(f"Dropping non-land categories: {parcel.loc[dropped, 'PROPERTY_CATEGORY'].value_counts().to_dict()}")
parcel = parcel[~dropped].copy()

# ── 4. Stacked records summed onto their footprint, then account dedup ───────
# Condo units (all on the complex polygon), undivided-interest owners and boundary-split
# accounts share one polygon; their values are partial and are summed (skill §2a/§6).
parcel = merge_stacked_footprints(parcel, "acct", ["LAND_VAL", "IMPR_VAL", "MARKET_VALUE"],
                                  area_col="LAND_ACRE", dup_subset=["OWNER_NAME"])
parcel = dedup_accounts(parcel, "acct", ["LAND_VAL", "IMPR_VAL", "MARKET_VALUE", "STATE_USE_CODE"])
assert not parcel["acct"].duplicated().any()
# Condo developments drawn as unit outlines (Parkside, Wellington Place I, S Arlington Prof
# Bldg...) are merged onto their land (skill §6b): their own §23.18 common-area parcel, else a
# nominal parent, else the unit hull grown into unparcelled ground (never into another parcel,
# exempt or off-roll ones included, nor public road right-of-way).
roads_cache = DATA_DIR / f"{CITY}-{STATE}-roads.parquet"
if roads_cache.exists():
    roads = gpd.read_parquet(roads_cache)
else:
    roads = road_row_polygons(boundary.bounds)
    roads.to_parquet(roads_cache)
blockers = gpd.GeoDataFrame(pd.concat([raw[["acct", "geometry"]], off_roll[["geometry"]]],
                                      ignore_index=True), geometry="geometry", crs=4326)
parcel = merge_condo_stubs(parcel, "PARCEL_LEGAL_DESCRIPTION", ["LAND_VAL", "IMPR_VAL", "MARKET_VALUE"],
                           area_col="LAND_ACRE", blockers=blockers, roads=roads)
# Remaining association common areas (Texas Tax Code §23.18: valued at a nominal $1 because
# their value is already in the members' lots) are not land-value parcels: shipped as-is they
# render as $0 land and count as "Vacant". Drop them like exempt land.
legal = parcel["PARCEL_LEGAL_DESCRIPTION"].fillna("").str.upper()
nominal_common = legal.str.contains("COMMON") & parcel["LAND_VAL"].fillna(0).le(1000) \
    & parcel["land_source"].isna()
log(f"Dropping {int(nominal_common.sum()):,} nominal-value association common areas "
    f"({parcel.loc[nominal_common].to_crs(parcel.estimate_utm_crs()).area.sum() / 4046.86:,.0f} acres)")
parcel = parcel[~nominal_common].copy()
log(f"After stack merge + dedup + condo merge -> {len(parcel):,}")

ex = parcel.copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex["land_value"] = ex["LAND_VAL"].astype(float)
ex["improvement_value"] = ex["IMPR_VAL"].astype(float)
ex["property_land_use_refined"] = classify_property_refined(ex, fetch_footprints=False)
log(f"After exempt/refine -> {len(ex):,}")

# ── 5. Canonical fields — appraiser LAND_ACRE denominator (geodesic fallback) ─
log("Computing GIS areas...")
ex["geom_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
ex["reported_sqft"] = ex["LAND_ACRE"] * SQFT_PER_ACRE
ex.loc[~(ex["reported_sqft"] >= 1), "reported_sqft"] = np.nan
ratio = (ex["geom_area_sqft"] / ex["reported_sqft"]).replace([np.inf, -np.inf], np.nan)
log(f"Area cross-check GIS / LAND_ACRE: median {ratio.median():.3f}, "
    f"off by >25%: {int((ratio.sub(1).abs() > .25).sum()):,} of {int(ratio.notna().sum()):,}")
# LAND_ACRE agrees with the polygon (median ratio 1.001) for ~95k parcels, but on ~6.7k it is
# garbage: square feet in the acres field (3,136 "acres" on a 3,038 sqft lot), implausible
# fractions (0.0001 ac on a 7,110 sqft lot -> $9,183/sqft), or well under the drawn lot.
# Trust it only within 25% of the polygon; otherwise use the polygon.
use_reported = (ex["reported_sqft"] > 0) & (ratio.sub(1).abs() <= 0.25)
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["geom_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = ex["MARKET_VALUE"].where(ex["MARKET_VALUE"] > 0,
                                                   ex["land_value"] + ex["improvement_value"])
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")
# TAD property page by 8-digit account number.
ex["link"] = "https://www.tad.org/property?account=" + ex["acct"].astype(str)

# ── 6. Export ────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant"]
final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final = gpd.GeoDataFrame(fix_geometry(gpd.GeoDataFrame(final, geometry="geometry", crs=4326)),
                         geometry="geometry", crs="EPSG:4326")
out = DATA_DIR / f"{CITY}-{STATE}-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"{CITY}-{STATE}-parcels_{datetime.now():%Y_%m_%d}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"land ${final.current_full_land_value.sum():,.0f}  impr ${final.improvement_value.sum():,.0f}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")

# ── §6a smoke alarms: condo / unit stubs ──────────────────────────────────────
log("--- condo/stub smoke alarms (skill §6a) ---")
a = ex["geom_area_sqft"]
lv = pd.to_numeric(final["land_value_per_sqft"], errors="coerce")
log(f"  footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
log(f"  sub-500 / sub-1000 sqft footprints: {int((a < 500).sum()):,} / {int((a < 1000).sum()):,}")
log(f"  land $/sqft p50/p95/p99/max: ${lv.median():,.2f} / ${lv.quantile(.95):,.2f} / "
    f"${lv.quantile(.99):,.2f} / ${lv.max():,.2f}")
rp = final.geometry.representative_point()
vc = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)).value_counts()
log(f"  stacked clusters: {int((vc > 1).sum())}, max stack: {int(vc.max())}")
holes = final.geometry.apply(lambda g: sum(
    len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
log(f"  parcels with interior rings (holes): {int((holes > 0).sum()):,}")
log(f"  zero land value (renders as gp-error): "
    f"{int((final['current_full_land_value'].fillna(0) <= 0).sum()):,}")
log(f"  bounds: {[round(v, 4) for v in final.total_bounds]}")

# ── Singularity sweep: land $/sqft >= 4x the 300 m neighbourhood and in the city's top 10% ──
# Policy: artifacts (value on the wrong footprint) are fixed above at the source; slivers are
# hidden (likely_remnant); what remains must be genuine appraiser values, which are published.
# Any condo still flagged means a development's land was not found - investigate it.
sing = land_value_singularities(final)
legal_f = ex.loc[sing.index, "PARCEL_LEGAL_DESCRIPTION"].fillna("").str.upper()
cat_f = ex.loc[sing.index, "property_land_use_category"]


def _cause(i):
    lg, cat = legal_f[i], cat_f[i]
    if final.loc[i, "likely_remnant"] == 1:
        return "sliver <500 sqft (hidden)"
    if pd.notna(ex.loc[i, "land_source"]):
        if ex.loc[i, "land_source"] in ("common_area_parcel", "nominal_parent"):
            # Merged onto the development's own land (units + §23.18 common area): the value
            # now sits on the whole development; what remains is real pricing.
            return "development on its common-area land (appraiser value, published)"
        return f"condo, land from {ex.loc[i, 'land_source']} (UNRESOLVED)"
    if re.search(CONDO_LEGAL_RE, lg) and not re.search(CONDO_EXCLUDE_RE, lg):
        if ex.loc[i, "n_records"] > 1:
            # Units stacked on the complex's own plat polygon (common-area record included):
            # the value already sits on the development's full land as TAD draws it.
            return "condo complex on its plat polygon (appraiser value, published)"
        return "condo unit not merged (UNRESOLVED)"
    if re.search(r"TOWNHOME|TOWNHOUSE|TNHSE", lg) or sing.loc[i, "footprint_sqft"] < 2500:
        return "small fee-simple lot (appraiser value, published)"
    if cat in ("Commercial", "Industrial", "Multifamily", "Vacant Commercial"):
        return "commercial/multifamily land (appraiser value, published)"
    if cat in ("Single Family", "Vacant Residential"):
        return "residential lot (appraiser value, published)"
    return "other (review)"


sing["cause"] = [_cause(i) for i in sing.index]
sing["account"] = ex.loc[sing.index, "acct"].values
sing["legal"] = legal_f.values
sing["address"] = ex.loc[sing.index, "SITUS_ADDR"].values
sing.drop(columns="geometry", errors="ignore").to_csv(DATA_DIR / f"{CITY}-{STATE}-singularities.csv")
summary = sing.groupby("cause").agg(n=("ratio", "size"), median_ratio=("ratio", "median"),
                                    max_ratio=("ratio", "max"), land=("land_value", "sum"))
log(f"Singularity sweep: {len(sing):,} parcels >= 4x neighbours (of {len(final):,}); by cause:\n"
    f"{summary.round(1).to_string()}")
unresolved = sing["cause"].str.contains("UNRESOLVED")
if unresolved.any():
    log(f"  UNRESOLVED: {sing.loc[unresolved, ['account', 'ratio', 'legal']].head(20).to_string()}")
log("DONE")
