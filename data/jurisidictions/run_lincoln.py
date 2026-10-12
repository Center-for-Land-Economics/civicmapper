#!/usr/bin/env python3
"""
Build the City of Lincoln, NE canonical parcel parquet (Lancaster County).

Lincoln is Nebraska's capital and second-largest city (pop. ~295,000), home to the University
of Nebraska-Lincoln. Requested by Josh Duff, who pointed at gis.lincoln.ne.gov.

SOURCES — two public ArcGIS layers, no token, joined on the 13-digit Lancaster County parcel id:

1. GEOMETRY + county attributes: the joint City of Lincoln / Lancaster County Assessor
   TaxParcels layer
     https://gis.lincoln.ne.gov/public/rest/services/Assessor/TaxParcels/FeatureServer/0
   122,699 countywide polygons: PARCELID, CLASSCD (R1/R2/C1/C2/A1/A2), PRIMEUSE (a real use
   taxonomy), CNTASSDVAL (current total assessed value), condo fields (HighRise / UnitNumber).
   BUT ITS LAND VALUE FIELD IS EMPTY: LNDVALUE is NULL on all 122,699 rows (as are every
   taxable-value and taxes-owed field). The city GIS publishes NO land/improvement split
   anywhere — TaxParcels, ParcelInformation, PublicAnalytics, the /integration and /hosted
   server instances and the city's portal were all checked.

2. LAND / IMPROVEMENT SPLIT: the State of Nebraska's statewide parcel layer, which the OCIO
   compiles from each county's CAMA export
     https://giscat.ne.gov/enterprise/rest/services/StatewideParcelsExternal/FeatureServer/0
   County_ID '109' = Lancaster: 121,664 rows with Land_Value / Improvements_Value /
   Other_Impr_Value / Ag_Land_Value / Total_Assessed_Value (the four components sum to the
   total on 99.99% of rows). CAMAUpdate 2026-01-01.

   VINTAGE: these are the 2025 ROLL. Verified against the Assessor's own value history for
   4100 NW 57TH ST (0412201001000): 2025 land $52,000 / building $127,600 / total $179,600,
   exactly the statewide row. The city layer's CNTASSDVAL is the newer 2026 total ($203,400
   there; citywide the state total runs a median 3.3% below it). The 2026 roll has no public
   land split yet, so the whole map ships on the consistent 2025 roll rather than mixing a
   2026 total with a 2025 land figure. City boundary: CityCounty/CityLimits/MapServer/5
   (Lincoln City Limits, 2026, 105.1 sq mi).

Nebraska assesses residential and commercial real property at 100% of actual (market) value
and agricultural land at 75% (Neb. Rev. Stat. 77-201). So land/improvement values here are
market value, except Ag_Land_Value (49 parcels, $7.1M inside the city), which is 75%.

EXEMPTION — THE HARD PART. Lancaster County APPRAISES EXEMPT PROPERTY AT FULL VALUE (churches
carry $2-5M, Bryan Medical Center $321M) and neither source has a usable tax-status flag:
  - the statewide layer's Ownership_Type = 'Exempt' is NOT full exemption. It marks ANY
    exemption including the partial HOMESTEAD exemption (verified: 4100 NW 57TH ST is
    'Exempt' and the Assessor shows "Homestead Exemption"; 6,956 single-family homes carry it)
    and TIF-district apartment towers. Using it would delete ~15,600 taxable parcels.
  - Property class (Classification_Code digits 3-4) is '00' on every Lancaster row.
So exemption is composed from three signals (see `is_exempt`):
  1. PRIMEUSE in Public Use / Church / School / Central Assessed (state-assessed railroad and
     utility property — excluded like Utility elsewhere);
  2. statewide Ownership_Type in County / State / Federal (government owner);
  3. an institutional OWNER-NAME keyword (City of Lincoln, Board of Regents, Housing Authority,
     school district, church, nonprofit hospital, ...) matched at fetch time — only the matched
     keyword is cached, the owner name never touches disk (issue #12). Residential parcels are
     exempted on GOVERNMENT keywords only: a church- or hospital-owned house is often a
     taxable rental.

CONDOS — two patterns, both merged (skill §6, §2a sum rule: units are distinct value records):
  A. STACKED: high-rise and office condo units each carry an IDENTICAL copy of the whole lot
     polygon (Van Dorn Villa: 192 copies). Collapsed per footprint, values summed.
  B. BASE ACCOUNT + UNIT STUBS: a $0 'Condo Base Account' parcel (PARCELID ending 000) holds the
     regime's land polygon and its units (same first 10 digits, suffix 001..) are drawn as
     building-footprint stubs inside it (e.g. GEMINID CONDOMINIUM: a 94,864 sqft base + 15
     service-bay units of 1,800-8,400 sqft). Merged DOWN onto the base polygon (holes filled),
     units' values summed — skill §6b.

Outputs:
- data/jurisidictions/data/lincoln/lincoln-ne-parcels.parquet
- data/jurisidictions/data/lincoln/lincoln-ne-parcels_YYYY_MM_DD.parquet
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
sys.path.append(str(Path(__file__).resolve().parent))
from parcel_calculations import (  # noqa: E402
    add_improvement_ratio_fields,
    classify_property_refined,
    gis_area_sqft,
)
import lincoln_fetch as lf  # noqa: E402
from lincoln_fetch import log  # noqa: E402

_ap = argparse.ArgumentParser(description="Build the Lincoln, NE parcel parquet.")
_ap.add_argument("--no-condo-cap", action="store_true",
                 help="ship condo regimes' summed per-unit site values unchanged (see CONDO_CAP_RATIO)")
ARGS = _ap.parse_args()

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "lincoln"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SQFT_PER_ACRE = 43560.0
SQM_TO_SQFT = 10.763910416709722
UTM = "EPSG:32614"  # UTM 14N — Lincoln

EXEMPT_USES = {"Public Use", "Church", "School", "Central Assessed",
               "Game and Parks In Lieu of Tax", "Utility/Railroad"}
GOVT_OWNERSHIP = {"County", "State", "Federal"}
GOVT_KEYWORDS = {"CITY OF LINCOLN", "LINCOLN CITY OF", "LANCASTER COUNTY", "COUNTY OF LANCASTER",
                 "STATE OF NEBRASKA", "NEBRASKA STATE OF", "UNITED STATES", " USA ",
                 "BOARD OF REGENTS", "HOUSING AUTHORITY", "SCHOOL DIST", "PUBLIC SCHOOLS",
                 "AIRPORT AUTHORITY", "JOINT PUBLIC AGENCY", "PUBLIC BUILDING COMMISSION",
                 "NATURAL RESOURCES DIST"}
RESIDENTIAL_USES = {"Single Family", "Townhouse", "Duplex/Triplex", "Conversion-Apt",
                    "Multi-Family", "High Rise Condos", "Acreage", "Mobile Home",
                    "Mobile Home Court"}
UNIMPROVED_CLASSES = {"R2", "C2", "A2"}
# A merged condo regime whose summed per-unit land runs above this multiple of its neighbours'
# land $/sqft gets its land re-split at the neighbour rate (total value unchanged). See below.
CONDO_CAP_RATIO = 3.0
CONDO_K = 15


def _polygonal(g):
    """Keep only the polygonal part (make_valid can return a GeometryCollection)."""
    if g is None or g.is_empty:
        return g
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    if g.geom_type == "GeometryCollection":
        polys = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon") and not p.is_empty]
        if not polys:
            return None
        return polys[0] if len(polys) == 1 else unary_union(polys)
    return None


def _union(geoms):
    gs = [g for g in geoms if g is not None and not g.is_empty]
    try:
        return unary_union(gs)
    except Exception:  # noqa: BLE001 - last-resort repair for stubborn topology
        return unary_union([_polygonal(make_valid(g)).buffer(0) for g in gs])


def _fill_holes(g):
    """Solid exterior: a condo base polygon's holes are punched-out unit footprints (§6b)."""
    if g is None or g.is_empty:
        return g
    if g.geom_type == "Polygon":
        return Polygon(g.exterior)
    if g.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in g.geoms])
    return g


def _dominant(series):
    s = series.dropna()
    return s.mode().iloc[0] if len(s) else None


# ── fetch ─────────────────────────────────────────────────────────────────────
boundary = lf.fetch_boundary(DATA_DIR / "lincoln-boundary.geojson")
state = lf.fetch_state_values(DATA_DIR / "lincoln-ne-state-values.parquet")
raw = lf.fetch_county_parcels(DATA_DIR / "lincoln-ne-county-geometry.parquet", boundary.total_bounds)
owners = lf.fetch_owner_flags(DATA_DIR / "lincoln-ne-owner-flags.parquet", boundary.total_bounds)
log(f"County rows in envelope: {len(raw):,} | statewide Lancaster rows: {len(state):,}")

# ── clip to the authoritative city limits (centroid-within) ───────────────────
raw = raw.set_crs("EPSG:4326") if raw.crs is None else raw.to_crs("EPSG:4326")
raw = raw[raw["PARCELID"].notna() & raw.geometry.notna()].copy()
n_bad = int((~raw.geometry.is_valid).sum())
raw["geometry"] = raw["geometry"].apply(
    lambda g: g if g.is_valid else _polygonal(make_valid(g)))
raw = raw[raw.geometry.notna() & ~raw.geometry.is_empty].copy()
log(f"Repaired {n_bad:,} invalid source geometries")
city = boundary.geometry.union_all()
inside = gpd.GeoSeries(raw.geometry.representative_point(), crs="EPSG:4326").within(city)
raw = raw[inside.values].copy()
log(f"Inside Lincoln city limits: {len(raw):,} rows / {raw['PARCELID'].nunique():,} PARCELIDs")

# ── multi-polygon PARCELIDs: union geometry, FIRST value (skill §2/§2a) ───────
dup = raw[raw["PARCELID"].duplicated(keep=False)]
log(f"PARCELIDs with >1 geometry row: {dup['PARCELID'].nunique():,} ({len(dup):,} rows)")
if len(dup):
    assert int((dup.groupby("PARCELID")["CNTASSDVAL"].nunique(dropna=False) > 1).sum()) == 0, \
        "multi-row PARCELID with differing values — not a broadcast, re-check before taking first"
attr_cols = [c for c in raw.columns if c not in ("geometry", "PARCELID")]
parcel = raw.groupby("PARCELID", sort=False).agg({c: "first" for c in attr_cols}).reset_index()
parcel["geometry"] = raw.groupby("PARCELID", sort=False)["geometry"].apply(_union).values
parcel = gpd.GeoDataFrame(parcel, geometry="geometry", crs="EPSG:4326")
assert parcel["PARCELID"].is_unique

# ── join the 2025 land/improvement split from the statewide layer ─────────────
st = state[state["Parcel_ID"].fillna("").str.len() > 0].drop_duplicates("Parcel_ID")
for c in ["Land_Value", "Improvements_Value", "Other_Impr_Value", "Ag_Land_Value",
          "Total_Assessed_Value"]:
    st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0)
parcel = parcel.merge(st, left_on="PARCELID", right_on="Parcel_ID", how="left")
parcel = gpd.GeoDataFrame(parcel, geometry="geometry", crs="EPSG:4326")
matched = parcel["Parcel_ID"].notna()
parcel["land_val"] = parcel["Land_Value"] + parcel["Ag_Land_Value"]
parcel["bld_val"] = parcel["Improvements_Value"] + parcel["Other_Impr_Value"]
parcel["tot_appr_val"] = parcel["Total_Assessed_Value"]
parcel["value_source"] = np.where(matched, "state-2025", None)
# Parcels platted after the 2025 roll have no statewide record. An UNIMPROVED one (R2/C2/A2 —
# 99.9% carry zero improvements in the roll) is all land, so its 2026 county total IS its land
# value. An improved one cannot be split and is dropped (logged below).
newlot = ~matched & parcel["CLASSCD"].isin(UNIMPROVED_CLASSES) & parcel["CNTASSDVAL"].gt(0)
parcel.loc[newlot, "land_val"] = parcel.loc[newlot, "CNTASSDVAL"]
parcel.loc[newlot, "bld_val"] = 0.0
parcel.loc[newlot, "tot_appr_val"] = parcel.loc[newlot, "CNTASSDVAL"]
parcel.loc[newlot, "value_source"] = "county-2026-unimproved"
nosplit = ~matched & ~newlot
log(f"Joined to the 2025 statewide roll: {int(matched.sum()):,}; new unimproved lots valued from "
    f"the 2026 county total: {int(newlot.sum()):,} (${parcel.loc[newlot, 'land_val'].sum() / 1e6:,.1f}M); "
    f"unmatched improved/unclassed, no land split -> dropped: {int(nosplit.sum()):,} "
    f"(2026 total ${parcel.loc[nosplit, 'CNTASSDVAL'].fillna(0).sum() / 1e6:,.1f}M, "
    f"classes {parcel.loc[nosplit, 'CLASSCD'].fillna('none').value_counts().to_dict()})")
parcel = parcel[~nosplit].copy()
for c in ["land_val", "bld_val", "tot_appr_val"]:
    parcel[c] = pd.to_numeric(parcel[c], errors="coerce").fillna(0)

# ── exemption (row level, before the condo merge) ─────────────────────────────
parcel = parcel.merge(owners, on="PARCELID", how="left")
parcel = gpd.GeoDataFrame(parcel, geometry="geometry", crs="EPSG:4326")
kw = parcel["inst_owner_kw"]
is_res = parcel["PRIMEUSE"].isin(RESIDENTIAL_USES)
by_use = parcel["PRIMEUSE"].isin(EXEMPT_USES)
by_govt = parcel["Ownership_Type"].isin(GOVT_OWNERSHIP)
by_kw = kw.notna() & (~is_res | kw.isin(GOVT_KEYWORDS))
parcel["is_exempt"] = by_use | by_govt | by_kw
log(f"Exempt rows: {int(parcel['is_exempt'].sum()):,} — by use {int(by_use.sum()):,}, govt owner "
    f"{int((by_govt & ~by_use).sum()):,}, institutional owner keyword "
    f"{int((by_kw & ~by_use & ~by_govt).sum()):,}; land ${parcel.loc[parcel['is_exempt'], 'land_val'].sum() / 1e9:,.2f}B")
log(f"  owner-keyword-only exemptions by keyword: "
    f"{kw[by_kw & ~by_use & ~by_govt].value_counts().to_dict()}")

# ── condo merge ───────────────────────────────────────────────────────────────
parcel["p10"] = parcel["PARCELID"].str[:10]
parcel["suffix"] = parcel["PARCELID"].str[10:]
parcel["_sqft"] = parcel.to_crs(UTM).geometry.area * SQM_TO_SQFT
parcel["_units"] = 1
parcel["condo_merged"] = 0
parcel["CATEGORY_SRC"] = parcel["PRIMEUSE"]
VALUE_COLS = ["land_val", "bld_val", "tot_appr_val", "BLDGAREA", "RESFLRAREA"]
for c in ["BLDGAREA", "RESFLRAREA"]:
    parcel[c] = pd.to_numeric(parcel[c], errors="coerce").fillna(0)


def _merge_row(units, geom, rep_id):
    row = units.sort_values("land_val", ascending=False).iloc[0].to_dict()
    row.update({c: units[c].sum() for c in VALUE_COLS})
    row.update({
        "PARCELID": rep_id, "geometry": geom, "_units": len(units), "condo_merged": 1,
        "CATEGORY_SRC": _dominant(units["PRIMEUSE"]), "is_exempt": False,
        "value_source": _dominant(units["value_source"]),
    })
    return row


# A. base account + units, merged DOWN onto the base polygon
base = parcel[parcel["PRIMEUSE"].eq("Condo Base Account") & parcel["suffix"].eq("000")]
taxable = parcel[~parcel["is_exempt"]]
dev_rows, consumed = [], set()
for _, b in base.iterrows():
    units = taxable[taxable["p10"].eq(b["p10"]) & taxable["suffix"].ne("000")]
    if units.empty:
        continue
    geom = _fill_holes(_union([b.geometry] + list(units.geometry)))
    dev_rows.append(_merge_row(units, geom, b["PARCELID"]))
    consumed.update(units["PARCELID"])
    consumed.add(b["PARCELID"])
n_a_units = sum(r["_units"] for r in dev_rows)
log(f"Condo A (base account + units): {n_a_units:,} units -> {len(dev_rows):,} developments")
rest = parcel[~parcel["PARCELID"].isin(consumed)].copy()
# A base account left unmerged because its units post-date the 2025 roll (e.g. 1255 LIBRA DR:
# five 2,700 sqft units with no 2025 record) still carries the whole lot's pre-conversion land
# value, on a polygon with the unit footprints punched out. Give it the solid lot.
_lone_base = rest["PRIMEUSE"].eq("Condo Base Account") & rest["land_val"].gt(0)
rest.loc[_lone_base, "geometry"] = rest.loc[_lone_base, "geometry"].apply(_fill_holes)
log(f"  unmerged base accounts carrying land value (holes filled): {int(_lone_base.sum()):,}")

# B. stacked units on an identical footprint (no base account in the layer, or a different p10)
rp = rest.geometry.representative_point()
rest["_fp"] = (rp.x.round(6).astype(str) + "," + rp.y.round(6).astype(str) + ","
               + rest["_sqft"].round(-1).astype(str))
fp_n = rest.groupby("_fp")["PARCELID"].transform("size")
stk = rest[fp_n.gt(1)]
stack_rows, stack_consumed = [], set()
for _, grp in stk.groupby("_fp"):
    units = grp[~grp["is_exempt"] & grp["PRIMEUSE"].ne("Condo Base Account")]
    stack_consumed.update(grp["PARCELID"])
    if units.empty:
        continue
    stack_rows.append(_merge_row(units, _union(list(grp.geometry)), units["PARCELID"].iloc[0]))
log(f"Condo B (stacked identical footprints): {stk['_fp'].nunique():,} stacks / {len(stk):,} rows "
    f"-> {len(stack_rows):,} parcels ({sum(r['_units'] for r in stack_rows):,} taxable units summed)")
rest = rest[~rest["PARCELID"].isin(stack_consumed)].drop(columns="_fp")
merged = pd.DataFrame(dev_rows + stack_rows)
parcel = gpd.GeoDataFrame(pd.concat([rest, merged.drop(columns="_fp", errors="ignore")],
                                    ignore_index=True), geometry="geometry", crs="EPSG:4326")
assert parcel["PARCELID"].is_unique
if len(merged):
    _m = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    _psf = _m["land_val"] / (_m.to_crs(UTM).geometry.area * SQM_TO_SQFT)
    log(f"  merged condo land $/sqft p50 ${_psf.median():,.2f} p95 ${_psf.quantile(.95):,.2f} "
        f"max ${_psf.max():,.2f}")
    _m["psf"] = _psf.values
    _m["sqft"] = (_m.to_crs(UTM).geometry.area * SQM_TO_SQFT).values
    log("  hottest merged developments:\n" + _m.sort_values("psf", ascending=False).head(10)[
        ["PARCELID", "CNVYNAME", "CATEGORY_SRC", "_units", "land_val", "sqft", "psf"]].to_string())

# ── exemption filter + no-land drop ───────────────────────────────────────────
parcel["exemption_flag"] = parcel["is_exempt"].astype(int)
zero = ~parcel["is_exempt"] & parcel["land_val"].le(0)
log(f"Excluded exempt: {int(parcel['is_exempt'].sum()):,}; excluded taxable-but-no-land-value "
    f"(HOA outlots, $0 base accounts, garages): {int(zero.sum()):,} "
    f"{parcel.loc[zero, 'PRIMEUSE'].value_counts().head(6).to_dict()}")
ex = parcel[~parcel["is_exempt"] & ~zero].copy()
log(f"Shipped parcels -> {len(ex):,}")

# ── classification (PRIMEUSE is the assessor's own taxonomy) ──────────────────
USE_MAP = {
    "Single Family": "Single Family", "Acreage": "Single Family",
    "Townhouse": "Townhome",
    "Duplex/Triplex": "Multifamily", "Multi-Family": "Multifamily", "Conversion-Apt": "Multifamily",
    "High Rise Condos": "Condominium",
    "Mobile Home Court": "Mobile Home", "Mobile Home": "Mobile Home",
    "Service": "Commercial", "Office": "Commercial", "Retail": "Commercial",
    "Store Front": "Commercial", "Fast Food Restaurant": "Commercial",
    "Convenience Store": "Commercial", "Hotel/Motel": "Commercial",
    "Other - Commercial": "Commercial",
    "Industrial": "Industrial",
    "Multi-Use": "Mixed Use",
    "Agricultural": "Agricultural / Rural",
    "Vacant Land": "Vacant Land",
    # HOA common areas, IOLL land and assemblage remnants — median $0.30/sqft. Not developable
    # vacant land, so deliberately NOT labelled 'Vacant' (it would flood the vacant tab).
    "Vacant land with IOLL/Common Area/Assemblage": "Common Area / IOLL",
}


def categorize(use, cls, impr):
    use = use if isinstance(use, str) else ""
    cls = cls if isinstance(cls, str) else ""
    if use == "Agricultural" or (cls in ("A1", "A2") and use in ("", "Acreage")):
        return "Agricultural / Rural"
    if cls in UNIMPROVED_CLASSES and (impr or 0) <= 0 and use not in (
            "Vacant land with IOLL/Common Area/Assemblage",):
        return "Vacant Land"     # R2/C2 lots: assessor class says unimproved
    if use in USE_MAP:
        return USE_MAP[use]
    if cls and cls.startswith("C"):
        return "Commercial"
    if cls and cls.startswith("R"):
        return "Single Family"
    return "Vacant Land" if (impr or 0) <= 0 else "Other"


ex["property_land_use_category"] = [
    categorize(u, c, i) for u, c, i in zip(ex["CATEGORY_SRC"], ex["CLASSCD"], ex["bld_val"])]
ex["land_value"] = ex["land_val"]
ex["improvement_value"] = ex["bld_val"]
ex["full_market_value"] = ex["tot_appr_val"]
ex["bld_ar"] = ex["BLDGAREA"].where(ex["BLDGAREA"] > 0, ex["RESFLRAREA"]).fillna(0)

# ── areas (geodesic polygon area: stacked/merged condos make stated areas meaningless) ──
ex["geometry"] = ex["geometry"].apply(
    lambda g: g if (g is None or g.is_valid) else _polygonal(make_valid(g)))
log("Computing geodesic areas...")
ex["land_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["land_area_sqft"] < 1, "land_area_sqft"] = np.nan
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["area_source"] = "gis"
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

# ── condo land: re-split implausible regimes at the neighbours' land rate ─────
# Lancaster values a condo unit's "land" as a FIXED PER-UNIT SITE VALUE set by the building
# (Georgian Place: $90,000 on each of 48 units; Lincoln Building: $90,000 x 50), not as a share
# of the lot. Summed onto the lot that is $4.3M on 5,495 sqft = $786/sqft, against $64.80/sqft
# — the assessor's own uniform downtown land rate — on every non-condo parcel within 500 m. Skill
# §6b step 5: a merged development's land $/sqft must land in the neighbours' range, or the
# summing is wrong. So where a merged regime runs > CONDO_CAP_RATIO x the median land $/sqft of
# its CONDO_K nearest non-condo neighbours, its land is re-split at that neighbour rate x the
# lot area and the remainder moves to improvements: the assessor's TOTAL is untouched, and the
# assessor's own land figure is kept in `assessor_land_value` (Boston precedent, run_boston.py).
# Regimes within range (most suburban townhouse/office condos) keep the assessor's sum.
# --no-condo-cap ships the summed site values unchanged.
from scipy.spatial import cKDTree  # noqa: E402

ex["assessor_land_value"] = ex["land_value"]
ex["condo_land_reallocated"] = 0
_pts = ex.to_crs(UTM).geometry.representative_point()
_xy = np.column_stack([_pts.x.to_numpy(), _pts.y.to_numpy()])
_psf = ex["land_value"] / ex["land_area_sqft"]
_donor = (ex["condo_merged"].eq(0) & ex["land_value"].gt(0) & ex["land_area_sqft"].ge(500)
          & ~ex["property_land_use_category"].isin(["Condominium", "Common Area / IOLL"])
          & ~ex["PRIMEUSE"].eq("High Rise Condos")).to_numpy()
_tgt = ex["condo_merged"].eq(1).to_numpy()
_, _idx = cKDTree(_xy[_donor]).query(_xy[_tgt], k=CONDO_K)
_nb = pd.Series(np.median(_psf.to_numpy()[_donor][_idx], axis=1), index=ex.index[_tgt])
_ratio = _psf[_tgt] / _nb
_hot = _ratio[_ratio > CONDO_CAP_RATIO].index
_new_land = (_nb[_hot] * ex.loc[_hot, "land_area_sqft"]).round(0)
_new_land = np.minimum(_new_land, ex.loc[_hot, "full_market_value"])
log(f"Condo land check: {int(_tgt.sum()):,} merged regimes; {len(_hot):,} run > {CONDO_CAP_RATIO:g}x "
    f"their neighbours' land $/sqft (summed ${ex.loc[_hot, 'land_value'].sum() / 1e6:,.1f}M vs "
    f"${_new_land.sum() / 1e6:,.1f}M at the neighbour rate) — "
    f"{'NOT re-split (--no-condo-cap)' if ARGS.no_condo_cap else 're-split, total unchanged'}")
log("  regimes over the cap:\n" + pd.DataFrame({
    "PARCELID": ex.loc[_hot, "PARCELID"], "name": ex.loc[_hot, "CNVYNAME"].str[:34],
    "units": ex.loc[_hot, "_units"], "summed_land": ex.loc[_hot, "land_value"],
    "psf": _psf[_hot].round(1), "neighbour_psf": _nb[_hot].round(1),
    "new_land": _new_land}).sort_values("psf", ascending=False).to_string())
if not ARGS.no_condo_cap and len(_hot):
    ex.loc[_hot, "land_value"] = _new_land
    ex.loc[_hot, "improvement_value"] = ex.loc[_hot, "full_market_value"] - _new_land
    ex.loc[_hot, "condo_land_reallocated"] = 1

ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other", "Agricultural / Rural", "Common Area / IOLL"),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    bld_ar_col="bld_ar", fetch_footprints=False)
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")
ex["link"] = ("https://orion.lancaster.ne.gov/appraisal/publicaccess/Property-Detail/PropertyNumber/"
              + ex["PARCELID"].astype(str))

# ── formula-allocation test (Provo lesson) ────────────────────────────────────
imp = ex[(ex["improvement_value"] > 0) & (ex["land_value"] > 0)]
_r = imp["land_value"] / (imp["land_value"] + imp["improvement_value"])
_t = pd.DataFrame({"n": _r.groupby(imp["property_land_use_category"]).size(),
                   "median_land_share": _r.groupby(imp["property_land_use_category"]).median().round(3),
                   "share_within_0.5pt_of_median": _r.groupby(imp["property_land_use_category"]).apply(
                       lambda s: round(float((abs(s - s.median()) < 0.005).mean()), 3)),
                   "distinct_land_values": imp.groupby("property_land_use_category")["land_value"].nunique()})
log("land share of total by category (a fixed share = formula-allocated land):\n"
    + _t.sort_values("n", ascending=False).to_string())

# ── export ───────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant", "assessor_land_value", "condo_land_reallocated"]
for c in COLUMNS:
    if c not in ex.columns:
        ex[c] = np.nan
final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")
assert final.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()
out = DATA_DIR / "lincoln-ne-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"lincoln-ne-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet",
                 index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"value_source: {ex['value_source'].value_counts().to_dict()}")
log(f"land value total: ${final['current_full_land_value'].sum() / 1e9:,.2f}B | "
    f"market value total: ${final['full_market_value'].sum() / 1e9:,.2f}B")

# ── §6a smoke alarms ─────────────────────────────────────────────────────────
log("--- condo/stub smoke alarms (skill §6a) ---")
a = ex["land_area_sqft"]
lv = pd.to_numeric(final["land_value_per_sqft"], errors="coerce")
shown = lv[final["likely_remnant"] == 0]
log(f"  footprint sqft p1/p5/p10/p50: {[round(a.quantile(q)) for q in (.01, .05, .10, .50)]}")
log(f"  sub-500 / sub-1000 sqft footprints: {int((a < 500).sum()):,} / {int((a < 1000).sum()):,}")
log(f"  land $/sqft ALL ROWS    p50/p95/p99/max: ${lv.median():,.2f} / ${lv.quantile(.95):,.2f} / "
    f"${lv.quantile(.99):,.2f} / ${lv.max():,.2f}")
log(f"  land $/sqft AS RENDERED p50/p95/p99/max: ${shown.median():,.2f} / ${shown.quantile(.95):,.2f} / "
    f"${shown.quantile(.99):,.2f} / ${shown.max():,.2f}  (likely_remnant excluded)")
holes = final.geometry.apply(lambda g: 0 if g is None else sum(
    len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
log(f"  parcels with interior rings (holes): {int((holes > 0).sum()):,}")
rp = final.geometry.representative_point()
_vc = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)).value_counts()
log(f"  stacked footprints: {int((_vc > 1).sum()):,} clusters, max stack {int(_vc.max())}")
log(f"  zero/neg land value (renders as gp-error): "
    f"{int((final['current_full_land_value'].fillna(0) <= 0).sum()):,}")
top = ex.assign(lvps=lv.values).sort_values("lvps", ascending=False).head(8)
log("  top land $/sqft parcels:\n" + top[["PARCELID", "SITEADDRESS", "property_land_use_category",
                                          "land_value", "land_area_sqft", "lvps", "_units"]].to_string())
log(f"  bounds: {[round(v, 4) for v in final.total_bounds]}")
log("DONE")
