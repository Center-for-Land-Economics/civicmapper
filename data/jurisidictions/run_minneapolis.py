#!/usr/bin/env python3
"""
Build the City of Minneapolis, MN canonical parcel parquet.

Minneapolis is in Hennepin County, but the CITY runs its own Assessor's office and publishes
its own roll, and both sources below are city-only, so no county clip is needed (the
geometry layer is the city's own parcel fabric; every row has CITY = 'MINNEAPOLIS').

Sources (both City of Minneapolis, AGOL org `afSMGVsC7QlRK1kZ`, public, no token):
- Values: "Assessing Department Parcel Data 2026" — a TABLE (no geometry), one row per
  building, 132,079 rows. ASMTYEAR 2026 / TAXYR 2027 (the 2026 assessment, taxes payable 2027).
  LANDVALUE / BUILDINGVALUE / TOTALVALUE are the assessor's ESTIMATED MARKET VALUE (Minnesota
  assesses at 100% of market; class rates are applied later, to tax capacity, not here).
    https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services/Assessing_Department_Parcel_Data_2026/FeatureServer/0
  The PIN column is literally named "F_No_column_name_" and is "p" + the 13-digit Hennepin PID.
- Geometry: the city's parcel fabric as published inside its "TylerParkingProd" service,
  layer 8 "PARCELS" (130,985 polygons, PID unique, data last edited 2026-05-13):
    https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services/TylerParkingProd/FeatureServer/8
  Why not Hennepin County's own LAND_PROPERTY/MapServer/1 or the Met Council regional parcel
  layers? gis.hennepin.us, arcgis.metc.state.mn.us and gisdata.mn.gov were all unreachable from
  the environment this was built in (skill §1: county hosts are often firewalled — use an AGOL
  mirror). This layer is the city's own copy, current, complete, and keyed on the same PID.
  Older city copies (Property_Information_Map, 2018) and UMN class-project copies were rejected.

JOIN: assessor PID -> geometry PID, inner. The assessor table repeats a PID once per BUILDING
(625 PIDs, 2-6 rows) with the parcel's values BROADCAST identically on every row — asserted
below, then `first` (skill §2; summing would multiply the value). Building above-ground area is
summed across those rows.

EXEMPT: the Minneapolis assessor APPRAISES exempt property too (skill §5, St. Louis County
pattern): 7,022 ISEXEMPT='Y' parcels carry $15.7B of land value (the University of Minnesota,
parks, city/county/state, churches, the airport-commission land...). Dropped on ISEXEMPT='Y',
the assessor's own flag. UTILITY (21 taxable) is dropped as well (skill §5), and so is
COMMERCIAL RAILROAD (319 taxable, $427M land): railroad operating property in Minnesota is
valued by the state Department of Revenue and apportioned onto placeholder records, so the
"land value" is not a land value at all. 120 records are $0, the median is $0.11/sqft, and
1-55 Hennepin Ave carry $223M on 8 stacked copies of a 0.7-acre polygon (PARCELAREA 0), which
rendered at $6,900/sqft. Railroad land behaves like utility land, so it is left off.

CONDOS / CO-OPS (skill §6): Hennepin maps every condo unit, condo garage/storage unit and
co-op unit as an IDENTICAL COPY of the development's lot polygon (966 stacks, up to 1,059
copies), and each unit carries its own land and building value (distinct value records, NOT a
broadcast). So every stack of identical footprints is collapsed to ONE parcel with land and
building SUMMED (skill §2a "sum vs first": distinct records on one footprint -> sum). The
development's COMMON AREA / CO-OP MASTER record ($0) sits in the same stack and simply adds 0.
COMMON AREA / CO-OP MASTER records that are NOT in a stack (townhouse-association land,
private drives) carry $0 and are dropped rather than shipped as $0 error parcels.

Classification: from the assessor's PROPERTYTYPE (29 readable labels). There is no
surface-parking land-use code (only "Parking Garage/Ramp" building use, which is a structure
and is judged on its ratio — parcel_calculations.classify_property_refined). Surface parking
is tagged afterwards from detected lots (data/scripts/apply_parking_category.py).

Outputs:
- data/jurisidictions/data/minneapolis/minneapolis-mn-parcels.parquet
- data/jurisidictions/data/minneapolis/minneapolis-mn-parcels_YYYY_MM_DD.parquet

~125k parcels -> PMTiles + H3 (bake with --drop-remnants; the city sets hideRemnants).

Owner names are NOT shipped (OWNERNAME is read but never exported; upload_city_dev.py also
blocks owner-name columns).

Link: Hennepin County's property-information page, pidresult.jsp?pid=<13-digit PID>. The County
host could not be reached from the build environment, so the link format was not live-checked.
"""
from __future__ import annotations

import json
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
from parcel_calculations import (add_improvement_ratio_fields, classify_property_refined,  # noqa: E402
                                 gis_area_sqft)

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "minneapolis"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "minneapolis-mn-geometry.parquet"
ROLL_CACHE = DATA_DIR / "minneapolis-mn-assessor-2026.parquet"

ORG = "https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services"
ROLL_URL = f"{ORG}/Assessing_Department_Parcel_Data_2026/FeatureServer/0"
GEOM_URL = f"{ORG}/TylerParkingProd/FeatureServer/8"
LINK_FMT = "https://www16.co.hennepin.mn.us/pins/pidresult.jsp?pid={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
UTM = "EPSG:26915"  # NAD83 / UTM 15N
SQM_TO_SQFT = 10.763910416709722


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── fetch (ObjectId-range pagination, cached) ──────────────────────────────────
def pull(url: str, cache: Path, geo: bool, page: int) -> pd.DataFrame:
    if cache.exists():
        log(f"Using cached {cache.name}")
        return gpd.read_parquet(cache) if geo else pd.read_parquet(cache)
    ids = requests.get(f"{url}/query", params={"where": "1=1", "returnIdsOnly": "true", "f": "json"},
                       headers=HEADERS, timeout=120).json()
    oids, key = sorted(ids["objectIds"]), ids["objectIdFieldName"]
    log(f"Pulling {len(oids):,} rows from {url.split('/services/')[1]}")
    parts = []
    for i in range(0, len(oids), page):
        lo, hi = oids[i], oids[min(i + page, len(oids)) - 1]
        for attempt in range(5):
            try:
                r = requests.get(f"{url}/query", params={
                    "where": f"{key}>={lo} AND {key}<={hi}", "outFields": "*",
                    "returnGeometry": str(geo).lower(), "outSR": 4326,
                    "f": "geojson" if geo else "json"}, headers=HEADERS, timeout=240)
                r.raise_for_status()
                j = r.json()
                break
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt + 1} @ {lo}: {type(e).__name__}: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"pull failed at objectid {lo}")
        parts.append(gpd.GeoDataFrame.from_features(j["features"], crs="EPSG:4326") if geo
                     else pd.DataFrame([f["attributes"] for f in j["features"]]))
    df = pd.concat(parts, ignore_index=True)
    df = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326") if geo else df
    df.to_parquet(cache, index=False)
    log(f"  cached -> {cache.name} ({len(df):,} rows)")
    return df


# ── 1. roll ────────────────────────────────────────────────────────────────────
roll = pull(ROLL_URL, ROLL_CACHE, geo=False, page=1000)
roll["pid"] = roll["F_No_column_name_"].astype(str).str.strip().str.lstrip("p")
log(f"Roll rows: {len(roll):,}  (ASMTYEAR {roll.ASMTYEAR.unique().tolist()}, TAXYR {roll.TAXYR.unique().tolist()})")
bad = ~roll["pid"].str.fullmatch(r"\d{13}")
log(f"  malformed PIDs dropped: {int(bad.sum())} (land ${roll.loc[bad, 'LANDVALUE'].sum():,})")
roll = roll[~bad].copy()

# One row per BUILDING: the parcel's values are broadcast. Assert, then take first (skill §2).
dup = roll[roll["pid"].duplicated(keep=False)]
nun = dup.groupby("pid")[["LANDVALUE", "BUILDINGVALUE", "TOTALVALUE", "PROPERTYTYPE", "ISEXEMPT"]].nunique()
log(f"  PIDs with >1 roll row (one per building): {dup.pid.nunique():,} ({len(dup):,} rows)")
assert (nun.max(axis=1) <= 1).all(), "per-building rows carry DIFFERENT values — not a broadcast"
bld_area = roll.groupby("pid")["ABOVEGROUNDAREA"].sum(min_count=1)
roll = roll.drop_duplicates("pid", keep="first").set_index("pid")
roll["bld_ar"] = bld_area
roll = roll.reset_index()
log(f"  unique PIDs: {len(roll):,}  land ${roll.LANDVALUE.sum():,}  bldg ${roll.BUILDINGVALUE.sum():,}")

# ── 2. exempt + out-of-scope ───────────────────────────────────────────────────
ex = roll["ISEXEMPT"].eq("Y")
log(f"EXEMPT (ISEXEMPT='Y') dropped: {int(ex.sum()):,} parcels, land ${roll.loc[ex, 'LANDVALUE'].sum():,}")
log(f"  exempt by type: {roll.loc[ex, 'PROPERTYTYPE'].value_counts().head(8).to_dict()}")
util = roll["PROPERTYTYPE"].isin(["UTILITY", "COMMERCIAL RAILROAD"]) & ~ex
log(f"UTILITY + COMMERCIAL RAILROAD (state-assessed operating property) dropped: "
    f"{int(util.sum())} parcels, land ${roll.loc[util, 'LANDVALUE'].sum():,} "
    f"{roll.loc[util, 'PROPERTYTYPE'].value_counts().to_dict()}")
roll = roll[~ex & ~util].copy()
TAXABLE_LAND = int(roll["LANDVALUE"].sum())
TAXABLE_BLDG = int(roll["BUILDINGVALUE"].sum())

# ── 3. geometry + join ─────────────────────────────────────────────────────────
geom = pull(GEOM_URL, GEOM_CACHE, geo=True, page=2000)
geom["PID"] = geom["PID"].astype(str).str.strip()
log(f"Geometry rows: {len(geom):,}; CITY values: {geom['CITY'].value_counts().to_dict()}")
geom = geom[geom["PID"].str.fullmatch(r"\d{13}") & geom.geometry.notna()].copy()
# Preflight (skill §2a): a PID on several geometry rows must be UNIONED, never deduped.
gd = geom[geom["PID"].duplicated(keep=False)]
log(f"  PIDs with >1 geometry row: {gd.PID.nunique()} -> unary_union")
if len(gd):
    u = geom.groupby("PID")["geometry"].apply(lambda gs: unary_union(list(gs)))
    geom = geom.drop_duplicates("PID").set_index("PID")
    geom["geometry"] = u
    geom = geom.reset_index()
geom["geometry"] = geom["geometry"].apply(lambda g: g if g.is_valid else g.buffer(0))
geom = gpd.GeoDataFrame(geom[["PID", "CONDONO", "geometry"]], geometry="geometry", crs="EPSG:4326")

parcel = geom.merge(roll, left_on="PID", right_on="pid", how="inner", validate="one_to_one")
parcel = gpd.GeoDataFrame(parcel, geometry="geometry", crs="EPSG:4326")
unm = roll[~roll["pid"].isin(geom["PID"])]
log(f"Joined: {len(parcel):,} of {len(roll):,} taxable roll PIDs have a polygon")
log(f"  taxable roll PIDs with no polygon: {len(unm):,} (land ${unm.LANDVALUE.sum():,}, "
    f"bldg ${unm.BUILDINGVALUE.sum():,}) — new plats/condos after the fabric snapshot; by type: "
    f"{unm.PROPERTYTYPE.value_counts().head(6).to_dict()}")

# ── 4. condo / co-op stacks -> one parcel per footprint, values SUMMED ─────────
pu = parcel.to_crs(UTM)
rp = pu.geometry.representative_point()
pu["_sqm"] = pu.geometry.area
pu["_stack"] = (rp.x.round(1).astype(str) + "," + rp.y.round(1).astype(str) + ","
                + pu["_sqm"].round(0).astype(str))
vc = pu["_stack"].value_counts()
stacked = pu["_stack"].map(vc).gt(1)
log(f"Stacked identical footprints: {int((vc > 1).sum()):,} stacks holding {int(stacked.sum()):,} "
    f"records (max {int(vc.max())}); types: "
    f"{pu.loc[stacked, 'PROPERTYTYPE'].value_counts().head(8).to_dict()}")

single = pu[~stacked].copy()
single["n_units"] = 1
CONDO_TYPES = {"RESIDENTIAL 1 UNIT - CONDOMINIUM", "RESIDENTIAL CONDOMINIUM GARAGE/MISC",
               "RESIDENTIAL 1 UNIT - CO-OP", "RESIDENTIAL CO-OP MASTER", "COMMON AREA"}
rows = []
for key, grp in pu[stacked].groupby("_stack", sort=False):
    vals = grp[grp["TOTALVALUE"].fillna(0) > 0]
    lead = (vals if len(vals) else grp).sort_values("TOTALVALUE", ascending=False).iloc[0].to_dict()
    lead["LANDVALUE"] = float(grp["LANDVALUE"].sum())
    lead["BUILDINGVALUE"] = float(grp["BUILDINGVALUE"].sum())
    lead["TOTALVALUE"] = float(grp["TOTALVALUE"].sum())
    lead["bld_ar"] = float(grp["bld_ar"].sum())
    lead["n_units"] = int(len(grp))
    lead["geometry"] = unary_union(list(grp.geometry.values)).buffer(0)
    # dominant use among the VALUED units (garage units are tiny, so weight by value)
    w = (vals if len(vals) else grp).groupby("PROPERTYTYPE")["TOTALVALUE"].sum()
    lead["PROPERTYTYPE_STACK"] = w.idxmax() if len(w) else lead["PROPERTYTYPE"]
    lead["_stack_kind"] = ("coop" if grp["PROPERTYTYPE"].str.contains("CO-OP").any()
                           else "condo" if grp["PROPERTYTYPE"].isin(CONDO_TYPES).any()
                           else "other")
    rows.append(lead)
merged = gpd.GeoDataFrame(rows, geometry="geometry", crs=UTM)
log(f"  -> {len(merged):,} stack parcels: {merged['_stack_kind'].value_counts().to_dict()}")
oth = merged[merged["_stack_kind"].eq("other")]
if len(oth):
    log(f"  non-condo stacks (distinct PIDs sharing one footprint, values summed): {len(oth)}; "
        f"types {oth['PROPERTYTYPE_STACK'].value_counts().head(5).to_dict()}")
single["PROPERTYTYPE_STACK"] = single["PROPERTYTYPE"]
single["_stack_kind"] = ""
parcel = gpd.GeoDataFrame(pd.concat([single, merged], ignore_index=True), geometry="geometry", crs=UTM)
assert int(parcel["LANDVALUE"].sum()) + int(unm["LANDVALUE"].sum()) == TAXABLE_LAND, "land not conserved"
assert int(parcel["BUILDINGVALUE"].sum()) + int(unm["BUILDINGVALUE"].sum()) == TAXABLE_BLDG

# Free-standing $0 association land (common area / co-op master not under any units).
ca0 = parcel["PROPERTYTYPE_STACK"].isin(["COMMON AREA", "RESIDENTIAL CO-OP MASTER"]) & \
    parcel["TOTALVALUE"].fillna(0).eq(0)
log(f"Free-standing $0 common-area / co-op-master parcels dropped: {int(ca0.sum())}")
parcel = parcel[~ca0].copy()

# Overlap check: a valued parcel lying inside another (not identical) footprint is double land.
_rp = gpd.GeoDataFrame(geometry=parcel.geometry.representative_point(), crs=UTM)
_hits = gpd.sjoin(_rp, parcel[["geometry"]], predicate="within")
_hits = _hits[_hits.index != _hits["index_right"]]
log(f"Parcels whose point falls inside ANOTHER shipped parcel: {_hits.index.nunique():,}")
parcel = parcel.to_crs("EPSG:4326")


# ── 5. classification ──────────────────────────────────────────────────────────
def categorize(pt: str, kind: str, bld: float) -> str:
    pt = str(pt or "").upper()
    if kind == "coop" or "CO-OP" in pt:
        return "Co-op"
    if kind == "condo" or "CONDOMINIUM" in pt:
        return "Condominium"
    if pt.startswith("VACANT LAND - INDUSTRIAL"):
        return "Vacant Industrial"
    if pt.startswith("VACANT LAND - COMMERCIAL"):
        return "Vacant Commercial"
    if pt.startswith("VACANT LAND"):
        return "Vacant Residential"
    if pt.startswith("RESIDENTIAL 1 UNIT"):
        return "Single Family"
    if pt in ("RESIDENTIAL 2 UNITS", "RESIDENTIAL 3 UNITS"):
        return "Residential 2-3 Units"
    if pt in ("APARTMENT", "FRATERNITY/SORORITY HOUSING"):
        return "Multifamily"
    if pt.startswith("MIXED USE"):
        return "Mixed Use"
    if pt in ("COMMERCIAL", "MEDICAL CARE FACILITY"):
        return "Commercial"
    if pt == "INDUSTRIAL":
        return "Industrial"
    if pt == "COMMERCIAL RAILROAD":
        return "Railroad"
    if pt == "FARM":
        return "Agricultural / Rural"
    if pt == "RESIDENTIAL MISC STRUCTURE":
        return "Residential Accessory"
    if pt == "COMMON AREA":
        return "Common Area"
    return "Other"


parcel["property_land_use_category"] = [categorize(p, k, b) for p, k, b in zip(
    parcel["PROPERTYTYPE_STACK"], parcel["_stack_kind"], parcel["BUILDINGVALUE"])]
parcel["exemption_flag"] = 0  # exempt parcels were dropped above on the assessor's own flag

# ── 6. canonical fields ────────────────────────────────────────────────────────
log("Computing geodesic areas...")
parcel["geom_area_sqft"] = parcel["geometry"].apply(gis_area_sqft)
parcel.loc[parcel["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
_src = pd.to_numeric(parcel["PARCELAREA"], errors="coerce").where(parcel["n_units"].eq(1))
_r = (parcel["geom_area_sqft"] / _src.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
log(f"Area cross-check vs assessor PARCELAREA: median ratio {_r.median():.4f}, "
    f"off by >5%: {int((_r.sub(1).abs() > 0.05).sum()):,} of {int(_r.notna().sum()):,}")
parcel["land_area_sqft"] = parcel["geom_area_sqft"]
parcel["area_source"] = "gis"
parcel["land_area_acres"] = parcel["land_area_sqft"] / SQFT_PER_ACRE
parcel["likely_remnant"] = (parcel["land_area_sqft"] < 500).astype(int)

parcel["land_value"] = parcel["LANDVALUE"].astype(float)
parcel["improvement_value"] = parcel["BUILDINGVALUE"].astype(float)
parcel["full_market_value"] = parcel["land_value"] + parcel["improvement_value"]
den = parcel["land_area_sqft"].replace(0, np.nan)
parcel["full_market_value_per_sqft"] = parcel["full_market_value"] / den
parcel["land_value_per_sqft"] = parcel["land_value"] / den
parcel["improvement_value_per_sqft"] = parcel["improvement_value"] / den
parcel = add_improvement_ratio_fields(parcel, land_col="land_value", improvement_col="improvement_value")

parcel["property_land_use_refined"] = classify_property_refined(
    parcel, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Railroad", "Agricultural / Rural", "Common Area", "Other"),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    bld_ar_col="bld_ar", fetch_footprints=False)

parcel["parcel_id"] = parcel["PID"]
parcel["link"] = parcel["PID"].map(LINK_FMT.format)
parcel["address"] = parcel["ADDRESSFORMATTED"].fillna("").astype(str).str.strip()
parcel.loc[parcel["n_units"].gt(1), "address"] = parcel["address"].str.replace(
    r"\s+(UNIT|#)\s*\S+$", "", regex=True)
parcel["neighborhood"] = parcel["NEIGHBORHOOD"].fillna("").astype(str).str.title()
parcel["community"] = parcel["COMMUNITY"].fillna("").astype(str).str.title()
parcel["zoning"] = parcel["ZONING"].fillna("").astype(str)
parcel["assessor_property_type"] = parcel["PROPERTYTYPE_STACK"].fillna("").astype(str).str.title()
parcel["n_condo_units"] = parcel["n_units"].astype(int)
parcel["tax_year"] = 2027

# ── 7. export ──────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "parcel_id", "address", "neighborhood", "community", "zoning",
           "assessor_property_type", "n_condo_units", "tax_year",
           "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant"]
final = parcel[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final["geometry"] = final["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")
assert not final["parcel_id"].duplicated().any(), "duplicate parcel_id in output"
out = DATA_DIR / "minneapolis-mn-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"minneapolis-mn-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet",
                 index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"land ${final.current_full_land_value.sum():,.0f}  impr ${final.improvement_value.sum():,.0f}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")

# ── §6a smoke alarms ───────────────────────────────────────────────────────────
log("--- condo/stub smoke alarms (skill §6a) ---")
a = parcel["geom_area_sqft"]
lv = pd.to_numeric(final["land_value_per_sqft"], errors="coerce")
log(f"  footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
log(f"  sub-500 / sub-1000 sqft footprints: {int((a < 500).sum()):,} / {int((a < 1000).sum()):,}")
log(f"  land $/sqft p50/p95/p99/max: ${lv.median():,.2f} / ${lv.quantile(.95):,.2f} / "
    f"${lv.quantile(.99):,.2f} / ${lv.max():,.2f}")
for cat in ("Condominium", "Co-op", "Single Family", "Commercial"):
    s = lv[final["property_land_use_category"].eq(cat)]
    log(f"    {cat:14s} land $/sqft p50 ${s.median():,.2f}  p99 ${s.quantile(.99):,.2f}")
rp = final.geometry.representative_point()
vc = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)).value_counts()
log(f"  stacked clusters: {int((vc > 1).sum())}, max stack: {int(vc.max())}")
holes = final.geometry.apply(lambda g: 0 if g is None else sum(
    len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
log(f"  parcels with interior rings (holes): {int((holes > 0).sum()):,}")
log(f"  zero land value (renders as gp-error): "
    f"{int((final['current_full_land_value'].fillna(0) <= 0).sum()):,}")
top = final.sort_values("land_value_per_sqft", ascending=False).head(10)
for r in top.itertuples():
    log(f"    top $/sqft: {r.parcel_id} {r.address[:30]:30s} {r.property_land_use_category:14s} "
        f"{r.land_area_acres * SQFT_PER_ACRE:9,.0f} sqft  ${r.land_value_per_sqft:,.0f}/sqft")
log(f"  bounds: {[round(v, 4) for v in final.total_bounds]}")
log("DONE")
