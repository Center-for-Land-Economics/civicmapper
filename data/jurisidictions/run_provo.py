#!/usr/bin/env python3
"""
Build the City of Provo, UT canonical parcel parquet.

Provo is the Utah County seat (pop. 115,162) and home to Brigham Young University.

SOURCE — Utah County's own live assessor layer (public, no token):
  https://maps.utahcounty.gov/arcgis/rest/services/Assessor/TaxParcelAll_NoLabel/MapServer/0
  285,300 countywide parcels, ONE ROW PER PARCEL, 92 fields: geometry + MKT_LAND_VALUE /
  MKT_IMP_VALUE / MKT_CUR_VALUE + deed ACREAGE + TAXABLE_CODE + a real use taxonomy
  (PROP_TYPE_DESCR / SPC_PROP_TYP_DESCR) + building areas. ASMT_YEAR 2027 (the current roll).
  City boundary: UGRC `UtahMunicipalBoundaries`, NAME='Provo' AND COUNTYNBR='25'.

DO NOT USE UGRC's statewide LIR layer (`Parcels_Utah_LIR`) for Utah County. It looks like the
obvious one-stop source — same publisher as the boundary, clean LIR schema — and this ETL was
first built on it. It is NOT usable here for two reasons:

  1. IT IS MISSING VALUES ON ~20% OF TAXABLE PARCELS, silently. LAND_MKT_VALUE/TOTAL_MKT_VALUE
     come back NULL on parcels that plainly carry value. Verified against the assessor's own
     value history: 4095 W CENTER ST (a 15.9-acre mini-warehouse complex) is NULL in LIR and
     $6.07M land / $14.0M total at the county; 3581 S MOUNTAIN VISTA PKWY is NULL in LIR and
     $1.27M land / $9.63M total. Because Utah does not assess exempt land at all, "no value"
     reads as "exempt", so these parcels were being silently DROPPED — the LIR build shipped
     18,700 parcels / $5.26B of land against the county's 27,273 / $7.26B, i.e. it was missing
     ~8,500 taxable parcels and 28% of the city's land value, and downtown University Avenue
     rendered as a hole.
  2. It carries no exemption flag at all (TAXEXEMPT_TYPE is NULL on every Utah County row),
     whereas the county layer has an explicit TAXABLE_CODE.

The LIR layer's one genuine advantage — a 152-value building taxonomy in BLDG_SQFT_INFO — is
superseded by the county's SPC_PROP_TYP_DESCR, which is cleaner and parcel-level. The same
caution probably applies to LIR sheets for other Utah counties: check value coverage against
the county assessor before trusting one.

Outputs:
- data/jurisidictions/data/provo/provo-ut-parcels.parquet
- data/jurisidictions/data/provo/provo-ut-parcels_YYYY_MM_DD.parquet

Notes:
- VALUES ARE FULL MARKET VALUE, not Utah's 55% primary-residential taxable basis. Verified
  live against the assessor's own value history for two parcels: MKT_* matches the table
  headed "Market Value" row-for-row. (TXBL_CUR_VALUE is the reduced taxable figure — not used.)
- EXEMPTION is explicit: TAXABLE_CODE 100 = taxable, everything else is an exemption class.
  The big ones in Provo are 920 (7,433 acres — the Uinta National Forest sections and the lake
  flats), 959 (3,444 acres — BYU, schools, city land), 600/601 (404 acres — condo/HOA common
  area, which the condo merge below consumes before the exempt filter runs) and 998 (ROW).
- OWNER_NAME is deliberately NOT requested from the source at all (issue #12): it never enters
  the pipeline, rather than being fetched and dropped later.
- CONDOS/PUDs are mapped as building-footprint stubs and are MERGED down onto their plat's
  common-area land (skill §6a/§6b, run_olympia.py recipe) — see the merge block.
- ~19k parcels, but baked to PMTiles + H3 anyway: Provo's condo/townhome stock means small
  parcels are a large share, and those drop out below ~z13 on the browser GeoParquet path
  (the Olympia low-zoom sparseness fix).
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
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import (  # noqa: E402
    add_improvement_ratio_fields,
    check_area_agreement,
    classify_property_refined,
    gis_area_sqft,
)

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "provo"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "provo-ut-county-geometry.parquet"
BOUNDARY_CACHE = DATA_DIR / "provo-boundary.geojson"

PARCELS_URL = ("https://maps.utahcounty.gov/arcgis/rest/services/Assessor/"
               "TaxParcelAll_NoLabel/MapServer/0/query")
BOUNDARY_URL = ("https://services1.arcgis.com/99lidPhWCzftIe9K/arcgis/rest/services/"
                "UtahMunicipalBoundaries/FeatureServer/0/query")
# Utah County is COUNTYNBR 25; kept so a future statewide re-point can't grab a same-named place.
BOUNDARY_WHERE = "NAME='Provo' AND COUNTYNBR='25'"

# OWNER_NAME is deliberately absent — see the module docstring.
OUT_FIELDS = (
    "PARCEL_NO,PARCELID,SITE_FULL_ADDRESS,ACREAGE,TAX_DISTRICT,TAX_CITY,TAXABLE_CODE,"
    "ACCOUNT_TYPE,PROP_TYPE_DESCR,SPC_PROP_TYP_DESCR,ASMT_CODE_DESCR,NEIGHBORHOOD,"
    "MKT_LAND_VALUE,MKT_IMP_VALUE,MKT_CUR_VALUE,ASMT_YEAR,TOTAL_UNIT_COUNT,"
    "TOTAL_ABOVE_GRADE_AREA,GLA_WEIGHTED_YRBLT,TOTAL_IMP_COUNT"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}
PAGE = 1000
SQFT_PER_ACRE = 43560.0
SQM_TO_SQFT = 10.763910416709722
UTM = "EPSG:32612"  # UTM 12N — Provo
TAXABLE_CODE_TAXABLE = 100


def _polygonal(g):
    """Keep only the polygonal part of a geometry.

    make_valid() turns a self-intersecting polygon into a GeometryCollection when the repair
    sheds a degenerate spur, and a bare LineString/GeometryCollection breaks both the hole
    diagnostic and the PMTiles bake. Drop the non-areal pieces and keep the polygons.
    """
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


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── fetch ─────────────────────────────────────────────────────────────────────
def fetch_boundary() -> gpd.GeoDataFrame:
    if not BOUNDARY_CACHE.exists():
        r = requests.get(BOUNDARY_URL, params={
            "where": BOUNDARY_WHERE, "outFields": "NAME,POPLASTCENSUS",
            "returnGeometry": "true", "outSR": 4326, "f": "geojson"},
            headers=HEADERS, timeout=120)
        r.raise_for_status()
        BOUNDARY_CACHE.write_bytes(r.content)
    b = gpd.read_file(BOUNDARY_CACHE).to_crs("EPSG:4326")
    if not len(b):
        raise RuntimeError("Provo municipal boundary came back empty")
    return b


def fetch_parcels(bounds) -> gpd.GeoDataFrame:
    """Countywide layer filtered to the city's bounding envelope, cached to parquet.

    The envelope (not TAX_CITY) is the fetch filter and the authoritative municipal boundary
    does the actual clip below — a tax-city/situs label is not a jurisdiction (playbook §4).
    """
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    minx, miny, maxx, maxy = bounds
    geom_params = {
        "geometry": json.dumps({"xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
                                "spatialReference": {"wkid": 4326}}),
        "geometryType": "esriGeometryEnvelope", "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
    }
    total = requests.get(PARCELS_URL, params={**geom_params, "where": "1=1",
                         "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=120).json()["count"]
    log(f"Pulling {total:,} rows in the Provo envelope (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(5):
            try:
                r = requests.get(PARCELS_URL, params={
                    **geom_params, "where": "1=1", "outFields": OUT_FIELDS,
                    "returnGeometry": "true", "resultOffset": off, "resultRecordCount": PAGE,
                    "outSR": 4326, "orderByFields": "OBJECTID", "f": "geojson"},
                    headers=HEADERS, timeout=300)
                r.raise_for_status()
                body = json.loads(r.content)
                # A MapServer returns HTTP 200 with an {"error": ...} body on overload; without
                # this check that parses as "zero features" and silently truncates the pull.
                if "error" in body:
                    raise RuntimeError(str(body["error"])[:200])
                feats = body.get("features", [])
                gdf = (gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
                       if feats else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
                break
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt + 1} @off {off}: {type(e).__name__}: {e}")
                time.sleep(5 * (attempt + 1))
        if gdf is None:
            raise RuntimeError(f"Parcel pull failed at offset {off}")
        if not len(gdf):
            break
        pages.append(gdf)
        off += len(gdf)
        if off % 10000 < PAGE:
            log(f"  fetched {off:,}/{total:,}")
        if len(gdf) < PAGE:      # a SHORT page means the end — do not break on 0 only
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


boundary = fetch_boundary()
raw = fetch_parcels(boundary.total_bounds)
log(f"Envelope rows: {len(raw):,}")

# ── clip to the authoritative city boundary (centroid-within) ─────────────────
if raw.crs is None:
    raw = raw.set_crs("EPSG:4326")
elif raw.crs.to_epsg() != 4326:
    raw = raw.to_crs("EPSG:4326")
# 109 source polygons are self-intersecting; make_valid (not buffer(0), which can empty a
# bow-tie) repairs them. Left unrepaired they raise a GEOS side-location-conflict inside the
# multi-polygon dedup below.
n_bad = int((~raw.geometry.is_valid).sum())
raw["geometry"] = raw["geometry"].apply(
    lambda g: g if (g is not None and g.is_valid) else _polygonal(make_valid(g)))
raw = raw[raw["geometry"].notna() & ~raw.geometry.is_empty].copy()
log(f"Repaired {n_bad:,} invalid source geometries")
city = boundary.geometry.union_all()
inside = gpd.GeoSeries(raw.geometry.representative_point(), crs="EPSG:4326").within(city)
raw = raw[inside.values].copy()
log(f"Inside the Provo municipal boundary: {len(raw):,} rows")

for c in ["MKT_LAND_VALUE", "MKT_IMP_VALUE", "MKT_CUR_VALUE", "ACREAGE", "TAXABLE_CODE",
          "TOTAL_UNIT_COUNT", "TOTAL_ABOVE_GRADE_AREA"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw["pid"] = raw["PARCEL_NO"].astype("Int64").astype(str).str.zfill(9)
raw = raw[raw["pid"].str.isdigit()].copy()

# ── dedup parcels split across several GIS polygons ──────────────────────────
# Values are parcel-level and repeat identically on every polygon, so they take `first` and are
# NEVER summed (skill §2, the Dallas N-x inflation bug); the shapes are unioned.
log(f"Rows -> parcels: {len(raw):,} -> {raw['pid'].nunique():,}")
first_cols = ["SITE_FULL_ADDRESS", "ACREAGE", "TAX_DISTRICT", "TAX_CITY", "TAXABLE_CODE",
              "ACCOUNT_TYPE", "PROP_TYPE_DESCR", "SPC_PROP_TYP_DESCR", "ASMT_CODE_DESCR",
              "NEIGHBORHOOD", "MKT_LAND_VALUE", "MKT_IMP_VALUE", "MKT_CUR_VALUE", "ASMT_YEAR",
              "TOTAL_UNIT_COUNT", "TOTAL_ABOVE_GRADE_AREA", "GLA_WEIGHTED_YRBLT"]
parcel = raw.groupby("pid", dropna=False).agg({c: "first" for c in first_cols}).reset_index()


def _union(geoms):
    gs = [g for g in geoms if g is not None]
    try:
        return unary_union(gs)
    except Exception:  # noqa: BLE001 - last-resort repair for stubborn topology
        return unary_union([_polygonal(make_valid(g)).buffer(0) for g in gs])


parcel["geometry"] = raw.groupby("pid", dropna=False)["geometry"].apply(_union).values
parcel = gpd.GeoDataFrame(parcel, geometry="geometry", crs="EPSG:4326")
log(f"After PARCEL_NO dedup -> {len(parcel):,}")

parcel["land_val"] = parcel["MKT_LAND_VALUE"]
parcel["bld_val"] = parcel["MKT_IMP_VALUE"].fillna(0)
parcel["tot_appr_val"] = parcel["MKT_CUR_VALUE"].fillna(
    parcel["land_val"].fillna(0) + parcel["bld_val"])
parcel["plat"] = parcel["pid"].str[:5]  # Utah County serial MM:PPP:LLLL — MM:PPP is the plat map
parcel["taxable"] = parcel["TAXABLE_CODE"].eq(TAXABLE_CODE_TAXABLE) & parcel["land_val"].gt(0)

# ── condo / PUD units -> merge DOWN onto their plat's common-area land (§6b) ──
# Each unit is mapped as its own building-footprint stub (~550-1,500 sqft) carrying a share of
# the development's land value, and the development's REAL land is one or more exempt parcels in
# the same plat (usually TAXABLE_CODE 600/601, "common area"). The county's ACREAGE agrees with
# the stub polygon (median ratio 1.001), so there is no independent land area to fall back on:
# left alone, Provo's student-housing stock renders as thousands of pencils — CONDO stubs sit at
# a median $52/sqft and run past $250 against a citywide median of $24.
#
# The gate is PLAT DOMINANCE, not adjacency. A plat merges only when its stubs are >=60% of its
# valued parcels. That separates a pure condo/PUD regime (plat 37118: 120 stubs + one 3.9-acre
# common parcel, zero ordinary lots) from a MIXED subdivision (plat 36196: 8 stubs among 23
# valued parcels, the rest ordinary 4,500 sqft townhouse lots at a normal $18/sqft, sharing a
# 1.07-acre HOA open space). Merging the mixed case would hand the whole HOA open space to 8
# units that do not own it — the failure run_lynchburg.py guards against with its "common-area
# parcels never chain" rule. Adjacency alone was tried first and got exactly that wrong.
STUB_SQFT = 2000.0       # above this a small parcel is a real lot with a yard, not a unit stub
COMMON_MIN_SQFT = 200.0  # ignore slivers; real common-area parcels run 4k-300k sqft
STUB_SHARE = 0.60        # stubs must dominate the plat's valued parcels

parcel["_sqft"] = parcel.to_crs(UTM).geometry.area * SQM_TO_SQFT
parcel["_is_stub"] = parcel["taxable"] & parcel["_sqft"].lt(STUB_SQFT)
parcel["_is_common"] = ~parcel["taxable"] & parcel["_sqft"].ge(COMMON_MIN_SQFT)

_g = parcel.groupby("plat")
_stats = pd.DataFrame({"n_val": _g["taxable"].sum(), "n_stub": _g["_is_stub"].sum(),
                       "n_com": _g["_is_common"].sum()})
_stats["share"] = _stats["n_stub"] / _stats["n_val"].replace(0, np.nan)
merge_plats = set(_stats.index[(_stats["n_stub"] >= 2) & (_stats["n_com"] >= 1)
                               & (_stats["share"] >= STUB_SHARE)])
log(f"Condo/PUD plats to merge: {len(merge_plats):,}")


def _fill_holes(g):
    """Solid exterior. A common parcel's holes are the punched-out unit footprints, which belong
    to the development — unlike a fee-simple lot carved out of a parent (skill §6b)."""
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


consumed, dev_rows = set(), []
for plat in sorted(merge_plats):
    blk = parcel[parcel["plat"].eq(plat)]
    units, commons = blk[blk["_is_stub"]], blk[blk["_is_common"]]
    row = units.iloc[0].to_dict()
    row.update({
        "pid": units["pid"].iloc[0],          # link to a representative unit's assessor page
        "geometry": _fill_holes(_union(list(units.geometry) + list(commons.geometry))),
        "land_val": units["land_val"].sum(),
        "bld_val": units["bld_val"].sum(),
        "tot_appr_val": units["tot_appr_val"].sum(),
        "TOTAL_ABOVE_GRADE_AREA": units["TOTAL_ABOVE_GRADE_AREA"].sum(),
        "TOTAL_UNIT_COUNT": units["TOTAL_UNIT_COUNT"].sum(),
        "PROP_TYPE_DESCR": _dominant(units["PROP_TYPE_DESCR"]),
        "SPC_PROP_TYP_DESCR": _dominant(units["SPC_PROP_TYP_DESCR"]),
        "ACREAGE": np.nan,                    # per-unit shares; the union polygon is the land
        "taxable": True,
        "_merged_units": len(units),
    })
    dev_rows.append(row)
    consumed.update(units["pid"])
    consumed.update(commons["pid"])

if dev_rows:
    devs = gpd.GeoDataFrame(pd.DataFrame(dev_rows), geometry="geometry", crs="EPSG:4326")
    parcel = parcel[~parcel["pid"].isin(consumed)].copy()
    parcel["_merged_units"] = 1
    parcel = gpd.GeoDataFrame(pd.concat([parcel, devs], ignore_index=True),
                              geometry="geometry", crs="EPSG:4326")
    _psf = devs["land_val"] / (devs.to_crs(UTM).geometry.area * SQM_TO_SQFT)
    log(f"Condo merge: {int(devs['_merged_units'].sum()):,} unit stubs -> {len(devs):,} "
        f"development parcels (land $/sqft p50 ${_psf.median():,.0f}, max ${_psf.max():,.0f})")
else:
    parcel["_merged_units"] = 1
_left = parcel["_is_stub"].fillna(False) & parcel["_merged_units"].eq(1)
log(f"Unit-sized parcels left individual (no common-area land mapped in their plat): "
    f"{int(_left.sum()):,}")

# ── exemption filter ─────────────────────────────────────────────────────────
# The county's own TAXABLE_CODE is authoritative: 100 = taxable, everything else names an
# exemption class. Parcels coded taxable but carrying no land value (a handful of pending
# DEFAULT/CONDO records) are dropped too — they would render as zero-value gp-error parcels.
parcel["exemption_flag"] = (~parcel["taxable"]).astype(int)
_by_code = parcel[parcel["exemption_flag"] == 1].groupby(
    parcel["TAXABLE_CODE"].fillna(-1)).size().sort_values(ascending=False)
log(f"Excluded (exempt / no land value): {int(parcel['exemption_flag'].sum()):,} "
    f"— top codes {dict(list(_by_code.items())[:6])}")
ex = parcel[parcel["exemption_flag"] == 0].copy()
ex = ex.drop(columns=["_sqft", "_is_stub", "_is_common", "taxable"], errors="ignore")
log(f"Shipped parcels -> {len(ex):,}")

# ── classification ───────────────────────────────────────────────────────────
# The county publishes a real use taxonomy, so this is a direct mapping rather than a heuristic.
# SPC_PROP_TYP_DESCR is consulted first because it distinguishes Condo from Townhome from Twin
# inside the coarser CONDO/PUD buckets, and names parking structures.
SPC_MAP = {
    "Single Family Res": "Single Family", "Twin": "Single Family",
    "Twin - Detached": "Single Family", "Res Adjoining": "Single Family",
    "Modular Home": "Single Family",
    "Condo": "Condominium",
    "Townhome": "Townhome", "Vac Townhome Lot": "Vacant Land",
    "Manuf Home": "Mobile Home", "Mobile Homes": "Mobile Home",
    "Mobile Home Park": "Mobile Home",
    "Parking Structure": "Parking Garage",
    "Schools": "Commercial", "Community Center": "Commercial",
    "Group Care-Nrsg-Retire-Res Prim": "Multifamily", "Assisted Living": "Multifamily",
    "Mixed Use Residential/Retail": "Mixed Use", "Mixed Use ": "Mixed Use",
    "Dairy": "Agricultural / Rural", "Greenhouses": "Agricultural / Rural",
    "Detached Imps Only": "Vacant Land", "Undevelopable": "Vacant Land",
    "Unbuildable Com lot": "Vacant Land", "Unbuildable Com w/Det": "Vacant Land",
}
TYPE_MAP = {
    "SINGLE FAMILY RES": "Single Family",
    "CONDO": "Condominium", "IMPROVED CONDOS/PUD": "Condominium",
    "PUD": "Townhome",
    "DUPLEX": "Multifamily", "TRIPLEX": "Multifamily", "FOURPLEX": "Multifamily",
    "APARTMENTS": "Multifamily", "MULTIPLE RES": "Multifamily",
    "MULTIPLE UNIT MIX": "Multifamily", "RES CONVERSION TO APT": "Multifamily",
    "STUDENT HOUSING": "Multifamily", "SUBSIDIZE HOUSING": "Multifamily",
    "MANUF HOME": "Mobile Home", "MOBILE HOMES": "Mobile Home",
    "MOBILE HOME PARK": "Mobile Home",
    "COMMERCIAL": "Commercial", "RETAIL": "Commercial", "FOOD": "Commercial",
    "AUTO": "Commercial", "LODGING": "Commercial",
    "OFFICE <50,000 sf": "Commercial", "OFFICE >50,000 sf": "Commercial",
    "COMMERCIAL WITH RES EXEMPTION": "Commercial",
    "INDUSTRIAL": "Industrial",
    "VACANT": "Vacant Land", "VACANT COMMERCIAL": "Vacant Land",
    "VACANT APARTMENT": "Vacant Land",
    "MIXED USE Com/Apt/Res/HD": "Mixed Use",
    "AGRICULTURAL": "Agricultural / Rural",
    "PUBLIC": "Other", "EXEMPT": "Other", "PARTIALLY EXEMPT COUNTY": "Other",
}
# Last resort when both type fields are DEFAULT/blank (196 parcels): the account's own bucket.
ACCOUNT_MAP = {"RESIDENTIAL": "Single Family", "HGHDENRES": "Multifamily",
               "APARTMENTS": "Multifamily", "COMMERCIAL": "Commercial"}


def _txt(v):
    """Source strings arrive as NaN floats when the column is empty — normalise before .strip()."""
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


def categorize(spc, ptype, acct, impr):
    spc = _txt(spc)
    if spc in SPC_MAP:
        return SPC_MAP[spc]
    if spc.startswith("Vac"):                 # Vac Res / Vac Sub Lot / Vac Com Lot / ...
        return "Vacant Land"
    ptype = _txt(ptype).upper()
    if ptype in TYPE_MAP:
        return TYPE_MAP[ptype]
    if "Units on Lot" in spc or "-plex on Lot" in spc:
        return "Multifamily"
    cat = ACCOUNT_MAP.get(_txt(acct).upper())
    if cat:
        return cat
    return "Vacant Land" if (impr or 0) <= 0 else "Other"


ex["property_land_use_category"] = [
    categorize(s, p, a, i) for s, p, a, i in zip(
        ex["SPC_PROP_TYP_DESCR"], ex["PROP_TYPE_DESCR"], ex["ACCOUNT_TYPE"], ex["bld_val"])]

ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")
ex["bld_ar"] = pd.to_numeric(ex["TOTAL_ABOVE_GRADE_AREA"], errors="coerce").fillna(0)
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other", "Agricultural / Rural"),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    bld_ar_col="bld_ar",
    fetch_footprints=False)

# ── areas: deed acreage, guarded, geodesic fallback ──────────────────────────
ex["geometry"] = ex["geometry"].apply(
    lambda g: g if (g is None or g.is_valid) else _polygonal(make_valid(g)))
log("Computing geodesic areas...")
ex["geom_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
check_area_agreement(ex["geom_area_sqft"], ex["ACREAGE"] * SQFT_PER_ACRE,
                     label="deed ACREAGE", log=log)

ex["reported_sqft"] = pd.to_numeric(ex["ACREAGE"], errors="coerce") * SQFT_PER_ACRE
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan
# Richmond guard: deed acreage is trusted only when it is in the same ballpark as the polygon.
# It is NaN on merged developments (per-unit shares would be nonsense against the union
# footprint), so those always fall through to the polygon.
ratio = ex["reported_sqft"] / ex["geom_area_sqft"].replace(0, np.nan)
use_reported = ex["reported_sqft"].gt(0) & (ratio.between(0.5, 2.0) | ex["geom_area_sqft"].isna())
log(f"Deed acreage rejected as implausible (outside 0.5-2.0x polygon): "
    f"{int((ex['reported_sqft'].gt(0) & ~use_reported).sum()):,}")
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["geom_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = pd.to_numeric(ex["tot_appr_val"], errors="coerce")
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

# Utah County's public parcel lookup. Verified live: the serial is accepted with or without a
# leading zero, and an unknown serial returns a blank record rather than an error page.
ex["link"] = "https://www.utahcounty.gov/LandRecords/Property.asp?av_serial=" + ex["pid"].astype(str)

# ── export ───────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant"]
for c in COLUMNS:
    if c not in ex.columns:
        ex[c] = np.nan
final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final["geometry"] = final["geometry"].apply(
    lambda g: g if (g is None or g.is_valid) else _polygonal(make_valid(g)))
final = gpd.GeoDataFrame(final, geometry="geometry", crs=ex.crs)
if final.crs is None or final.crs.to_epsg() != 4326:
    final = final.to_crs("EPSG:4326")
out = DATA_DIR / "provo-ut-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"provo-ut-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet",
                 index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"land value total: ${pd.to_numeric(final['current_full_land_value']).sum() / 1e9:,.2f}B | "
    f"market value total: ${pd.to_numeric(final['full_market_value']).sum() / 1e9:,.2f}B")

# ── §6a smoke alarms ─────────────────────────────────────────────────────────
log("--- condo/stub smoke alarms (skill §6a) ---")
a = ex["geom_area_sqft"]
lv = pd.to_numeric(final["land_value_per_sqft"], errors="coerce")
shown = lv[final["likely_remnant"] == 0]
log(f"  footprint sqft p1/p5/p10/p50: {[round(a.quantile(q)) for q in (.01, .05, .10, .50)]}")
log(f"  sub-500 / sub-1000 sqft footprints: {int((a < 500).sum()):,} / {int((a < 1000).sum()):,}")
log(f"  land $/sqft ALL ROWS    p50/p95/p99/max: ${lv.median():,.2f} / ${lv.quantile(.95):,.2f} / "
    f"${lv.quantile(.99):,.2f} / ${lv.max():,.2f}")
log(f"  land $/sqft AS RENDERED p50/p95/p99/max: ${shown.median():,.2f} / ${shown.quantile(.95):,.2f} / "
    f"${shown.quantile(.99):,.2f} / ${shown.max():,.2f}  (likely_remnant excluded, hideRemnants=true)")
holes = final.geometry.apply(lambda g: 0 if g is None else sum(
    len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
log(f"  parcels with interior rings (holes): {int((holes > 0).sum()):,}")
rp = final.geometry.representative_point()
_vc = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)).value_counts()
log(f"  stacked footprints: {int((_vc > 1).sum()):,} clusters, max stack {int(_vc.max())}")
log(f"  zero/neg land value (renders as gp-error): "
    f"{int((pd.to_numeric(final['current_full_land_value'], errors='coerce').fillna(0) <= 0).sum()):,}")
log(f"  bounds: {[round(v, 4) for v in final.total_bounds]}")
log("DONE")
