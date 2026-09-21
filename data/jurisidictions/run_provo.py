#!/usr/bin/env python3
"""
Build the City of Provo, UT canonical parcel parquet.

Provo is the Utah County seat (pop. 115,162) and home to Brigham Young University.

Source (UGRC / Utah Geospatial Resource Center, public, no token):
- Utah Utah County Parcels LIR:
  https://services1.arcgis.com/99lidPhWCzftIe9K/arcgis/rest/services/Parcels_Utah_LIR/FeatureServer/0
  327,655 countywide rows. A ONE-STOP layer in the state's LIR (Land Information Record)
  standard schema: geometry + LAND_MKT_VALUE + TOTAL_MKT_VALUE + PARCEL_ACRES + PROP_CLASS +
  BLDG_SQFT/BLDG_SQFT_INFO + BUILT_YR, all in one place. No joins, no manual downloads.
  CURRENT_ASOF = 2025-10-30, i.e. the 2025 assessment roll.
- City boundary: Utah Municipal Boundaries (same publisher), NAME='Provo', COUNTYNBR='25'.

The county's own GIS host (maps.utahcounty.gov) is a thin viewer shell; the UGRC mirror is
the authoritative machine-readable copy and is the one the county's open-data page points at.

Outputs:
- data/jurisidictions/data/provo/provo-ut-parcels.parquet
- data/jurisidictions/data/provo/provo-ut-parcels_YYYY_MM_DD.parquet

Notes / the four traps this ETL had to clear:

1. VALUES ARE FULL MARKET VALUE, not Utah's 55% primary-residential taxable basis.
   Verified live against the assessor's own value history for two parcels (04:017:0017 and
   04:002:0034): the LIR numbers match the table headed "Market Value" row-for-row for 2025.
   So no Georgia-style 40%-of-market correction is needed here. NOTE the assessor has already
   published 2026 values; LIR is one roll behind (2025). That is the published state dataset,
   so it is what ships.

2. THE LIR LAYER HAS ONE ROW PER BUILDING, NOT PER PARCEL. 38,687 Provo rows collapse to
   30,310 parcels; one apartment complex (04:021:0031, 865 N 160 W) is 216 rows carrying the
   SAME $23.3M total on every one. Summing values across an account's rows would have inflated
   it 216x (skill §2, the Dallas bug). So the PARCEL_ID dedup takes values as `first`, sums
   BLDG_SQFT across the buildings, keeps the LARGEST building's type for classification, and
   unions the geometry.

3. THERE IS NO EXEMPTION FLAG. TAXEXEMPT_TYPE is NULL on every Utah County row and PROP_CLASS
   ='Tax Exempt' covers only 25 parcels, because exempt land in Utah simply is not assessed:
   it carries NULL values. So "no land value" IS the exemption signal here, and it is what the
   exempt filter keys on. The unvalued set is dominated by exactly what you would expect —
   whole 640-acre PLSS sections of Uinta National Forest in the mountains east of town, the
   municipal airport (912 S AVIATION DR, 828 acres), BYU, parks and ROW.

4. CONDOS / PUDs ARE MAPPED AS BUILDING-FOOTPRINT STUBS (skill §6a/§6b). 3,220 valued parcels
   have a sub-1,000 sqft footprint, and PARCEL_ACRES agrees with the polygon (median ratio
   1.001), so the assessor carries no independent land area for them either — the development's
   real land is a SEPARATE, UNVALUED common-area parcel in the same plat. Left alone, Provo's
   student-housing stock renders as thousands of pencils: land $/sqft ran to $351 against a
   citywide median of $25. So units are MERGED DOWN onto their common-area land (run_olympia.py
   recipe): see the merge block for the plat-dominance gate that keeps mixed subdivisions out.

Classification: PROP_CLASS is too coarse to use alone — 14,665 Provo rows come back "Unknown"
(the LIR translation has no mapping for Utah County's condo/townhome/multi-unit codes) and the
duplex at 534 S 100 W is filed "Commercial" because Utah taxes non-primary residential at the
full rate. BLDG_SQFT_INFO carries a genuinely rich 152-value building taxonomy instead
(`int:_two_story` = interior townhouse unit, `12_unit_building`, `fourplex:_two_story`,
`storage_warehouse`, ...), so categorize() keys on the LARGEST building's type and falls back
to PROP_CLASS/HOUSE_CNT only for parcels whose only structure is accessory (shed, carport).

~19k shipped parcels, but baked to PMTiles + H3 anyway: Provo's condo/townhome stock means a
large share of parcels are small, which is exactly the population that drops out below ~z13 on
the browser GeoParquet path (the Olympia low-zoom sparseness fix, memory geoparquet-lowzoom-sparse).
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
GEOM_CACHE = DATA_DIR / "provo-ut-geometry.parquet"
BOUNDARY_CACHE = DATA_DIR / "provo-boundary.geojson"

UGRC = "https://services1.arcgis.com/99lidPhWCzftIe9K/arcgis/rest/services"
PARCELS_URL = f"{UGRC}/Parcels_Utah_LIR/FeatureServer/0/query"
BOUNDARY_URL = f"{UGRC}/UtahMunicipalBoundaries/FeatureServer/0/query"
# Utah County is COUNTYNBR 25. The NAME filter alone would also match nothing else, but the
# county number is kept so a future statewide re-point cannot pick up a same-named place.
BOUNDARY_WHERE = "NAME='Provo' AND COUNTYNBR='25'"

OUT_FIELDS = (
    "OBJECTID,PARCEL_ID,SERIAL_NUM,PARCEL_ADD,PARCEL_CITY,TAXEXEMPT_TYPE,TAX_DISTRICT,"
    "TOTAL_MKT_VALUE,LAND_MKT_VALUE,PARCEL_ACRES,PROP_CLASS,PRIMARY_RES,HOUSE_CNT,"
    "SUBDIV_NAME,BLDG_SQFT,BLDG_SQFT_INFO,FLOORS_CNT,BUILT_YR,CURRENT_ASOF"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}
PAGE = 2000
SQFT_PER_ACRE = 43560.0
SQM_TO_SQFT = 10.763910416709722
UTM = "EPSG:32612"  # UTM 12N — Provo


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

    The envelope (not PARCEL_CITY) is the fetch filter, and the authoritative municipal
    boundary does the actual clip below — PARCEL_CITY is a situs/postal label and is
    demonstrably wrong at the edges (119 parcels inside Provo carry a blank city, and one
    'Orem'-labelled parcel falls inside the Provo boundary). Playbook §4.
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
                    headers=HEADERS, timeout=240)
                r.raise_for_status()
                feats = json.loads(r.content).get("features", [])
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
        # maxRecordCount is 2000; a SHORT page means the end (do not break on 0 only —
        # see the Seattle pagination note in the add-city skill).
        if len(gdf) < PAGE:
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
raw["geometry"] = raw["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
raw = raw[raw["geometry"].notnull() & raw["geometry"].apply(
    lambda g: getattr(g, "is_valid", False) and not g.is_empty)].copy()
city = boundary.geometry.union_all()
inside = gpd.GeoSeries(raw.geometry.representative_point(), crs="EPSG:4326").within(city)
raw = raw[inside.values].copy()
log(f"Inside the Provo municipal boundary: {len(raw):,} rows")

for c in ["TOTAL_MKT_VALUE", "LAND_MKT_VALUE", "PARCEL_ACRES", "BLDG_SQFT", "FLOORS_CNT"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw["pid"] = raw["PARCEL_ID"].astype(str).str.strip()
raw = raw[raw["pid"].ne("") & raw["pid"].ne("None") & raw["pid"].ne("nan")].copy()

# ── dedup: ONE ROW PER BUILDING -> one row per parcel ─────────────────────────
# Values are parcel-level and repeated identically on every building row, so they take
# `first` and are NEVER summed (skill §2). BLDG_SQFT is per-building and DOES sum. The
# building TYPE kept is the largest building's, so a house with a tool shed classifies as a
# house rather than as whatever row the assessor happened to list first.
nrows, npar = len(raw), raw["pid"].nunique()
log(f"Building rows -> parcels: {nrows:,} -> {npar:,} "
    f"(max rows on one parcel: {int(raw['pid'].value_counts().max())})")
raw = raw.sort_values("BLDG_SQFT", ascending=False, na_position="last")
first_cols = ["PARCEL_ADD", "PARCEL_CITY", "SERIAL_NUM", "TAX_DISTRICT", "TAXEXEMPT_TYPE",
              "TOTAL_MKT_VALUE", "LAND_MKT_VALUE", "PARCEL_ACRES", "PROP_CLASS", "PRIMARY_RES",
              "HOUSE_CNT", "SUBDIV_NAME", "BLDG_SQFT_INFO", "BUILT_YR"]
agg = {c: "first" for c in first_cols}
agg["BLDG_SQFT"] = "sum"
agg["FLOORS_CNT"] = "max"
parcel = raw.groupby("pid", dropna=False).agg(agg).reset_index()
parcel["geometry"] = raw.groupby("pid", dropna=False)["geometry"].apply(
    lambda gs: unary_union([g for g in gs if g is not None])).values
parcel = gpd.GeoDataFrame(parcel, geometry="geometry", crs="EPSG:4326")
parcel["geometry"] = parcel["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
log(f"After PARCEL_ID dedup -> {len(parcel):,}")

parcel["land_val"] = parcel["LAND_MKT_VALUE"]
parcel["tot_appr_val"] = parcel["TOTAL_MKT_VALUE"]
parcel["bld_val"] = (parcel["tot_appr_val"] - parcel["land_val"]).clip(lower=0)
parcel["plat"] = parcel["pid"].str[:5]  # Utah County serial MM:PPP:LLLL — MM:PPP is the plat map

# ── condo / PUD units -> merge DOWN onto their plat's common-area land (§6b) ──
# Structure, verified across ~300 Provo plats: each unit is mapped as its own building-footprint
# stub (~550-1,100 sqft) carrying a share of the development's land value ($45k-$300k), and the
# development's REAL land is one or more UNVALUED parcels in the same plat. Spring Creek
# (plat 35771) is the canonical case: 49 stubs at $305,500 land each sitting beside two unvalued
# parcels of 6.90 + 3.79 acres. Summed onto that land the development lands at ~$32/sqft, i.e.
# right on the citywide median of ~$25 — which is the calibration test skill §6b step 5 asks for.
#
# The gate is PLAT DOMINANCE, not mere adjacency. A plat merges only when its stubs are >=60%
# of its valued parcels. That is what separates a pure condo/PUD regime (plat 37118: 120 stubs,
# one 3.9-acre common parcel, 0 ordinary lots) from a MIXED subdivision (plat 36196: 8 stubs
# among 23 valued parcels, most of them ordinary 4,500 sqft townhouse lots at a perfectly normal
# $18/sqft, plus a 1.07-acre HOA open space serving all of them). Merging the mixed case would
# hand the whole HOA open space to 8 units that do not own it — the failure run_lynchburg.py
# guards against with its "common-area parcels never chain" rule.
STUB_SQFT = 2000.0      # above this a small parcel is a real lot with a yard, not a unit stub
COMMON_MIN_SQFT = 200.0  # ignore slivers; real common-area parcels run 4k-300k sqft
STUB_SHARE = 0.60        # stubs must dominate the plat's valued parcels

_utm = parcel.to_crs(UTM)
parcel["_sqft"] = _utm.geometry.area * SQM_TO_SQFT
parcel["_valued"] = parcel["land_val"].notna() & parcel["land_val"].gt(0)
parcel["_is_stub"] = parcel["_valued"] & parcel["_sqft"].lt(STUB_SQFT)
parcel["_is_common"] = ~parcel["_valued"] & parcel["_sqft"].ge(COMMON_MIN_SQFT)

_g = parcel.groupby("plat")
_stats = pd.DataFrame({"n_val": _g["_valued"].sum(), "n_stub": _g["_is_stub"].sum(),
                       "n_com": _g["_is_common"].sum()})
_stats["share"] = _stats["n_stub"] / _stats["n_val"].replace(0, np.nan)
merge_plats = set(_stats.index[(_stats["n_stub"] >= 2) & (_stats["n_com"] >= 1)
                               & (_stats["share"] >= STUB_SHARE)])
log(f"Condo/PUD plats to merge: {len(merge_plats):,}")


def _fill_holes(g):
    """Solid exterior. The common parcel's holes are the punched-out unit footprints, which
    belong to the development — unlike a fee-simple lot carved out of a parent (skill §6b)."""
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
    units = blk[blk["_is_stub"]]
    commons = blk[blk["_is_common"]]
    geom = _fill_holes(unary_union(list(units.geometry) + list(commons.geometry)))
    row = units.iloc[0].to_dict()
    row.update({
        "pid": units["pid"].iloc[0],          # link to a representative unit's assessor page
        "geometry": geom,
        "land_val": units["land_val"].sum(),
        "tot_appr_val": units["tot_appr_val"].sum(),
        "bld_val": units["bld_val"].sum(),
        "BLDG_SQFT": units["BLDG_SQFT"].sum(),
        "BLDG_SQFT_INFO": _dominant(units["BLDG_SQFT_INFO"]),
        "PROP_CLASS": _dominant(units["PROP_CLASS"]),
        "HOUSE_CNT": str(int(pd.to_numeric(units["HOUSE_CNT"], errors="coerce").fillna(1).sum())),
        "PARCEL_ACRES": np.nan,               # per-unit shares; the union polygon is the land
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
    _dev_sqft = devs.to_crs(UTM).geometry.area * SQM_TO_SQFT
    _psf = devs["land_val"] / _dev_sqft
    log(f"Condo merge: {int(devs['_merged_units'].sum()):,} unit stubs -> {len(devs):,} "
        f"development parcels (land $/sqft p50 ${_psf.median():,.0f}, max ${_psf.max():,.0f})")
else:
    parcel["_merged_units"] = 1
_left = parcel["_valued"].fillna(True) & parcel["_sqft"].lt(STUB_SQFT) & parcel["_merged_units"].eq(1)
log(f"Unit-sized parcels left individual (no common-area land mapped in their plat): "
    f"{int(_left.sum()):,}")
parcel = parcel.drop(columns=["_sqft", "_valued", "_is_stub", "_is_common"], errors="ignore")

# ── exemption flag ───────────────────────────────────────────────────────────
# Utah County publishes NO exemption field (TAXEXEMPT_TYPE is null on every row), because
# exempt land is simply not assessed. "No land value" IS the flag. The parcels this drops are
# the national forest sections east of town, the airport, BYU, schools, parks, ROW and the
# leftover HOA open space that no condo development claimed above.
parcel["exemption_flag"] = (~(parcel["land_val"].notna() & parcel["land_val"].gt(0))).astype(int)
log(f"Unassessed (exempt / common area / ROW) -> excluded: {int(parcel['exemption_flag'].sum()):,}")
ex = parcel[parcel["exemption_flag"] == 0].copy()
log(f"Shipped parcels -> {len(ex):,}")

# ── classification ───────────────────────────────────────────────────────────
# Keyed on BLDG_SQFT_INFO (the largest building's type) because PROP_CLASS is unusable on its
# own here: 14,665 Provo rows are "Unknown" and Utah files non-primary residential under
# "Commercial" (it is taxed at the full rate), so a duplex rented to students reads Commercial.
SF_TYPES = {
    "one_story", "two_story", "split_level", "bi-level", "one_and_one_half", "log:_one_story",
    "a-frame", "cabin", "basement_home", "livable_space", "guest_house",
}
TOWNHOME_TYPES = {
    "end:_one_story", "end:_two_story", "end:_split_level",
    "int:_one_story", "int:_two_story", "int:_split_level",
}
MF_SMALL_TYPES = {"duplex", "triplex", "fourplex:_one_story", "fourplex:_two_story"}
MF_TYPES = {
    "multiple_residence", "apartments_(high-rise)", "dormitory_residence_halls", "rooming_house",
    "multi_res_-_assisted_living", "multi_res_senior_citizen", "home_for_the_elderly",
    "group_care_homes", "convalescent_hospital", "office_-_apartment", "mixed_retail_w/_res_units",
}
MOBILE_TYPES = {
    "one-section_12'_wide", "one-section_14'_wide", "one-section_16'_wide",
    "two-section_20'_wide", "two-section_24'_wide", "two-section_28'_wide",
    "transient_labor_cabin",
}
INDUSTRIAL_TYPES = {
    "indust_light_mfg", "indust_engineering_(r&d)", "storage_warehouse", "cold_storage_warehouse",
    "distribution_warehouse", "light_indust_warehouse_shell", "mini_warehouse",
    "industrial_flex_(mall)", "industrial_flex_(mall)_shell", "material_storage",
    "material_shelters", "material_storage_shed", "quonset_commercial", "storage_garage",
    "storage_hangar", "light_commercial_(shop)", "light_commercial_utility",
    "computer_data_center", "mini_lube_garage", "service_garage",
}
AG_PREFIXES = ("farm_", "barn", "greenhouse", "dairy_", "poultry_", "stable", "equestrian_",
               "concrete_poured", "concrete_stave", "steel:_")
AG_TYPES = {"loafing_shed", "open_hay_shed", "horse_arena", "kennel", "shed_-_equipment",
            "arch-rib_(quonset)_implement", "arch-rib_(quonset)_utility"}
# Structures that never tell you what a parcel IS — a shed, a carport or a detached garage sits
# next to whatever the real use is. These fall through to the PROP_CLASS/HOUSE_CNT fallback.
ACCESSORY_TYPES = {
    "shed_tool", "shed:_wood", "shed:_aluminum", "shed:_steel", "prefab_storage/shed",
    "secure_storage_modular_shed", "shed_office_structure", "detached", "built-in",
    "individual_-_open_carport", "individual_-_closed_carport", "multi_-_open_carport",
    "multi_-_closed_carport", "pavilion", "restroom_bldg", "bath_house", "clubhouse",
    "recreational_(pool)_enclosure", "mechanical_penthouse",
}


def categorize(bld_type, prop_class, house_cnt, land, total):
    """Provo property category. Building type first, assessor class only as the fallback."""
    t = str(bld_type or "").strip().lower()
    cls = str(prop_class or "").strip()
    try:
        units = int(float(house_cnt))
    except (TypeError, ValueError):
        units = 0
    impr = (total or 0) - (land or 0)

    if cls == "Vacant":
        return "Vacant Land"
    if t == "parking_structure":
        return "Parking Garage"
    if t in TOWNHOME_TYPES:
        return "Townhome"
    if t.endswith("_unit_building"):
        # "12_unit_building" etc. — an apartment/condo building. Which one it is depends on
        # whether the parcel is the whole building or one stacked unit inside it; after the
        # merge above a surviving record is the development, so Multifamily is the honest label.
        return "Multifamily"
    if t in MF_SMALL_TYPES or t in MF_TYPES:
        return "Multifamily"
    if t in MOBILE_TYPES:
        return "Mobile Home"
    if t in SF_TYPES:
        return "Multifamily" if units >= 3 else "Single Family"
    if t in INDUSTRIAL_TYPES:
        return "Industrial"
    if t in AG_TYPES or t.startswith(AG_PREFIXES):
        return "Agricultural / Rural"
    if t and t not in ACCESSORY_TYPES:
        return "Commercial"          # the remaining ~90 types are all commercial uses
    # No building, or an accessory-only one: fall back to the assessor class.
    if cls == "Residential":
        return "Multifamily" if units >= 3 else "Single Family"
    if cls == "Commercial":
        return "Commercial"
    if impr <= 0:
        return "Vacant Land"
    return "Other"


ex["property_land_use_category"] = [
    categorize(t, c, h, lv, tv) for t, c, h, lv, tv in zip(
        ex["BLDG_SQFT_INFO"], ex["PROP_CLASS"], ex["HOUSE_CNT"], ex["land_val"], ex["tot_appr_val"])]

ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")
ex["bld_ar"] = pd.to_numeric(ex["BLDG_SQFT"], errors="coerce").fillna(0)
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other", "Agricultural / Rural"),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    bld_ar_col="bld_ar",
    fetch_footprints=False)

# ── areas: assessor acreage, guarded, geodesic fallback ──────────────────────
ex["geometry"] = ex["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
log("Computing geodesic areas...")
ex["geom_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
check_area_agreement(ex["geom_area_sqft"], ex["PARCEL_ACRES"] * SQFT_PER_ACRE,
                     label="PARCEL_ACRES", log=log)

ex["reported_sqft"] = pd.to_numeric(ex["PARCEL_ACRES"], errors="coerce") * SQFT_PER_ACRE
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan
# Richmond guard: reported acreage is trusted only when it is in the same ballpark as the
# polygon. It is set to NaN on merged developments above (per-unit shares would be nonsense
# against the union footprint), so those always use the polygon.
ratio = ex["reported_sqft"] / ex["geom_area_sqft"].replace(0, np.nan)
use_reported = ex["reported_sqft"].gt(0) & (ratio.between(0.5, 2.0) | ex["geom_area_sqft"].isna())
log(f"Reported acreage rejected as implausible (outside 0.5-2.0x polygon): "
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
# Canonical column set only. The source also carries the situs address, subdivision and year
# built, which the app never reads; the LIR schema carries no owner name at all, which is the
# right side of the issue-#12 rule.
COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant"]
for c in COLUMNS:
    if c not in ex.columns:
        ex[c] = np.nan
final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final["geometry"] = final["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
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
