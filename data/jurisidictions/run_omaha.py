#!/usr/bin/env python3
"""
Build the City of Omaha, NE canonical parcel parquet.

Omaha is the Douglas County seat and Nebraska's largest city (pop. ~486k, ~149 sq mi).

SOURCE — Douglas County / City of Omaha joint GIS (DOGIS), public ArcGIS Server, no token:
  https://dcgis.org/server/rest/services/vector/Cityworks_Parcels_DCHD/FeatureServer/0
  217,384 countywide rows: geometry + the Assessor's LAND_VALUE / IMPROVEMEN / TOTAL_VALU +
  CLASS (R/C/I/M/F/E/U) + DCAACCTYPE (account type, e.g. "Ex Government") + CONDO flag +
  BLDG_DESC + ACRES/SQ_FEET + an ASSESSOR deep link (Beacon) per PIN.
  City boundary: https://dcgis.org/server/rest/services/Hosted/City_Limits_(source)_view/FeatureServer/0
  (town='Omaha', 148.9 sq mi).

WHY THIS LAYER and not `vector/Parcels_public`: the "public" parcel layer is the same 217,384
polygons but carries NO value fields at all (PIN, owner, address, legal, building only). The
Cityworks parcel layers (CCSM / DCHD / PRPP / EQC — one parcel layer republished under four names
for the City's work-order system) carry the Assessor's values. `vector/Parcels` needs a token.

OWNER_NAME / OWNER / ADDRESS1-2 / OWNER_CITY are deliberately NOT requested (issue #12): they never
enter the pipeline.

Outputs:
- data/jurisidictions/data/omaha/omaha-ne-parcels.parquet
- data/jurisidictions/data/omaha/omaha-ne-parcels_YYYY_MM_DD.parquet

Notes:
- VALUES ARE 100% OF ACTUAL (MARKET) VALUE. Nebraska assesses residential and commercial real
  property at 100% of actual value (Neb. Rev. Stat. 77-201(1)) — the median Omaha house here is
  $247k, i.e. market, not a fractional basis; no Georgia-style ratio correction. Agricultural and
  horticultural land is assessed at 75% (77-201(2)), but only ~130 Omaha parcels are CLASS F, so
  it is documented rather than grossed up.
- EXEMPTION is explicit: CLASS E (exempt — government, religious, charitable, educational,
  cemetery) and CLASS U / DCAACCTYPE "State Assd ..." (railroad, pipeline and public-utility
  property, which the State values centrally and the county carries at $0) are excluded. Both
  carry $0 land here. "... Par Ex ..." (partially exempt) and "... Govt Lse" (taxable leasehold
  on government land) accounts DO carry assessed value and are kept.
- STACKED FOOTPRINTS (the Fort Collins pattern, playbook §5): Douglas County draws EVERY condo
  unit as the WHOLE development's lot polygon, so a tower is up to ~400 identical polygons, each
  carrying one unit's land + improvement value. Separately, Omaha TIF ("OMAHA REDEVELOPMENT")
  parcels are split into two PINs on one identical polygon — a base account and a TIF-excess
  account with DIFFERENT values. Both are distinct value records on one footprint, so the ETL
  collapses identical footprints into one parcel and SUMS the values (skill §2a: "several value
  records with distinct values for one footprint -> sum"). Multi-polygon PINs (5 in the city) are
  the opposite case — one value broadcast to several polygons — and take FIRST + union.
- TOWNHOME/PUD FOOTPRINT LOTS (skill §6a/§6b): a few newer subdivisions plat each townhouse as a
  ~500-sqft footprint lot with the open land in $100 HOA outlots. Those merge down onto their
  outlots under a plat-dominance gate (see §3b). Rowhouse subdivisions with NO outlot (Rows at
  SoMa, Dahlman Rows ...) are genuine fee-simple lots and stay individual.
- CONDO LAND IS THE ASSESSOR'S OWN SPLIT: units carry ~3% of their value as land (vs ~10% for
  houses), so summed condo developments read a little low on land. Published as-is.
- ~152k shipped parcels -> PMTiles + H3.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
from shapely import make_valid
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import (  # noqa: E402
    add_improvement_ratio_fields,
    check_area_agreement,
    classify_property_refined,
    gis_area_sqft,
)

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "omaha"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "omaha-ne-county-geometry.parquet"
BOUNDARY_CACHE = DATA_DIR / "omaha-boundary.geojson"

PARCELS_URL = ("https://dcgis.org/server/rest/services/vector/Cityworks_Parcels_DCHD/"
               "FeatureServer/0/query")
BOUNDARY_URL = ("https://dcgis.org/server/rest/services/Hosted/City_Limits_(source)_view/"
                "FeatureServer/0/query")
BOUNDARY_WHERE = "town='Omaha'"

# Owner fields are deliberately absent — see the module docstring.
OUT_FIELDS = (
    "PIN,PROPERTY_A,HOUSE,STREET_DIR,STREET_NAM,STREET_TYP,APARTMENT,PROP_CITY,PROP_ZIP,"
    "SEC_TWN_RN,ADDITION_N,BLOCK,LOT,LAND_VALUE,IMPROVEMEN,TOTAL_VALU,DCAACCTYPE,CLASS,SID,"
    "TAX_DIST,CITY,REDEVLPMNT,NUMBLDGS,BLDG_SF,BLDG_STORY,BLDG_YRBLT,BLDG_DESC,QUALITY,"
    "CONDITION,CONDO,SUBNO,ACRES,SQ_FEET,ASSESSOR,X_COORD,Y_COORD"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}
PAGE = 2000  # the layer's maxRecordCount
SQFT_PER_ACRE = 43560.0
NE_STATE_PLANE_FT = "EPSG:26852"  # NAD83 / Nebraska (ftUS) — the layer's native CRS
BEACON = ("https://beacon.schneidercorp.com/Application.aspx?AppID=1221&LayerID=37105"
          "&PageTypeID=4&PageID=14212&KeyValue=")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _polygonal(g):
    """Keep only the polygonal part of a make_valid() result (GeometryCollection -> polygons)."""
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
    gs = [g for g in geoms if g is not None]
    try:
        return unary_union(gs)
    except Exception:  # noqa: BLE001 - last-resort repair for stubborn topology
        return unary_union([_polygonal(make_valid(g)).buffer(0) for g in gs])


def _txt(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


# ── fetch ─────────────────────────────────────────────────────────────────────
def fetch_boundary() -> gpd.GeoDataFrame:
    if not BOUNDARY_CACHE.exists():
        r = requests.get(BOUNDARY_URL, params={
            "where": BOUNDARY_WHERE, "outFields": "town,area_sq_mi",
            "returnGeometry": "true", "outSR": 4326, "f": "geojson"},
            headers=HEADERS, timeout=120)
        r.raise_for_status()
        BOUNDARY_CACHE.write_bytes(r.content)
    b = gpd.read_file(BOUNDARY_CACHE).to_crs("EPSG:4326")
    if not len(b):
        raise RuntimeError("Omaha city-limits polygon came back empty")
    return b


def fetch_parcels() -> gpd.GeoDataFrame:
    """Whole county (217k rows, ~2 min), cached. The city clip happens spatially below."""
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    total = requests.get(PARCELS_URL, params={"where": "1=1", "returnCountOnly": "true",
                                              "f": "json"}, headers=HEADERS, timeout=120).json()["count"]
    log(f"Pulling {total:,} countywide rows (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(6):
            try:
                r = requests.get(PARCELS_URL, params={
                    "where": "1=1", "outFields": OUT_FIELDS, "returnGeometry": "true",
                    "resultOffset": off, "resultRecordCount": PAGE, "outSR": 4326,
                    "orderByFields": "OBJECTID", "f": "geojson"}, headers=HEADERS, timeout=300)
                r.raise_for_status()
                body = json.loads(r.content)
                # ArcGIS returns HTTP 200 + {"error": ...} on overload; without this it parses
                # as zero features and silently truncates the pull.
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
        if off % 20000 < PAGE:
            log(f"  fetched {off:,}/{total:,}")
        if len(gdf) < PAGE:      # a SHORT page is the end
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    if len(geom) != total:
        raise RuntimeError(f"Pulled {len(geom):,} rows but the layer reports {total:,}")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


boundary = fetch_boundary()
raw = fetch_parcels()
log(f"Countywide rows: {len(raw):,}")

# ── repair + clip to the authoritative city limits (representative point within) ──
n_bad = int((~raw.geometry.is_valid).sum())
raw["geometry"] = raw["geometry"].apply(
    lambda g: g if (g is not None and g.is_valid) else _polygonal(make_valid(g)))
raw = raw[raw["geometry"].notna() & ~raw.geometry.is_empty].copy()
log(f"Repaired {n_bad:,} invalid source geometries")
city = boundary.geometry.union_all()
inside = gpd.GeoSeries(raw.geometry.representative_point(), crs="EPSG:4326").within(city)
log("Clip cross-check vs the layer's CITY tax-district label (NOT used to filter): "
    f"{raw.loc[inside.values, 'CITY'].fillna('NULL').value_counts().to_dict()}")
raw = raw[inside.values].copy()
log(f"Inside the Omaha city limits: {len(raw):,} rows")

for c in ["LAND_VALUE", "IMPROVEMEN", "TOTAL_VALU", "ACRES", "SQ_FEET", "CONDO", "BLDG_SF",
          "NUMBLDGS"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw["PIN"] = raw["PIN"].astype(str).str.strip()

# ── 1. multi-polygon PINs: one value broadcast to several polygons -> FIRST + union ──
dup = raw[raw["PIN"].duplicated(keep=False)]
log(f"PINs with >1 geometry row (preflight, skill §2a): {dup['PIN'].nunique():,} "
    f"({len(dup):,} rows)")
for c in ["LAND_VALUE", "IMPROVEMEN", "TOTAL_VALU"]:
    n_var = int(dup.groupby("PIN")[c].nunique().gt(1).sum())
    assert n_var == 0, f"{c} differs across a PIN's polygons on {n_var} PINs — not a broadcast"
attr_cols = [c for c in raw.columns if c not in ("geometry", "PIN")]
pin = raw.groupby("PIN", sort=False).agg({c: "first" for c in attr_cols}).reset_index()
pin["geometry"] = raw.groupby("PIN", sort=False)["geometry"].apply(_union).values
pin = gpd.GeoDataFrame(pin, geometry="geometry", crs="EPSG:4326")
assert not pin["PIN"].duplicated().any()
log(f"After PIN dedup -> {len(pin):,}")

# ── 2. exemption, record level ────────────────────────────────────────────────
acct = pin["DCAACCTYPE"].fillna("").astype(str)
pin["_exempt"] = (pin["CLASS"].isin(["E", "U"])
                  | acct.str.startswith("Ex ")
                  | acct.str.startswith("State Assd")
                  | pin["CLASS"].isna())
log("Exempt records by account type: "
    f"{pin.loc[pin['_exempt'], 'DCAACCTYPE'].fillna('NULL').value_counts().head(8).to_dict()}")
log(f"  exempt records' land value total: ${pin.loc[pin['_exempt'], 'LAND_VALUE'].sum():,.0f} "
    "(expect ~0 — Douglas County carries exempt/state-assessed parcels at $0)")
for c in ["LAND_VALUE", "IMPROVEMEN", "TOTAL_VALU"]:
    pin[c] = pin[c].where(~pin["_exempt"], 0).fillna(0)

# ── 3. stacked footprints: condo units + TIF base/excess splits -> ONE parcel, SUM ──
# Key = the polygon itself (normalised, snapped to 1e-7 deg ~ 1 cm), so only truly identical
# footprints collapse. Every record on a stack is a distinct value record (a unit, or the TIF
# excess account), so the values SUM. Exempt records add $0 but still count toward the stack.
pin["_gk"] = [hashlib.md5(shapely.normalize(shapely.set_precision(g, 1e-7)).wkb).hexdigest()
              for g in pin.geometry]
# A handful of stacks were digitised twice with sub-foot vertex differences (16 pairs, ~32 acres,
# e.g. PINs 0644190490/0644190492), so the exact key misses them. Fold any two footprints that
# overlap >=95% of BOTH areas into one key (union-find) before aggregating.
_fp = pin.groupby("_gk", sort=False)["geometry"].first()
_fp = gpd.GeoDataFrame({"_gk": _fp.index}, geometry=_fp.values, crs="EPSG:4326").to_crs(NE_STATE_PLANE_FT)
_pairs = gpd.sjoin(_fp, _fp, predicate="intersects")
_pairs = _pairs[_pairs.index < _pairs["index_right"]]
_parent = {k: k for k in _fp["_gk"]}


def _find(k):
    while _parent[k] != k:
        _parent[k] = _parent[_parent[k]]
        k = _parent[k]
    return k


_n_fuzzy = 0
for i, j in zip(_pairs.index, _pairs["index_right"]):
    a, b = _fp.geometry.iloc[i], _fp.geometry.iloc[j]
    inter = a.intersection(b).area
    if inter >= 0.95 * a.area and inter >= 0.95 * b.area:
        _parent[_find(_fp["_gk"].iloc[j])] = _find(_fp["_gk"].iloc[i])
        _n_fuzzy += 1
pin["_gk"] = pin["_gk"].map(_find)
log(f"Near-identical footprints folded into one stack (>=95% mutual overlap): {_n_fuzzy:,} pairs")
pin["_n"] = pin.groupby("_gk")["PIN"].transform("size")
st = pin[pin["_n"] > 1]
log(f"Stacked footprints: {st['_gk'].nunique():,} footprints carry {len(st):,} records "
    f"(max stack {int(pin['_n'].max())}); condo-unit records among them: "
    f"{int(st['CONDO'].eq(1).sum()):,}; TIF ('OMAHA REDEVELOPMENT') records: "
    f"{int(st['REDEVLPMNT'].fillna('').str.contains('REDEV').sum()):,}")

# The record that speaks for the stack (category, building, link) is the most valuable TAXABLE one.
pin["_rank"] = pin["TOTAL_VALU"] + np.where(pin["_exempt"], -1e15, 0)
pin = pin.sort_values(["_gk", "_rank"], ascending=[True, False])
g = pin.groupby("_gk", sort=False)
parcel = g.agg({c: "first" for c in attr_cols + ["PIN"]})
parcel["LAND_VALUE"] = g["LAND_VALUE"].sum()
parcel["IMPROVEMEN"] = g["IMPROVEMEN"].sum()
parcel["TOTAL_VALU"] = g["TOTAL_VALU"].sum()
parcel["BLDG_SF"] = g["BLDG_SF"].sum()
parcel["_records"] = g["PIN"].size()
parcel["_condo_units"] = g["CONDO"].apply(lambda s: int((s == 1).sum()))
parcel["_n_taxable"] = g["_exempt"].apply(lambda s: int((~s).sum()))
parcel["_tif"] = g["REDEVLPMNT"].apply(lambda s: s.fillna("").str.contains("REDEV").any())
parcel["geometry"] = g["geometry"].apply(lambda s: s.iloc[0] if len(s) == 1 else _union(list(s)))
parcel = gpd.GeoDataFrame(parcel.reset_index(drop=True), geometry="geometry", crs="EPSG:4326")
log(f"After footprint collapse -> {len(parcel):,} parcels "
    f"({int((parcel['_condo_units'] > 1).sum()):,} condo developments, "
    f"{int(parcel['_tif'].sum()):,} TIF parcels)")

# ── 3b. townhome/PUD footprint lots -> merge DOWN onto their subdivision's outlots (§6b) ──
# Omaha's newer townhome/rowhouse subdivisions plat each unit as a BUILDING-FOOTPRINT lot
# (Villages on Sprague: 129 lots of 528 sqft at $10,000 land each) and put the development's
# open land in outlots ("OLA", "OLB" ...) that the assessor carries at a nominal $100 or exempts.
# Left alone the units render as pencils at ~$19/sqft beside $100 outlots at $0.0004/sqft, with
# the neighbourhood at ~$2-3. The gate is PLAT DOMINANCE as in run_provo.py — a subdivision
# (SUBNO) merges only when its sub-2,000-sqft valued lots are >=60% of its valued parcels and it
# has at least one common outlot — so a normal subdivision with a few small lots is untouched, and
# the huge catch-all SUBNOs ("LANDS", "CITY LOTS") can never qualify. Outlots deeded to the City
# (Ex Government: detention ponds, streets) are not HOA land and are never merged.
STUB_SQFT = 2000.0
COMMON_MIN_SQFT = 200.0
COMMON_MAX_LAND = 1000.0   # the assessor's nominal outlot value is $100
STUB_SHARE = 0.60
parcel["_sqft"] = parcel.to_crs(NE_STATE_PLANE_FT).geometry.area
parcel["_taxable"] = (parcel["_n_taxable"] > 0) & (parcel["LAND_VALUE"] > 0)
_gov = parcel["DCAACCTYPE"].fillna("").str.startswith(("Ex Government", "State Assd"))
_sub = parcel["SUBNO"].fillna("").str.strip()
parcel["_is_stub"] = (parcel["_taxable"] & parcel["_sqft"].lt(STUB_SQFT)
                      & parcel["_records"].eq(1) & _sub.ne("")
                      & parcel["LAND_VALUE"].gt(COMMON_MAX_LAND))  # a $100 lot is an outlot
parcel["_is_common"] = (~_gov & parcel["_sqft"].ge(COMMON_MIN_SQFT) & parcel["_records"].eq(1)
                        & (parcel["_n_taxable"].eq(0) | parcel["LAND_VALUE"].le(COMMON_MAX_LAND)))
_g = parcel.assign(_sub=_sub).groupby("_sub")
_stats = pd.DataFrame({"n_val": _g["_taxable"].sum(), "n_stub": _g["_is_stub"].sum(),
                       "n_com": _g["_is_common"].sum()})
_stats["share"] = _stats["n_stub"] / _stats["n_val"].replace(0, np.nan)
merge_subs = set(_stats.index[(_stats["n_stub"] >= 2) & (_stats["n_com"] >= 1)
                              & (_stats["share"] >= STUB_SHARE)]) - {""}
log(f"Townhome/PUD subdivisions to merge onto their outlots: {len(merge_subs):,}")
parcel["_pud_units"] = 0
dev_rows, consumed = [], set()
for sub in sorted(merge_subs):
    blk = parcel[_sub.eq(sub)]
    units = blk[blk["_is_stub"]]
    commons = blk[blk["_is_common"]]
    row = units.sort_values("TOTAL_VALU", ascending=False).iloc[0].to_dict()
    row.update({
        "geometry": _union(list(units.geometry) + list(commons.geometry)),
        "LAND_VALUE": units["LAND_VALUE"].sum() + commons["LAND_VALUE"].sum(),
        "IMPROVEMEN": units["IMPROVEMEN"].sum() + commons["IMPROVEMEN"].sum(),
        "TOTAL_VALU": units["TOTAL_VALU"].sum() + commons["TOTAL_VALU"].sum(),
        "BLDG_SF": units["BLDG_SF"].sum(),
        "_records": len(units) + len(commons),
        "_n_taxable": int(units["_n_taxable"].sum() + commons["_n_taxable"].sum()),
        "_pud_units": len(units),
    })
    dev_rows.append(row)
    consumed.update(units.index)
    consumed.update(commons.index)
if dev_rows:
    devs = gpd.GeoDataFrame(pd.DataFrame(dev_rows), geometry="geometry", crs="EPSG:4326")
    _psf = devs["LAND_VALUE"] / devs.to_crs(NE_STATE_PLANE_FT).geometry.area
    _before = parcel.loc[parcel.index.isin(consumed) & parcel["_is_stub"]]
    log(f"PUD merge: {int(devs['_pud_units'].sum()):,} footprint lots + "
        f"{len(consumed) - int(devs['_pud_units'].sum()):,} outlots -> {len(devs):,} developments; "
        f"land $/sqft p50 ${(_before['LAND_VALUE'] / _before['_sqft']).median():,.2f} as lots -> "
        f"${_psf.median():,.2f} merged (max ${_psf.max():,.2f})")
    parcel = gpd.GeoDataFrame(pd.concat([parcel.drop(index=list(consumed)), devs],
                                        ignore_index=True), geometry="geometry", crs="EPSG:4326")
_left = parcel["_is_stub"].fillna(False) & parcel["_pud_units"].eq(0)
log(f"Footprint-sized lots left individual (no outlot in their subdivision, or a mixed "
    f"subdivision): {int(_left.sum()):,}")

# ── 4. exemption filter (parcel level) ────────────────────────────────────────
# A footprint is shipped when at least one taxable record on it carries land value. Taxable
# records with $0 land are mostly TIF-excess / condo-common shells already summed into a stack;
# the stand-alone ones would render as zero-value gp-error parcels, so they drop too.
parcel["exemption_flag"] = ((parcel["_n_taxable"] == 0) | (parcel["LAND_VALUE"] <= 0)).astype(int)
_n0 = int(((parcel["_n_taxable"] > 0) & (parcel["LAND_VALUE"] <= 0)).sum())
log(f"Excluded: {int(parcel['exemption_flag'].sum()):,} parcels "
    f"({int((parcel['_n_taxable'] == 0).sum()):,} wholly exempt/state-assessed, "
    f"{_n0:,} taxable-but-$0-land)")
ex = parcel[parcel["exemption_flag"] == 0].copy()
log(f"Shipped parcels -> {len(ex):,}")

# ── 5. classification ─────────────────────────────────────────────────────────
PARKING_DESC = {"Parking Structure", "Parking Levels", "Underground Parking Garage"}
MIXED_DESC = {"Mixed Retail w/ Res Units", "Mixed Retail w/ Office Units"}
MF_DESC_PREFIX = ("Apartment", "Duplex", "Triplex", "Multiple -", "Multiple Residences",
                  "Group Care", "Convlsnt", "Fraternity")
MOBILE_DESC = {"Single Wide", "Double Wide", "Mobile Home Parks  *CODE"}


def categorize(r) -> str:
    cls = _txt(r["CLASS"])
    desc = _txt(r["BLDG_DESC"])
    acct = _txt(r["DCAACCTYPE"])
    impr = r["IMPROVEMEN"] or 0
    if r["_condo_units"] >= 2 or (r["CONDO"] == 1 and desc.startswith("Condo")):
        return "Condominium" if cls in ("R", "M") else "Commercial Condominium"
    if r["_pud_units"] >= 2:
        return "Townhome" if cls in ("R", "M") else "Commercial"
    if desc == "Parking Lot":
        return "Parking Lot"
    if impr <= 0 and not desc:
        return "Vacant Land"
    if cls == "F" or acct.startswith("Agricultur") or acct.startswith("Farm"):
        return "Agricultural / Rural"
    if desc in PARKING_DESC:
        return "Parking Garage"
    if desc in MIXED_DESC:
        return "Mixed Use"
    if desc in MOBILE_DESC:
        return "Mobile Home"
    if desc.startswith("Townhouse"):
        return "Townhome"
    if cls == "M" or desc.startswith(MF_DESC_PREFIX):
        return "Multifamily"
    if cls == "I":
        return "Industrial"
    if cls == "C":
        return "Commercial"
    if cls == "R":
        return "Single Family"
    return "Other"


ex["property_land_use_category"] = ex.apply(categorize, axis=1)
ex["land_value"] = ex["LAND_VALUE"].astype(float)
ex["improvement_value"] = ex["IMPROVEMEN"].astype(float)
ex["bld_ar"] = ex["BLDG_SF"].fillna(0)
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other", "Agricultural / Rural"),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    bld_ar_col="bld_ar", fetch_footprints=False)

# ── 6. areas: assessor SQ_FEET, guarded, geodesic fallback ────────────────────
log("Computing geodesic areas...")
ex["geom_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
ex["reported_sqft"] = ex["SQ_FEET"].where(ex["SQ_FEET"] > 0)
# Condo units carry SQ_FEET 0 and stacks sum to nonsense, so a collapsed stack always uses the
# polygon (which IS the development's lot).
ex.loc[ex["_records"] > 1, "reported_sqft"] = np.nan
check_area_agreement(ex["geom_area_sqft"], ex["reported_sqft"], label="assessor SQ_FEET", log=log)
ratio = ex["reported_sqft"] / ex["geom_area_sqft"].replace(0, np.nan)
use_reported = ex["reported_sqft"].gt(0) & (ratio.between(0.5, 2.0) | ex["geom_area_sqft"].isna())
log(f"Assessor SQ_FEET rejected as implausible (outside 0.5-2.0x polygon): "
    f"{int((ex['reported_sqft'].gt(0) & ~use_reported).sum()):,}")
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["geom_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = ex["TOTAL_VALU"].astype(float)
_mismatch = (ex["full_market_value"] - (ex["land_value"] + ex["improvement_value"])).abs() > 1
log(f"TOTAL != LAND + IMPROVEMENT on {int(_mismatch.sum()):,} parcels")
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

# Douglas County Assessor's parcel page on Beacon (the layer's own ASSESSOR field, rebuilt from
# the PIN so a blank field still links). For a condo stack this is the most valuable unit's page.
ex["link"] = BEACON + ex["PIN"].astype(str)
ex["parcel_id"] = ex["PIN"].astype(str)
ex["address"] = ex["PROPERTY_A"].fillna("").astype(str).str.strip().replace("", None)
# Value records summed onto this footprint: condo units, PUD footprint lots + outlots, or a TIF
# base/excess pair. 1 for an ordinary parcel.
ex["value_records"] = ex["_records"].astype(int)
ex["tif_parcel"] = ex["_tif"].fillna(False).astype(int)

# ── export ───────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant", "parcel_id", "address", "value_records", "tif_parcel"]
final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final["geometry"] = final["geometry"].apply(
    lambda g: g if (g is None or g.is_valid) else _polygonal(make_valid(g)))
final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")
out = DATA_DIR / "omaha-ne-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"omaha-ne-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet",
                 index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"land value total: ${final['current_full_land_value'].sum() / 1e9:,.2f}B | "
    f"market value total: ${final['full_market_value'].sum() / 1e9:,.2f}B")

# ── post-flight: the shipped totals must equal the taxable roll's totals ──────
roll_land = pin.loc[~pin["_exempt"], "LAND_VALUE"].sum()
shipped_land = final["current_full_land_value"].sum()
dropped_land = parcel.loc[parcel["exemption_flag"] == 1, "LAND_VALUE"].sum()
log(f"Roll (taxable PINs in city) land ${roll_land:,.0f} = shipped ${shipped_land:,.0f} "
    f"+ dropped ${dropped_land:,.0f}")
assert abs(roll_land - shipped_land - dropped_land) < 1, "value lost or duplicated in the collapse"

# ── formula-allocated land test (the Provo lesson) ────────────────────────────
log("--- land share of total value by category (a spike at one value = formula allocation) ---")
_share = final["current_full_land_value"] / final["full_market_value"].replace(0, np.nan)
for cat, s in _share.groupby(final["property_land_use_category"]):
    s = s.dropna()
    if len(s) < 50:
        continue
    top = s.round(3).value_counts(normalize=True)
    log(f"  {cat:24s} n={len(s):>7,} median {s.median():.3f}  most common share "
        f"{top.index[0]:.3f} on {top.iloc[0]:.1%} of parcels")

# ── §6a smoke alarms ─────────────────────────────────────────────────────────
log("--- condo/stub smoke alarms (skill §6a) ---")
a = ex["geom_area_sqft"]
lv = final["land_value_per_sqft"]
shown = lv[final["likely_remnant"] == 0]
log(f"  footprint sqft p1/p5/p10/p50: {[round(a.quantile(q)) for q in (.01, .05, .10, .50)]}")
log(f"  sub-500 / sub-1000 sqft footprints: {int((a < 500).sum()):,} / {int((a < 1000).sum()):,}")
log(f"  land $/sqft ALL ROWS    p50/p95/p99/max: ${lv.median():,.2f} / ${lv.quantile(.95):,.2f} / "
    f"${lv.quantile(.99):,.2f} / ${lv.max():,.2f}")
log(f"  land $/sqft AS RENDERED p50/p95/p99/max: ${shown.median():,.2f} / ${shown.quantile(.95):,.2f} / "
    f"${shown.quantile(.99):,.2f} / ${shown.max():,.2f}  (likely_remnant excluded)")
_cd = final[ex["_condo_units"].values >= 2]
log(f"  condo developments: {len(_cd):,}, land $/sqft p50/p90/max: "
    f"${_cd['land_value_per_sqft'].median():,.2f} / ${_cd['land_value_per_sqft'].quantile(.9):,.2f} / "
    f"${_cd['land_value_per_sqft'].max():,.2f}")
holes = final.geometry.apply(lambda g: 0 if g is None else sum(
    len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
log(f"  parcels with interior rings (holes): {int((holes > 0).sum()):,}")
rp = final.geometry.representative_point()
_vc = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)).value_counts()
log(f"  stacked footprints: {int((_vc > 1).sum()):,} clusters, max stack {int(_vc.max())}")
_sp = final[["geometry"]].to_crs(NE_STATE_PLANE_FT)
_sp["geometry"] = _sp.buffer(-1)  # ignore shared edges
_ov = gpd.sjoin(_sp, _sp, predicate="intersects")
_ov = _ov[_ov.index < _ov["index_right"]]
_ova = sum(_sp.geometry.iloc[i].intersection(_sp.geometry.iloc[j]).area
           for i, j in zip(_ov.index, _ov["index_right"]))
log(f"  overlapping parcel pairs: {len(_ov):,} ({_ova / SQFT_PER_ACRE:,.1f} acres overlapped)")
_tiny = final["land_area_acres"] * SQFT_PER_ACRE < 1500
log(f"  shipped parcels under 1,500 sqft: {int(_tiny.sum()):,} "
    f"({final.loc[_tiny, 'property_land_use_category'].value_counts().head(4).to_dict()})")
log(f"  zero/neg land value (renders as gp-error): "
    f"{int((final['current_full_land_value'].fillna(0) <= 0).sum()):,}")
log(f"  bounds: {[round(v, 4) for v in final.total_bounds]}")
log("DONE")
