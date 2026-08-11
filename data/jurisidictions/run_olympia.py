#!/usr/bin/env python3
"""
Build the City of Olympia, WA canonical parcel parquet.

Olympia is in Thurston County. Thurston County GeoData publishes a single
one-stop countywide parcel layer that carries geometry + assessor land/building/
total values + use type + exemption type, so this is fully automated (no manual
appraisal-roll join). The county layer is countywide (~131k parcels), so we clip
to the City of Olympia municipal boundary (centroid-within) — the postal
SITUS_CITY='OLYMPIA' covers a much larger unincorporated area and is unreliable
for jurisdiction (playbook §4).

Sources (public, no token):
- Parcels (geometry + values + use + exempt), Thurston County GeoData:
  https://tconline.co.thurston.wa.us/server/rest/services/Common_Layers/Parcels/FeatureServer/4
  Fields used: PARCEL_NO, USE_CODE, PROP_TYPE, PROP_SUBTY, LAND_VALUE, BLDG_VALUE,
  TOTAL_VALUE, TOTAL_ACRES, TAXABLE, EXEMPT_TY, STATUS_IND, OWNER_NAME, SITUS_*.
- City of Olympia municipal boundary (City of Olympia's own AGOL org):
  https://services.arcgis.com/a83cWFJpXhezKJzd/arcgis/rest/services/City_Limits/FeatureServer/0

Outputs:
- data/jurisidictions/data/olympia/olympia-wa-parcels.parquet
- data/jurisidictions/data/olympia/olympia-wa-parcels_YYYY_MM_DD.parquet

Notes:
- ~20-25k Olympia parcels -> small enough for the browser GeoParquet path (no PMTiles
  / H3 hexes), like Newport News / South Bend. Registered with usePmtiles omitted.
- Classification keys off PROP_TYPE (RES/CNU/MUL/APT/MOB/LND/PRK/OFF/RTL/... — much
  cleaner than the 2-digit USE_CODE). LND -> Vacant Land, PRK -> Parking.
- Exempt: drop truly public/institutional land (EXEMPT_TY in Government Property /
  DoR Institutional / Housing Authority / Tribal Lands, or PROP_TYPE='XMP', or
  TAXABLE='N'). KEEP Senior/Disabled, Historical, Home Improvement, Multi-Family
  Urban Housing — those are tax-relief programs on otherwise normal-valued parcels,
  not exempt land use, and dropping them would gut residential coverage.
- $/sqft denominator is TOTAL_ACRES (acres -> sqft), fallback geodesic polygon area.
- Condo units can stack on a shared footprint -> same-footprint collapse (SUM values
  + stated area) as in run_newportnews.py / run_tulsa.py.
- link: Thurston County Assessor property search (?parcel=<PARCEL_NO>).
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
from shapely.geometry import box
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))          # so `import data.scripts.*` resolves
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "olympia"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "olympia-wa-geometry.parquet"

PARCELS_URL = ("https://tconline.co.thurston.wa.us/server/rest/services/"
               "Common_Layers/Parcels/FeatureServer/4/query")
BOUNDARY_URL = ("https://services.arcgis.com/a83cWFJpXhezKJzd/arcgis/rest/services/"
                "City_Limits/FeatureServer/0/query")
OUT_FIELDS = ("PARCEL_NO,USE_CODE,PROP_TYPE,PROP_SUBTY,LAND_VALUE,BLDG_VALUE,TOTAL_VALUE,"
              "TOTAL_ACRES,TAXABLE,EXEMPT_TY,STATUS_IND,CURR_USE,OWNER_NAME,SITUS_STRE,SITUS_CITY")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
geod = Geod(ellps="WGS84")

# Exempt types that mark truly public / institutional land we should drop. Tax-relief
# programs (Senior/Disabled, Historical, Home Improvement, Multi Family Urban Housing,
# Less than $500) are intentionally NOT in this set — they sit on normal parcels.
EXEMPT_DROP = {
    "Government Property",
    "DoR Institutional",
    "Housing Authority",
    "Tribal Lands for Government Services",
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_boundary():
    """City of Olympia municipal boundary as a single (multi)polygon, EPSG:4326."""
    r = requests.get(BOUNDARY_URL, params={
        "where": "1=1", "outFields": "OBJECTID", "returnGeometry": "true",
        "outSR": 4326, "f": "geojson"}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    bnd = gpd.read_file(io.BytesIO(r.content))
    if bnd.crs is None:
        bnd = bnd.set_crs(4326)
    elif bnd.crs.to_epsg() != 4326:
        bnd = bnd.to_crs(4326)
    poly = unary_union([g.buffer(0) for g in bnd.geometry if g is not None])
    log(f"Olympia boundary: {len(bnd)} feature(s); bbox {[round(x,5) for x in bnd.total_bounds]}")
    return poly, list(bnd.total_bounds)


def fetch_parcels(bbox):
    """Pull Thurston parcels intersecting the Olympia bbox (envelope filter)."""
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    minx, miny, maxx, maxy = bbox
    spatial = {
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
    }
    total = requests.get(PARCELS_URL, params={**spatial, "where": "1=1",
                         "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} Thurston parcels in Olympia bbox (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(5):
            try:
                r = requests.get(PARCELS_URL, params={
                    **spatial, "where": "1=1", "outFields": OUT_FIELDS,
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
        if off % 10000 < PAGE:
            log(f"  fetched {off:,}/{total:,}")
        if len(gdf) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


boundary_poly, bbox = fetch_boundary()
geom = fetch_parcels(bbox)

# ── clip to City of Olympia (centroid within municipal boundary) ──────────────
if geom.crs is None:
    geom = geom.set_crs("EPSG:4326")
elif geom.crs.to_epsg() != 4326:
    geom = geom.to_crs("EPSG:4326")
geom["geometry"] = geom["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
geom = geom[geom["geometry"].notnull() & geom["geometry"].apply(
    lambda x: getattr(x, "is_valid", False) and not x.is_empty)].copy()
rep = geom.geometry.representative_point()
inside = rep.within(boundary_poly)
log(f"Centroid-in-Olympia clip: {int(inside.sum()):,} of {len(geom):,} bbox parcels")
parcel = geom[inside.values].copy()

# ── capture common-area land parcels (the real footprint units merge DOWN onto) ──────
# Condo / business-park units are placeholder stubs; the development's actual land is a
# separate polygon whose PARCEL_NO is the plat root — the units' leading digits (units
# 35340000800 -> common parcel 3534) — and which carries NO assessor attributes (value
# lives in the units). These have a null STATUS_IND so the active-records filter below drops
# them: capture them FIRST, dissolving multi-polygon plats into one geometry per plat.
_cpn = parcel["PARCEL_NO"].astype(str).str.strip()
_is_common = _cpn.str.fullmatch(r"\d{2,10}") & parcel["PROP_TYPE"].isna() & parcel["LAND_VALUE"].isna()
commons = parcel.loc[_is_common, ["geometry"]].copy()
commons["cpn"] = _cpn[_is_common].values
commons = commons[commons.geometry.notna() & ~commons.geometry.is_empty]
commons = commons.dissolve(by="cpn").reset_index() if len(commons) else commons
log(f"Common-area land parcels captured: {len(commons):,}")

# ── field cleanup ─────────────────────────────────────────────────────────────
parcel["acct"] = parcel["PARCEL_NO"].astype(str).str.strip()
parcel = parcel[parcel["acct"].ne("") & parcel["acct"].ne("None") & parcel["acct"].ne("nan")]
for c in ["LAND_VALUE", "BLDG_VALUE", "TOTAL_VALUE", "TOTAL_ACRES"]:
    parcel[c] = pd.to_numeric(parcel[c], errors="coerce")
for c in ["PROP_TYPE", "USE_CODE", "EXEMPT_TY", "TAXABLE", "STATUS_IND"]:
    parcel[c] = parcel[c].astype(str).str.strip()
# keep only active assessment records (drop ROW / non-assessed STATUS=None)
n0 = len(parcel)
parcel = parcel[parcel["STATUS_IND"].str.upper() == "A"].copy()
log(f"Active (STATUS_IND='A') parcels -> {len(parcel):,} (dropped {n0 - len(parcel):,})")

parcel = parcel.rename(columns={"LAND_VALUE": "land_val", "BLDG_VALUE": "bld_val"})
parcel["tot_appr_val"] = parcel["land_val"].fillna(0) + parcel["bld_val"].fillna(0)
parcel["stated_acres"] = parcel["TOTAL_ACRES"]

# ── dedup multi-polygon parcels (values first, geometry unioned) ──────────────
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
log(f"Rows sharing a PARCEL_NO (multi-polygon): {ndup:,}")
if ndup:
    first_cols = [c for c in parcel.columns if c not in ("geometry", "acct")]
    agg = {c: "first" for c in first_cols}
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
log(f"After PARCEL_NO dedup -> {len(parcel):,}")

# ── collapse unit "stub" placeholders onto their development land parcel ──────
# Thurston maps multi-unit properties (condos, business-park suites) as tiny placeholder unit
# stubs carrying per-unit value but little/no stated land -> value/tiny-footprint = $hundreds/
# sqft -> thin, very tall 3D slivers. The development's REAL land is a separate common-area
# parcel (captured above) whose PARCEL_NO is the units' plat-root prefix (units 35340000800 ->
# common 3534). MERGE each development's stubs DOWN onto that common parcel: sum land + building
# value, use the common parcel's true footprint -> correct $/sqft, no synthesized geometry.
# ~99% of stubs match a common parcel by prefix; the few that don't fall back to a building-
# snapped hull. A stub is flagged by EFFECTIVE land area (stated acreage if present, else geom)
# so the rule catches every placeholder size (100/168/224/400 sqft) and both zero-share and
# small-share units, while leaving real houses on small lots alone. No-land stubs (mobile homes /
# personal property on leased PARK land, land_val==0) own no land and are dropped.
STUB_MAX_SQFT = 1000.0   # placeholder footprints are ~100-400 sqft; real parcels are >2,000
STUB_LAND_SQFT = 3000.0  # effective land (stated acreage if present, else geom) below which a
                         # building must be a multi-unit share, not a real lot. Catches condo
                         # units whose stated share is 0 OR a small fraction (e.g. 0.02 ac = 871
                         # sqft, the Conger Ave condos) while excluding real houses on small lots
                         # (RES tiny-geom placeholders carry a real ~0.29 ac stated area).
MIN_WIDTH_M = 16.0  # footprint short-side floor: keeps a collinear stub row from becoming a thin wall
_proj = parcel.to_crs(32610)
parcel["_area_sqft"] = (_proj.geometry.area * 10.7639).values

# Pre-match every parcel to a development common-area land parcel by longest PARCEL_NO prefix
# (units of a plat share its number; the common parcel IS that plat root). Used to flag units
# and, in the merge, to place them on the real land.
_cpns = sorted((str(p) for p in commons["cpn"]), key=len, reverse=True) if len(commons) else []
def _match_common(acct):
    for c in _cpns:
        if acct.startswith(c):
            return c
    return None
parcel["_dev"] = parcel["acct"].astype(str).apply(_match_common)

def _stub_mask(df):
    """A placeholder UNIT: tiny footprint + a building, AND either it belongs to a known
    development plat (matched a common-area parcel) OR its effective land area (stated acreage
    if present, else geom) is tiny. The dev-membership test catches units whose stated share
    happens to exceed the size cap (e.g. a 0.07 ac Conger Ave share); the eff-land test catches
    developments with no common-area parcel. Real houses on small lots match neither."""
    a = df["_area_sqft"]
    stated_sqft = pd.to_numeric(df["stated_acres"], errors="coerce").fillna(0) * 43560.0
    eff_land = stated_sqft.where(stated_sqft > 0, a)
    bld = pd.to_numeric(df["bld_val"], errors="coerce").fillna(0)
    # A tiny parcel that belongs to a known development plat is a unit even with no building
    # (e.g. a vacant unit share, PROP_TYPE='LND'); merge it too. Parcels with no plat match need
    # a building + tiny effective land (developments that lack a common-area parcel).
    return (a < STUB_MAX_SQFT) & (df["_dev"].notna() | ((bld > 0) & (eff_land < STUB_LAND_SQFT)))

is_stub = _stub_mask(parcel)
_landval = pd.to_numeric(parcel["land_val"], errors="coerce").fillna(0)
n_drop = int((is_stub & (_landval <= 0)).sum())
parcel = parcel[~(is_stub & (_landval <= 0))].copy()
log(f"Dropped {n_drop:,} no-land unit stubs (mobile homes / personal property on leased land)")

is_stub = _stub_mask(parcel)
_landval = pd.to_numeric(parcel["land_val"], errors="coerce").fillna(0)
merge_mask = is_stub & (_landval > 0)
n_merge = int(merge_mask.sum())
if n_merge:
    stubs = parcel[merge_mask].to_crs(32610).reset_index(drop=True)
    rest = parcel[~merge_mask].copy()

    # cgeom: development common-area land geometry by plat number (stubs already carry _dev,
    # the matched plat, from the pre-match above). The assessor's common-area polygon is the
    # land with the individual unit footprints punched out (interior rings) — for a land parcel
    # we want the SOLID outer boundary, so fill those holes.
    from shapely.geometry import Polygon, MultiPolygon
    def _fill_holes(g):
        if g is None or g.is_empty:
            return g
        if g.geom_type == "Polygon":
            return Polygon(g.exterior)
        if g.geom_type == "MultiPolygon":
            return MultiPolygon([Polygon(p.exterior) for p in g.geoms])
        return g
    cgeom = {}
    if len(commons):
        commons_p = commons.to_crs(32610)
        cgeom = {str(p): _fill_holes(g) for p, g in zip(commons_p["cpn"], commons_p.geometry)
                 if g is not None and not g.is_empty}
    n_matched = int(stubs["_dev"].notna().sum())

    # Overture buildings only needed for the (rare) stubs with no common-area land parcel:
    # give those a building-snapped convex hull, widened to MIN_WIDTH_M so a collinear row of
    # stubs is not a thin wall. Skipped entirely when every stub matched a common parcel.
    bld = None
    if bool((stubs["_dev"].isna()).any()):
        try:
            from data.scripts.classify_parking_surface import fetch_overture_buildings
            _b = fetch_overture_buildings(tuple(parcel.to_crs(4326).total_bounds)).to_crs(32610)
            bld = _b[_b.geometry.notna() & ~_b.geometry.is_empty][["geometry"]].reset_index(drop=True)
            log(f"  Overture buildings for {int(stubs['_dev'].isna().sum())} unmatched stub(s): {len(bld):,}")
        except Exception as e:  # noqa: BLE001
            log(f"  Overture buildings unavailable ({type(e).__name__}); unmatched stubs use plain hull")

    def shortside(geom):
        xs, ys = geom.minimum_rotated_rectangle.exterior.coords.xy
        p = list(zip(xs, ys))
        return min(((p[i][0]-p[i+1][0])**2 + (p[i][1]-p[i+1][1])**2) ** 0.5 for i in range(4))

    def hull_footprint(stub_geoms):
        u = unary_union(list(stub_geoms))
        if bld is not None:
            hit = [i for i in bld.sindex.query(u, predicate="intersects")
                   if bld.geometry.iloc[i].intersects(u)]
            if hit:
                u = unary_union([u, unary_union(bld.geometry.iloc[hit].values)])
        foot = u.convex_hull
        try:
            sw = shortside(foot)
            if sw < MIN_WIDTH_M:
                foot = foot.buffer((MIN_WIDTH_M - sw) / 2.0)
        except Exception:  # noqa: BLE001 - degenerate (point/line) hull
            foot = foot.buffer(MIN_WIDTH_M / 2.0)
        return foot

    # group matched stubs by development (common parcel); unmatched by plat-root prefix
    stubs["_grp"] = stubs["_dev"].where(stubs["_dev"].notna(),
                                        "P:" + stubs["acct"].astype(str).str[:4])
    sum_cols = ["land_val", "bld_val", "tot_appr_val"]
    rows = []
    for _, grp in stubs.groupby("_grp"):
        row = grp.iloc[0].to_dict()
        dev = grp["_dev"].iloc[0]
        if dev is not None and dev in cgeom:
            row["geometry"] = cgeom[dev]                      # real development land parcel
        else:
            row["geometry"] = hull_footprint(grp.geometry.values)  # fallback
        for c in sum_cols:
            row[c] = float(pd.to_numeric(grp[c], errors="coerce").fillna(0).sum())
        row["stated_acres"] = 0.0  # use the (common-area / hull) geometry as the land denominator
        rows.append(row)
    merged = gpd.GeoDataFrame(rows, geometry="geometry", crs=32610)
    merged["geometry"] = merged.geometry.apply(lambda g: g if (g is not None and g.is_valid) else g.buffer(0))
    merged = merged.drop(columns=["_dev", "_grp"], errors="ignore").to_crs(4326)
    parcel = gpd.GeoDataFrame(pd.concat([rest, merged], ignore_index=True),
                              geometry="geometry", crs="EPSG:4326")
    log(f"Merged {n_merge:,} unit stubs -> {len(rows):,} development parcels "
        f"({n_matched:,} onto common-area land, {n_merge - n_matched:,} via building-snap hull)")
parcel["_collapsed"] = 0
parcel = parcel.drop(columns=["_area_sqft", "_dev"], errors="ignore")
log(f"After stub collapse -> {len(parcel):,}")

# ── exemption flag + classification ──────────────────────────────────────────
parcel["exemption_flag"] = (
    (parcel["PROP_TYPE"].str.upper() == "XMP")
    | (parcel["TAXABLE"].str.upper() == "N")
    | (parcel["EXEMPT_TY"].isin(EXEMPT_DROP))
).astype(int)


def categorize(pt, use):
    pt = str(pt or "").strip().upper()
    use = str(use or "").strip()
    if pt == "PRK":
        # PROP_TYPE='PRK' is a mobile-home / manufactured-home PARK (the leased land that
        # ~1,560 MOB personal-property units sit on), NOT a surface parking lot.
        return "Mobile Home"
    if pt == "LND":
        return "Vacant Land"
    if pt == "RES":
        return "Single Family"
    if pt == "CNU":
        return "Condominium"
    if pt in ("MUL", "APT"):
        return "Multifamily"
    if pt == "MOB":
        return "Mobile Home"
    if pt in ("OFF", "RTL", "RST", "SRV", "MED", "TRN", "BTH", "COM", "HOS"):
        return "Commercial"
    if pt in ("IND", "WHS"):
        return "Industrial"
    return "Other"


parcel["PROPERTY_CATEGORY"] = [categorize(pt, u) for pt, u in
                               zip(parcel["PROP_TYPE"], parcel["USE_CODE"])]

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


# ── canonical fields — TOTAL_ACRES denominator, geodesic fallback ────────────
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
rep_sqft = pd.to_numeric(ex.get("stated_acres", np.nan), errors="coerce") * SQFT_PER_ACRE
ex["reported_sqft"] = rep_sqft
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

# Collapsed condo stacks: shared complex footprint as land denominator (not one unit's
# stated area). Larger of summed per-unit stated area and union polygon geodesic area.
col = ex["_collapsed"] == 1
ex.loc[col, "reported_sqft"] = np.maximum(
    pd.to_numeric(ex.loc[col, "reported_sqft"], errors="coerce").fillna(0.0),
    pd.to_numeric(ex.loc[col, "geom_area_sqft"], errors="coerce").fillna(0.0))
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

# Thurston County Assessor parcel detail page. Must be basic.asp?pn=<parcel_no> —
# front.asp?parcel= just lands on the search form (the "takes you to the general page" bug).
ex["link"] = ("https://tcproperty.co.thurston.wa.us/propsql/basic.asp?pn="
              + ex["acct"].astype(str))

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
out = DATA_DIR / "olympia-wa-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"olympia-wa-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")
log(f"land_value_per_sqft: p50=${final['land_value_per_sqft'].median():.0f} "
    f"p99=${final['land_value_per_sqft'].quantile(.99):.0f} "
    f"max=${final['land_value_per_sqft'].max():.0f}")
log("DONE")
