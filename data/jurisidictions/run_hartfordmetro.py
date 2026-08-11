#!/usr/bin/env python3
"""
Build the Hartford (CT) metro parcel parquet — the contiguous urban core, all from
ONE source so the land-value surface is seamless across town lines.

Connecticut has no county governments; the "City of Hartford" is a tiny ~18 sq mi
core inside a ~1M-person metro of independent towns. For a land-value tool the
town boundary is an administrative artifact — land value is economically continuous
across it — so this city shows the whole urban core and tags each parcel by town
(Houston-style jurisdiction treatment: light up/dim towns via the dropdown).

Source (one layer, all towns, identical schema):
  Connecticut CAMA & Parcel Layer 2024 (CT GIS Office)
  https://services3.arcgis.com/3FL1kr7L4LvwA2Kb/arcgis/rest/services/Connecticut_CAMA_and_Parcel_Layer_2024/FeatureServer/0
  Filtered to the 12 contiguous core towns (Town_Name IN CORE_TOWNS).

Why VALUE-ONLY (no land-use categories): the statewide layer's per-parcel land-use
comes as town-specific CAMA/zoning codes (West Hartford "R-10", Manchester "RA",
Glastonbury "RR"…) with NO shared codebook and NO populated description — so there is
no way to classify use across towns. (The standalone `hartford` city gets real
categories only because the CITY of Hartford publishes its own decoded layer; the
other towns don't.) So this metro build ships the land/building VALUE split +
jurisdiction, and category filters / the Underused tab are disabled in the frontend.

Staggered revaluation: CT revalues every 5 years on offset schedules, so values are in
mixed base years (Hartford 2021, West Hartford 2022, East Hartford 2023, most others
2024). Shown as-is; `valuation_year` ships per parcel and is disclosed in the popup.

Condos: CT does not assess condo land per unit, so condo units come through the state
layer with Appraised_Land=0 (often building=0 too). They're kept but flag as erroneous
in land modes (REALLANDVA<=0), same as the standalone Hartford city.

Exempt: no use codes here, so exemption is an OWNER-keyword heuristic (government /
institutional / religious / utility). Imperfect but excludes the obvious distorting
government & institutional parcels.

Outputs:
- data/jurisidictions/data/hartfordmetro/hartfordmetro-ct-parcels.parquet
- data/jurisidictions/data/hartfordmetro/hartfordmetro-jurisdiction-overlay.geojson
- dated snapshot
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
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields  # noqa: E402
from pyproj import Geod  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "hartfordmetro"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "hartfordmetro-ct-raw.parquet"

STATE_URL = ("https://services3.arcgis.com/3FL1kr7L4LvwA2Kb/arcgis/rest/services/"
             "Connecticut_CAMA_and_Parcel_Layer_2024/FeatureServer/0/query")
CORE_TOWNS = ["Hartford", "West Hartford", "East Hartford", "Newington", "Wethersfield",
              "Bloomfield", "Windsor", "Manchester", "South Windsor", "Rocky Hill",
              "Glastonbury", "Farmington"]
FIELDS = ("Town_Name,Parcel_ID,Location_1,Owner,Appraised_Land,Appraised_Building,"
          "Appraised_Outbuilding,Valuation_Year,Land_Acres")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
geod = Geod(ellps="WGS84")

# Owner-keyword exemption (no use codes available metro-wide). Conservative: clear
# government / institutional / religious / utility owners only.
EXEMPT_KW = [
    "CITY OF", "TOWN OF", "STATE OF CONN", "STATE OF CT", " CONNECTICUT ", "UNITED STATES",
    "U S A", "U.S.A", "FEDERAL", "HOUSING AUTHORITY", "BOARD OF ED", "BOARD OF EDUCATION",
    "METROPOLITAN DISTRICT", "REDEVELOPMENT", "HOUSING AUTH", "PUBLIC SCHOOL",
    "CHURCH", "PARISH", "ARCHDIOCESE", "DIOCESE", "CONGREGATION", "TEMPLE", "SYNAGOGUE",
    "MOSQUE", "MINISTR", "ROMAN CATHOLIC", "ST ", "SAINT ", "CEMETERY", "UNIVERSITY",
    "COLLEGE", "ACADEMY", "HOSPITAL", "YMCA", "YWCA", "HABITAT FOR", "RED CROSS",
    "SALVATION ARMY", "GOODWILL", "COMMUNITY CHEST", "EVERSOURCE", "UNITED ILLUMINAT",
    "CONN LIGHT", "WATER POLLUTION", "WATER AUTHORITY", "TRANSIT", "HOUSING & DEV",
    "COUNCIL OF", "FOUNDATION", "TRUST FOR PUBLIC",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_towns():
    if GEOM_CACHE.exists():
        log(f"Using cache: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    town_list = ",".join(f"'{t}'" for t in CORE_TOWNS)
    where = f"Town_Name IN ({town_list})"
    total = requests.get(STATE_URL, params={"where": where, "returnCountOnly": "true",
                         "f": "json"}, headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} parcels across {len(CORE_TOWNS)} towns (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(6):
            try:
                r = requests.get(STATE_URL, params={
                    "where": where, "outFields": FIELDS, "returnGeometry": "true",
                    "resultOffset": off, "resultRecordCount": PAGE, "outSR": 4326,
                    "orderByFields": "OBJECTID", "f": "geojson"},
                    headers=HEADERS, timeout=240)
                r.raise_for_status()
                gdf = gpd.read_file(io.BytesIO(r.content))
                break
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt+1} @off {off}: {type(e).__name__}: {e}")
                time.sleep(5 * (attempt + 1))
        if gdf is None:
            raise RuntimeError(f"Fetch failed at offset {off}")
        if not len(gdf):
            break
        pages.append(gdf)
        off += len(gdf)
        if off % 20000 < PAGE:
            log(f"  fetched {off:,}/{total:,}")
        if len(gdf) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


raw = fetch_towns()
if raw.crs is None:
    raw = raw.set_crs(4326)
elif raw.crs.to_epsg() != 4326:
    raw = raw.to_crs(4326)
raw["geometry"] = raw["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
raw = raw[raw.geometry.notnull() & ~raw.geometry.is_empty].copy()
log(f"Raw parcels: {len(raw):,}; per-town: {raw['Town_Name'].value_counts().to_dict()}")

for c in ["Appraised_Land", "Appraised_Building", "Appraised_Outbuilding"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

# ── dedup: account-broadcast collapse, then footprint collapse ────────────────
# The CT state layer splits ONE account across many GIS polygons and BROADCASTS the
# account's full value onto each (verified: 97% of multi-polygon same-address groups
# carry IDENTICAL land+building). Left alone this both N×-inflates value (skill §2)
# and renders each broadcast polygon as its own tall $/sqft tower (and over-sums the
# H3 hexes). Parcel_ID/link_1 are unusable keys here (blank ' ' / null outside
# Hartford), so key the account on (Town, address, land, building, outbuilding):
# identical value at one address = one account — distinct real parcels essentially
# never share all of address+land+bldg (the 3% of same-address groups with DIFFERING
# values stay separate, which is correct: they're genuine per-unit assessments).
# Blank addresses (~1%, all value-0 ROW/water, dropped later) get a unique key so they
# never merge. Union footprints; take each value ONCE (never sum).
raw["_addr"] = raw["Location_1"].fillna("").astype(str).str.strip().str.upper()
blank = raw["_addr"].str.fullmatch(r"0?\s*") | raw["_addr"].isin(["0", "UNKNOWN"])
raw["_acct"] = (raw["Town_Name"].astype(str) + "|" + raw["_addr"] + "|"
                + raw["Appraised_Land"].astype(str) + "|" + raw["Appraised_Building"].astype(str)
                + "|" + raw["Appraised_Outbuilding"].astype(str))
raw.loc[blank, "_acct"] = "BLANK|" + raw.index.to_series()[blank].astype(str)
log(f"Rows sharing an account (broadcast split): {int(raw['_acct'].duplicated(keep=False).sum()):,}")
first_cols = ["Town_Name", "Parcel_ID", "Location_1", "Owner", "Valuation_Year",
              "Appraised_Land", "Appraised_Building", "Appraised_Outbuilding"]


def _collapse(gdf, key):
    g = gdf.groupby(key, dropna=False)
    out = g.agg({c: "first" for c in first_cols if c in gdf.columns})
    out["geometry"] = g["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]))
    return gpd.GeoDataFrame(out.reset_index(drop=True), geometry="geometry", crs="EPSG:4326")


acct = _collapse(raw, "_acct")
log(f"After broadcast collapse -> {len(acct):,}")
# Second pass: any remaining exact-footprint stacks (condo units sharing one polygon
# with DIFFERING values) — collapse by representative point so overlapping polygons
# don't double-render / double-count in the hexes. Value first (land=0 for CT condos).
rp = acct.geometry.representative_point()
acct["_rp"] = rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)
parcel = _collapse(acct, "_rp")
log(f"After footprint collapse -> {len(parcel):,}")

# ── value model ───────────────────────────────────────────────────────────────
parcel["land_value"] = parcel["Appraised_Land"]
parcel["improvement_value"] = parcel["Appraised_Building"] + parcel["Appraised_Outbuilding"]
parcel["full_market_value"] = parcel["land_value"] + parcel["improvement_value"]

# ── exemption (owner-keyword heuristic) ───────────────────────────────────────
own = parcel["Owner"].fillna("").astype(str).str.upper()
pat = "|".join(k.replace(" ", r"\s") for k in EXEMPT_KW)
parcel["exemption_flag"] = own.str.contains(pat, regex=True, na=False).astype(int)
log(f"Exempt (owner heuristic): {int(parcel['exemption_flag'].sum()):,} "
    f"({100*parcel['exemption_flag'].mean():.1f}%)")
ex = parcel[parcel["exemption_flag"] == 0].copy()
# drop parcels with no value at all (bare ROW / water slivers with 0/0)
noval = (ex["full_market_value"] <= 0)
log(f"Zero-value parcels dropped: {int(noval.sum()):,}")
ex = ex[~noval | (ex["land_value"] > 0)].copy()  # keep land>0 even if total edge-case

# ── jurisdiction tag + reval year ─────────────────────────────────────────────
ex["jurisdiction"] = ex["Town_Name"]
ex["valuation_year"] = pd.to_numeric(ex["Valuation_Year"], errors="coerce").astype("Int64")

# ── canonical fields — geodesic area denominator ──────────────────────────────
def gis_area_sqft(g):
    if g is None or g.is_empty:
        return np.nan
    if g.geom_type == "Polygon":
        lon, lat = g.exterior.coords.xy
        a, _ = geod.polygon_area_perimeter(lon, lat)
        return abs(a) * 10.763910416709722
    if g.geom_type == "MultiPolygon":
        return sum(gis_area_sqft(p) for p in g.geoms)
    return np.nan


log("Computing GIS areas...")
ex["land_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["land_area_sqft"] < 1, "land_area_sqft"] = np.nan
ex["area_source"] = "gis"
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

# No metro-wide land-use classification available (see header) -> single placeholder
# category so the frontend's category machinery has a valid field; categories/Underused
# are disabled in the city config.
ex["property_land_use_category"] = "Unclassified"
ex["property_land_use_refined"] = None
ex["link"] = ""

# ── export ────────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "jurisdiction", "valuation_year", "exemption_flag",
           "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant"]
for c in COLUMNS:
    if c not in ex.columns:
        ex[c] = np.nan
final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final["geometry"] = final["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")
final = final[final.geometry.notnull() & ~final.geometry.is_empty].copy()
out = DATA_DIR / "hartfordmetro-ct-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"hartfordmetro-ct-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,}")

# ── town-boundary overlay (dissolve parcels by town) ──────────────────────────
log("Building jurisdiction overlay (dissolve by town)...")
diss = final.dissolve(by="jurisdiction").reset_index()[["jurisdiction", "geometry"]]
diss = diss.rename(columns={"jurisdiction": "name"})
diss["geometry"] = diss["geometry"].simplify(0.00015)
ov = DATA_DIR / "hartfordmetro-jurisdiction-overlay.geojson"
diss.to_file(ov, driver="GeoJSON")
log(f"SAVED {ov} ({len(diss)} towns)")

log(f"per-town parcels: {final['jurisdiction'].value_counts().to_dict()}")
log(f"reval years: {final['valuation_year'].value_counts(dropna=False).to_dict()}")
log(f"land_value<=0 (condos/erroneous): {int((final['current_full_land_value']<=0).sum()):,}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")
lv = final.loc[final["current_full_land_value"] > 0, "land_value_per_sqft"]
log(f"land_value_per_sqft (land>0): p50=${lv.median():.1f} p99=${lv.quantile(.99):.0f} max=${lv.max():.0f}")
log("DONE")
