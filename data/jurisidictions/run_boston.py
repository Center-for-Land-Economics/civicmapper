#!/usr/bin/env python3
"""
Build the Greater Boston canonical parcel parquet: Boston + Cambridge + Somerville + Brookline.

ONE CivicMapper city (`boston`, `boston-ma-parcels.*`) stitched from four Massachusetts
municipalities (DMV / Hartford-metro pattern). Each parcel carries a `jurisdiction` column
driving the region toggle (already in H3_CATEGORICAL_FIELDS, so the bake emits
metadata.groups.jurisdiction + per-region land totals).

Massachusetts assesses at 100% of full and fair cash value (M.G.L. c.59 s.38), so every figure
here is full market value — no assessment-ratio fix is needed (contrast Georgia's 40%).

SOURCES (public, no token; verified 2026-09-28)
- Boston (assessed 1/1/2025, FY2026 bills) — City of Boston, Analyze Boston:
    values   "Property Assessment FY2026" CSV (184,552 records, one per PID)
             https://data.boston.gov/dataset/property-assessment
    geometry Parcels26 (98,451 polygons, MassGIS L3 schema, MAP_PAR_ID == GIS_ID)
             https://gisportal.boston.gov/arcgis/rest/services/Parcels/Parcels26/MapServer/0
  Joined on GIS_ID -> MAP_PAR_ID: 99.998% of records match (4 unmatched, $3.6M).
  NOT MassGIS for Boston: the statewide layer still carries Boston's FY2023 roll.
- Cambridge (FY2026), Somerville (FY2025), Brookline (FY2027) — MassGIS standardized
  "Level 3" Property Tax Parcels (L3_TAXPAR_POLY_ASSESS: polygons + assessor records):
    https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/Massachusetts_Property_Tax_Parcels/FeatureServer/0
  TOWN_ID 49 / 274 / 46. One row per assessor record; condo units are STACKED on their
  building-lot polygon. FY is each town's latest certified roll as loaded by MassGIS
  (Somerville lags one year).
Owner/mailing fields are never requested from MassGIS and are dropped from the Boston CSV at
read time (issue #12).

TRAPS HANDLED
1. CONDO LAND IS $0 IN ALL FOUR TOWNS (Massachusetts practice). Condo units carry their whole
   value as building; the condo "main"/master record (Boston LU=CM, Cambridge CONDO-BLDG) holds
   the lot area at $0. Boston alone: 74k residential units ($65.2B) + commercial/parking condo
   units sit on ~11k lots with ZERO assessor land. Units are summed onto their lot (Boston:
   group by GIS_ID, which every unit shares with its CM main; MassGIS: group by the stacked
   LOC_ID polygon). Left at $0 the densest housing in the region (Back Bay, South End, Beacon
   Hill, most of Somerville and Brookline) reads as land worth nothing. So by default the lot's
   land is ESTIMATED (--no-condo-impute reproduces the Providence/Hartford "$0 + flagged"
   treatment):
       est  = lot area x median land $/sqft of the 15 nearest non-condo taxable parcels in the
              SAME municipality (assessor-valued land, >=500 sqft, not a remnant)
       land = min(est, 70% of the lot's total assessed value); improvement = total - land
   Every such parcel carries `condo_land_imputed = 1` and keeps the assessor's own figure in
   `assessor_land_value`. Imputation only runs where unsplit ($0-land) units are >=50% of the
   lot's value AND no exempt record on the same lot carries land value — a taxable building
   on EXEMPT land (e.g. Massport / BPDA ground leases) is genuinely $0 land to the taxpayer and
   keeps the assessor's $0. Imputed parcels are never labelled "Underdeveloped" (their land
   share is our estimate, not the assessor's).
2. EXEMPT: Boston LU E (fully exempt) and EA (Ch.121A urban-redevelopment properties, which pay
   an excise in lieu of property tax — GROSS_TAX = 0); MassGIS towns: DOR use class 9xx
   (Harvard, MIT, Tufts, BU, hospitals, government, churches). Dropped PER RECORD before the
   lot roll-up, so a condo building with one exempt unit keeps its taxable units.
3. MA use codes are town-extended: Cambridge 1014/1021/0112, Somerville 1040/112C/920V. The DOR
   class is the first three digits, except a 4-char code with a leading 0 (0112 -> 112).
   3-char codes starting 0 (013, 031) are mixed use.
4. MassGIS PROP_IDs occasionally span several LOC_ID polygons (a parcel in two pieces): those
   polygons are joined into one footprint (union-find) and each PROP_ID's value is counted ONCE
   (skill §2 — never multiply an account by its polygon count).
5. Utility classes (42x, 43x telecom/cell) excluded per the add-city skill.

Outputs:
- data/jurisidictions/data/boston/boston-ma-parcels.parquet  (+ dated snapshot)

Usage:
    python data/jurisidictions/run_boston.py                 # build (uses raw/ caches)
    python data/jurisidictions/run_boston.py --no-condo-impute
    python data/jurisidictions/run_boston.py --fetch-only
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import (  # noqa: E402
    add_improvement_ratio_fields, classify_property_refined, gis_area_sqft)

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "boston"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh) CivicMapper-ETL", "Accept": "application/json"}
SQFT_PER_ACRE = 43560.0
MA_EPSG = 26986  # NAD83 / Massachusetts Mainland (metres)

BOSTON_ASSESS_CSV_URL = (
    "https://data.boston.gov/dataset/e02c44d2-3c64-459c-8fe2-e1ce5f38a035/resource/"
    "ee73430d-96c0-423e-ad21-c4cfb54c8961/download/fy2026-property-assessment-data_rev.csv")
BOSTON_PARCELS_URL = "https://gisportal.boston.gov/arcgis/rest/services/Parcels/Parcels26/MapServer/0/query"
MASSGIS_URL = ("https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/"
               "Massachusetts_Property_Tax_Parcels/FeatureServer/0/query")
MASSGIS_FIELDS = ("OBJECTID,MAP_PAR_ID,LOC_ID,POLY_TYPE,TOWN_ID,PROP_ID,BLDG_VAL,LAND_VAL,OTHER_VAL,"
                  "TOTAL_VAL,FY,LOT_SIZE,LOT_UNITS,USE_CODE,USE_DESC,SITE_ADDR,CITY,ZONING,UNITS,BLD_AREA")
BOSTON_KEEP = ["PID", "CM_ID", "GIS_ID", "ST_NUM", "ST_NUM2", "ST_NAME", "UNIT_NUM", "CITY", "ZIP_CODE",
               "LUC", "LU", "LU_DESC", "LAND_SF", "GROSS_AREA", "LAND_VALUE", "BLDG_VALUE",
               "SFYI_VALUE", "TOTAL_VALUE", "GROSS_TAX", "RES_UNITS", "COM_UNITS", "YR_BUILT"]
MASSGIS_TOWNS = {49: "Cambridge", 274: "Somerville", 46: "Brookline"}
EXPECTED_FY = {"Boston": 2026, "Cambridge": 2026, "Somerville": 2025, "Brookline": 2027}

# Per-parcel deep link only exists for Boston (properties.boston.gov, a hash-routed React app;
# route verified in github.com/CityOfBoston/boston-property-lookup frontend/src/App.tsx). The
# other towns get their assessor's lookup landing page (Providence/Richmond precedent).
LINKS = {
    "Cambridge": "https://www.cambridgema.gov/propertydatabase",
    "Somerville": "https://www.somervillema.gov/departments/finance/assessing",
    "Brookline": "https://www.brooklinema.gov/159/Assessors-Office",
}

CONDO_K = 15             # nearest donor parcels for the condo land estimate
CONDO_CAP = 0.70         # imputed land never exceeds this share of the lot's total value
UNSPLIT_MIN_SHARE = 0.5  # lot is "a condo lot" when $0-land units are >= this share of value


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# Fetch (cached to raw/)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_geojson(query_url: str, where: str, out_fields: str, page: int = 2000,
                  order_by: str = "OBJECTID") -> gpd.GeoDataFrame:
    total = requests.get(query_url, params={"where": where, "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=120).json().get("count")
    log(f"  pulling {total:,} features where {where!r}")
    pages, off = [], 0
    while True:
        gdf = None
        for attempt in range(6):
            try:
                r = requests.get(query_url, params={
                    "where": where, "outFields": out_fields, "returnGeometry": "true",
                    "resultOffset": off, "resultRecordCount": page, "outSR": 4326,
                    "orderByFields": order_by, "f": "geojson"}, headers=HEADERS, timeout=300)
                r.raise_for_status()
                # ArcGIS answers overload with HTTP 200 + an {"error": ...} body (Provo trap).
                if r.content[:1] != b"{" or b'"error"' in r.content[:200]:
                    raise RuntimeError(f"non-GeoJSON body: {r.content[:160]!r}")
                gdf = gpd.read_file(io.BytesIO(r.content))
                break
            except Exception as e:  # noqa: BLE001
                log(f"    retry {attempt + 1} @off {off}: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(4 * (attempt + 1))
        if gdf is None:
            raise RuntimeError(f"page failed at offset {off}: {query_url}")
        if not len(gdf):
            break
        pages.append(gdf)
        off += len(gdf)
        if total is not None and off >= total:
            break
    out = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    if total is not None and len(out) != total:
        raise RuntimeError(f"pulled {len(out):,} of {total:,} — truncated pull")
    return out


def load_boston_assessment() -> pd.DataFrame:
    cache = RAW_DIR / "boston-assessment-fy2026.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    csv = RAW_DIR / "fy2026-property-assessment.csv"
    if not csv.exists():
        log("Downloading Boston FY2026 property assessment CSV ...")
        r = requests.get(BOSTON_ASSESS_CSV_URL, headers=HEADERS, timeout=900, allow_redirects=True)
        r.raise_for_status()
        csv.write_bytes(r.content)
    df = pd.read_csv(csv, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[BOSTON_KEEP].copy()          # OWNER / MAIL_* never leave this function
    df.to_parquet(cache, index=False)
    return df


def load_boston_parcels() -> gpd.GeoDataFrame:
    cache = RAW_DIR / "boston-parcels26.parquet"
    if cache.exists():
        return gpd.read_parquet(cache)
    log("Pulling Boston Parcels26 geometry ...")
    g = fetch_geojson(BOSTON_PARCELS_URL, "1=1", "OBJECTID,MAP_PAR_ID,LOC_ID,POLY_TYPE,TOWN_ID")
    g.to_parquet(cache, index=False)
    return g


def load_massgis_town(town_id: int) -> gpd.GeoDataFrame:
    cache = RAW_DIR / f"massgis-town{town_id}.parquet"
    if cache.exists():
        return gpd.read_parquet(cache)
    log(f"Pulling MassGIS L3 town {town_id} ({MASSGIS_TOWNS[town_id]}) ...")
    g = fetch_geojson(MASSGIS_URL, f"TOWN_ID={town_id}", MASSGIS_FIELDS)
    g.to_parquet(cache, index=False)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Massachusetts DOR use-code classifier (shared by all four towns)
# ─────────────────────────────────────────────────────────────────────────────
COMMERCIAL_CONDO_WORDS = ("RETAIL", "RTL", "OFFICE", "OFC", "OFF ", "COMM", "(COM)", "INDUST", "IND ", "PKG-SP")
BOSTON_OVERRIDES = {"439": "Commercial", "444": "Industrial", "465": "Commercial",
                    "435": "Commercial", "436": "Commercial", "995": "Condominium"}


def dor_base(code: str) -> str:
    c = (code or "").strip().upper()
    return c[1:4] if (len(c) >= 4 and c[0] == "0") else c[:3]


def categorize(code: str, desc: str, jur: str) -> str:
    c = (code or "").strip().upper()
    d = (desc or "").upper()
    if not c:
        return "Other"
    base = dor_base(c)
    if jur == "Boston" and base in BOSTON_OVERRIDES:
        return BOSTON_OVERRIDES[base]
    if base[:1] == "9":
        return "Exempt"
    if "CONDO" in d or "CNDO" in d or base == "102":
        if base[:1] in ("0", "1") and not any(w in d for w in COMMERCIAL_CONDO_WORDS):
            return "Condominium"
        return "Commercial Condominium"
    if len(c) == 3 and c[0] == "0":
        return "Mixed Use"
    if not base[:3].isdigit():
        return "Other"
    b = int(base)
    if b == 101:
        return "Single Family"
    if b in (104, 105):
        return "Two & Three Family"
    if 111 <= b <= 114 or b == 120 or 125 <= b <= 127:
        return "Apartments (4+ units)"
    if b == 116:
        return "Parking Garage"
    if b == 119:
        return "Parking Lot"
    if 130 <= b <= 132:
        return "Vacant Land"
    if 100 <= b < 200:
        return "Other Residential"
    if 200 <= b < 300:
        return "Open Space"
    if 300 <= b <= 302:
        return "Hotel"
    if 320 <= b <= 329:
        return "Retail"
    if 330 <= b <= 339 and "PARK" in d:
        return "Parking Lot" if "LOT" in d else "Parking Garage"
    if b == 336 or b == 338:
        return "Parking Garage"
    if b in (337, 387):
        return "Parking Lot"
    if 340 <= b <= 349:
        return "Office"
    if 390 <= b <= 393:
        return "Vacant Land"
    if 300 <= b < 400:
        return "Commercial"
    if 420 <= b <= 439:
        return "Utility"
    if 440 <= b <= 442:
        return "Vacant Land"
    if 400 <= b < 500:
        return "Industrial"
    if 500 <= b < 600:
        return "Personal Property"
    if 600 <= b < 900:
        return "Open Space"
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# Adapters -> (records DataFrame, footprints GeoDataFrame)
#   records: jurisdiction, fkey, land, bldg, total, use_code, use_desc, exempt, is_unit,
#            is_main, stated_sqft, address, fy
#   footprints: fkey, geometry
# ─────────────────────────────────────────────────────────────────────────────
def boston_adapter() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    a = load_boston_assessment()
    g = load_boston_parcels()
    for c in ["LAND_VALUE", "BLDG_VALUE", "TOTAL_VALUE", "LAND_SF", "GROSS_TAX"]:
        a[c] = num(a[c])
    log(f"Boston: {len(a):,} assessment records ({a['PID'].nunique():,} PIDs), "
        f"{len(g):,} Parcels26 polygons")
    if a["PID"].duplicated().any():
        raise SystemExit("Boston: duplicate PIDs in the assessment file — dedup rule needed")
    st = (a["ST_NUM"].fillna("").str.strip() + " " + a["ST_NAME"].fillna("").str.strip()).str.strip()
    lu = a["LU"].fillna("").str.strip()
    rec = pd.DataFrame({
        "jurisdiction": "Boston",
        "rec_id": a["PID"],
        "fkey": "B" + a["GIS_ID"].astype(str),
        "gis_id": a["GIS_ID"].astype(str),
        "land": a["LAND_VALUE"].fillna(0.0),
        "bldg": a["BLDG_VALUE"].fillna(0.0),
        "total": a["TOTAL_VALUE"].fillna(0.0),
        "use_code": a["LUC"].fillna("").str.strip(),
        "use_desc": a["LU_DESC"].fillna("").str.strip(),
        "exempt": lu.isin(["E", "EA"]),
        "is_unit": lu.isin(["CD", "CC", "CP"]),
        "is_main": (a["PID"] == a["GIS_ID"]),
        "condo_marker": lu.isin(["CD", "CC", "CP", "CM"]),
        "stated_sqft": a["LAND_SF"],
        "address": st,
        "fy": 2026,
    })
    # A handful of Boston records carry BLDG+LAND != TOTAL (SFYI "yard items" sit in TOTAL);
    # improvement = total - land keeps land + improvement == total.
    rec["bldg"] = (rec["total"] - rec["land"]).clip(lower=0)
    geo = g[["MAP_PAR_ID", "geometry"]].copy()
    geo["fkey"] = "B" + geo["MAP_PAR_ID"].astype(str)
    geo = geo.dissolve(by="fkey", as_index=False)[["fkey", "geometry"]]
    return rec, gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326")


def _union_find_footprints(g: pd.DataFrame) -> pd.Series:
    """LOC_IDs linked by a shared PROP_ID collapse to one footprint key."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, grp in g.dropna(subset=["PROP_ID"]).groupby("PROP_ID")["LOC_ID"]:
        locs = grp.unique()
        for other in locs[1:]:
            ra, rb = find(locs[0]), find(other)
            if ra != rb:
                parent[rb] = ra
    return g["LOC_ID"].map(find)


def massgis_adapter(town_id: int) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    jur = MASSGIS_TOWNS[town_id]
    g = load_massgis_town(town_id)
    g = g[g["POLY_TYPE"].isin(["FEE", "TAX"])].copy()
    for c in ["LAND_VAL", "BLDG_VAL", "OTHER_VAL", "TOTAL_VAL", "LOT_SIZE"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    fy = g["FY"].dropna().astype(int).value_counts()
    log(f"{jur}: {len(g):,} MassGIS records on {g['LOC_ID'].nunique():,} polygons; FY {fy.to_dict()}")
    if fy.index[0] != EXPECTED_FY[jur]:
        raise SystemExit(f"{jur}: expected FY{EXPECTED_FY[jur]}, MassGIS has FY{fy.index[0]} — "
                         "a newer roll landed; update EXPECTED_FY and the notes")
    g["root"] = _union_find_footprints(g)
    n_multi = int((g.groupby("root")["LOC_ID"].nunique() > 1).sum())
    log(f"  footprints after joining PROP_IDs split across polygons: {g['root'].nunique():,} "
        f"({n_multi} multi-polygon)")
    # Each assessor record counted once: PROP_ID repeats across a split parcel's polygons carry
    # the SAME account value (skill §2 — first, never sum). Null PROP_ID rows stand alone.
    g["rec_id"] = g["PROP_ID"].fillna("OID" + g["OBJECTID"].astype(str))
    recs = g.sort_values("OBJECTID").drop_duplicates(["root", "rec_id"])
    code = recs["USE_CODE"].fillna("").str.strip()
    lot_sqft = recs["LOT_SIZE"] * SQFT_PER_ACRE  # LOT_UNITS is 'Acres' for all three towns
    addr = recs["SITE_ADDR"].fillna("").str.split("#").str[0].str.strip()
    land, total = recs["LAND_VAL"].fillna(0.0), recs["TOTAL_VAL"].fillna(0.0)
    unsplit = (land <= 0) & (total > 0)
    desc = recs["USE_DESC"].fillna("").str.upper()
    rec = pd.DataFrame({
        "jurisdiction": jur,
        "rec_id": recs["rec_id"],
        "fkey": jur[:1] + recs["root"].astype(str),
        "gis_id": recs["MAP_PAR_ID"].astype(str),
        "land": land,
        "bldg": (total - land).clip(lower=0),   # BLDG_VAL + OTHER_VAL
        "total": total,
        "use_code": code,
        "use_desc": recs["USE_DESC"].fillna("").str.strip(),
        "exempt": code.map(dor_base).str[:1].eq("9"),
        "is_unit": unsplit & (desc.str.contains("CONDO|CNDO") | code.map(dor_base).eq("102")),
        "is_main": ~unsplit & (lot_sqft > 0),
        # Condo regime marker: any unit OR the $0 master/common record (Cambridge CONDO-BLDG,
        # Brookline 'Non-Taxable Condominium Common Land').
        "condo_marker": desc.str.contains("CONDO|CNDO") | code.map(dor_base).eq("102"),
        "stated_sqft": lot_sqft.where(lot_sqft > 0),
        "address": addr,
        "fy": recs["FY"],
    })
    polys = g.drop_duplicates("LOC_ID")[["root", "geometry"]].copy()
    polys["fkey"] = jur[:1] + polys["root"].astype(str)
    geo = polys.dissolve(by="fkey", as_index=False)[["fkey", "geometry"]]
    return rec, gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326")


# ─────────────────────────────────────────────────────────────────────────────
# Roll records up to footprints
# ─────────────────────────────────────────────────────────────────────────────
def polygonal(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    parts = [p for p in getattr(geom, "geoms", []) if isinstance(p, (Polygon, MultiPolygon))]
    if not parts:
        return None
    u = unary_union(parts)
    return u if isinstance(u, (Polygon, MultiPolygon)) else None


def roll_up(rec: pd.DataFrame, geo: gpd.GeoDataFrame, stats: dict) -> gpd.GeoDataFrame:
    jur = rec["jurisdiction"].iloc[0]
    rec = rec.copy()
    rec["category"] = [categorize(c, d, jur) for c, d in zip(rec["use_code"], rec["use_desc"])]
    ex = rec[rec["exempt"]]
    s = stats.setdefault(jur, {})
    s["exempt_records"] = int(len(ex))
    s["exempt_land"] = float(ex["land"].sum())
    s["exempt_total"] = float(ex["total"].sum())
    exempt_land_fp = ex.groupby("fkey")["land"].sum()

    tx = rec[~rec["exempt"]]
    drop_cat = tx["category"].isin(["Utility", "Personal Property"])
    s["utility_records"] = int(drop_cat.sum())
    s["utility_total"] = float(tx.loc[drop_cat, "total"].sum())
    tx = tx[~drop_cat & (tx["total"] > 0)].copy()
    s["taxable_records"] = int(len(tx))
    s["taxable_land_assessor"] = float(tx["land"].sum())
    s["taxable_total"] = float(tx["total"].sum())
    tx["unsplit_val"] = np.where((tx["land"] <= 0) & (tx["total"] > 0), tx["total"], 0.0)
    tx["unit_n"] = tx["is_unit"].astype(int)

    gb = tx.groupby("fkey")
    fp = pd.DataFrame({
        "land": gb["land"].sum(), "bldg": gb["bldg"].sum(), "total": gb["total"].sum(),
        "unsplit_val": gb["unsplit_val"].sum(), "n_accounts": gb.size(),
        "n_condo_units": gb["unit_n"].sum(), "fy": gb["fy"].max(),
    })
    # Dominant category = the one carrying the most value on the lot (a condo building with a
    # ground-floor retail unit stays Condominium).
    dom = (tx.groupby(["fkey", "category"])["total"].sum().reset_index()
           .sort_values("total", ascending=False).drop_duplicates("fkey").set_index("fkey")["category"])
    fp["category"] = dom
    lead = tx.sort_values("total", ascending=False).drop_duplicates("fkey").set_index("fkey")
    fp["use_code"], fp["use_desc"] = lead["use_code"], lead["use_desc"]
    # Address + stated lot area come from the lot's main record (Boston: PID == GIS_ID, incl. the
    # $0 condo main; MassGIS: a record with a real LOT_SIZE), else the most valuable record.
    allrec = rec[~rec["exempt"]]
    main = allrec[allrec["is_main"]].sort_values("stated_sqft", ascending=False).drop_duplicates("fkey").set_index("fkey")
    fp["address"] = main["address"].reindex(fp.index).fillna(lead["address"])
    fp["stated_sqft"] = main["stated_sqft"].reindex(fp.index)
    fp["gis_id"] = main["gis_id"].reindex(fp.index).fillna(lead["gis_id"])
    fp["exempt_land_on_lot"] = exempt_land_fp.reindex(fp.index).fillna(0.0)
    fp["condo_regime"] = allrec.groupby("fkey")["condo_marker"].any().reindex(fp.index).fillna(False)
    fp["jurisdiction"] = jur
    fp = fp.reset_index()

    exempt_only = geo[geo["fkey"].isin(set(ex["fkey"]) - set(fp.index))]
    s["exempt_only_lots"] = int(len(exempt_only))
    s["exempt_only_acres"] = float(exempt_only.geometry.apply(gis_area_sqft).sum() / SQFT_PER_ACRE)
    out = geo.merge(fp, on="fkey", how="inner")
    missing = fp[~fp["fkey"].isin(geo["fkey"])]
    s["unmapped_lots"] = int(len(missing))
    s["unmapped_total"] = float(missing["total"].sum())
    out["geometry"] = out["geometry"].apply(polygonal)
    bad = out["geometry"].isna()
    s["bad_geometry"] = int(bad.sum())
    out = out[~bad].copy()
    log(f"{jur}: {len(out):,} taxable lots mapped; unmapped {len(missing):,} "
        f"(${missing['total'].sum() / 1e6:,.1f}M); exempt records {len(ex):,} "
        f"(${ex['land'].sum() / 1e9:,.2f}B land); utility dropped {s['utility_records']}")
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


# ─────────────────────────────────────────────────────────────────────────────
# Condo land estimate
# ─────────────────────────────────────────────────────────────────────────────
def impute_condo_land(ex: gpd.GeoDataFrame, stats: dict, enabled: bool) -> gpd.GeoDataFrame:
    from scipy.spatial import cKDTree

    ex["assessor_land_value"] = ex["land"]
    ex["condo_land_imputed"] = 0
    share = np.divide(ex["unsplit_val"], ex["total"], out=np.zeros(len(ex)), where=ex["total"] > 0)
    # A condominium lot: the lot is a condo regime (has a unit or a condo master record) AND its
    # $0-land records carry most of its value. A plain commercial building assessed at $0 land
    # with no condo regime (air rights, ground lease) is NOT estimated — it keeps the assessor's $0.
    condo_lot = ex["condo_regime"].astype(bool) & (share >= UNSPLIT_MIN_SHARE)
    leasehold = condo_lot & (ex["exempt_land_on_lot"] > 0)
    eligible = condo_lot & ~leasehold
    pts = ex.to_crs(MA_EPSG).geometry.representative_point()
    xy = np.column_stack([pts.x.to_numpy(), pts.y.to_numpy()])
    psf = ex["land"] / ex["land_area_sqft"]
    donor = (ex["land"] > 0) & (share < 0.1) & (ex["land_area_sqft"] >= 500)
    est_psf = pd.Series(np.nan, index=ex.index)
    for jur in ex["jurisdiction"].unique():
        dmask = donor & (ex["jurisdiction"] == jur)
        tmask = eligible & (ex["jurisdiction"] == jur)
        s = stats[jur]
        s["condo_lots"] = int((condo_lot & (ex["jurisdiction"] == jur)).sum())
        s["condo_lots_leasehold_kept_zero"] = int((leasehold & (ex["jurisdiction"] == jur)).sum())
        if not tmask.any():
            continue
        tree = cKDTree(xy[dmask.to_numpy()])
        _, idx = tree.query(xy[tmask.to_numpy()], k=CONDO_K)
        dpsf = psf[dmask].to_numpy()
        est_psf.loc[tmask] = np.median(dpsf[idx], axis=1)
    est = est_psf * ex["land_area_sqft"]
    new_land = np.minimum(est, CONDO_CAP * ex["total"])
    apply = eligible & new_land.notna() & (new_land > ex["land"])
    ex["condo_land_estimate_psf"] = est_psf.where(eligible)
    for jur in ex["jurisdiction"].unique():
        m = apply & (ex["jurisdiction"] == jur)
        s = stats[jur]
        s["condo_lots_imputed"] = int(m.sum())
        s["condo_land_imputed_added"] = float((new_land[m] - ex.loc[m, "land"]).sum())
        s["condo_cap_binding"] = int((m & (est > CONDO_CAP * ex["total"])).sum())
    new_land = new_land.round(0)  # whole dollars, like every assessor figure
    if enabled:
        ex.loc[apply, "land"] = new_land[apply]
        ex.loc[apply, "bldg"] = ex.loc[apply, "total"] - ex.loc[apply, "land"]
        ex.loc[apply, "condo_land_imputed"] = 1
    log(f"Condo lots: {int(condo_lot.sum()):,}; estimate {'APPLIED' if enabled else 'computed, NOT applied'} "
        f"to {int(apply.sum()):,}; kept $0 (taxable building on exempt land): {int(leasehold.sum()):,}")
    return ex


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--no-condo-impute", action="store_true",
                    help="Ship the assessor's $0 condo land (Providence/Hartford treatment).")
    args = ap.parse_args()

    stats: dict = {}
    b_rec, b_geo = boston_adapter()
    town = {t: massgis_adapter(t) for t in MASSGIS_TOWNS}
    if args.fetch_only:
        return
    parts = [roll_up(b_rec, b_geo, stats)] + [roll_up(r, g, stats) for r, g in town.values()]
    ex = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    log(f"Combined: {len(ex):,} taxable lots")

    # ── area: stated lot area when it agrees with the polygon (0.5-2.0x), else geodesic ──
    ex["geom_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
    ex.loc[ex["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
    rep = pd.to_numeric(ex["stated_sqft"], errors="coerce")
    ratio = rep / ex["geom_area_sqft"]
    use_rep = (rep > 0) & ratio.between(0.5, 2.0)
    ex["land_area_sqft"] = np.where(use_rep, rep, ex["geom_area_sqft"])
    ex["area_source"] = np.where(use_rep, "assessor", "gis")
    log(f"Area: assessor={int(use_rep.sum()):,} gis={int((~use_rep).sum()):,}")
    ex = ex[ex["land_area_sqft"] > 0].copy()
    ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

    ex = impute_condo_land(ex, stats, enabled=not args.no_condo_impute)

    # ── canonical fields ──
    ex["property_land_use_category"] = ex["category"]
    ex["current_full_land_value"] = ex["land"]
    ex["improvement_value"] = ex["bldg"]
    ex["full_market_value"] = ex["total"]
    ex["exemption_flag"] = 0
    refined = classify_property_refined(
        ex, sf_cutoff=0.67, other_cutoff=0.50,
        exclude_categories=("Other", "Open Space"),
        category_col="property_land_use_category",
        land_col="current_full_land_value", improvement_col="improvement_value",
        fetch_footprints=False)
    refined[(ex["condo_land_imputed"] == 1) & (refined == "Underdeveloped")] = None
    ex["property_land_use_refined"] = refined
    den = ex["land_area_sqft"]
    ex["land_area_acres"] = den / SQFT_PER_ACRE
    ex["land_value_per_sqft"] = ex["current_full_land_value"] / den
    ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
    ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
    ex = add_improvement_ratio_fields(ex, land_col="current_full_land_value", improvement_col="improvement_value")
    is_b = ex["jurisdiction"].eq("Boston")
    page = np.where(ex["n_condo_units"] > 0, "master-parcel", "details")
    ex["link"] = np.where(is_b, "https://properties.boston.gov/#/" + pd.Series(page, index=ex.index)
                          + "?parcelId=" + ex["gis_id"].astype(str),
                          ex["jurisdiction"].map(LINKS))
    ex["parcel_id"] = ex["gis_id"]
    # String, so the popup shows "2026" rather than a thousands-separated number.
    ex["fiscal_year"] = "FY" + ex["fy"].astype("Int64").astype(str)

    # ── smoke alarms (skill §6a) ──
    gp = ex.to_crs(MA_EPSG)
    a = gp.geometry.area * 10.7639
    lv = ex["land_value_per_sqft"]
    log(f"footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
    log(f"sub-500 / sub-1000 sqft valued: {int(((a < 500) & (ex['full_market_value'] > 0)).sum())} / "
        f"{int(((a < 1000) & (ex['full_market_value'] > 0)).sum())}")
    rp = gp.geometry.representative_point()
    vc = (rp.x.round(1).astype(str) + "," + rp.y.round(1).astype(str)).value_counts()
    log(f"stacked clusters: {int((vc > 1).sum())}; max stack {int(vc.max())}")
    holes = ex.geometry.apply(lambda g: sum(len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
    log(f"parcels with holes: {int((holes > 0).sum())}")
    for jur, d in ex.groupby("jurisdiction"):
        l = d["land_value_per_sqft"]
        imp = d[d["condo_land_imputed"] == 1]["land_value_per_sqft"]
        don = d[(d["condo_land_imputed"] == 0) & (d["current_full_land_value"] > 0)]["land_value_per_sqft"]
        log(f"  {jur:10s} land $/sqft p50/p99/max {l.median():.0f}/{l.quantile(.99):.0f}/{l.max():.0f} | "
            f"assessor-land lots p50 {don.median():.0f} | imputed condo lots p50 "
            f"{(imp.median() if len(imp) else float('nan')):.0f} (n={len(imp):,}) | "
            f"$0-land lots {int((d['current_full_land_value'] <= 0).sum()):,}")

    # ── export ──
    COLUMNS = ["geometry", "jurisdiction", "parcel_id", "address", "fiscal_year", "use_code", "use_desc",
               "exemption_flag", "property_land_use_category", "property_land_use_refined",
               "full_market_value", "full_market_value_per_sqft",
               "current_full_land_value", "land_value_per_sqft",
               "improvement_value", "improvement_value_per_sqft",
               "assessor_land_value", "condo_land_imputed", "n_accounts", "n_condo_units",
               "TLLDIMPROV", "IMPR_LAND_RATIO", "IMPR_LAND_PCT", "IMPR_PCT_TOTAL",
               "link", "land_area_acres", "area_source", "likely_remnant"]
    final = gpd.GeoDataFrame(ex[COLUMNS].reset_index(drop=True), geometry="geometry", crs="EPSG:4326")
    for c in ["n_accounts", "n_condo_units", "condo_land_imputed", "likely_remnant", "exemption_flag"]:
        final[c] = final[c].astype("int32")
    out = DATA_DIR / "boston-ma-parcels.parquet"
    final.to_parquet(out, index=False)
    final.to_parquet(DATA_DIR / f"boston-ma-parcels_{datetime.now():%Y_%m_%d}.parquet", index=False)
    log(f"SAVED {out} | rows {len(final):,}")

    log("category: " + str(final["property_land_use_category"].value_counts().to_dict()))
    log("refined: " + str(final["property_land_use_refined"].value_counts(dropna=False).to_dict()))
    print("\n=== PER-JURISDICTION SUMMARY ===")
    for jur, d in final.groupby("jurisdiction"):
        s = stats[jur]
        ex_share = s["exempt_land"] / (s["exempt_land"] + s["taxable_land_assessor"])
        print(f"{jur}: {d['fiscal_year'].max()}  lots {len(d):,}  "
              f"land ${d['current_full_land_value'].sum() / 1e9:,.2f}B "
              f"(assessor ${d['assessor_land_value'].sum() / 1e9:,.2f}B + condo est "
              f"${(d['current_full_land_value'] - d['assessor_land_value']).sum() / 1e9:,.2f}B)  "
              f"total ${d['full_market_value'].sum() / 1e9:,.2f}B  acres {d['land_area_acres'].sum():,.0f}")
        print(f"    exempt: {s['exempt_records']:,} records, land ${s['exempt_land'] / 1e9:,.2f}B "
              f"(total ${s['exempt_total'] / 1e9:,.2f}B) = {ex_share:.1%} of assessor land value; "
              f"condo lots {s.get('condo_lots', 0):,}, imputed {s.get('condo_lots_imputed', 0):,} "
              f"(+${s.get('condo_land_imputed_added', 0) / 1e9:,.2f}B, cap binding {s.get('condo_cap_binding', 0)}), "
              f"leasehold kept $0 {s.get('condo_lots_leasehold_kept_zero', 0)}; unmapped {s['unmapped_lots']} "
              f"(${s['unmapped_total'] / 1e6:,.1f}M); utility dropped {s['utility_records']}")
        print(f"    fully-exempt lots: {s['exempt_only_lots']:,} = {s['exempt_only_acres']:,.0f} acres vs "
              f"{d['land_area_acres'].sum():,.0f} taxable acres "
              f"({s['exempt_only_acres'] / (s['exempt_only_acres'] + d['land_area_acres'].sum()):.1%} of parcel acreage); "
              f"exempt share of land value incl. condo estimate: "
              f"{s['exempt_land'] / (s['exempt_land'] + d['current_full_land_value'].sum()):.1%}")
    print(f"TOTAL: {len(final):,} lots, land ${final['current_full_land_value'].sum() / 1e9:,.2f}B, "
          f"market ${final['full_market_value'].sum() / 1e9:,.2f}B")


if __name__ == "__main__":
    main()
