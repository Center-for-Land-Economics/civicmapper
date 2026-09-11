#!/usr/bin/env python3
"""
Build the City of Duluth, GA canonical parcel parquet.

Duluth is a SUBURBAN CITY IN GWINNETT COUNTY, GA (metro Atlanta, ~10.3 sq mi, pop ~31k).
Not Duluth MN (the big one on Lake Superior) and not Duluth, Minnesota's twin Superior WI.
The city key is `duluth` / state `ga` — if a Duluth MN is ever added it must take a
different key (e.g. `duluthmn`), because this one claimed the plain slug first.

Source — Gwinnett County GIS public AGOL org `RfpmnkSAQleRbndX` (public, no token). The
`Property_and_Tax` FeatureServer is a ONE-STOP source, but the values are NOT on the parcel
layer: geometry is layer 0 and the dollar values live in RELATED TABLES on the same service.
  - /0  Parcels             geometry + PIN + LRSN + EXEMPTION_TYPE + PARCELTYPE (309,658 countywide)
  - /3  Tax Master Table    LANDVAL1 / DWLGVAL1 / TOTVAL1 / PROPCLAS / PCDESC   (314,997 rows)
  - /10 Land Value Table    SQFT / ACRES (stated land area)                      (323,355 rows)
  - /8  Property Improvements Table — NOT used (building size is derived client-side, see
        the add-city skill §9: popup land/building size = value / value-per-sqft).
Join key is LRSN (integer, present on the parcel layer and both tables), with a PIN FALLBACK
that is not optional: the parcel layer leaves LRSN NULL on 665 real, valued Duluth parcels
(1,209 acres / $64.2M of land value / 223 houses + 370 townhomes). The Tax Master's `PIN`
matches the GIS `PIN` exactly and recovers 664 of them. Use `PIN`, never `RPIN` — and note
that tables /8 and /10 space-pad their PIN ("R1001 001     ") while /3 and the parcel layer
do not, which is why the fallback goes through /3 only.

KNOWN GAP — CONDOMINIUMS ARE NOT MAPPED BY GWINNETT:
  Condo units (PROPCLAS 106/355/356/357) exist in the Tax Master but have NO parcel polygon
  anywhere in the county — verified: 0 of 1,402 Duluth-addressed condo records are mapped.
  Their land is the development's `122`/`322` Condo Common Area polygon, which the assessor
  values at $0. So Duluth ships 26 common-area parcels (95.8 ac) at $0 land value, and the
  condo value itself is absent. Land-lot-weighted estimate of the in-city shortfall:
  ~$19.6M land (~1.6% of city land value) and ~$219M of improvements. There is no reliable
  unit->common-area key (PIN plat roots cover only 1 of the 26; addresses match 85 of 1,402),
  and LOCCITY='DULUTH' spans far more unincorporated area than the city, so inventing a merge
  would import out-of-city value. Documented rather than guessed. Revisit if Gwinnett ever
  publishes condo footprints or a unit->common-area relationship.

City boundary: `City_Area` FeatureServer/0 on the same org, CITY_NAME='DULUTH' (one polygon,
3 rings). Parcels are clipped CENTROID-WITHIN that polygon — do NOT filter on the Tax Master
`LOCCITY` field, which is the situs/mailing city and includes unincorporated "Duluth GA 30096"
addresses well outside the city limits (the playbook §4 trap).

GEORGIA 40% ASSESSMENT RATIO — the one number that will bite you:
  Georgia taxes at 40% of fair market value. In the Tax Master, LANDVAL1 / DWLGVAL1 / TOTVAL1
  are 100% FAIR MARKET VALUE and TAXTOT1 is the 40% assessed value (verified: TAXTOT1 ==
  0.40 * TOTVAL1 across the roll). CivicMapper ships MARKET value, so this uses LANDVAL1 /
  DWLGVAL1 / TOTVAL1 and ignores TAXTOT1. Using TAXTOT1 would silently under-report the city
  by 60%.

All value fields in the tables are STRINGS (space-padded) — every one needs to_numeric.

Outputs:
- data/jurisidictions/data/duluth/duluth-ga-parcels.parquet
- data/jurisidictions/data/duluth/duluth-ga-parcels_YYYY_MM_DD.parquet

Notes:
- ~10k parcels — well under the ~100k PMTiles rule of thumb, but baked to PMTiles + H3 anyway:
  small GeoParquet cities render SPARSE when zoomed out (MapLibre culls sub-pixel polygons and
  there is no aggregate layer to cover it). That is the Olympia 2026-07-01 lesson; baking H3
  from the start avoids shipping the bug and re-baking later. Bake with --drop-remnants.
- No assessor parking class: Gwinnett's PROPCLAS scheme has no surface-parking code, so
  parking for this city is OSM-only (there is nothing in the roll to cross-check it against).
"""
from __future__ import annotations

import io
import json
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from datetime import datetime
from pathlib import Path
from shapely.ops import unary_union
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "duluth"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "duluth-ga-geometry.parquet"
BND_CACHE = DATA_DIR / "duluth-ga-boundary.parquet"
TM_CACHE = DATA_DIR / "duluth-ga-taxmaster.parquet"
LV_CACHE = DATA_DIR / "duluth-ga-landvalue.parquet"
TM_PIN_CACHE = DATA_DIR / "duluth-ga-taxmaster-by-pin.parquet"

ORG = "https://services3.arcgis.com/RfpmnkSAQleRbndX/arcgis/rest/services"
PT = f"{ORG}/Property_and_Tax/FeatureServer"
CITY_AREA = f"{ORG}/City_Area/FeatureServer/0/query"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
geod = Geod(ellps="WGS84")

# Gwinnett's public property record search is qPublic (Schneider), AppID 1282 / LayerID 43872.
# The per-parcel KeyValue deep link could not be verified from this environment (qPublic
# answers 403 to non-browser clients), so — like Richmond — every parcel points at the search
# landing page rather than a guessed deep link that might 404 for users.
SEARCH_URL = ("https://qpublic.schneidercorp.com/Application.aspx"
              "?AppID=1282&LayerID=43872&PageTypeID=2&PageID=16058")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── fetch: city boundary ─────────────────────────────────────────────────────
def fetch_boundary():
    if BND_CACHE.exists():
        log(f"Using cached boundary: {BND_CACHE.name}")
        return gpd.read_parquet(BND_CACHE)
    r = requests.get(CITY_AREA, params={"where": "CITY_NAME='DULUTH'", "outFields": "CITY_NAME",
                     "returnGeometry": "true", "outSR": 4326, "f": "geojson"},
                     headers=HEADERS, timeout=120)
    r.raise_for_status()
    bd = gpd.read_file(io.BytesIO(r.content))
    if not len(bd):
        raise RuntimeError("City_Area returned no DULUTH polygon")
    bd.to_parquet(BND_CACHE, index=False)
    log(f"  cached boundary -> {BND_CACHE.name}")
    return bd


# ── fetch: parcel geometry within the Duluth envelope ────────────────────────
def fetch_parcels(bd):
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    xmin, ymin, xmax, ymax = bd.total_bounds
    env = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
           "spatialReference": {"wkid": 4326}}
    log("Pulling Gwinnett parcels intersecting the Duluth envelope...")
    pages, off = [], 0
    while True:
        g = None
        for attempt in range(5):
            try:
                r = requests.get(f"{PT}/0/query", params={
                    "where": "1=1", "geometry": json.dumps(env),
                    "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                    "spatialRel": "esriSpatialRelIntersects",
                    "outFields": ("PIN,TAXPIN,LRSN,ADDRESS,PARCELTYPE,EXEMPTION_TYPE,"
                                  "DEEDEDACREAGE,CALCULATEDACREAGE"),
                    "returnGeometry": "true", "resultOffset": off, "resultRecordCount": PAGE,
                    "outSR": 4326, "orderByFields": "OBJECTID", "f": "geojson",
                }, headers=HEADERS, timeout=240)
                r.raise_for_status()
                g = gpd.read_file(io.BytesIO(r.content))
                break
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt+1} @off {off}: {type(e).__name__}: {e}")
                time.sleep(5 * (attempt + 1))
        if g is None:
            raise RuntimeError(f"Parcel pull failed at offset {off}")
        if not len(g):
            break
        pages.append(g)
        off += len(g)
        # NOTE: do NOT break on a short page before checking it is really the last one —
        # the maxRecordCount pagination cap bit Seattle exactly this way.
        if len(g) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


# ── fetch: related tables, restricted to the LRSNs we actually mapped ────────
def _post_query(layer, where, fields):
    """POST one /query and return its attribute dicts, retrying transient failures."""
    for attempt in range(4):
        try:
            r = requests.post(f"{PT}/{layer}/query", data={
                "where": where, "outFields": fields, "returnGeometry": "false", "f": "json",
            }, headers=HEADERS, timeout=240)
            r.raise_for_status()
            js = r.json()          # raises if the server fell back to an HTML error page
            if "error" in js:
                raise RuntimeError(f"ArcGIS error: {js['error']}")
            return [f["attributes"] for f in js.get("features", [])]
        except Exception as e:  # noqa: BLE001
            log(f"  retry {attempt+1}: {type(e).__name__}: {e}")
            time.sleep(4 * (attempt + 1))
    return None


def fetch_taxmaster_by_pin(pins, fields):
    """Tax Master rows looked up by PIN, for parcels whose GIS LRSN is null.

    The parcel layer leaves LRSN NULL on a meaningful minority of real, valued parcels
    (665 inside Duluth — 1,209 acres, $64.2M of land value, incl. 223 single-family homes
    and 370 townhomes). Joining on LRSN alone silently drops every one of them, which is
    ~5% of the city's land value and looks like nothing more than a slightly low parcel
    count. The Tax Master's own PIN matches the GIS PIN exactly (use PIN, NOT the padded
    RPIN), and recovers 664/665. Never ship this city on an LRSN-only join.
    """
    log(f"Looking up {len(pins):,} null-LRSN parcels by PIN...")
    out = []
    for i in range(0, len(pins), 100):
        chunk = pins[i:i + 100]
        where = "PIN IN (" + ",".join("'" + p.replace("'", "''") + "'" for p in chunk) + ")"
        got = _post_query(3, where, fields)
        if got is None:
            raise RuntimeError(f"PIN lookup failed at chunk {i}")
        out += got
    df = pd.DataFrame(out)
    log(f"  recovered {len(df):,} tax-master rows by PIN")
    return df


def fetch_table(layer, fields, lrsns, cache):
    if cache.exists():
        log(f"Using cached table: {cache.name}")
        return pd.read_parquet(cache)
    log(f"Pulling table /{layer} for {len(lrsns):,} LRSNs...")
    out = []
    # CHUNK SIZE AND POST BOTH MATTER. A `LRSN IN (...)` list of 400 ids overflows the GET
    # URL length limit and the server answers with an HTML error page, not JSON — which a
    # naive retry loop will spin on forever instead of failing. POST keeps the where-clause
    # out of the URL; CHUNK stays modest so one bad chunk is cheap to retry.
    CHUNK = 150
    for i in range(0, len(lrsns), CHUNK):
        chunk = lrsns[i:i + CHUNK]
        where = f"LRSN IN ({','.join(map(str, chunk))})"
        got = None
        for attempt in range(4):
            try:
                r = requests.post(f"{PT}/{layer}/query", data={
                    "where": where, "outFields": fields, "returnGeometry": "false", "f": "json",
                }, headers=HEADERS, timeout=240)
                r.raise_for_status()
                js = r.json()          # raises if the server fell back to an HTML error page
                if "error" in js:
                    raise RuntimeError(f"ArcGIS error: {js['error']}")
                got = [f["attributes"] for f in js.get("features", [])]
                break
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt+1} @chunk {i}: {type(e).__name__}: {e}")
                time.sleep(4 * (attempt + 1))
        if got is None:
            raise RuntimeError(f"Table /{layer} pull failed at chunk {i}")
        out += got
        if (i // CHUNK) % 10 == 0 or i + CHUNK >= len(lrsns):
            log(f"  {min(i + CHUNK, len(lrsns)):,}/{len(lrsns):,}")
    df = pd.DataFrame(out)
    df.to_parquet(cache, index=False)
    log(f"  cached -> {cache.name} ({len(df):,} rows)")
    return df


bd = fetch_boundary()
geom = fetch_parcels(bd)

# ── clip to the city limits (centroid-within the authoritative boundary) ─────
poly = bd.to_crs("EPSG:4326").geometry.union_all()
geom = geom[geom.geometry.notnull() & ~geom.geometry.is_empty].copy()
geom["geometry"] = geom["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
inside = geom.geometry.representative_point().within(poly)
parcel = geom[inside].copy()
log(f"Parcels in envelope {len(geom):,} -> centroid-within Duluth {len(parcel):,}")

# Rights-of-way carry no assessment and are not real estate holdings; drop them up front.
n_row = int(parcel["PARCELTYPE"].astype(str).str.strip().eq("R/W").sum())
parcel = parcel[parcel["PARCELTYPE"].astype(str).str.strip().ne("R/W")].copy()
log(f"Dropped {n_row:,} R/W (right-of-way) parcels -> {len(parcel):,}")

TM_FIELDS = ("LRSN,PIN,LOCADDR,LOCCITY,OWNER1,LEGALAC,ZONING,ZONEDESC,"
             "DWLGVAL1,LANDVAL1,TOTVAL1,TAXTOT1,PROPCLAS,PCDESC")
parcel["LRSN"] = pd.to_numeric(parcel["LRSN"], errors="coerce")
parcel["PIN"] = parcel["PIN"].astype(str).str.strip()

# Backfill the null GIS LRSNs from the Tax Master via PIN (see fetch_taxmaster_by_pin).
missing = parcel["LRSN"].isna() & parcel["PIN"].ne("") & ~parcel["PIN"].isin(["R/W", "nan", "None"])
if missing.any():
    pins = sorted(set(parcel.loc[missing, "PIN"]))
    if TM_PIN_CACHE.exists():
        log(f"Using cached PIN lookup: {TM_PIN_CACHE.name}")
        by_pin = pd.read_parquet(TM_PIN_CACHE)
    else:
        by_pin = fetch_taxmaster_by_pin(pins, TM_FIELDS)
        by_pin.to_parquet(TM_PIN_CACHE, index=False)
    by_pin["PIN"] = by_pin["PIN"].astype(str).str.strip()
    pin2lrsn = (by_pin.dropna(subset=["LRSN"]).groupby("PIN")["LRSN"].first())
    filled = parcel.loc[missing, "PIN"].map(pin2lrsn)
    parcel.loc[missing, "LRSN"] = pd.to_numeric(filled, errors="coerce").values
    log(f"Recovered LRSN by PIN for {int(filled.notna().sum()):,} of {int(missing.sum()):,} "
        "null-LRSN parcels")

n_nolrsn = int(parcel["LRSN"].isna().sum())
parcel = parcel[parcel["LRSN"].notna()].copy()
parcel["LRSN"] = parcel["LRSN"].astype(np.int64)
log(f"Dropped {n_nolrsn:,} parcels still with no tax record -> {len(parcel):,}")

lrsns = sorted(set(int(x) for x in parcel["LRSN"].unique()))
tm = fetch_table(3, TM_FIELDS, lrsns, TM_CACHE)
lv = fetch_table(10, "LRSN,PIN,SQFT,ACRES,APPUPDD,NUMIMP,NUMDWLG", lrsns, LV_CACHE)

# ── join values onto geometry (LRSN) ─────────────────────────────────────────
# Both tables can carry >1 row per LRSN (appraisal history / multiple land segments).
# Values are ACCOUNT-LEVEL and identical across an LRSN's rows, so take `first` — summing
# here is the Dallas N-times-inflation bug (add-city skill §2). Land segments are the one
# exception: stated SQFT is per-segment and must SUM to the account's land area.
tm["LRSN"] = pd.to_numeric(tm["LRSN"], errors="coerce").astype("Int64")
lv["LRSN"] = pd.to_numeric(lv["LRSN"], errors="coerce").astype("Int64")
for c in ["DWLGVAL1", "LANDVAL1", "TOTVAL1", "TAXTOT1", "LEGALAC"]:
    tm[c] = pd.to_numeric(tm[c].astype(str).str.strip(), errors="coerce")
for c in ["PROPCLAS", "PCDESC", "OWNER1", "LOCADDR", "LOCCITY", "ZONING", "ZONEDESC"]:
    tm[c] = tm[c].astype(str).str.strip()
lv["SQFT"] = pd.to_numeric(lv["SQFT"].astype(str).str.strip(), errors="coerce")
lv["ACRES"] = pd.to_numeric(lv["ACRES"].astype(str).str.strip(), errors="coerce")

tm_first = tm.sort_values("LRSN").groupby("LRSN", as_index=False).first()
lv_sum = lv.groupby("LRSN", as_index=False).agg(stated_sqft=("SQFT", "sum"),
                                                stated_acres=("ACRES", "sum"))
log(f"Tax Master rows {len(tm):,} -> {len(tm_first):,} accounts; "
    f"Land Value rows {len(lv):,} -> {len(lv_sum):,} accounts")

parcel = parcel.merge(tm_first, on="LRSN", how="left", suffixes=("", "_tm"))
parcel = parcel.merge(lv_sum, on="LRSN", how="left")
n_noval = int(parcel["TOTVAL1"].isna().sum())
log(f"Joined values; {n_noval:,} parcels have NO tax-master record")

# Sanity-check the Georgia 40% ratio so a source change can never silently flip the
# meaning of these columns (see the module docstring).
_ok = parcel[(parcel["TOTVAL1"] > 0) & parcel["TAXTOT1"].notna()]
if len(_ok):
    ratio40 = (_ok["TAXTOT1"] / _ok["TOTVAL1"]).median()
    log(f"GA assessment-ratio check: median TAXTOT1/TOTVAL1 = {ratio40:.3f} (expect ~0.40)")
    if not 0.38 <= ratio40 <= 0.42:
        raise RuntimeError(f"Unexpected assessment ratio {ratio40:.3f} — TOTVAL1 may no longer "
                           "be fair market value. Re-check the Tax Master schema before shipping.")

parcel = parcel.rename(columns={"LANDVAL1": "land_val", "DWLGVAL1": "bld_val",
                                "TOTVAL1": "tot_appr_val"})

# ── dedup: one account split across several GIS polygons ─────────────────────
# Values are `first` (account-level, broadcast onto every polygon); per-polygon stated area
# already summed at the account level above, so it is `first` here too.
ndup = int(parcel.duplicated(subset=["LRSN"], keep=False).sum())
log(f"Rows sharing an LRSN (multi-polygon accounts): {ndup:,}")
if ndup:
    first_cols = [c for c in parcel.columns if c not in ("geometry", "LRSN")]
    agg = {c: "first" for c in first_cols}
    coll = parcel.groupby("LRSN", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("LRSN", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]))
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
log(f"After LRSN dedup -> {len(parcel):,}")

# ── exemption flag ───────────────────────────────────────────────────────────
# Gwinnett encodes exemption TWO ways and neither alone is sufficient:
#  1. PROPCLAS 600-699 is the exempt block (600 Vacant Exempt, 612 School, 620 Religious,
#     650-665 Gwinnett County departments, 671-686 the cities, 690/691 State of Georgia
#     and GDOT, 699 Exemption Pending). This is the RELIABLE signal.
#  2. The parcel layer's EXEMPTION_TYPE (E0-E9 / ST / SV / SH) is populated on only a
#     handful of records — 'NE' (Not Exempt) on almost everything — so it is a supplement,
#     not the test. Using it alone would leave every school, church and city park in.
# Utilities (700-799) are excluded as well: per the playbook their assessed values are not
# comparable real-estate values (same rule as the Texas 'J' state class).
pc = pd.to_numeric(parcel["PROPCLAS"], errors="coerce")
exempt_class = pc.between(600, 699)
exempt_code = parcel["EXEMPTION_TYPE"].astype(str).str.strip().isin(
    ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "ST", "SH", "SV"])
utility_class = pc.between(700, 799)
parcel["exemption_flag"] = (exempt_class | exempt_code).astype(int)
log(f"Exempt by PROPCLAS 6xx: {int(exempt_class.sum()):,}; "
    f"by EXEMPTION_TYPE code: {int(exempt_code.sum()):,}; "
    f"union: {int(parcel['exemption_flag'].sum()):,}; utility 7xx: {int(utility_class.sum()):,}")
parcel = parcel[(parcel["exemption_flag"] == 0) & ~utility_class].copy()
log(f"After exempt + utility filter -> {len(parcel):,}")

# Parcels with no tax record at all carry no value and cannot be placed on a value map.
parcel = parcel[parcel["tot_appr_val"].notna()].copy()
log(f"After dropping value-less records -> {len(parcel):,}")

# ── condo / stacked-footprint diagnostic (add-city skill §6a) ────────────────
rp = parcel.geometry.representative_point()
parcel["_rpkey"] = rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)
vc = parcel["_rpkey"].value_counts()
stacked_keys = vc[vc > 1].index
log(f"Stacked footprints: {len(stacked_keys):,}; max stack: {int(vc.max())}; "
    f"parcels involved: {int(vc[vc > 1].sum()):,}")
if len(stacked_keys):
    # Per-unit condo records sharing one ground polygon: SUM values and stated area across
    # the stack, union the geometry. (Gwinnett maps most condo complexes at the building
    # level, so this normally touches only a small tail — the log line above is the check.)
    is_stacked = parcel["_rpkey"].isin(stacked_keys)
    single = parcel[~is_stacked].copy()
    single["_collapsed"] = 0
    multi = parcel[is_stacked].copy()
    sum_cols = ["land_val", "bld_val", "tot_appr_val", "stated_sqft", "stated_acres"]
    first_cols = [c for c in multi.columns if c not in (["geometry", "_rpkey"] + sum_cols)]
    agg = {c: "sum" for c in sum_cols if c in multi.columns}
    agg.update({c: "first" for c in first_cols})
    coll = multi.groupby("_rpkey", dropna=False).agg(agg).reset_index()
    coll["geometry"] = multi.groupby("_rpkey", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None])).values
    coll["_collapsed"] = 1
    parcel = gpd.GeoDataFrame(pd.concat([single, gpd.GeoDataFrame(coll, geometry="geometry",
                              crs="EPSG:4326")], ignore_index=True),
                              geometry="geometry", crs="EPSG:4326")
else:
    parcel["_collapsed"] = 0
parcel = parcel.drop(columns=["_rpkey"], errors="ignore")
log(f"After condo footprint collapse -> {len(parcel):,}")


# ── classification (Gwinnett PROPCLAS, a numeric statewide-style block scheme) ─
# 100-199 residential | 200-299 multifamily & lodging | 300-499 commercial/industrial
# 500-599 residual & common area | 600-699 exempt (already dropped) | 700-799 utility (dropped)
# There is NO surface-parking class in this scheme, so no parcel is ever categorized
# 'Parking' here — Duluth's parking layer comes from OSM only.
VACANT = {100, 113, 119, 123, 124, 150, 299, 300, 500, 700}
SINGLE_FAMILY = {101, 109, 110, 112, 125, 151, 197, 198, 199}
TOWNHOME = {107}
CONDO = {106, 355, 356, 357}
MULTIFAMILY = {102, 103, 202, 203, 204, 211, 212, 214}
MOBILE_HOME = {108, 111, 213}
MIXED_USE = {105}
COMMON_AREA = {120, 122, 322, 520}
INDUSTRIAL = {392, 395, 396, 397, 398, 399, 401, 459, 501, 506}


def categorize(code, desc):
    c = pd.to_numeric(code, errors="coerce")
    d = str(desc or "").strip()
    if pd.isna(c):
        return "Other"
    c = int(c)
    if c in VACANT:
        return "Vacant Land"
    if c in SINGLE_FAMILY:
        return "Single Family"
    if c in TOWNHOME:
        return "Townhome"
    if c in CONDO:
        return "Condominium"
    if c in MULTIFAMILY:
        return "Multifamily"
    if c in MOBILE_HOME:
        return "Mobile Home"
    if c in MIXED_USE:
        return "Mixed Use"
    if c in COMMON_AREA:
        return "Common Area"
    if c in INDUSTRIAL:
        return "Industrial"
    if 140 <= c <= 141:
        return "Common Area"          # private drives held by non-HOA owners
    if 115 <= c <= 117:
        return "Vacant Land"          # conservation easement / environmentally sensitive
    if 252 <= c <= 257:
        return "Hotel"
    if 200 <= c <= 599:
        return "Commercial"
    if 100 <= c <= 199:
        return "Single Family"
    return "Other"


parcel["PROPERTY_CATEGORY"] = [categorize(c, d) for c, d in
                               zip(parcel["PROPCLAS"], parcel["PCDESC"])]

ex = parcel.copy()
ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other", "Common Area"),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    fetch_footprints=False)
log(f"Classified {len(ex):,} taxable parcels")


# ── land area: stated assessor SQFT, geodesic polygon area as the fallback ───
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
ex["reported_sqft"] = pd.to_numeric(ex.get("stated_sqft", np.nan), errors="coerce")
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

# Collapsed stacks: the denominator is the SHARED footprint, not one unit's share.
col = ex["_collapsed"] == 1
ex.loc[col, "reported_sqft"] = np.maximum(
    pd.to_numeric(ex.loc[col, "reported_sqft"], errors="coerce").fillna(0.0),
    pd.to_numeric(ex.loc[col, "geom_area_sqft"], errors="coerce").fillna(0.0))
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

# The assessor's stated SQFT is a land-segment figure and for condo / apartment / common-area
# records it is a PER-UNIT SHARE against the real shared polygon, which produces absurd
# $/sqft (smoke alarms #2/#4). The mapped polygon is authoritative, so stated area is used
# only when it sits in a sane band around the polygon; otherwise fall back to geodesic area.
ratio = ex["reported_sqft"] / ex["geom_area_sqft"].replace(0, np.nan)
use_reported = (ex["reported_sqft"] > 0) & ratio.between(0.5, 2.0)
ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["geom_area_sqft"])
ex["area_source"] = np.where(use_reported, "reported", "gis")
log(f"Area denominator: reported={int(use_reported.sum()):,} "
    f"gis-fallback={int((~use_reported).sum()):,} "
    f"(rejected stated area: {int(((ex['reported_sqft'] > 0) & ~ratio.between(0.5, 2.0)).sum()):,})")
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

ex["full_market_value"] = pd.to_numeric(ex.get("tot_appr_val", np.nan), errors="coerce")
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

ex["link"] = SEARCH_URL

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
final["geometry"] = final["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
final = gpd.GeoDataFrame(final, geometry="geometry", crs=ex.crs)
if final.crs is None or final.crs.to_epsg() != 4326:
    final = final.to_crs("EPSG:4326")
out = DATA_DIR / "duluth-ga-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"duluth-ga-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet",
                 index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")
log(f"land_value_per_sqft: p50=${final['land_value_per_sqft'].median():.2f} "
    f"p99=${final['land_value_per_sqft'].quantile(.99):.2f} "
    f"max=${final['land_value_per_sqft'].max():.2f}")
log(f"TOTAL land value: ${final['current_full_land_value'].sum():,.0f} over "
    f"{final['land_area_acres'].sum():,.0f} acres")
log("DONE")
