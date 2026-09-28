#!/usr/bin/env python3
"""
Build the City of Culver City, CA canonical parcel parquet.

Culver City (pop. ~40k, 5.0 sq mi) is an independent city in Los Angeles County, wedged between
the City of Los Angeles (Palms / Mar Vista / Westchester) and unincorporated Ladera Heights.
Requested by a user who pointed at the city's own GIS viewer (gisproxy.culvercity.org, "View
Additional Details" -> "Current Land Base Value"); the city's open-data download only carries
boundaries. Those viewer values are the LA County Assessor's roll, so this ETL goes to the county.

SOURCE — LA County's public parcel layer (eGIS, public, no token):
  https://public.gis.lacounty.gov/public/rest/services/LACounty_Cache/LACounty_Parcel/MapServer/0
  One row per Assessor ID Number (AIN), ~2.4M countywide. Geometry + the current secured roll
  (Roll_Year 2026): Roll_LandValue / Roll_ImpValue, Roll_LandBaseYear / Roll_ImpBaseYear,
  Roll_RealEstateExemp, the Assessor's 4-character UseCode + UseType / UseDescription, building
  year/units/sqft, and TaxRateCity. THE LAYER CARRIES NO OWNER NAMES (the county publishes those
  only through the Assessor portal one record at a time), so there is nothing to strip.
  City boundary: LA County eGIS Political_Boundaries/MapServer/19 (City Boundaries),
  CITY_NAME='Culver City'.

PROP 13 — READ BEFORE INTERPRETING ANY NUMBER HERE. California's Proposition 13 (1978) sets a
parcel's assessed value at its value when it last changed hands (or was newly built), the
"base year value", rising by at most 2% a year after that. So Roll_LandValue is NOT market
value: a lot held since the 1970s carries a 1975 land value inflated 2%/yr, while the identical
lot next door that sold in 2023 carries its 2023 purchase price. Land values are therefore
heavily understated for long-held parcels and vary block-by-block with sale date rather than
location. The ETL logs the magnitude (median assessed land $/sqft on ordinary single-family lots
by land base year) and ships `land_base_year` per parcel so the popup can show it.

Outputs:
- data/jurisidictions/data/culvercity/culvercity-ca-parcels.parquet
- data/jurisidictions/data/culvercity/culvercity-ca-parcels_YYYY_MM_DD.parquet

Notes:
- CLIP: representative-point-within the county's Culver City boundary. The county's own
  TaxRateCity='CULVER CITY' agrees with that clip on all but one of ~13.7k parcels (a parking
  lot straddling the city line), so the two jurisdiction signals are consistent.
- EXEMPTION: California does not put a value on government-owned land at all (UseCode 88xx
  "Government Parcel", Roll_LandValue NULL), and state-assessed utility property (81xx) is
  valued by the State Board of Equalization, not on this roll — both are dropped. Welfare /
  religious / college exemptions (churches, private schools, nonprofit housing) carry a normal
  roll value plus a Roll_RealEstateExemp equal to it; a parcel is exempt when that exemption is
  >= 99% of land + improvements. 1-4 unit residential is NEVER dropped on the exemption test:
  those exemptions are the disabled veterans' exemption (~$180k), which fully covers a
  long-held low-base-year house, and dropping it would both remove a taxable-class home and
  reveal the owner's veteran/disability status by omission. Parcels carrying only a nominal
  placeholder value (land < $1,000 and no improvements — condo common lots, pipeline and
  ROW strips, $9 mall-lot placeholders) are dropped as unassessed.
- CONDOS: LA County gives every condo unit its own AIN but maps it as an IDENTICAL COPY OF THE
  WHOLE LOT POLYGON (4,360 records in 132 stacks, max stack 404 at Tract 23852 on Green Valley
  Circle). Each unit carries its own real land share, so stacks collapse to ONE parcel per lot
  with land / improvement values SUMMED across the distinct AINs (never `first`: these are
  different accounts, not one account broadcast onto several polygons). Stacks are matched on
  footprint overlap (IoU >= 0.95), not exact geometry, because ~120 units' copies differ from
  their siblings by digitising noise. There is no separate common-area parcel underneath, so
  no merge-down is needed (unlike Olympia/Provo). Mixed stacks (Westfield Culver City's ten
  parcel-map lots on one footprint, a Culver Blvd building with a commercial module over
  residential units) collapse the same way.
- PUD TOWNHOMES are the one stub case: the Fox Hills PUD tracts map each townhouse as a
  building-sized fee lot with the grounds on separate $9 placeholder lots, so those are merged
  DOWN onto the common land per Assessor map page (see the PUD merge block).
- Result (2026-09-28 run): 13,678 records -> 8,744 parcels, $10.29B assessed land / $18.07B
  assessed total. Median single-family land $/sqft: $11.91 at a 1975-79 base year vs $238.23
  at 2020-26 (the Prop 13 section of the log).
- ~13k parcels -> browser GeoParquet would work, but baked to PMTiles + H3 anyway: condo-heavy
  small-lot cities render sparse when zoomed out on the GeoParquet path (Olympia lesson).
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
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import (  # noqa: E402
    add_improvement_ratio_fields,
    classify_property_refined,
    gis_area_sqft,
)

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "culvercity"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "culvercity-ca-county-parcels.parquet"
BOUNDARY_CACHE = DATA_DIR / "culvercity-boundary.geojson"

PARCELS_URL = ("https://public.gis.lacounty.gov/public/rest/services/LACounty_Cache/"
               "LACounty_Parcel/MapServer/0/query")
BOUNDARY_URL = ("https://public.gis.lacounty.gov/public/rest/services/LACounty_Dynamic/"
                "Political_Boundaries/MapServer/19/query")
BOUNDARY_WHERE = "CITY_NAME='Culver City'"

# The layer has no owner-name fields at all. Homeowners'/real-estate exemption AMOUNTS are
# fetched for the exemption test only and are never exported (they reveal owner-occupancy and
# disabled-veteran status per household).
OUT_FIELDS = (
    "AIN,SitusFullAddress,TaxRateCity,UseCode,UseType,UseDescription,YearBuilt1,"
    "Units1,Units2,Units3,Units4,Units5,SQFTmain1,SQFTmain2,SQFTmain3,SQFTmain4,SQFTmain5,"
    "Roll_Year,Roll_LandValue,Roll_ImpValue,Roll_RealEstateExemp,Roll_LandBaseYear,"
    "Roll_ImpBaseYear,ParcelTypeCode"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) CivicMapper-ETL",
           "Accept": "application/json"}
PAGE = 1000
SQFT_PER_ACRE = 43560.0
SQM_TO_SQFT = 10.763910416709722
UTM = "EPSG:32611"            # UTM 11N — Los Angeles
NOMINAL_LAND = 1000.0         # placeholder roll values ($9 / $22 / $129 ...) sit far below this
FULL_EXEMPT_SHARE = 0.99
STACK_IOU = 0.95


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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


# ── fetch ─────────────────────────────────────────────────────────────────────
def fetch_boundary() -> gpd.GeoDataFrame:
    if not BOUNDARY_CACHE.exists():
        r = requests.get(BOUNDARY_URL, params={
            "where": BOUNDARY_WHERE, "outFields": "CITY_NAME,FEAT_TYPE", "returnGeometry": "true",
            "outSR": 4326, "f": "geojson"}, headers=HEADERS, timeout=120)
        r.raise_for_status()
        BOUNDARY_CACHE.write_bytes(r.content)
    b = gpd.read_file(BOUNDARY_CACHE).to_crs("EPSG:4326")
    if len(b) != 1:
        raise RuntimeError(f"Expected one Culver City boundary feature, got {len(b)}")
    return b


def fetch_parcels(bounds) -> gpd.GeoDataFrame:
    """County layer filtered to the city's (padded) envelope, cached. The boundary does the clip."""
    if GEOM_CACHE.exists():
        log(f"Using cached parcels: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    minx, miny, maxx, maxy = bounds
    pad = 0.002
    geom_params = {
        "geometry": json.dumps({"xmin": minx - pad, "ymin": miny - pad, "xmax": maxx + pad,
                                "ymax": maxy + pad, "spatialReference": {"wkid": 4326}}),
        "geometryType": "esriGeometryEnvelope", "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
    }
    total = requests.get(PARCELS_URL, params={**geom_params, "where": "1=1",
                         "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=120).json()["count"]
    log(f"Pulling {total:,} rows in the Culver City envelope (paginated GeoJSON)...")
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
                # MapServers answer HTTP 200 with an {"error": ...} body on overload — without
                # this check that reads as "zero features" and silently truncates the pull.
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
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    if len(geom) != total:
        raise RuntimeError(f"Pulled {len(geom):,} rows but the server counted {total:,}")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


boundary = fetch_boundary()
raw = fetch_parcels(boundary.total_bounds)
log(f"Envelope rows: {len(raw):,}")

# ── clip to the authoritative city boundary (representative point within) ────
n_bad = int((~raw.geometry.is_valid).sum())
raw["geometry"] = raw["geometry"].apply(
    lambda g: g if (g is not None and g.is_valid) else _polygonal(make_valid(g)))
raw = raw[raw["geometry"].notna() & ~raw.geometry.is_empty].copy()
log(f"Repaired {n_bad:,} invalid source geometries")
city = boundary.geometry.union_all()
inside = gpd.GeoSeries(raw.geometry.representative_point(), crs="EPSG:4326").within(city).values
trc = raw["TaxRateCity"].eq("CULVER CITY").values
log(f"Boundary clip vs TaxRateCity: both {int((inside & trc).sum()):,} | boundary only "
    f"{int((inside & ~trc).sum()):,} | TaxRateCity only {int((~inside & trc).sum()):,}")
raw = raw[inside].copy()
log(f"Inside the Culver City boundary: {len(raw):,} rows, {raw['AIN'].nunique():,} distinct AINs")
if raw["AIN"].duplicated().any():
    raise RuntimeError("Duplicate AINs in the county layer — expected one row per AIN")

for c in ["Roll_LandValue", "Roll_ImpValue", "Roll_RealEstateExemp", "YearBuilt1",
          "Roll_LandBaseYear", "Roll_ImpBaseYear", "ParcelTypeCode",
          *[f"Units{i}" for i in range(1, 6)], *[f"SQFTmain{i}" for i in range(1, 6)]]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw["UseCode"] = raw["UseCode"].fillna("").astype(str).str.strip()
raw["uc2"] = raw["UseCode"].str[:2]
raw["uc4"] = raw["UseCode"].str[3:4]
raw["land_val"] = raw["Roll_LandValue"]
raw["bld_val"] = raw["Roll_ImpValue"].fillna(0)
raw["units"] = raw[[f"Units{i}" for i in range(1, 6)]].fillna(0).sum(axis=1)
raw["bld_ar"] = raw[[f"SQFTmain{i}" for i in range(1, 6)]].fillna(0).sum(axis=1)
log(f"Roll years: {raw['Roll_Year'].value_counts(dropna=False).to_dict()}")

# ── per-record exemption (BEFORE the condo collapse, so exempt units never sum in) ──
tot = raw["land_val"].fillna(0) + raw["bld_val"]
exempt_share = raw["Roll_RealEstateExemp"].fillna(0) / tot.replace(0, np.nan)
res_1to4 = raw["uc2"].isin(["01", "02", "03", "04"])
reason = pd.Series("", index=raw.index)
reason[raw["uc2"].eq("81")] = "state-assessed utility"
reason[raw["uc2"].eq("88") | raw["UseType"].eq("Government")] = "government"
reason[(reason == "") & raw["land_val"].isna()] = "no roll value"
reason[(reason == "") & exempt_share.ge(FULL_EXEMPT_SHARE) & ~res_1to4] = "welfare/religious exemption"
reason[(reason == "") & raw["land_val"].lt(NOMINAL_LAND) & raw["bld_val"].lt(NOMINAL_LAND)] = "nominal placeholder value"
raw["exempt_reason"] = reason
log(f"Dropped per record: {reason[reason != ''].value_counts().to_dict()}")
ex_rec = raw[reason != ""]
log(f"  (dropped records carry ${ex_rec['land_val'].fillna(0).sum() / 1e6:,.1f}M roll land value; "
    f"1-4 unit homes with a >=99% exemption kept: "
    f"{int((exempt_share.ge(FULL_EXEMPT_SHARE) & res_1to4).sum())})")
rec = raw[reason == ""].copy()

# ── condo stacks: N identical copies of the lot polygon -> one parcel per lot ──
# Union-find over footprint pairs. Two records are copies of one lot when their footprints agree
# to IoU >= STACK_IOU — or, when BOTH are condo-type records (ParcelTypeCode 1), when the overlap
# covers >= 90% of the smaller and the two are within 0.7x of each other in size: a few units'
# lot copies were digitised from an older, slightly larger lot line (Indian Wood Rd: one unit's
# copy is 116k sqft against its 86 siblings' 99k), and a strict IoU leaves that one unit as its
# own near-$0/sqft parcel lying on top of the development.
rec = rec.reset_index(drop=True)
is_condo_rec = rec["ParcelTypeCode"].eq(1).to_numpy()
rp = rec.to_crs(UTM)
area_m2 = rp.geometry.area.to_numpy()
pairs = gpd.sjoin(rp[["geometry"]], rp[["geometry"]], predicate="intersects")
pairs = pairs[pairs.index < pairs["index_right"]]
parent = np.arange(len(rec))


def _find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


li, ri = pairs.index.to_numpy(), pairs["index_right"].to_numpy()
geo = rp.geometry.to_numpy()
for a, b in zip(li, ri):
    lo, hi = sorted((area_m2[a], area_m2[b]))
    both_condo = is_condo_rec[a] and is_condo_rec[b]
    if hi <= 0 or lo / hi < (0.7 if both_condo else STACK_IOU):
        continue
    inter = geo[a].intersection(geo[b]).area
    iou = inter / (area_m2[a] + area_m2[b] - inter)
    if iou >= STACK_IOU or (both_condo and inter / lo >= 0.90):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra
rec["lot"] = [_find(i) for i in range(len(rec))]
sizes = rec["lot"].map(rec["lot"].value_counts())
log(f"Footprint stacks: {int((rec.groupby('lot').size() > 1).sum()):,} lots carrying "
    f"{int((sizes > 1).sum()):,} records (max stack {int(sizes.max())}); condo-type records "
    f"(ParcelTypeCode 1) in a stack: {int(((sizes > 1) & rec['ParcelTypeCode'].eq(1)).sum()):,} of "
    f"{int(rec['ParcelTypeCode'].eq(1).sum()):,}")

# ── classification (per record, then dominant-by-value per lot) ──────────────
COMMERCIAL_MAP = {"11": "Retail", "12": "Retail", "13": "Retail", "14": "Retail", "15": "Retail",
                  "16": "Retail", "21": "Retail", "17": "Office", "19": "Office",
                  "18": "Hotel"}


def categorize(uc2: str, uc4: str, bld: float, land: float) -> str:
    if not uc2:
        return "Other"
    # A 'V' (vacant) code the Assessor never updated after construction is not vacant: Ivy
    # Station (8825 National Blvd) still carries 100V with $144M of improvements on the roll.
    if uc4 == "V" and bld <= 0.5 * land:
        if uc2 < "10":
            return "Vacant Residential"
        if uc2 < "30":
            return "Vacant Commercial"
        if uc2 < "40":
            return "Vacant Industrial"
        return "Vacant Other"
    if uc2 in ("27", "38"):
        # The Assessor's "Parking Lots" use covers both surface lots and decks. A structure is a
        # building (issue #12 house rule), so it must not be force-labelled underused.
        return "Parking Lot" if bld < land else "Parking Structure"
    if uc2 == "01":
        if uc4 in ("C", "E"):
            return "Condominium"
        if uc4 == "D":
            return "Townhome / PUD"
        return "Single Family"
    if uc2 in ("02", "03", "04"):
        return "Multifamily (2-4 units)"
    if uc2 in ("05", "06", "07", "08"):
        return "Multifamily (5+ units)"
    if uc2 == "09":
        return "Mobile Home Park"
    if uc2 < "30":
        return COMMERCIAL_MAP.get(uc2, "Commercial")
    if uc2 == "35":
        return "Studio / Media Production"
    if uc2 < "40":
        return "Industrial"
    if uc2 < "50":
        return "Agricultural / Rural"
    if uc2.startswith("6"):
        return "Recreational"
    if uc2.startswith("7"):
        return "Institutional"
    return "Other"


rec["category"] = [categorize(a, b, c, d) for a, b, c, d in zip(
    rec["uc2"], rec["uc4"], rec["bld_val"], rec["land_val"].fillna(0))]
rec["tot_val"] = rec["land_val"] + rec["bld_val"]


def _dominant_by_value(df):
    s = df.groupby("category")["tot_val"].sum()
    return s.idxmax()


grp = rec.groupby("lot")
lots = pd.DataFrame({
    "n_records": grp.size(),
    "land_val": grp["land_val"].sum(),
    "bld_val": grp["bld_val"].sum(),
    "units": grp["units"].sum(),
    "bld_ar": grp["bld_ar"].sum(),
    "by_min": grp["Roll_LandBaseYear"].min(),
    "by_max": grp["Roll_LandBaseYear"].max(),
    "by_median": grp["Roll_LandBaseYear"].median(),
    "yr_built": grp["YearBuilt1"].min(),
})
# Representative record per lot = the highest-value one (address, AIN for the link).
rep = rec.sort_values("tot_val", ascending=False).drop_duplicates("lot").set_index("lot")
lots["ain"] = rep["AIN"]
lots["address"] = rep["SitusFullAddress"].fillna("").str.strip()
lots["use_description"] = rep["UseDescription"]
lots["use_code"] = rep["UseCode"]
lots["category"] = grp.apply(_dominant_by_value, include_groups=False)
cats_per_lot = grp["category"].agg(lambda s: set(s))
res_cats = {"Condominium", "Single Family", "Townhome / PUD", "Multifamily (2-4 units)",
            "Multifamily (5+ units)"}
mixed = cats_per_lot.apply(lambda s: bool(s & res_cats) and bool(s - res_cats))
lots.loc[mixed[mixed].index, "category"] = "Mixed Use"
# Geometry: the largest copy in the stack (copies agree to IoU >= 0.95).
rec["_a"] = area_m2
geom_rep = rec.sort_values("_a", ascending=False).drop_duplicates("lot").set_index("lot")["geometry"]
lots = gpd.GeoDataFrame(lots, geometry=geom_rep.reindex(lots.index).values, crs="EPSG:4326")
lots = lots.reset_index(drop=True)


def _base_year_label(r):
    if pd.isna(r["by_min"]):
        return None
    if r["by_min"] == r["by_max"]:
        return str(int(r["by_min"]))
    n = int(r["_merged_units"]) if r.get("_merged_units", 0) > 0 else int(r["n_records"])
    return f"{int(r['by_min'])}-{int(r['by_max'])} ({n} units, median {int(r['by_median'])})"


# ── townhome / PUD units -> merge DOWN onto their tract's common land (skill §6b) ──
# The Fox Hills PUD tracts (Raintree / Tara Ter / Butterfield Ct, map book 4296) map each
# townhouse as its own fee lot the size of its building footprint (~1,250-1,800 sqft) carrying
# the unit's share of the land value, while the development's grounds, drives and pools are
# separate lots the Assessor carries at a $9 placeholder (010D, land $9 / impr $0). Left as-is
# the units render as a field of $600-750/sqft pencils against a ~$240 citywide recent-sale rate,
# and the grounds vanish with the nominal-value filter. So: per Assessor map page (AIN[:7]),
# take the connected clusters of small valued residential lots + nominal common lots and fold
# each into one development parcel (values summed, common land's footprint included, holes
# filled). Gate = PLAT DOMINANCE (the Provo lesson): a page merges only when its small stubs are
# >= 60% of the page's valued parcels, so an ordinary subdivision with a couple of small lots
# and an HOA strip never hands the strip to lots that don't own it.
STUB_SQFT = 2000.0
COMMON_MIN_SQFT = 2000.0
STUB_SHARE = 0.60
RES_STUB_CATS = {"Single Family", "Townhome / PUD", "Condominium", "Multifamily (2-4 units)"}
lots["_sqft"] = lots.to_crs(UTM).geometry.area * SQM_TO_SQFT
lots["page"] = lots["ain"].astype(str).str[:7]
lots["_stub"] = (lots["n_records"].eq(1) & lots["_sqft"].lt(STUB_SQFT)
                 & lots["category"].isin(RES_STUB_CATS) & lots["land_val"].gt(0))
commons = raw[raw["exempt_reason"].eq("nominal placeholder value")
              & raw["uc2"].isin(["01", "02", "03", "04", "05"])].copy()
commons["_sqft"] = commons.to_crs(UTM).geometry.area * SQM_TO_SQFT
commons = commons[commons["_sqft"].ge(COMMON_MIN_SQFT)].copy()
commons["page"] = commons["AIN"].astype(str).str[:7]
_page = lots.groupby("page").agg(n_val=("land_val", "size"), n_stub=("_stub", "sum"))
_page["share"] = _page["n_stub"] / _page["n_val"]
merge_pages = set(_page.index[(_page["n_stub"] >= 2) & (_page["share"] >= STUB_SHARE)]) \
    & set(commons["page"])


def _fill_holes(g):
    from shapely.geometry import MultiPolygon, Polygon
    if g is None or g.is_empty:
        return g
    if g.geom_type == "Polygon":
        return Polygon(g.exterior)
    if g.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in g.geoms])
    return g


dev_rows, consumed_lots = [], set()
n_commons_used = 0
for page in sorted(merge_pages):
    st = lots[lots["page"].eq(page) & lots["_stub"]]
    cm = commons[commons["page"].eq(page)]
    nodes = [("s", i, g) for i, g in zip(st.index, st.to_crs(UTM).geometry)] + \
            [("c", i, g) for i, g in zip(cm.index, cm.to_crs(UTM).geometry)]
    par = list(range(len(nodes)))

    def _f(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    buf = [n[2].buffer(0.5) for n in nodes]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if buf[i].intersects(buf[j]):
                par[_f(j)] = _f(i)
    comps: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        comps.setdefault(_f(i), []).append(i)
    for members in comps.values():
        s_idx = [nodes[m][1] for m in members if nodes[m][0] == "s"]
        c_idx = [nodes[m][1] for m in members if nodes[m][0] == "c"]
        if len(s_idx) < 2 or not c_idx:
            continue
        geom = _fill_holes(unary_union(list(lots.loc[s_idx].geometry)
                                       + list(commons.loc[c_idx].geometry)))
        # Units whose lot is a little over STUB_SQFT (end units run 2,000-2,600 sqft) sit INSIDE the
        # development once its holes are filled; left out they'd render on top of it. Absorb them.
        cand = lots[lots["page"].eq(page) & lots["n_records"].eq(1)
                    & lots["category"].isin(RES_STUB_CATS) & ~lots.index.isin(s_idx)
                    & ~lots.index.isin(list(consumed_lots))]
        extra = cand.index[cand.geometry.representative_point().within(geom)].tolist()
        s_idx = s_idx + extra
        u = lots.loc[s_idx]
        top = u.sort_values("land_val", ascending=False).iloc[0]
        cat_val = (u["land_val"] + u["bld_val"]).groupby(u["category"]).sum()
        yrs = u["by_min"].dropna()
        dev_rows.append({
            "n_records": int(u["n_records"].sum()) + len(c_idx),
            "land_val": u["land_val"].sum(), "bld_val": u["bld_val"].sum(),
            "units": u["units"].sum(), "bld_ar": u["bld_ar"].sum(),
            "by_min": yrs.min() if len(yrs) else np.nan,
            "by_max": yrs.max() if len(yrs) else np.nan,
            "by_median": yrs.median() if len(yrs) else np.nan,
            "yr_built": u["yr_built"].min(),
            "ain": top["ain"], "address": top["address"],
            "use_description": top["use_description"], "use_code": top["use_code"],
            "category": cat_val.idxmax(),
            "geometry": geom,
            "_merged_units": len(s_idx),
        })
        consumed_lots.update(s_idx)
        n_commons_used += len(c_idx)

if dev_rows:
    devs = gpd.GeoDataFrame(pd.DataFrame(dev_rows), geometry="geometry", crs="EPSG:4326")
    lots = lots.drop(index=list(consumed_lots))
    lots["_merged_units"] = 0
    lots = gpd.GeoDataFrame(pd.concat([lots, devs], ignore_index=True), geometry="geometry",
                            crs="EPSG:4326")
    _psf = devs["land_val"] / (devs.to_crs(UTM).geometry.area * SQM_TO_SQFT)
    log(f"PUD merge: {int(devs['_merged_units'].sum()):,} townhome/unit lots + {n_commons_used:,} "
        f"nominal common lots -> {len(devs):,} development parcels on {len(merge_pages):,} map "
        f"pages (land $/sqft p50 ${_psf.median():,.0f}, max ${_psf.max():,.0f})")
else:
    lots["_merged_units"] = 0
_left = lots["_stub"].fillna(False).astype(bool)
log(f"Small residential lots left individual (no merge page / no common lot): {int(_left.sum()):,}")
lots = lots.drop(columns=["_sqft", "_stub", "page"], errors="ignore")

lots["land_base_year"] = lots.apply(_base_year_label, axis=1)
lots["assessor_records"] = lots["n_records"].astype(int)
log(f"Records -> parcels: {len(rec):,} -> {len(lots):,}")

ex = lots.copy()
ex["exemption_flag"] = 0
ex["property_land_use_category"] = ex["category"]
ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other", "Agricultural / Rural", "Institutional", "Recreational"),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    bld_ar_col="bld_ar", fetch_footprints=False)

# ── areas (geodesic, hole-subtracting) ────────────────────────────────────────
log("Computing geodesic areas...")
ex["land_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["land_area_sqft"] < 1, "land_area_sqft"] = np.nan
ex["area_source"] = "gis"
ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
# Standard sub-500-sqft sliver rule, plus one narrow extension: a sub-1,000-sqft fragment
# carrying >= $2,000/sqft of land. That catches exactly one record — AIN 4209029023, a 620 sqft
# "PAR C" strip beside a Jefferson Blvd office block carrying $5.31M of land ($8,559/sqft, ~10x
# anything else in the city) — a genuine Assessor record whose value belongs to an assemblage,
# the Hartford 0 OLIVE ST class of anomaly. Hidden by hideRemnants, not altered.
ex["likely_remnant"] = ((ex["land_area_sqft"] < 500)
                        | ((ex["land_area_sqft"] < 1000)
                           & (ex["land_value"] / ex["land_area_sqft"] >= 2000))).astype(int)
log(f"likely_remnant: {int(ex['likely_remnant'].sum()):,}")

ex["full_market_value"] = ex["land_value"] + ex["improvement_value"]
den = ex["land_area_sqft"].replace(0, np.nan)
ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
ex["land_value_per_sqft"] = ex["land_value"] / den
ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

# LA County Assessor portal detail page (verified live: /parceldetail/<10-digit AIN> -> 200).
ex["link"] = "https://portal.assessor.lacounty.gov/parceldetail/" + ex["ain"].astype(str)

# ── export ───────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant", "ain", "address", "use_description", "land_base_year",
           "assessor_records"]
final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")
out = DATA_DIR / "culvercity-ca-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"culvercity-ca-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet",
                 index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"land value total: ${final['current_full_land_value'].sum() / 1e9:,.2f}B | "
    f"assessed (land+impr) total: ${final['full_market_value'].sum() / 1e9:,.2f}B")

# ── Prop 13: how far assessed land lags by base year ─────────────────────────
log("--- Prop 13 base-year effect (ordinary single-family lots, one AIN per lot) ---")
sf = ex[(ex["category"] == "Single Family") & (ex["assessor_records"] == 1)
        & ex["land_value_per_sqft"].notna()].copy()
sf["by"] = sf["by_min"]
bins = [0, 1979, 1989, 1999, 2009, 2019, 2030]
labels = ["1975-79", "1980s", "1990s", "2000s", "2010s", "2020-26"]
tbl = sf.groupby(pd.cut(sf["by"], bins, labels=labels), observed=True)["land_value_per_sqft"].agg(
    ["count", "median"])
for k, r in tbl.iterrows():
    log(f"  land base year {k:>8}: {int(r['count']):5,} lots, median assessed land ${r['median']:,.2f}/sqft")
old = sf.loc[sf["by"] < 2000, "land_value_per_sqft"].median()
new = sf.loc[sf["by"] >= 2020, "land_value_per_sqft"].median()
log(f"  pre-2000 base year median ${old:,.2f}/sqft vs 2020+ ${new:,.2f}/sqft -> {new / old:,.1f}x")
log(f"  share of SF lots with a pre-2000 land base year: {(sf['by'] < 2000).mean():.0%}")
wtd = ex.groupby(pd.cut(ex["by_median"], bins, labels=labels), observed=True)["land_value"].sum()
log(f"  roll land value by (median) base year: {({k: round(v / 1e9, 2) for k, v in wtd.items()})} ($B)")
# What the SF lots would carry at the 2020+ median rate (a crude, location-blind yardstick).
_rate = sf.loc[sf["by"] >= 2020, "land_value_per_sqft"].median()
log(f"  SF lots: roll land ${sf['land_value'].sum() / 1e9:,.2f}B vs "
    f"${(sf['land_area_sqft'] * _rate).sum() / 1e9:,.2f}B at the 2020+ median rate")

# ── §6a smoke alarms ─────────────────────────────────────────────────────────
log("--- condo/stub smoke alarms (skill §6a) ---")
a = ex["land_area_sqft"]
lv = pd.to_numeric(final["land_value_per_sqft"], errors="coerce")
log(f"  footprint sqft p1/p5/p10/p50: {[round(a.quantile(q)) for q in (.01, .05, .10, .50)]}")
log(f"  sub-500 / sub-1000 sqft footprints: {int((a < 500).sum()):,} / {int((a < 1000).sum()):,}")
log(f"  land $/sqft p50/p95/p99/max: ${lv.median():,.2f} / ${lv.quantile(.95):,.2f} / "
    f"${lv.quantile(.99):,.2f} / ${lv.max():,.2f}")
holes = final.geometry.apply(lambda g: 0 if g is None else sum(
    len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
log(f"  parcels with interior rings (holes): {int((holes > 0).sum()):,}")
rp2 = final.geometry.representative_point()
_vc = (rp2.x.round(6).astype(str) + "," + rp2.y.round(6).astype(str)).value_counts()
log(f"  stacked footprints left: {int((_vc > 1).sum()):,} clusters, max stack {int(_vc.max())}")
_inside_other = gpd.sjoin(gpd.GeoDataFrame(geometry=rp2, crs="EPSG:4326"),
                          final[["geometry"]], predicate="within")
log(f"  parcels whose rep point lies inside ANOTHER shipped parcel: "
    f"{int((_inside_other.index != _inside_other['index_right']).sum()):,}")
log(f"  zero/neg land value: {int((final['current_full_land_value'].fillna(0) <= 0).sum()):,}")
log(f"  bounds: {[round(v, 4) for v in final.total_bounds]}")
log("DONE")
