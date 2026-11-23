#!/usr/bin/env python3
"""
Build the Somerset County, NJ (all 21 municipalities) + Princeton (Mercer) parcel parquet from the
NJ Office of GIS statewide "Parcels and MOD-IV Composite" layer.

Source (one-stop: geometry + MOD-IV tax-list values + NJ property class):
  NJOGIS  Parcels_Composite_NJ_WM/FeatureServer/0
  https://services2.arcgis.com/XVOqAjTOJ5P6ngMu/arcgis/rest/services/Parcels_Composite_NJ_WM/FeatureServer/0
  filtered COUNTY='SOMERSET' OR PCL_MUN='1114' (126,226 rows, published 2026-01-05). Owner-name fields are
  never requested.

VALUE YEAR: the layer's MOD-IV values are the TAX YEAR 2025 assessments. Verified by
summing taxable (non-class-15) LAND_VAL+IMPRVT_VAL per town against the Somerset County
Board of Taxation's Final Equalization Tables: e.g. Bernards $9.26B vs $9.30B (2025 table)
vs $9.88B (2026 table); Franklin $16.45B vs $16.68B (2025) vs $17.35B (2026). Every town
lines up with 2025, not 2026.

EQUALIZATION: NJ towns assess at different shares of market value. Most Somerset towns
reassess annually and sit near 100%, but five do not (2025 ratios: Montgomery 59.97%,
North Plainfield 57.23%, Raritan 66.45%, Somerville 63.46%, South Bound Brook 66.23%).
Unadjusted, a county-wide land map would show false cliffs at those town lines. So every
assessed value is divided by its town's 2025 average ratio of assessed to true value
(Final Equalization Table, County of Somerset, 2025, column 1(b)):
  https://www.nj.gov/njbonds/treasury/taxation/pdf/lpt/county_equalized/2025/2025Somerset.pdf
The assessor's own (unequalized) figures ship alongside as assessed_land_value /
assessed_improvement_value.
EXCEPTION: class 3B qualified farmland is assessed at its Farmland Assessment Act use value
(N.J.S.A. 54:4-23.1 et seq.), not at a share of market value, so its land is NOT equalized.

NJ PROPERTY CLASSES (N.J.A.C. 18:12-2.2): 1 vacant, 2 residential (1-4 family), 3A farm
regular (homestead), 3B farm qualified (farmland-assessed), 4A commercial, 4B industrial,
4C apartments (5+), 5A/5B railroad, 15A-F exempt (public school, other school, public
property, church & charitable, cemetery, other exempt).
  - 15A-F: exempt -> dropped (exemption_flag=1 before the drop).
  - 5A/5B: railroad -> dropped (state-assessed utility land, not a market value).

CONDOS (skill §6/§6b, Olympia pattern): 16,275 condo-unit records (qualifier C####) are
mapped as unit-footprint stubs (median ~700 sqft) punched as HOLES out of the development's
"COMMON ELEMENTS" parcel, which carries $0 and is usually class 1 or 15F. Merged DOWN:
  1. Each unit's representative point is matched to the non-unit polygon whose hole-filled
     exterior contains it (88% of units -> 141 host parcels).
  2. Host geometry: rings that contain a unit are filled (the punched-out units), all other
     holes are kept (fee-simple lots), then unioned with the unit stubs.
  3. Values: units' land + improvement are SUMMED (distinct value records); the host's own
     value is added only if the host is taxable (a $0 or exempt common area adds nothing).
  4. Units with no host polygon (12%, e.g. office condos on Omni Dr, North Plainfield garden
     condos) are grouped by block/lot; groups of 3+ get the convex hull of their stubs minus
     any other parcel it overlaps, smaller groups keep the unioned stubs.
Common-element hosts are captured BEFORE the exempt/zero-value filters so they survive.

FARM QUALIFIERS (Q####) and billboards (B##): extra value records on an existing lot. Their
polygons are small pieces inside the base lot, so they are merged onto the base lot the same
way (values summed, geometry unioned). Without a base lot they stand alone.

ZERO-VALUE parcels (HOA open space, detention basins, road remnants with $0 land and $0
improvements) are dropped after the merges.

Outputs (one combined build, split into TWO CivicMapper places on 2026-11-23: Princeton is in
MERCER County, not Somerset, so it ships as its own city key instead of a 22nd Somerset town):
- data/jurisidictions/data/somerset/somerset-nj-parcels.parquet (+ dated snapshot): the 21
  Somerset County municipalities only (city key `somerset`)
- data/jurisidictions/data/princeton/princeton-nj-parcels.parquet (+ dated snapshot): Princeton
  (PCL_MUN 1114) only (city key `princeton`)
Princeton is still processed in the same run (same layer, same condo/additional-lot logic, and the
remnant test's nearest-neighbour medians can see across the Montgomery / Rocky Hill line); it is
split off only at export.
- viz/public/somerset-jurisdiction-overlay.geojson (21 Somerset towns)
"""
from __future__ import annotations

import io
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined, gis_area_sqft  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "somerset"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_CACHE = DATA_DIR / "somerset-nj-raw.parquet"
UNJOINED_CACHE = DATA_DIR / "somerset-nj-unjoined.parquet"
CADASTRAL_URL = "https://maps.nj.gov/arcgis/rest/services/Framework/Cadastral/MapServer/0/query"
UNJOINED_WHERE = "(PCL_MUN LIKE '18%' OR PCL_MUN='1114') AND LAND_VAL IS NULL"  # Somerset 1801-1821 + Princeton
OVERLAY = ROOT / "viz" / "public" / "somerset-jurisdiction-overlay.geojson"
PRINCETON_MUN = "1114"  # Mercer County; exported as its own city (princeton-nj-parcels.parquet)
PRINCETON_DIR = ROOT / "data" / "jurisidictions" / "data" / "princeton"

URL = ("https://services2.arcgis.com/XVOqAjTOJ5P6ngMu/arcgis/rest/services/"
       "Parcels_Composite_NJ_WM/FeatureServer/0/query")
WHERE = "COUNTY='SOMERSET' OR PCL_MUN='1114'"  # + Princeton (Mercer)
FIELDS = ("OBJECTID,PAMS_PIN,PCL_MUN,PCLBLOCK,PCLLOT,PCLQCODE,PROP_CLASS,MUN_NAME,PROP_LOC,"
          "LAND_VAL,IMPRVT_VAL,NET_VALUE,CALC_ACRE,BLDG_DESC,LAND_DESC,BLDG_CLASS,PROP_USE,"
          "FAC_NAME,YR_CONSTR,DWELL,COMM_DWELL,ADD_LOTS1,ADD_LOTS2,PCL_PBDATE")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh) Chrome/124 Safari/537.36"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
NJ_FT = 3424  # NAD83 / New Jersey (ftUS)

# PCL_MUN -> (display name, 2025 average ratio of assessed to true value, col 1(b) of the
# 2025 Final Equalization Table, County of Somerset, certified 3/10/2025).
MUNIS = {
    "1801": ("Bedminster", 1.0391), "1802": ("Bernards", 1.0149),
    "1803": ("Bernardsville", 1.0276), "1804": ("Bound Brook", 1.0161),
    "1805": ("Branchburg", 1.0179), "1806": ("Bridgewater", 0.9707),
    "1807": ("Far Hills", 1.0631), "1808": ("Franklin", 1.0429),
    "1809": ("Green Brook", 1.0211), "1810": ("Hillsborough", 0.9672),
    "1811": ("Manville", 1.0330), "1812": ("Millstone", 0.9442),
    "1813": ("Montgomery", 0.5997), "1814": ("North Plainfield", 0.5723),
    "1815": ("Peapack-Gladstone", 1.0364), "1816": ("Raritan", 0.6645),
    "1817": ("Rocky Hill", 1.0022), "1818": ("Somerville", 0.6346),
    "1819": ("South Bound Brook", 0.6623), "1820": ("Warren", 1.0525),
    "1821": ("Watchung", 1.0462),
    # Princeton is in MERCER County; its ratio is col 1(b) of the 2025 Final Equalization
    # Table, County of Mercer (https://www.nj.gov/njbonds/treasury/taxation/pdf/lpt/county_equalized/2025/2025Mercer.pdf).
    "1114": ("Princeton", 0.6650),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── 1. Fetch (cached) ────────────────────────────────────────────────────────
def _page(off):
    for attempt in range(8):
        try:
            r = requests.get(URL, params={
                "where": WHERE, "outFields": FIELDS, "returnGeometry": "true", "outSR": 4326,
                "resultOffset": off, "resultRecordCount": PAGE, "orderByFields": "OBJECTID",
                "f": "geojson"}, headers=HEADERS, timeout=300)
            r.raise_for_status()
            return gpd.read_file(io.BytesIO(r.content))
        except Exception as e:  # noqa: BLE001
            log(f"  retry {attempt + 1} @off {off}: {type(e).__name__}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"fetch failed at offset {off}")


def fetch():
    if RAW_CACHE.exists():
        log(f"Using cache: {RAW_CACHE.name}")
        return gpd.read_parquet(RAW_CACHE)
    total = requests.get(URL, params={"where": WHERE, "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=60).json()["count"]
    log(f"Pulling {total:,} parcels (8 parallel pages of {PAGE})...")
    with ThreadPoolExecutor(8) as ex:
        pages = list(ex.map(_page, range(0, total, PAGE)))
    gdf = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    assert gdf["OBJECTID"].is_unique and len(gdf) == total, (len(gdf), total)
    gdf.to_parquet(RAW_CACHE, index=False)
    log(f"  cached -> {RAW_CACHE.name}")
    return gdf


raw = fetch()
raw = raw.set_crs(4326) if raw.crs is None else raw.to_crs(4326)
raw["geometry"] = raw["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
raw = raw[raw.geometry.notnull() & ~raw.geometry.is_empty].reset_index(drop=True)
assert raw["PAMS_PIN"].is_unique, "PAMS_PIN not unique"
assert set(raw["PCL_MUN"]) <= set(MUNIS), set(raw["PCL_MUN"]) - set(MUNIS)
log(f"Raw parcels: {len(raw):,} (layer published "
    f"{pd.to_datetime(raw['PCL_PBDATE'], unit='ms').max():%Y-%m-%d})")

for c in ("LAND_VAL", "IMPRVT_VAL"):
    raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0).astype(float)
raw["cls"] = raw["PROP_CLASS"].fillna("").astype(str).str.strip().str.upper()
raw["qual"] = raw["PCLQCODE"].fillna("").astype(str).str.strip().str.upper()
raw["bl"] = (raw["PCL_MUN"] + "_" + raw["PCLBLOCK"].fillna("").str.strip() + "_"
             + raw["PCLLOT"].fillna("").str.strip())
raw["exempt_cls"] = raw["cls"].str.startswith("15")
log(f"classes: {raw['cls'].value_counts().to_dict()}")
log(f"qualifier prefixes: {raw['qual'].str[:1].replace('', '-').value_counts().to_dict()}")

# ── 2. Year check: taxable assessed totals per town (compare to equalization table) ──
tx = raw[~raw["exempt_cls"]]
tot = (tx["LAND_VAL"] + tx["IMPRVT_VAL"]).groupby(tx["PCL_MUN"]).sum()
log("taxable assessed total by town ($M; compare 2025 equalization table col 1(a)): "
    + ", ".join(f"{MUNIS[k][0]} {v / 1e6:,.0f}" for k, v in tot.items()))


# ── 2b. Additional lots: union unjoined polygons into their primary tax lot ───
# A MOD-IV record that covers several mapped lots sits on its PRIMARY lot only; the other
# lots are listed in ADD_LOTS1/2 ("L81,82", "L2-5", "37") and their polygons are absent from
# the composite layer (84% of Manville records, 9% of Franklin). Left alone, the whole
# property's value lands on one 25-ft lot: Manville showed a $88/sqft median against $20-30
# next door. The state's full cadastral layer keeps those polygons, unjoined (LAND_VAL null).
# Pull them, parse the lot lists, and union each listed lot into its primary — but only if it
# is the same block AND physically touches the growing primary footprint (within 5 ft).
def _norm_lot(s):
    s = str(s).strip().upper().lstrip("0") or "0"
    return s.rstrip("0").rstrip(".") if "." in s else s


def parse_add_lots(txt):
    """Lot numbers from an ADD_LOTS free-text field, or [] if it isn't a plain lot list."""
    t = str(txt or "").strip().upper()
    t = re.sub(r"\bC-?GAR\b", "", t).strip()
    if not re.fullmatch(r"(LOTS?|L)?[\s\d.,\-&]+", t):
        return []  # 'AKA BLK 5', 'END UNIT', 'PHASE 2', ... are notes, not lot lists
    t = re.sub(r"^(LOTS?|L)", "", t)
    lots = []
    for tok in re.split(r"[,&\s]+", t):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m and 0 < int(m.group(2)) - int(m.group(1)) <= 50:
            lots += [str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1)]
        elif re.fullmatch(r"\d+(\.\d+)?", tok):
            lots.append(tok)
    return [_norm_lot(x) for x in lots]


if not UNJOINED_CACHE.exists():
    def _upage(off):
        for attempt in range(8):
            try:
                r = requests.get(CADASTRAL_URL, params={
                    "where": UNJOINED_WHERE, "outFields": "OBJECTID,PAMS_PIN,PCL_MUN,PCLBLOCK,PCLLOT,PCLQCODE",
                    "returnGeometry": "true", "outSR": 4326, "resultOffset": off,
                    "resultRecordCount": 1000, "orderByFields": "OBJECTID", "f": "geojson"},
                    headers=HEADERS, timeout=300)
                r.raise_for_status()
                return gpd.read_file(io.BytesIO(r.content))
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt + 1} @off {off}: {e}")
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"unjoined fetch failed at {off}")
    n_unj = requests.get(CADASTRAL_URL, params={"where": UNJOINED_WHERE, "returnCountOnly": "true",
                                                "f": "json"}, headers=HEADERS, timeout=90).json()["count"]
    with ThreadPoolExecutor(6) as ex:
        _u = gpd.GeoDataFrame(pd.concat(list(ex.map(_upage, range(0, n_unj, 1000))),
                                        ignore_index=True), crs=4326)
    assert len(_u) == n_unj
    _u.to_parquet(UNJOINED_CACHE, index=False)
unj = gpd.read_parquet(UNJOINED_CACHE).to_crs(4326)
unj = unj[unj["PCLQCODE"].fillna("").str.strip().eq("")].copy()  # plain lots only
unj["geometry"] = unj.geometry.apply(lambda x: x if x.is_valid else x.buffer(0))
unj["key"] = (unj["PCL_MUN"] + "_" + unj["PCLBLOCK"].fillna("").str.strip() + "_"
              + unj["PCLLOT"].map(_norm_lot))
unj = unj.drop_duplicates("key")
unj_by_key = dict(zip(unj["key"], unj.index))
unj_ft = unj.to_crs(NJ_FT).geometry
log(f"Unjoined (no MOD-IV record) plain-lot polygons in the cadastral layer: {len(unj):,}")

raw_ft = raw.to_crs(NJ_FT).geometry
claimed, n_rec = set(), 0
for i, row in raw[raw["qual"].eq("")].iterrows():
    lots = parse_add_lots(row["ADD_LOTS1"]) + parse_add_lots(row["ADD_LOTS2"])
    if not lots:
        continue
    blk = f"{row['PCL_MUN']}_{str(row['PCLBLOCK']).strip()}_"
    grow_ft, parts = raw_ft[i], []
    for lot in dict.fromkeys(lots):
        u = unj_by_key.get(blk + lot)
        if u is None or u in claimed:
            continue
        if unj_ft[u].distance(grow_ft) <= 5:
            claimed.add(u)
            parts.append(u)
            grow_ft = grow_ft.union(unj_ft[u])
    if parts:
        n_rec += 1
        raw.at[i, "geometry"] = unary_union([raw.at[i, "geometry"]] + [unj.geometry[u] for u in parts]).buffer(0)
left = unj.drop(index=list(claimed))
log(f"  additional lots unioned into their primary: {len(claimed):,} polygons onto {n_rec:,} records; "
    f"unclaimed: {len(left):,} {left['PCL_MUN'].map(lambda m: MUNIS[m][0]).value_counts().head(6).to_dict()}")


def fill_rings(geom, keep_ring):
    """Rebuild polygon(s) keeping only interior rings for which keep_ring(ring) is True."""
    parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    out = [Polygon(p.exterior, [r for r in p.interiors if keep_ring(r)]) for p in parts]
    return out[0] if len(out) == 1 else MultiPolygon(out)


def exterior_only(geom):
    return fill_rings(geom, lambda r: False)


# ── 2c. Fee-simple townhouse lots punched out of a common-element parcel ──────
# Same structure as condos without the C qualifier: class-2 townhouse lots drawn as the
# unit footprint (often < 1,000 sqft) in holes of a $0 "COMMON ELEMENTS" parcel (Franklin's
# Enclave Circle, much of Bridgewater). Left alone they render as pencils at ~$216/sqft
# against a ~$19 town median. Rule (Olympia §6b, per development, not per lot): a $0 /
# exempt / class-1 parcel with holes is a common host; if the class-2 lots under 3,000 sqft
# in its holes have a median footprint under 2,000 sqft, it is a townhouse development and
# all of those lots merge onto it with the condo units below.
ft_area = raw.to_crs(NJ_FT).geometry.area
_nh = raw.geometry.apply(lambda g: sum(len(p.interiors) for p in (
    g.geoms if g.geom_type == "MultiPolygon" else [g])))
_common = ((_nh > 0) & raw["qual"].eq("")
           & (((raw["LAND_VAL"] + raw["IMPRVT_VAL"]) <= 0) | raw["exempt_cls"] | raw["cls"].eq("1")))
_ch = gpd.GeoDataFrame({"host": raw.index[_common]},
                       geometry=raw.geometry[_common].apply(exterior_only).values, crs=4326)
_cand = raw[raw["qual"].eq("") & raw["cls"].eq("2") & (ft_area < 3000)]
_cp = gpd.GeoDataFrame({"i": _cand.index.values},
                       geometry=_cand.geometry.representative_point().values, crs=4326)
_j = gpd.sjoin(_cp, _ch, predicate="within")
_j["harea"] = _j["host"].map(lambda h: raw.geometry[h].area)
_j = _j.sort_values("harea").drop_duplicates("i")
_j["fp"] = ft_area[_j["i"]].values
_dev = _j.groupby("host")["fp"].median()
_th_hosts = _dev[_dev < 2000].index
raw["townhouse_unit"] = raw.index.isin(_j.loc[_j["host"].isin(_th_hosts), "i"])
log(f"Townhouse lots in common-element developments: {int(raw['townhouse_unit'].sum()):,} "
    f"in {len(_th_hosts)} developments ({_j['host'].nunique() - len(_th_hosts)} hosts with "
    f"larger lots left as separate parcels)")

# ── 3. Condo units -> merge DOWN onto their common-element host ───────────────
is_unit = raw["qual"].str.startswith("C") | raw["townhouse_unit"]
units = raw[is_unit].copy()
others = raw[~is_unit].copy()
log(f"Condo unit + townhouse records: {len(units):,} in {units['bl'].nunique():,} block/lots")

nholes = others.geometry.apply(lambda g: sum(len(p.interiors) for p in (
    g.geoms if g.geom_type == "MultiPolygon" else [g])))
hostcand = others[nholes > 0]
hf = gpd.GeoDataFrame({"host": hostcand.index.values},
                      geometry=hostcand.geometry.apply(exterior_only).values, crs=4326)
upts = gpd.GeoDataFrame({"unit": units.index.values},
                        geometry=units.geometry.representative_point().values, crs=4326)
j = gpd.sjoin(upts, hf, predicate="within", how="left")
# nested hosts: keep the smallest containing host
j["harea"] = j["host"].map(lambda h: np.nan if pd.isna(h) else others.geometry[int(h)].area)
j = j.sort_values("harea").drop_duplicates("unit").set_index("unit")
units["host"] = j["host"].reindex(units.index)
matched = units[units["host"].notna()].copy()
orphans = units[units["host"].isna()].copy()
matched["host"] = matched["host"].astype(int)
log(f"  units in a host common parcel: {len(matched):,} ({100 * len(matched) / len(units):.0f}%) "
    f"-> {matched['host'].nunique()} hosts; orphan units: {len(orphans):,}")


def unit_category(classes: pd.Series) -> str:
    top = classes.value_counts().idxmax()
    return {"2": "Condo / Townhouse Development", "4A": "Commercial", "4B": "Industrial",
            "4C": "Apartments (5+ units)"}.get(top, "Condo / Townhouse Development")


merged_rows = []
for h, grp in matched.groupby("host"):
    host = others.loc[h]
    pts = list(grp.geometry.representative_point())
    geom = fill_rings(host.geometry, lambda r: not any(Polygon(r).contains(p) for p in pts))
    geom = unary_union([geom] + list(grp.geometry)).buffer(0)
    host_taxable = (not host["exempt_cls"]) and host["cls"] not in ("5A", "5B")
    land = grp["LAND_VAL"].sum() + (host["LAND_VAL"] if host_taxable else 0)
    impr = grp["IMPRVT_VAL"].sum() + (host["IMPRVT_VAL"] if host_taxable else 0)
    host_has_value = host_taxable and host["LAND_VAL"] + host["IMPRVT_VAL"] > 0
    cat = (None if not host_has_value or host["cls"] in ("1", "15") else host["cls"])
    merged_rows.append({
        "PAMS_PIN": host["PAMS_PIN"], "PCL_MUN": host["PCL_MUN"], "bl": host["bl"],
        "cls": cat or ("CONDO", grp["cls"]), "qual": "", "LAND_VAL": land, "IMPRVT_VAL": impr,
        "PROP_LOC": grp["PROP_LOC"].iloc[0] if not host_has_value else host["PROP_LOC"],
        "DWELL": np.nan, "BLDG_DESC": host["BLDG_DESC"], "exempt_cls": False,
        "condo_units": len(grp), "geometry": geom})
# orphan units: cluster stubs that sit within 75 ft of each other in the same town + block
# (one development is usually split across several block/lots, so grouping by lot alone
# leaves a forest of 1-2 unit "pencils").
other_sindex = others.sindex
orph_ft = orphans.to_crs(NJ_FT)
orphans["cluster"] = None
for (mun, blk), grp in orph_ft.groupby([orph_ft["PCL_MUN"], orph_ft["PCLBLOCK"].fillna("").str.strip()]):
    blobs = unary_union(list(grp.geometry.buffer(75)))
    blobs = list(blobs.geoms) if blobs.geom_type == "MultiPolygon" else [blobs]
    rp = grp.geometry.representative_point()
    for k, blob in enumerate(blobs):
        orphans.loc[rp[rp.within(blob)].index, "cluster"] = f"{mun}_{blk}_{k}"
assert orphans["cluster"].notna().all()
log(f"  orphan units clustered into {orphans['cluster'].nunique():,} developments")
for cl, grp in orphans.groupby("cluster"):
    bl = grp["bl"].iloc[0]
    stubs = unary_union(list(grp.geometry)).buffer(0)
    geom = stubs
    if len(grp) >= 3:
        hull = stubs.convex_hull
        hit = others.iloc[other_sindex.query(hull, predicate="intersects")]
        if len(hit):
            hull = hull.difference(unary_union(list(hit.geometry)))
        geom = unary_union([hull, stubs]).buffer(0)
    merged_rows.append({
        "PAMS_PIN": "CONDO_" + cl, "PCL_MUN": grp["PCL_MUN"].iloc[0], "bl": bl,
        "cls": ("CONDO", grp["cls"]), "qual": "", "LAND_VAL": grp["LAND_VAL"].sum(),
        "IMPRVT_VAL": grp["IMPRVT_VAL"].sum(), "PROP_LOC": grp["PROP_LOC"].iloc[0],
        "DWELL": np.nan, "BLDG_DESC": grp["BLDG_DESC"].iloc[0], "exempt_cls": False,
        "condo_units": len(grp), "geometry": geom})
merged = gpd.GeoDataFrame(merged_rows, geometry="geometry", crs=4326)
merged["condo_cat"] = merged["cls"].apply(lambda c: unit_category(c[1]) if isinstance(c, tuple) else None)
merged["cls"] = merged["cls"].apply(lambda c: "CONDO" if isinstance(c, tuple) else c)
log(f"  condo developments built: {len(merged):,} (from {len(units):,} unit records); "
    f"unit land ${units['LAND_VAL'].sum():,.0f} -> merged land ${merged['LAND_VAL'].sum():,.0f}")
others = others.drop(index=matched["host"].unique())

# ── 4. Farm qualifiers / billboards -> merge onto their base lot ─────────────
is_extra = others["qual"].ne("")
extra = others[is_extra & ~others["exempt_cls"]]
base = others[~is_extra]
base_by_bl = pd.Series(base.index, index=base["bl"])
base_by_bl = base_by_bl[~base_by_bl.index.duplicated()]
has_base = extra["bl"].isin(base_by_bl.index)
log(f"Extra-qualifier records (taxable): {len(extra):,}; with a base lot: {int(has_base.sum()):,}")
for bl, grp in extra[has_base].groupby("bl"):
    b = base_by_bl[bl]
    others.at[b, "LAND_VAL"] += grp["LAND_VAL"].sum()
    others.at[b, "IMPRVT_VAL"] += grp["IMPRVT_VAL"].sum()
    others.at[b, "geometry"] = unary_union([others.at[b, "geometry"]] + list(grp.geometry)).buffer(0)
    if others.at[b, "cls"] in ("3A", "3B") or grp["cls"].isin(["3A", "3B"]).any():
        others.at[b, "farm_merged"] = True
others = others.drop(index=extra[has_base].index)
others["condo_units"] = 0
parcel = gpd.GeoDataFrame(pd.concat([others, merged], ignore_index=True), geometry="geometry",
                          crs=4326)
assert parcel["PAMS_PIN"].is_unique

# ── 5. Exempt / railroad / zero-value drops ──────────────────────────────────
parcel["exemption_flag"] = parcel["exempt_cls"].fillna(False).astype(int)
parcel["gis_area_sqft"] = parcel.geometry.apply(gis_area_sqft)
ex = parcel[parcel["exemption_flag"] == 1]
log(f"Exempt (class 15A-F): {len(ex):,} parcels, {ex['gis_area_sqft'].sum() / SQFT_PER_ACRE:,.0f} ac, "
    f"assessed land ${ex['LAND_VAL'].sum():,.0f}")
for c, g in ex.groupby("cls"):
    log(f"    {c:4s} {len(g):6,} parcels {g['gis_area_sqft'].sum() / SQFT_PER_ACRE:9,.0f} ac")
rr = parcel["cls"].isin(["5A", "5B"])
log(f"Railroad (5A/5B) dropped: {int(rr.sum())}")
parcel = parcel[(parcel["exemption_flag"] == 0) & ~rr].copy()
zero = (parcel["LAND_VAL"] + parcel["IMPRVT_VAL"]) <= 0
log(f"Zero-value parcels dropped: {int(zero.sum()):,} "
    f"({parcel.loc[zero, 'gis_area_sqft'].sum() / SQFT_PER_ACRE:,.0f} ac); classes "
    f"{parcel.loc[zero, 'cls'].value_counts().to_dict()}")
parcel = parcel[~zero].copy()

# ── 6. Equalize to market value ──────────────────────────────────────────────
parcel["jurisdiction"] = parcel["PCL_MUN"].map(lambda m: MUNIS[m][0])
parcel["assessment_ratio"] = parcel["PCL_MUN"].map(lambda m: MUNIS[m][1])
farm_q = parcel["cls"].eq("3B")
land_ratio = np.where(farm_q, 1.0, parcel["assessment_ratio"])
parcel["assessed_land_value"] = parcel["LAND_VAL"]
parcel["assessed_improvement_value"] = parcel["IMPRVT_VAL"]
parcel["land_value"] = (parcel["LAND_VAL"] / land_ratio).round(0)
parcel["improvement_value"] = (parcel["IMPRVT_VAL"] / parcel["assessment_ratio"]).round(0)
parcel["full_market_value"] = parcel["land_value"] + parcel["improvement_value"]
log(f"Equalized: assessed land ${parcel['LAND_VAL'].sum():,.0f} -> market land "
    f"${parcel['land_value'].sum():,.0f}; total ${parcel['full_market_value'].sum():,.0f}")

# ── 7. Categories ────────────────────────────────────────────────────────────
OPEN_SPACE = r"OPEN SPACE|COMMON|DETENTION|RETENTION|DRAINAGE|BASIN|BUFFER|CONSERVATION"


def categorize(row):
    c = row["cls"]
    if c == "CONDO":
        return row["condo_cat"]
    if c == "1":
        txt = f"{row.get('BLDG_DESC') or ''} {row.get('PROP_LOC') or ''}".upper()
        return "Open Space / Common Area" if re.search(OPEN_SPACE, txt) else "Vacant Land"
    if c == "2":
        d = row.get("DWELL")
        return "Multifamily (2-4 units)" if pd.notna(d) and d >= 2 else "Single Family"
    if c == "3A" or bool(row.get("farm_merged") == True):  # noqa: E712
        return "Farm (Homestead)"
    if c == "3B":
        return "Farmland (Qualified)"
    return {"4A": "Commercial", "4B": "Industrial", "4C": "Apartments (5+ units)"}.get(c, "Other")


parcel["property_land_use_category"] = parcel.apply(categorize, axis=1)
parcel["nj_property_class"] = parcel["cls"].replace({"CONDO": "2 (condo units)"})

# ── 8. Canonical fields ──────────────────────────────────────────────────────
parcel["geometry"] = parcel["geometry"].apply(lambda x: x if x.is_valid else x.buffer(0))
parcel["land_area_sqft"] = parcel.geometry.apply(gis_area_sqft)
parcel.loc[parcel["land_area_sqft"] < 1, "land_area_sqft"] = np.nan
parcel["area_source"] = "gis"
parcel["land_area_acres"] = parcel["land_area_sqft"] / SQFT_PER_ACRE
parcel["likely_remnant"] = (parcel["land_area_sqft"] < 500).astype(int)
den = parcel["land_area_sqft"].replace(0, np.nan)
parcel["full_market_value_per_sqft"] = parcel["full_market_value"] / den
parcel["land_value_per_sqft"] = parcel["land_value"] / den
parcel["improvement_value_per_sqft"] = parcel["improvement_value"] / den

# Orphan condo clusters (no common-element parcel) whose ground is simply not mapped carry the
# development's whole land value on the unit boxes, e.g. Princeton's Palmer Square: 46 units,
# $15.8M land on 0.11 acres = ~$3,300/sqft. There is no real footprint to put it on, so flag
# them like slivers (hidden via hideRemnants, dropped from hexes via --drop-remnants) when their
# land $/sqft is > 5x the median of the 15 nearest ordinary parcels.
_orph = parcel["PAMS_PIN"].astype(str).str.startswith("CONDO_")
_ord = parcel[(parcel["condo_units"] == 0) & parcel["land_value_per_sqft"].gt(0)]
_ordpt = _ord.to_crs(NJ_FT).geometry.representative_point()
from scipy.spatial import cKDTree  # noqa: E402
_tree = cKDTree(np.c_[_ordpt.x, _ordpt.y])
_op = parcel[_orph].to_crs(NJ_FT).geometry.representative_point()
_, _nn = _tree.query(np.c_[_op.x, _op.y], k=15)
_nbr = np.median(_ord["land_value_per_sqft"].to_numpy()[_nn], axis=1)
_spike = parcel.loc[_orph, "land_value_per_sqft"].to_numpy() > 5 * _nbr
parcel["condo_footprint_unmapped"] = 0
parcel.loc[parcel.index[_orph][_spike], ["likely_remnant", "condo_footprint_unmapped"]] = 1
_fl = parcel[parcel["condo_footprint_unmapped"] == 1]
log(f"Orphan condo clusters flagged as unmapped-footprint (hidden): {len(_fl)} of {int(_orph.sum())} "
    f"({int(_fl['condo_units'].sum()):,} units, land ${_fl['land_value'].sum():,.0f})")
parcel = add_improvement_ratio_fields(parcel, land_col="land_value", improvement_col="improvement_value")

parcel["property_land_use_refined"] = classify_property_refined(
    parcel, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Farmland (Qualified)", "Farm (Homestead)", "Open Space / Common Area",
                        "Other"),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    state_class_col="__none__", fetch_footprints=False)
# Farmland-assessed land is a use value; don't let the ratio test call a farm "Underdeveloped".
farmish = parcel["property_land_use_category"].isin(["Farmland (Qualified)", "Farm (Homestead)",
                                                      "Open Space / Common Area"])
parcel.loc[farmish, "property_land_use_refined"] = None

parcel["link"] = np.nan  # no stable public per-parcel deep link
parcel["parcel_number"] = parcel["PAMS_PIN"].astype(str)
parcel["address"] = parcel["PROP_LOC"].fillna("").astype(str).str.strip().replace("", np.nan)
parcel["assessment_ratio_pct"] = (parcel["assessment_ratio"] * 100).round(2)

# ── 9. Export ────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "jurisdiction", "exemption_flag", "property_land_use_category",
           "property_land_use_refined", "full_market_value", "full_market_value_per_sqft",
           "land_value", "land_value_per_sqft", "improvement_value", "improvement_value_per_sqft",
           "TLLDIMPROV", "IMPR_LAND_RATIO", "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link",
           "land_area_acres", "area_source", "likely_remnant", "parcel_number", "address",
           "nj_property_class", "assessed_land_value", "assessed_improvement_value",
           "assessment_ratio_pct", "condo_units"]
final = parcel[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")
final = final[final.geometry.notnull() & ~final.geometry.is_empty].copy()
assert final["parcel_number"].duplicated().sum() == 0
# Split: Princeton (Mercer) becomes its own city; the Somerset file is the 21 Somerset towns only.
is_pr = final["jurisdiction"].eq(MUNIS[PRINCETON_MUN][0])
PRINCETON_DIR.mkdir(parents=True, exist_ok=True)
stamp = f"{datetime.now():%Y_%m_%d}"
for _dir, _slug, _part in ((DATA_DIR, "somerset-nj", final[~is_pr]), (PRINCETON_DIR, "princeton-nj", final[is_pr])):
    _out = _dir / f"{_slug}-parcels.parquet"
    _part.to_parquet(_out, index=False)
    _part.to_parquet(_dir / f"{_slug}-parcels_{stamp}.parquet", index=False)
    log(f"SAVED {_out} | rows {len(_part):,} | land ${_part.current_full_land_value.sum():,.0f}")
out = DATA_DIR / "somerset-nj-parcels.parquet"
log(f"combined (log stats below cover both) | rows {len(final):,}")
log(f"land ${final.current_full_land_value.sum():,.0f} over {final.land_area_acres.sum():,.0f} acres"
    f" | impr ${final.improvement_value.sum():,.0f} | total ${final.full_market_value.sum():,.0f}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"per-town: {final['jurisdiction'].value_counts().to_dict()}")

# ── 10. Town overlay ─────────────────────────────────────────────────────────
src = raw.copy()
src = src[src["PCL_MUN"] != PRINCETON_MUN]  # Princeton is its own city; no Somerset overlay ring
src["name"] = src["PCL_MUN"].map(lambda m: MUNIS[m][0])
diss = src[["name", "geometry"]].dissolve(by="name").reset_index()
diss["geometry"] = diss.geometry.buffer(0.00002).buffer(-0.00002).simplify(0.00015)
diss.to_file(OVERLAY, driver="GeoJSON")
log(f"SAVED {OVERLAY.relative_to(ROOT)} ({len(diss)} towns)")

# ── 11. Smoke alarms (skill §6a) ─────────────────────────────────────────────
log("--- smoke alarms ---")
gp = final.to_crs(NJ_FT)
a = gp.geometry.area
lv = pd.to_numeric(final["land_value_per_sqft"], errors="coerce")
log(f"  footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
log(f"  sub-500 / sub-1000 sqft with value: {int(((a < 500) & (final.full_market_value > 0)).sum()):,}"
    f" / {int(((a < 1000) & (final.full_market_value > 0)).sum()):,}")
log(f"  land $/sqft p50/p95/p99/max: ${lv.median():,.2f} / ${lv.quantile(.95):,.2f} / "
    f"${lv.quantile(.99):,.2f} / ${lv.max():,.2f}")
rp = final.geometry.representative_point()
vc = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)).value_counts()
log(f"  stacked clusters: {int((vc > 1).sum())}, max stack: {int(vc.max())}")
holes = final.geometry.apply(lambda g: sum(len(p.interiors) for p in (
    g.geoms if g.geom_type == "MultiPolygon" else [g])))
log(f"  parcels with holes: {int((holes > 0).sum()):,}")
cd = final[final["condo_units"] > 0]
log(f"  condo developments land $/sqft p10/p50/p90: "
    f"{[round(x, 2) for x in cd['land_value_per_sqft'].quantile([.1, .5, .9])]} "
    f"(all parcels p50 ${lv.median():.2f})")
log(f"  per-town median land $/sqft: "
    + ", ".join(f"{k} ${v:.1f}" for k, v in final.groupby('jurisdiction')['land_value_per_sqft'].median().sort_values().items()))
log(f"  bounds: {[round(v, 4) for v in final.total_bounds]}")
log("DONE")
