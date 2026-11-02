#!/usr/bin/env python3
"""
Build the City of Harrisonburg, VA canonical parcel parquet.

Harrisonburg is a Virginia INDEPENDENT CITY (FIPS 51660), not part of the surrounding
Rockingham County. Every source below is city-only, so no county clip is needed.

Geometry and values live in different places (skill §1, "geometry + separate appraisal
export"). The city's GIS layers carry owner/zoning/tax-map number but NO dollar values; the
values are published as the assessor's bulk reassessment book (a PDF), joined on the tax-map
number. Per-parcel value pages (the Vision/VGSI site) are NOT scraped.

Sources (both public, no token, no login):
- Geometry: City of Harrisonburg GIS, CityServices/MyParcelTaxData FeatureServer layer 0
  ("Parcel Boundaries"), ~13.6k polygons:
  https://gis.harrisonburgva.gov/arcgis/rest/services/CityServices/MyParcelTaxData/FeatureServer/0
  Fields used: TaxMapNum (join key, e.g. "001 A 13"), usecode (ZONING district, e.g. R1/B2/M1 —
  despite the name it is not a land-use code), hbrislink (per-parcel Vision/VGSI page), SQFeet.
- Values: "2026 Reassessment Book (proposed)", City of Harrisonburg Real Estate Office,
  linked from https://www.harrisonburgva.gov/real-estate :
  https://www.harrisonburgva.gov/sites/default/files/revenue/2026%20Reassessment%20Book%20(proposed).pdf
  1,273 pages, sworn by the City Assessor 2025-10-27. It is the "REGULAR" (taxable) roll of the
  2026 general reassessment at 100% fair market value: per record the map designation, the
  Virginia land-book class code (1 single family urban / 3 multiple-unit residential /
  4 commercial & industrial), LAND value, BUILDING & IMPROVEMENT value and TOTAL.
  Parsed with `pdftotext -layout` (poppler-utils; `brew install poppler` / `apt install
  poppler-utils`). The parser is checked against the book's own DISTRICT TOTALS page and the
  run aborts if the count or the land/improvement sums don't reconcile to the dollar.

Outputs:
- data/jurisidictions/data/harrisonburg/harrisonburg-va-parcels.parquet
- data/jurisidictions/data/harrisonburg/harrisonburg-va-parcels_YYYY_MM_DD.parquet

Notes / caveats (see also the notes in viz/src/cities/harrisonburg.json):
- Roll year: 2026 general reassessment, PROPOSED values (effective 2026-01-01, published before
  appeals). Board-of-Equalization changes after publication are not reflected.
- Exempt parcels: the REGULAR book lists only taxable real estate, so exempt land (JMU, Eastern
  Mennonite, city/county/state/federal, churches, schools) is simply absent from the value side
  and drops out of the inner join. State-assessed public-service-corporation property is also
  not in the book. No owner-name heuristic is needed or used.
- Taxable LEASEHOLDS on exempt land (map numbers ending in "T", e.g. "025 P 2T" = a $27M
  building on JMU land) have no polygon of their own in GIS. They are left off (logged below)
  rather than painted onto the whole exempt parent parcel.
- Land-use (agricultural/open-space deferral) parcels: the book's LAND column is the full fair
  market land value; the deferred land-use value is printed separately and the book's TOTAL for
  those ~33 parcels is land-use value + building. This ETL uses FAIR MARKET land + building for
  full_market_value so all parcels are on the same basis.
- Classification: the book's class code is coarse (3 classes) and has no vacancy or parking
  code, so: improvement value == 0 -> Vacant Land; class 1 -> Single Family; class 3 ->
  Multifamily; class 4 -> Industrial when zoned M1/M1C, otherwise Commercial. Surface parking
  is tagged afterwards from the detected parking lots (data/scripts/apply_parking_category.py).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
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
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined, gis_area_sqft  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "harrisonburg"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "harrisonburg-va-geometry.parquet"
BOOK_PDF = DATA_DIR / "harrisonburg-2026-reassessment-book.pdf"
BOOK_TXT = DATA_DIR / "harrisonburg-2026-reassessment-book.txt"

PARCELS_URL = ("https://gis.harrisonburgva.gov/arcgis/rest/services/CityServices/"
               "MyParcelTaxData/FeatureServer/0/query")
BOOK_URL = ("https://www.harrisonburgva.gov/sites/default/files/revenue/"
            "2026%20Reassessment%20Book%20(proposed).pdf")
OUT_FIELDS = "OBJECTID,PARCELID,TaxMapNum,usecode,statecode,hbrislink,SQFeet,PID"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
UTM = "EPSG:32617"  # UTM 17N (Harrisonburg ~78.87W)

# The book's own DISTRICT TOTALS page (page 1,273). The parser must reproduce these exactly.
BOOK_TOTAL_COUNT = 12_714
BOOK_TOTAL_LAND = 1_561_980_600
BOOK_TOTAL_IMPR = 4_852_215_700

INDUSTRIAL_ZONES = {"M1", "M1C"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def norm_key(s) -> str:
    """Tax-map numbers are space-padded differently in GIS ("001 B  1 2") and the book
    ("001 B 1 2"), so runs of spaces are collapsed to one. Spaces must NOT be stripped
    entirely: "033 R 1 2" (lots 1 and 2) and "033 R 12" (lot 12) are different parcels."""
    return re.sub(r"\s+", " ", str(s or "")).strip().upper()


# ── 1. values: the 2026 reassessment book ──────────────────────────────────────
def fetch_book_text() -> str:
    if not BOOK_TXT.exists():
        if not BOOK_PDF.exists():
            log(f"Downloading reassessment book -> {BOOK_PDF.name}")
            r = requests.get(BOOK_URL, headers=HEADERS, timeout=300)
            r.raise_for_status()
            BOOK_PDF.write_bytes(r.content)
        if not shutil.which("pdftotext"):
            raise SystemExit("pdftotext not found — install poppler (brew install poppler / "
                             "apt install poppler-utils)")
        subprocess.run(["pdftotext", "-layout", str(BOOK_PDF), str(BOOK_TXT)], check=True)
    return BOOK_TXT.read_text(encoding="utf-8", errors="replace")


# One record per block between the 120-dash rules. The first line holds owner | map
# designation | class code | land | building & improvements | total. Owner names longer than
# the 45-char column run straight into the map designation ("...STAL036 E 4"), so the map
# designation is anchored from the RIGHT: "<3-digit map> <1-2 letter block>[lot]" then lot
# tokens separated by single spaces, then 2+ spaces and the 1-digit class code.
REC = re.compile(r"(\d{3} [A-Z]{1,2}(?:\d\S*)?(?: \S+)*?)\s{2,}(\d)\s+(-?[\d,]+)\s+(-?[\d,]+)"
                 r"\s+(-?[\d,]+)\s*$", re.M)
LANDUSE_LINE = re.compile(r"\s{2,}([\d,]+)\s*$")


def parse_book(text: str) -> pd.DataFrame:
    rows = []
    for blk in re.split(r"\n-{100,}\n", text):
        lines = blk.splitlines()
        hits = [(i, REC.search(l)) for i, l in enumerate(lines) if "TOTALS" not in l]
        hits = [(i, m) for i, m in hits if m]
        if not hits:
            continue
        if len(hits) > 1:
            raise RuntimeError(f"block with {len(hits)} records:\n{blk[:500]}")
        i, m = hits[0]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        land, bld, tot = (int(m.group(k).replace(",", "")) for k in (3, 4, 5))
        # Land-use (deferral) parcels print the land-use value at the end of the street line.
        lu = LANDUSE_LINE.search(nxt)
        zoning = lines[i + 2][45:].strip() if i + 2 < len(lines) else ""
        rows.append({"map": m.group(1).strip(), "book_class": m.group(2), "land_val": land,
                     "bld_val": bld, "book_total": tot,
                     "landuse_val": int(lu.group(1).replace(",", "")) if lu else np.nan,
                     "book_zoning": zoning.split()[0] if zoning else ""})
    return pd.DataFrame(rows)


book = parse_book(fetch_book_text())
log(f"Book records parsed: {len(book):,}  land ${book.land_val.sum():,}  impr ${book.bld_val.sum():,}")
assert len(book) == BOOK_TOTAL_COUNT, f"parsed {len(book)} records, book says {BOOK_TOTAL_COUNT}"
assert book.land_val.sum() == BOOK_TOTAL_LAND, "land total does not reconcile with DISTRICT TOTALS"
assert book.bld_val.sum() == BOOK_TOTAL_IMPR, "improvement total does not reconcile"
book["k"] = book["map"].map(norm_key)
assert not book["k"].duplicated().any(), "duplicate map designations in book"
is_lu = book["land_val"] + book["bld_val"] != book["book_total"]
log(f"  land-use deferral records (TOTAL = land-use value + bldg): {int(is_lu.sum())} — "
    f"fair-market land used for these")
log(f"  class counts: {book.book_class.value_counts().to_dict()}")


# ── 2. geometry ────────────────────────────────────────────────────────────────
def fetch_parcels():
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    total = requests.get(PARCELS_URL, params={"where": "1=1", "returnCountOnly": "true",
                         "f": "json"}, headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} Harrisonburg parcels (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(5):
            try:
                r = requests.get(PARCELS_URL, params={
                    "where": "1=1", "outFields": OUT_FIELDS, "returnGeometry": "true",
                    "resultOffset": off, "resultRecordCount": PAGE, "outSR": 4326,
                    "orderByFields": "OBJECTID", "f": "geojson",
                }, headers=HEADERS, timeout=240)
                r.raise_for_status()
                feats = json.loads(r.content).get("features", [])
                gdf = (gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
                       if feats else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
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
        if len(gdf) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


raw = fetch_parcels()
raw["k"] = raw["TaxMapNum"].map(norm_key)
raw = raw[raw["k"].ne("") & raw["k"].ne("NONE")].copy()
raw["geometry"] = raw["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
raw = raw[raw["geometry"].notnull() & ~raw["geometry"].is_empty].copy()
log(f"GIS parcels with a tax-map number + valid geometry: {len(raw):,}")

# Preflight (skill §2a): one tax-map number on several polygon rows -> UNION them, never keep one.
d = raw[raw["k"].duplicated(keep=False)]
log(f"Keys with >1 geometry row: {d['k'].nunique():,} ({len(d):,} rows) -> unary_union")
first_cols = [c for c in raw.columns if c not in ("geometry", "k")]
geo = raw.groupby("k", dropna=False).agg({c: "first" for c in first_cols}).reset_index()
geo["geometry"] = raw.groupby("k")["geometry"].apply(lambda gs: unary_union(list(gs))).values
geo = gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326")
assert not geo["k"].duplicated().any()

# ── 3. join (1:1 on the tax-map number; both sides unique, so no broadcast) ────
parcel = geo.merge(book, on="k", how="inner", validate="one_to_one")
parcel = gpd.GeoDataFrame(parcel, geometry="geometry", crs="EPSG:4326")
unmatched = book[~book["k"].isin(geo["k"])]
lease = unmatched["map"].str.upper().str.endswith("T")
log(f"Joined: {len(parcel):,} of {len(book):,} book records have a polygon")
log(f"  unmatched book records: {len(unmatched)} — {int(lease.sum())} taxable LEASEHOLDS on exempt "
    f"land (map # ends in T; land ${unmatched[lease].land_val.sum():,} / impr "
    f"${unmatched[lease].bld_val.sum():,}), {int((~lease).sum())} other (recent splits/combos: "
    f"{unmatched[~lease]['map'].tolist()})")
log(f"  GIS polygons with no taxable book record (exempt / common area / state-assessed): "
    f"{len(geo) - len(parcel):,}")
assert parcel["land_val"].sum() + unmatched["land_val"].sum() == BOOK_TOTAL_LAND

parcel["zoning"] = parcel["usecode"].fillna(parcel["book_zoning"]).astype(str).str.strip()

# ── 3b. condo units -> merge DOWN onto the development's common-area land (skill §6b) ──
# Harrisonburg maps each condo UNIT as its own building-slice polygon (e.g. "004 B 1366-A".."-L",
# twelve ~474 sqft slices of one building) carrying that unit's land share, and garden/stacked
# condos as IDENTICAL polygons per floor ("031 B 1010-1/-5/-9"). The development's real land is a
# separate common-area polygon ("UNIT OWNERS ASSOCIATION", "004 B 3", "081 E 1" ...) with the
# unit footprints punched out as holes; it is not on the taxable roll, so the inner join above
# dropped it. Left alone that renders as a forest of $60-260/sqft pencils floating over a hole.
# Recipe (run_lynchburg.py / run_olympia.py): unit candidates = footprint < UNIT_SQFT or stacked
# on another valued footprint; group candidates that touch; attach a non-roll polygon when at
# least MIN_UNITS_ON_CA candidates touch it; each group with a common area, or of >= 2 touching
# or stacked units, becomes ONE parcel: values SUMMED (distinct per-unit records, not a
# broadcast — asserted), footprint = union with holes filled, minus any valued parcel that still
# sits in a hole (a fee-simple lot that was not merged).
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402

UNIT_SQFT = 1000.0
MIN_UNITS_ON_CA = 3


def _fill_holes(g):
    if g is None or g.is_empty:
        return g
    if g.geom_type == "Polygon":
        return Polygon(g.exterior)
    if g.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in g.geoms])
    return g


pu = parcel.to_crs(UTM)
pu["_sqft"] = pu.geometry.area * 10.763910416709722
_rp = pu.geometry.representative_point()
pu["_rk"] = _rp.x.round(1).astype(str) + "," + _rp.y.round(1).astype(str)
_stacked = pu["_rk"].duplicated(keep=False)
nonroll = geo[~geo["k"].isin(parcel["k"])].to_crs(UTM)
# Office/retail CONDO units (class 4) are bigger than residential stubs (1,100-4,100 sqft) but the
# same structure: building footprints punched out of a common-area polygon that is off the roll
# ("1880 RESERVOIR ST CONDO A", "PRO POINTE PROPERTY ASSOC", "VALLEY PROFESSIONAL COMMONS").
# Left alone they render at $75-175/sqft against ~$15 next door. A class-4 parcel sitting in a
# HOLE of an off-roll polygon is a unit of that regime whatever its size, as long as the common
# area is plausibly its site (<= COM_CA_MAX_RATIO x the units' footprint and <= 10 acres — not a
# campus or park that happens to surround a taxable lot). Residential lots in HOA holes are NOT
# included: those are fee-simple townhouse lots that own their land (skill §6b).
COM_CA_MAX_RATIO = 20.0
_nr_filled = gpd.GeoDataFrame({"ci": nonroll.index},
                              geometry=[_fill_holes(g) for g in nonroll.geometry.values], crs=UTM)
_c4 = pu[pu["book_class"].eq("4")]
_c4pt = gpd.GeoDataFrame({"pi": _c4.index}, geometry=_c4.geometry.representative_point().values, crs=UTM)
_h = gpd.sjoin(_c4pt, _nr_filled, predicate="within")
_h = _h[[not nonroll.geometry[ci].contains(pt) for ci, pt in zip(_h["ci"], _h.geometry)]]
_h["_u_sqft"] = pu.loc[_h["pi"], "_sqft"].values
_ca_sqft = nonroll.geometry.area * 10.763910416709722
_tot = _h.groupby("ci")["_u_sqft"].sum()
_ok_ci = {ci for ci, us in _tot.items()
          if _ca_sqft[ci] <= COM_CA_MAX_RATIO * us and _ca_sqft[ci] <= 10 * SQFT_PER_ACRE}
_com_units = _h[_h["ci"].isin(_ok_ci)]
log(f"Commercial condo units in holes of off-roll common areas: {len(_com_units)} on "
    f"{len(_ok_ci)} common areas ({_h['ci'].nunique() - len(_ok_ci)} surrounding off-roll "
    f"polygons rejected as too large to be the units' site)")
cand = pu[(pu["_sqft"] < UNIT_SQFT) | _stacked | pu.index.isin(_com_units["pi"])].copy()
log(f"Condo unit candidates: {len(cand):,} ({int((pu['_sqft'] < UNIT_SQFT).sum()):,} sub-"
    f"{UNIT_SQFT:.0f} sqft, {int(_stacked.sum()):,} stacked on another valued footprint, "
    f"{len(_com_units)} commercial units in common-area holes)")

# union-find over candidates (touching / stacked) and common-area polygons
_parents = pu[~pu.index.isin(cand.index)]
_nodes = ([("u", i) for i in cand.index] + [("c", i) for i in nonroll.index]
          + [("p", i) for i in _parents.index])
_pos = {n: k for k, n in enumerate(_nodes)}
_par = list(range(len(_nodes)))


def _find(a):
    while _par[a] != a:
        _par[a] = _par[_par[a]]
        a = _par[a]
    return a


def _union(a, b):
    ra, rb = _find(a), _find(b)
    if ra != rb:
        _par[rb] = ra


_cbuf = cand.geometry.buffer(0.5)
_cidx = list(cand.index)
for k, g in enumerate(_cbuf.values):
    for j in _cbuf.sindex.query(g, predicate="intersects"):
        if j > k:
            _union(_pos[("u", _cidx[k])], _pos[("u", _cidx[int(j)])])
_ca_hits = gpd.sjoin(cand[["geometry"]].assign(geometry=cand.geometry.buffer(1.0)),
                     nonroll[["geometry"]], predicate="intersects")
_ca_n = _ca_hits.groupby("index_right").size()
_ca_ok = set(_ca_n[_ca_n >= MIN_UNITS_ON_CA].index)
for ui, ci in zip(_ca_hits.index, _ca_hits["index_right"]):
    if ci in _ca_ok:
        _union(_pos[("u", ui)], _pos[("c", ci)])
for pi, ci in zip(_com_units["pi"], _com_units["ci"]):
    _union(_pos[("u", pi)], _pos[("c", ci)])
_ca_ok |= set(_com_units["ci"])

# A unit group with no common area touching it can instead sit in a HOLE of its development's
# parcel (Madison Manor: ~116 stacked garden-condo units in holes of "031 B 1", MADISON MANOR
# INC, 6.9 ac carrying a nominal $68k land / $171k clubhouse). Attach that hole-parent when it
# is off the roll (common area), or on the roll at under PARENT_MAX_LAND_PSF land $/sqft — a
# nominally-valued association parcel. A normally-valued parcel that merely wraps around a
# condo building ("012 M 3", an unrelated commercial lot around 1360 Hunters Rd) is NOT merged.
PARENT_MAX_LAND_PSF = 1.0
_filled = gpd.GeoDataFrame(
    {"node": [("c", i) for i in nonroll.index] + [("p", i) for i in _parents.index]},
    geometry=[_fill_holes(g) for g in nonroll.geometry.values]
    + [_fill_holes(g) for g in _parents.geometry.values], crs=UTM)
_actual = dict(zip(_filled["node"], list(nonroll.geometry.values) + list(_parents.geometry.values)))
_comp0 = pd.Series({("u", i): _find(_pos[("u", i)]) for i in cand.index})
_roots_with_ca = {_find(_pos[("c", i)]) for i in _ca_ok}
for root, members in _comp0.groupby(_comp0).groups.items():
    if root in _roots_with_ca:
        continue
    pt = unary_union([pu.geometry[n[1]] for n in members]).representative_point()
    for j in _filled.sindex.query(pt, predicate="within"):
        node = _filled["node"].iloc[j]
        if _actual[node].contains(pt):
            continue                                    # not a hole — the unit is ON it
        if node[0] == "p":
            r = pu.loc[node[1]]
            if r["land_val"] / max(r["_sqft"], 1.0) >= PARENT_MAX_LAND_PSF or len(members) < MIN_UNITS_ON_CA:
                continue
        _union(_pos[node], root)

comp = pd.Series({n: _find(_pos[n]) for n in _nodes})
merge_rows, drop_k = [], []
for root, members in comp.groupby(comp).groups.items():
    us = [n[1] for n in members if n[0] == "u"]
    cs = [n[1] for n in members if n[0] == "c"]
    ps = [n[1] for n in members if n[0] == "p"]
    if not us or (len(us) < 2 and not cs and not ps):
        continue
    grp = pu.loc[us + ps]
    # Per-unit records with their own values on one development -> SUM is right here. If every
    # unit carried the SAME values that could also be a broadcast; units of one building
    # legitimately often do, so this is logged rather than asserted.
    row = grp.iloc[0].to_dict()
    geoms = list(grp.geometry.values) + list(nonroll.loc[cs].geometry.values)
    # NB: `grp` holds the units plus any valued hole-parent; the parent's own land/building
    # value is summed in with the units' below (it is a distinct roll record, not a broadcast).
    row["geometry"] = _fill_holes(unary_union(geoms).buffer(0))
    for c in ("land_val", "bld_val", "book_total"):
        row[c] = float(grp[c].sum())
    _ucls = pu.loc[us, "book_class"]
    row["book_class"] = _ucls.value_counts().index[0]
    row["_condo"] = "res" if _ucls.isin(["1", "3"]).mean() >= 0.5 else "com"
    row["_n_units"] = len(us)
    row["_n_ca"] = len(cs) + len(ps)
    row["map"] = f"{pu.loc[us[0], 'map']} (+{len(us) - 1} units)"
    merge_rows.append(row)
    drop_k += list(grp["k"])

merged = gpd.GeoDataFrame(merge_rows, geometry="geometry", crs=UTM)
keep = pu[~pu["k"].isin(drop_k)]
# A valued parcel that was NOT merged (e.g. a fee-simple townhouse lot) can sit in one of the
# filled holes; carve it back out so nothing is drawn — or counted as land — twice.
_kidx = keep.sindex
_n_carved = 0
for i, g in zip(merged.index, merged.geometry.values):
    hits = [j for j in _kidx.query(g, predicate="intersects")
            if keep.geometry.values[j].intersection(g).area > 1.0]
    if hits:
        _n_carved += len(hits)
        merged.at[i, "geometry"] = g.difference(unary_union([keep.geometry.values[j] for j in hits])).buffer(0)
log(f"Condo merge: {len(drop_k):,} unit records + {int(merged['_n_ca'].sum())} common-area "
    f"polygons -> {len(merged):,} development parcels ({_n_carved} unmerged valued lots carved "
    f"back out of filled holes)")
_ac = merged.geometry.area * 10.763910416709722
for r, sq in sorted(zip(merged.to_dict("records"), _ac), key=lambda t: -t[0]["_n_units"])[:25]:
    log(f"    {r['map'][:30]:30s} {r['_condo']} units={r['_n_units']:3d} ca={r['_n_ca']} "
        f"{sq / SQFT_PER_ACRE:6.2f} ac land ${r['land_val']:>11,.0f}  ${r['land_val'] / sq:6.2f}/sqft")
parcel = gpd.GeoDataFrame(pd.concat([keep, merged], ignore_index=True), geometry="geometry",
                          crs=UTM).to_crs("EPSG:4326")
parcel["_condo"] = parcel["_condo"].fillna("")
parcel = parcel.drop(columns=["_sqft", "_rk"], errors="ignore")
assert parcel["land_val"].sum() + unmatched["land_val"].sum() == BOOK_TOTAL_LAND
log(f"After condo merge -> {len(parcel):,} parcels")

# ── 4. classification ─────────────────────────────────────────────────────────
def categorize(cls, bld, zoning, condo):
    if condo == "res":
        return "Condominium"
    if condo == "com":
        return "Commercial"
    if (bld or 0) <= 0:
        return "Vacant Land"
    if cls == "1":
        return "Single Family"
    if cls == "3":
        return "Multifamily"
    if cls == "4":
        return "Industrial" if zoning in INDUSTRIAL_ZONES else "Commercial"
    return "Other"


parcel["property_land_use_category"] = [categorize(c, b, z, m) for c, b, z, m in zip(
    parcel["book_class"], parcel["bld_val"], parcel["zoning"], parcel["_condo"])]
# Every shipped parcel is on the taxable REGULAR roll, so none is exempt. The flag is still
# written (the app and validator expect it); exempt land never entered the join.
parcel["exemption_flag"] = 0

# ── 5. canonical fields ───────────────────────────────────────────────────────
log("Computing GIS areas...")
parcel["geom_area_sqft"] = parcel["geometry"].apply(gis_area_sqft)
parcel.loc[parcel["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
_src = pd.to_numeric(parcel["SQFeet"], errors="coerce").where(parcel["_condo"].eq(""))
_agree = (parcel["geom_area_sqft"] / _src.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
log(f"Area cross-check vs source SQFeet — median ratio {_agree.median():.4f} (expect ~1.0), "
    f"rows off by >5%: {int((_agree.sub(1).abs() > 0.05).sum()):,}")
parcel["land_area_sqft"] = parcel["geom_area_sqft"]
parcel["area_source"] = "gis"
parcel["land_area_acres"] = parcel["land_area_sqft"] / SQFT_PER_ACRE
parcel["likely_remnant"] = (parcel["land_area_sqft"] < 500).astype(int)

parcel["land_value"] = parcel["land_val"].astype(float)
parcel["improvement_value"] = parcel["bld_val"].astype(float)
parcel["full_market_value"] = parcel["land_value"] + parcel["improvement_value"]
den = parcel["land_area_sqft"].replace(0, np.nan)
parcel["full_market_value_per_sqft"] = parcel["full_market_value"] / den
parcel["land_value_per_sqft"] = parcel["land_value"] / den
parcel["improvement_value_per_sqft"] = parcel["improvement_value"] / den
parcel = add_improvement_ratio_fields(parcel, land_col="land_value", improvement_col="improvement_value")

parcel["property_land_use_refined"] = classify_property_refined(
    parcel, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other",),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    fetch_footprints=False)

# Vision/VGSI per-parcel page (the city's own "Real Estate Info System"); the GIS layer stores
# it as http — the site serves https.
parcel["link"] = parcel["hbrislink"].astype(str).str.replace("http://", "https://", regex=False)
parcel.loc[~parcel["link"].str.startswith("https://"), "link"] = "https://gis.vgsi.com/harrisonburgva/"

# ── 6. export ─────────────────────────────────────────────────────────────────
COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
           "likely_remnant"]
final = parcel[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
final["geometry"] = final["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")
out = DATA_DIR / "harrisonburg-va-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"harrisonburg-va-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet",
                 index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"land ${final.current_full_land_value.sum():,.0f}  impr ${final.improvement_value.sum():,.0f}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")

# ── §6a smoke alarms: condo / unit stubs ──────────────────────────────────────
log("--- condo/stub smoke alarms (skill §6a) ---")
a = parcel["geom_area_sqft"]
lv = pd.to_numeric(final["land_value_per_sqft"], errors="coerce")
log(f"  footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
log(f"  sub-500 / sub-1000 sqft footprints: {int((a < 500).sum()):,} / {int((a < 1000).sum()):,}")
log(f"  land $/sqft p50/p95/p99/max: ${lv.median():,.2f} / ${lv.quantile(.95):,.2f} / "
    f"${lv.quantile(.99):,.2f} / ${lv.max():,.2f}")
rp = final.geometry.representative_point()
vc = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)).value_counts()
log(f"  stacked clusters: {int((vc > 1).sum())}, max stack: {int(vc.max())}")
holes = final.geometry.apply(lambda g: 0 if g is None else sum(
    len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
log(f"  parcels with interior rings (holes): {int((holes > 0).sum()):,}")
log(f"  zero land value (renders as gp-error): "
    f"{int((final['current_full_land_value'].fillna(0) <= 0).sum()):,}")
log(f"  bounds: {[round(v, 4) for v in final.total_bounds]}")
log("DONE")
