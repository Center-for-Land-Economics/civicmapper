#!/usr/bin/env python3
"""
Build the City of Lynchburg, VA canonical parcel parquet.

Lynchburg is a Virginia INDEPENDENT CITY (FIPS 51680) — not Lynchburg TN (Moore County,
of whiskey fame) and not Campbell/Bedford/Amherst County land around it. The city's own
parcel layer is already city-only, so no county clip is needed.

Source (City of Lynchburg GIS open data, public, no token):
- OpenData/ODPDynamic MapServer layer 41 ("Parcel"):
  https://mapviewer.lynchburgva.gov/ArcGIS/rest/services/OpenData/ODPDynamic/MapServer/41
  ~32.6k parcels. One-stop layer: geometry + current assessor land/improvement/total values
  + property class code & description + acreage + owner, all in one place. No joins, no
  manual downloads.

  Cross-check on 2026-08-29: the City Assessor's office publicly states it values ~32,525
  parcels of which 1,645 are exempt (https://www.lynchburgva.gov/167/City-Assessor). The
  live layer returns 32,600 parcels / 1,650 EXEMPT-* — i.e. this open-data layer IS the
  complete assessor roll, not a stale or partial extract.

Outputs:
- data/jurisidictions/data/lynchburg/lynchburg-va-parcels.parquet
- data/jurisidictions/data/lynchburg/lynchburg-va-parcels_YYYY_MM_DD.parquet

Notes:
- ~32.6k parcels -> small enough for the browser GeoParquet path (no PMTiles / H3 hexes),
  like Newport News and South Bend. Registered with usePmtiles omitted (default off).
- Values: Current_Land / Current_Imp / Current_Total. The assessor's own Current_Total is
  used directly for full_market_value (falling back to land+imp) so the roll is preserved.
  Lynchburg reassesses BIENNIALLY, so values are a 2025-2026 roll.
- Classification: PropClas/PCDesc is an unusually clean, purpose-built taxonomy — the
  VACANT-* classes (100/300/400/500) and COMMERCIAL - PARKING (407) map 1:1 onto the app's
  Vacant / Parking Lot refined buckets. No heuristics needed for those.
- EXEMPT: institutional exempt classes (710-790) -> exemption_flag=1, excluded.
  IMPORTANT EXCEPTION — the DAV classes (791/792/793/795, "EXEMPT - OTHER (DAV nn%)") are
  DISABLED-AMERICAN-VETERAN tax relief granted to the OWNER, not institutionally exempt
  land. Verified live: all 259 carry a normal full market land assessment, and 253 of them
  also carry improvement value (e.g. $40k land / $195.6k improvement, NumDwlg=1, individual
  person owners) — i.e. they are ordinary houses. The remaining 6 are vacant lots with real
  land value and no building. Excluding them would punch 259 holes in residential
  neighborhoods for no data reason, so they are KEPT with exemption_flag=0 and categorized
  by what is actually on them (Single Family when improved, Vacant Land when not).
- PUBLIC SERVICE CORPORATION (801, 135 parcels) is EXCLUDED: verified live that every one
  has Current_Land = 0 AND Current_Imp = 0. These are state-assessed utility/railroad
  parcels valued by the VA State Corporation Commission, not the local assessor, so they
  carry no local value at all and would render as zero-value error parcels.
- Land area: LegalAc (acres) is populated on only ~41% of parcels, and on condo/unit records
  it is a per-unit SHARE rather than the footprint. So reported acreage is used only when it
  is within 0.5-2.0x the geodesic polygon area (the run_richmond.py guard); otherwise the
  geodesic polygon area is the denominator. Where LegalAc IS populated it agrees closely
  with the polygon (0.34 vs 0.3399, 0.326 vs 0.3262), so the fallback is safe.
- Condos: 712 condo-class (105) parcels. Each UNIT is mapped as its own ~360 sqft slice of the
  building carrying the unit's land-value share (~$12-20k), while the development's actual
  land is a separate "COMMON AREA" parcel (108/499) assessed at $0. Left alone that renders as
  a forest of $30-460/sqft pencils next to a $0 gp-error lot (skill §6a). So units are MERGED
  DOWN onto their development's common-area land (skill §6b / run_olympia.py): grouped by the
  Legal1 development name and by touching geometry, values summed, footprint = union of the
  units + common area with holes filled. Units with no common area and real-sized lots
  (detached "villa" condos) are left as individual parcels.
"""
from __future__ import annotations

import json
import re
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
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined, gis_area_sqft  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "lynchburg"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "lynchburg-va-geometry.parquet"

PARCELS_URL = ("https://mapviewer.lynchburgva.gov/ArcGIS/rest/services/"
               "OpenData/ODPDynamic/MapServer/41/query")
WHERE = "1=1"
# Only what the ETL actually reads. Shape_Acres is the source's own GIS area and is used
# purely as a cross-check on our geodesic computation (see the area-agreement guard below).
# Legal1 is the legal description; its leading segment is the DEVELOPMENT name that links
# condo units to their common-area land parcel (see the condo merge below).
OUT_FIELDS = ("OBJECTID,Parcel_ID,PropClas,Current_Land,Current_Imp,Current_Total,"
              "LegalAc,Shape_Acres,NumDwlg,FinSize,Legal1")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0
PAGE = 2000
geod = Geod(ellps="WGS84")

# Institutional exemptions -> excluded. NOTE 791/792/793/795 (DAV) are deliberately absent:
# those are disabled-veteran OWNER relief on ordinary houses, not exempt institutional land.
EXEMPT_CLASSES = {"710", "720", "730", "741", "742", "743", "744", "745",
                  "760", "770", "780", "790"}
DAV_CLASSES = {"791", "792", "793", "795"}
PUBLIC_SERVICE_CLASS = "801"  # state-assessed utility; zero local value -> excluded


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_parcels():
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    total = requests.get(PARCELS_URL, params={"where": WHERE, "returnCountOnly": "true",
                         "f": "json"}, headers=HEADERS, timeout=120).json().get("count", 0)
    log(f"Pulling {total:,} Lynchburg parcels (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(5):
            try:
                r = requests.get(PARCELS_URL, params={
                    "where": WHERE, "outFields": OUT_FIELDS, "returnGeometry": "true",
                    "resultOffset": off, "resultRecordCount": PAGE, "outSR": 4326,
                    "orderByFields": "OBJECTID", "f": "geojson",
                }, headers=HEADERS, timeout=240)
                r.raise_for_status()
                # NB: gpd.read_file(BytesIO(...)) fails here under geopandas 1.1.4 / pyogrio
                # ("FeatureError: URL rejected: No host part in the URL" — GDAL tries to treat
                # the buffer as a URL). The payload is valid GeoJSON, so parse it directly.
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
        if off % 10000 < PAGE:
            log(f"  fetched {off:,}/{total:,}")
        if len(gdf) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


geom = fetch_parcels()
geom["acct"] = geom["Parcel_ID"].astype(str).str.strip()
geom = geom[geom["acct"].ne("") & geom["acct"].ne("None") & geom["acct"].ne("nan")]
for c in ["Current_Land", "Current_Imp", "Current_Total", "LegalAc", "NumDwlg", "FinSize"]:
    geom[c] = pd.to_numeric(geom[c], errors="coerce")
geom["prop_class"] = geom["PropClas"].astype(str).str.strip()

parcel = geom.rename(columns={"Current_Land": "land_val", "Current_Imp": "bld_val"})
parcel["tot_appr_val"] = pd.to_numeric(parcel["Current_Total"], errors="coerce")
parcel["tot_appr_val"] = parcel["tot_appr_val"].fillna(
    parcel["land_val"].fillna(0) + parcel["bld_val"].fillna(0))
parcel["stated_acres"] = parcel["LegalAc"]

if parcel.crs is None:
    parcel = parcel.set_crs("EPSG:4326")
elif parcel.crs.to_epsg() != 4326:
    parcel = parcel.to_crs("EPSG:4326")
parcel["geometry"] = parcel["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
parcel = parcel[parcel["geometry"].notnull() & parcel["geometry"].apply(
    lambda x: getattr(x, "is_valid", False) and not x.is_empty)].copy()
log(f"Valid-geometry parcels -> {len(parcel):,}")

# ── dedup multi-polygon parcels (values first — NEVER sum, geometry unioned) ──
# A single account split across several GIS polygons must not have its account-level value
# summed per polygon (the Dallas N-x inflation bug); take `first` and union the shapes.
ndup = parcel.duplicated(subset=["acct"], keep=False).sum()
log(f"Rows sharing a Parcel_ID (multi-polygon): {ndup:,}")
if ndup:
    first_cols = [c for c in parcel.columns if c not in ("geometry", "acct")]
    agg = {c: "first" for c in first_cols}
    coll = parcel.groupby("acct", dropna=False).agg(agg).reset_index()
    gu = parcel.groupby("acct", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]) if any(x is not None for x in gs) else None)
    coll["geometry"] = gu.values
    parcel = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
log(f"After Parcel_ID dedup -> {len(parcel):,}")

# ── condo same-footprint collapse (>1 distinct account at one point -> SUM) ───
# Lynchburg maps condo units as adjacent slices rather than a shared stack, so this is
# expected to find very little. Kept as a defensive net + diagnostic (playbook §5 / skill §6).
rp = parcel.geometry.representative_point()
parcel["_rpkey"] = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str))
vc = parcel["_rpkey"].value_counts()
stacked_keys = vc[vc > 1].index
log(f"Stacked footprints: {len(stacked_keys):,}; max stack: {int(vc.max())}; "
    f"parcels involved: {int(vc[vc > 1].sum()):,}")
if len(stacked_keys):
    is_stacked = parcel["_rpkey"].isin(stacked_keys)
    single = parcel[~is_stacked].copy()
    single["_collapsed"] = 0
    multi = parcel[is_stacked].copy()
    # Stacked units are per-unit shares of ONE shared parcel: sum values AND the per-unit
    # stated area, then reconcile the denominator against the union polygon further below.
    sum_cols = ["land_val", "bld_val", "tot_appr_val", "stated_acres"]
    first_cols = [c for c in multi.columns if c not in (["geometry", "_rpkey"] + sum_cols)]
    agg = {c: "sum" for c in sum_cols if c in multi.columns}
    agg.update({c: "first" for c in first_cols})
    coll = multi.groupby("_rpkey", dropna=False).agg(agg).reset_index()
    gu = multi.groupby("_rpkey", dropna=False)["geometry"].apply(
        lambda gs: unary_union([x for x in gs if x is not None]))
    coll["geometry"] = gu.values
    coll["_collapsed"] = 1
    coll = gpd.GeoDataFrame(coll, geometry="geometry", crs="EPSG:4326")
    parcel = gpd.GeoDataFrame(pd.concat([single, coll], ignore_index=True),
                              geometry="geometry", crs="EPSG:4326")
else:
    parcel["_collapsed"] = 0
parcel = parcel.drop(columns=["_rpkey"], errors="ignore")
log(f"After condo footprint collapse -> {len(parcel):,}")

# ── exemption flag + classification ──────────────────────────────────────────
parcel["exemption_flag"] = parcel["prop_class"].isin(EXEMPT_CLASSES).astype(int)


def categorize(cls, impr, dwellings):
    """Map Lynchburg PropClas -> app property category.

    The assessor's class code is authoritative and complete (0 nulls verified), so this is a
    direct code mapping rather than a text heuristic.
    """
    cls = str(cls or "").strip()
    if cls in ("100", "300", "400", "500"):
        return "Vacant Land"
    if cls in ("101", "103", "106"):          # detached / townhouse / attached single family
        return "Single Family"
    if cls in ("102", "104"):                 # 2 units / 3-4 units
        return "Multifamily"
    if cls in ("301", "303"):                 # apartment complex / converted house
        return "Multifamily"
    if cls == "105":
        return "Condominium"
    if cls == "107":
        # A house sitting on commercially-zoned land — a prime underutilization candidate, so
        # it deliberately does NOT get the stricter 'Single Family' land-share cutoff.
        return "Residential (Commercially Zoned)"
    if cls in ("108", "499"):                 # residential / commercial HOA common area
        return "Common Area"
    if cls == "302":
        return "Mobile Home"
    if cls == "407":
        return "Parking"
    if cls == "423":
        return "Parking Garage"               # a STRUCTURE, not surface parking — see override below
    if cls.startswith("4"):
        return "Commercial"
    if cls.startswith("5"):
        return "Industrial"
    if cls in DAV_CLASSES:
        # Disabled-veteran owner relief on an ordinary parcel — categorize by what is on it.
        # 6 of the 259 DAV parcels are VACANT LOTS (no improvement, no dwelling, no floor
        # area) carrying real land value. They must land in "Vacant Land": "Other" is in
        # exclude_categories below, which would block the improvement==0 -> Vacant rule too
        # and hide them from BOTH the original and refined vacant views.
        if (impr or 0) > 0 or (dwellings or 0) >= 1:
            return "Single Family"
        return "Vacant Land"
    if cls in EXEMPT_CLASSES:
        return "Exempt"
    if cls == PUBLIC_SERVICE_CLASS:
        return "Utility"
    return "Other"


parcel["PROPERTY_CATEGORY"] = [categorize(c, i, n) for c, i, n in zip(
    parcel["prop_class"], parcel["bld_val"], parcel["NumDwlg"])]

# Drop institutionally-exempt land AND the zero-value state-assessed utility parcels.
drop_utility = parcel["prop_class"].eq(PUBLIC_SERVICE_CLASS)
log(f"Excluding {int(parcel['exemption_flag'].sum()):,} exempt + "
    f"{int(drop_utility.sum()):,} public-service-corporation (zero local value) parcels")
ex = parcel[(parcel["exemption_flag"] == 0) & (~drop_utility)].copy()

# ── condo units -> merge DOWN onto their development's common-area land (skill §6b) ──────
# Lynchburg maps each condo unit (class 105) as its own slice of the building and carries the
# unit's land-value share on that slice; the development's real land is a separate COMMON AREA
# parcel (108/499) assessed at $0. Group units with their common-area parcel(s) — linked by the
# Legal1 development name ("PARKSIDE GRANDE, BLDG 1, UNIT 101" <-> "PARKSIDE GRANDE, COMMON
# AREA") OR by touching geometry (0.5 m) — and collapse each group to ONE parcel: land/building
# values summed, footprint = union of units + common area with holes filled. Same recipe as
# run_olympia.py; the assessor's own common-area polygon is the land, nothing is synthesized.
from shapely.geometry import Polygon, MultiPolygon

UTM = "EPSG:32617"
STUB_SQFT = 1500.0  # units-only groups merge only when they are slices, not detached villa lots


def _fill_holes(g):
    if g is None or g.is_empty:
        return g
    if g.geom_type == "Polygon":
        return Polygon(g.exterior)
    if g.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in g.geoms])
    return g


_GENERIC = re.compile(r"\b(CONDOS?|CONDOMINIUMS?|CONOMINIUM|TOWNHOMES?|TOWNHOUSES?)\b")
_TOKEN = re.compile(r"^(BLK|BLOCK|LOTS?|UNITS?|BLDG|BUILDING|PH|PHASE|COMMON AREA|PART|RESIDUE|REV PARCEL|PARCEL)\b")


def _dev_key(legal):
    """Development name from Legal1, e.g. "PARKSIDE GRANDE, BLDG 1, UNIT 101" -> PARKSIDE GRANDE.

    - Generic suffixes are dropped so "CHELSEA HOUSE, BLDG A, UNIT 3" matches its lot
      "CHELSEA HOUSE CONDO, COMMON AREA" (also EAST RIDGE CONDOMINIUM(S), the CONOMINIUM typo...).
    - Downtown parcels read "DOWNTOWN, BLK 93, 1220 MAIN ST CONDOS, UNIT 3": the first segment is
      the whole district, so the building name is the first non-token segment after it.
    - A few common-area lots lead with the token: "COMMON AREA, TOWN CENTER, BLK F, LOT 8,
      PARKSIDE GRANDE" -> the development is the LAST non-token segment.
    """
    segs = [x.strip() for x in str(legal or "").upper().split(",") if x.strip()]
    if not segs:
        return ""
    if segs[0] == "DOWNTOWN":
        rest = [x for x in segs[1:] if not _TOKEN.match(x)]
        name = re.sub(r"\b(UNITS?)\b.*$", "", rest[0]).strip() if rest else ""
        return "DOWNTOWN | " + name if name else "DOWNTOWN"
    if segs[0] == "COMMON AREA":
        rest = [x for x in segs[1:] if not _TOKEN.match(x)]
        name = rest[-1] if rest else ""
    else:
        name = segs[0]
    name = _GENERIC.sub("", name)
    return re.sub(r"\s+", " ", name).strip(" -@")


ex["_dev"] = ex["Legal1"].apply(_dev_key)
is_ca = ex["prop_class"].isin(("108", "499"))
# A "unit" is class 105 (RESIDENTIAL - CONDOMINIUM) OR any other parcel whose legal description
# is a UNIT ("PIEDMONT OFFICE CONDOS, UNIT III", "IVY CREEK TOWNHOMES, UNIT 204", "CARRIAGE SQUARE
# CONDO, UNIT 27"): the assessor files townhome condos under 103/106, office/retail condos under
# 402/415/416/430 and disabled-veteran condo units under 791, all mapped as the same building-slice
# stubs inside a $0 common-area donut. Parcels whose legal says LOT (fee-simple townhouse and
# Town Center shop lots inside an HOA common area) are NOT units: they own their small lot and
# are left alone (playbook §6b: don't fill legit fee-simple holes). A UNIT parcel only merges if
# it actually joins a common-area group below; otherwise it is untouched.
# "LAKESIDE PLAZA CONDO" / "MCCONVILLE PARK CONDO UNIT 1" style legals count too (CONDO without
# UNIT); non-105 candidates are capped at 15,000 sqft so a whole valued condo-regime parent parcel
# is never mistaken for a unit.
_legal_u = ex["Legal1"].fillna("").str.upper()
_is_unit_legal = _legal_u.str.contains(r"\bUNITS?\b|\bCONDO", regex=True) & (ex.geometry.to_crs(UTM).area * 10.763910416709722 < 15000)
is_unit = (ex["prop_class"].eq("105") | _is_unit_legal) & ~is_ca
mp = ex[is_unit | is_ca].to_crs(UTM)
mp["_sqft"] = mp.geometry.area * 10.763910416709722
mp["_is_unit"] = is_unit[mp.index].values
units = mp[mp["_is_unit"]]
cas = mp[~mp["_is_unit"]]

# 1) Unit groups: union-find over units only — touching slices (same building) or same dev name.
upos = {i: k for k, i in enumerate(units.index)}
_parent = list(range(len(units)))


def _find(a):
    while _parent[a] != a:
        _parent[a] = _parent[_parent[a]]
        a = _parent[a]
    return a


def _union(a, b):
    ra, rb = _find(a), _find(b)
    if ra != rb:
        _parent[rb] = ra


_ubuf = units.geometry.buffer(0.5)
_usidx = _ubuf.sindex
for k, g in enumerate(_ubuf.values):
    for j in _usidx.query(g, predicate="intersects"):
        if j > k:
            _union(k, int(j))
for dev in set(units["_dev"]) - {""}:
    ks = [upos[i] for i in units.index[units["_dev"].eq(dev)]]
    for k in ks[1:]:
        _union(ks[0], k)
ugrp = pd.Series([_find(upos[i]) for i in units.index], index=units.index)
grp_devs = units.groupby(ugrp)["_dev"].agg(lambda s: set(s) - {""})

# 2) Attach common-area parcels to unit groups: by development NAME first (the association's own
#    land); a CA that matches no name attaches to a unit group it TOUCHES only if that group has
#    no named CA yet. Common-area parcels never chain to each other — otherwise a master-planned
#    community's HOA open space (Wyndhurst "PARK AREA RESIDUE", Cornerstone "REV PARCEL A") gets
#    swallowed into whichever condo it happens to border and dilutes its $/sqft.
ca_grp = {}
named = set()
for i, dev in zip(cas.index, cas["_dev"]):
    if not dev:
        continue
    hits = [g for g, devs in grp_devs.items() if dev in devs]
    if hits:
        g = max(hits, key=lambda g: int((ugrp == g).sum()))
        ca_grp[i] = g
        named.add(g)
_cabuf = cas.geometry.buffer(0.5)
for i, g in zip(cas.index, _cabuf.values):
    if i in ca_grp:
        continue
    touched = ugrp.loc[units.index[_usidx.query(g, predicate="intersects")]]
    touched = [t for t in set(touched) if t not in named]
    if touched:
        ca_grp[i] = max(touched, key=lambda t: int((touched == t) if False else (ugrp == t).sum()))

# 3) Merge each group that has a CA, or is a multi-unit group of building SLICES (no CA at all —
#    the slices' union is the building footprint). Multi-unit groups of real-sized lots (detached
#    "villa" condos on their own land) are left as individual parcels.
_gs = pd.DataFrame({"units": ugrp.value_counts()})
_gs["cas"] = pd.Series(ca_grp).value_counts().reindex(_gs.index).fillna(0).astype(int)
_gs["unit_med_sqft"] = units.groupby(ugrp)["_sqft"].median()
_gs["all_105"] = units.groupby(ugrp)["prop_class"].agg(lambda s: bool(s.eq("105").all()))
# Merge a group when it has common-area land, or when it is a multi-unit group of class-105
# building slices with no common area at all (their union is the building footprint). UNIT-legal
# parcels of other classes only ever merge onto a common-area lot.
# At least two units either way: one lone "unit" next to a common-area lot is a mis-keyed legal,
# not a development, and would just dilute its value over land it does not own.
_merge = _gs.index[(_gs.units > 1) & ((_gs.cas > 0) | ((_gs.unit_med_sqft < STUB_SQFT) & _gs["all_105"]))]
_sum_cols = ["land_val", "bld_val", "tot_appr_val", "NumDwlg", "FinSize"]
rows, drop_idx = [], []
for gid in _merge:
    u_idx = list(ugrp.index[ugrp == gid])
    c_idx = [i for i, g in ca_grp.items() if g == gid]
    grp = mp.loc[u_idx + c_idx]
    row = mp.loc[u_idx[0]].to_dict()
    row["geometry"] = _fill_holes(unary_union(list(grp.geometry.values)))
    for c in _sum_cols:
        row[c] = float(pd.to_numeric(grp[c], errors="coerce").fillna(0).sum())
    row["stated_acres"] = np.nan          # the merged footprint is the land denominator
    # Residential unit classes (1xx, DAV 79x) -> Condominium; office/retail condo units (4xx) ->
    # Commercial, so the improvement-ratio rule judges the complex like any other commercial parcel.
    _ucls = mp.loc[u_idx, "prop_class"].astype(str)
    _res = (_ucls.str.startswith("1") | _ucls.isin(DAV_CLASSES)).mean()
    row["PROPERTY_CATEGORY"] = "Condominium" if _res >= 0.5 else "Commercial"
    row["_dev"] = " / ".join(sorted(grp_devs[gid])) or "(unnamed)"
    row["_n_units"] = len(u_idx)
    row["_n_ca"] = len(c_idx)
    rows.append(row)
    drop_idx += u_idx + c_idx
if rows:
    merged = gpd.GeoDataFrame(rows, geometry="geometry", crs=UTM)
    merged["geometry"] = merged.geometry.apply(lambda g: g if g.is_valid else g.buffer(0))
    merged = merged.to_crs("EPSG:4326")
    n_u = int(mp.loc[drop_idx, "_is_unit"].sum())
    n_c = len(drop_idx) - n_u
    ex = gpd.GeoDataFrame(pd.concat([ex.drop(index=drop_idx), merged], ignore_index=True),
                          geometry="geometry", crs="EPSG:4326")
    log(f"Condo merge: {n_u:,} units + {n_c:,} common-area parcels -> {len(rows):,} development parcels")
    _acres = merged.to_crs(UTM).geometry.area * 10.763910416709722 / SQFT_PER_ACRE
    _psf = merged["land_val"] / (_acres * SQFT_PER_ACRE)
    for r, a, v in sorted(zip(merged.to_dict("records"), _acres, _psf), key=lambda t: -t[0]["_n_units"]):
        log(f"    {r['_dev'][:40]:40s} {r['PROPERTY_CATEGORY'][:5]} units={r['_n_units']:3d} ca={r['_n_ca']:2d} "
            f"{a:7.2f} ac  land ${r['land_val']:>12,.0f}  ${v:6.2f}/sqft")
    multi = merged[merged["_dev"].str.contains(" / ")]
    if len(multi):
        log(f"  NOTE: {len(multi)} merged parcel(s) span >1 development name (touching slices): "
            f"{multi['_dev'].tolist()}")
left = ex[ex["prop_class"].eq("105")]
log(f"Class-105 condo units left as individual parcels: {len(left):,} "
    f"(median footprint {left.to_crs(UTM).geometry.area.median() * 10.7639:,.0f} sqft)")
ex = ex.drop(columns=["_dev", "_n_units", "_n_ca", "_is_unit"], errors="ignore")

ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
ex["land_value"] = pd.to_numeric(ex["land_val"], errors="coerce")
ex["improvement_value"] = pd.to_numeric(ex["bld_val"], errors="coerce")
ex["bld_ar"] = pd.to_numeric(ex["FinSize"], errors="coerce").fillna(0)
ex["property_land_use_refined"] = classify_property_refined(
    ex, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Other", "Common Area"),
    category_col="property_land_use_category",
    land_col="land_value", improvement_col="improvement_value",
    bld_ar_col="bld_ar",
    fetch_footprints=False)

# Parking GARAGES need no local handling any more: classify_property_refined now excludes
# parking structures from the 'Parking Lot' force-label repo-wide, so class 423 is judged
# by the ordinary improvement ratio automatically.

log(f"After exempt/utility filter -> {len(ex):,}")

# ── canonical fields — LegalAc denominator, guarded, geodesic fallback ───────
ex["geometry"] = ex["geometry"].apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
log("Computing GIS areas...")
ex["geom_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
ex.loc[ex["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan

# Cross-check our geodesic area against the source's own GIS area. These should agree to
# well under a percent; a systematic divergence means the area computation is wrong (this
# is what catches an exterior-only area that ignores interior rings). Log, don't fail —
# individual parcels can legitimately differ, a shifted MEDIAN cannot.
_src_sqft = pd.to_numeric(ex["Shape_Acres"], errors="coerce") * SQFT_PER_ACRE
_agree = (ex["geom_area_sqft"] / _src_sqft.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
log(f"Area cross-check vs source Shape_Acres — median ratio {_agree.median():.4f} "
    f"(expect ~1.0000), rows off by >5%: {int((_agree.sub(1).abs() > 0.05).sum()):,}")
rep = pd.to_numeric(ex.get("stated_acres", np.nan), errors="coerce") * SQFT_PER_ACRE
ex["reported_sqft"] = rep
ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

# Collapsed stacks: the shared complex footprint is the denominator, not one unit's share.
col = ex["_collapsed"] == 1
if col.any():
    ex.loc[col, "reported_sqft"] = np.maximum(
        pd.to_numeric(ex.loc[col, "reported_sqft"], errors="coerce").fillna(0.0),
        pd.to_numeric(ex.loc[col, "geom_area_sqft"], errors="coerce").fillna(0.0))
    ex.loc[ex["reported_sqft"] < 1, "reported_sqft"] = np.nan

# Richmond guard: reported acreage is only trusted when it is in the same ballpark as the
# actual polygon. Condo/unit records carry a per-unit SHARE of the development's land, which
# would otherwise produce astronomical $/sqft on a normal-sized footprint.
ratio = ex["reported_sqft"] / ex["geom_area_sqft"].replace(0, np.nan)
# `| geom_area_sqft.isna()` so a parcel with good reported acreage but no usable polygon
# area keeps its acreage instead of falling through to NaN (ratio would be NaN -> False).
use_reported = ex["reported_sqft"].gt(0) & (ratio.between(0.5, 2.0) | ex["geom_area_sqft"].isna())
n_rejected = int((ex["reported_sqft"].gt(0) & ~use_reported).sum())
log(f"Reported acreage rejected as implausible (outside 0.5-2.0x polygon): {n_rejected:,}")
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

# Assessor parcel viewer (the city's own public lookup, linked from lynchburgva.gov/962).
# NO PER-PARCEL DEEP LINK EXISTS. Lynchburg migrated its ParcelViewer to the CivQuest SPA
# (the old mapviewer.lynchburgva.gov/ParcelViewer/ now 302s there), and that app ignores URL
# query parameters — ?parcel=, ?search= and ?q= were all tested live and every one lands on
# the empty "Search for a parcel ID, owner, or address" state. So this links to the viewer
# root rather than fabricating a per-parcel URL that would silently fail. The popup shows the
# Parcel ID and Address next to it, which are exactly what that viewer's search box takes.
# Revisit if CivQuest ever adds deep links.
ex["link"] = "https://lynchburgva.civ.quest/"

# ── export ────────────────────────────────────────────────────────────────────
# Canonical column set only — identical to run_newportnews.py. The source layer also
# carries owner, address, neighborhood and year-built, but those are deliberately NOT
# exported: the app never reads them (the metric dropdowns are a fixed allowlist and the
# underutilization rule is purely a value ratio), the skill's "Popups: derive, don't bake"
# rule is against carrying columns the app doesn't need, and owner names in particular
# should not be baked into a bulk dataset just because the assessor publishes them
# one-at-a-time. Neighborhood is worth revisiting later as a proper jurisdictionGroups
# region toggle (Lynchburg land values are computed as neighborhood rate x area), not as
# a stray popup column.
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
out = DATA_DIR / "lynchburg-va-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"lynchburg-va-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet",
                 index=False)
log(f"SAVED {out} | rows {len(final):,}")
log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
log(f"area_source: {final['area_source'].value_counts().to_dict()}")
log(f"likely_remnant: {int(final['likely_remnant'].sum()):,}")

# ── §6a smoke alarms: prove units/condos are not mis-mapped stubs ─────────────
log("--- condo/stub smoke alarms (skill §6a) ---")
a = ex["geom_area_sqft"]                                   # geodesic footprint, computed above
lv = pd.to_numeric(final["land_value_per_sqft"], errors="coerce")
shown = lv[final["likely_remnant"] == 0]                   # what the app actually renders
log(f"  footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
log(f"  sub-500 / sub-1000 sqft footprints: {int((a < 500).sum()):,} / {int((a < 1000).sum()):,}")
# Both populations are printed, explicitly labelled, at 2dp. Reporting a single blended set
# of percentiles invites quoting an all-rows p99 next to a rendered max.
log(f"  land $/sqft ALL ROWS    p50/p95/p99/max: ${lv.median():,.2f} / ${lv.quantile(.95):,.2f} / "
    f"${lv.quantile(.99):,.2f} / ${lv.max():,.2f}")
log(f"  land $/sqft AS RENDERED p50/p95/p99/max: ${shown.median():,.2f} / ${shown.quantile(.95):,.2f} / "
    f"${shown.quantile(.99):,.2f} / ${shown.max():,.2f}  (likely_remnant excluded, hideRemnants=true)")
holes = final.geometry.apply(lambda g: 0 if g is None else sum(
    len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
log(f"  parcels with interior rings (holes): {int((holes > 0).sum()):,}")
log(f"  zero/neg land value (renders as gp-error): "
    f"{int((pd.to_numeric(final['current_full_land_value'], errors='coerce').fillna(0) <= 0).sum()):,}")
log(f"  bounds: {[round(v, 4) for v in final.total_bounds]}")
log("DONE")
