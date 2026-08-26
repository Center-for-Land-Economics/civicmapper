#!/usr/bin/env python3
"""
Build Lakewood, CO's canonical parcel parquet from live Jefferson County services.

Lakewood is a municipality inside Jefferson County, so the parcel source is
COUNTYWIDE (~259k parcels) and MUST be restricted to city limits. We clip on the
county's authoritative municipal-boundary layer (centroid-within), not on the
situs/mailing city string -- `PRPCTYNAM` is case-inconsistent and 3,063 parcels
carry a "LAKEWOOD" address while sitting in unincorporated Jeffco.

Sources
-------
- Parcels (geometry + actual values + abstract class, one-stop):
  https://gisportal.jeffco.us/server2/rest/services/Parcel/FeatureServer/20
- Municipal boundaries (authoritative city limits):
  https://gisportal.jeffco.us/server/rest/services/City/FeatureServer/4

Jeffco schema notes (probed live 2026-08-20)
-------------------------------------------
- `TOTACTLNDV` / `TOTACTIMPV` / `TOTACTVAL` are Colorado *actual* (market)
  values. `ASMASD*` are assessed values (post-ratio) -- we want actual.
- `TAXCLS` is the Colorado 4-digit abstract class. Verified empirically:
    0xxx vacant land          1112/1115 single family      1120/1125 apartments
    1230/1212/1225/1250 condo 2xxx commercial             3xxx industrial
    4xxx agricultural         5xxx natural resources      9xxx EXEMPT
  The 9xxx = exempt inference was validated against owner names: City of
  Lakewood, Jefferson County School District R1, churches, CDOT/State Highway,
  metro districts, housing authority, Colorado Christian University, and the
  United States of America (the former Denver Federal Center, class 9119).
- `TOTACR` IS NOT TRUSTWORTHY. Class 9139 (school district) sums to 296,348
  acres inside a 28,000-acre city, and condo units report 0. All areas are
  therefore computed geodesically from geometry.
- `PIN` is 'ROW' for road right-of-way (6,033 polygons in Lakewood) and 'WATER'
  for water bodies. These are not assessable parcels and are dropped.

The condo trap (the Fort Collins lesson, present here) -- THREE regimes
-----------------------------------------------------------------------
Jefferson County maps THREE distinct parcel regimes, and they must be told apart
by whether the account owns ground *in fee*, not by `POLYCAT` (an earlier version
of this ETL keyed off `POLYCAT in (2,3)` and silently missed regime A entirely).

The reliable discriminator is `LGLSQFT`, the deeded legal area. For ordinary
fee-simple parcels `LGLSQFT / polygon_area` has a median of 1.001 (verified over
40,130 parcels), so `LGLSQFT <= 0` means "this account owns no ground in fee".

  regime N  fee-simple            ~40,130  LGLSQFT > 0.  Polygon IS the deeded
                                           lot. Nothing to do.
  regime A  attached / townhome    ~6,741  LGLSQFT = 0, `STTSTRC` = 'Townhomes',
                                           REAL per-unit land value (~$543M
                                           citywide). Polygon is the BUILDING
                                           FOOTPRINT only (median 794 sqft); the
                                           development's ground is a separate HOA
                                           tract. Because the height field is
                                           land_value_per_sqft, an unmerged
                                           townhome with $115k of land on a
                                           450 sqft footprint extrudes at
                                           $255/sqft against a true ~$18.
  regime C  true condominium       ~5,239  LGLSQFT = 0, class 1230/2245/3230,
                                           land value $0 (see limitation below).
                                           Unit polygons stacked on one
                                           footprint.

Both A and C are merged DOWN onto the development's ground tract. Linkage is by
PIN plat root (first three PIN groups, e.g. units `39-363-06-0xx` -> tract
`39-363-06-016`) and then by spatial adjacency, which is essential: several
developments' HOA tracts live in an ADJACENT plat root (root `49-173-03`'s units
sit on `49-172-06-191`, "SECOND GREEN MOUNTAIN TOWNHOUSE CORP", 18.3 acres).

*** The spatial test MUST be `intersects`, never `within`. *** The tract has each
unit footprint punched out as an interior ring, so a unit is never geometrically
`within` its own tract -- `within` matches ZERO units (measured). Holes are closed
with fill_holes() once the units are merged back in.

Guards that matter (each one is load-bearing; removing any re-breaks the map):
- A ground tract that has ANY unit on its plat root or adjacent to it is never
  dropped. The reverse -- dropping tracts that absorbed nothing -- is what
  deleted 533 acres of townhome ground in the 2026-08-20 build.
- `LGLSQFT <= 0` alone is NOT sufficient to call something a unit. 383 genuine
  fee-simple parcels merely lack a recorded `LGLSQFT` (e.g. `39-284-03-031`,
  Applewood Knolls, a real 1.28-acre lot; a 5.6-acre church). They are excluded
  by `TOTACR <= 0.10` -- a real lot states its acreage, a unit states ~0 -- plus
  a footprint cap for anything not stacked. They are additionally protected by
  the fact that nothing merges unless it links to a ground tract.
- Most ground tracts have a blank `TAXCLS`, but a handful are coded as vacant
  land (0xxx) with a nominal value and an association owner -- e.g. Sienna Park's
  `49-274-16-059`, 2.33 acres, $700, owner "SIENNA TOWNHOMES". Those count as
  tracts too, or 104 townhomes keep spiking.

Value aggregation, and why it differs from the account dedup below:
- Regime A/C unit values are GENUINE PER-UNIT ALLOCATIONS, so land and
  improvement values are SUMMED across a development's units.
- The `SCH` account dedup does the OPPOSITE (`first`), because there one
  account's single value is broadcast across its several GIS polygons and
  summing would inflate it N-fold.
  These two are easy to confuse. They are not the same operation.

*** KNOWN DATA LIMITATION (surfaced, not silently absorbed) ***
Regime C condo land value is genuinely absent from the Jeffco feed -- not hidden
in another field and not recoverable by any join. Of 5,800 class-1230 accounts in
the county pull, exactly ONE carries any land value. We therefore merge regime C
geometry but leave its land value at $0 rather than imputing one: CivicMapper
shows assessor data as published, and inventing land value in a public data
viewer is not acceptable. Those developments render flat/zero-height, which is
honest, and each carries a `land_value_basis` note explaining why so a viewer is
not left to conclude the land is worthless. ~5,200 condo parcels carrying ~$1.64B
of improvements are affected, so Lakewood's citywide land-value total
UNDERSTATES true land value. The ETL reports the magnitude so it can be cited.

Outputs
-------
- data/jurisidictions/data/lakewood/lakewood-co-parcels.parquet
- data/jurisidictions/data/lakewood/lakewood-co-parcels_YYYY_MM_DD.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Geod
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.parcel_calculations import add_improvement_ratio_fields  # noqa: E402

requests.packages.urllib3.disable_warnings()  # county host serves an incomplete chain

PARCEL_QUERY_URL = (
    "https://gisportal.jeffco.us/server2/rest/services/Parcel/FeatureServer/20/query"
)
BOUNDARY_QUERY_URL = (
    "https://gisportal.jeffco.us/server/rest/services/City/FeatureServer/4/query"
)
CITY_NAME = "LAKEWOOD"
PAGE_SIZE = 2000
UTM_EPSG = 32613  # UTM 13N, Colorado Front Range

OUTPUT_DIR = Path("data/jurisidictions/data/lakewood")
RAW_PATH = OUTPUT_DIR / "lakewood-co-raw-parcels.parquet"
OUT_PATH = OUTPUT_DIR / "lakewood-co-parcels.parquet"

RAW_FIELDS = [
    "OBJECTID", "PIN", "PINDESC", "SCH", "PARCELID", "AIN", "POLYCAT", "STS",
    "OWNNAM", "DBA", "PRPADDRESS", "PRPCTYNAM", "BPCTYCD", "SUBNAM", "NHDNAM",
    "TAXCLS", "TAXCLS2", "TAXCLS3", "VALACR", "VALACT", "VALACR2", "VALACT2",
    "TOTACR", "LGLSQFT", "TOTACTLNDV", "TOTACTIMPV", "TOTACTVAL",
    "STTSTRC", "STTTYPUSE", "STTYRBLT", "STTGRSAREA", "STTNBRUNT", "STTNBRBLDG",
    "ASMASDLND", "ASMASDIMP", "ASMASDTOT", "MILL_LEVY",
]

# Non-assessable PIN sentinels used by Jeffco for right-of-way / water polygons.
NON_PARCEL_PINS = {"ROW", "WATER"}

# Slivers below this are flagged so the frontend + hex bake can drop them; a real
# Lakewood lot starts around 2,000 sqft, so 500 is comfortably below any real lot.
REMNANT_SQFT = 500

# --- unit / ground-tract detection (see "The condo trap" in the module docstring) ---
# A unit account owns no ground in fee (LGLSQFT <= 0) AND the assessor's own
# structure type says it is attached housing. `STTSTRC` is the discriminator, not
# a size heuristic: an earlier attempt gated on TOTACR <= 0.10 acres + a footprint
# cap and it BOTH swallowed real detached homes on small lots (49-273-10-060, a
# 0.099-acre single-family lot with $127,500 of land, and two more on Ames Street)
# AND missed 270 genuine condo units whose stated acreage exceeds the cap (Belmar
# Plaza units state 0.816 acres each on a shared 35,000 sqft building footprint).
# Structure type gets both right, because it is what the assessor actually
# recorded rather than something inferred from geometry.
UNIT_TOWNHOME_STRUCTURE = "Townhomes"
# Every condo structure type in the feed: 'Condo, Res: Attached', 'Office/Condo',
# 'Warehouse/Condo', 'Retail/Condo', 'Condos, Res: Low Rise (1-3)',
# 'Industrial/Condo'. Matched as a substring so a new variant is picked up.
UNIT_CONDO_STRUCTURE_PAT = r"Condo"
# Colorado abstract classes for condominiums, as a backstop where STTSTRC is blank.
UNIT_CONDO_CLASSES = {"1230", "1212", "1225", "1250", "1240", "2245", "3230"}
# A ground tract coded as vacant land rather than left blank (the Sienna Park
# pattern) must be big, essentially valueless, and carry several units on its root.
TRACT_MIN_SQFT = 5_000
TRACT_MAX_LAND_VALUE = 10_000
TRACT_MIN_UNITS_IN_ROOT = 2
# Spatial-adjacency safety net: only merge a unit into a tract that dwarfs it.
TRACT_MIN_AREA_RATIO = 3.0

# Safety net for government land miscoded as taxable. The playbook warns that
# government parcels are frequently miscoded as commercial with zero
# exempt-value fields, so class 9xxx alone is not assumed sufficient.
#
# These are ANCHORED regexes, deliberately. Loose substring matching produced
# only false positives when audited against this feed: "JEFFCO" caught
# "JEFFCO RENTALS LLC", "USA " caught "BOKER USA INC" / "DESOUSA TRISTA" /
# "SIRAGUSA ANDREW", "FEDERAL" caught "FEDERAL NATIONAL MORTGAGE ASSOCIATION" --
# 19 private parcels wrongly excluded and zero genuine catches.
#
# Audit result (2026-08-20): class 9xxx already covers every genuinely public
# holding, including the former Denver Federal Center (17/17, class 9119) and
# all City of Lakewood park land incl. Bear Creek Lake Park and Belmar Park
# (400/400, classes 9149/9140/9148/9130). The only public-sounding owners
# OUTSIDE 9xxx are three parcels the assessor deliberately codes taxable: the
# Jefferson County Education Association (a private union), a single-family
# home held by the State of Colorado, and a Lakewood Housing Authority
# mixed-income complex. Those correctly stay in the taxable set.
GOV_OWNER_PATTERNS = [
    r"^CITY OF LAKEWOOD\b",
    # Exact county-government forms only. A bare "JEFFERSON COUNTY" prefix is
    # NOT enough: "JEFFERSON COUNTY EDUCATION ASSOCIATION" is a private union
    # that the assessor correctly codes taxable.
    r"^(JEFFERSON COUNTY|COUNTY OF JEFFERSON)(\s+(OF\s+)?COLORADO)?$",
    r"^JEFFERSON COUNTY OPEN SPACE\b",
    r"\bSCHOOL DISTRICT\b",
    r"^UNITED STATES OF AMERICA\b",
    r"\bDEPARTMENT OF TRANSPORTATION\b",
    r"^STATE HIGHWAY\b",
    r"\bREGIONAL TRANSPORTATION DISTRICT\b",
    r"\bURBAN DRAINAGE AND FLOOD\b",
    r"\bFIRE PROTECTION DISTRICT\b",
    r"^WEST METRO FIRE\b",
    r"\bLIBRARY DISTRICT\b",
    r"^REGENTS OF THE UNIVERSITY\b",
    r"^COLORADO SCHOOL OF MINES\b",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Lakewood canonical parcel parquet.")
    p.add_argument("--use-cache", action="store_true",
                   help="Reuse the cached raw parquet instead of re-downloading.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
def request_json(url: str, params: dict, *, timeout: int = 300) -> dict:
    r = requests.get(url, params=params, timeout=timeout, verify=False)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS error from {url}: {payload['error']}")
    return payload


def fetch_boundary() -> gpd.GeoDataFrame:
    payload = request_json(BOUNDARY_QUERY_URL, {
        "f": "geojson", "where": f"NAME='{CITY_NAME}'", "outFields": "NAME,CC",
        "returnGeometry": "true", "outSR": 4326,
    })
    gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    if gdf.empty:
        raise RuntimeError("Failed to fetch the Lakewood municipal boundary.")
    print(f"Boundary polygons: {len(gdf)}")
    return gdf


def download_parcels(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fetch every county parcel whose geometry intersects the Lakewood bbox.

    The bbox is a coarse server-side prefilter only; the authoritative
    centroid-within clip happens locally in restrict_to_city().
    """
    xmin, ymin, xmax, ymax = unary_union(boundary.geometry).bounds
    base = {
        "f": "geojson", "where": "1=1", "outFields": ",".join(RAW_FIELDS),
        "returnGeometry": "true", "outSR": 4326,
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope", "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects", "orderByFields": "OBJECTID",
    }
    total = request_json(PARCEL_QUERY_URL,
                         {**base, "f": "json", "returnCountOnly": "true"})["count"]
    print(f"Parcels intersecting the Lakewood bbox: {total:,}")

    frames, offset = [], 0
    while offset < total:
        for attempt in range(4):
            try:
                payload = request_json(PARCEL_QUERY_URL, {
                    **base, "resultOffset": offset, "resultRecordCount": PAGE_SIZE})
                feats = payload.get("features", [])
                break
            except Exception as exc:  # transient county-host timeouts
                print(f"   retry {attempt} at offset {offset}: {str(exc)[:110]}")
                time.sleep(4)
        else:
            raise RuntimeError(f"Failed to fetch page at offset {offset}")
        if not feats:
            break
        frames.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326"))
        offset += len(feats)
        print(f"  fetched {offset:,}/{total:,}")
        time.sleep(0.05)

    raw = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                           geometry="geometry", crs="EPSG:4326")
    for f in RAW_FIELDS:
        if f not in raw.columns:
            raw[f] = np.nan
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(RAW_PATH, index=False)
    print(f"Saved raw cache: {RAW_PATH} ({len(raw):,} rows)")
    return raw


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
_GEOD = Geod(ellps="WGS84")


def geodesic_area_sqft(geom) -> float:
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type == "Polygon":
        lon, lat = geom.exterior.coords.xy
        area_m2, _ = _GEOD.polygon_area_perimeter(lon, lat)
        holes = 0.0
        for ring in geom.interiors:
            lo, la = ring.coords.xy
            part, _ = _GEOD.polygon_area_perimeter(lo, la)
            holes += abs(part)
        return max(abs(area_m2) - holes, 0.0) * 10.763910416709722
    if geom.geom_type == "MultiPolygon":
        return sum(geodesic_area_sqft(p) for p in geom.geoms)
    return np.nan


def safe_union(geoms):
    """unary_union that survives the county's self-intersecting polygons.

    A handful of Jeffco tract polygons are topologically invalid, and GEOS raises
    "side location conflict" when unioning them. Repair with make_valid (buffer(0)
    as a last resort) and retry, so one bad ring cannot abort the whole merge.
    """
    gs = [g for g in geoms if g is not None and not g.is_empty]
    if not gs:
        return None
    try:
        return unary_union(gs)
    except Exception:
        pass
    repaired = []
    for g in gs:
        if g.is_valid:
            repaired.append(g)
            continue
        try:
            r = make_valid(g)
        except Exception:
            r = g.buffer(0)
        if r is not None and not r.is_empty:
            # make_valid can emit lines/points from degenerate rings; keep areas only.
            if r.geom_type in ("GeometryCollection", "MultiLineString", "LineString", "Point"):
                polys = [p for p in getattr(r, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
                r = unary_union(polys) if polys else None
            if r is not None and not r.is_empty:
                repaired.append(r)
    if not repaired:
        return None
    try:
        return unary_union(repaired)
    except Exception:
        return unary_union([g.buffer(0) for g in repaired])


def fill_holes(geom):
    """Drop interior rings.

    An association polygon is the development's land with each unit footprint
    punched out; once the units are merged in, those holes belong to the
    development and must be closed.
    """
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom


# --------------------------------------------------------------------------- #
# city restriction
# --------------------------------------------------------------------------- #
def restrict_to_city(gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Centroid-within clip on the authoritative municipal boundary + validation."""
    bnd = unary_union(boundary.geometry)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    inside = gdf.geometry.representative_point().within(bnd)

    kept, dropped = gdf[inside].copy(), gdf[~inside]
    print("\n--- city-limits restriction ---")
    print(f"Parcels centroid-within Lakewood: {len(kept):,} (dropped {len(dropped):,})")
    print("Kept, by county jurisdiction code (L = Lakewood):")
    print(kept["BPCTYCD"].fillna("NULL").value_counts().head(8).to_string())
    print("Dropped, by county jurisdiction code (neighbouring municipalities):")
    print(dropped["BPCTYCD"].fillna("NULL").value_counts().head(10).to_string())
    lakewood_addr_dropped = int(
        dropped["PRPCTYNAM"].fillna("").astype(str).str.upper().eq("LAKEWOOD").sum())
    print(f"Dropped parcels carrying a 'LAKEWOOD' situs address: "
          f"{lakewood_addr_dropped:,} (mailing city != jurisdiction; correct to drop)")
    stray = int(kept["BPCTYCD"].fillna("").eq("L").sum())
    print(f"Kept parcels coded BPCTYCD='L': {stray:,}")
    print(f"Bounds: {[round(v, 4) for v in kept.total_bounds]}")
    return kept


# --------------------------------------------------------------------------- #
# condo / common-area handling
# --------------------------------------------------------------------------- #
def plat_root(pin: pd.Series) -> pd.Series:
    """First three PIN groups -- the plat root a development's units share."""
    return pin.fillna("").astype(str).str.strip().str.split("-").str[:3].str.join("-")


def identify_units_and_tracts(gdf: gpd.GeoDataFrame) -> tuple[pd.Series, pd.Series]:
    """Split the frame into condo/townhome UNITS and their GROUND TRACTS.

    Keyed off `LGLSQFT` (deeded legal area), not `POLYCAT`. See the module
    docstring: `POLYCAT in (2,3)` misses 6,535 of the 6,741 regime-A townhome
    units, which is what produced the spikes-and-gaps artifact.
    """
    tax = gdf["_tax"]
    pin_ok = (~gdf["_pin"].isin(NON_PARCEL_PINS)) & (gdf["_pin"] != "")
    lgl = pd.to_numeric(gdf["LGLSQFT"], errors="coerce").fillna(0)
    land = pd.to_numeric(gdf["TOTACTLNDV"], errors="coerce").fillna(0)
    area = gdf["_fp_sqft"]
    strc = gdf["STTSTRC"].fillna("").astype(str).str.strip()

    # No ground held in fee -> this account is a candidate unit...
    cand = (tax != "") & pin_ok & (lgl <= 0)
    # ...and the assessor's structure type decides whether it really is one.
    stacked = gdf["_pt_key"].map(gdf["_pt_key"][cand].value_counts()).fillna(0) > 1
    is_unit = cand & (
        strc.eq(UNIT_TOWNHOME_STRUCTURE)
        | strc.str.contains(UNIT_CONDO_STRUCTURE_PAT, case=False, na=False)
        | tax.isin(UNIT_CONDO_CLASSES)
        | (strc.eq("") & stacked)  # unlabelled stubs piled on one footprint
    )

    units_per_root = gdf.loc[is_unit].groupby("_root")["_pin"].size()
    root_units = gdf["_root"].map(units_per_root).fillna(0)

    # Ground tracts: normally a blank TAXCLS (captured BEFORE any exempt/blank
    # filter, per the playbook, so they survive to participate in the merge)...
    is_tract = (tax == "") & pin_ok
    # ...but a few are coded as vacant land with a nominal value and an
    # association owner (Sienna Park's 49-274-16-059: 2.33 ac, $700).
    is_tract |= (
        tax.str.startswith("0")
        & pin_ok
        & (area > TRACT_MIN_SQFT)
        & (land <= TRACT_MAX_LAND_VALUE)
        & (root_units >= TRACT_MIN_UNITS_IN_ROOT)
    )
    is_tract &= ~is_unit  # a tract is never also a unit
    # A PIN is ONE parcel that may be drawn as several polygons. If any of them is
    # the development's ground, they all are -- otherwise a twin row is left behind
    # in `others`, and the SCH dedup below (which takes `first`) can discard the
    # merged development's summed value in favour of the leftover's nominal one.
    # Measured: without this, PIN 49-124-26-008 lost $563,500.
    tract_pins = set(gdf.loc[is_tract, "_pin"])
    is_tract |= gdf["_pin"].isin(tract_pins) & ~is_unit

    print("\n--- unit / ground-tract identification ---")
    print(f"Candidate no-fee-ground accounts (LGLSQFT<=0, classed): {int(cand.sum()):,}")
    print(f"  -> units (attached/condo per STTSTRC):               {int(is_unit.sum()):,}")
    excluded = cand & ~is_unit
    print(f"  -> excluded as genuine parcels missing LGLSQFT:       {int(excluded.sum()):,} "
          f"(${land[excluded].sum() / 1e6:,.1f}M land preserved in place)")
    print("     their structure types: "
          + ", ".join(f"{k or '(blank)'} {v}" for k, v in
                      strc[excluded].value_counts().head(5).items()))
    print(f"Ground tracts captured: {int(is_tract.sum()):,} "
          f"(blank TAXCLS {int(((tax == '') & pin_ok & ~is_unit).sum()):,}, "
          f"vacant-coded {int((is_tract & (tax != '')).sum()):,})")
    old_test = int((gdf["POLYCAT"].isin([2, 3]) & (tax != "")).sum())
    print(f"[regression guard] the old POLYCAT-based test would have found only "
          f"{old_test:,} units")
    return is_unit, is_tract


def merge_condo_developments(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge condo/townhome unit parcels down onto their ground tract.

    Ground tracts (blank TAXCLS, a real PIN, real footprint, $0 value) are
    captured BEFORE any exempt/blank filtering, per the playbook, so they can
    still participate in the merge.
    """
    gdf = gdf.copy()
    gdf["_pin"] = gdf["PIN"].fillna("").astype(str).str.strip()
    gdf["_tax"] = gdf["TAXCLS"].fillna("").astype(str).str.strip()
    gdf["_root"] = plat_root(gdf["_pin"])
    # Projected footprint area + a stacking key, needed by the unit test.
    gdf["_fp_sqft"] = gdf.to_crs(UTM_EPSG).geometry.area * 10.763910416709722
    _rp = gdf.geometry.representative_point()
    gdf["_pt_key"] = _rp.x.round(6).astype(str) + "," + _rp.y.round(6).astype(str)

    is_unit, is_assoc = identify_units_and_tracts(gdf)

    assoc = gdf[is_assoc].copy()
    units = gdf[is_unit].copy()
    others = gdf[~is_assoc & ~is_unit].copy()

    # A plat's common area can be several polygons sharing one PIN; dissolve them
    # so each association PIN is a single merge target.
    if assoc["_pin"].duplicated().any():
        n_before = len(assoc)
        geom = assoc.groupby("_pin")["geometry"].apply(safe_union)
        assoc = assoc.drop_duplicates("_pin").set_index("_pin")
        assoc["geometry"] = geom
        assoc = gpd.GeoDataFrame(assoc.reset_index(), geometry="geometry", crs=gdf.crs)
        print(f"Dissolved multi-polygon association plats: {n_before:,} -> {len(assoc):,}")

    print("\n--- condo / townhome merge ---")
    print(f"Ground tracts available as merge targets: {len(assoc):,} "
          f"({assoc['_fp_sqft'].sum() / 43560:,.1f} acres)")
    print(f"Unit parcels to merge down: {len(units):,} "
          f"(carrying ${units['TOTACTLNDV'].pipe(pd.to_numeric, errors='coerce').fillna(0).sum() / 1e6:,.1f}M of land value "
          f"on {units['_fp_sqft'].sum() / 43560:,.1f} acres of footprint)")

    # smoke alarm: how much stacking are we actually fixing?
    vc = gdf["_pt_key"].value_counts()
    print(f"Stacked clusters before merge: {int((vc > 1).sum()):,} (max stack {int(vc.max())})")

    if units.empty or assoc.empty:
        print("Nothing to merge.")
        return gdf.drop(columns=["_pin", "_tax", "_root"])

    # Link unit -> ground tract: plat root first, then spatial ADJACENCY.
    # Where a root holds several tracts, the largest is the development's ground.
    root_to_assoc = (assoc.sort_values("_fp_sqft", ascending=False)
                     .drop_duplicates("_root").set_index("_root")["_pin"].to_dict())
    units["_target"] = units["_root"].map(root_to_assoc)
    by_root = int(units["_target"].notna().sum())

    # Spatial fallback. `intersects`, NOT `within`: the tract has every unit
    # footprint punched out as an interior ring, so no unit is ever `within` its
    # own tract (measured: `within` matches zero). Several developments' HOA
    # tracts genuinely live in an adjacent plat root, which is what this catches.
    missing = units[units["_target"].isna()]
    if len(missing):
        a3 = assoc.to_crs(UTM_EPSG)[["_pin", "geometry", "_fp_sqft"]].rename(
            columns={"_pin": "_apin", "_fp_sqft": "_a_sqft"})
        m3 = missing.to_crs(UTM_EPSG)[["_pin", "geometry", "_fp_sqft"]]
        j = gpd.sjoin(m3, a3, how="inner", predicate="intersects")
        # Safety net: only merge into a tract that dwarfs the unit, so an ordinary
        # parcel that merely abuts some tract can never be swallowed by it.
        j = j[j["_a_sqft"] >= TRACT_MIN_AREA_RATIO * j["_fp_sqft"]]
        j = j.sort_values("_a_sqft", ascending=False).drop_duplicates("_pin")
        fill = dict(zip(j["_pin"], j["_apin"]))
        units.loc[units["_target"].isna(), "_target"] = (
            units.loc[units["_target"].isna(), "_pin"].map(fill))
    by_spatial = int(units["_target"].notna().sum()) - by_root
    unmatched = units[units["_target"].isna()]
    print(f"Linked by PIN plat root: {by_root:,}; by spatial adjacency: {by_spatial:,}; "
          f"unmatched: {len(unmatched):,}")
    if len(unmatched):
        u_land = pd.to_numeric(unmatched["TOTACTLNDV"], errors="coerce").fillna(0)
        print(f"  unmatched detail: ${u_land.sum() / 1e6:,.2f}M land on "
              f"{unmatched['_fp_sqft'].sum() / 43560:,.2f} acres of footprint; "
              f"{int((u_land > 0).sum()):,} carry land value and keep a "
              f"footprint-based $/sqft (left as published, not imputed)")
        print(unmatched.groupby("_tax")["_pin"].size().head(6).to_string())

    matched = units[units["_target"].notna()].copy()
    val_cols = ["TOTACTLNDV", "TOTACTIMPV", "TOTACTVAL"]
    for c in val_cols:
        matched[c] = pd.to_numeric(matched[c], errors="coerce").fillna(0)
        assoc[c] = pd.to_numeric(assoc[c], errors="coerce").fillna(0)

    # Land and improvement values are SUMMED here -- these are genuine per-unit
    # allocations, unlike the account dedup below, where one account's value is
    # broadcast across its polygons and `first` is correct. Do not conflate them.
    agg = matched.groupby("_target").agg(
        _n_units=("_pin", "size"),
        _lnd=("TOTACTLNDV", "sum"),
        _imp=("TOTACTIMPV", "sum"),
        _val=("TOTACTVAL", "sum"),
        _tax_mode=("_tax", lambda s: s.value_counts().index[0]),
        _sub=("SUBNAM", "first"),
        _use=("STTTYPUSE", "first"),
        _strc=("STTSTRC", lambda s: (s.dropna().value_counts().index[0]
                                     if s.notna().any() else None)),
        _units_geom=("geometry", safe_union),
    )

    assoc = assoc.set_index("_pin")
    hit = assoc.index.intersection(agg.index)
    # the merged development footprint: association land + unit footprints, holes closed
    new_geom = []
    for pin in hit:
        g = safe_union([assoc.loc[pin, "geometry"], agg.loc[pin, "_units_geom"]])
        new_geom.append(fill_holes(g))
    assoc.loc[hit, "geometry"] = gpd.GeoSeries(new_geom, index=hit, crs=assoc.crs)
    for c, src in [("TOTACTLNDV", "_lnd"), ("TOTACTIMPV", "_imp"), ("TOTACTVAL", "_val")]:
        assoc.loc[hit, c] = assoc.loc[hit, c].values + agg.loc[hit, src].values
    # give the merged development the units' dominant class + structure type, so a
    # townhome development classifies as attached housing rather than as one
    # enormous single-family home.
    assoc.loc[hit, "TAXCLS"] = agg.loc[hit, "_tax_mode"].values
    assoc.loc[hit, "SUBNAM"] = assoc.loc[hit, "SUBNAM"].fillna(
        pd.Series(agg.loc[hit, "_sub"].values, index=hit))
    assoc.loc[hit, "STTTYPUSE"] = assoc.loc[hit, "STTTYPUSE"].fillna(
        pd.Series(agg.loc[hit, "_use"].values, index=hit))
    assoc.loc[hit, "STTSTRC"] = pd.Series(agg.loc[hit, "_strc"].values, index=hit)
    assoc["_merged_units"] = 0
    assoc.loc[hit, "_merged_units"] = agg.loc[hit, "_n_units"].values
    assoc = assoc.reset_index().rename(columns={"index": "_pin"})

    print(f"Developments formed: {len(hit):,} "
          f"(absorbing {int(agg.loc[hit, '_n_units'].sum()):,} units, "
          f"${agg.loc[hit, '_lnd'].sum() / 1e6:,.1f}M land, "
          f"{assoc.loc[assoc['_merged_units'] > 0, '_fp_sqft'].sum() / 43560:,.1f} acres "
          f"of tract ground recovered)")

    # Ground tracts that absorbed NOTHING are normally open space / drainage /
    # medians with no class and no value -- not assessable parcels, so they are
    # dropped. Two exceptions are retained, because dropping ground that sits
    # under or among housing is exactly the defect this build fixes (the
    # 2026-08-20 build deleted 533 acres of live townhome ground):
    #   1. tracts that absorbed units (regimes A and C), and
    #   2. tracts serving FEE-SIMPLE townhomes (regime D). Those units hold their
    #      ~1,300 sqft footprint in fee (LGLSQFT > 0), so they are correctly
    #      mapped and are NOT merged -- but their HOA common ground is still a
    #      real, separately-deeded polygon and must not vanish. It is kept at the
    #      county's published $0 land value; nothing is imputed.
    attached = (gdf["STTSTRC"].fillna("").astype(str) == "Townhomes")
    attached_roots = set(gdf.loc[attached, "_root"])
    retain = (assoc["_merged_units"] > 0) | assoc["_root"].isin(attached_roots)
    if (~retain).any():
        cand_tracts = assoc[~retain]
        att = gdf.loc[attached, ["geometry"]].to_crs(UTM_EPSG)
        if len(att):
            ct = cand_tracts[["geometry"]].to_crs(UTM_EPSG)
            touched = set(gpd.sjoin(ct, att, how="inner", predicate="intersects").index)
            retain |= assoc.index.isin(touched)
    assoc["_common_area"] = ((assoc["_merged_units"] == 0) & retain).astype(int)

    leftover = assoc[~retain]
    kept_ca = assoc[assoc["_common_area"] == 1]
    print(f"Unmerged tracts RETAINED as HOA common area beside fee-simple townhomes: "
          f"{len(kept_ca):,} ({kept_ca['_fp_sqft'].sum() / 43560:,.1f} acres, "
          f"published land value $0)")
    print(f"Unmerged tracts dropped (genuinely valueless open space/drainage): "
          f"{len(leftover):,} ({leftover['_fp_sqft'].sum() / 43560:,.1f} acres, "
          f"${pd.to_numeric(leftover['TOTACTLNDV'], errors='coerce').fillna(0).sum():,.0f} land value)")
    assoc = assoc[retain]

    out = pd.concat([others, assoc, unmatched], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)

    key = out.geometry.representative_point()
    k = key.x.round(5).astype(str) + "," + key.y.round(5).astype(str)
    vc = k.value_counts()
    print(f"Stacked clusters after merge: {int((vc > 1).sum()):,} "
          f"(max stack {int(vc.max()) if len(vc) else 0})")
    return out.drop(columns=["_pin", "_tax", "_root", "_units_geom", "_target",
                             "_fp_sqft", "_pt_key"], errors="ignore")


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #
def collapse_duplicate_accounts(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Collapse multiple GIS polygons sharing one assessor schedule number.

    Values are account-level and broadcast identically across an account's
    polygons, so they take `first` -- NEVER `sum` (summing is the N x
    value-inflation bug). Geometry is unioned.
    """
    gdf = gdf.copy()
    sch = gdf["SCH"].fillna("").astype(str).str.strip()
    has = sch != ""
    dup_mask = has & sch.duplicated(keep=False)
    print("\n--- account dedup ---")
    print(f"Rows sharing a schedule number: {int(dup_mask.sum()):,}")
    if not dup_mask.any():
        return gdf

    gdf["_sch"] = sch
    single = gdf[~dup_mask]
    multi = gdf[dup_mask]

    value_cols = ["TOTACTLNDV", "TOTACTIMPV", "TOTACTVAL",
                  "ASMASDLND", "ASMASDIMP", "ASMASDTOT", "TOTACR", "LGLSQFT"]
    other_cols = [c for c in multi.columns
                  if c not in set(value_cols + ["geometry", "_sch"])]
    agg = {c: "first" for c in value_cols}          # account-level -> first
    agg.update({c: "first" for c in other_cols})
    collapsed = multi.groupby("_sch", dropna=False).agg(agg).reset_index()
    geom = multi.groupby("_sch", dropna=False)["geometry"].apply(safe_union)
    collapsed["geometry"] = geom.values

    out = pd.concat([single, gpd.GeoDataFrame(collapsed, geometry="geometry", crs=gdf.crs)],
                    ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)
    print(f"{len(multi):,} duplicate rows -> {len(collapsed):,} accounts; "
          f"total {len(gdf):,} -> {len(out):,}")
    return out.drop(columns=["_sch"], errors="ignore")


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def classify_category(tax: str, use: str, impr: float, strc: str = "") -> str:
    """Lakewood/Jeffco-specific mapping from the Colorado 4-digit abstract class.

    `STTSTRC` (the assessor's own structure type) overrides the abstract class for
    attached housing. Colorado codes a townhome 1112, the same as a detached house,
    so class alone labelled 6,586 townhome units and 1,444 fee-simple townhome lots
    "Single Family Residential". The assessor does distinguish them -- STTSTRC is
    literally 'Townhomes' -- so we use it.
    """
    t = (tax or "").strip()
    u = (use or "").strip()
    s = (strc or "").strip()
    if not t:
        return "Unclassified"
    if t.startswith("9"):
        return "Exempt"
    if t.startswith("0"):
        return "Vacant Land"
    if t.startswith("1"):
        if t in {"1230", "1212", "1225", "1250", "1240"}:
            return "Residential Condominium"
        if t in {"1120", "1125", "1150", "1177"}:
            return "Multi-Family Residential"
        if t == "1140":
            return "Residential Other"
        if s == "Townhomes":
            return "Townhome / Attached Residential"
        return "Single Family Residential"
    if t.startswith("2"):
        if t == "2245":
            return "Commercial Condominium"
        if t == "2115" or "Hotel" in u or "Motel" in u:
            return "Lodging"
        if t == "2135":
            return "Industrial"
        if t == "2120":
            return "Office"
        if t in {"2125", "2150"}:
            return "Commercial Other"
        return "Commercial"
    if t.startswith("3"):
        return "Industrial"
    if t.startswith("4"):
        return "Agricultural"
    if t.startswith("5"):
        return "Natural Resources"
    return "Other"


def classify_refined(row: pd.Series) -> str | None:
    """Vacant / Parking Lot / Underdeveloped -- drives the underutilization tab."""
    cat = str(row.get("property_land_use_category") or "")
    use = str(row.get("STTTYPUSE") or "")
    land = float(row.get("current_full_land_value") or 0)
    impr = float(row.get("improvement_value") or 0)
    total = land + impr

    if "Parking" in use:
        return "Parking Lot"
    if cat == "Vacant Land":
        return "Vacant"
    # A parcel with land value and no structure at all is vacant in substance.
    if impr <= 0 and land > 0:
        return "Vacant"
    # Underdeveloped: improvements worth less than the land beneath them.
    # Condos are excluded -- Jeffco records $0 land for them, so the ratio is
    # meaningless and every condo would otherwise look fully developed/undeveloped.
    if "Condominium" in cat:
        return None
    if total > 0 and land > 0 and impr < 0.5 * total:
        return "Underdeveloped"
    return None


def flag_exempt(gdf: gpd.GeoDataFrame) -> pd.Series:
    """exemption_flag: class 9xxx, plus a government-owner safety net."""
    tax = gdf["TAXCLS"].fillna("").astype(str).str.strip()
    owner = gdf["OWNNAM"].fillna("").astype(str).str.upper()
    by_class = tax.str.startswith("9")
    by_owner = pd.Series(False, index=gdf.index)
    for pat in GOV_OWNER_PATTERNS:
        by_owner |= owner.str.contains(pat, regex=True, na=False)
    flag = (by_class | by_owner).astype(int)
    print("\n--- exemptions ---")
    print(f"Exempt by abstract class 9xxx: {int(by_class.sum()):,}")
    print(f"Additional exempt caught by government-owner heuristic: "
          f"{int((by_owner & ~by_class).sum()):,}")
    print(f"Total exempt: {int(flag.sum()):,}")
    sample = gdf.loc[(by_owner & ~by_class), "OWNNAM"].value_counts().head(8)
    if len(sample):
        print("Owner-heuristic catches (top):")
        print(sample.to_string())
    return flag


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_export(raw: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = restrict_to_city(raw, boundary)

    # Drop right-of-way / water sentinels: not assessable parcels.
    pin = gdf["PIN"].fillna("").astype(str).str.strip()
    n_row = int(pin.isin(NON_PARCEL_PINS).sum())
    gdf = gdf[~pin.isin(NON_PARCEL_PINS)].copy()
    print(f"\nDropped non-parcel ROW/WATER polygons: {n_row:,} -> {len(gdf):,} rows")

    gdf = merge_condo_developments(gdf)
    gdf = collapse_duplicate_accounts(gdf)

    for c in ["TOTACTLNDV", "TOTACTIMPV", "TOTACTVAL"]:
        gdf[c] = pd.to_numeric(gdf[c], errors="coerce").fillna(0)

    gdf["exemption_flag"] = flag_exempt(gdf)

    def clean(s: pd.Series) -> pd.Series:
        t = s.fillna("").astype(str).str.strip()
        return t.mask(t.eq(""), np.nan)

    gdf["parcel_id"] = clean(gdf["PIN"])
    gdf["schedule_number"] = clean(gdf["SCH"])
    gdf["owner"] = clean(gdf["OWNNAM"])
    gdf["property_address"] = clean(gdf["PRPADDRESS"])
    gdf["subdivision"] = clean(gdf["SUBNAM"])
    gdf["neighborhood"] = clean(gdf["NHDNAM"])
    gdf["tax_class"] = clean(gdf["TAXCLS"])
    gdf["structure_use"] = clean(gdf["STTTYPUSE"])
    gdf["structure_type"] = clean(gdf["STTSTRC"])
    gdf["merged_unit_count"] = (
        pd.to_numeric(gdf["_merged_units"], errors="coerce").fillna(0).astype(int)
        if "_merged_units" in gdf.columns else 0)
    gdf["year_built"] = pd.to_numeric(gdf["STTYRBLT"], errors="coerce")
    gdf["building_sqft"] = pd.to_numeric(gdf["STTGRSAREA"], errors="coerce")
    gdf["assessor_url"] = np.where(
        gdf["schedule_number"].notna(),
        "https://jeffco.us/assessor/property-records-search/?schedule="
        + gdf["schedule_number"].fillna("").astype(str),
        None,
    )

    gdf["property_land_use_category"] = [
        classify_category(t, u, i, s) for t, u, i, s in
        zip(gdf["TAXCLS"].fillna(""), gdf["STTTYPUSE"].fillna(""), gdf["TOTACTIMPV"],
            gdf["STTSTRC"].fillna(""))
    ]

    # Retained HOA common-area tracts have a blank TAXCLS, which would otherwise
    # classify as "Unclassified" and be dropped again. Name them for what they are.
    if "_common_area" in gdf.columns:
        ca = pd.to_numeric(gdf["_common_area"], errors="coerce").fillna(0) > 0
        gdf.loc[ca, "property_land_use_category"] = "Common Area (HOA)"
        print(f"\nRetained HOA common-area tracts categorised: {int(ca.sum()):,}")

    # Per-parcel provenance for the land-value figure. Regime C condos genuinely
    # have NO published land value; rather than silently rendering them as
    # zero-height "worthless" land, every parcel says where its number came from.
    # This is a plain data column surfaced through the city dictionary -- no
    # frontend change needed (only dictionary fields reach the popup).
    merged_n = (pd.to_numeric(gdf["_merged_units"], errors="coerce").fillna(0)
                if "_merged_units" in gdf.columns else pd.Series(0.0, index=gdf.index))
    gdf["land_value_basis"] = np.where(
        (gdf["TOTACTLNDV"] <= 0) & gdf["property_land_use_category"].str.contains("Condominium"),
        "Not published: Jefferson County books condominium value entirely to "
        "improvements and records no land value for these units",
        np.where(
            gdf["property_land_use_category"].eq("Common Area (HOA)"),
            "Common ground held by a homeowners association; Jefferson County "
            "publishes no separate land value for it",
            np.where(
                merged_n > 0,
                "Assessor land value, summed across " + merged_n.astype(int).astype(str)
                + " attached units on their common ground tract",
                "Assessor land value for this parcel",
            ),
        ),
    )

    gdf["current_full_land_value"] = gdf["TOTACTLNDV"].clip(lower=0)
    gdf["improvement_value"] = gdf["TOTACTIMPV"].clip(lower=0)
    gdf["full_market_value"] = gdf["current_full_land_value"] + gdf["improvement_value"]

    # Areas are geodesic from geometry: Jeffco's TOTACR is unusable (see docstring).
    gdf["area_sqft"] = gdf.geometry.apply(geodesic_area_sqft)
    gdf["land_area_sqft"] = gdf["area_sqft"]
    gdf["stated_acres"] = pd.to_numeric(gdf["TOTACR"], errors="coerce")

    gdf["property_land_use_refined"] = gdf.apply(classify_refined, axis=1)

    gdf["land_value_per_sqft"] = gdf["current_full_land_value"] / gdf["area_sqft"]
    gdf["improvement_value_per_sqft"] = gdf["improvement_value"] / gdf["area_sqft"]
    gdf["full_market_value_per_sqft"] = gdf["full_market_value"] / gdf["area_sqft"]
    gdf["REALLANDVA"] = gdf["current_full_land_value"]
    gdf["REALIMPROV"] = gdf["improvement_value"]
    gdf["REALLANDVA_per_sqft"] = gdf["land_value_per_sqft"]
    gdf["REALIMPROV_per_sqft"] = gdf["improvement_value_per_sqft"]
    gdf = add_improvement_ratio_fields(gdf, land_col="REALLANDVA",
                                       improvement_col="REALIMPROV")
    gdf["TLLDIMPROV_per_sqft"] = gdf["full_market_value_per_sqft"]
    gdf["likely_remnant"] = (gdf["area_sqft"] < REMNANT_SQFT).astype(int)

    print("\n--- category value counts (pre-exempt-removal) ---")
    print(gdf["property_land_use_category"].value_counts().to_string())

    before = len(gdf)
    gdf = gdf[gdf["exemption_flag"] == 0].copy()
    print(f"\nRemoved {before - len(gdf):,} exempt parcels -> {len(gdf):,} remaining")

    # Drop categories the app does not ship.
    drop_cats = {"Natural Resources", "Unclassified"}
    n = len(gdf)
    gdf = gdf[~gdf["property_land_use_category"].isin(drop_cats)].copy()
    print(f"Removed {n - len(gdf):,} rows in non-shipped categories {sorted(drop_cats)}")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf["area_sqft"] > 0].copy()

    print("\n--- FINAL category value counts ---")
    print(gdf["property_land_use_category"].value_counts().to_string())
    print("\n--- FINAL refined value counts ---")
    print(gdf["property_land_use_refined"].value_counts(dropna=False).to_string())

    keep = [
        "geometry", "parcel_id", "schedule_number", "owner", "property_address",
        "subdivision", "neighborhood", "tax_class", "structure_use", "structure_type",
        "year_built", "building_sqft", "assessor_url",
        "land_value_basis", "merged_unit_count",
        "property_land_use_category", "property_land_use_refined",
        "current_full_land_value", "improvement_value", "full_market_value",
        "area_sqft", "land_area_sqft", "stated_acres",
        "land_value_per_sqft", "improvement_value_per_sqft", "full_market_value_per_sqft",
        "REALLANDVA", "REALIMPROV", "REALLANDVA_per_sqft", "REALIMPROV_per_sqft",
        "TLLDIMPROV", "TLLDIMPROV_per_sqft",
        "IMPR_LAND_RATIO", "IMPR_LAND_PCT", "IMPR_PCT_TOTAL",
        "exemption_flag", "likely_remnant",
    ]
    out = gpd.GeoDataFrame(gdf[[c for c in keep if c in gdf.columns]],
                           geometry="geometry", crs="EPSG:4326")
    return out


def report_quality(out: gpd.GeoDataFrame) -> None:
    """Playbook smoke alarms -- run on the FINAL frame before export."""
    print("\n================ QUALITY REPORT ================")
    a = out["area_sqft"]
    lvps = out["land_value_per_sqft"].replace([np.inf, -np.inf], np.nan)
    print("footprint sqft p1/p5/p50/p99: %s" %
          [round(a.quantile(q)) for q in (.01, .05, .50, .99)])
    print("sub-500 sqft: %d   sub-1000 sqft: %d" % (int((a < 500).sum()), int((a < 1000).sum())))
    print("land $/sqft p50/p99/max: %.1f / %.1f / %.1f" %
          (lvps.median(), lvps.quantile(.99), lvps.max()))
    holes = out.geometry.apply(
        lambda g: 0 if g is None else
        sum(len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
    print("parcels with holes: %d" % int((holes > 0).sum()))

    key = out.geometry.representative_point()
    k = key.x.round(5).astype(str) + "," + key.y.round(5).astype(str)
    vc = k.value_counts()
    print("stacked clusters: %d (max stack %d)" %
          (int((vc > 1).sum()), int(vc.max()) if len(vc) else 0))

    land = out["current_full_land_value"].sum()
    impr = out["improvement_value"].sum()
    acres = out["area_sqft"].sum() / 43560.0
    print("\nCITY TOTALS")
    print("  parcels:          %d" % len(out))
    print("  land value:       $%.0f" % land)
    print("  improvement val:  $%.0f" % impr)
    print("  land share:       %.1f%%" % (100 * land / (land + impr)))
    print("  acres:            %.1f" % acres)
    print("  land $/acre:      $%.0f" % (land / acres))

    # --- the spikes-and-gaps regression check -------------------------------- #
    # The 3D height field is land_value_per_sqft. When townhome units sit on a
    # footprint with their ground tract deleted, the extreme tail of this
    # distribution is entirely artifact: ~800 sqft polygons carrying a full
    # unit's land value. A healthy build has a tail dominated by genuinely small
    # or genuinely valuable parcels, not by uniform sub-1000 sqft footprints.
    print("\nLAND $/SQFT TAIL COMPOSITION (artifact detector)")
    v = out.assign(_ppsf=lvps).dropna(subset=["_ppsf"])
    v = v[v["_ppsf"] > 0]
    for label, q in [("top 1%", 0.99), ("top 0.1%", 0.999)]:
        tail = v[v["_ppsf"] >= v["_ppsf"].quantile(q)]
        if not len(tail):
            continue
        small = int((tail["area_sqft"] < 1500).sum())
        print(f"  {label:9s} n={len(tail):5d}  min ${tail['_ppsf'].min():,.0f}/sqft  "
              f"median footprint {tail['area_sqft'].median():,.0f} sqft  "
              f"sub-1500-sqft share {100 * small / len(tail):5.1f}%")
        print(f"    {'':7s} top categories: "
              f"{', '.join(f'{k} {n}' for k, n in tail['property_land_use_category'].value_counts().head(3).items())}")

    print("\nNAMED-DEVELOPMENT SPOT CHECKS (land $/sqft; artifacts read high)")
    for name in ["HAMPDEN VILLA", "GREEN MOUNTAIN TOWNHOUSE", "PHEASANT CREEK",
                 "JEFFERSON GREEN", "VICTORIA VILLAGE", "SAN FRANCISCO WEST",
                 "VILLA WEST", "AMMONS PARK", "KIPLING KLUB", "WALKER PARK",
                 "SIENNA PARK"]:
        sub = out[out["subdivision"].fillna("").str.upper().str.contains(name, na=False)]
        if not len(sub):
            print(f"  {name:26s} -- no parcels")
            continue
        land = sub["current_full_land_value"].sum()
        ac = sub["area_sqft"].sum() / 43560
        ppsf = land / sub["area_sqft"].sum() if sub["area_sqft"].sum() else float("nan")
        units = int(sub.get("merged_unit_count", pd.Series(0, index=sub.index)).sum())
        print(f"  {name:26s} parcels={len(sub):4d} units={units:4d} "
              f"land=${land / 1e6:7.2f}M acres={ac:7.2f} ${ppsf:7.1f}/sqft")

    zero_land = out[out["current_full_land_value"] <= 0]
    print("\nZERO-LAND-VALUE PARCELS (the Jeffco condo limitation)")
    print("  count: %d  (%.1f%% of parcels)" % (len(zero_land), 100 * len(zero_land) / len(out)))
    print("  improvements they carry: $%.0f" % zero_land["improvement_value"].sum())
    print("  acres they occupy:       %.1f" % (zero_land["area_sqft"].sum() / 43560.0))
    print("  by category:")
    print(zero_land["property_land_use_category"].value_counts().head(8).to_string())
    print("================================================\n")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    boundary = fetch_boundary()

    if args.use_cache and RAW_PATH.exists():
        raw = gpd.read_parquet(RAW_PATH)
        raw = gpd.GeoDataFrame(raw, geometry="geometry", crs="EPSG:4326")
        print(f"Loaded raw cache: {RAW_PATH} ({len(raw):,} rows)")
    else:
        raw = download_parcels(boundary)

    out = build_export(raw, boundary)
    report_quality(out)

    out.to_parquet(OUT_PATH, index=False)
    stamp = datetime.now().strftime("%Y_%m_%d")
    snap = OUTPUT_DIR / f"lakewood-co-parcels_{stamp}.parquet"
    out.to_parquet(snap, index=False)
    print(f"Wrote {OUT_PATH} ({len(out):,} parcels)")
    print(f"Wrote {snap}")


if __name__ == "__main__":
    main()
