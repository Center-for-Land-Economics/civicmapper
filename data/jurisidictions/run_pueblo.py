#!/usr/bin/env python3
"""
Build Pueblo's canonical parcel parquet from live Pueblo County services.

Sources:
- Parcels (authoritative, ArcGIS Online):
  https://services1.arcgis.com/IL17xsvNU5Bmw3RY/ArcGIS/rest/services/County_Parcels/FeatureServer/0
- Municipal boundaries:
  https://maps.co.pueblo.co.us/outside/rest/services/Landbase/PuebloCounty_MunicipalCountyBoundaries/MapServer/0
- City zoning:
  https://maps.co.pueblo.co.us/outside/rest/services/Landbase/PuebloCounty_ZoningCountyCity/MapServer/0
- Per-parcel ground truth used while developing this ETL (no bulk endpoint):
  https://puebloco-search.gsacorp.io/parcel/<PAR_TXT>  (states "Acreage", the
  deeded area, which is what proved the sliver diagnosis below)

Why the parcel source moved (probed live 2026-08-26)
----------------------------------------------------
The old source, `maps.co.pueblo.co.us/.../PuebloCounty_Parcels/MapServer/1`,
renamed its columns: `Owner` -> `OwnerName`, `Neighborhood` -> `Neighborho`,
`LandActualValue` -> `LandActual`, `ImprovementsActualValue` -> `Improvem_1`,
and `AssessorURL` is gone. Requesting this ETL's own field list against it now
returns HTTP 400, so the ETL could not be re-run at all. The county's AGOL layer
`County_Parcels/FeatureServer/0` carries exactly the original field names plus
`LegalDescription`, `Zoning` and `NoDisplay`, so we point there instead. That
layer also states `TaxExempt` as 'yes'/'no' rather than 'Y'/'N' -- see below.

The condominium common-element trap (the defect Greg saw on the 3D map)
----------------------------------------------------------------------
Pueblo records commercial, office, medical and townhome condominiums as:
  - one GROUND tract per development, owned by the association, whose
    `LegalDescription` says "COMMON ELEMENT ...", "GCE ...", "CE ...", and which
    carries $0 (or a nominal) land value; and
  - one parcel per UNIT whose legal states an undivided interest in that ground
    ("+ INT IN CE", "+ 1/15TH OF COMMON GROUND", "1/26 INT IN + TO THE COMMON
    GROUND", "+ INTERST IN COMMON ELEMENT", ...).

The unit polygons are BUILDING FOOTPRINTS, not lots. Measured over the 57 in-city
tracts: the tracts hold 56.0 acres against only 17.9 acres of unit footprint. So
each unit's land_value_per_sqft -- the field the 3D map extrudes -- is divided by
roughly a quarter of the ground it actually occupies (Tuscany Villas' 95 units
read $12.31/sqft against a true $3.41), while the tract beside them carries $0 and
renders as a flat void. That is the "units floating above the ground" artifact.

We therefore merge each tract with its own units: union the geometry, sum the
values. Total land value and total acreage are untouched -- only the denominator
is repaired. Two developments (Champion Villa, Champion Villa North) additionally
have their units drawn ON TOP of the tract, so unioning also removes genuinely
double-counted acreage.

*** Do NOT try to identify this regime by assessor class code. *** An earlier
investigation tested only class 1130 (residential condominiums, which in Pueblo
genuinely do carry per-unit land value), concluded "Pueblo condos are clean", and
missed this entire commercial/mixed-use regime. `LegalDescription` is the only
field that states the regime, and the county's spelling of it is inconsistent
(COMMOM, ELEMENET, COMDOMINIUM, INTERST, INC for INT are all live in the feed).

Slivers, and what they actually turned out to be
------------------------------------------------
Checked against the assessor's stated acreage, the tiny-polygon-with-big-value
parcels are three different things, and only the first two are defects:
  1. FRAGMENT GEOMETRY -- the polygon is a scrap of a much larger deeded parcel.
     `0408101008` is 269 sqft of a 13.74-acre parcel ($153/sqft); `0524115017` is
     391 sqft of 195.6 acres; `1512224013` is 308 sqft of a 0.10-acre lot.
  2. UNATTRIBUTED POLYGONS -- no owner, no legal, $0/$0: a GIS polygon no account
     joined to. 18 of these sit in the city (14.0 acres, $0 land). One is
     `0419301007`, the 15.4-sqft polygon that carried $608,098 and a $39,362/sqft
     spike in the shipped build. They are dropped.
  3. GENUINELY TINY PARCELS -- real deeded scraps ("W 3.00 FT OF LOT 15",
     "A PARCEL 1 FT X 50 FT"). 75 of them, carrying $100-$5,400 each. Their
     $/sqft is honest but meaningless for a colour ramp.

Cases 1 and 3 are flagged `likely_remnant` rather than deleted, so their value
still counts in totals and only the per-parcel extruded layer skips them (set
`hideRemnants: true` in viz/src/cities/pueblo.json). The flag is NOT a bare size
threshold: it is size OR a neighborhood-relative $/sqft outlier, because Pueblo's
assessor values land at a near-uniform rate per neighborhood (both Eco Walk units
are exactly $7.35/sqft), which makes the ratio distribution tight enough -- p99.9
is 10.5x the neighborhood median -- for a 10x test to catch `0431447006` (528
sqft, above any size cutoff) without touching genuinely valuable parcels.

Duplicate PAR_TXT rows must be collapsed with `first`, never `sum`
-----------------------------------------------------------------
One account can be drawn as several polygons, and the county repeats the SAME
value on every row (verified: all 24 in-city duplicate groups have exactly one
distinct land value across their rows). Summing them fabricated $11,150,454 of
land value, $10.7M of it on a single parcel -- `1513303005` (Quik Trip) appears
7 times at $1,791,271 and was booked at $12,538,897.

Outputs:
- data/jurisidictions/data/pueblo/pueblo-co-parcels.parquet
- data/jurisidictions/data/pueblo/pueblo-co-parcels_YYYY_MM_DD.parquet
"""

from __future__ import annotations

import argparse
import glob
import re
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Geod
from shapely.ops import unary_union
from shapely.validation import make_valid

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.parcel_calculations import add_improvement_ratio_fields


PARCEL_QUERY_URL = (
    "https://services1.arcgis.com/IL17xsvNU5Bmw3RY/ArcGIS/rest/services/"
    "County_Parcels/FeatureServer/0/query"
)
BOUNDARY_QUERY_URL = (
    "https://maps.co.pueblo.co.us/outside/rest/services/"
    "Landbase/PuebloCounty_MunicipalCountyBoundaries/MapServer/0/query"
)
ZONING_QUERY_URL = (
    "https://maps.co.pueblo.co.us/outside/rest/services/"
    "Landbase/PuebloCounty_ZoningCountyCity/MapServer/0/query"
)
CITY_NAME = "PUEBLO"
PAGE_SIZE = 1000
OUTPUT_DIR = Path("data/jurisidictions/data/pueblo")
RAW_FIELDS = [
    "OBJECTID",
    "PAR_NUM",
    "PAR_TXT",
    "Owner",
    "TaxExempt",
    "Neighborhood",
    "Subdivision",
    "LandActualValue",
    "ImprovementsActualValue",
    "AssessorURL",
    # LegalDescription is the ONLY field that states the condominium regime.
    "LegalDescription",
    "NoDisplay",
]
ZONING_FIELDS = ["OBJECTID", "ZONE_CODE", "MAP_AMENDMENT", "ZoningURL"]

STATE_PLANE_EPSG = 2232          # NAD83 / Colorado South (ftUS) -- planar areas
SQM_TO_SQFT = 10.763910416709722

# --- common-element ground detection ---------------------------------------- #
# Matched ANYWHERE in the legal description, not anchored: "PAR A GENERAL COMMON
# ELEMENT BELMONT COMMONS" and "ELEVENTH FAIRWAY SUB AMD COMMON GROUND KNOWN AS
# PAR A" both bury the marker mid-string, and an anchored pattern left 8 tracts
# rendering as voids. COMMOM / ELEMENET are the county's own typos.
CE_PATTERN = re.compile(
    r"(?:(?:GENERAL|LIMITED)\s+)?COMMO[NM]\s+(?:ELEMEN[ET]?T?S?|AREA|GROUND|PROPERTY)"
    r"|(?<![A-Z0-9])(?:GCE|LCE)(?![A-Z0-9])"
    r"|^CE\b"
)
# A UNIT states its own undivided interest in that ground. Every spelling below is
# live in the feed; each was found by re-reading parcels an earlier iteration of
# this pattern had misfiled as tracts:
#   "UNIT 2 ... + INT IN CE"                  "SUITE 110 ... + INTEREST IN GCE"
#   "UNIT 100 ... + INT IN COMMON ELEMENT"    "UNIT B + 1/15TH OF COMMON GROUND"
#   "UNIT 2137 + 7.143% OF COMMON GROUND"     "UNIT 104 ... 1/26 INT IN + TO THE
#   "LOT 11 ... 1/15TH INT IN COMMON AREA"     COMMON GROUND"
#   "UNIT 5C ALSO UNDIVIDED 1/19 INT IN ALL COMMOM GROUND"   (COMMOM typo)
#   "UNIT 22 ... 1/30 INC IN COMMON GROUND"                  (INC for INT)
#   "UNIT 3 ... + INTERST IN COMMON ELEMENT"                 (INTERST typo)
#   "CU #100A LCE ..."                                       (CU = condo unit)
UNIT_INTEREST_PATTERN = re.compile(
    r"(?:\bINT\b|\bINTE?R?E?ST\b|\bINC\b)\s+(?:IN|TO)\b.{0,30}?"
    r"(?:COMMO[NM]|GCE|LCE|(?<![A-Z])C\s?E(?![A-Z]))"
    r"|\d\s*/\s*\d+\s*(?:ST|ND|RD|TH)?\s+(?:OF|IN)\b.{0,20}?(?:COMMO[NM]|GCE|LCE)"
    r"|\d\s*%\s*(?:OF\s+)?(?:THE\s+)?COMMO[NM]"
    r"|^CU\s*#"
)
# Tokens carrying no identity, so they must not drive development-name matching.
# "LOT"/"UNIT" are deliberately absent: "LOT 1 LLC" is a development name.
_NAME_STOPWORDS = {
    "COMMON", "ELEMENT", "ELEMENTS", "AREA", "GROUND", "PROPERTY", "CE", "GCE",
    "LCE", "THE", "OF", "IN", "A", "AN", "AND", "ALSO", "CONDOMINIUM",
    "CONDOMINIUMS", "CONDO", "CONDOS", "AMENDED", "AMENDMENT", "AMD", "PLAT",
    "MAP", "NO", "FILING", "PHASE", "SUB", "SUBDIVISION", "SPECIAL", "PLAN",
    "OWNERS", "ASSOCIATION", "ASSOC", "ASSN", "FIRST", "SECOND", "THIRD",
    "PAR", "PARCEL", "TR", "TRCT", "TRACT", "BLK", "BLOCK", "KNOWN", "AS", "SITE",
}
_ABBREV = {"SUBDIVISION": "SUB", "CONDOMINIUMS": "CONDOMINIUM",
           "ASSOCIATION": "ASSN", "ASSOC": "ASSN", "AMENDMENT": "AMENDED",
           "AMD": "AMENDED"}
# Sentinel rows: 9999999999 is literally owned by "TEMPLATE PARCEL".
TEMPLATE_ID_PATTERN = re.compile(r"^9{8}")
# Thresholds for the "implied common ground" test (see merge_condo_developments).
TRACT_MIN_SQFT = 5_000              # a development's shared ground is never tiny
TRACT_NOMINAL_LAND_CAP = 50_000     # observed nominal tract values: $100 - $40,000

# --- sliver / remnant flagging ---------------------------------------------- #
REMNANT_SQFT = 500
OUTLIER_REL_PPSF = 10.0             # citywide p99.9 of the ratio is 10.5
OUTLIER_AREA_SQFT = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Pueblo canonical parcel parquet.")
    parser.add_argument("--use-cache", action="store_true", help="Reuse the latest cached raw parquet if present.")
    parser.add_argument("--skip-upload", action="store_true", help="Accepted for CLI compatibility; upload is handled separately.")
    return parser.parse_args()


def request_json(url: str, params: dict, *, timeout: int = 240) -> dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS error from {url}: {payload['error']}")
    return payload


def fetch_count(url: str, where: str = "1=1") -> int:
    payload = request_json(
        url,
        {
            "f": "json",
            "where": where,
            "returnCountOnly": "true",
        },
        timeout=120,
    )
    return int(payload["count"])


def fetch_page(url: str, fields: list[str], offset: int, *, where: str = "1=1") -> gpd.GeoDataFrame:
    payload = request_json(
        url,
        {
            "f": "geojson",
            "where": where,
            "outFields": ",".join(fields),
            "returnGeometry": "true",
            "outSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "orderByFields": fields[0],
        },
    )
    gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    for field in fields:
        if field not in gdf.columns:
            gdf[field] = np.nan
    return gdf


def latest_cached_raw(raw_glob: str) -> Path | None:
    files = sorted(glob.glob(raw_glob))
    return Path(files[-1]) if files else None


def download_layer(url: str, fields: list[str], raw_path: Path, *, where: str = "1=1", label: str) -> gpd.GeoDataFrame:
    total = fetch_count(url, where=where)
    print(f"Found {total:,} {label} records")
    frames: list[gpd.GeoDataFrame] = []
    offset = 0
    while offset < total:
        frame = fetch_page(url, fields, offset, where=where)
        if frame.empty:
            break
        frames.append(frame)
        offset += len(frame)
        print(f"  fetched {offset:,}/{total:,}")
        time.sleep(0.03)

    raw = pd.concat(frames, ignore_index=True)
    raw_gdf = gpd.GeoDataFrame(raw, geometry="geometry", crs="EPSG:4326")
    raw_gdf.to_parquet(raw_path, index=False)
    print(f"Saved raw cache: {raw_path}")
    return raw_gdf


def fetch_pueblo_boundary() -> gpd.GeoDataFrame:
    payload = request_json(
        BOUNDARY_QUERY_URL,
        {
            "f": "geojson",
            "where": f"UPPER(City_Name)='{CITY_NAME}'",
            "outFields": "City_Name",
            "returnGeometry": "true",
            "outSR": 4326,
        },
    )
    boundary = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    if boundary.empty:
        raise RuntimeError("Failed to fetch Pueblo municipal boundary.")
    return boundary


def geodesic_area_sqft(geom) -> float:
    geod = Geod(ellps="WGS84")
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type == "Polygon":
        lon, lat = geom.exterior.coords.xy
        area_m2, _ = geod.polygon_area_perimeter(lon, lat)
        hole_area = 0.0
        for ring in geom.interiors:
            lon_h, lat_h = ring.coords.xy
            part_area, _ = geod.polygon_area_perimeter(lon_h, lat_h)
            hole_area += abs(part_area)
        return max(abs(area_m2) - hole_area, 0.0) * 10.763910416709722
    if geom.geom_type == "MultiPolygon":
        return sum(geodesic_area_sqft(part) for part in geom.geoms)
    return np.nan


def safe_union(geoms):
    """unary_union that survives the county's self-intersecting rings."""
    gs = [g for g in geoms if g is not None and not g.is_empty]
    if not gs:
        return None
    try:
        return unary_union(gs)
    except Exception:
        pass
    repaired = []
    for g in gs:
        try:
            r = g if g.is_valid else make_valid(g)
        except Exception:
            r = g.buffer(0)
        if r is None or r.is_empty:
            continue
        if r.geom_type not in ("Polygon", "MultiPolygon"):
            polys = [p for p in getattr(r, "geoms", [])
                     if p.geom_type in ("Polygon", "MultiPolygon")]
            r = unary_union(polys) if polys else None
        if r is not None and not r.is_empty:
            repaired.append(r)
    if not repaired:
        return None
    try:
        return unary_union(repaired)
    except Exception:
        return unary_union([g.buffer(0) for g in repaired])


def normalize_legal(series: pd.Series) -> pd.Series:
    return (series.fillna("").astype(str).str.upper()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def strip_formerly(text: str) -> str:
    """Drop the '... FORMERLY #05-364-49-012' provenance tail."""
    return re.sub(r"\s*(?:FORMERLY|FORMER\s*#|NO\s+FORMER)\b.*$", "", text).strip()


def plat_root(par_txt: pd.Series) -> pd.Series:
    """First three groups of the 10-char id (05|364|49|013 -> 0536449)."""
    return par_txt.fillna("").astype(str).str.strip().str[:7]


def name_tokens(text: str) -> set:
    body = CE_PATTERN.sub("", strip_formerly(text))
    body = re.sub(r"^\s*(?:IN|OF|FOR)\b", "", body).strip()
    toks = set()
    for raw in re.findall(r"[A-Z0-9]+", body):
        tok = _ABBREV.get(raw, raw)
        if tok not in _NAME_STOPWORDS:
            toks.add(tok)
    return toks


def unattributed_mask(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Polygons with no assessor record behind them at all.

    No owner, no legal description, $0 land and $0 improvement -- a GIS polygon no
    account joined to. 18 sit inside the city (14.0 acres, $0 land value), one of
    them `0419301007`: the 15.4-sqft polygon that carried $608,098 and a
    $39,362/sqft spike in the shipped build. Dropping them removes that artifact
    structurally instead of relying on the county having since unlinked the value.
    """
    blank = lambda col: gdf[col].fillna("").astype(str).str.strip().eq("")
    land = pd.to_numeric(gdf["LandActualValue"], errors="coerce").fillna(0)
    impr = pd.to_numeric(gdf["ImprovementsActualValue"], errors="coerce").fillna(0)
    return blank("Owner") & blank("LegalDescription") & (land <= 0) & (impr <= 0)


def drop_non_parcels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tmpl = gdf["PAR_TXT"].fillna("").astype(str).str.match(TEMPLATE_ID_PATTERN)
    orph = unattributed_mask(gdf)
    drop = tmpl | orph
    if drop.any():
        acres = gdf[drop].to_crs(STATE_PLANE_EPSG).geometry.area.sum() / 43560
        land = pd.to_numeric(gdf.loc[drop, "LandActualValue"], errors="coerce").fillna(0).sum()
        print(f"Dropped {int(drop.sum())} non-parcel polygons "
              f"({int(tmpl.sum())} template/sentinel ids, "
              f"{int((orph & ~tmpl).sum())} unattributed) -- "
              f"${land:,.0f} land value, {acres:.2f} acres")
    return gdf[~drop].copy()


def collapse_duplicate_parcels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Collapse multi-polygon accounts by UNIONING geometry and taking `first`.

    *** Values are NOT summed. *** One account drawn as several polygons repeats
    the SAME value on every row -- verified: all 24 in-city duplicate groups carry
    exactly one distinct land value across their rows. The previous `sum` here
    fabricated $11,150,454 of land value, $10.7M of it on `1513303005` alone
    (Quik Trip, 7 polygons x $1,791,271 booked as $12,538,897).
    """
    dup = gdf.duplicated(subset=["PAR_TXT"], keep=False)
    print(f"Duplicate PAR_TXT rows: {int(dup.sum()):,} across "
          f"{gdf.loc[dup, 'PAR_TXT'].nunique()} accounts")
    if not dup.any():
        return gdf

    land = pd.to_numeric(gdf.loc[dup, "LandActualValue"], errors="coerce").fillna(0)
    first = gdf[dup].assign(_l=land).groupby("PAR_TXT")["_l"].first().sum()
    print(f"  land value that summing would have fabricated: ${land.sum() - first:,.0f}")

    cols = [c for c in gdf.columns if c not in ("geometry", "PAR_TXT")]
    collapsed = gdf.groupby("PAR_TXT", dropna=False).agg({c: "first" for c in cols}).reset_index()
    collapsed["geometry"] = gdf.groupby("PAR_TXT", dropna=False)["geometry"].apply(safe_union).values
    result = gpd.GeoDataFrame(collapsed, geometry="geometry", crs=gdf.crs)
    print(f"Rows after duplicate collapse: {len(result):,}")
    return result


def _token_filter(pool: pd.DataFrame, ce_tokens: set) -> pd.DataFrame:
    """Keep pool rows whose legal description names the same development.

    Requires >=50% of the tract's distinctive tokens. This is the guard that stops
    a merge from swallowing an unrelated fee-simple neighbour: plat root 0536423
    holds the "LOT 1 LLC AMENDMENT TWO" condominium beside four independent
    Historic Arkansas Riverwalk lots worth $740k, and only the condominium's own
    suite parcel clears the threshold.
    """
    if pool.empty or not ce_tokens:
        return pool.iloc[0:0]
    keep = [idx for idx, leg in pool["_leg"].items()
            if (t := name_tokens(leg)) and len(ce_tokens & t) / len(ce_tokens) >= 0.5]
    return pool.loc[keep]


def merge_condo_developments(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge each common-element ground tract with the units that own it.

    Union of geometry + SUM of value, so total land value and total acreage are
    untouched and only the denominator is repaired. See the module docstring for
    the measurements that justify merging rather than flagging.
    """
    gdf = gdf.copy()
    gdf["_leg"] = normalize_legal(gdf["LegalDescription"])
    gdf["_root"] = plat_root(gdf["PAR_TXT"])
    gdf["_merged_units"] = 0
    gdf["_is_common_element"] = 0
    gdf["_sqft"] = gdf.to_crs(STATE_PLANE_EPSG).geometry.area

    _land = pd.to_numeric(gdf["LandActualValue"], errors="coerce").fillna(0)
    _impr = pd.to_numeric(gdf["ImprovementsActualValue"], errors="coerce").fillna(0)

    gdf["_has_interest"] = gdf["_leg"].str.contains(UNIT_INTEREST_PATTERN)
    # Structural guard, independent of wording: common ground is GROUND. A parcel
    # carrying improvements is a unit with a building on it, whatever its legal
    # happens to mention. This is what stopped ~150 Chatalet / Sovereign Court /
    # Cottonwood Creek units from being read as tracts.
    gdf["_is_ce"] = (gdf["_leg"].str.contains(CE_PATTERN)
                     & ~gdf["_has_interest"] & (_impr <= 0))

    # Secondary, wording-free tract test. Some subdivisions record the shared
    # ground as a bare "PAR A" / "TRCT A" with no common-element wording, and it is
    # the UNITS that point at it ("ALSO A 1/15 INT IN THE COMMON GROUND KNOWN AS
    # PARCEL A"). Nor is the tract's value reliably $0: Two Pines at Outlook's
    # "PAR A B C + D" carries $100, Seven Oaks' "PAR A" $15,000, Residence at
    # Outlook's "PAR B" $20,000. Requiring $0 left 471 units (36 acres) unmerged
    # and still reading ~4x high. Guards, each load-bearing: no improvements
    # (ground, not a building); substantial area (not a sliver mis-keyed as a
    # tract); land under a nominal cap AND under 25% of the units' own land, so an
    # ordinary fee-simple neighbour can never be absorbed.
    interest_per_root = gdf.loc[gdf["_has_interest"]].groupby("_root")["PAR_TXT"].size()
    root_interest = gdf["_root"].map(interest_per_root).fillna(0)
    root_unit_land = gdf["_root"].map(
        gdf.loc[gdf["_has_interest"]].assign(_l=_land).groupby("_root")["_l"].sum()).fillna(0)
    lettered = gdf["_leg"].str.contains(
        r"^(?:PAR|PARCEL|TR|TRCT|TRACT)\s+[\"']?[A-Z][\"']?(?:\s|\+|$)", regex=True, na=False)
    implied = (lettered & (root_interest >= 2) & ~gdf["_is_ce"] & (_impr <= 0)
               & (gdf["_sqft"] >= TRACT_MIN_SQFT)
               & (_land <= TRACT_NOMINAL_LAND_CAP)
               & (_land <= 0.25 * root_unit_land))
    gdf.loc[implied, "_is_ce"] = True

    print("\n--- common-element / condominium merge ---")
    print(f"Parcels stating an undivided interest in common ground: "
          f"{int(gdf['_has_interest'].sum()):,}")
    print(f"Common-element tracts: {int(gdf['_is_ce'].sum())} "
          f"({int(implied.sum())} of them implied by a lettered nominal-value tract)")

    ce_ids = sorted(set(gdf.loc[gdf["_is_ce"], "PAR_TXT"]))
    if not ce_ids:
        return gdf.drop(columns=["_leg", "_root", "_is_ce", "_has_interest", "_sqft"])

    proj_geom = dict(zip(gdf.index, gdf.to_crs(STATE_PLANE_EPSG).geometry))
    absorbed: set = set()
    n_merged = n_units = 0

    for ce_id in ce_ids:
        ce_rows = gdf[(gdf["PAR_TXT"] == ce_id) & gdf["_is_ce"]]
        if ce_rows.empty:
            continue
        ce_tokens = set()
        for leg in ce_rows["_leg"]:
            ce_tokens |= name_tokens(leg)
        root = ce_rows["_root"].iloc[0]

        # Candidate pool: same plat root, then physical adjacency, then the bare
        # structural signal for tracts that carry no matchable name.
        pool = gdf[(gdf["_root"] == root) & (gdf["PAR_TXT"] != ce_id) & ~gdf["_is_ce"]]
        cand = _token_filter(pool, ce_tokens)
        if cand.empty:
            ce_geom = safe_union([proj_geom[i] for i in ce_rows.index])
            if ce_geom is not None:
                near = gdf[(gdf["PAR_TXT"] != ce_id) & ~gdf["_is_ce"]]
                buf = ce_geom.buffer(5.0)
                touch = [i for i in near.index
                         if proj_geom[i] is not None and proj_geom[i].intersects(buf)]
                cand = _token_filter(near.loc[touch], ce_tokens)
        if cand.empty:
            cand = pool[pool["_has_interest"]]
        cand = cand[~cand["PAR_TXT"].isin(absorbed)]

        if cand.empty:
            # A tract with no linkable units keeps its own published value. It is
            # NOT dropped -- it is real, separately-deeded ground.
            gdf.loc[ce_rows.index, "_is_common_element"] = 1
            continue

        keep = ce_rows.index[0]
        land = float(pd.to_numeric(ce_rows["LandActualValue"], errors="coerce").fillna(0).sum()
                     + pd.to_numeric(cand["LandActualValue"], errors="coerce").fillna(0).sum())
        impr = float(pd.to_numeric(ce_rows["ImprovementsActualValue"], errors="coerce").fillna(0).sum()
                     + pd.to_numeric(cand["ImprovementsActualValue"], errors="coerce").fillna(0).sum())
        merged = safe_union([proj_geom[i] for i in list(ce_rows.index) + list(cand.index)])
        merged_wgs = gpd.GeoSeries([merged], crs=STATE_PLANE_EPSG).to_crs(gdf.crs).iloc[0]

        gdf.loc[keep, "geometry"] = merged_wgs
        gdf.loc[keep, "LandActualValue"] = land
        gdf.loc[keep, "ImprovementsActualValue"] = impr
        gdf.loc[keep, "_merged_units"] = len(cand)
        gdf.loc[keep, "_is_common_element"] = 1
        # A tract flagged tax-exempt while its units are taxable must not be
        # dropped by the exempt filter -- that is exactly what re-opens the hole.
        # Riverwalk Place's tract is TaxExempt='yes' in the live feed.
        if (cand["TaxExempt"].fillna("").astype(str).str.strip().str.lower() == "no").any():
            gdf.loc[keep, "TaxExempt"] = "no"
        for col in ("Neighborhood", "Subdivision", "AssessorURL"):
            blank = pd.isna(gdf.at[keep, col]) or str(gdf.at[keep, col]).strip() in ("", "None")
            if blank and cand[col].notna().any():
                gdf.at[keep, col] = cand[col].dropna().iloc[0]

        gdf = gdf.drop(index=list(cand.index) + [i for i in ce_rows.index if i != keep])
        absorbed |= set(cand["PAR_TXT"])
        n_merged += 1
        n_units += len(cand)

    print(f"Developments formed: {n_merged} (absorbing {n_units:,} unit parcels)")
    # Honest residual: units whose tract is absent from the feed keep a
    # footprint-based $/sqft. Nothing is imputed for them.
    left = gdf[gdf["_has_interest"] & (gdf["_merged_units"] == 0) & ~gdf["_is_ce"]]
    if len(left):
        ll = pd.to_numeric(left["LandActualValue"], errors="coerce").fillna(0)
        print(f"UNRESOLVED: {len(left):,} parcels state an undivided interest but no "
              f"tract could be linked (${ll.sum():,.0f} land on "
              f"{left['_sqft'].sum() / 43560:.1f} ac of footprint) -- left as published")

    out = gdf.drop(columns=["_leg", "_root", "_is_ce", "_has_interest", "_sqft"])
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


def flag_remnants(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Mark polygons whose land $/sqft is an artifact of a broken denominator."""
    area = pd.to_numeric(gdf["area_sqft"], errors="coerce")
    land = pd.to_numeric(gdf["current_full_land_value"], errors="coerce").fillna(0)
    ppsf = (land / area).replace([np.inf, -np.inf], np.nan)

    tiny = area < REMNANT_SQFT
    med = ppsf.where(ppsf > 0).groupby(gdf["neighborhood"]).transform("median")
    outlier = ((ppsf / med) > OUTLIER_REL_PPSF) & (area < OUTLIER_AREA_SQFT)

    gdf["likely_remnant"] = (tiny | outlier.fillna(False)).astype(int)
    flagged = gdf["likely_remnant"] == 1
    print("\n--- remnant / sliver flagging ---")
    print(f"  sub-{REMNANT_SQFT}-sqft polygons: {int(tiny.sum()):,}")
    print(f"  neighborhood $/sqft outliers (>{OUTLIER_REL_PPSF:.0f}x median, "
          f"<{OUTLIER_AREA_SQFT:,} sqft): {int(outlier.fillna(False).sum()):,}")
    print(f"  likely_remnant total: {int(flagged.sum()):,} "
          f"(${land[flagged].sum():,.0f} land, {area[flagged].sum() / 43560:.2f} acres) "
          f"-- retained in the parquet, hidden from the per-parcel layer by "
          f"hideRemnants in viz/src/cities/pueblo.json")
    return gdf


def classify_original(zone_code: str, exempt_flag: int, improvement_value: float) -> str:
    zone = (zone_code or "").strip().upper()
    if zone:
        return zone
    if exempt_flag:
        return "EXEMPT"
    if improvement_value <= 0:
        return "UNCLASSIFIED VACANT"
    return "UNCLASSIFIED"


def classify_refined(row: pd.Series) -> str | None:
    zone = str(row.get("zone_code") or "").strip().upper()
    land = float(pd.to_numeric(row.get("current_full_land_value"), errors="coerce") or 0.0)
    improvement = float(pd.to_numeric(row.get("improvement_value"), errors="coerce") or 0.0)
    total = land + improvement

    if improvement <= 0 and land > 0:
        return "Vacant"

    commercial_like = zone.startswith(("B", "I"))
    if commercial_like and total > 0 and improvement < 0.5 * total:
        return "Underdeveloped"

    return None


def build_export(raw_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame, zoning_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = raw_gdf.copy()
    gdf = gdf[gdf.geometry.notna()].copy()

    boundary_union = unary_union(boundary_gdf.geometry)
    gdf = gdf[gdf.geometry.intersects(boundary_union)].copy()

    for col in ["LandActualValue", "ImprovementsActualValue"]:
        gdf[col] = pd.to_numeric(gdf.get(col), errors="coerce").fillna(0)

    gdf = drop_non_parcels(gdf)
    gdf = collapse_duplicate_parcels(gdf)
    # Runs BEFORE the exempt filter at the end of this function: a common-element
    # tract can itself be flagged exempt, and dropping it before the merge is what
    # leaves its units floating (Riverwalk Place's tract is TaxExempt='yes').
    gdf = merge_condo_developments(gdf)

    # This layer states TaxExempt as 'yes'/'no', not 'Y'/'N'. The previous
    # `.eq("Y")` test therefore matched NOTHING: measured against the live feed it
    # flagged 0 parcels where 2,620 are exempt, leaking $148,523,653 of exempt land
    # value into the map. Both spellings are accepted so either source works.
    exempt_txt = gdf["TaxExempt"].fillna("").astype(str).str.strip().str.upper()
    gdf["exemption_flag"] = exempt_txt.isin({"Y", "YES", "TRUE", "1"}).astype(int)
    print(f"Exempt parcels flagged: {int(gdf['exemption_flag'].sum()):,}")

    zone_subset = zoning_gdf[["zone_code", "zoning_url", "geometry"]].copy()
    centroids = gdf.geometry.representative_point()
    centroids_gdf = gpd.GeoDataFrame(gdf[["PAR_TXT"]].copy(), geometry=centroids, crs=gdf.crs)
    zone_join = gpd.sjoin(centroids_gdf, zone_subset, how="left", predicate="within")
    zone_join = zone_join.groupby("PAR_TXT", dropna=False).first().reset_index()
    gdf = gdf.merge(zone_join[["PAR_TXT", "zone_code", "zoning_url"]], on="PAR_TXT", how="left")

    def clean_text(series: pd.Series) -> pd.Series:
        text = series.fillna("").astype(str).str.strip()
        return text.mask(text.eq(""), np.nan)

    gdf["owner"] = clean_text(gdf["Owner"])
    gdf["neighborhood"] = clean_text(gdf["Neighborhood"])
    gdf["subdivision"] = clean_text(gdf["Subdivision"])
    gdf["assessor_url"] = clean_text(gdf["AssessorURL"])
    gdf["legal_description"] = clean_text(gdf["LegalDescription"])
    gdf["merged_unit_count"] = pd.to_numeric(
        gdf.get("_merged_units", 0), errors="coerce").fillna(0).astype(int)
    gdf["is_common_element"] = pd.to_numeric(
        gdf.get("_is_common_element", 0), errors="coerce").fillna(0).astype(int)
    # Per-parcel provenance for the land-value figure, so a viewer looking at a
    # merged development is not left guessing why one polygon covers 95 addresses.
    gdf["land_value_basis"] = np.where(
        gdf["merged_unit_count"] > 0,
        "Assessor land value, summed across " + gdf["merged_unit_count"].astype(str)
        + np.where(gdf["merged_unit_count"] == 1, " condominium unit", " condominium units")
        + " and the common-element ground they jointly own",
        np.where(
            gdf["is_common_element"] == 1,
            "Common-element ground; Pueblo County publishes no separate land value "
            "for it where the value is carried by the units",
            "Assessor land value for this parcel"))

    gdf["property_land_use_category"] = [
        classify_original(zone, exempt, improvement)
        for zone, exempt, improvement in zip(
            gdf["zone_code"].fillna(""),
            gdf["exemption_flag"],
            gdf["ImprovementsActualValue"],
        )
    ]

    gdf["current_full_land_value"] = gdf["LandActualValue"].clip(lower=0)
    gdf["improvement_value"] = gdf["ImprovementsActualValue"].clip(lower=0)
    gdf["full_market_value"] = gdf["current_full_land_value"] + gdf["improvement_value"]
    gdf["property_land_use_refined"] = gdf.apply(classify_refined, axis=1)
    gdf["area_sqft"] = gdf.geometry.apply(geodesic_area_sqft)
    gdf["land_value_per_sqft"] = gdf["current_full_land_value"] / gdf["area_sqft"]
    gdf["improvement_value_per_sqft"] = gdf["improvement_value"] / gdf["area_sqft"]
    gdf["full_market_value_per_sqft"] = gdf["full_market_value"] / gdf["area_sqft"]
    gdf["REALLANDVA"] = gdf["current_full_land_value"]
    gdf["REALIMPROV"] = gdf["improvement_value"]
    gdf["REALLANDVA_per_sqft"] = gdf["land_value_per_sqft"]
    gdf["REALIMPROV_per_sqft"] = gdf["improvement_value_per_sqft"]

    gdf = add_improvement_ratio_fields(
        gdf,
        land_col="REALLANDVA",
        improvement_col="REALIMPROV",
    )
    gdf = flag_remnants(gdf)

    before = len(gdf)
    gdf = gdf[gdf["exemption_flag"] == 0].copy()
    print(f"\nRemoved {before - len(gdf):,} exempt parcels -> {len(gdf):,} remaining")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    export_columns = [
        "geometry",
        "PAR_TXT",
        "zone_code",
        "property_land_use_category",
        "property_land_use_refined",
        "exemption_flag",
        "current_full_land_value",
        "improvement_value",
        "full_market_value",
        "REALLANDVA",
        "REALIMPROV",
        "area_sqft",
        "land_value_per_sqft",
        "improvement_value_per_sqft",
        "full_market_value_per_sqft",
        "REALLANDVA_per_sqft",
        "REALIMPROV_per_sqft",
        "TLLDIMPROV",
        "IMPR_LAND_RATIO",
        "IMPR_LAND_PCT",
        "IMPR_PCT_TOTAL",
        "owner",
        "neighborhood",
        "subdivision",
        "assessor_url",
        "zoning_url",
        "legal_description",
        "likely_remnant",
        "merged_unit_count",
        "is_common_element",
        "land_value_basis",
    ]
    export = gdf[[c for c in export_columns if c in gdf.columns]].copy()
    export = export.rename(columns={"PAR_TXT": "parcel_id"})
    return export


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y_%m_%d")
    raw_glob = str(OUTPUT_DIR / "pueblo-co-raw_*.parquet")
    raw_path = OUTPUT_DIR / f"pueblo-co-raw_{today}.parquet"
    zoning_path = OUTPUT_DIR / f"pueblo-co-zoning_{today}.parquet"
    output_path = OUTPUT_DIR / "pueblo-co-parcels.parquet"
    dated_path = OUTPUT_DIR / f"pueblo-co-parcels_{today}.parquet"

    boundary = fetch_pueblo_boundary()
    print(f"Fetched Pueblo boundary fragments: {len(boundary):,}")

    raw_gdf: gpd.GeoDataFrame
    if args.use_cache:
        cached = latest_cached_raw(raw_glob)
        if cached and cached.exists():
            print(f"Using cached raw parcels: {cached}")
            raw_gdf = gpd.read_parquet(cached)
        else:
            raw_gdf = download_layer(PARCEL_QUERY_URL, RAW_FIELDS, raw_path, label="Pueblo County parcel")
    else:
        raw_gdf = download_layer(PARCEL_QUERY_URL, RAW_FIELDS, raw_path, label="Pueblo County parcel")

    zoning_gdf = download_layer(ZONING_QUERY_URL, ZONING_FIELDS, zoning_path, label="Pueblo zoning")
    zoning_gdf = zoning_gdf.rename(columns={"ZONE_CODE": "zone_code", "ZoningURL": "zoning_url"})
    zoning_gdf = zoning_gdf[zoning_gdf.geometry.notna()].copy()

    export = build_export(raw_gdf, boundary, zoning_gdf)
    export.to_parquet(output_path, index=False)
    export.to_parquet(dated_path, index=False)

    print(f"Saved canonical parquet: {output_path}")
    print(f"Saved dated snapshot : {dated_path}")
    print(f"Final parcel count   : {len(export):,}")
    print("\nRefined category counts:")
    print(export["property_land_use_refined"].value_counts(dropna=False).to_string())
    print("\nOriginal category counts:")
    print(export["property_land_use_category"].value_counts(dropna=False).head(20).to_string())


if __name__ == "__main__":
    main()
