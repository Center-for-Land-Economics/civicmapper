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

The condo trap (the Fort Collins lesson, present here)
-----------------------------------------------------
Jeffco books condominium value 100% to improvements: class 1230 units have
`TOTACTVAL == TOTACTIMPV` and NO land component (`ASMASDLND` is null). The
development's land sits on a separate common-area/association parcel whose
`TAXCLS` IS BLANK and whose value is $0 -- exactly the parcel a naive exempt or
"drop rows with no class" filter would delete, leaving 5,394 unit polygons
stacked on missing land (max observed stack: 75 parcels on one point).

Because those unit polygons OVERLAP the association polygon, keeping both would
double-count the city's land area. So we merge each development's units DOWN onto
its association parcel, linked by the PIN plat root (first three PIN groups, e.g.
units `39-331-01-0xx` -> association `39-331-01-105`), with a spatial-intersection
fallback. Improvement values are summed onto the real footprint.

*** KNOWN DATA LIMITATION (surfaced, not silently absorbed) ***
Condo land value is genuinely absent from the Jeffco feed -- not hidden in
another field and not recoverable by any join. ~5,200 condo parcels carrying
~$1.64B of improvements have $0 land value, so Lakewood's citywide land-value
total UNDERSTATES true land value. We do not impute it. The ETL reports the
magnitude so it can be cited as a caveat.

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


def merge_condo_developments(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge stacked condo unit parcels down onto their association parcel.

    Association parcels (blank TAXCLS, a real PIN, real footprint, $0 value) are
    captured BEFORE any exempt/blank filtering, per the playbook, so they can
    still participate in the merge.
    """
    gdf = gdf.copy()
    gdf["_pin"] = gdf["PIN"].fillna("").astype(str).str.strip()
    gdf["_tax"] = gdf["TAXCLS"].fillna("").astype(str).str.strip()
    gdf["_root"] = plat_root(gdf["_pin"])

    is_assoc = (gdf["_tax"] == "") & (~gdf["_pin"].isin(NON_PARCEL_PINS)) & (gdf["_pin"] != "")
    is_unit = gdf["POLYCAT"].isin([2, 3]) & (gdf["_tax"] != "")

    assoc = gdf[is_assoc].copy()
    units = gdf[is_unit].copy()
    others = gdf[~is_assoc & ~is_unit].copy()

    # A plat's common area can be several polygons sharing one PIN; dissolve them
    # so each association PIN is a single merge target.
    if assoc["_pin"].duplicated().any():
        n_before = len(assoc)
        geom = assoc.groupby("_pin")["geometry"].apply(
            lambda gs: unary_union([g for g in gs if g is not None]))
        assoc = assoc.drop_duplicates("_pin").set_index("_pin")
        assoc["geometry"] = geom
        assoc = gpd.GeoDataFrame(assoc.reset_index(), geometry="geometry", crs=gdf.crs)
        print(f"Dissolved multi-polygon association plats: {n_before:,} -> {len(assoc):,}")

    print("\n--- condo / common-area merge ---")
    print(f"Association (common-area) parcels captured: {len(assoc):,}")
    print(f"Stacked condo unit parcels: {len(units):,}")

    # smoke alarm: how much stacking are we actually fixing?
    key = gdf.geometry.representative_point()
    k = key.x.round(5).astype(str) + "," + key.y.round(5).astype(str)
    vc = k.value_counts()
    print(f"Stacked clusters before merge: {int((vc > 1).sum()):,} (max stack {int(vc.max())})")

    if units.empty or assoc.empty:
        print("Nothing to merge.")
        return gdf.drop(columns=["_pin", "_tax", "_root"])

    # link unit -> association: plat root first, spatial intersection as fallback
    root_to_assoc = (assoc.drop_duplicates("_root").set_index("_root")["_pin"].to_dict())
    units["_target"] = units["_root"].map(root_to_assoc)
    by_root = int(units["_target"].notna().sum())

    missing = units[units["_target"].isna()]
    if len(missing):
        a3 = assoc.to_crs(UTM_EPSG)[["_pin", "geometry"]].rename(columns={"_pin": "_apin"})
        m3 = missing.to_crs(UTM_EPSG)[["_pin", "geometry"]]
        j = gpd.sjoin(m3, a3, how="left", predicate="intersects")
        j = j.dropna(subset=["_apin"]).drop_duplicates("_pin")
        fill = dict(zip(j["_pin"], j["_apin"]))
        units.loc[units["_target"].isna(), "_target"] = (
            units.loc[units["_target"].isna(), "_pin"].map(fill))
    by_spatial = int(units["_target"].notna().sum()) - by_root
    unmatched = units[units["_target"].isna()]
    print(f"Linked by PIN plat root: {by_root:,}; by spatial overlap: {by_spatial:,}; "
          f"unmatched: {len(unmatched):,}")

    matched = units[units["_target"].notna()].copy()
    val_cols = ["TOTACTLNDV", "TOTACTIMPV", "TOTACTVAL"]
    for c in val_cols:
        matched[c] = pd.to_numeric(matched[c], errors="coerce").fillna(0)
        assoc[c] = pd.to_numeric(assoc[c], errors="coerce").fillna(0)

    agg = matched.groupby("_target").agg(
        _n_units=("_pin", "size"),
        _lnd=("TOTACTLNDV", "sum"),
        _imp=("TOTACTIMPV", "sum"),
        _val=("TOTACTVAL", "sum"),
        _tax_mode=("_tax", lambda s: s.value_counts().index[0]),
        _sub=("SUBNAM", "first"),
        _use=("STTTYPUSE", "first"),
        _units_geom=("geometry", lambda gs: unary_union([g for g in gs if g is not None])),
    )

    assoc = assoc.set_index("_pin")
    hit = assoc.index.intersection(agg.index)
    # the merged development footprint: association land + unit footprints, holes closed
    new_geom = []
    for pin in hit:
        g = unary_union([assoc.loc[pin, "geometry"], agg.loc[pin, "_units_geom"]])
        new_geom.append(fill_holes(g))
    assoc.loc[hit, "geometry"] = gpd.GeoSeries(new_geom, index=hit, crs=assoc.crs)
    for c, src in [("TOTACTLNDV", "_lnd"), ("TOTACTIMPV", "_imp"), ("TOTACTVAL", "_val")]:
        assoc.loc[hit, c] = assoc.loc[hit, c].values + agg.loc[hit, src].values
    # give the merged development the units' dominant class so it classifies as condo
    assoc.loc[hit, "TAXCLS"] = agg.loc[hit, "_tax_mode"].values
    assoc.loc[hit, "SUBNAM"] = assoc.loc[hit, "SUBNAM"].fillna(
        pd.Series(agg.loc[hit, "_sub"].values, index=hit))
    assoc.loc[hit, "STTTYPUSE"] = assoc.loc[hit, "STTTYPUSE"].fillna(
        pd.Series(agg.loc[hit, "_use"].values, index=hit))
    assoc["_merged_units"] = 0
    assoc.loc[hit, "_merged_units"] = agg.loc[hit, "_n_units"].values
    assoc = assoc.reset_index().rename(columns={"index": "_pin"})

    print(f"Developments formed: {len(hit):,} (absorbing {int(agg.loc[hit, '_n_units'].sum()):,} units)")

    # Association parcels that absorbed nothing are common area / open space with
    # no class and no value -- they are not assessable parcels, so drop them.
    leftover = assoc[assoc["_merged_units"] == 0]
    print(f"Unmerged association parcels dropped: {len(leftover):,}")
    assoc = assoc[assoc["_merged_units"] > 0]

    out = pd.concat([others, assoc, unmatched], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)

    key = out.geometry.representative_point()
    k = key.x.round(5).astype(str) + "," + key.y.round(5).astype(str)
    vc = k.value_counts()
    print(f"Stacked clusters after merge: {int((vc > 1).sum()):,} "
          f"(max stack {int(vc.max()) if len(vc) else 0})")
    return out.drop(columns=["_pin", "_tax", "_root", "_units_geom", "_target"],
                    errors="ignore")


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
    geom = multi.groupby("_sch", dropna=False)["geometry"].apply(
        lambda gs: unary_union([g for g in gs if g is not None]))
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
def classify_category(tax: str, use: str, impr: float) -> str:
    """Lakewood/Jeffco-specific mapping from the Colorado 4-digit abstract class."""
    t = (tax or "").strip()
    u = (use or "").strip()
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
    gdf["year_built"] = pd.to_numeric(gdf["STTYRBLT"], errors="coerce")
    gdf["building_sqft"] = pd.to_numeric(gdf["STTGRSAREA"], errors="coerce")
    gdf["assessor_url"] = np.where(
        gdf["schedule_number"].notna(),
        "https://jeffco.us/assessor/property-records-search/?schedule="
        + gdf["schedule_number"].fillna("").astype(str),
        None,
    )

    gdf["property_land_use_category"] = [
        classify_category(t, u, i) for t, u, i in
        zip(gdf["TAXCLS"].fillna(""), gdf["STTTYPUSE"].fillna(""), gdf["TOTACTIMPV"])
    ]

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
        "subdivision", "neighborhood", "tax_class", "structure_use", "year_built",
        "building_sqft", "assessor_url",
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
