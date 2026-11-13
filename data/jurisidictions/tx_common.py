"""Shared helpers for Texas city ETLs (appraisal-district parcel feeds).

Texas appraisal districts all report the statewide SPTB "state class" code, publish through a
handful of ArcGIS schemas, and share the same dedup traps, so the pieces that were copy-pasted
across run_dallas.py / run_sanantonio.py / run_bcs.py live here for new Texas cities.
Existing scripts are left as they are (they ship as-is); new cities import from this module.

    from tx_common import fetch_arcgis_geojson, categorize_sptb, dedup_accounts
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.ops import unary_union

HEADERS = {"Accept": "application/json, text/plain, */*"}
SQFT_PER_ACRE = 43560.0

# Categories never shipped (not land-value parcels): see the add-city skill §5.
NON_LAND_CATEGORIES = {"Mineral / Oil & Gas", "Personal Property / Inventory", "Utility"}

# Owner-name fragments for public / institutional land. Matched as substrings of the
# upper-cased owner name. City-specific names are passed in addition to these.
#
# Keep these specific: a loose fragment ("USA ", "FEDERAL ") also matches private owners
# (USA Rentals, Federal Express). The ETL logs every matched owner so the list can be audited.
PUBLIC_OWNER_KEYWORDS = [
    "STATE OF TEXAS", "TEXAS DEPARTMENT", "TEXAS DEPT", "TXDOT",
    "UNITED STATES OF AMERICA", "UNITED STATES GOVT", "UNITED STATES GOVERNMENT",
    "US GOVT", "U.S. GOVERNMENT", "HOUSING AUTHORITY",
    " ISD", "INDEPENDENT SCHOOL", "BOARD OF REGENTS", "UNIVERSITY OF TEXAS",
    "COMMUNITY COLLEGE", "COLLEGE DISTRICT",
    "WATER DISTRICT", "RIVER AUTHORITY", "TRANSPORTATION AUTHORITY", "TRANSIT AUTHORITY",
    "HOSPITAL DISTRICT",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_arcgis_geojson(layer_url: str, *, where: str = "1=1", out_fields: str = "*",
                         geometry: dict | None = None, cache: Path | None = None,
                         page: int = 2000, order_by: str = "OBJECTID") -> gpd.GeoDataFrame:
    """Paginated GeoJSON pull from an ArcGIS FeatureServer layer, cached to parquet.

    `geometry` is an optional envelope dict {xmin, ymin, xmax, ymax} in EPSG:4326 used as a
    spatial pre-filter (intersects). Pages are ordered by `order_by` so offsets are stable.
    """
    if cache is not None and cache.exists():
        log(f"Using cached pull: {cache.name}")
        return gpd.read_parquet(cache)
    q = f"{layer_url.rstrip('/')}/query"
    base = {"where": where, "f": "json"}
    if geometry is not None:
        base.update({"geometry": ",".join(str(geometry[k]) for k in ("xmin", "ymin", "xmax", "ymax")),
                     "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                     "spatialRel": "esriSpatialRelIntersects"})
    total = requests.get(q, params={**base, "returnCountOnly": "true"}, headers=HEADERS,
                         timeout=180).json().get("count", 0)
    log(f"Pulling {total:,} features from {layer_url.split('/services/')[-1]} ...")
    pages, off = [], 0
    while off < total:
        gdf = None
        for attempt in range(5):
            try:
                r = requests.get(q, params={**base, "outFields": out_fields, "returnGeometry": "true",
                                            "orderByFields": order_by, "resultOffset": off,
                                            "resultRecordCount": page, "outSR": 4326,
                                            "f": "geojson"}, headers=HEADERS, timeout=300)
                r.raise_for_status()
                gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
                break
            except Exception as e:  # noqa: BLE001 - retried, then raised below
                log(f"  retry {attempt + 1} @offset {off}: {type(e).__name__}: {str(e)[:80]}")
                time.sleep(5 * (attempt + 1))
        if gdf is None:
            raise RuntimeError(f"ArcGIS pull failed at offset {off}")
        if not len(gdf):
            break
        pages.append(gdf)
        off += len(gdf)
        if off % 20000 < page:
            log(f"  fetched {off:,}/{total:,}")
    out = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    if len(out) != total:
        log(f"  WARNING: pulled {len(out):,} rows but the server counted {total:,}")
    if cache is not None:
        out.to_parquet(cache, index=False)
        log(f"  cached -> {cache.name} ({len(out):,} rows)")
    return out


def categorize_sptb(code) -> str:
    """Texas SPTB state class -> coarse property category (statewide codes)."""
    raw = str(code or "").strip().upper()
    if not raw or raw in ("NAN", "NONE"):
        return "Other"
    if raw.startswith("X"):
        return "Exempt"
    if raw.startswith("A"):
        return "Single Family"
    if raw.startswith("B"):
        return "Multifamily"
    if raw.startswith("C"):
        return "Vacant Commercial" if raw.startswith("C1C") or raw.startswith("C2") else "Vacant Residential"
    if raw[0] in ("D", "E"):
        return "Agricultural / Rural"
    if raw.startswith("F"):
        return "Industrial" if raw.startswith("F2") else "Commercial"
    if raw.startswith("G"):
        return "Mineral / Oil & Gas"
    if raw[0] in ("J", "U"):
        return "Utility"
    if raw[0] in ("L", "M", "N", "O", "S"):
        return "Personal Property / Inventory"
    return "Other"


def owner_is_public(owner: pd.Series, extra: list[str] = ()) -> pd.Series:
    kw = [k for k in (*PUBLIC_OWNER_KEYWORDS, *extra) if k]
    pat = "|".join(pd.Series(kw).str.replace(r"([.()&+?*\[\]^$|\\])", r"\\\1", regex=True))
    return owner.fillna("").astype(str).str.upper().str.contains(pat, regex=True)


def merge_stacked_footprints(gdf: gpd.GeoDataFrame, key: str, sum_cols: list[str],
                             area_col: str | None = None,
                             dup_subset: list[str] | None = None) -> gpd.GeoDataFrame:
    """Sum every value record that sits on the identical polygon (skill §2a / §6).

    Texas rolls stack several records on one footprint: condo units mapped onto the whole
    complex (each owning "% OF COMMON AREA"), undivided-interest owners each carrying their
    share, and one lot split into accounts by a school-district or TIF boundary. All of them
    are partial values of the same land, so they are SUMMED onto the footprint. Exact repeats
    (same account, polygon and values) are dropped first so nothing is double-counted.

    The surviving row keeps the attributes of its highest-land-value record (category, link)
    and gains `n_records`. Its `area_col` (the appraiser's land size, which for a condo unit
    is its share or 0) is blanked so the denominator falls back to the footprint area.
    """
    g = gdf.copy()
    g["_gk"] = g.geometry.apply(lambda x: x.wkb_hex if x is not None else "")
    before = len(g)
    g = g.drop_duplicates(subset=[key, "_gk", *(dup_subset or []), *sum_cols])
    g["n_records"] = g.groupby("_gk")[key].transform("size")
    multi = g["n_records"] > 1
    log(f"Stacked footprints: {g.loc[multi, '_gk'].nunique():,} polygons carrying "
        f"{int(multi.sum()):,} records (exact repeats dropped: {before - len(g):,}); "
        f"largest stack {int(g['n_records'].max())}")
    if multi.any():
        m = g[multi].sort_values(sum_cols[0], ascending=False)
        rep = m.drop_duplicates("_gk").set_index("_gk")
        sums = m.groupby("_gk")[sum_cols].sum(min_count=1)
        rep[sum_cols] = sums.loc[rep.index, sum_cols]
        if area_col:
            rep[area_col] = np.nan
        g = pd.concat([g[~multi], rep.reset_index()], ignore_index=True)
    return gpd.GeoDataFrame(g.drop(columns="_gk"), geometry="geometry", crs=gdf.crs)


INSTITUTIONAL_RE = (r"CITY|COUNTY|STATE|TEXAS|\bISD\b|SCHOOL|UNIV|REGENTS|CHURCH|\bCH\b|CHRCH|"
                    r"BAPTIST|METHODIST|CATHOLIC|PRESB|MINISTR|LUTHER|EPISCOPAL|CHRIST|TEMPLE|"
                    r"MOSQUE|ISLAMIC|MASONIC|DISTRICT|AUTHORITY|FOUNDATION|HOUSING|U S A|"
                    r"UNITED STATES|POSTAL|DIOCESE|ASSEMBLY|CONGREGATION|IGLESIA|CHAPEL|"
                    r"FELLOWSHIP|HABITAT|YMCA|HOSPITAL|COLLEGE|ACADEMY|MISSION|CEMETERY")


def owner_is_institutional(owner: pd.Series) -> pd.Series:
    """Broad public / religious / charitable owner test. Only meaningful combined with a
    value-based exemption signal (e.g. $0 taxable), never on its own."""
    return owner.fillna("").astype(str).str.upper().str.contains(INSTITUTIONAL_RE, regex=True)


def dedup_accounts(gdf: gpd.GeoDataFrame, key: str, value_cols: list[str]) -> gpd.GeoDataFrame:
    """Collapse an account split across several polygons into one row (skill §2 / §2a).

    Asserts that the account's values are identical on every piece (a broadcast), then takes
    them once (`first`, never sum) and unions every piece's geometry so the footprint is at the
    same grain as the value it is divided by. Run merge_stacked_footprints first where the roll
    stacks several records on one polygon.
    """
    dup = gdf[key].duplicated(keep=False)
    log(f"Keys with >1 geometry row: {gdf.loc[dup, key].nunique():,} ({int(dup.sum()):,} rows)")
    if not dup.any():
        return gdf
    d = gdf[dup]
    for c in value_cols:
        n_distinct = d.groupby(key)[c].nunique(dropna=False)
        bad = n_distinct[n_distinct > 1]
        if len(bad):
            raise AssertionError(f"{len(bad)} accounts carry differing '{c}' across their pieces "
                                 f"(e.g. {list(bad.index[:5])}) - not a broadcast; decide sum vs first")
    first_cols = [c for c in gdf.columns if c not in ("geometry", key)]
    single = gdf[~dup]
    coll = d.groupby(key, sort=False).agg({c: "first" for c in first_cols})
    coll["geometry"] = d.groupby(key, sort=False)["geometry"].apply(
        lambda gs: unary_union([g for g in gs if g is not None]))
    coll = gpd.GeoDataFrame(coll.reset_index(), geometry="geometry", crs=gdf.crs)
    return gpd.GeoDataFrame(pd.concat([single, coll], ignore_index=True), geometry="geometry", crs=gdf.crs)


def fix_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf["geometry"] = gdf["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
    return gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]


def clip_centroid_within(gdf: gpd.GeoDataFrame, boundary) -> gpd.GeoDataFrame:
    """Keep parcels whose representative point falls inside the authoritative boundary."""
    pts = gdf.geometry.representative_point()
    return gdf[pts.within(boundary)].copy()


CONDO_LEGAL_RE = r"CONDO|CONDMINIUM|OF COMMON AREA|% IN CE|\bCE\b"
DEV_SPLIT_RE = r"^(.*?)(?:\s+BLDG|\s+BUILDING|\s+BLOCK|\s+LOT|\s+UNIT|\s+SUITE|\s+OFFICE CENTER|\s+APT|$)"


def merge_condo_stubs(gdf: gpd.GeoDataFrame, legal_col: str, sum_cols: list[str],
                      area_col: str | None = None, stub_max_sqft: float = 3000.0,
                      min_units: int = 3, nominal_land: float = 1000.0) -> gpd.GeoDataFrame:
    """Merge condo units mapped as tiny per-unit polygons onto their development (skill §6b).

    Units are recognised by a condo legal description (CONDO / "% OF COMMON AREA" / CE) on a
    footprint under `stub_max_sqft`, grouped by the development name that prefixes the legal
    description (text before BLDG/BLOCK/LOT/UNIT/SUITE). Fee-simple townhomes (no condo
    wording) are left alone: their small lots are real lots.

    Each development is merged DOWN onto a nominal-value parcel that contains its units (the
    developer's common area, e.g. $1 land) when one exists; otherwise onto the convex hull of
    its units minus any other parcel's land (§6b step 4 fallback). Values are summed; the
    surviving row keeps the highest-land-value unit's attributes, `n_records` = units merged,
    and `area_col` is blanked so the denominator is the merged footprint.
    """
    from shapely.ops import unary_union
    utm = gdf.estimate_utm_crs()
    g = gdf.to_crs(utm).reset_index(drop=True)
    legal = g[legal_col].fillna("").astype(str).str.upper()
    sqft = g.geometry.area * 10.7639
    is_unit = legal.str.contains(CONDO_LEGAL_RE, regex=True) & (sqft < stub_max_sqft)
    g["_dev"] = legal.str.extract(DEV_SPLIT_RE)[0].str.strip()
    sizes = g[is_unit].groupby("_dev").size()
    devs = sizes[sizes >= min_units].index
    unit_idx = g.index[is_unit & g["_dev"].isin(devs)]
    if not len(unit_idx):
        log("Condo stubs: none")
        return gdf
    others = g.drop(index=unit_idx)
    land0 = pd.to_numeric(others[sum_cols[0]], errors="coerce").fillna(0)
    oidx = others.sindex
    new_rows, drop_idx, via_parent = [], set(unit_idx), 0
    for dev, units in g.loc[unit_idx].groupby("_dev"):
        pts = units.geometry.representative_point()
        hull = unary_union(list(units.geometry)).convex_hull
        cand = others.iloc[list(oidx.query(hull, predicate="intersects"))]
        parent = cand[(land0.loc[cand.index] <= nominal_land)
                      & cand.geometry.apply(lambda p: pts.within(p).mean() >= 0.8)]
        rep = units.sort_values(sum_cols[0], ascending=False).iloc[0].copy()
        if len(parent):
            pi = parent.geometry.area.idxmax()
            geom = parent.loc[pi, "geometry"]
            drop_idx.add(pi)
            vals = units[sum_cols].sum(min_count=1) + parent.loc[pi, sum_cols].fillna(0)
            via_parent += 1
        else:
            blockers = cand.geometry
            geom = hull.difference(unary_union(list(blockers))) if len(blockers) else hull
            geom = unary_union([geom, *units.geometry])
            vals = units[sum_cols].sum(min_count=1)
        rep[sum_cols] = vals.values
        rep["geometry"] = geom
        rep["n_records"] = len(units)
        if area_col:
            rep[area_col] = np.nan
        new_rows.append(rep)
    out = pd.concat([g.drop(index=list(drop_idx)), pd.DataFrame(new_rows)], ignore_index=True)
    out = gpd.GeoDataFrame(out.drop(columns="_dev"), geometry="geometry", crs=utm).to_crs(gdf.crs)
    log(f"Condo stubs: {len(unit_idx):,} units in {len(devs)} developments merged "
        f"({via_parent} onto a nominal-value common-area parcel, {len(devs) - via_parent} onto "
        f"the unit hull): {sorted(devs)}")
    return out
