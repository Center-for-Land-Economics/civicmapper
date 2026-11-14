"""Shared helpers for Texas city ETLs (appraisal-district parcel feeds).

Texas appraisal districts all report the statewide SPTB "state class" code, publish through a
handful of ArcGIS schemas, and share the same dedup traps, so the pieces that were copy-pasted
across run_dallas.py / run_sanantonio.py / run_bcs.py live here for new Texas cities.
Existing scripts are left as they are (they ship as-is); new cities import from this module.

    from tx_common import fetch_arcgis_geojson, categorize_sptb, dedup_accounts
"""
from __future__ import annotations

import re
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


def _esri_polygon(geom) -> str:
    """Shapely (Multi)Polygon in EPSG:4326 -> ArcGIS JSON polygon geometry string."""
    import json
    parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    rings = []
    for p in parts:
        rings.append([list(c) for c in p.exterior.coords])
        rings.extend([list(c) for c in r.coords] for r in p.interiors)
    return json.dumps({"rings": rings, "spatialReference": {"wkid": 4326}})


def fetch_arcgis_geojson(layer_url: str, *, where: str = "1=1", out_fields: str = "*",
                         geometry: dict | None = None, polygon=None, cache: Path | None = None,
                         page: int = 2000, order_by: str = "OBJECTID", workers: int = 4,
                         id_field: str = "OBJECTID") -> gpd.GeoDataFrame:
    """Paginated GeoJSON pull from an ArcGIS FeatureServer layer, cached to parquet.

    Spatial pre-filter (intersects), either `geometry` = envelope dict {xmin, ymin, xmax, ymax}
    or `polygon` = a shapely (Multi)Polygon in EPSG:4326 (sent by POST; simplify it first - a
    city outline instead of its bounding box cut Fort Worth's pull from 741k to ~330k rows).
    The filter is resolved once to object IDs, which are then fetched in chunks of `page`,
    `workers` at a time, each chunk cached beside `cache` so an interrupted pull resumes.
    """
    from concurrent.futures import ThreadPoolExecutor
    if cache is not None and cache.exists():
        log(f"Using cached pull: {cache.name}")
        return gpd.read_parquet(cache)
    q = f"{layer_url.rstrip('/')}/query"
    base = {"where": where}
    if polygon is not None:
        base.update({"geometry": _esri_polygon(polygon), "geometryType": "esriGeometryPolygon",
                     "inSR": 4326, "spatialRel": "esriSpatialRelIntersects"})
    elif geometry is not None:
        base.update({"geometry": ",".join(str(geometry[k]) for k in ("xmin", "ymin", "xmax", "ymax")),
                     "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                     "spatialRel": "esriSpatialRelIntersects"})
    # Resolve the spatial filter ONCE to object IDs, then fetch by ID chunks: no spatial
    # predicate and no deep offsets per page (offset paging over a polygon query timed out
    # repeatedly past ~60k rows on Fort Worth). Chunks are cached so a failure resumes.
    ids = requests.post(q, data={**base, "returnIdsOnly": "true", "f": "json"}, headers=HEADERS,
                        timeout=600).json()
    oid_field = ids.get("objectIdFieldName", id_field)
    oids = sorted(ids.get("objectIds") or [])
    total = len(oids)
    chunks = [oids[i:i + page] for i in range(0, total, page)]
    part_dir = (cache.parent / f".{cache.stem}_parts") if cache is not None else None
    if part_dir is not None:
        part_dir.mkdir(parents=True, exist_ok=True)
    log(f"Pulling {total:,} features from {layer_url.split('/services/')[-1]} "
        f"({len(chunks)} chunks, {workers} at a time) ...")

    def fetch(i: int) -> gpd.GeoDataFrame:
        part = part_dir / f"{i:05d}.parquet" if part_dir is not None else None
        if part is not None and part.exists():
            return gpd.read_parquet(part)
        for attempt in range(8):
            try:
                r = requests.post(q, data={"objectIds": ",".join(map(str, chunks[i])),
                                           "outFields": out_fields, "returnGeometry": "true",
                                           "outSR": 4326, "f": "geojson"}, headers=HEADERS, timeout=600)
                r.raise_for_status()
                js = r.json()
                if "features" not in js:
                    raise RuntimeError(str(js.get("error", js))[:120])
                gdf = gpd.GeoDataFrame.from_features(js["features"], crs="EPSG:4326")
                if part is not None:
                    gdf.to_parquet(part, index=False)
                return gdf
            except Exception as e:  # noqa: BLE001 - retried, then raised below
                log(f"  retry {attempt + 1} chunk {i}: {type(e).__name__}: {str(e)[:80]}")
                time.sleep(min(120, 10 * (attempt + 1)))
        raise RuntimeError(f"ArcGIS pull failed at chunk {i} (re-run to resume)")

    pages, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for gdf in ex.map(fetch, range(len(chunks))):
            pages.append(gdf)
            done += len(gdf)
            if done % 20000 < page:
                log(f"  fetched {done:,}/{total:,}")
    out = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    if id_field in out.columns:
        out = out.drop_duplicates(id_field)
    if len(out) != total:
        log(f"  WARNING: pulled {len(out):,} rows but the server counted {total:,}")
    if cache is not None:
        out.to_parquet(cache, index=False)
        log(f"  cached -> {cache.name} ({len(out):,} rows)")
        if part_dir is not None:
            import shutil
            shutil.rmtree(part_dir, ignore_errors=True)
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
        # Local "...C" codes (TAD's C1C / C2C) mark commercial vacant land. Statewide C2 is
        # COLONIA lots (residential), so a bare C2 is residential (El Paso: 97k colonia lots).
        return "Vacant Commercial" if raw in ("C1C", "C2C") or raw.startswith(("C1C", "C2C")) else "Vacant Residential"
    if raw[0] in ("D", "E"):
        return "Agricultural / Rural"
    if raw.startswith("F"):
        return "Industrial" if raw.startswith("F2") else "Commercial"
    if raw.startswith("G"):
        return "Mineral / Oil & Gas"
    if raw[0] in ("J", "U"):
        return "Utility"
    if raw[0] == "O":
        # SPTB O = real-property RESIDENTIAL INVENTORY: lots and new homes held by builders
        # (8.6k in Fort Worth's new subdivisions). Real land - not personal property.
        return "Residential Inventory"
    if raw[0] in ("L", "M", "N", "S"):
        return "Personal Property / Inventory"
    return "Other"


def owner_is_public(owner: pd.Series, extra: list[str] = ()) -> pd.Series:
    kw = [k for k in (*PUBLIC_OWNER_KEYWORDS, *extra) if k]
    # Short keywords must end on a word boundary: " ISD" is a school district, not owner
    # ISDELL / ISDARELY. Long ones stay prefixes ("...IMPROVEMENT CORP" -> "CORPORATION").
    pat = "|".join(k + (r"\b" if len(raw.strip()) <= 5 and re.search(r"\w$", raw) else "")
                   for raw, k in zip(kw, pd.Series(kw).str.replace(r"([.()&+?*\[\]^$|\\])", r"\\\1", regex=True)))
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


# Condo regime = condo wording or a FRACTIONAL share of the common element ("& .0129585 OF
# COMMON AREA", "14.095% COMMON AREA", "33.14% IN CE"). Plain "& PART OF COMMON AREA" is a
# fee-simple lot in a planned development and is excluded (see CONDO_EXCLUDE_RE).
CONDO_LEGAL_RE = (r"CONDO|CONDMINIUM|[0-9.]+\s*%?\s*(?:OF\s+)?(?:COMMON AREA|CE)\b|% IN CE|\bCE\b"
                  # El Paso phrasings: "& 15.30 UND% IN COMMON AREA", "PLUS 8.75 % INT IN COM AREA",
                  # "& UND 2.11% INT IN CMN AREA"
                  r"|UND\s*%\s+IN\s+(?:COMMON|COM|CMN)\b|%\s*(?:UND\s+)?INT\s+IN\s+(?:THE\s+)?(?:COMMON|COM|CMN)\b"
                  # fractional shares: "& UND 1/50 INT IN COMMON AREA"
                  r"|UND\s+\d+/\d+\s+INT\s+IN\s+(?:THE\s+)?(?:COMMON|COM|CMN)\b"
                  # Collin CAD: "MIDWAY MEDICAL CENTER, BLDG 3108, UNIT 200, 9.79 % CMN INT"
                  r"|%\s*CMN\s+INT\b")
CONDO_EXCLUDE_RE = r"PART OF (?:THE )?(?:COMMON AREA|CE)\b"
DEV_SPLIT_RE = (r"^(.*?)(?:\s+BLDG|\s+BUILDING|\s+BLOCK|\s+LOT|\s+UNIT|\s+SUITE|\s+SITE|\s+TR\b|\s+TRACT\b|\s+REPLAT\b|\s+OFFICE CENTER|"
                r"\s+APT|\s+COMMON|\s+ALL UNITS|$)")

# Public road right-of-way half-widths (m) by Overture road class, used to keep a gap fill out
# of streets. service/footway/path are deliberately absent: those are parking aisles and walks
# INSIDE a site.
ROW_HALF_WIDTH_M = {"motorway": 30, "trunk": 25, "primary": 18, "secondary": 15, "tertiary": 12,
                    "residential": 9, "unclassified": 9, "living_street": 7, "unknown": 7}


def development_key(legal: pd.Series) -> pd.Series:
    """Development name from a condo / subdivision legal description, across CAD formats.

    Tarrant: "CLOISTERS CONDOMINIUMS, THE BLOCK S LOT 1212 & .0020829 OF COMMON AREA"
    El Paso: "NORTH MESA PROF CTR CONDO #1 SOUTH (1895.00 SQ FT) & 11.46 UND% IN COMMON AREA",
             "UNT D32 BLD D PIAZZA ESCONDIDA CONDOMINIUMS PLUS 2.526 % INT IN COM AREA",
             "A PLACITAS DE ZARAGOSA OFFICE CONDOS 1 #A-1 (2722.00 SQ FT) & 12.70% UND INT ..."
    Drops parentheticals, the share clause (& ... / PLUS ...), unit / building / suite tokens,
    '#' unit numbers, a lone leading building letter and trailing compass words, then cuts at
    BLOCK / LOT / UNIT / ... as before. Phase numbers stay (separate regimes, separate common areas).
    """
    s = legal.fillna("").astype(str).str.upper()
    s = s.str.replace(r"\([^)]*\)", " ", regex=True)
    s = s.str.split(r"\s&\s|\s&$|\sPLUS\s", n=1, regex=True).str[0]
    # Leading unit / building tokens only (El Paso "UNT D32 BLD D PIAZZA ..."); trailing ones
    # ("... CONDOS BLDG F UNIT 1101-8") are cut by DEV_SPLIT_RE below.
    s = s.str.replace(r"^(?:(?:UNT|UNIT|BLD|BLDG|BUILDING|SUITE|STE|APT)\s+[A-Z0-9-]+\s*)+", " ", regex=True)
    s = s.str.replace(r"#\s*\S+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.str.extract(DEV_SPLIT_RE)[0].fillna("").str.strip()
    s = s.str.replace(r"^[A-Z]\s+(?=\S{3,})", "", regex=True)
    s = s.str.replace(r"^\d+\s+(?=[A-Z]{3,})", "", regex=True)   # leading block number ("105 VISTA DEL SOL")
    s = s.str.replace(r"(?:\s+(?:NORTH|SOUTH|EAST|WEST|N|S|E|W))+$", "", regex=True)
    # Trailing unit codes with letters ("... CONDOMINIUMS F-1", "G9", "C-6"); bare numbers stay
    # because they are phases ("PLACITAS DE ZARAGOSA OFFICE CONDOS 1" vs "... 2").
    s = s.str.replace(r"\s+[A-Z]{1,2}-?\d+[A-Z]?$", "", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def road_row_polygons(bounds: tuple, release: str | None = None) -> gpd.GeoDataFrame:
    """Public road right-of-way as buffered Overture road centrelines (EPSG:4326)."""
    import sys
    import pyarrow.compute as pc
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from data.scripts import overture_source
    from data.scripts.classify_parking_surface import OVERTURE_RELEASE
    rel = overture_source.resolve_release(release or OVERTURE_RELEASE)
    seg = overture_source.read("transportation", "segment", tuple(bounds), release=rel,
                               columns=["class", "geometry"], where=pc.field("subtype") == "road")
    seg = seg[seg["class"].isin(ROW_HALF_WIDTH_M)].copy()
    utm = seg.estimate_utm_crs()
    su = seg.to_crs(utm)
    su["geometry"] = [g.buffer(ROW_HALF_WIDTH_M[c]) for g, c in zip(su.geometry, su["class"])]
    log(f"Road right-of-way: {len(su):,} Overture {rel} road segments buffered")
    return su.to_crs(4326)


def _fill_holes(geom):
    from shapely.geometry import MultiPolygon, Polygon
    parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    return MultiPolygon([Polygon(p.exterior) for p in parts]) if len(parts) > 1 else Polygon(parts[0].exterior)


def _unit_numbered(legal: pd.Series) -> bool:
    L = legal.fillna("").str.upper()
    unit = L.str.contains(r"\bUNIT\s+[A-Z]?-?\d+", regex=True)
    lot = L.str.contains(r"\b(?:LOT|LT|LTS|BLK|BLOCK)\b", regex=True)
    return bool(len(L)) and float((unit & ~lot).mean()) >= 0.8


def merge_condo_stubs(gdf: gpd.GeoDataFrame, legal_col: str, sum_cols: list[str],
                      area_col: str | None = None, stub_max_sqft: float = 3000.0,
                      min_units: int = 2, nominal_land: float = 1000.0,
                      blockers: gpd.GeoDataFrame | None = None,
                      roads: gpd.GeoDataFrame | None = None, fill_m: float = 40.0,
                      footprint_lot_sqft: float = 1500.0) -> gpd.GeoDataFrame:
    """Merge condo units mapped as unit outlines onto their development's land (skill §6b).

    A development (legal-description prefix before BLDG/BLOCK/LOT/UNIT/SUITE/COMMON) qualifies
    when it has >= `min_units` condo units (CONDO / "% OF COMMON AREA" / CE wording) under
    `stub_max_sqft`; then EVERY condo unit of it is merged (selecting members by size split
    S Arlington Prof Bldg in two). Fee-simple townhomes (no condo wording) keep their lots.

    The development's land, in order of preference:
      1. its own common-area parcel(s): same development name, "COMMON" in the legal
         description, nominal land value (Texas Tax Code §23.18 values association common
         area nominally because its value sits in the units), within 200 m of the units;
         holes (the punched-out unit outlines) are filled;
      2. a nominal-value parcel containing >= 80% of the units;
      3. the unit hull grown into adjacent unparcelled ground, up to `fill_m`, never into
         another parcel (`blockers`, which should include exempt and off-roll polygons) or
         public road right-of-way (`roads`).
    Values are summed (units + nominal parents); the surviving row keeps the highest-land-value
    unit's attributes, `n_records` = units merged, `land_source` says which rule applied, and
    `area_col` is blanked so the denominator is the merged footprint.
    """
    from shapely.ops import unary_union
    utm = gdf.estimate_utm_crs()
    g = gdf.to_crs(utm).reset_index(drop=True)
    legal = g[legal_col].fillna("").astype(str).str.upper()
    sqft = g.geometry.area * 10.7639
    land = pd.to_numeric(g[sum_cols[0]], errors="coerce").fillna(0)
    is_common = legal.str.contains(r"COMMON|OPEN SPACE", regex=True) & (land <= nominal_land)
    is_condo = legal.str.contains(CONDO_LEGAL_RE, regex=True) \
        & ~legal.str.contains(CONDO_EXCLUDE_RE, regex=True) & ~is_common
    g["_dev"] = development_key(legal)
    # A development qualifies by unit-outline stubs OR by having its own nominal common-area
    # parcel near its condo records (catches complexes whose units were already summed onto one
    # polygon, and developments of large unit outlines: Crestgrove, Crestbluff, Chelsea Park).
    is_stub = is_condo & (sqft < stub_max_sqft)
    sizes = g[is_stub].groupby("_dev").size()
    devs = set(sizes[sizes >= min_units].index)
    # ... or by having >= min_units separately drawn condo units of any size (office condos
    # with large unit outlines and no common-area parcel, e.g. Woodland Springs Office Park).
    units_any = g[is_condo].groupby("_dev").size()
    devs |= set(units_any[units_any >= min_units].index)
    condo_union = g[is_condo].groupby("_dev")["geometry"].apply(lambda s: unary_union(list(s)))
    for dev, cg in g[is_common & g["_dev"].isin(condo_union.index)].groupby("_dev"):
        if cg.geometry.distance(condo_union[dev]).min() <= 200:
            devs.add(dev)
    members = is_condo & g["_dev"].isin(devs)
    # Footprint-lot developments: fee-simple townhome lots drawn as building footprints (median
    # under `footprint_lot_sqft`) beside the association's nominal common-area / open-space
    # parcels (Inniswood, Main 7). Under Texas Tax Code §23.18 the common land's value is in the
    # lots, so they are the same case as condos. Real lots in planned developments (0.2-acre
    # Stoneridge lots "& PART OF COMMON AREA") are far above the size test and stay as they are.
    small_lot = ~is_common & ~is_condo & (sqft < stub_max_sqft)
    lot_stats = g[small_lot].groupby("_dev").agg(n=("geometry", "size"))
    lot_stats["med_sqft"] = sqft[small_lot].groupby(g.loc[small_lot, "_dev"]).median()
    # Footprint lots must be the MAJORITY of the subdivision's lots: a big single-family
    # subdivision with a few remnant slivers and an HOA greenbelt (Sendera Ranch, Marine Creek
    # Ranch) is not a townhome development.
    all_lots = g[~is_common & ~is_condo].groupby("_dev").size()
    lot_stats["share"] = lot_stats["n"] / all_lots.reindex(lot_stats.index)
    fdevs = set()
    unit_devs = {d for d in lot_stats.index
                 if _unit_numbered(g.loc[small_lot & (g["_dev"] == d), legal_col])}
    lot_ok = (lot_stats["med_sqft"] < footprint_lot_sqft) | lot_stats.index.isin(unit_devs)
    for dev, st in lot_stats[(lot_stats["n"] >= min_units) & (lot_stats["share"] >= 0.5)
                             & lot_ok].iterrows():
        lots_geom = unary_union(list(g.loc[small_lot & (g["_dev"] == dev), "geometry"]))
        cg = g[is_common & (g["_dev"] == dev)]
        if len(cg) and cg.geometry.distance(lots_geom).min() <= 200:
            fdevs.add(dev)
        elif _unit_numbered(g.loc[small_lot & (g["_dev"] == dev), legal_col]):
            # Unit outlines with no association parcel drawn: lots numbered as UNITs with no
            # block / lot (Collin: "PARKBLUFF ADDITION PHASE I, UNIT 112", 700-980 sqft; West Plano
            # Medical Center office units ~2,000 sqft; any size under the stub cap) are condo
            # units drawn one by one, not fee-simple lots ("OSO RIVER ESTATES UNIT 3 BLK 1 LOT 1A"
            # is a subdivision phase and has a LOT).
            fdevs.add(dev)
    members |= small_lot & g["_dev"].isin(fdevs)
    devs |= fdevs
    unit_idx = g.index[members]
    if not len(unit_idx):
        log("Condo stubs: none")
        return gdf
    others = g.drop(index=unit_idx)
    oidx = others.sindex
    bl = blockers.to_crs(utm) if blockers is not None else others
    bidx = bl.sindex
    rd = roads.to_crs(utm) if roads is not None else None
    ridx = rd.sindex if rd is not None else None
    new_rows, drop_idx, how = [], set(unit_idx), {"common_area_parcel": 0, "nominal_parent": 0, "unit_hull_fill": 0}
    # Split each development name into spatial clusters (units within 150 m of each other), so a
    # name that repeats across the city never hulls units kilometres apart into one parcel.
    ug = g.loc[unit_idx]
    cluster = pd.Series(index=ug.index, dtype=object)
    for dev, units in ug.groupby("_dev"):
        blobs = unary_union(list(units.geometry.buffer(75)))
        blobs = list(getattr(blobs, "geoms", [blobs]))
        pts = units.geometry.representative_point()
        for i, b in enumerate(blobs):
            cluster.loc[pts[pts.within(b)].index] = f"{dev}#{i}"
    for dev_c, units in ug.groupby(cluster):
        dev = dev_c.rsplit("#", 1)[0]
        ugeom = unary_union(list(units.geometry))
        pts = units.geometry.representative_point()
        commons = others[(others["_dev"] == dev) & is_common.loc[others.index]
                         & (others.geometry.distance(ugeom) <= 200)]
        rep = units.sort_values(sum_cols[0], ascending=False).iloc[0].copy()
        vals = units[sum_cols].sum(min_count=1)
        if len(commons):
            geom = unary_union([_fill_holes(unary_union(list(commons.geometry))), ugeom])
            drop_idx.update(commons.index)
            vals = vals + commons[sum_cols].fillna(0).sum()
            src = "common_area_parcel"
        else:
            cand = others.iloc[list(oidx.query(ugeom.convex_hull, predicate="intersects"))]
            parent = cand[(land.loc[cand.index] <= nominal_land)
                          & cand.geometry.apply(lambda p: pts.within(p).mean() >= 0.8)]
            if len(parent):
                pi = parent.geometry.area.idxmax()
                geom = unary_union([_fill_holes(parent.loc[pi, "geometry"]), ugeom])
                drop_idx.add(pi)
                vals = vals + parent.loc[pi, sum_cols].fillna(0)
                src = "nominal_parent"
            else:
                member_ids = set(units.get("acct", pd.Series(dtype=str)))
                zone = ugeom.convex_hull.buffer(fill_m)
                near = bl.iloc[list(bidx.query(zone, predicate="intersects"))]
                if "acct" in near.columns:
                    near = near[~near["acct"].isin(member_ids)]
                near = near[~near.geometry.within(ugeom.buffer(0.5))]
                block = list(near.geometry)
                if rd is not None:
                    block += list(rd.iloc[list(ridx.query(zone, predicate="intersects"))].geometry)
                gap = zone.difference(unary_union(block)) if block else zone
                parts = [p for p in getattr(gap, "geoms", [gap]) if p.distance(ugeom) < 1.0 and p.area > 20]
                geom = unary_union([ugeom, *parts])
                src = "unit_hull_fill"
        how[src] += 1
        rep[sum_cols] = vals.values
        rep["geometry"] = geom
        rep["n_records"] = len(units)
        rep["land_source"] = src
        if area_col:
            rep[area_col] = np.nan
        new_rows.append(rep)
    out = pd.concat([g.drop(index=list(drop_idx)), pd.DataFrame(new_rows)], ignore_index=True)
    out = gpd.GeoDataFrame(out.drop(columns="_dev"), geometry="geometry", crs=utm).to_crs(gdf.crs)
    log(f"Footprint-lot townhome developments merged like condos: {sorted(fdevs)}")
    log(f"Condo developments: {len(unit_idx):,} units in {len(devs)} developments merged "
        f"(land from: {how}; {len(drop_idx) - len(unit_idx)} common-area/nominal parcels absorbed)")
    return out
