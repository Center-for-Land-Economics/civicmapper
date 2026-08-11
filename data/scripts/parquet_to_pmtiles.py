from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from shapely.geometry import box

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.parcel_calculations import add_improvement_ratio_fields
from data.parquet_registry import CITY_PARQUETS, list_cities, resolve_city


AGGREGATE_LAYER_SPECS_BY_CITY: dict[str, list[tuple[str, int]]] = {
    "baltimore": [("parcels_low", 400)],
    # Denver benefits from a medium-detail bridge layer between the coarse
    # citywide aggregate and the full parcel layer.
    "denver": [("parcels_low", 600), ("parcels_mid", 250)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert parquet files to PMTiles format with pre-computed metadata."
    )
    parser.add_argument(
        "--city",
        help="City key to convert (e.g. denver).",
    )
    parser.add_argument(
        "--file",
        help="Override local parquet path.",
    )
    parser.add_argument(
        "--connection-string",
        default=os.getenv("AZURE_STORAGE_CONNECTION_STRING", ""),
        help="Azure Storage connection string (AZURE_STORAGE_CONNECTION_STRING).",
    )
    parser.add_argument(
        "--container",
        default=os.getenv("AZURE_DEV_CONTAINER", "parquets-dev"),
        help="Target container name (defaults to AZURE_DEV_CONTAINER).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload PMTiles and metadata to Azure Blob Storage.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination blob if it exists.",
    )
    parser.add_argument(
        "--tippecanoe",
        default="tippecanoe",
        help="Path to tippecanoe binary (default: tippecanoe).",
    )
    parser.add_argument(
        "--pmtiles",
        default="pmtiles",
        help="Path to pmtiles binary (default: pmtiles).",
    )
    parser.add_argument(
        "--wsl",
        action="store_true",
        help=(
            "Run tippecanoe and pmtiles via WSL (Windows Subsystem for Linux). "
            "Use on Windows when tippecanoe is installed inside WSL2. "
            "Paths are automatically converted to /mnt/<drive>/... form."
        ),
    )
    parser.add_argument(
        "--h3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build the H3 hexagon low-zoom aggregate (one 'parcels_low' layer, each "
            "hex carrying the same field names as parcels — rates area-weighted, totals "
            "summed). This is the STANDARD low-zoom layer for every PMTiles city and is "
            "ON by default. Pass --no-h3 only to fall back to the legacy square-grid "
            "aggregate (kept for reproducing old bakes)."
        ),
    )
    parser.add_argument(
        "--drop-remnants",
        action="store_true",
        help=(
            "Drop tiny sub-500-sqft sliver remnants (likely_remnant=1) before baking. "
            "Their per-sqft is a real account value on a fragment polygon, which can "
            "dominate the area-weighted value of a small/sparse low-zoom hex and show "
            "as a spike. Use for cities with hideRemnants:true in cities.ts so the hex "
            "layer matches the filtered detail layer."
        ),
    )
    return parser.parse_args()


def windows_path_to_wsl(path: Path) -> str:
    """Convert a Windows absolute path to its WSL /mnt/ equivalent.

    Examples:
        C:\\Users\\foo\\bar.geojson  ->  /mnt/c/Users/foo/bar.geojson
        L:\\git\\project\\out.mbtiles -> /mnt/l/git/project/out.mbtiles
    """
    s = str(path).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:]  # already starts with /
        return f"/mnt/{drive}{rest}"
    return s


def resolve_local_path(city: str | None, override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    if city:
        # Try data/jurisidictions/data/<city> first, then data/output/final
        city_meta = CITY_PARQUETS.get(city)
        if city_meta:
            legacy_path = Path("data/jurisidictions/data") / city / city_meta.legacy_filename
            if legacy_path.exists():
                return legacy_path
        return Path("data/output/final") / f"{city}.parquet"
    raise ValueError("Either --city or --file must be provided")


# Fields the PMTiles frontend colours/scales by — the ones we precompute quantile breaks +
# robust percentiles for. Single source of truth, shared with refresh_metadata_percentiles.py.
RENDER_PERCENTILE_FIELDS = [
    "REALLANDVA_per_sqft",
    "REALIMPROV_per_sqft",
    "TLLDIMPROV_per_sqft",
    "IMPR_LAND_PCT",
    "land_value_per_sqft",
    "improvement_value_per_sqft",
    "full_market_value_per_sqft",
]


def compute_render_percentiles(gdf: gpd.GeoDataFrame):
    """For each render field present with data, return (field, percentiles, quantileBreaks).

    percentiles carries p1/p99 (height + fallback) and p999 (the robust colour-domain top — a high
    that reaches the expensive end WITHOUT a lone sliver-inflated outlier/max wasting the range).
    Shared by the full bake AND the metadata-only refresh so the two never drift.
    """
    out = []
    for field in RENDER_PERCENTILE_FIELDS:
        if field in gdf.columns:
            values = gdf[field].dropna()
            if len(values) > 0:
                breaks = [float(np.percentile(values, p)) for p in (20, 40, 60, 80)]
                pct = {
                    "p1": float(np.percentile(values, 1)),
                    "p99": float(np.percentile(values, 99)),
                    "p999": float(np.percentile(values, 99.9)),
                }
                out.append((field, pct, breaks))
    return out


def compute_metadata(gdf: gpd.GeoDataFrame, dev_category_field: str, orig_category_field: str) -> dict:
    """Compute metadata: statistics, categories, underutilized totals, quantile breaks."""
    metadata: dict = {
        "statistics": {},
        "categories": {"refined": [], "original": []},
        "underutilizedTotals": {
            "Vacant": 0,
            "Parking Lot": 0,
            "Underdeveloped": 0,
            "totalNonExempt": 0,
        },
        "quantileBreaks": {},
        "percentiles": {},
    }

    # Compute statistics for all numeric fields
    numeric_cols = gdf.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == "geometry":
            continue
        values = gdf[col].dropna()
        if len(values) > 0:
            metadata["statistics"][col] = {
                "min": float(values.min()),
                "max": float(values.max()),
            }

    # Compute categories
    if dev_category_field in gdf.columns:
        refined_cats = gdf[dev_category_field].dropna().unique().tolist()
        metadata["categories"]["refined"] = sorted([str(c) for c in refined_cats if c])
    if orig_category_field in gdf.columns:
        orig_cats = gdf[orig_category_field].dropna().unique().tolist()
        metadata["categories"]["original"] = sorted([str(c) for c in orig_cats if c])

    # NOTE: region-group breakdowns (metadata["groups"]) are emitted by encode_categoricals(),
    # which also integer-encodes the categorical columns — so the id<->name mapping there matches
    # the integer ids baked into the tiles. (Runs after this, before write/bake.)

    # Compute underutilized totals
    if "REALLANDVA" in gdf.columns and "exemption_flag" in gdf.columns:
        exempt_mask = gdf["exemption_flag"] == 0
        non_exempt = gdf[exempt_mask]
        metadata["underutilizedTotals"]["totalNonExempt"] = float(
            non_exempt["REALLANDVA"].sum()
        )

        for cat in ["Vacant", "Parking Lot", "Underdeveloped"]:
            if dev_category_field in gdf.columns:
                cat_mask = (gdf[dev_category_field] == cat) & exempt_mask
                metadata["underutilizedTotals"][cat] = float(
                    gdf.loc[cat_mask, "REALLANDVA"].sum()
                )

        # Underdeveloped improvement-share breakdown: land value + parcel count
        # bucketed by improvements as a % of total parcel value. The frontend
        # (viz/src/main.ts UNDERDEV_BUCKETS) reads these flat keys.
        if dev_category_field in gdf.columns and "REALIMPROV" in gdf.columns:
            ud = gdf[(gdf[dev_category_field] == "Underdeveloped") & exempt_mask]
            land = ud["REALLANDVA"].astype(float)
            impr = ud["REALIMPROV"].astype(float)
            total = land + impr
            impr_pct = (impr / total.where(total > 0)) * 100
            for key, lo, hi in [("lt10", 0, 10), ("10_25", 10, 25), ("25_50", 25, 101)]:
                bucket = (impr_pct >= lo) & (impr_pct < hi)
                metadata["underutilizedTotals"][f"Underdeveloped_{key}"] = float(
                    land[bucket.fillna(False)].sum()
                )
                metadata["underutilizedTotals"][f"Underdeveloped_{key}_count"] = int(
                    bucket.fillna(False).sum()
                )

    # Quantile breaks + robust percentiles (incl. p99.9 colour-domain top) for the render fields.
    for field, pct, breaks in compute_render_percentiles(gdf):
        metadata["quantileBreaks"][field] = breaks
        metadata["percentiles"][field] = pct

    # Geographic extent in EPSG:4326 ([minLon, minLat, maxLon, maxLat]). The web
    # client uses this to fit the map to the city instead of hard-coding per-city
    # coordinates. (The client also reads bounds straight from the PMTiles header;
    # this is a metadata-level fallback.)
    try:
        geo = gdf.to_crs(4326) if gdf.crs is not None and gdf.crs.to_epsg() != 4326 else gdf
        minx, miny, maxx, maxy = geo.total_bounds
        metadata["bounds"] = [float(minx), float(miny), float(maxx), float(maxy)]
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️ Could not compute bounds for metadata: {exc}")

    return metadata


def convert_parquet_to_geojson(parquet_path: Path, output_path: Path) -> None:
    """Convert parquet to GeoJSON."""
    print(f"Loading parquet: {parquet_path}")
    gdf = gpd.read_parquet(parquet_path)

    print(f"Loaded {len(gdf):,} features")
    print(f"Writing GeoJSON: {output_path}")
    write_geojson(gdf, output_path)
    print(f"✅ GeoJSON written: {output_path}")


def encode_categoricals(gdf: gpd.GeoDataFrame, metadata: dict) -> None:
    """Integer-encode the categorical region fields IN PLACE and emit their id<->name maps.

    These fields (jurisdiction / council_district / super_neighborhood / civic_club) are long
    strings repeated across ~1.3M features (hexes + parcels) — they dominate each tile's string
    pool and make decode/bucket-build slow (the profiled main-thread bottleneck). Storing a tiny
    int id per feature instead shrinks tiles and speeds the load path.

    metadata["groups"][field] = {"ids": [name_for_id_0, name_for_id_1, ...], "counts": {name: n}}
    so the tile stores id `i` and the frontend recovers the name via ids[i] (and maps name->id to
    build the layer filter). Ids are ordered by descending count (deterministic). Must run AFTER
    compute_metadata (which reads the original strings) and BEFORE write_geojson/build_h3_aggregate
    (so parcels AND the per-hex dominant value are encoded with the same mapping).
    """
    groups: dict = {}
    for field in H3_CATEGORICAL_FIELDS:
        if field not in gdf.columns:
            continue
        vc = gdf[field].dropna().value_counts()  # descending count
        if vc.empty:
            continue
        names = [str(k) for k in vc.index]
        counts = {n: int(c) for n, c in zip(names, vc.values)}
        id_of = {k: i for i, k in enumerate(vc.index)}
        ids_col = gdf[field].map(id_of)
        if ids_col.isna().any():
            ids_col = ids_col.fillna(-1)  # unmapped (e.g. a stray null) -> sentinel, matches nothing
        gdf[field] = ids_col.astype("int64")
        groups[field] = {"ids": names, "counts": counts}
        print(f"  encoded {field}: {len(names)} ids")
    if groups:
        metadata["groups"] = groups
    if "jurisdiction" in groups:
        metadata["jurisdictions"] = groups["jurisdiction"]["counts"]  # back-compat name->count alias


def add_region_value_totals(gdf: gpd.GeoDataFrame, metadata: dict) -> None:
    """Add value/acre totals for the Land Value blurb ("$Y of land value over X acres in Z"):
      - metadata['cityTotals'] = {acres, land, impr, total} — citywide, emitted for EVERY city, so
        cities with no region groups still get a blurb (as if every region were selected).
      - metadata['groups'][field]['totals'][region] — per-region breakdown, when region groups exist.

    Runs AFTER encode_categoricals: the region columns are now int ids, so names are recovered via
    groups[field]['ids']. Acres come from the projected geometry (equal-area EPSG:6933), NOT the
    land_area_acres column, which can carry corrupt outliers (see land-area-acres notes). Skipped
    quietly if the value columns are absent — the frontend just hides the blurb then."""
    if "current_full_land_value" in gdf.columns:
        land = pd.to_numeric(gdf["current_full_land_value"], errors="coerce")
    elif "REALLANDVA" in gdf.columns:
        land = pd.to_numeric(gdf["REALLANDVA"], errors="coerce")
    else:
        return
    land = land.fillna(0.0)
    if "improvement_value" in gdf.columns:
        impr = pd.to_numeric(gdf["improvement_value"], errors="coerce").fillna(0.0)
    elif "REALIMPROV" in gdf.columns:
        impr = pd.to_numeric(gdf["REALIMPROV"], errors="coerce").fillna(0.0)
    else:
        # Combined/land-value-only cities ship no improvement column; land still sums fine.
        impr = pd.Series(0.0, index=gdf.index)
    total = pd.to_numeric(gdf["full_market_value"], errors="coerce").fillna(0.0) \
        if "full_market_value" in gdf.columns else (land + impr)
    try:
        acres = gdf.geometry.to_crs(6933).area / 4046.8564224
    except Exception as exc:
        print(f"  value totals skipped (geometry area failed: {exc})")
        return
    metadata["cityTotals"] = {"acres": round(float(acres.sum()), 1), "land": round(float(land.sum())),
                              "impr": round(float(impr.sum())), "total": round(float(total.sum()))}
    print(f"  citywide totals: ${metadata['cityTotals']['total']/1e9:.1f}B total "
          f"over {metadata['cityTotals']['acres']:,.0f} acres")
    groups = metadata.get("groups")
    if not groups:
        return
    for field, g in groups.items():
        ids = g.get("ids")
        if field not in gdf.columns or not ids:
            continue
        agg = pd.DataFrame({"_id": gdf[field], "acres": acres, "land": land, "impr": impr, "total": total}) \
            .groupby("_id").sum()
        totals = {}
        for idv, r in agg.iterrows():
            i = int(idv)
            if 0 <= i < len(ids):
                totals[ids[i]] = {"acres": round(float(r.acres), 1), "land": round(float(r.land)),
                                  "impr": round(float(r.impr)), "total": round(float(r.total))}
        g["totals"] = totals
    print(f"  region value totals added for: {sorted(groups.keys())}")


def write_geojson(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Write GeoDataFrame to GeoJSON.

    Round float attributes to 2 decimals first. Full-precision (17-significant-digit)
    floats across ~100k+ features overflow tippecanoe's per-tile attribute value pool
    and get mis-encoded — e.g. land_value_per_sqft picking up an unrelated parcel's
    dollar value, producing false multi-thousand-$/sqft spikes on the map. Rounding
    collapses the distinct-value explosion; display precision is unaffected. NaN is
    preserved (written as null); only ±inf is normalized to null.
    """
    gdf = gdf.copy()
    geom_col = gdf.geometry.name
    for c in gdf.columns:
        if c != geom_col and pd.api.types.is_float_dtype(gdf[c]):
            gdf[c] = gdf[c].replace([np.inf, -np.inf], np.nan).round(2)
    gdf.to_file(output_path, driver="GeoJSON")


def build_low_zoom_aggregate(
    gdf: gpd.GeoDataFrame, cell_size_m: int = 1000
) -> gpd.GeoDataFrame:
    """Aggregate parcels into a coarse grid for low-zoom rendering."""
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf_3857 = gdf.to_crs(3857)
    minx, miny, maxx, maxy = gdf_3857.total_bounds
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        return gpd.GeoDataFrame(geometry=[], crs=gdf_3857.crs)

    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)
    grid_cells = [box(x, y, x + cell_size_m, y + cell_size_m) for x in xs for y in ys]
    if not grid_cells:
        return gpd.GeoDataFrame(geometry=[], crs=gdf_3857.crs)

    grid = gpd.GeoDataFrame({"geometry": grid_cells}, crs=gdf_3857.crs)
    joined = gpd.sjoin(gdf_3857, grid, predicate="intersects")
    if joined.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf_3857.crs)

    numeric_cols = joined.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "index_right"]
    grouped = joined.groupby("index_right")[numeric_cols].median()
    grouped["feature_count"] = joined.groupby("index_right").size()

    grid_agg = grid.loc[grouped.index].copy()
    grid_agg = grid_agg.join(grouped)
    return grid_agg.to_crs(gdf.crs)


def get_aggregate_layer_specs(city: str | None) -> list[tuple[str, int]]:
    if city:
        return AGGREGATE_LAYER_SPECS_BY_CITY.get(city, [("parcels_low", 1000)])
    return [("parcels_low", 1000)]


# H3 hexagon aggregate resolutions baked as low-zoom layers (coarse -> fine),
# each gated to a tile-zoom band. Real parcels take over at parcelMinZoom — z13 by
# default, or earlier when plan_h3_ladder() prunes fine bands (see below).
H3_RESOLUTIONS = [7, 8, 10, 11, 12]
# Tightened 2026-06-24: every mid/high zoom uses one finer H3 resolution than before, so far
# less detail is erased into coarse hexes when zoomed out. Per-zoom resolution is now
#   z0-6 -> r7,  z7-8 -> r8,  z9 -> r10,  z10-11 -> r11,  z12 -> r12
# (was z9 -> r9, z10-11 -> r10, z12 -> r11). The fine_cap (1M parcels) still drops r>10 on
# NYC/county-scale inputs (the last kept res, r10, then stretches to cover z9-12), and the
# >600k binning path keeps r11/r12 tractable on big cities.
# 2026-07-13: plan_h3_ladder() additionally drops any band whose hex is smaller than the
# city's MEDIAN parcel (r12 ~300 m² vs typical ~450-900 m² lots — a 2026-07 sweep found this
# on 22 of 27 PMTiles cities; only NYC/Baltimore rowhouse lots are finer than r12). Unlike
# the count caps, that prune hands off to real parcels earlier (parcelMinZoom moves down)
# instead of stretching a coarser band — hexes that subdivide single lots aren't aggregation,
# just false hex texture, and they cost MORE features than the parcels they proxy.
H3_ZOOM_BANDS = {7: (0, 6), 8: (7, 8), 10: (9, 9), 11: (10, 11), 12: (12, 12)}

# Rate fields (per-sqft, ratios) -> area-weighted mean across a hex's parcels.
# NOTE: the per-sqft rates (land/improvement/full_market _per_sqft) are NO LONGER baked — they're
# dropped from the tiles and computed client-side as value/(land_area_acres·43560). The hex value
# follows as Σvalue/Σarea (a per-sqft of the aggregated cell), since both numerator and denominator
# are carried as apportioned totals below. Saves 3 continuous-float fields per parcel + per hex.
H3_RATE_FIELDS = [
    "IMPR_LAND_PCT", "IMPR_LAND_RATIO", "IMPR_PCT_TOTAL",
    "REALLANDVA_per_sqft", "REALIMPROV_per_sqft", "TLLDIMPROV_per_sqft",
]
# Total/value fields -> summed (apportioned by piece area) across a hex's parcels. land_area_acres
# is the shared denominator for the client-side per-sqft computation.
H3_TOTAL_FIELDS = [
    "current_full_land_value", "improvement_value", "full_market_value",
    "TLLDIMPROV", "REALLANDVA", "REALIMPROV", "land_area_acres",
]
# So the "Vacant & Underdeveloped" tab has a low-zoom layer too (it can't category-
# FILTER hexes the way it filters parcels), each hex carries the share of its land
# that is underutilized + the underutilized land value. Categories match the viz's
# UNDERUTILIZED_DEFAULTS.
H3_CATEGORY_FIELD = "property_land_use_refined"
H3_UNDERUTILIZED = {"Vacant", "Underdeveloped", "Parking Lot"}
# Categorical fields carried onto each hex as the DOMINANT value (the category covering the
# most parcel area in that hex). Lets the frontend filter/treat hexes by jurisdiction at
# low zoom — a hex straddling a city border is assigned to whichever side covers more of it.
H3_CATEGORICAL_FIELDS = ["jurisdiction", "council_district", "super_neighborhood", "civic_club",
                         # Seattle (King County): NMA district / neighborhood region toggles.
                         # NYC: borough / neighborhood (NTA) region toggles.
                         # Only present on cities whose ETL/augment adds them; skipped otherwise.
                         "neighborhood_district", "neighborhood", "borough"]


def plan_h3_ladder(gdf: gpd.GeoDataFrame) -> dict:
    """Decide the per-city H3 ladder and the hex->parcel handoff zoom (parcelMinZoom).

    Two prune mechanisms with different remedies:
      - count caps (perf): above H3_R12_CAP_PARCELS/H3_FINE_CAP_PARCELS the fine bands are
        too many hexes to bake/serve; the coarsest survivor STRETCHES over the rest of the
        hex zooms and parcels still start at 13 (county-scale z12 parcel tiles are too
        heavy — see docs/pmtiles-h3-hexagons.md).
      - median-parcel prune (fidelity): a band whose hex is smaller than the city's MEDIAN
        parcel subdivides typical lots instead of aggregating them. Those bands are skipped
        and parcels take over at the pruned band's zoom instead (earlier parcelMinZoom) —
        at that point real parcels are FEWER features than the hexes they'd replace.

    Areas are true m² (parcels via equal-area EPSG:6933, hexes via h3.cell_area at the
    data's median lat/lng) — NOT web-mercator areas, which are lat-inflated.
    """
    src = gdf if gdf.crs is not None else gdf.set_crs(4326)
    area_m2 = src.geometry.to_crs(6933).area.to_numpy()
    pos = area_m2[area_m2 > 0]
    median_m2 = float(np.median(pos)) if len(pos) else 0.0

    rep = src.geometry.representative_point()
    cy, cx = float(np.nanmedian(rep.y.to_numpy())), float(np.nanmedian(rep.x.to_numpy()))
    return plan_h3_ladder_core(len(gdf), median_m2, cy, cx)


def plan_h3_ladder_core(n_rows: int, median_m2: float, lat: float, lng: float) -> dict:
    """Decision core of plan_h3_ladder, on scalar inputs so audit tooling
    (audit_hex_parcel_handoff.py) can apply the same rule without a full gdf."""
    import h3

    fine_cap = int(os.environ.get("H3_FINE_CAP_PARCELS", "1000000"))  # >1M: r10 max (NYC, full counties)
    r12_cap = int(os.environ.get("H3_R12_CAP_PARCELS", "600000"))     # >600k: drop r12, r11 max
    by_count = [r for r in H3_RESOLUTIONS
                if not (n_rows > fine_cap and r > 10)
                and not (n_rows > r12_cap and r > 11)]
    cy, cx = lat, lng

    hex_m2 = {r: float(h3.cell_area(h3.latlng_to_cell(cy, cx, r), unit="m^2")) for r in by_count}
    # hex areas shrink monotonically with res, so the median prune drops a suffix of the
    # ladder; always keep at least the coarsest band so low zoom stays covered.
    res_list = [r for r in by_count if hex_m2[r] >= median_m2] or by_count[:1]
    median_pruned = [r for r in by_count if r not in res_list]

    default_minzoom = max(mx for _, mx in H3_ZOOM_BANDS.values()) + 1  # 13
    parcel_minzoom = (min(H3_ZOOM_BANDS[r][0] for r in median_pruned)
                      if median_pruned else default_minzoom)

    return {
        "res_list": res_list,
        "hex_maxzoom": parcel_minzoom - 1,
        "parcel_minzoom": parcel_minzoom,
        "median_parcel_m2": median_m2,
        "hex_m2": hex_m2,
        "median_pruned": median_pruned,
    }


def _h3_cell_poly(cell, h3):
    from shapely.geometry import Polygon
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def _h3_pieces_overlay(p, resolution, h3):
    """Small inputs: EXACT areal overlay of parcels against a hex grid covering the bbox.
    Returns per-(parcel,hex) rows with piece_area + parcel_area + the parcel attrs."""
    minx, miny, maxx, maxy = p.total_bounds
    pad = 0.02
    bbox_poly = h3.LatLngPoly([
        (miny - pad, minx - pad), (maxy + pad, minx - pad),
        (maxy + pad, maxx + pad), (miny - pad, maxx + pad)])
    cells = list(h3.h3shape_to_cells(bbox_poly, resolution))
    if not cells:
        return None
    hexes = gpd.GeoDataFrame({"h3": cells},
                             geometry=[_h3_cell_poly(c, h3) for c in cells], crs=4326)
    p3 = p.to_crs(3857).copy()
    p3["parcel_area"] = p3.geometry.area
    pieces = gpd.overlay(p3, hexes.to_crs(3857), how="intersection", keep_geom_type=True)
    if pieces.empty:
        return None
    pieces["piece_area"] = pieces.geometry.area
    return pd.DataFrame(pieces.drop(columns="geometry"))


def _h3_pieces_binning(p, resolution, attr_cols, h3):
    """Large/county inputs: centroid + polyfill BINNING (no GEOS overlay — the overlay builds
    millions of bbox hex polygons and crashes at county scale). Each parcel goes to its
    centroid hex (piece = min(parcel_area, hex_area)); parcels much larger than a hex are
    ALSO polyfilled into the interior hexes they cover (piece = hex_area each), so big tracts
    still tile their hexes — reproducing the overlay's coverage and the same area-weighted
    math. Cost scales with parcel count, not bbox area."""
    p3 = p.to_crs(3857)
    parcel_area = p3.geometry.area.to_numpy()
    rep = p.geometry.representative_point()
    lat = rep.y.to_numpy(); lng = rep.x.to_numpy()
    # hex area in the SAME (3857) units as parcel_area, sampled at the data's latitude.
    cy, cx = float(np.nanmedian(lat)), float(np.nanmedian(lng))
    sample = _h3_cell_poly(h3.latlng_to_cell(cy, cx, resolution), h3)
    hex_area = float(gpd.GeoSeries([sample], crs=4326).to_crs(3857).area.iloc[0])

    cc = [h3.latlng_to_cell(float(la), float(lo), resolution) for la, lo in zip(lat, lng)]
    base = pd.DataFrame({c: p[c].to_numpy() for c in attr_cols})
    base["h3"] = cc
    base["parcel_area"] = parcel_area
    base["piece_area"] = np.minimum(parcel_area, hex_area)
    frames = [base]

    # Parcels meaningfully larger than a hex tile multiple hexes — polyfill their interiors so
    # the surface doesn't gap inside big tracts. Threshold keeps the per-parcel loop bounded.
    big_idx = np.where(parcel_area > hex_area * 3.0)[0]
    if len(big_idx):
        geoms = p.geometry.to_numpy()
        ext_i, ext_c = [], []
        for i in big_idx:
            geom = geoms[i]
            parts = ([geom] if geom.geom_type == "Polygon"
                     else list(geom.geoms) if geom.geom_type == "MultiPolygon" else [])
            cset = set()
            for poly in parts:
                try:
                    ring = [(yy, xx) for xx, yy in poly.exterior.coords]
                    holes = [[(yy, xx) for xx, yy in r.coords] for r in poly.interiors]
                    cset.update(h3.h3shape_to_cells(h3.LatLngPoly(ring, *holes), resolution))
                except Exception:
                    pass
            cset.discard(cc[i])  # centroid cell already counted in `base`
            for c in cset:
                ext_i.append(i); ext_c.append(c)
        if ext_i:
            extra = pd.DataFrame({c: p[c].to_numpy()[ext_i] for c in attr_cols})
            extra["h3"] = ext_c
            extra["parcel_area"] = parcel_area[ext_i]
            extra["piece_area"] = hex_area
            frames.append(extra)
    return pd.concat(frames, ignore_index=True)


def build_h3_aggregate(gdf: gpd.GeoDataFrame, resolution: int) -> gpd.GeoDataFrame:
    """Aggregate parcels into H3 hexagons at the given resolution.

    Per hex: rate fields (per-sqft, ratios) -> area-weighted mean; total/$ fields ->
    proportional area split; underutilized_pct -> share of covered area; categorical
    (jurisdiction) -> dominant value by area. Only hexes that receive parcel area are emitted.

    Two equivalent code paths produce the same per-(parcel,hex) `pieces`:
      - small inputs: EXACT areal overlay against a bbox hex grid.
      - large/county inputs (>600k parcels): centroid + polyfill BINNING — the overlay builds
        millions of bbox hex polygons and crashes GEOS at county extent, so we bin parcels by
        cell instead (cost scales with parcels, not bbox area), keeping the same aggregation.
    """
    import h3

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    g = gdf.to_crs(4326)

    rate_fields = [c for c in H3_RATE_FIELDS if c in g.columns]
    total_fields = [c for c in H3_TOTAL_FIELDS if c in g.columns]
    cat_fields = [c for c in H3_CATEGORICAL_FIELDS if c in g.columns]
    uu_field = H3_CATEGORY_FIELD if H3_CATEGORY_FIELD in g.columns else None
    uu_lv_field = "current_full_land_value" if "current_full_land_value" in g.columns else None
    attr_cols = list(dict.fromkeys(
        rate_fields + total_fields + cat_fields
        + ([uu_field] if uu_field else []) + ([uu_lv_field] if uu_lv_field else [])))

    p = g[["geometry"] + attr_cols].copy()
    p["geometry"] = p.geometry.buffer(0)  # fix self-intersections
    p = p[p.geometry.notna() & ~p.geometry.is_empty].reset_index(drop=True)
    if p.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    import os
    binning_min = int(os.environ.get("H3_BINNING_MIN", "600000"))
    if len(p) > binning_min:
        pieces = _h3_pieces_binning(p, resolution, attr_cols, h3)
    else:
        pieces = _h3_pieces_overlay(p, resolution, h3)
    if pieces is None or len(pieces) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    # ---- shared per-hex aggregation (identical for both paths) ----
    for f in rate_fields:
        pieces["_n_" + f] = pieces[f].astype(float).fillna(0) * pieces["piece_area"]
    for f in total_fields:
        pieces["_t_" + f] = (pieces[f].astype(float).fillna(0)
                             * pieces["piece_area"] / pieces["parcel_area"].replace(0, np.nan))
    if uu_field:
        uu_mask = pieces[uu_field].isin(H3_UNDERUTILIZED)
        pieces["_uu_area"] = pieces["piece_area"].where(uu_mask, 0.0)
        if uu_lv_field:
            pieces["_uu_lv"] = (pieces[uu_lv_field].astype(float).fillna(0)
                                * pieces["piece_area"] / pieces["parcel_area"].replace(0, np.nan)
                                ).where(uu_mask, 0.0)
    grp = pieces.groupby("h3", sort=False)
    area_sum = grp["piece_area"].sum()
    out = pd.DataFrame({"feature_count": grp.size().astype(int)})
    for f in rate_fields:
        out[f] = (grp["_n_" + f].sum() / area_sum).astype(float)
    for f in total_fields:
        out[f] = grp["_t_" + f].sum().astype(float)
    if uu_field:
        out["underutilized_pct"] = (100.0 * grp["_uu_area"].sum() / area_sum).astype(float)
        if uu_lv_field:
            out["underutilized_land_value"] = grp["_uu_lv"].sum().astype(float)
    out = out.reset_index()

    # Drop negligible-coverage hexes. An edge-clip piece can leave a hex with a covered
    # `land_area_acres` so tiny that write_geojson's 2-decimal rounding zeroes it — and the
    # frontend's per-sqft (value / (land_area_acres * 43560)) then divides by ~0, rendering the
    # hex as an infinite-height tower that's invisible at parcel zoom (real parcels have real
    # areas). Unrounded these hexes are harmless (both value and area scale with the covered
    # fraction, so $/sqft equals the covering parcel's own rate) — the artifact is purely the
    # rounding zeroing the denominator. <0.005 ac (~218 sqft) of covered land is never a whole
    # parcel (sub-500-sqft remnants are already dropped upstream), so these are spurious slivers
    # of a parcel's edge — drop them so the rounded denominator can never be 0.
    if "land_area_acres" in out.columns:
        _la = pd.to_numeric(out["land_area_acres"], errors="coerce").round(2)
        out = out[_la > 0].reset_index(drop=True)

    # Dominant categorical (e.g. jurisdiction) per hex = the value covering the most area.
    for cf in cat_fields:
        dom = (pieces.groupby(["h3", cf])["piece_area"].sum().reset_index()
               .sort_values(["h3", "piece_area"]).drop_duplicates("h3", keep="last")
               .set_index("h3")[cf])
        out[cf] = out["h3"].map(dom)

    # Emit full hexagons only for occupied cells.
    out["geometry"] = [_h3_cell_poly(c, h3) for c in out["h3"]]
    return gpd.GeoDataFrame(out.drop(columns=["h3"]), geometry="geometry", crs=4326)


def check_binary(binary_name: str, binary_path: str, use_wsl: bool = False) -> None:
    """Check if a binary exists and is executable."""
    if use_wsl:
        if shutil.which("wsl") is None:
            raise FileNotFoundError(
                "WSL (Windows Subsystem for Linux) not found. "
                "Install WSL2 from https://learn.microsoft.com/windows/wsl/install "
                f"then install {binary_name} inside WSL with:\n"
                f"  wsl -- sudo apt-get install -y {binary_name}"
            )
        # Verify the binary is available inside WSL.
        # Use `bash -lc "command -v ..."` (login shell) so the full PATH is
        # searched — a non-login bash -c misses /usr/local/bin etc.
        probe = subprocess.run(
            ["wsl", "--", "bash", "-lc", f"command -v {binary_name}"],
            capture_output=True, text=True,
        )
        print(f"  [diag] WSL probe for {binary_name}: rc={probe.returncode}", flush=True)
        print(f"  [diag] stdout: {probe.stdout.strip()!r}", flush=True)
        print(f"  [diag] stderr: {probe.stderr.strip()!r}", flush=True)
        if probe.returncode != 0:
            raise FileNotFoundError(
                f"{binary_name} not found in WSL. Install it with:\n"
                f"  python data/scripts/install_tippecanoe.py\n"
                f"or manually:\n"
                f"  wsl -- sudo apt-get install -y {binary_name}"
            )
        return

    if shutil.which(binary_path) is None:
        msg = (
            f"{binary_name} not found in PATH. "
            f"Install it with: brew install {binary_name} (macOS) or "
            f"see https://github.com/felt/tippecanoe (tippecanoe) / "
            f"https://github.com/protomaps/PMTiles (pmtiles)"
        )
        if platform.system() == "Windows":
            msg += (
                "\n\nOn Windows, install WSL2 and tippecanoe/pmtiles inside it:\n"
                "  wsl -- sudo apt-get update && sudo apt-get install -y tippecanoe\n"
                "Then re-run with --wsl flag (the notebook does this automatically)."
            )
        raise FileNotFoundError(msg)


def create_mbtiles(
    geojson_path: Path,
    mbtiles_path: Path,
    tippecanoe_bin: str,
    layer_specs: list[tuple[str, Path]] | None = None,
    use_wsl: bool = False,
) -> None:
    """Create MBTiles using tippecanoe.

    Args:
        layer_specs: List of (layer_name, file_path) tuples
        use_wsl: If True, invoke tippecanoe via 'wsl -e tippecanoe' and
                 convert Windows paths to /mnt/<drive>/... form.
    """
    check_binary("tippecanoe", tippecanoe_bin, use_wsl)

    def p(path: Path) -> str:
        return windows_path_to_wsl(path) if use_wsl else str(path)

    prefix = ["wsl", "-e", "tippecanoe"] if use_wsl else [tippecanoe_bin]
    print(f"Creating MBTiles: {mbtiles_path}")
    cmd = prefix + [
        "-o", p(mbtiles_path),
        "-z", "14",
        "-Z", "0",
        "--coalesce-densest-as-needed",
        "--extend-zooms-if-still-dropping",
    ]
    if layer_specs:
        for layer_name, file_path in layer_specs:
            cmd.extend(["-L", f"{layer_name}:{p(file_path)}"])
    else:
        cmd.extend(["-l", "parcels", p(geojson_path)])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"tippecanoe failed: {result.stderr}")
    print(f"MBTiles created: {mbtiles_path}")


def build_pmtiles_via_wsl_native(
    layer_specs: list[tuple[str, Path]],
    pmtiles_path: Path,
    tippecanoe_bin: str = "tippecanoe",
    pmtiles_bin: str = "pmtiles",
    densest_flags: str = "--coalesce-densest-as-needed --extend-zooms-if-still-dropping",
) -> None:
    """Run tippecanoe + pmtiles entirely on the WSL-native ext4 filesystem.

    The input GeoJSON, tippecanoe scratch (-t), and output MBTiles all live on
    native ext4; only the final PMTiles is copied back to /mnt/c. tippecanoe does
    heavy random access on its input + SQLite output, and over the /mnt/c 9p bridge
    that is ~40x slower (measured: ~1h vs ~90s for Houston's 606k parcels). The
    one-time sequential `cp` of the inputs into ext4 is cheap by comparison.
    """
    check_binary("tippecanoe", tippecanoe_bin, use_wsl=True)
    check_binary("pmtiles", pmtiles_bin, use_wsl=True)
    cps, lflags = [], []
    for i, (name, path) in enumerate(layer_specs):
        src = windows_path_to_wsl(path)
        dst = f"$T/in{i}.geojson"
        cps.append(f'cp "{src}" "{dst}"')
        lflags.append(f'-L {name}:"{dst}"')
    out_pm = windows_path_to_wsl(pmtiles_path)
    script = (
        "set -e; T=$(mktemp -d); "
        + "; ".join(cps) + "; "
        + f"tippecanoe -o \"$T/out.mbtiles\" -t \"$T\" -z 14 -Z 0 "
        + f"{densest_flags} {' '.join(lflags)}; "
        + "pmtiles convert \"$T/out.mbtiles\" \"$T/out.pmtiles\"; "
        + f"cp \"$T/out.pmtiles\" \"{out_pm}\"; rm -rf \"$T\""
    )
    print("Creating PMTiles via WSL (native ext4 staging: inputs + scratch + output)")
    print(f"Running: wsl -e bash -lc <copy inputs native, tippecanoe+pmtiles, out={out_pm}>")
    result = subprocess.run(["wsl", "-e", "bash", "-lc", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"WSL native PMTiles build failed: {result.stderr}")
    print(f"PMTiles created: {pmtiles_path}")


def build_pmtiles_h3_wsl_native(
    parcels_geojson: Path,
    hex_geojson: Path,
    pmtiles_path: Path,
    hex_maxzoom: int,
    parcel_minzoom: int,
    tippecanoe_bin: str = "tippecanoe",
    pmtiles_bin: str = "pmtiles",
    under_geojson: Path | None = None,
) -> None:
    """H3 bake on WSL-native ext4, joined with tile-join. Up to three tippecanoe runs:
      - hexes:   z0..hex_maxzoom, the H3 summary (layer `parcels_low`). Carries an
                 `underutilized_pct` field for the 3D land-value tab's low-zoom hexes.
      - parcels: parcel_minzoom..14, full detail, --no-tile-size-limit (NOTHING
                 dropped — drop-densest deleted ~98% of parcels; coalescing 600k
                 parcels into low-zoom tiles stalls tippecanoe, so the FULL parcel
                 layer is only tiled at the handoff zoom and the hexes own below).
      - parcels_under (optional): z0..14, the underutilized subset only (Vacant /
                 Parking / Underdeveloped — tens of thousands, not the full ~300k).
                 --drop-densest-as-needed thins it at low zoom and keeps full detail
                 high, so the "Vacant & Underdeveloped" tab renders at EVERY zoom.
    `tile-join -pk` merges without re-imposing the 500KB cap. Everything runs on
    native ext4; only the final PMTiles is copied back to /mnt/c.
    """
    check_binary("tippecanoe", tippecanoe_bin, use_wsl=True)
    check_binary("pmtiles", pmtiles_bin, use_wsl=True)
    p = windows_path_to_wsl(parcels_geojson)
    h = windows_path_to_wsl(hex_geojson)
    out = windows_path_to_wsl(pmtiles_path)
    under_run = ""
    under_join = ""
    if under_geojson is not None:
        u = windows_path_to_wsl(under_geojson)
        under_run = (
            f'cp "{u}" "$T/u.geojson"; '
            f'tippecanoe -o "$T/under.mbtiles" -t "$T" -Z0 -z14 '
            f'--drop-densest-as-needed -l parcels_under "$T/u.geojson"; '
        )
        under_join = '"$T/under.mbtiles" '
    script = (
        "set -e; T=$(mktemp -d); "
        f'cp "{p}" "$T/p.geojson"; cp "{h}" "$T/h.geojson"; '
        f'tippecanoe -o "$T/hex.mbtiles" -t "$T" -Z0 -z{hex_maxzoom} '
        f'--no-tile-size-limit --no-feature-limit -L parcels_low:"$T/h.geojson"; '
        f'tippecanoe -o "$T/parc.mbtiles" -t "$T" -Z{parcel_minzoom} -z14 '
        f'--no-tile-size-limit --no-feature-limit -l parcels "$T/p.geojson"; '
        f'{under_run}'
        f'tile-join -f -pk -o "$T/c.mbtiles" "$T/hex.mbtiles" "$T/parc.mbtiles" {under_join}; '
        f'pmtiles convert "$T/c.mbtiles" "$T/c.pmtiles"; '
        f'cp "$T/c.pmtiles" "{out}"; rm -rf "$T"'
    )
    print(f"Creating PMTiles via WSL (hexes z0-{hex_maxzoom} + full parcels z{parcel_minzoom}-14"
          f"{' + parcels_under z0-14' if under_geojson is not None else ''}, tile-join)")
    result = subprocess.run(["wsl", "-e", "bash", "-lc", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"WSL H3 PMTiles build failed: {result.stderr[-2000:]}")
    print(f"PMTiles created: {pmtiles_path}")


def build_pmtiles_h3_native(
    parcels_geojson: Path,
    hex_geojson: Path,
    pmtiles_path: Path,
    hex_maxzoom: int,
    parcel_minzoom: int,
    tippecanoe_bin: str = "tippecanoe",
    pmtiles_bin: str = "pmtiles",
    tile_join_bin: str = "tile-join",
    under_geojson: Path | None = None,
) -> None:
    """macOS/Linux H3 bake — the SAME runs as build_pmtiles_h3_wsl_native but with
    binaries invoked directly (no /mnt/c bridge, so no ext4 staging/copy-back).
    Includes the optional all-zoom `parcels_under` subset layer when provided.

    NOTE: mirrors the proven WSL command sequence exactly (identical flags/order),
    but has not been run on a Mac yet. Requires tippecanoe + tile-join + pmtiles on
    PATH (macOS: `brew install tippecanoe pmtiles`).
    """
    for b in (tippecanoe_bin, tile_join_bin, pmtiles_bin):
        if shutil.which(b) is None:
            raise FileNotFoundError(
                f"{b} not found in PATH. Install on macOS with: brew install tippecanoe pmtiles"
            )
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        hexm, parc, cmb = t / "hex.mbtiles", t / "parc.mbtiles", t / "c.mbtiles"
        runs = [
            [tippecanoe_bin, "-o", str(hexm), "-t", td, "-Z0", f"-z{hex_maxzoom}",
             "--no-tile-size-limit", "--no-feature-limit", "-L", f"parcels_low:{hex_geojson}"],
            [tippecanoe_bin, "-o", str(parc), "-t", td, f"-Z{parcel_minzoom}", "-z14",
             "--no-tile-size-limit", "--no-feature-limit", "-l", "parcels", str(parcels_geojson)],
        ]
        join_inputs = [str(hexm), str(parc)]
        if under_geojson is not None:
            underm = t / "under.mbtiles"
            runs.append(
                [tippecanoe_bin, "-o", str(underm), "-t", td, "-Z0", "-z14",
                 "--drop-densest-as-needed", "-l", "parcels_under", str(under_geojson)])
            join_inputs.append(str(underm))
        runs.append([tile_join_bin, "-f", "-pk", "-o", str(cmb), *join_inputs])
        runs.append([pmtiles_bin, "convert", str(cmb), str(pmtiles_path)])
        print(f"Creating PMTiles natively (hexes z0-{hex_maxzoom} + parcels z{parcel_minzoom}-14"
              f"{' + parcels_under z0-14' if under_geojson is not None else ''}, tile-join)")
        for cmd in runs:
            print("Running:", " ".join(cmd))
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-2000:]}")
    print(f"PMTiles created: {pmtiles_path}")


def convert_mbtiles_to_pmtiles(
    mbtiles_path: Path, pmtiles_path: Path, pmtiles_bin: str, use_wsl: bool = False
) -> None:
    """Convert MBTiles to PMTiles."""
    check_binary("pmtiles", pmtiles_bin, use_wsl)

    def p(path: Path) -> str:
        return windows_path_to_wsl(path) if use_wsl else str(path)

    prefix = ["wsl", "-e", "pmtiles"] if use_wsl else [pmtiles_bin]
    print(f"Converting MBTiles to PMTiles: {pmtiles_path}")
    cmd = prefix + ["convert", p(mbtiles_path), p(pmtiles_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pmtiles convert failed: {result.stderr}")
    print(f"PMTiles created: {pmtiles_path}")


def upload_file(
    container_client, local_path: Path, blob_name: str, overwrite: bool
) -> None:
    """Upload file to Azure Blob Storage."""
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")
    blob_client = container_client.get_blob_client(blob_name)
    with local_path.open("rb") as handle:
        blob_client.upload_blob(handle, overwrite=overwrite)
    print(f"✅ Uploaded {local_path.name} -> {container_client.container_name}/{blob_name}")


def main() -> None:
    args = parse_args()

    if not args.city and not args.file:
        cities = ", ".join(list_cities())
        raise SystemExit(f"Provide --city <name> or --file <path>. Available cities: {cities}")

    # Resolve input path
    parquet_path = resolve_local_path(args.city, args.file)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # Determine output names. resolve_city() raises for cities outside parquet_registry (e.g. the
    # dev-only 'harris'); when an explicit --file was given we don't need the registry — derive the
    # output names from the parquet stem instead (matches the no---city branch).
    city_meta = None
    if args.city:
        try:
            city_meta = resolve_city(args.city)
        except ValueError:
            if not args.file:
                raise
    if city_meta is not None:
        pmtiles_filename = f"{city_meta.city}-{city_meta.state}-parcels.pmtiles"
        metadata_filename = f"{city_meta.city}-{city_meta.state}-parcels-metadata.json"
    else:
        base = parquet_path.stem  # e.g. 'harris-tx-parcels' -> harris-tx-parcels.pmtiles
        pmtiles_filename = f"{base}.pmtiles"
        metadata_filename = f"{base}-metadata.json"

    # Load parquet and compute metadata
    print("=" * 60)
    print("Step 1: Loading parquet and computing metadata")
    print("=" * 60)
    gdf = gpd.read_parquet(parquet_path)
    print(f"Loaded {len(gdf):,} features")

    # Drop sub-500-sqft sliver remnants before building BOTH layers + metadata, so the
    # low-zoom hex aggregate matches the frontend-filtered detail layer. A remnant's
    # per-sqft (a real account value on a fragment polygon) can dominate a small/sparse
    # hex's area-weighted value and render as a spike.
    if args.drop_remnants and "likely_remnant" in gdf.columns:
        before = len(gdf)
        gdf = gdf[gdf["likely_remnant"].fillna(0).astype(int) != 1].copy()
        print(f"Dropped {before - len(gdf):,} likely_remnant slivers -> {len(gdf):,} features")

    # Ensure land_area_acres exists — it's the per-sqft DENOMINATOR. The frontend computes per-sqft
    # heights as value/(land_area_acres*43560) on BOTH the parcel layer AND the summed low-zoom hexes
    # (land_area_acres is in H3_TOTAL_FIELDS), so a missing/empty column makes the denominator 0 and
    # every feature renders FLAT. Older city ETLs never emitted it; derive it here from the projected
    # geometry (equal-area EPSG:6933, m²/4046.8564 = acres), matching the geometry-area approach used
    # in augment_houston_metadata_region_totals.py. Cities that already carry real values (Houston,
    # the TX cities) are left untouched.
    _la = pd.to_numeric(gdf["land_area_acres"], errors="coerce") if "land_area_acres" in gdf.columns else None
    if _la is None or _la.fillna(0).le(0).all():
        src = gdf if gdf.crs is not None else gdf.set_crs(4326)
        gdf["land_area_acres"] = (src.geometry.to_crs(6933).area / 4046.8564224).to_numpy()
        med = float(pd.to_numeric(gdf["land_area_acres"], errors="coerce").median())
        print(f"Derived land_area_acres from geometry (EPSG:6933) for {len(gdf):,} features "
              f"(median {med:.3f} ac) — source parquet lacked it")

    # Determine category fields (city-specific)
    dev_category_field = "property_land_use_refined"
    orig_category_field = "property_land_use_category"
    if args.city and args.city == "southbend":
        dev_category_field = "property_category_refined"
        orig_category_field = "PROPERTY_CATEGORY"

    # Ensure required fields exist
    if "REALIMPROV" not in gdf.columns or "REALLANDVA" not in gdf.columns:
        # Try to compute from alternative field names
        if "improvement_value" in gdf.columns:
            gdf["REALIMPROV"] = gdf["improvement_value"]
        if "current_full_land_value" in gdf.columns:
            gdf["REALLANDVA"] = gdf["current_full_land_value"]
        elif "land_value" in gdf.columns:
            gdf["REALLANDVA"] = gdf["land_value"]

    # Add derived fields if missing
    gdf = add_improvement_ratio_fields(
        gdf, land_col="REALLANDVA", improvement_col="REALIMPROV"
    )

    # Underutilized subset (the refined dev category is set: Vacant / Parking Lot /
    # Underdeveloped). The "Vacant & Underdeveloped" tab ONLY ever renders these, so we
    # give them their own `parcels_under` layer tiled across ALL zooms — it's a small
    # fraction of the parcels (tens of thousands, not the full ~300k), so it tiles
    # low-zoom cleanly where the full parcel layer would stall / drop ~98%. This is what
    # lets that tab render when zoomed out instead of being blank below parcelMinZoom.
    under_gdf = None
    under_layer_name = None
    if args.h3 and dev_category_field in gdf.columns:
        _refined = gdf[dev_category_field].astype(str).str.strip()
        under_mask = gdf[dev_category_field].notna() & _refined.ne("") & _refined.str.lower().ne("nan")
        under_gdf = gdf[under_mask].copy()
        if len(under_gdf):
            under_layer_name = "parcels_under"
            print(f"Underutilized subset (all-zoom `parcels_under` layer): {len(under_gdf):,} parcels")
        else:
            under_gdf = None

    # Decide the H3 ladder + hex->parcel handoff BEFORE the metadata save: the handoff zoom
    # ships in the metadata JSON (uploaded together with the tiles), and the frontend prefers
    # it over the dictionary's static parcelMinZoom — so tiles and handoff can't drift apart.
    h3_plan = plan_h3_ladder(gdf) if args.h3 else None
    if h3_plan:
        kept = ",".join(f"r{r}" for r in h3_plan["res_list"])
        print(f"H3 ladder: {kept} (hexes z0-{h3_plan['hex_maxzoom']}, parcels z{h3_plan['parcel_minzoom']}+); "
              f"median parcel {h3_plan['median_parcel_m2']:,.0f} m²"
              + (f" — pruned r{h3_plan['median_pruned']} (finer than median parcel)"
                 if h3_plan["median_pruned"] else ""))

    # Compute metadata
    metadata = compute_metadata(gdf, dev_category_field, orig_category_field)
    print(f"Computed metadata: {len(metadata['statistics'])} fields, "
          f"{len(metadata['categories']['refined'])} refined categories")
    if h3_plan:
        metadata["parcelMinZoom"] = h3_plan["parcel_minzoom"]
    # Integer-encode the categorical region fields (shrinks tiles; see encode_categoricals).
    # In place on gdf so parcels AND the per-hex dominant value share one id<->name mapping.
    encode_categoricals(gdf, metadata)
    # Per-region $/acre totals for the Land Value blurb (must run after encode — uses the id maps).
    add_region_value_totals(gdf, metadata)
    # Tell the frontend the underutilized tab can use the all-zoom subset layer. Cities
    # baked before this existed lack the key, so the frontend falls back to `parcels`.
    if under_layer_name:
        metadata["underutilizedSourceLayer"] = under_layer_name

    # Save metadata JSON
    metadata_path = parquet_path.parent / metadata_filename
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {metadata_path}")

    # Check dependencies before starting conversion
    print("=" * 60)
    print("Step 2: Checking dependencies")
    print("=" * 60)
    use_wsl = args.wsl
    if platform.system() == "Windows" and not use_wsl:
        # Auto-suggest --wsl if tippecanoe not found natively on Windows
        print(f"  [diag] platform={platform.system()!r}, tippecanoe in PATH={shutil.which(args.tippecanoe)!r}, wsl in PATH={shutil.which('wsl')!r}", flush=True)
        if shutil.which(args.tippecanoe) is None and shutil.which("wsl") is not None:
            print("tippecanoe not found in PATH; WSL detected — switching to --wsl mode", flush=True)
            use_wsl = True
    print(f"  [diag] use_wsl={use_wsl}", flush=True)
    try:
        check_binary("tippecanoe", args.tippecanoe, use_wsl)
        check_binary("pmtiles", args.pmtiles, use_wsl)
        print("All dependencies found" + (" (via WSL)" if use_wsl else ""))
    except FileNotFoundError as e:
        print(f"[X] {e}")
        raise SystemExit(1)

    # Convert to PMTiles
    print("=" * 60)
    print("Step 3: Converting to PMTiles")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        geojson_path = tmp / "input.geojson"
        mbtiles_path = tmp / "output.mbtiles"
        pmtiles_path = parquet_path.parent / pmtiles_filename

        # Per-sqft rates are computed client-side from value + land_area_acres, so they're NOT
        # written to the parcel/under tiles (kept in `gdf` above only for compute_metadata's
        # stats/breaks + the hex Σ apportioning). Dropping these continuous floats is the win.
        DROP_FROM_TILES = ["land_value_per_sqft", "improvement_value_per_sqft", "full_market_value_per_sqft"]
        slim = lambda g: g.drop(columns=[c for c in DROP_FROM_TILES if c in g.columns])

        # Convert parquet to GeoJSON
        print("Writing base GeoJSON")
        write_geojson(slim(gdf), geojson_path)

        # Underutilized subset GeoJSON (all-zoom `parcels_under` layer).
        under_geojson_path = None
        if under_gdf is not None and len(under_gdf):
            under_geojson_path = tmp / "parcels_under.geojson"
            write_geojson(slim(under_gdf), under_geojson_path)

        # Build aggregate layers used before the full parcel layer is dense enough.
        print("Building aggregate layers")
        layer_specs: list[tuple[str, Path]] = [("parcels", geojson_path)]
        aggregate_count = 0
        if args.h3:
            # Merge all H3 resolutions into ONE 'parcels_low' layer, each feature
            # gated to its zoom band via a per-feature `tippecanoe` member, so only
            # the right resolution renders at each zoom. The viz keeps a single
            # low-zoom layer and hands off to real parcels at parcelMinZoom.
            from shapely.geometry import mapping
            hex_path = tmp / "parcels_low.geojson"
            feats: list[dict] = []
            # Ladder + handoff decided by plan_h3_ladder() (count caps for county-scale perf,
            # median-parcel prune for fidelity — see its docstring). The last surviving res
            # stretches its band to the top of the hex zoom range (the `res == res_list[-1]`
            # block below), so coverage stays full up to the parcel handoff.
            res_list = h3_plan["res_list"]
            overall_max_z = h3_plan["hex_maxzoom"]
            for res in res_list:
                mn, mx = H3_ZOOM_BANDS[res]
                if res == res_list[-1]:
                    mx = overall_max_z
                hex_gdf = build_h3_aggregate(gdf, resolution=res)
                if hex_gdf.empty:
                    print(f"⚠️ H3 r{res} empty; skipping")
                    continue
                for _, row in hex_gdf.iterrows():
                    # Round floats to 2 dp — same tippecanoe value-pool overflow as the
                    # parcel layer (see write_geojson); the hex layer is written by hand
                    # here so it needs the same treatment or hexes show false mega-spikes.
                    props = {k: (None if (isinstance(v, float) and pd.isna(v))
                                 else round(v, 2) if isinstance(v, float) else v)
                             for k, v in row.items() if k != "geometry"}
                    feats.append({
                        "type": "Feature",
                        "tippecanoe": {"minzoom": mn, "maxzoom": mx},
                        "properties": props,
                        "geometry": mapping(row.geometry),
                    })
                print(f"  parcels_low r{res}: {len(hex_gdf):,} hexes (z{mn}-{mx})")
            if feats:
                with open(hex_path, "w") as f:
                    json.dump({"type": "FeatureCollection", "features": feats}, f)
                layer_specs.append(("parcels_low", hex_path))
                aggregate_count = 1
        else:
            for layer_name, cell_size in get_aggregate_layer_specs(args.city):
                aggregate_gdf = build_low_zoom_aggregate(gdf, cell_size_m=cell_size)
                if aggregate_gdf.empty:
                    print(f"⚠️ Aggregate layer {layer_name} empty; skipping")
                    continue
                aggregate_geojson_path = tmp / f"{layer_name}.geojson"
                write_geojson(aggregate_gdf, aggregate_geojson_path)
                layer_specs.append((layer_name, aggregate_geojson_path))
                aggregate_count += 1
        if aggregate_count:
            print(f"✅ Created {aggregate_count} aggregate layer(s)")
        else:
            print("⚠️ Aggregate layers empty; continuing with parcels only")

        if args.h3 and aggregate_count:
            # H3: hexes (z0..hex_maxzoom, incl. underutilized_pct) + full parcels
            # (parcel_minzoom..14), tile-join'd. WSL stages on native ext4 (avoids the
            # slow /mnt/c bridge); macOS/Linux run the binaries directly.
            hex_maxzoom = h3_plan["hex_maxzoom"]
            parcel_minzoom = h3_plan["parcel_minzoom"]
            if use_wsl:
                build_pmtiles_h3_wsl_native(
                    geojson_path, hex_path, pmtiles_path,
                    hex_maxzoom, parcel_minzoom, args.tippecanoe, args.pmtiles,
                    under_geojson=under_geojson_path,
                )
            else:
                build_pmtiles_h3_native(
                    geojson_path, hex_path, pmtiles_path,
                    hex_maxzoom, parcel_minzoom, args.tippecanoe, args.pmtiles,
                    under_geojson=under_geojson_path,
                )
        elif use_wsl:
            # Non-H3 (square-grid aggregate): single run, coalesce low zooms.
            build_pmtiles_via_wsl_native(
                layer_specs, pmtiles_path, args.tippecanoe, args.pmtiles,
                "--coalesce-densest-as-needed --extend-zooms-if-still-dropping",
            )
        else:
            create_mbtiles(geojson_path, mbtiles_path, args.tippecanoe, layer_specs=layer_specs, use_wsl=False)
            convert_mbtiles_to_pmtiles(mbtiles_path, pmtiles_path, args.pmtiles, use_wsl=False)

    print(f"✅ PMTiles created: {pmtiles_path}")

    # Upload if requested
    if args.upload:
        if not args.connection_string:
            raise SystemExit(
                "Missing connection string. Set AZURE_STORAGE_CONNECTION_STRING or pass --connection-string."
            )

        print("=" * 60)
        print("Step 4: Uploading to Azure Blob Storage")
        print("=" * 60)

        # Tune for large PMTiles: chunk into 4 MB blocks with long timeouts + retries, so a slow
        # link doesn't time out on a single-shot PUT of a 100+ MB file (matches upload_city_dev.py).
        _MB = 1024 * 1024
        blob_service = BlobServiceClient.from_connection_string(
            args.connection_string,
            max_single_put_size=4 * _MB,
            max_block_size=4 * _MB,
            connection_timeout=300,
            read_timeout=600,
            retry_total=8,
        )
        container_client = blob_service.get_container_client(args.container)

        upload_file(container_client, pmtiles_path, pmtiles_filename, args.overwrite)
        upload_file(container_client, metadata_path, metadata_filename, args.overwrite)

        print("=" * 60)
        print("✅ Conversion complete!")
        print(f"PMTiles: {pmtiles_filename}")
        print(f"Metadata: {metadata_filename}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("✅ Conversion complete!")
        print(f"PMTiles: {pmtiles_path}")
        print(f"Metadata: {metadata_path}")
        print("=" * 60)
        print("Use --upload to upload to Azure Blob Storage")


if __name__ == "__main__":
    main()
