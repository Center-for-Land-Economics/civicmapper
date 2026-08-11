"""parking_lot_extraction.py

End-to-end ETL pipeline for surface parking lot detection.

Pipeline:
  1. Load existing city parcel GeoParquet (read-only)
  2. Derive city bounding box from parcel footprint
  3. Fetch NAIP aerial imagery via Microsoft Planetary Computer STAC API
  4. Tile imagery and run UIUC SegFormer-large-parking model inference
  5. Vectorize pixel masks to GeoJSON polygons with post-processing
  6. Spatially join parking polygons to parcels, compute land value metrics
  7. Write isolated output to data/parking/{city}/ (never modifies parcel files)
  8. Optionally upload to Azure Blob Storage at parking/ subfolder

Usage:
    python data/scripts/parking_lot_extraction.py --city southbend
    python data/scripts/parking_lot_extraction.py --city southbend --upload
    python data/scripts/parking_lot_extraction.py --city southbend --file /path/to/parcels.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from urllib.parse import urlsplit

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.ops import unary_union

# Add project root to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.parquet_registry import CITY_PARQUETS, list_cities, resolve_city

# Suppress noisy deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# OSM place query strings per city (used only if --use-osm-supplement flag is set)
# Primary pipeline uses ML segmentation on NAIP imagery
# ---------------------------------------------------------------------------
CITY_OSM_QUERIES: dict[str, str] = {
    "southbend": "South Bend, Indiana, USA",
    "syracuse": "Syracuse, New York, USA",
    "spokane": "Spokane, Washington, USA",
    "rochester": "Rochester, New York, USA",
    "bellingham": "Bellingham, Washington, USA",
    "morgantown": "Morgantown, West Virginia, USA",
    "denver": "Denver, Colorado, USA",
    "cincinnati": "Cincinnati, Ohio, USA",
    "ibx": "New York City, New York, USA",
    "stpaul": "Saint Paul, Minnesota, USA",
    "nyc": "New York City, New York, USA",
    "baltimore": "Baltimore, Maryland, USA",
    "albuquerque": "City of Albuquerque, New Mexico, USA",
    "fortcollins": "Fort Collins, Colorado, USA",
    "cleveland": "Cleveland, Ohio, USA",
    "columbus": "Columbus, Ohio, USA",
    "charlottesville": "Charlottesville, Virginia, USA",
    "pueblo": "Pueblo, Colorado, USA",
    "portland": "Portland, Oregon, USA",
    "houston": "Houston, Texas, USA",
    "austin": "Austin, Texas, USA",
    "dallas": "Dallas, Texas, USA",
    "sanantonio": "San Antonio, Texas, USA",
    # Bryan + College Station as one unit. The OSM fetch + clip both use the
    # parcel-footprint bbox/geometry (which already spans both cities), so this
    # place string is only the required gate value, not the fetch extent.
    "bcs": "College Station, Texas, USA",
    "rockville": "Rockville, Maryland, USA",
    "detroit": "Detroit, Michigan, USA",
    "chicago": "Chicago, Illinois, USA",
    "tulsa": "Tulsa, Oklahoma, USA",
    "newportnews": "Newport News, Virginia, USA",
    "olympia": "Olympia, Washington, USA",
    "seattle": "Seattle, Washington, USA",
    "vancouver": "Vancouver, Washington, USA",
    "washington": "Washington, District of Columbia, USA",
    "tallinn": "Tallinn, Estonia",
    "copenhagen": "Copenhagen, Denmark",
}

PORTLAND_BOUNDARY_QUERY_URL = "https://www3.multco.us/gisagspublic/rest/services/DART/LevyCode/MapServer/4/query"
PORTLAND_CITY_NAME = "CITY OF PORTLAND"

# Minimum parking polygon area to keep (sqft). Filters out model noise.
MIN_PARKING_AREA_SQFT = 500  # ~1 standard parking space = ~162 sqft; 500 sqft ≈ 3 spaces
# Maximum plausible surface parking lot area (acres). Features larger than this
# are false positives — airport tarmac, distribution centres, etc.  The largest
# real-world surface parking lots (mega-mall / stadium) top out around 30–35 ac;
# dev-validated Spokane data caps at 37.6 ac.
MAX_PARKING_AREA_ACRES = 40
# Morphological opening disk radius in pixels.  At 0.6 m/px this is ~1.8 m
# physical radius — enough to break 1–2 px false connections between separate
# surfaces and remove isolated noise, without eroding real parking edges.
MORPH_OPEN_RADIUS_PX = 3
# Concurrent NAIP tile fetches. Tiles stream through rasterio warps that are
# network-bound; a county-scale fetch is ~12 h sequential vs ~2 h at 6 workers.
NAIP_FETCH_WORKERS = 6
# NDVI above this is treated as vegetation and stripped from the parking mask.
# The published parking model is RGB-only and over-predicts on grass and merges
# adjacent lots across grassy medians; the paper's NIR insight is that vegetation
# stands out in near-infrared. We capture that benefit without retraining by
# computing NDVI = (NIR - Red)/(NIR + Red) from the 4-band NAIP we already fetch
# and zeroing vegetated pixels before vectorizing. Pavement NDVI ≈ 0; grass ≳ 0.2.
NDVI_VEG_THRESHOLD = 0.20


def parse_naip_item_id(item_id: str) -> tuple[str, float]:
    """
    Split a NAIP STAC item ID into:
    - a stable footprint key (same physical quad across vintages/resolutions)
    - the native ground sample distance in meters when encoded in the ID

    Examples:
    - m_4108127_se_17_030_20230526_20231018 -> ("m_4108127_se_17", 0.30)
    - m_4108127_se_17_060_20210605          -> ("m_4108127_se_17", 0.60)
    """
    parts = item_id.split("_")

    # NAIP item IDs end with one or two YYYYMMDD date stamps. Remove them first.
    while parts and re.fullmatch(r"\d{8}", parts[-1]):
        parts.pop()

    native_resolution_m = float("inf")
    if parts and re.fullmatch(r"\d{3}", parts[-1]):
        native_resolution_m = int(parts.pop()) / 100.0

    footprint_key = "_".join(parts) if parts else item_id
    return footprint_key, native_resolution_m


def native_resolution_m_for_item(item: "pystac.Item") -> float:
    """
    Return the native NAIP resolution in meters, preferring explicit STAC metadata
    and falling back to the item ID convention.
    """
    gsd = item.properties.get("gsd") if getattr(item, "properties", None) else None
    try:
        if gsd is not None:
            return float(gsd)
    except (TypeError, ValueError):
        pass

    _, parsed_resolution = parse_naip_item_id(item.id)
    return parsed_resolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract surface parking lot geometries from NAIP aerial imagery "
            "using the UIUC SegFormer-large-parking model, then link to parcel land values."
        )
    )
    parser.add_argument(
        "--city",
        help="City key to process (e.g. southbend). Run with no args to see available cities.",
    )
    parser.add_argument(
        "--file",
        help="Override path to the city parcel GeoParquet (reads only, never modified).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for parking GeoParquet and metadata JSON. "
            "Defaults to data/parking/{city}/ relative to project root."
        ),
    )
    parser.add_argument(
        "--naip-start",
        default="2021-01-01",
        help="NAIP imagery search start date (YYYY-MM-DD). Default: 2021-01-01.",
    )
    parser.add_argument(
        "--naip-end",
        default="2023-12-31",
        help="NAIP imagery search end date (YYYY-MM-DD). Default: 2023-12-31.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.6,
        help="Imagery resolution in meters. Default: 0.6 (best NAIP vintage). Model trained at 0.3m.",
    )
    parser.add_argument(
        "--use-osm-supplement",
        action="store_true",
        help="Also fetch OSM parking polygons and merge as supplemental source.",
    )
    parser.add_argument(
        "--osm-only",
        action="store_true",
        help=(
            "Skip NAIP fetch and ML inference entirely. "
            "Use OpenStreetMap parking polygons as the sole source. "
            "Fast and reliable for cities with good OSM coverage."
        ),
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help=(
            "Skip ML inference step (useful if you already have mask GeoTIFF). "
            "Expects {output-dir}/mask.tif to exist."
        ),
    )
    parser.add_argument(
        "--connection-string",
        default=os.getenv("AZURE_STORAGE_CONNECTION_STRING", ""),
        help="Azure Storage connection string (AZURE_STORAGE_CONNECTION_STRING env var).",
    )
    parser.add_argument(
        "--container",
        default=os.getenv("AZURE_DEV_CONTAINER", "parquets-dev"),
        help="Target Azure Blob Storage container (AZURE_DEV_CONTAINER env var).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload output files to Azure Blob Storage at parking/ subfolder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing blobs when uploading.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available (slower but works anywhere).",
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="Skip the surface-vs-structure classification step (Overture buildings + "
             "OSM tags). Parking is written without parking_type/confidence columns.",
    )
    parser.add_argument(
        "--force-citywide-mosaic",
        action="store_true",
        help=(
            "Force the legacy workflow that mosaics NAIP into one citywide raster "
            "before inference. By default, large jobs use tile-first inference."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Parking probability threshold (0.0-1.0). Pixels with model confidence "
            "below this are classified as background. Higher values reduce false "
            "positives. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--rethreshold",
        action="store_true",
        help=(
            "Skip NAIP fetch and inference entirely. Re-read the saved probability "
            "map (parking_probs.tif), apply --threshold, morph opening, vectorize, "
            "and spatial-join. Runs in seconds instead of hours."
        ),
    )
    parser.add_argument(
        "--no-nir-strip",
        action="store_false",
        dest="nir_strip",
        help=(
            "Disable the NDVI vegetation strip. By default the 4-band NAIP NIR "
            "channel is used to remove vegetated pixels from the parking mask "
            "(reduces over-prediction on grass and lots merged across medians)."
        ),
    )
    parser.add_argument(
        "--no-remove-roads",
        action="store_false",
        dest="remove_roads",
        help=(
            "Disable OSM road-buffer removal. By default road centrelines are "
            "buffered by highway class and subtracted from detected parking "
            "(roads are the most common false positive)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Resolve local parcel file path
# ---------------------------------------------------------------------------

def resolve_parcel_path(city: str | None, override: str | None) -> Path:
    """Return path to the existing city parcel GeoParquet (read-only)."""
    if override:
        p = Path(override).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Parcel file not found: {p}")
        return p
    if city:
        city_meta = CITY_PARQUETS.get(city)
        if city_meta:
            # Try canonical location first
            canonical = (
                Path("data/jurisidictions/data") / city / city_meta.canonical_filename
            )
            if canonical.exists():
                return canonical
            # Fall back to legacy filename
            legacy = (
                Path("data/jurisidictions/data") / city / city_meta.legacy_filename
            )
            if legacy.exists():
                return legacy
        raise FileNotFoundError(
            f"Could not find parcel parquet for city '{city}'. "
            f"Pass --file to specify the path explicitly."
        )
    raise ValueError("Either --city or --file must be provided.")


# ---------------------------------------------------------------------------
# Step 2: Derive city bounding box from parcel footprint
# ---------------------------------------------------------------------------

def bbox_from_parcels(parcels: gpd.GeoDataFrame) -> list[float]:
    """Return [minx, miny, maxx, maxy] in EPSG:4326 from parcel footprint."""
    if parcels.crs is None:
        raise ValueError("Parcel GeoDataFrame has no CRS set.")
    if parcels.crs.to_epsg() != 4326:
        parcels = parcels.to_crs(4326)
    minx, miny, maxx, maxy = parcels.total_bounds
    print(f"  Bounding box (EPSG:4326): [{minx:.6f}, {miny:.6f}, {maxx:.6f}, {maxy:.6f}]")
    return [float(minx), float(miny), float(maxx), float(maxy)]


def fetch_portland_boundary_geometry():
    """Return the official Portland municipal boundary as a single geometry."""
    response = requests.get(
        PORTLAND_BOUNDARY_QUERY_URL,
        params={
            "f": "geojson",
            "where": f"Tax_Dist_Desc='{PORTLAND_CITY_NAME}'",
            "outFields": "Tax_Dist_Desc",
            "returnGeometry": "true",
            "outSR": 4326,
        },
        timeout=240,
    )
    response.raise_for_status()
    payload = response.json()
    features = payload.get("features", [])
    if not features:
        raise RuntimeError("Failed to fetch Portland municipal boundary.")
    boundary = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    return unary_union(boundary.geometry)


def city_boundary_geometry(city_key: str, parcels: gpd.GeoDataFrame):
    """
    Return a geometry that represents the city's valid parking extent.

    Portland uses an official municipal boundary because the parking ETL can
    start from a bbox-wide OSM query. Other cities fall back to the parcel
    footprint we already loaded for the city.
    """
    if city_key == "portland":
        return fetch_portland_boundary_geometry()

    if parcels.crs is None or parcels.crs.to_epsg() != 4326:
        parcels = parcels.to_crs(4326)
    # buffer(0) cleans up self-intersections / side-location conflicts that
    # otherwise crash unary_union (seen on Pueblo). Cheap and idempotent on
    # already-valid geometries.
    cleaned = parcels.geometry.buffer(0)
    return unary_union(cleaned)


def clip_parking_to_city_boundary(
    parking: gpd.GeoDataFrame,
    boundary_geometry,
) -> gpd.GeoDataFrame:
    """Keep only parking features whose representative point is within the city boundary."""
    if parking.empty:
        return parking

    if parking.crs is None or parking.crs.to_epsg() != 4326:
        parking = parking.to_crs(4326)

    representative_points = parking.representative_point()
    inside_mask = representative_points.within(boundary_geometry)
    dropped = int((~inside_mask).sum())
    if dropped:
        print(f"  Dropping {dropped:,} parking polygons outside the city boundary")
    return parking.loc[inside_mask].copy()


# ---------------------------------------------------------------------------
# Step 3: Fetch NAIP imagery via Microsoft Planetary Computer STAC
# ---------------------------------------------------------------------------

def fetch_naip_imagery(
    bbox: list[float],
    date_range: str,
    resolution_m: float,
    output_dir: Path,
) -> Path:
    """
    Fetch NAIP imagery for bbox via Microsoft Planetary Computer STAC API.

    Strategy: every network read goes through rasterio — each source COG is
    clipped/warped to a local per-tile GeoTIFF cache, then a purely LOCAL
    gdalbuildvrt + gdalwarp produce the unified citywide mosaic. The GDAL CLI
    never touches the network: the Windows osgeo-wheel CLI links a c-ares
    resolver that cannot resolve the NAIP blob host, while rasterio's curl
    stack works everywhere.

    Robustness:
    - per-tile clipping with SAS re-sign retry and local caching
    - isolates flaky source COGs so a single bad tile does not kill a whole city
    - cached tiles are reused across runs (extent-checked), so re-runs are cheap

    Returns path to the merged+reprojected GeoTIFF (RGB+NIR) clipped to bbox.
    """
    import subprocess
    try:
        import planetary_computer
        import pystac_client
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.shutil import copy as rio_copy
        from rasterio.transform import from_origin
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import transform_bounds
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Install with: pip install pystac-client planetary-computer rasterio"
        ) from e

    def gdal_env() -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        env.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
        env.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF")
        env.setdefault("GDAL_HTTP_MAX_RETRY", "4")
        env.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
        env.setdefault("GDAL_HTTP_RETRY_CODES", "429,500,502,503,504")
        # The osgeo-wheel gdalwarp/gdalbuildvrt need PROJ_LIB/GDAL_DATA to find
        # proj.db; point them at the wheel's own data dirs so runs don't depend
        # on ambient shell configuration.
        try:
            import osgeo
            osgeo_data = Path(osgeo.__file__).parent / "data"
            if (osgeo_data / "proj" / "proj.db").exists():
                env.setdefault("PROJ_LIB", str(osgeo_data / "proj"))
                env.setdefault("PROJ_DATA", str(osgeo_data / "proj"))
            if (osgeo_data / "gdal").is_dir():
                env.setdefault("GDAL_DATA", str(osgeo_data / "gdal"))
        except ImportError:
            pass
        return env

    def run_checked(cmd: list[str], *, label: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(cmd, capture_output=True, text=True, env=gdal_env())
        if result.returncode != 0:
            raise RuntimeError(f"{label} failed:\n{result.stderr}")
        return result

    def build_vrt_from_filelist(filelist_path: Path, vrt_path: Path) -> None:
        run_checked(
            [
                "gdalbuildvrt",
                "-input_file_list", str(filelist_path),
                "-allow_projection_difference",
                str(vrt_path),
            ],
            label="gdalbuildvrt",
        )
        print(f"  Built VRT: {vrt_path}", flush=True)

    def direct_warp(vrt_path: Path, naip_path: Path) -> None:
        run_checked(
            [
                "gdalwarp",
                "-te", str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]),
                "-te_srs", "EPSG:4326",
                "-t_srs", f"EPSG:{epsg_utm}",
                "-tr", str(resolution_m), str(resolution_m),
                "-r", "bilinear",
                "-multi",
                "-wo", "NUM_THREADS=ALL_CPUS",
                # Treat source zeros as nodata — avoids zero-fill gaps from per-tile
                # clips bleeding into the final mosaic and causing false-positive
                # parking detection in uncovered edge areas.
                "-srcnodata", "0",
                "-dstnodata", "0",
                "-of", "GTiff",
                "-co", "COMPRESS=LZW",
                # Tiled output is required for chunked inference: window reads
                # from a striped GTiff decompress entire raster rows (~GBs per
                # chunk at county scale), tiled reads only touch needed blocks.
                "-co", "TILED=YES",
                "-co", "BLOCKXSIZE=512",
                "-co", "BLOCKYSIZE=512",
                "-co", "BIGTIFF=IF_SAFER",
                "-overwrite",
                str(vrt_path), str(naip_path),
            ],
            label="gdalwarp",
        )

    def tile_key_from_url(url: str) -> str:
        return Path(urlsplit(url).path).name

    def bbox_intersection(a: list[float], b: list[float]) -> list[float] | None:
        minx = max(a[0], b[0])
        miny = max(a[1], b[1])
        maxx = min(a[2], b[2])
        maxy = min(a[3], b[3])
        if minx >= maxx or miny >= maxy:
            return None
        return [float(minx), float(miny), float(maxx), float(maxy)]

    def item_clip_bbox(item: "pystac.Item") -> list[float] | None:
        item_bounds = item.bbox
        if not item_bounds or len(item_bounds) != 4:
            return None
        return bbox_intersection(
            [float(item_bounds[0]), float(item_bounds[1]), float(item_bounds[2]), float(item_bounds[3])],
            bbox,
        )

    def cached_tile_matches_expected(out_path: Path, expected_bbox: list[float]) -> bool:
        # `gdalwarp` snaps the output grid in projected space, so the resulting
        # WGS84 bounds can drift by ~0.0015 degrees from the raw STAC bbox while
        # still representing the same local tile. Keep this loose enough to
        # accept valid local caches but far tighter than the old citywide rasters.
        tolerance_deg = 0.0025
        try:
            with rasterio.open(out_path) as src:
                cached_bounds = transform_bounds(
                    src.crs,
                    "EPSG:4326",
                    *src.bounds,
                    densify_pts=21,
                )
        except Exception:
            return False

        deltas = [
            abs(cached_bounds[0] - expected_bbox[0]),
            abs(cached_bounds[1] - expected_bbox[1]),
            abs(cached_bounds[2] - expected_bbox[2]),
            abs(cached_bounds[3] - expected_bbox[3]),
        ]
        return max(deltas) <= tolerance_deg

    def _reset_sas_cache() -> None:
        # planetary_computer caches SAS tokens in module state; after a 403,
        # clear whatever cache attribute this version exposes so the next
        # sign() must fetch a fresh token. Best-effort by design.
        sas_module = getattr(planetary_computer, "sas", None)
        for attr in ("TOKEN_CACHE", "_CACHE", "_cache"):
            cache = getattr(sas_module, attr, None)
            if hasattr(cache, "clear"):
                cache.clear()

    def clip_tiles_locally(
        original_items: list["pystac.Item"],
        output_dir: Path,
    ) -> Path:
        clipped_dir = output_dir / "naip_tiles"
        clipped_dir.mkdir(parents=True, exist_ok=True)
        local_paths: list[Path] = []
        failed_tiles: list[str] = []

        item_by_name = {
            tile_key_from_url(item.assets["image"].href): item
            for item in original_items
        }

        def attempt_clip(
            tile_name: str,
            tile_url: str,
            out_path: Path,
            clip_bbox: list[float],
        ) -> None:
            # Same warp gdalwarp used to do (-te/-te_srs/-t_srs/-tr/-r bilinear),
            # expressed as a rasterio WarpedVRT streamed to disk block-by-block.
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", f"EPSG:{epsg_utm}", *clip_bbox, densify_pts=21
            )
            width = max(1, int(np.ceil((right - left) / resolution_m)))
            height = max(1, int(np.ceil((top - bottom) / resolution_m)))
            dst_transform = from_origin(left, top, resolution_m, resolution_m)
            env_opts = {
                "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
                "GDAL_HTTP_MAX_RETRY": "4",
                "GDAL_HTTP_RETRY_DELAY": "2",
                "GDAL_HTTP_RETRY_CODES": "429,500,502,503,504",
            }
            try:
                with rasterio.Env(**env_opts):
                    with rasterio.open(tile_url) as src:
                        with WarpedVRT(
                            src,
                            crs=f"EPSG:{epsg_utm}",
                            transform=dst_transform,
                            width=width,
                            height=height,
                            resampling=Resampling.bilinear,
                        ) as warped:
                            rio_copy(
                                warped,
                                str(out_path),
                                driver="GTiff",
                                compress="LZW",
                                bigtiff="IF_SAFER",
                            )
            except Exception as e:
                out_path.unlink(missing_ok=True)
                raise RuntimeError(f"rasterio clip ({tile_name}) failed: {e}") from e

        def fetch_one(idx: int, tile_name: str, item) -> tuple[str, Path | None, bool]:
            """Fetch/reuse one tile. Returns (tile_name, path or None, hard_failure)."""
            clip_bbox = item_clip_bbox(item)
            if clip_bbox is None:
                print(
                    f"  Skipping tile {idx}/{total} with no city overlap in STAC bbox: {tile_name}",
                    flush=True,
                )
                return tile_name, None, False

            out_path = clipped_dir / tile_name
            if out_path.exists() and out_path.stat().st_size > 0:
                if cached_tile_matches_expected(out_path, clip_bbox):
                    print(f"  Reusing cached tile {idx}/{total}: {tile_name}", flush=True)
                    return tile_name, out_path, False
                print(
                    f"  Cached tile {idx}/{total} has stale extent; rebuilding: {tile_name}",
                    flush=True,
                )
                out_path.unlink(missing_ok=True)

            print(f"  Clipping tile {idx}/{total}: {tile_name}", flush=True)
            try:
                # Sign per tile at clip time: SAS tokens live ~1 h, much shorter
                # than a county-scale fetch — signing all items up front 403s
                # every tile past the first hour.
                tile_url = planetary_computer.sign(item).assets["image"].href
                attempt_clip(tile_name, tile_url, out_path, clip_bbox)
                return tile_name, out_path, False
            except Exception as first_error:
                print(
                    f"    First attempt failed for {tile_name}; re-signing and retrying",
                    flush=True,
                )
                try:
                    _reset_sas_cache()
                    refreshed_url = planetary_computer.sign(item).assets["image"].href
                    attempt_clip(tile_name, refreshed_url, out_path, clip_bbox)
                    return tile_name, out_path, False
                except Exception as second_error:
                    print(f"    Failed tile {tile_name}", flush=True)
                    print(f"       First error: {first_error}", flush=True)
                    print(f"       Second error: {second_error}", flush=True)
                    out_path.unlink(missing_ok=True)
                    return tile_name, None, True

        total = len(item_by_name)
        # Tiles are independent and rasterio releases the GIL during network
        # I/O, so a small thread pool parallelizes the fetch ~NAIP_FETCH_WORKERS×.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=NAIP_FETCH_WORKERS) as pool:
            futures = [
                pool.submit(fetch_one, idx, tile_name, item)
                for idx, (tile_name, item) in enumerate(
                    sorted(item_by_name.items()), start=1
                )
            ]
            for future in futures:
                tile_name, path, hard_failure = future.result()
                if path is not None:
                    local_paths.append(path)
                elif hard_failure:
                    failed_tiles.append(tile_name)

        if not local_paths:
            raise RuntimeError("Per-tile NAIP clipping failed for every tile.")

        failed_path = output_dir / "naip_failed_tiles.txt"
        if failed_tiles:
            failed_path.write_text("\n".join(failed_tiles) + "\n", encoding="utf-8")
            print(
                f"  {len(failed_tiles)} tile(s) failed and were skipped; details: {failed_path}",
                flush=True,
            )
        else:
            failed_path.unlink(missing_ok=True)

        local_filelist = output_dir / "naip_local_filelist.txt"
        local_filelist.write_text("".join(f"{path}\n" for path in local_paths), encoding="utf-8")
        local_vrt = output_dir / "naip_local_mosaic.vrt"
        build_vrt_from_filelist(local_filelist, local_vrt)
        return local_vrt

    print(f"  Searching Planetary Computer STAC for NAIP ({date_range}) in bbox...", flush=True)
    # NOTE: no sign_inplace modifier — items keep their raw (unsigned) hrefs
    # and are signed per tile at clip time, because SAS tokens expire after
    # ~1 h while a county-scale fetch runs for several.
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )
    search = catalog.search(
        collections=["naip"],
        bbox=bbox,
        datetime=date_range,
        # County-scale bboxes exceed 150 quads across vintages; 200 silently
        # truncated coverage.
        max_items=2000,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No NAIP imagery found for bbox {bbox} in date range {date_range}. "
            "Try widening --naip-start/--naip-end or check that the city is in the continental US."
        )
    print(f"  Found {len(items)} NAIP tile(s) – selecting best item per footprint...", flush=True)

    # Deduplicate by stable quad footprint, not the full item ID.
    # Some NAIP footprints can appear multiple times across vintages and native
    # resolutions (for example 0.30 m and 0.60 m products covering the same quad).
    # Keep a single best source tile per footprint so we do not run inference twice
    # on overlapping imagery and then dissolve the duplicates into bloated polygons.
    by_quad: dict[str, "pystac.Item"] = {}
    for item in items:
        key, _ = parse_naip_item_id(item.id)
        existing = by_quad.get(key)
        if existing is None:
            by_quad[key] = item
            continue

        existing_resolution = native_resolution_m_for_item(existing)
        candidate_resolution = native_resolution_m_for_item(item)
        existing_datetime = existing.datetime or pd.Timestamp.min.to_pydatetime()
        candidate_datetime = item.datetime or pd.Timestamp.min.to_pydatetime()

        if (
            candidate_resolution < existing_resolution
            or (
                candidate_resolution == existing_resolution
                and candidate_datetime > existing_datetime
            )
        ):
            by_quad[key] = item
    unique_items = list(by_quad.values())
    print(
        f"  Deduplicated to {len(unique_items)} unique footprint tile(s) "
        f"(finest native resolution, then most-recent vintage).",
        flush=True,
    )

    # Derive target UTM CRS from bbox centroid longitude
    lon_center = (bbox[0] + bbox[2]) / 2
    utm_zone = int((lon_center + 180) / 6) + 1
    epsg_utm = 32600 + utm_zone
    print(f"  Target CRS: EPSG:{epsg_utm} at {resolution_m}m resolution", flush=True)

    # Fetch every tile through rasterio into the local cache, then build the
    # unified citywide mosaic with a purely local VRT + gdalwarp.
    naip_path = output_dir / "naip_imagery.tif"
    local_vrt = clip_tiles_locally(unique_items, output_dir)
    print(
        "  Per-tile cache ready; warping local VRT to unified citywide mosaic...",
        flush=True,
    )
    direct_warp(local_vrt, naip_path)

    # Verify output
    with rasterio.open(naip_path) as src:
        bands, height, width = src.count, src.height, src.width
        crs_epsg = src.crs.to_epsg()
    print(
        f"  NAIP imagery written: {naip_path} "
        f"({width}×{height} px, {bands} bands, EPSG:{crs_epsg})",
        flush=True,
    )
    return naip_path


# ---------------------------------------------------------------------------
# Step 4: Run SegFormer inference
# ---------------------------------------------------------------------------

def _disk_kernel(radius: int) -> np.ndarray:
    """Create a circular structuring element for morphological operations."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


def _project_for_area(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Project a GeoDataFrame to its local UTM zone for accurate area/length.

    EPSG:3857 (Web Mercator) is NOT equal-area — it overstates area by a factor
    of ~1/cos²(latitude) (≈1.33× at 30°N, worse farther from the equator). Using
    it for `.area` inflated every parking acreage and parcel-area figure. UTM is
    locally equal-area to <0.1% across a city, so use it for all area math.
    Already-projected inputs (e.g. the UTM NAIP mask grid) are returned as-is.
    """
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    if gdf.crs.is_projected:
        return gdf
    return gdf.to_crs(gdf.estimate_utm_crs())


def strip_vegetation(
    binary_mask: np.ndarray,
    output_dir: Path,
    ndvi_threshold: float = NDVI_VEG_THRESHOLD,
) -> np.ndarray:
    """
    Zero out parking-mask pixels that are vegetation, using NDVI from the 4-band
    NAIP mosaic (band 1 = Red, band 4 = NIR). Captures the paper's near-infrared
    benefit — grass around and between lots reflects strongly in NIR — without
    retraining the RGB model. Removes grass the model over-predicted and helps
    split lots erroneously merged across grassy medians.

    No-op (returns the mask unchanged) when the NAIP mosaic is missing, lacks a
    NIR band, or does not align with the mask grid.
    """
    import rasterio

    naip_path = output_dir / "naip_imagery.tif"
    if not naip_path.exists():
        print("  NDVI strip skipped: naip_imagery.tif not found")
        return binary_mask
    with rasterio.open(naip_path) as src:
        if src.count < 4:
            print(f"  NDVI strip skipped: NAIP has {src.count} bands (need 4 incl. NIR)")
            return binary_mask
        red = src.read(1).astype(np.float32)
        nir = src.read(4).astype(np.float32)
    if red.shape != binary_mask.shape:
        print(
            f"  NDVI strip skipped: NAIP {red.shape} != mask {binary_mask.shape} "
            "(grids not aligned)"
        )
        return binary_mask

    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    before = int(binary_mask.sum())
    stripped = binary_mask.copy()
    stripped[ndvi > ndvi_threshold] = 0
    after = int(stripped.sum())
    print(
        f"  NDVI vegetation strip (>{ndvi_threshold}): {before:,} → {after:,} "
        f"parking pixels ({100 * (before - after) / max(before, 1):.1f}% removed)"
    )
    return stripped


def load_segformer_bundle(use_cpu: bool = False):
    """
    Load the parking SegFormer model once and reuse it across tiles.

    The UTEL-UIUC checkpoint is published as a PyTorch Lightning checkpoint rather
    than a full Transformers repo, so we reconstruct the HF SegFormer architecture
    from the B5 config and load `best_model.ckpt` directly.
    """
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import (
            SegformerConfig,
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. Install with: pip install torch transformers huggingface_hub"
        ) from e

    model_id = "UTEL-UIUC/SegFormer-large-parking"
    base_model_id = "nvidia/segformer-b5-finetuned-ade-640-640"
    if use_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  Loading SegFormer model ({model_id}) on {device.upper()}...")

    processor = SegformerImageProcessor.from_pretrained(base_model_id)
    config = SegformerConfig.from_pretrained(base_model_id)
    config.num_labels = 2
    config.id2label = {0: "background", 1: "parking"}
    config.label2id = {"background": 0, "parking": 1}

    checkpoint_path = hf_hub_download(model_id, "best_model.ckpt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }

    model = SegformerForSemanticSegmentation(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "SegFormer checkpoint load mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    model = model.to(device)
    model.eval()
    return processor, model, device, model_id

def run_segformer_inference(
    naip_path: Path,
    output_dir: Path,
    use_cpu: bool = False,
    processor=None,
    model=None,
    device: str | None = None,
    threshold: float = 0.5,
    nir_strip: bool = True,
    chunk_size_px: int = 6144,
    batch_size: int = 8,
) -> Path:
    """
    Run UIUC SegFormer-large-parking model on NAIP imagery.

    Streams the raster in spatial chunks so memory stays O(chunk_size_px²)
    regardless of city size, while producing BIT-IDENTICAL output to
    whole-raster processing: the 512×512 sliding-window grid is computed once
    over the full raster, and each chunk runs exactly the global windows that
    overlap its centre region (reading their union as border context). The
    model is strongly phase-sensitive — computing window positions relative
    to each chunk's read origin instead shifts probabilities enough to flip
    ~25% of near-threshold parking pixels — so the global grid is load-bearing.

    Per chunk: sliding-window inference → averaged probabilities → threshold
    → NDVI vegetation strip (NAIP band 4) → morphological opening, then the
    chunk centre is written to disk. A quantized probability map (uint8,
    0–255) is written alongside the mask so --rethreshold stays cheap.
    chunk_size_px is a multiple of the 512 px GTiff block size so chunk
    writes stay block-aligned.

    Model: UTEL-UIUC/SegFormer-large-parking (Apache 2.0, Hugging Face)
    Paper: "A Pipeline and NIR-Enhanced Dataset for Parking Lot Segmentation" WACV 2025
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Install with: pip install torch"
        ) from e

    import time

    import rasterio
    from rasterio.windows import Window
    from scipy.ndimage import binary_opening as _binary_opening

    TILE_SIZE = 512  # pixels (model was trained on 512x512)
    OVERLAP = 64     # pixel overlap between inference windows to reduce edge artifacts

    if processor is None or model is None or device is None:
        processor, model, device, _ = load_segformer_bundle(use_cpu=use_cpu)

    mask_path = output_dir / "parking_mask.tif"
    probs_path = output_dir / "parking_probs.tif"

    with rasterio.open(naip_path) as src:
        width, height, n_bands = src.width, src.height, src.count
        # Build clean GTiff profiles from scratch — never inherit source driver
        # keys (a VRT source would otherwise poison the output profile).
        base_profile = {
            "driver": "GTiff",
            "count": 1,
            "width": width,
            "height": height,
            "crs": src.crs,
            "transform": src.transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "bigtiff": "IF_SAFER",
        }
    mask_profile = dict(base_profile, dtype=rasterio.uint8, nodata=None)
    probs_profile = dict(base_profile, dtype=rasterio.uint8, nodata=None)

    do_nir = nir_strip and n_bands >= 4
    if nir_strip and not do_nir:
        print(f"  NDVI strip skipped: NAIP has {n_bands} bands (need 4 incl. NIR)")

    # Global sliding-window grid over the FULL raster (identical to the
    # original whole-raster implementation, including the appended final
    # row/column so the raster edge is always covered).
    step = TILE_SIZE - OVERLAP
    gy = list(range(0, height - TILE_SIZE + 1, step))
    gx = list(range(0, width - TILE_SIZE + 1, step))
    if not gy or gy[-1] + TILE_SIZE < height:
        gy.append(max(0, height - TILE_SIZE))
    if not gx or gx[-1] + TILE_SIZE < width:
        gx.append(max(0, width - TILE_SIZE))

    y_chunks = list(range(0, height, chunk_size_px))
    x_chunks = list(range(0, width, chunk_size_px))
    total_chunks = len(x_chunks) * len(y_chunks)
    print(
        f"  Chunked inference on {naip_path.name}: {width}×{height} px → "
        f"{total_chunks} chunk(s) ({len(y_chunks)} rows × {len(x_chunks)} cols, "
        f"chunk={chunk_size_px} px), global grid {len(gy)}×{len(gx)} windows "
        f"of {TILE_SIZE} px (overlap {OVERLAP} px)"
    )

    # ----------------------------------------------------------------------
    # Chunk-level resume: a journal records every completed chunk so an
    # interrupted run (crash, reboot, Windows Update...) restarts where it
    # left off instead of re-inferring the whole raster. The header ties the
    # journal to the raster/params; any mismatch discards it.
    # ----------------------------------------------------------------------
    journal_path = output_dir / "inference_chunks_done.txt"
    journal_header = (
        f"{width}x{height} chunk={chunk_size_px} thr={threshold} nir={int(do_nir)}"
    )
    done_chunks: set[tuple[int, int]] = set()
    resume = False
    if journal_path.exists() and mask_path.exists() and probs_path.exists():
        journal_lines = journal_path.read_text(encoding="utf-8").splitlines()
        if journal_lines and journal_lines[0] == journal_header:
            for line in journal_lines[1:]:
                cy_s, _, cx_s = line.partition(",")
                done_chunks.add((int(cy_s), int(cx_s)))
            resume = True
    if resume:
        print(
            f"  Resuming: {len(done_chunks)}/{total_chunks} chunks already complete "
            f"(pixel totals below cover only this run's chunks)"
        )
    else:
        # Remove stale outputs so GDAL never opens them with a mismatched layout
        mask_path.unlink(missing_ok=True)
        probs_path.unlink(missing_ok=True)
        journal_path.write_text(journal_header + "\n", encoding="utf-8")

    kernel = _disk_kernel(MORPH_OPEN_RADIUS_PX)
    px_thresholded = 0
    px_after_strip = 0
    px_after_open = 0

    def _open_out(path: Path, profile: dict):
        # Resume reopens the existing raster in update mode; a fresh run
        # creates it from the profile.
        if resume:
            return rasterio.open(path, "r+")
        return rasterio.open(path, "w", **profile)

    run_t0 = time.perf_counter()
    chunks_done_run = 0
    with rasterio.open(naip_path) as src, \
         _open_out(mask_path, mask_profile) as mask_dst, \
         _open_out(probs_path, probs_profile) as probs_dst:
        for chunk_idx, (cy, cx) in enumerate(
            [(y, x) for y in y_chunks for x in x_chunks], start=1
        ):
            if (cy, cx) in done_chunks:
                continue
            chunk_t0 = time.perf_counter()
            cy_end = min(cy + chunk_size_px, height)
            cx_end = min(cx + chunk_size_px, width)

            # Global windows overlapping this chunk's centre region. Their
            # union is the read region; every centre pixel receives exactly
            # the same window contributions as in a whole-raster run.
            wy = [y0 for y0 in gy if y0 < cy_end and y0 + TILE_SIZE > cy]
            wx = [x0 for x0 in gx if x0 < cx_end and x0 + TILE_SIZE > cx]
            ry0 = min(wy)
            rx0 = min(wx)
            ry1 = min(height, max(y0 + TILE_SIZE for y0 in wy))
            rx1 = min(width, max(x0 + TILE_SIZE for x0 in wx))
            read_h = ry1 - ry0
            read_w = rx1 - rx0

            pct = 100 * chunk_idx / total_chunks
            print(
                f"  [{pct:5.1f}%] Chunk {chunk_idx}/{total_chunks}: "
                f"rows {cy}–{cy_end}, cols {cx}–{cx_end} "
                f"({len(wy) * len(wx)} windows)",
                flush=True,
            )

            chunk_data = src.read(window=Window(rx0, ry0, read_w, read_h))

            # --------------------------------------------------------------
            # Sliding-window inference with weighted averaging over overlaps
            # --------------------------------------------------------------
            full_mask = np.zeros((read_h, read_w), dtype=np.float32)
            weight_mask = np.zeros((read_h, read_w), dtype=np.float32)

            pending: list[tuple[int, int, int, int, np.ndarray]] = []

            def flush_batch() -> None:
                if not pending:
                    return
                inputs = processor(
                    images=[p[4] for p in pending], return_tensors="pt", do_resize=False
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                upsampled = torch.nn.functional.interpolate(
                    outputs.logits,  # (B, num_labels, H/4, W/4)
                    size=(TILE_SIZE, TILE_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                batch_probs = torch.softmax(upsampled, dim=1)[:, 1].cpu().numpy()
                for (y0, x0, y1, x1, _), probs in zip(pending, batch_probs):
                    probs = probs[: y1 - y0, : x1 - x0]
                    full_mask[y0:y1, x0:x1] += probs
                    weight_mask[y0:y1, x0:x1] += 1.0
                pending.clear()

            for gy0 in wy:
                for gx0 in wx:
                    y0 = gy0 - ry0
                    x0 = gx0 - rx0
                    y1 = min(y0 + TILE_SIZE, read_h)
                    x1 = min(x0 + TILE_SIZE, read_w)
                    patch = chunk_data[:, y0:y1, x0:x1]

                    pad_h = TILE_SIZE - (y1 - y0)
                    pad_w = TILE_SIZE - (x1 - x0)
                    if pad_h > 0 or pad_w > 0:
                        patch = np.pad(
                            patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
                        )

                    # Skip windows that are predominantly nodata (zero-filled
                    # gaps at bbox edges) — the model produces garbage on
                    # black imagery.
                    if (patch[0] == 0).mean() > 0.5:
                        continue

                    rgb = patch[:3].transpose(1, 2, 0).astype(np.uint8)
                    pending.append((y0, x0, y1, x1, rgb))
                    if len(pending) >= batch_size:
                        flush_batch()
            flush_batch()

            avg_probs = full_mask / np.where(weight_mask == 0, 1.0, weight_mask)

            # --------------------------------------------------------------
            # Threshold → NDVI strip → morphological opening (on the full
            # read region so the kernel never sees a seam), then write the
            # centre only.
            # --------------------------------------------------------------
            inner = (slice(cy - ry0, cy_end - ry0), slice(cx - rx0, cx_end - rx0))
            out_window = Window(cx, cy, cx_end - cx, cy_end - cy)

            probs_dst.write(
                np.rint(avg_probs[inner] * 255).astype(np.uint8), 1, window=out_window
            )

            binary_chunk = (avg_probs >= threshold).astype(np.uint8)
            px_thresholded += int(binary_chunk[inner].sum())

            if do_nir:
                red = chunk_data[0].astype(np.float32)
                nir = chunk_data[3].astype(np.float32)
                denom = nir + red
                with np.errstate(invalid="ignore"):
                    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
                binary_chunk[ndvi > NDVI_VEG_THRESHOLD] = 0
            px_after_strip += int(binary_chunk[inner].sum())

            binary_chunk = _binary_opening(binary_chunk, structure=kernel).astype(
                np.uint8
            )
            px_after_open += int(binary_chunk[inner].sum())

            mask_dst.write(binary_chunk[inner], 1, window=out_window)

            with open(journal_path, "a", encoding="utf-8") as journal:
                journal.write(f"{cy},{cx}\n")
            chunks_done_run += 1
            chunk_dt = time.perf_counter() - chunk_t0
            elapsed = time.perf_counter() - run_t0
            remaining = total_chunks - len(done_chunks) - chunks_done_run
            eta_h = remaining * (elapsed / chunks_done_run) / 3600
            print(
                f"    chunk {chunk_idx} done in {chunk_dt:.0f}s "
                f"({len(wy) * len(wx) / max(chunk_dt, 1e-9):.1f} win/s incl. I/O) — "
                f"ETA {eta_h:.1f} h for {remaining} remaining chunk(s)",
                flush=True,
            )

    print(f"  Probability map saved: {probs_path}")
    print(f"  Threshold {threshold}: {px_thresholded:,} parking pixels")
    if do_nir:
        print(
            f"  NDVI vegetation strip (>{NDVI_VEG_THRESHOLD}): "
            f"{px_thresholded:,} → {px_after_strip:,} parking pixels "
            f"({100 * (px_thresholded - px_after_strip) / max(px_thresholded, 1):.1f}% removed)"
        )
    print(
        f"  Morphological opening (r={MORPH_OPEN_RADIUS_PX}px): "
        f"{px_after_strip:,} → {px_after_open:,} parking pixels "
        f"({100 * (px_after_strip - px_after_open) / max(px_after_strip, 1):.1f}% removed)"
    )
    print(f"  Mask written: {mask_path}")
    return mask_path


def rethreshold_from_probs(
    output_dir: Path,
    threshold: float = 0.5,
    nir_strip: bool = True,
) -> Path:
    """
    Re-read a saved parking_probs.tif and produce a new parking_mask.tif at the
    given threshold.  Runs in seconds — no model, no NAIP fetch needed.

    Use this after a full inference run to rapidly iterate on the threshold
    without re-running the expensive ML step.
    """
    import rasterio
    from scipy.ndimage import binary_opening as _binary_opening

    probs_path = output_dir / "parking_probs.tif"
    mask_path = output_dir / "parking_mask.tif"

    if not probs_path.exists():
        raise FileNotFoundError(
            f"No probability map found at {probs_path}. "
            "Run a full inference first (without --rethreshold) to generate it."
        )

    print(f"  Reading probability map: {probs_path}")
    with rasterio.open(probs_path) as src:
        avg_probs = src.read(1)
        profile = src.profile.copy()
    if avg_probs.dtype == np.uint8:
        # Inference stores probabilities quantized to uint8 (0–255)
        avg_probs = avg_probs.astype(np.float32) / 255.0

    binary_mask = (avg_probs >= threshold).astype(np.uint8)
    print(f"  Threshold {threshold}: {int(binary_mask.sum()):,} parking pixels")

    if nir_strip:
        binary_mask = strip_vegetation(binary_mask, output_dir)

    kernel = _disk_kernel(MORPH_OPEN_RADIUS_PX)
    before_px = int(binary_mask.sum())
    binary_mask = _binary_opening(binary_mask, structure=kernel).astype(np.uint8)
    after_px = int(binary_mask.sum())
    print(
        f"  Morphological opening (r={MORPH_OPEN_RADIUS_PX}px): "
        f"{before_px:,} → {after_px:,} parking pixels "
        f"({100 * (before_px - after_px) / max(before_px, 1):.1f}% removed)"
    )

    profile.update(count=1, dtype=rasterio.uint8, nodata=None,
                   compress="lzw", bigtiff="IF_SAFER")
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(binary_mask, 1)

    print(f"  Mask written: {mask_path}")
    return mask_path


def dissolve_parking_polygons(
    parking_gdf: gpd.GeoDataFrame,
    min_area_sqft: float = MIN_PARKING_AREA_SQFT,
) -> gpd.GeoDataFrame:
    """
    Merge overlapping/adjacent parking polygons across tile boundaries and
    recompute area metrics in a projected CRS.
    """
    from shapely.ops import unary_union

    if parking_gdf.empty:
        return parking_gdf

    parking_proj = parking_gdf.to_crs(3857)
    merged = unary_union(parking_proj.geometry.tolist())
    merged_gdf = gpd.GeoDataFrame(geometry=[merged], crs=3857).explode(index_parts=False)
    merged_gdf = merged_gdf[merged_gdf.geometry.notna() & ~merged_gdf.geometry.is_empty].copy()
    merged_gdf["parking_area_sqft"] = merged_gdf.geometry.area * 10.7639
    merged_gdf = merged_gdf[merged_gdf["parking_area_sqft"] >= min_area_sqft].copy()
    max_area_sqft = MAX_PARKING_AREA_ACRES * 43560.0
    merged_gdf = merged_gdf[merged_gdf["parking_area_sqft"] <= max_area_sqft].copy()
    merged_gdf["parking_area_acres"] = merged_gdf["parking_area_sqft"] / 43560.0
    merged_gdf["source"] = "ml_segmentation"
    merged_gdf["confidence"] = 1.0
    merged_gdf = merged_gdf.to_crs(4326)
    merged_gdf = merged_gdf.reset_index(drop=True)
    return merged_gdf[
        ["geometry", "source", "confidence", "parking_area_sqft", "parking_area_acres"]
    ]


def run_segformer_inference_on_tiles(
    tile_dir: Path,
    output_dir: Path,
    use_cpu: bool = False,
    overwrite: bool = False,
) -> gpd.GeoDataFrame:
    """
    Run parking inference tile-by-tile using cached NAIP rasters, persist per-tile
    vector outputs for resumability, then dissolve them into one citywide layer.
    """
    tile_paths = sorted(tile_dir.glob("*.tif"))
    if not tile_paths:
        raise FileNotFoundError(f"No NAIP tiles found in {tile_dir}")

    tile_vector_dir = output_dir / "tile_vectors"
    tile_mask_dir = output_dir / "tile_masks"
    tile_vector_dir.mkdir(parents=True, exist_ok=True)
    tile_mask_dir.mkdir(parents=True, exist_ok=True)

    tile_gdfs: list[gpd.GeoDataFrame] = []
    total_tiles = len(tile_paths)
    print(f"  Running tile-first inference across {total_tiles} cached NAIP tiles...")
    processor, model, device, _ = load_segformer_bundle(use_cpu=use_cpu)

    for idx, tile_path in enumerate(tile_paths, start=1):
        tile_name = tile_path.stem
        tile_mask_path = tile_mask_dir / f"{tile_name}-mask.tif"
        tile_vector_path = tile_vector_dir / f"{tile_name}.parquet"

        if tile_vector_path.exists() and not overwrite:
            tile_gdf = gpd.read_parquet(tile_vector_path)
            if not tile_gdf.empty and tile_gdf.crs is None:
                tile_gdf = tile_gdf.set_crs(4326)
            tile_gdfs.append(tile_gdf)
            print(f"    Reusing tile vectors {idx}/{total_tiles}: {tile_name}")
            continue

        print(f"    Inferring tile {idx}/{total_tiles}: {tile_name}")
        mask_path = run_segformer_inference(
            naip_path=tile_path,
            output_dir=tile_mask_dir,
            use_cpu=use_cpu,
            processor=processor,
            model=model,
            device=device,
        )

        if mask_path != tile_mask_path:
            try:
                mask_path.rename(tile_mask_path)
            except OSError:
                tile_mask_path.write_bytes(mask_path.read_bytes())
                mask_path.unlink(missing_ok=True)
        tile_gdf = vectorize_mask(tile_mask_path, min_area_sqft=MIN_PARKING_AREA_SQFT)
        tile_gdf.to_parquet(tile_vector_path, index=False)
        tile_gdfs.append(tile_gdf)

    if not tile_gdfs:
        return gpd.GeoDataFrame(
            columns=["geometry", "source", "confidence", "parking_area_sqft", "parking_area_acres"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    combined = pd.concat(tile_gdfs, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    print(f"  Combining {len(combined_gdf):,} raw tile polygons across the city...")
    dissolved = dissolve_parking_polygons(combined_gdf, min_area_sqft=MIN_PARKING_AREA_SQFT)
    print(f"  Citywide dissolved parking polygons: {len(dissolved):,}")
    return dissolved


# ---------------------------------------------------------------------------
# Step 5: Vectorize mask → parking polygons with post-processing
# ---------------------------------------------------------------------------

def vectorize_mask(
    mask_path: Path,
    min_area_sqft: float = MIN_PARKING_AREA_SQFT,
    block_px: int = 8192,
) -> gpd.GeoDataFrame:
    """
    Convert binary parking mask GeoTIFF to GeoDataFrame of parking polygons.

    Streams the mask in blocks so memory stays O(block²) at any city size
    (whole-array PIL ModeFilter cannot even allocate at county scale). Each
    block is read with a margin larger than the ModeFilter halo, so filtered
    block centres are identical to a whole-array pass; polygons touching an
    interior block edge are unioned across blocks afterwards, so seams never
    split a lot.

    Post-processing:
    - ModeFilter(13) majority-vote smoothing (matches UTEL-UIUC reference pipeline)
    - Fill small holes (< 50 sqft) within detected parking areas
    - Remove tiny fragments (< min_area_sqft)
    - Simplify polygon edges (tolerance = 0.5m)
    - Reproject to EPSG:4326
    """
    import rasterio
    import rasterio.features
    from PIL import Image, ImageFilter
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
    from shapely.geometry import shape
    from shapely.ops import unary_union

    print(f"  Vectorizing mask: {mask_path.name}...")

    FILTER_SIZE = 13  # ModeFilter neighbourhood (reference-pipeline value)
    BLOCK = block_px
    MARGIN = 16  # > FILTER_SIZE // 2, so block centres match a whole-array filter

    plain_geoms: list = []
    seam_geoms: list = []
    n_parking = 0

    with rasterio.open(mask_path) as src:
        transform = src.transform
        crs = src.crs
        width, height = src.width, src.height
        eps = abs(transform.a) * 1e-6

        for row0 in range(0, height, BLOCK):
            for col0 in range(0, width, BLOCK):
                row1 = min(row0 + BLOCK, height)
                col1 = min(col0 + BLOCK, width)
                rr0 = max(0, row0 - MARGIN)
                rc0 = max(0, col0 - MARGIN)
                rr1 = min(height, row1 + MARGIN)
                rc1 = min(width, col1 + MARGIN)
                block = src.read(1, window=Window(rc0, rr0, rc1 - rc0, rr1 - rr0))
                if not block.any():
                    continue

                pil_mask = Image.fromarray(block * 255, mode="L")
                pil_mask = pil_mask.filter(ImageFilter.ModeFilter(size=FILTER_SIZE))
                filtered = (np.array(pil_mask) >= 128).astype(np.uint8)
                center = filtered[row0 - rr0 : row1 - rr0, col0 - rc0 : col1 - rc0]
                if not center.any():
                    continue
                n_parking += int(center.sum())

                block_window = Window(col0, row0, col1 - col0, row1 - row0)
                block_transform = window_transform(block_window, transform)

                # CRS coordinates of this block's interior edges (i.e. seams
                # shared with a neighbouring block, not the raster border).
                left_x = (transform * (col0, 0))[0] if col0 > 0 else None
                right_x = (transform * (col1, 0))[0] if col1 < width else None
                top_y = (transform * (0, row0))[1] if row0 > 0 else None
                bottom_y = (transform * (0, row1))[1] if row1 < height else None

                for geom_json, _val in rasterio.features.shapes(
                    center, mask=(center == 1), transform=block_transform
                ):
                    geom = shape(geom_json)
                    minx, miny, maxx, maxy = geom.bounds
                    touches_seam = (
                        (left_x is not None and abs(minx - left_x) < eps)
                        or (right_x is not None and abs(maxx - right_x) < eps)
                        or (top_y is not None and abs(maxy - top_y) < eps)
                        or (bottom_y is not None and abs(miny - bottom_y) < eps)
                    )
                    (seam_geoms if touches_seam else plain_geoms).append(geom)

    print(f"  ModeFilter({FILTER_SIZE}) applied: {n_parking:,} parking pixels")

    if seam_geoms:
        merged = unary_union(seam_geoms)
        if merged.geom_type == "Polygon":
            merged_parts = [merged]
        else:
            merged_parts = [g for g in merged.geoms if g.geom_type == "Polygon"]
        print(
            f"  Merged {len(seam_geoms):,} block-seam fragment(s) into "
            f"{len(merged_parts):,} polygon(s)"
        )
        geoms = plain_geoms + merged_parts
    else:
        geoms = plain_geoms

    if not geoms:
        print("  Warning: No parking polygons detected in mask.")
        return gpd.GeoDataFrame(
            columns=["geometry", "source", "confidence", "parking_area_sqft", "parking_area_acres"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    print(f"  Raw polygons from mask: {len(geoms):,}")

    # Build GeoDataFrame in mask CRS
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs)

    # Reproject to local UTM for accurate (equal-area) area calculations.
    gdf_proj = _project_for_area(gdf)
    gdf_proj["area_m2"] = gdf_proj.geometry.area
    gdf_proj["parking_area_sqft"] = gdf_proj["area_m2"] * 10.7639

    # Remove small fragments
    gdf_proj = gdf_proj[gdf_proj["parking_area_sqft"] >= min_area_sqft].copy()
    print(f"  After min-area filter (>= {min_area_sqft} sqft): {len(gdf_proj):,} polygons")

    # Remove implausibly large features (airports, distribution centres)
    gdf_proj["parking_area_acres"] = gdf_proj["parking_area_sqft"] / 43560.0
    max_area_sqft = MAX_PARKING_AREA_ACRES * 43560.0
    n_before = len(gdf_proj)
    gdf_proj = gdf_proj[gdf_proj["parking_area_sqft"] <= max_area_sqft].copy()
    n_removed = n_before - len(gdf_proj)
    if n_removed:
        print(
            f"  Removed {n_removed} feature(s) exceeding {MAX_PARKING_AREA_ACRES} ac max area"
        )

    # Simplify edges (0.5m tolerance in projected CRS)
    gdf_proj["geometry"] = gdf_proj.geometry.simplify(0.5, preserve_topology=True)

    # Fill small holes within polygons (holes < 50 sqft)
    def fill_holes(geom):
        from shapely.geometry import Polygon
        if geom.geom_type == "Polygon":
            exterior = geom.exterior
            kept_interiors = [
                ring for ring in geom.interiors
                if Polygon(ring).area * 10.7639 >= 50
            ]
            return Polygon(exterior, kept_interiors)
        return geom

    gdf_proj["geometry"] = gdf_proj.geometry.apply(fill_holes)

    # Reproject back to WGS84
    gdf_wgs84 = gdf_proj.to_crs(4326)
    gdf_wgs84 = gdf_wgs84.rename(columns={"area_m2": "_area_m2"})

    # Recompute area in local UTM after simplification
    gdf_proj_final = _project_for_area(gdf_wgs84)
    gdf_wgs84["parking_area_sqft"] = gdf_proj_final.geometry.area * 10.7639
    gdf_wgs84["parking_area_acres"] = gdf_wgs84["parking_area_sqft"] / 43560.0
    gdf_wgs84["source"] = "ml_segmentation"
    gdf_wgs84["confidence"] = 1.0  # binary threshold; future: store avg probability

    # Drop internal area column
    gdf_wgs84 = gdf_wgs84.drop(columns=["_area_m2"], errors="ignore")

    print(f"  Final parking polygons: {len(gdf_wgs84):,}")
    return gdf_wgs84[
        ["geometry", "source", "confidence", "parking_area_sqft", "parking_area_acres"]
    ]


# ---------------------------------------------------------------------------
# Step 5b (optional): Fetch OSM parking as supplement
# ---------------------------------------------------------------------------

def fetch_osm_parking(city_key: str, bbox: list[float] | None = None) -> gpd.GeoDataFrame:
    """Fetch OSM parking polygons as supplemental source."""
    try:
        import osmnx as ox
        from shapely.geometry import box
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. Install with: pip install osmnx"
        ) from e

    # overpass-api.de's round-robin DNS includes a dead backend IP
    # (65.109.112.52 was refusing TCP connections during the 2026-06-01 Mac
    # rollout); the healthy IP is 162.55.144.139. osmnx + urllib3's connection
    # pool will repeatedly pick whichever IP DNS returns first and not failover
    # to the second on connection refused, so retries don't help — every call
    # to the dead IP fails the same way. Pin name resolution to the known-good
    # IP via socket.getaddrinfo monkey-patch.
    # Override targets via env:
    #   OSMNX_OVERPASS_URL    — full URL (default overpass-api.de)
    #   OSMNX_OVERPASS_HOST_IP — IP to pin overpass-api.de to (default 162.55.144.139)
    _override = os.getenv("OSMNX_OVERPASS_URL")
    if _override:
        ox.settings.overpass_url = _override
    ox.settings.doh_url_template = None

    # overpass-api.de has 4 backends behind a round-robin DNS:
    #   IPv4: 65.109.112.52, 162.55.144.139  (BOTH frequently refusing TCP on 2026-06-01)
    #   IPv6: 2a01:4f8:261:3c4f::2, 2a01:4f9:3051:3e48::2  (alive — curl connects fine)
    # osmnx's `_config_dns` calls `socket.gethostbyname(hostname)` to pick ONE
    # IP, then forces urllib3 to use only that IP. `gethostbyname` returns
    # IPv4 only — so osmnx always pins to a (currently dead) IPv4 backend
    # and never tries the working IPv6 ones.
    # Fix: make `gethostbyname` raise gaierror for the overpass host so osmnx
    # falls back to `_resolve_host_via_doh` — which (because we already
    # disabled DoH) returns the hostname itself, leaving `socket.getaddrinfo`
    # unrestricted. urllib3 then iterates through getaddrinfo's full result
    # set (v4 + v6) and connects to whichever responds first.
    import socket as _socket
    _pinned_host = "overpass-api.de"
    if not getattr(_socket, "_overpass_pinned", False):
        _orig_gethostbyname = _socket.gethostbyname
        def _gethostbyname_force_doh(host):
            if host == _pinned_host:
                # Force osmnx to skip its single-IP pin so getaddrinfo's full
                # v4+v6 list is used; lets urllib3 fail over to IPv6 when v4 dies.
                raise _socket.gaierror("forced: use full getaddrinfo for overpass round-robin")
            return _orig_gethostbyname(host)
        _socket.gethostbyname = _gethostbyname_force_doh
        _socket._overpass_pinned = True
        print(f"  Unpinned {_pinned_host} (gethostbyname → gaierror → full getaddrinfo v4+v6 round-robin)")

    query = CITY_OSM_QUERIES.get(city_key)
    if not query:
        print(f"  Warning: No OSM query defined for city '{city_key}'. Skipping OSM supplement.")
        return gpd.GeoDataFrame()

    tags = {
        "amenity": "parking",
        "landuse": "parking",
        "parking": ["surface", "lane"],
    }
    print(f"  Fetching OSM parking for: {query}")
    # Overpass-api.de rate-limits concurrent requests from one IP — when
    # multiple cities are extracted in parallel, the second and later jobs see
    # "Connection refused" and the whole extract aborts. Retry with backoff so
    # a transient block doesn't waste 15 minutes of downstream work.
    import time
    last_err = None
    gdf = None
    for attempt in range(6):
        try:
            if bbox:
                gdf = ox.features_from_polygon(box(*bbox), tags=tags)
            else:
                gdf = ox.features_from_place(query, tags=tags)
            break
        except Exception as e:
            last_err = e
            wait = min(60, 5 * (2 ** attempt))  # 5, 10, 20, 40, 60, 60 s
            print(f"  OSM fetch attempt {attempt+1}/6 failed: {e}; retrying in {wait}s")
            time.sleep(wait)
    if gdf is None:
        print(f"  Warning: OSM fetch failed after retries: {last_err}. Continuing without OSM supplement.")
        return gpd.GeoDataFrame()

    # Keep only polygon geometries
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf.to_crs(4326)

    # Compute area in local UTM (equal-area)
    gdf_proj = _project_for_area(gdf)
    gdf["parking_area_sqft"] = gdf_proj.geometry.area * 10.7639
    gdf["parking_area_acres"] = gdf["parking_area_sqft"] / 43560.0
    gdf["source"] = "osm_supplement"
    gdf["confidence"] = 1.0

    gdf = gdf[["geometry", "source", "confidence", "parking_area_sqft", "parking_area_acres"]]
    print(f"  OSM parking polygons: {len(gdf):,}")
    return gdf


def _unpin_overpass_dns() -> None:
    """Make socket.gethostbyname raise for overpass-api.de so osmnx falls back to
    the full getaddrinfo (v4+v6) result set instead of pinning one (often-dead)
    IPv4 backend. Idempotent; shared by the OSM parking and road fetches."""
    import socket as _socket
    if getattr(_socket, "_overpass_pinned", False):
        return
    _orig = _socket.gethostbyname

    def _patched(host):
        if host == "overpass-api.de":
            raise _socket.gaierror("forced: use full getaddrinfo for overpass round-robin")
        return _orig(host)

    _socket.gethostbyname = _patched
    _socket._overpass_pinned = True


# Half-width (metres) to buffer an OSM road centreline by, per highway class.
# Roads are the landscape feature most often confused with parking lots; the
# paper (sec 4.2.4) subtracts road buffers from predictions. Widths are
# conservative half-carriageway estimates incl. a small margin; when an explicit
# `lanes` tag is present we use lanes*1.75 m + 1.5 m if larger.
ROAD_HALF_WIDTH_M = {
    "motorway": 18, "trunk": 14, "primary": 11, "secondary": 9, "tertiary": 7,
    "residential": 5, "unclassified": 6, "service": 3.5,
    "motorway_link": 10, "trunk_link": 9, "primary_link": 8, "secondary_link": 7,
}
ROAD_HALF_WIDTH_DEFAULT_M = 6.0


def remove_road_overlap(
    parking_gdf: gpd.GeoDataFrame,
    bbox: list[float],
) -> gpd.GeoDataFrame:
    """
    Subtract OSM road buffers from ML parking polygons (paper sec 4.2.4).

    Roads are the most common false-positive source for surface-parking
    segmentation. We fetch OSM road centrelines for the bbox, buffer each by a
    half-width from its highway class (or lane count when tagged), and subtract
    the union from the parking polygons. Slivers left below the min-area
    threshold are dropped. Graceful no-op when OSM is unreachable.
    """
    if parking_gdf.empty:
        return parking_gdf
    try:
        import osmnx as ox
        from shapely.geometry import box
    except ImportError:
        print("  Road removal skipped: osmnx not installed")
        return parking_gdf

    _unpin_overpass_dns()
    ox.settings.doh_url_template = None

    lon_center = (bbox[0] + bbox[2]) / 2
    utm_epsg = 32600 + int((lon_center + 180) / 6) + 1

    try:
        roads = ox.features_from_polygon(box(*bbox), tags={"highway": True})
    except Exception as e:  # noqa: BLE001 — OSM is best-effort here
        print(f"  Road removal skipped (OSM fetch failed: {e})")
        return parking_gdf

    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
    if roads.empty:
        print("  Road removal: no OSM roads found in bbox")
        return parking_gdf
    roads = roads.to_crs(utm_epsg)

    def _half_width(row) -> float:
        hw = row.get("highway")
        hw = hw[0] if isinstance(hw, list) else hw
        base = ROAD_HALF_WIDTH_M.get(hw, ROAD_HALF_WIDTH_DEFAULT_M)
        lanes = row.get("lanes")
        try:
            lanes = float(lanes[0] if isinstance(lanes, list) else lanes)
            return max(base, lanes * 1.75 + 1.5)
        except (TypeError, ValueError):
            return base

    road_union = roads.geometry.buffer(roads.apply(_half_width, axis=1).values).union_all()

    proj = parking_gdf.to_crs(utm_epsg)
    before_ac = proj.geometry.area.sum() / 4046.8564224
    proj["geometry"] = proj.geometry.difference(road_union)
    proj = proj[~proj.geometry.is_empty & proj.geometry.notna()].copy()
    proj = proj.explode(index_parts=False)
    proj["parking_area_sqft"] = proj.geometry.area * 10.7639
    proj = proj[proj["parking_area_sqft"] >= MIN_PARKING_AREA_SQFT].copy()
    proj["parking_area_acres"] = proj["parking_area_sqft"] / 43560.0
    after_ac = proj.geometry.area.sum() / 4046.8564224
    print(
        f"  Road removal: {before_ac:.1f} → {after_ac:.1f} acres "
        f"({len(roads):,} road lines; {len(proj):,} polygons after cut)"
    )
    return proj.to_crs(4326)


def merge_parking_sources(
    ml_gdf: gpd.GeoDataFrame,
    osm_gdf: gpd.GeoDataFrame,
    overlap_threshold: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Merge ML-detected and OSM parking polygons.
    OSM polygons that overlap >= overlap_threshold with ML output are dropped
    (already captured by ML). Non-overlapping OSM polygons are added.
    """
    if osm_gdf.empty:
        return ml_gdf

    ml_proj = ml_gdf.to_crs(3857)
    osm_proj = osm_gdf.to_crs(3857)

    # For each OSM polygon, check overlap with ML output
    osm_to_keep = []
    for idx, osm_row in osm_proj.iterrows():
        osm_area = osm_row.geometry.area
        if osm_area == 0:
            continue
        # Check intersection with any ML polygon
        intersecting = ml_proj[ml_proj.geometry.intersects(osm_row.geometry)]
        if intersecting.empty:
            osm_to_keep.append(idx)
            continue
        overlap_area = intersecting.geometry.intersection(osm_row.geometry).area.sum()
        if overlap_area / osm_area < overlap_threshold:
            osm_to_keep.append(idx)

    osm_supplement = osm_gdf.loc[osm_to_keep].copy()
    print(f"  OSM supplement (non-overlapping with ML): {len(osm_supplement):,} polygons")

    combined = pd.concat([ml_gdf, osm_supplement], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Step 6: Spatial join parking polygons to parcels
# ---------------------------------------------------------------------------

def identify_parcel_id_column(parcels: gpd.GeoDataFrame) -> str:
    """Identify the parcel ID column in the parcel GeoDataFrame."""
    candidates = [
        "parcel_id", "PARCEL_ID", "APN", "apn", "PIN", "pin",
        "parcelID", "ParcelID", "PARID", "pid", "PID",
        "objectid", "OBJECTID", "GIS_PIN", "gis_pin",
    ]
    for col in candidates:
        if col in parcels.columns:
            return col
    # Fall back to index
    return None


def clip_parking_to_parcels(
    parking: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    min_area_sqft: float = MIN_PARKING_AREA_SQFT,
) -> gpd.GeoDataFrame:
    """
    Clip parking polygons to parcel boundaries.

    This splits large blobs that span multiple parcels into separate
    parcel-bounded features with straight edges following property lines.
    Parking fragments outside all parcels are kept as-is.
    """
    print(f"  Clipping {len(parking):,} parking polygons to parcel boundaries...")

    # Normalise CRS
    if parcels.crs is None or parcels.crs.to_epsg() != 4326:
        parcels = parcels.to_crs(4326)
    if parking.crs is None or parking.crs.to_epsg() != 4326:
        parking = parking.to_crs(4326)

    # Work in projected CRS for accurate area calculations
    parking_proj = parking.to_crs(3857).reset_index(drop=True)
    parcels_proj = parcels.to_crs(3857).reset_index(drop=True)

    # Overlay: intersection of parking with parcels → parcel-bounded pieces
    clipped = gpd.overlay(
        parking_proj[["geometry"]],
        parcels_proj[["geometry"]],
        how="intersection",
        keep_geom_type=True,
    )

    # Also keep parking fragments that don't overlap any parcel (exempt areas)
    diff = gpd.overlay(
        parking_proj[["geometry"]],
        parcels_proj[["geometry"]],
        how="difference",
        keep_geom_type=True,
    )
    combined = pd.concat([clipped, diff], ignore_index=True)

    # Recompute area and re-filter
    combined["parking_area_sqft"] = combined.geometry.area * 10.7639
    combined["parking_area_acres"] = combined["parking_area_sqft"] / 43560.0

    n_before = len(combined)
    combined = combined[combined["parking_area_sqft"] >= min_area_sqft].copy()

    # Remove implausibly large features
    max_area_sqft = MAX_PARKING_AREA_ACRES * 43560.0
    combined = combined[combined["parking_area_sqft"] <= max_area_sqft].copy()

    # Simplify edges slightly for cleaner output
    combined["geometry"] = combined.geometry.simplify(0.3, preserve_topology=True)

    # Reproject back to WGS84
    result = combined.to_crs(4326)
    result["source"] = "ml_segmentation"
    result["confidence"] = 1.0

    # Recompute area after simplification
    result_proj = result.to_crs(3857)
    result["parking_area_sqft"] = result_proj.geometry.area * 10.7639
    result["parking_area_acres"] = result["parking_area_sqft"] / 43560.0

    print(f"  After parcel clipping: {len(result):,} features "
          f"(was {len(parking):,} blobs, {n_before - len(combined):,} tiny fragments removed)")

    return result[["geometry", "source", "confidence", "parking_area_sqft", "parking_area_acres"]]


def deduplicate_overlapping_parking(
    parking: gpd.GeoDataFrame,
    min_overlap_sqft: float = 1.0,
) -> gpd.GeoDataFrame:
    """Carve overlapping land out of duplicate OSM parking polygons.

    OSM mappers sometimes digitize the same lot twice (different edits by
    different users over time). With area-based metrics summed across all
    polygons that double-counts both the surface area AND the dollar value.
    Across our 18 cities this affects 16 of them — usually <1% of total
    area but in extreme cases (Baltimore: 114 duplicate polys) it's worth
    fixing.

    Method: sort polygons by area descending. The largest in each overlap
    cluster keeps its full geometry; later (smaller) ones have the union of
    all already-kept neighbors subtracted from them. End state: pairwise
    intersections of polygons all have zero area, so summing
    parking_area_sqft (and the derived classifier surface_area_sqft and
    effective_surface_land_value) yields the correct deduplicated totals.

    Per-polygon identity (popups, classification, parcel link) is preserved.

    Polygons that get fully covered by larger neighbors become empty and
    are dropped.
    """
    if len(parking) < 2:
        return parking
    g = _project_for_area(parking).reset_index(drop=True)
    # Sort largest first so we keep the most-coverage polygons intact
    g["_area_m2"] = g.geometry.area
    order = g["_area_m2"].sort_values(ascending=False).index.tolist()
    sindex = g.sindex

    new_geom = list(g.geometry)
    overlaps_carved = 0
    overlap_area_m2 = 0.0
    for idx_pos, i in enumerate(order):
        gi = new_geom[i]
        if gi is None or gi.is_empty:
            continue
        # Find candidates that come EARLIER in the order (i.e. larger or equal)
        hits = list(sindex.intersection(gi.bounds))
        earlier = [j for j in hits if j != i
                   and order.index(j) < idx_pos
                   and new_geom[j] is not None and not new_geom[j].is_empty]
        if not earlier:
            continue
        # Subtract the union of all earlier polygons from this one
        from shapely.ops import unary_union
        prior_union = unary_union([new_geom[j] for j in earlier])
        if not gi.intersects(prior_union):
            continue
        inter_area = gi.intersection(prior_union).area
        if inter_area < min_overlap_sqft / 10.7639:
            continue
        carved = gi.difference(prior_union)
        # difference() can return GeometryCollections with non-polygonal bits;
        # keep only the polygonal pieces.
        if carved.geom_type == "GeometryCollection":
            from shapely.geometry import MultiPolygon, Polygon
            polys = [g for g in carved.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            carved = MultiPolygon([p for p in polys for p in (p.geoms if p.geom_type == "MultiPolygon" else [p])]) if polys else carved
        new_geom[i] = carved
        overlaps_carved += 1
        overlap_area_m2 += inter_area

    g["geometry"] = new_geom
    # Drop now-empty geoms
    g = g[~g.geometry.is_empty & g.geometry.notna()].copy()
    # Recompute areas + scale ratio so derived dollar/surface columns shrink
    # proportionally (matters when the input already has classifier columns —
    # the inline ETL path runs dedup BEFORE the classifier so those columns
    # don't exist yet; the offline-patcher path does have them).
    g["_area_m2_new"] = g.geometry.area
    old_sqft = g["_area_m2"] * 10.7639
    new_sqft = g["_area_m2_new"] * 10.7639
    ratio = (new_sqft / old_sqft).where(old_sqft > 0, 1.0)
    SCALABLE_DERIVED = [
        "surface_area_sqft",
        "surface_area_acres",
        "effective_surface_land_value",
        "estimated_parking_land_value",
    ]
    for col in SCALABLE_DERIVED:
        if col in g.columns:
            g[col] = (g[col].astype(float) * ratio).astype(float)
    g["parking_area_sqft"] = new_sqft
    g["parking_area_acres"] = new_sqft / 43560.0

    overlap_sqft = overlap_area_m2 * 10.7639
    print(f"  Dedup: carved overlap from {overlaps_carved:,} polygons; "
          f"removed {overlap_sqft/43560.0:.2f} acres of duplicate land")

    g = g.drop(columns=["_area_m2", "_area_m2_new"]).to_crs(4326)
    return g


def spatial_join_to_parcels(
    parking: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Spatially join parking polygons to parcel data.

    For each parking polygon, find the dominant parcel (largest intersection area)
    and compute:
    - land_value_per_sqft
    - estimated_parking_land_value
    - parcel_category
    """
    print(f"  Joining {len(parking):,} parking polygons to {len(parcels):,} parcels...")

    # Normalize CRS
    if parcels.crs is None or parcels.crs.to_epsg() != 4326:
        parcels = parcels.to_crs(4326)

    # Reproject both to the same local UTM zone for accurate (equal-area) area
    # calculations and a valid spatial join (must share one CRS).
    area_crs = parcels.estimate_utm_crs()
    parking_proj = parking.to_crs(area_crs).reset_index(drop=True)
    parcels_proj = parcels.to_crs(area_crs).reset_index(drop=True)

    # Add spatial index for fast lookup
    parcel_sindex = parcels_proj.sindex

    # Identify key parcel fields
    parcel_id_col = identify_parcel_id_column(parcels)

    # Identify land value field
    land_value_candidates = ["REALLANDVA", "land_value", "current_full_land_value"]
    land_value_col = next(
        (c for c in land_value_candidates if c in parcels.columns), None
    )

    # Identify category field
    category_candidates = [
        "property_land_use_refined", "property_category_refined",
        "property_land_use_category", "PROPERTY_CATEGORY",
        "land_use", "LAND_USE", "land_use_code",
    ]
    category_col = next((c for c in category_candidates if c in parcels.columns), None)

    # Identify exemption flag field
    exemption_col = next((c for c in parcels.columns if c.lower() == "exemption_flag"), None)

    # NOTE: We always compute parcel area from the projected geometry rather than
    # reading any stored area column.  Stored columns (e.g. Morgantown's area_sqft)
    # can carry unit / projection errors that are ~9× too small, which would inflate
    # the land-value-per-sqft rate and produce wildly over-stated parking valuations.

    print(f"  Using: land_value={land_value_col}, category={category_col}, "
          f"id={parcel_id_col}, exemption={exemption_col}")

    # For each parking polygon, find dominant parcel by intersection area
    # Also track exemption: flag any lot that intersects an exempt parcel OR has
    # less than 10% of its area covered by any parcel (city omitted exempt parcels).
    results = []
    for park_idx in range(len(parking_proj)):
        park_geom = parking_proj.geometry.iloc[park_idx]

        # Candidate parcels via spatial index
        candidate_idxs = list(parcel_sindex.intersection(park_geom.bounds))
        if not candidate_idxs:
            # No parcel bounding-box overlap at all → uncovered → exempt
            results.append({
                "parking_idx": park_idx,
                "parcel_id": None,
                "parcel_land_value": np.nan,
                "parcel_area_sqft": np.nan,
                "parcel_category": None,
                "is_exempt": True,
            })
            continue

        # Find dominant parcel (largest intersection area) + track exemption/coverage
        best_area = 0.0
        best_parcel = None
        total_covered_area = 0.0
        has_exempt_intersection = False
        for ci in candidate_idxs:
            parcel_geom = parcels_proj.geometry.iloc[ci]
            try:
                inter = park_geom.intersection(parcel_geom)
                inter_area = inter.area
            except Exception:
                inter_area = 0.0
            if inter_area <= 0:
                continue
            total_covered_area += inter_area
            if exemption_col is not None:
                flag = parcels_proj[exemption_col].iloc[ci]
                if flag == 1 or flag is True:
                    has_exempt_intersection = True
            if inter_area > best_area:
                best_area = inter_area
                best_parcel = ci

        if best_parcel is None:
            # Bounding boxes overlapped but no actual intersection → uncovered → exempt
            results.append({
                "parking_idx": park_idx,
                "parcel_id": None,
                "parcel_land_value": np.nan,
                "parcel_area_sqft": np.nan,
                "parcel_category": None,
                "is_exempt": True,
            })
            continue

        # Exempt if: any overlap with an exempt parcel, OR >90% of lot area has no parcel
        park_area_m2 = park_geom.area
        coverage_ratio = total_covered_area / park_area_m2 if park_area_m2 > 0 else 0.0
        is_exempt = has_exempt_intersection or (coverage_ratio < 0.10)

        parcel_row = parcels_proj.iloc[best_parcel]

        # Get parcel ID
        pid = str(parcel_row[parcel_id_col]) if parcel_id_col else str(best_parcel)

        # Get land value
        lv = float(parcel_row[land_value_col]) if land_value_col and pd.notna(parcel_row.get(land_value_col)) else np.nan

        # Always compute parcel area from the projected geometry (sqft).
        # Stored area columns can carry unit/projection errors (e.g. Morgantown's
        # area_sqft is ~9× too small), which would inflate land-value-per-sqft rates.
        pa = float(parcels_proj.geometry.iloc[best_parcel].area * 10.7639)

        # Get category
        cat = str(parcel_row[category_col]) if category_col else None

        results.append({
            "parking_idx": park_idx,
            "parcel_id": pid,
            "parcel_land_value": lv,
            "parcel_area_sqft": pa,
            "parcel_category": cat,
            "is_exempt": is_exempt,
        })

    result_df = pd.DataFrame(results).set_index("parking_idx")

    # Compute derived metrics
    parking_out = parking.copy()
    parking_out["parcel_id"] = result_df["parcel_id"].values
    parking_out["parcel_land_value"] = result_df["parcel_land_value"].values
    parking_out["parcel_area_sqft"] = result_df["parcel_area_sqft"].values
    parking_out["parcel_category"] = result_df["parcel_category"].values
    parking_out["is_exempt"] = result_df["is_exempt"].astype(bool).values

    # land_value_per_sqft = parcel land value / geometry-computed parcel area
    # (parcel_area_sqft is always derived from projected geometry, never a stored column)
    parking_out["land_value_per_sqft"] = np.where(
        parking_out["parcel_area_sqft"].values > 0,
        parking_out["parcel_land_value"].values / parking_out["parcel_area_sqft"].values,
        np.nan,
    )

    # estimated_parking_land_value = parking area × land value per sqft
    parking_out["estimated_parking_land_value"] = (
        parking_out["parking_area_sqft"] * parking_out["land_value_per_sqft"]
    ).fillna(0.0)

    print(
        f"  Joined: {parking_out['parcel_id'].notna().sum():,} / {len(parking_out):,} "
        f"parking polygons matched to parcels"
    )
    return parking_out


# ---------------------------------------------------------------------------
# Step 7: Compute city-wide totals metadata
# ---------------------------------------------------------------------------

def _overture_release_tag() -> str:
    try:
        from data.scripts.classify_parking_surface import OVERTURE_RELEASE
        return f"overture-{OVERTURE_RELEASE}"
    except Exception:
        return "overture"


def compute_parking_metadata(
    parking: gpd.GeoDataFrame,
    city_key: str,
    state: str,
    resolution_m: float,
    naip_date_range: str,
    model_id: str = "UTEL-UIUC/SegFormer-large-parking",
) -> dict:
    """Compute city-wide parking totals for the metadata JSON."""
    surface_mask = parking["source"].str.contains("ml_segmentation", na=False)
    surface = parking[surface_mask]
    source_values = sorted(set(parking["source"].dropna().astype(str).tolist()))
    ml_only = bool(source_values) and set(source_values) == {"ml_segmentation"}
    osm_only = bool(source_values) and set(source_values) == {"osm", "osm_supplement"} or set(source_values) == {"osm"}
    mixed_sources = len(source_values) > 1

    total_area_sqft = float(parking["parking_area_sqft"].sum())
    surface_area_sqft = float(surface["parking_area_sqft"].sum())
    total_value = float(parking["estimated_parking_land_value"].sum())
    matched_mask = parking["parcel_id"].notna()
    matched = parking[matched_mask]
    matched_area_sqft = float(matched["parking_area_sqft"].sum())
    matched_value = float(matched["estimated_parking_land_value"].sum())

    # Taxable / exempt split
    exempt_mask = parking["is_exempt"].fillna(False).astype(bool)
    taxable_value = float(parking.loc[~exempt_mask, "estimated_parking_land_value"].sum())
    exempt_value  = float(parking.loc[exempt_mask,  "estimated_parking_land_value"].sum())
    taxable_count = int((~exempt_mask).sum())
    exempt_count  = int(exempt_mask.sum())

    # Statistics computed over taxable lots only
    taxable = parking[~exempt_mask]
    lv_per_sqft = taxable["land_value_per_sqft"].dropna()
    mean_lv_per_sqft = float(lv_per_sqft.mean()) if len(lv_per_sqft) > 0 else 0.0

    # Surface-vs-structure classification breakdown (present only when the
    # classifier ran). The headline "underutilized surface parking" value counts
    # only confident `surface` lots, pro-rated to their effective unbuilt area.
    classification = None
    if "parking_type" in parking.columns:
        eff = parking.get("effective_surface_land_value")
        eff = eff.fillna(0) if eff is not None else pd.Series(0.0, index=parking.index)
        by_type = {}
        for t in ["surface", "structure", "uncertain", "excluded"]:
            m = parking["parking_type"] == t
            by_type[t] = {
                "count": int(m.sum()),
                "effective_surface_land_value": float(eff[m].sum()),
                "surface_area_sqft": float(parking.loc[m, "surface_area_sqft"].sum())
                if "surface_area_sqft" in parking.columns else 0.0,
            }
        classification = {
            "byType": by_type,
            "surface_estimated_land_value": by_type["surface"]["effective_surface_land_value"],
            "uncertain_estimated_land_value": by_type["uncertain"]["effective_surface_land_value"],
            "surface_feature_count": by_type["surface"]["count"],
            "structure_feature_count": by_type["structure"]["count"],
            "excluded_feature_count": by_type["excluded"]["count"],
            "uncertain_feature_count": by_type["uncertain"]["count"],
            "building_source": _overture_release_tag(),
        }

        # Parcel-context split of `surface`: the high-confidence subset is the
        # defensible headline (real off-street parking on built parcels); the
        # low-confidence subset is bare/vacant lots, open materials yards, and
        # utility/rail ROW that look like pavement but aren't off-street parking.
        if "context_confidence" in parking.columns:
            surf_m = parking["parking_type"] == "surface"
            hi = surf_m & (parking["context_confidence"] == "high")
            lo = surf_m & (parking["context_confidence"] == "low")
            classification["parcel_context_applied"] = True
            classification["surface_high_context_feature_count"] = int(hi.sum())
            classification["surface_high_context_area_sqft"] = float(
                parking.loc[hi, "surface_area_sqft"].sum()) if "surface_area_sqft" in parking.columns else 0.0
            classification["surface_high_context_land_value"] = float(eff[hi].sum())
            classification["surface_low_context_feature_count"] = int(lo.sum())
            classification["surface_low_context_area_sqft"] = float(
                parking.loc[lo, "surface_area_sqft"].sum()) if "surface_area_sqft" in parking.columns else 0.0
            classification["surface_low_context_land_value"] = float(eff[lo].sum())

    result = {
        "parkingTotals": {
            "total_area_sqft": total_area_sqft,
            "total_area_acres": total_area_sqft / 43560.0,
            "surface_area_sqft": surface_area_sqft,
            "surface_area_acres": surface_area_sqft / 43560.0,
            "matched_area_sqft": matched_area_sqft,
            "matched_area_acres": matched_area_sqft / 43560.0,
            "total_estimated_land_value": total_value,
            "matched_estimated_land_value": matched_value,
            "taxable_estimated_land_value": taxable_value,
            "exempt_estimated_land_value": exempt_value,
            "matched_feature_count": int(matched_mask.sum()),
            "taxable_feature_count": taxable_count,
            "exempt_feature_count": exempt_count,
            "mean_land_value_per_sqft": mean_lv_per_sqft,
            "feature_count": len(parking),
            "detection_sources": source_values,
            "coverage_type": (
                "ml_segmentation" if ml_only else
                "osm_only" if osm_only else
                "mixed" if mixed_sources else
                "unknown"
            ),
            "imagery_source": f"naip_{naip_date_range}" if not osm_only else None,
            "imagery_resolution_m": resolution_m if not osm_only else None,
            "model": model_id if not osm_only else None,
        }
    }
    if classification is not None:
        result["classificationTotals"] = classification
    return result


# ---------------------------------------------------------------------------
# Step 8: Upload to Azure Blob Storage
# ---------------------------------------------------------------------------

def upload_to_azure(
    local_path: Path,
    blob_name: str,
    connection_string: str,
    container: str,
    overwrite: bool,
) -> None:
    """Upload a file to Azure Blob Storage at parking/ subfolder."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. Install with: pip install azure-storage-blob"
        ) from e

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container)
    blob_client = container_client.get_blob_client(blob_name)

    with local_path.open("rb") as handle:
        blob_client.upload_blob(handle, overwrite=overwrite)
    print(f"  Uploaded: {local_path.name} -> {container}/{blob_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.city and not args.file:
        cities = ", ".join(list_cities())
        raise SystemExit(
            f"Provide --city <name> or --file <path>.\nAvailable cities: {cities}"
        )

    # Resolve city metadata
    city_meta = resolve_city(args.city) if args.city else None
    city_key = args.city or Path(args.file).stem.split("-")[0]
    state = city_meta.state if city_meta else "xx"

    # Determine output directory (isolated from existing parcel data)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("data/parking") / city_key
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    # Output filenames. Prefer the registry's canonical names (they carry the country
    # slug for non-US cities, e.g. tallinn-harju-ee-parking-lots.parquet); fall back to
    # the city-state form for --file runs without registry metadata.
    if city_meta is not None:
        parking_filename = city_meta.parking_filename
        metadata_filename = city_meta.parking_metadata_filename
    else:
        parking_filename = f"{city_key}-{state}-parking-lots.parquet"
        metadata_filename = f"{city_key}-{state}-parking-lots-metadata.json"
    parking_path = output_dir / parking_filename
    metadata_path = output_dir / metadata_filename

    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f"Step 1: Loading parcel GeoParquet for '{city_key}' (read-only)")
    print("=" * 60)
    parcel_path = resolve_parcel_path(args.city, args.file)
    print(f"  Parcel file: {parcel_path}")
    parcels = gpd.read_parquet(parcel_path)
    if parcels.crs is None:
        parcels = parcels.set_crs(4326)
    print(f"  Loaded {len(parcels):,} parcels")

    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Deriving city bounding box from parcel footprint")
    print("=" * 60)
    bbox = bbox_from_parcels(parcels)
    boundary_geometry = city_boundary_geometry(city_key, parcels)

    # -----------------------------------------------------------------------
    if args.osm_only:
        # ── OSM-only mode: skip NAIP fetch and ML inference entirely ──────
        print("=" * 60)
        print("Step 3-4: OSM-only mode – skipping NAIP fetch and ML inference")
        print("=" * 60)
        print("  Fetching OSM parking polygons as primary source...")
        parking_gdf = fetch_osm_parking(city_key, bbox=bbox)
        if parking_gdf.empty:
            raise RuntimeError(
                f"No OSM parking polygons found for '{city_key}'. "
                "Check CITY_OSM_QUERIES or try a different city."
            )
        parking_gdf["source"] = "osm"
        print(f"  OSM returned {len(parking_gdf):,} parking polygons.")

    else:
        mask_path = output_dir / "parking_mask.tif"
        tile_dir = output_dir / "naip_tiles"
        naip_path: Path | None = None

        if args.rethreshold:
            # ── Fast re-threshold mode ─────────────────────────────────
            print("=" * 60)
            print("Step 3-4: Re-thresholding saved probability map (no inference)")
            print("=" * 60)
            mask_path = rethreshold_from_probs(
                output_dir=output_dir,
                threshold=args.threshold,
                nir_strip=args.nir_strip,
            )
        elif not args.skip_inference:
            naip_path = output_dir / "naip_imagery.tif"
            if naip_path.exists() and not args.overwrite:
                print(f"Step 3: NAIP imagery already exists, skipping fetch: {naip_path}")
            else:
                print("=" * 60)
                print("Step 3: Fetching NAIP imagery via Planetary Computer STAC")
                print("=" * 60)
                # Always build a single citywide mosaic (unified pixel grid).
                # fetch_naip_imagery caches per-tile clips locally, then runs one
                # final gdalwarp to create a consistent UTM raster. Running
                # inference on a per-tile VRT (tile-first) causes each tile's
                # resampling grid to diverge, producing irregular polygon edges
                # and 5–10× over-detection — always use the unified mosaic.
                naip_path = fetch_naip_imagery(
                    bbox=bbox,
                    date_range=f"{args.naip_start}/{args.naip_end}",
                    resolution_m=args.resolution,
                    output_dir=output_dir,
                )

            print("=" * 60)
            print("Step 4: Running SegFormer-large-parking inference")
            print("=" * 60)
            mask_path = run_segformer_inference(
                naip_path=naip_path,
                output_dir=output_dir,
                use_cpu=args.cpu,
                threshold=args.threshold,
                nir_strip=args.nir_strip,
            )
        else:
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"--skip-inference specified but mask not found: {mask_path}"
                )
            print(f"Skipping inference, using existing mask: {mask_path}")

        # -----------------------------------------------------------------------
        if args.rethreshold or args.skip_inference or naip_path is not None:
            print("=" * 60)
            print("Step 5: Vectorizing mask to parking polygons")
            print("=" * 60)
            parking_gdf = vectorize_mask(mask_path, min_area_sqft=MIN_PARKING_AREA_SQFT)

            if len(parking_gdf) == 0:
                print("Warning: No parking polygons detected. Check imagery quality and mask output.")

            if args.remove_roads and len(parking_gdf):
                print("=" * 60)
                print("Step 5a: Removing OSM road overlap from parking polygons")
                print("=" * 60)
                parking_gdf = remove_road_overlap(parking_gdf, bbox)

        # -----------------------------------------------------------------------
        if args.use_osm_supplement:
            print("=" * 60)
            print("Step 5b: Fetching OSM parking supplement")
            print("=" * 60)
            osm_gdf = fetch_osm_parking(city_key, bbox=bbox)
            parking_gdf = merge_parking_sources(parking_gdf, osm_gdf)

    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 5c: Clipping parking polygons to city boundary")
    print("=" * 60)
    parking_gdf = clip_parking_to_city_boundary(parking_gdf, boundary_geometry)

    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 5d: Deduplicating overlapping parking polygons")
    print("=" * 60)
    parking_gdf = deduplicate_overlapping_parking(parking_gdf)

    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 6: Spatial join to parcels")
    print("=" * 60)
    parking_gdf = spatial_join_to_parcels(parking_gdf, parcels)

    # -----------------------------------------------------------------------
    if not args.no_classify and len(parking_gdf):
        print("=" * 60)
        print("Step 6b: Classify surface vs structure (Overture buildings + OSM tags)")
        print("=" * 60)
        try:
            from data.scripts.classify_parking_surface import classify_parking
            cls_bbox = tuple(parking_gdf.to_crs(4326).total_bounds)
            parking_gdf = classify_parking(parking_gdf, cls_bbox, parcels=parcels)
            vc = parking_gdf["parking_type"].value_counts().to_dict()
            print(f"  Classified: {vc}")
        except Exception as e:
            # Never fail the whole ETL on the classifier — fall back to unclassified.
            print(f"  WARNING: surface/structure classification failed ({e}); "
                  f"writing parking without classification columns.")

    # Add city/state metadata
    parking_gdf["city"] = city_key
    parking_gdf["state"] = state

    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 7: Writing output files")
    print("=" * 60)

    # Ensure geometry is valid and CRS is set
    parking_gdf = parking_gdf[parking_gdf.geometry.notna()].copy()
    if parking_gdf.crs is None:
        parking_gdf = parking_gdf.set_crs(4326)

    # Write GeoParquet (isolated storage — parcel files untouched)
    parking_gdf.to_parquet(parking_path, index=False)
    print(f"  Parking GeoParquet: {parking_path}")
    print(f"    Features: {len(parking_gdf):,}")
    print(f"    Columns: {list(parking_gdf.columns)}")

    # Write metadata JSON
    metadata = compute_parking_metadata(
        parking=parking_gdf,
        city_key=city_key,
        state=state,
        resolution_m=args.resolution,
        naip_date_range=f"{args.naip_start}/{args.naip_end}",
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata JSON: {metadata_path}")
    totals = metadata["parkingTotals"]
    print(f"    Total area: {totals['total_area_acres']:.1f} acres")
    print(f"    Est. land value: ${totals['total_estimated_land_value']:,.0f}")

    # -----------------------------------------------------------------------
    if args.upload:
        if not args.connection_string:
            raise SystemExit(
                "Missing Azure connection string. "
                "Set AZURE_STORAGE_CONNECTION_STRING or pass --connection-string."
            )
        print("=" * 60)
        print("Step 8: Uploading to Azure Blob Storage (parking/ subfolder)")
        print("=" * 60)
        # Store under parking/ subfolder to keep separate from parcel files
        upload_to_azure(
            local_path=parking_path,
            blob_name=f"parking/{parking_filename}",
            connection_string=args.connection_string,
            container=args.container,
            overwrite=args.overwrite,
        )
        upload_to_azure(
            local_path=metadata_path,
            blob_name=f"parking/{metadata_filename}",
            connection_string=args.connection_string,
            container=args.container,
            overwrite=args.overwrite,
        )
    else:
        print("Run with --upload to upload to Azure Blob Storage.")

    print("=" * 60)
    print(f"Done! Parking lot extraction complete for '{city_key}'.")
    print(f"  GeoParquet: {parking_path}")
    print(f"  Metadata:   {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
