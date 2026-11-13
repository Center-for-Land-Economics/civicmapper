"""overture_source.py — read Overture Maps GeoParquet straight from its public S3 bucket.

Pure pyarrow (no DuckDB extensions, no Overpass), so it works anywhere pyarrow can reach
`overturemaps-us-west-2.s3.us-west-2.amazonaws.com` — including sandboxes that can't download
DuckDB's httpfs/spatial extensions or reach overpass-api.de. Honours HTTPS_PROXY.

Used by the parking pipeline as an OSM delivery path: Overture's base theme carries OSM
`amenity=parking` polygons (type=infrastructure, class=parking) with the original OSM tags in
`source_tags`, so the parking layer, the surface-vs-structure tags and the gas-station
exclusion can all come from one source. Validated 2026-11 against Dallas's Overpass-built
parking layer: 98.9% of its parking area is present in Overture.

    from data.scripts.overture_source import parking_polygons, fuel_stations, buildings
"""
from __future__ import annotations

import os
import re
import urllib.request
from functools import lru_cache

import geopandas as gpd
import pandas as pd
import shapely

BUCKET = "overturemaps-us-west-2"
REGION = "us-west-2"
LIST_URL = (f"https://{BUCKET}.s3.{REGION}.amazonaws.com/"
            "?list-type=2&prefix=release/&delimiter=/")

# OSM tags the parking classifier reads (see classify_parking_surface.attach_osm_tags).
PARKING_TAGS = ["parking", "building", "building:levels", "parking:levels", "levels", "layer"]


@lru_cache(maxsize=1)
def available_releases() -> list[str]:
    """Live Overture releases on S3 (oldest first). Overture prunes old ones."""
    with urllib.request.urlopen(LIST_URL, timeout=60) as r:
        xml = r.read().decode()
    return sorted(set(re.findall(r"release/([0-9]{4}-[0-9]{2}-[0-9]{2}\.[0-9]+)/", xml)))


def resolve_release(pinned: str | None = None) -> str:
    """Return `pinned` if it is still on S3, else the newest live release (with a warning).

    OVERTURE_RELEASE in the environment overrides the pin.
    """
    pinned = os.getenv("OVERTURE_RELEASE") or pinned
    try:
        live = available_releases()
    except Exception as e:  # noqa: BLE001 - listing is best-effort; trust the pin
        print(f"  Overture: could not list releases ({e}); using pin {pinned}")
        return pinned
    if pinned in live:
        return pinned
    newest = live[-1]
    if pinned:
        print(f"  Overture: pinned release {pinned} is no longer on S3; using newest {newest}")
    return newest


def _fs():
    import pyarrow.fs as pafs
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    kw = {"proxy_options": proxy} if proxy else {}
    return pafs.S3FileSystem(anonymous=True, region=REGION, **kw)


def read(theme: str, type_: str, bbox: tuple, *, release: str, columns: list[str],
         where=None) -> pd.DataFrame | gpd.GeoDataFrame:
    """Read one Overture theme/type inside bbox=(minx, miny, maxx, maxy).

    Features whose bbox intersects the query bbox are returned. `where` is an optional extra
    pyarrow.compute expression. If 'geometry' is requested a GeoDataFrame (EPSG:4326) is returned.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    minx, miny, maxx, maxy = bbox
    path = f"{BUCKET}/release/{release}/theme={theme}/type={type_}/"
    dset = ds.dataset(path, filesystem=_fs(), format="parquet")
    b = lambda k: pc.field("bbox", k)  # noqa: E731
    flt = (b("xmax") >= minx) & (b("xmin") <= maxx) & (b("ymax") >= miny) & (b("ymin") <= maxy)
    if where is not None:
        flt = flt & where
    tbl = dset.to_table(columns=columns, filter=flt)
    df = tbl.drop(["geometry"]).to_pandas() if "geometry" in columns else tbl.to_pandas()
    if "geometry" not in columns:
        return df
    geom = shapely.from_wkb(tbl["geometry"].to_pylist())
    return gpd.GeoDataFrame(df, geometry=geom, crs=4326)


def _tags_to_columns(gdf: gpd.GeoDataFrame, keys: list[str]) -> gpd.GeoDataFrame:
    tags = [dict(t) if t is not None else {} for t in gdf.pop("source_tags")]
    for k in keys:
        gdf[k] = [t.get(k) for t in tags]
    return gdf


def parking_polygons(bbox: tuple, release: str) -> gpd.GeoDataFrame:
    """OSM amenity=parking areas (Overture base/infrastructure, class=parking) with OSM tags.

    Street-side and lane parking are dropped, matching the Overpass path, which only asked for
    parking=surface|lane plus untagged amenity/landuse=parking areas (lane is kept, as there).
    """
    import pyarrow.compute as pc
    gdf = read("base", "infrastructure", bbox, release=release,
               columns=["id", "class", "source_tags", "geometry"],
               where=pc.field("class") == "parking")
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = _tags_to_columns(gdf, PARKING_TAGS)
    gdf = gdf[gdf["parking"].fillna("") != "street_side"]
    return gdf.drop(columns=["class"]).rename(columns={"id": "overture_id"}).reset_index(drop=True)


FUEL_CATEGORIES = {"gas_station", "fueling_station", "truck_gas_station"}


def fuel_stations(bbox: tuple, release: str, min_confidence: float = 0.5) -> gpd.GeoDataFrame:
    """Gas stations from Overture places (stands in for OSM amenity=fuel).

    The places schema keys categories as `basic_category` + `taxonomy.primary` (the older
    `categories` struct was retired in 2026). Low-confidence places are dropped.
    """
    gdf = read("places", "place", bbox, release=release,
               columns=["basic_category", "taxonomy", "confidence", "geometry"])
    prim = [(t or {}).get("primary") for t in gdf.pop("taxonomy")]
    is_fuel = gdf["basic_category"].isin(FUEL_CATEGORIES) | pd.Series(prim, index=gdf.index).isin(FUEL_CATEGORIES)
    gdf = gdf[is_fuel & (gdf["confidence"].fillna(0) >= min_confidence)]
    return gdf[["geometry"]].reset_index(drop=True)


def buildings(bbox: tuple, release: str) -> gpd.GeoDataFrame:
    """Building footprints with height / num_floors / class / subtype."""
    return read("buildings", "building", bbox, release=release,
                columns=["height", "num_floors", "class", "subtype", "geometry"])
