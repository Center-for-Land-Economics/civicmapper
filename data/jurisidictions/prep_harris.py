#!/usr/bin/env python3
"""Build the full-Harris-County parcel parquet for the dev-only `harris` city.

The shipped Houston bake dropped Unincorporated Harris County (to halve tile size). The full-Harris
backup still has every parcel but only the `jurisdiction` tag — it predates the council/super/civic
region columns. This takes that full backup and re-adds those 3 region columns by spatially joining
parcel centroids to the same Houston Shape Files used by augment_houston_regions.py, writing

    data/jurisidictions/data/harris/harris-tx-parcels.parquet

(jurisdiction + land_area_acres + the value/category columns are already present). Then bake it like
any other city:

    python data/scripts/rebake_pmtiles_cities.py --only harris

which globs this parquet, bakes harris-tx-parcels.pmtiles + metadata (the land_area_acres fallback
applies if needed), uploads, and bumps pmtilesVersion. The `harris` city is dev-only (cities.ts), so
it never appears in the picker and won't load on a deployed host.

Mirrors augment_houston_regions.py's join (same LAYERS / CRS handling / "(None)" convention).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
HOUSTON = HERE / "data" / "houston"
SRC = HOUSTON / "backup_full_harris_2026-06-16" / "houston-tx-parcels.parquet"  # full county
SHAPES = HOUSTON / "Houston Shape Files"
OUT_DIR = HERE / "data" / "harris"
OUT = OUT_DIR / "harris-tx-parcels.parquet"
NONE = "(None)"

# (geojson filename, source column, new parcel column, value transform) — identical to
# augment_houston_regions.py so harris's region tags match Houston's.
LAYERS = [
    ("City Council Districts.geojson", "DISTRICT", "council_district", lambda v: f"District {v}"),
    ("Super Neighborhoods.geojson", "SNBNAME", "super_neighborhood", lambda v: str(v)),
    ("Civic Clubs.geojson", "CivicName", "civic_club", lambda v: str(v)),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing full-Harris source: {SRC}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Reading {SRC} ...")
    parcel = gpd.read_parquet(SRC)
    log(f"  {len(parcel):,} parcels, crs={parcel.crs}")
    if "council_district" in parcel.columns:
        log("  (council_district already present — re-joining anyway to be safe)")

    log("Computing parcel centroids ...")
    cent = parcel.geometry.to_crs(3857).centroid.to_crs(4326)
    pts = gpd.GeoDataFrame(
        pd.DataFrame({"_row": range(len(parcel))}), geometry=cent.values, crs="EPSG:4326"
    )

    for fname, src_col, dst_col, transform in LAYERS:
        path = SHAPES / fname
        log(f"\nJoining {dst_col} from {fname} ...")
        layer = gpd.read_file(path)
        # Some files declare 4326 but carry Web-Mercator meters (e.g. Civic Clubs); detect by bounds.
        minx, miny, maxx, maxy = layer.total_bounds
        if max(abs(minx), abs(maxx)) > 180 or max(abs(miny), abs(maxy)) > 90:
            log(f"  bounds {layer.total_bounds} look like Web Mercator — overriding CRS to 3857")
            layer = layer.set_crs("EPSG:3857", allow_override=True)
        elif layer.crs is None:
            layer = layer.set_crs("EPSG:4326")
        layer = layer.to_crs("EPSG:4326")
        layer = layer[layer.geometry.notnull()].copy()
        inv = ~layer.geometry.is_valid
        if inv.any():
            log(f"  repairing {int(inv.sum())} invalid geometries with buffer(0)")
            layer.loc[inv, "geometry"] = layer.loc[inv, "geometry"].buffer(0)
        layer = layer[layer.geometry.notnull() & layer.geometry.is_valid].copy()
        layer = layer[[src_col, "geometry"]].rename(columns={src_col: "_val"})

        tagged = gpd.sjoin(pts, layer, how="left", predicate="within")
        tagged = tagged[~tagged.index.duplicated(keep="first")]
        vals = tagged["_val"].reindex(range(len(parcel)))
        parcel[dst_col] = vals.map(lambda v: transform(v) if pd.notna(v) else NONE).values

        vc = parcel[dst_col].value_counts(dropna=False)
        non_none = int((parcel[dst_col] != NONE).sum())
        log(f"  {dst_col}: {len(vc)} distinct, {non_none:,} parcels matched, "
            f"{int(vc.get(NONE, 0)):,} -> {NONE}")

    log(f"\nWriting {OUT} ...")
    parcel.to_parquet(OUT, index=False)
    log(f"Done. Columns now: {list(parcel.columns)}")
    log("\nNext: python data/scripts/rebake_pmtiles_cities.py --only harris")


if __name__ == "__main__":
    main()
