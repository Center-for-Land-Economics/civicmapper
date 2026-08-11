#!/usr/bin/env python3
"""Augment the Houston PARKING parquet with the same municipal-region grouping columns the
parcels carry, so the Parking tab can offer the identical region show/hide + boundary-overlay
widget as the Land Value tab.

Each parking lot is tagged by spatially joining its centroid against the SHIPPED overlay GeoJSON
layers in viz/public/ (the exact polygons the frontend draws and filters by `name`). Joining to
those — rather than the raw source shapefiles — guarantees the parking region values match both
the parcel tags and the on-map outlines:

    jurisdiction        <- houston-jurisdiction-overlay.geojson        (name, e.g. "Houston")
    council_district    <- houston-council_district-overlay.geojson    (name, e.g. "District A")
    super_neighborhood  <- houston-super_neighborhood-overlay.geojson  (name)
    civic_club          <- houston-civic_club-overlay.geojson          (name)

Lots outside a layer become "(None)" (matching the parcel convention). Operates in place on the
parking parquet (a backup copy is written first). Re-upload the parquet to the parquets-dev blob
afterwards to deploy — no PMTiles rebuild needed (parking is served as GeoParquet → GeoJSON).

This mirrors augment_houston_regions.py (the parcel counterpart).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
# Parking parquets live under data/parking/<city>/ (NOT git-tracked — local build artifacts
# uploaded to the parquets-dev blob). Matches classify_parking_surface.py's default.
DEFAULT_PARQUET = REPO_ROOT / "data" / "parking" / "houston" / "houston-tx-parking-lots.parquet"
DEFAULT_OVERLAY_DIR = REPO_ROOT / "viz" / "public"
NONE = "(None)"

# (overlay geojson filename, new parking column). Source column is always "name".
LAYERS = [
    ("houston-jurisdiction-overlay.geojson", "jurisdiction"),
    ("houston-council_district-overlay.geojson", "council_district"),
    ("houston-super_neighborhood-overlay.geojson", "super_neighborhood"),
    ("houston-civic_club-overlay.geojson", "civic_club"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_PARQUET,
                    help=f"parking parquet (default {DEFAULT_PARQUET})")
    ap.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR,
                    help=f"dir with the *-overlay.geojson files (default {DEFAULT_OVERLAY_DIR})")
    args = ap.parse_args()

    parquet: Path = args.input
    if not parquet.exists():
        raise SystemExit(f"Missing parking parquet: {parquet}")

    backup = parquet.with_suffix(".preregions.parquet")
    if not backup.exists():
        shutil.copy2(parquet, backup)
        log(f"Backed up current parquet -> {backup}")
    else:
        log(f"Backup already exists, leaving it untouched -> {backup}")

    log(f"Reading {parquet} ...")
    parking = gpd.read_parquet(parquet)
    if parking.crs is None:
        parking = parking.set_crs("EPSG:4326")
    log(f"  {len(parking):,} parking lots, crs={parking.crs}")

    # Centroids in a projected CRS, back to 4326 — same approach as the parcel join.
    log("Computing parking-lot centroids ...")
    cent = parking.geometry.to_crs(3857).centroid.to_crs(4326)
    pts = gpd.GeoDataFrame(
        pd.DataFrame({"_row": range(len(parking))}), geometry=cent.values, crs="EPSG:4326"
    )

    for fname, dst_col in LAYERS:
        path = args.overlay_dir / fname
        if not path.exists():
            raise SystemExit(f"Missing overlay layer: {path}")
        log(f"\nJoining {dst_col} from {fname} ...")
        layer = gpd.read_file(path)
        # Overlays are authored in 4326, but guard against mislabeled Web-Mercator just in case
        # (mirrors augment_houston_regions.py).
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
        layer = layer[["name", "geometry"]].rename(columns={"name": "_val"})

        tagged = gpd.sjoin(pts, layer, how="left", predicate="within")
        # A centroid can match >1 (overlapping boundary slivers) -> keep first.
        tagged = tagged[~tagged.index.duplicated(keep="first")]
        vals = tagged["_val"].reindex(range(len(parking)))
        parking[dst_col] = vals.map(lambda v: str(v) if pd.notna(v) else NONE).values

        vc = parking[dst_col].value_counts(dropna=False)
        non_none = int((parking[dst_col] != NONE).sum())
        log(f"  {dst_col}: {len(vc)} distinct, {non_none:,} lots matched, "
            f"{int(vc.get(NONE, 0)):,} -> {NONE}")
        log(f"  top: {vc.head(6).to_dict()}")

    log(f"\nWriting {parquet} ...")
    parking.to_parquet(parquet, index=False)
    log(f"Done. Columns now: {list(parking.columns)}")
    log("\nNext: re-upload this parquet to the parquets-dev blob (see upload_city_dev.py / data/.env SAS).")


if __name__ == "__main__":
    main()
