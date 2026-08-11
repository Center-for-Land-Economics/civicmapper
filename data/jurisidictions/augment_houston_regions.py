#!/usr/bin/env python3
"""Augment the Houston parcel parquet with extra municipal-region grouping columns.

The shipped Houston bake tags each parcel with its `jurisdiction` (the city it falls in).
This adds three MORE ways to partition the city, by spatially joining the parcel centroids
against GeoJSON layers the user supplied in `Houston Shape Files/`:

    council_district     <- City Council Districts.geojson   (DISTRICT, e.g. "District A")
    super_neighborhood   <- Super Neighborhoods.geojson       (SNBNAME)
    civic_club           <- Civic Clubs.geojson               (CivicName)

Each becomes a categorical column (parcels outside the layer -> "(None)") so the frontend can
offer them as alternative "region groups" alongside jurisdiction, and the PMTiles bake can tag
each hex with its dominant value per field (H3_CATEGORICAL_FIELDS in parquet_to_pmtiles.py).

Operates in place on data/jurisidictions/data/houston/houston-tx-parcels.parquet (a backup copy
is written first). Re-run parquet_to_pmtiles.py afterwards to rebuild the tiles + metadata.

This is the one-off counterpart to the same logic folded into run_houston_proto.py for future
canonical ETL runs (the spatial-join pattern mirrors run_houston_proto.py:258-263).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
HOUSTON = HERE / "data" / "houston"
PARQUET = HOUSTON / "houston-tx-parcels.parquet"
SHAPES = HOUSTON / "Houston Shape Files"
BACKUP = HOUSTON / "backup_full_harris_2026-06-16" / "houston-tx-parcels.no-unincorp.preregions.parquet"
NONE = "(None)"

# (geojson filename, source column, new parcel column, value transform)
LAYERS = [
    ("City Council Districts.geojson", "DISTRICT", "council_district", lambda v: f"District {v}"),
    ("Super Neighborhoods.geojson", "SNBNAME", "super_neighborhood", lambda v: str(v)),
    ("Civic Clubs.geojson", "CivicName", "civic_club", lambda v: str(v)),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    if not PARQUET.exists():
        raise SystemExit(f"Missing parquet: {PARQUET}")

    if not BACKUP.exists():
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PARQUET, BACKUP)
        log(f"Backed up current parquet -> {BACKUP}")
    else:
        log(f"Backup already exists, leaving it untouched -> {BACKUP}")

    log(f"Reading {PARQUET} ...")
    parcel = gpd.read_parquet(PARQUET)
    log(f"  {len(parcel):,} parcels, crs={parcel.crs}")

    # Centroids in a projected CRS, back to 4326 — same approach as the jurisdiction join.
    log("Computing parcel centroids ...")
    cent = parcel.geometry.to_crs(3857).centroid.to_crs(4326)
    pts = gpd.GeoDataFrame(
        pd.DataFrame({"_row": range(len(parcel))}), geometry=cent.values, crs="EPSG:4326"
    )

    for fname, src_col, dst_col, transform in LAYERS:
        path = SHAPES / fname
        log(f"\nJoining {dst_col} from {fname} ...")
        layer = gpd.read_file(path)
        # Some of these files declare EPSG:4326 but actually carry Web-Mercator (meter)
        # coordinates (e.g. Civic Clubs), so to_crs(4326) would be a silent no-op and match
        # nothing. Detect by bounds: lon/lat must be within [-180,180]/[-90,90].
        minx, miny, maxx, maxy = layer.total_bounds
        if max(abs(minx), abs(maxx)) > 180 or max(abs(miny), abs(maxy)) > 90:
            log(f"  bounds {layer.total_bounds} look like Web Mercator — overriding CRS to 3857")
            layer = layer.set_crs("EPSG:3857", allow_override=True)
        elif layer.crs is None:
            layer = layer.set_crs("EPSG:4326")
        layer = layer.to_crs("EPSG:4326")
        layer = layer[layer.geometry.notnull()].copy()
        # Repair self-intersecting polygons (e.g. council districts A/B/E) with buffer(0)
        # instead of dropping them — same fix the parcel ETL applies.
        inv = ~layer.geometry.is_valid
        if inv.any():
            log(f"  repairing {int(inv.sum())} invalid geometries with buffer(0)")
            layer.loc[inv, "geometry"] = layer.loc[inv, "geometry"].buffer(0)
        layer = layer[layer.geometry.notnull() & layer.geometry.is_valid].copy()
        layer = layer[[src_col, "geometry"]].rename(columns={src_col: "_val"})

        tagged = gpd.sjoin(pts, layer, how="left", predicate="within")
        # a centroid can match >1 (overlapping boundary slivers) -> keep first
        tagged = tagged[~tagged.index.duplicated(keep="first")]
        vals = tagged["_val"].reindex(range(len(parcel)))
        col = vals.map(lambda v: transform(v) if pd.notna(v) else NONE)
        parcel[dst_col] = col.values

        vc = parcel[dst_col].value_counts(dropna=False)
        non_none = int((parcel[dst_col] != NONE).sum())
        log(f"  {dst_col}: {len(vc)} distinct, {non_none:,} parcels matched, "
            f"{int(vc.get(NONE, 0)):,} -> {NONE}")
        log(f"  top: {vc.head(6).to_dict()}")

    log(f"\nWriting {PARQUET} ...")
    parcel.to_parquet(PARQUET, index=False)
    log(f"Done. Columns now: {list(parcel.columns)}")


if __name__ == "__main__":
    main()
