#!/usr/bin/env python3
"""Augment a city's PARKING parquet with the same region-grouping columns the parcels carry,
so the Parking tab offers the identical region show/hide + boundary-overlay widget as the
Land Value tab (parking.ts reads the region tags off the parking features directly).

Generic counterpart to augment_houston_parking_regions.py. Each parking lot is tagged by
spatially joining its centroid against the SHIPPED overlay GeoJSONs in viz/public/ (the exact
polygons the frontend draws + filters by `name`), so parking region values match both the parcel
tags and the on-map outlines. Lots outside a layer become "(None)".

Usage:
  python data/jurisidictions/augment_parking_regions.py --city seattle \
      --fields neighborhood neighborhood_district
  python data/jurisidictions/augment_parking_regions.py --city nyc \
      --fields neighborhood borough

For each <field>, joins to viz/public/<city>-<field>-overlay.geojson and writes a parking column
named <field>. Operates in place on data/parking/<city>/<city>-<state>-parking-lots.parquet
(backup written first). Re-upload the parquet to the parquets-dev blob afterwards; no bake needed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(ROOT / "data"))
from parquet_registry import resolve_city  # noqa: E402

OVERLAY_DIR = ROOT / "viz" / "public"
NONE = "(None)"


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True, help="city key (e.g. seattle, nyc)")
    ap.add_argument("--fields", nargs="+", required=True,
                    help="region field names; each maps to viz/public/<city>-<field>-overlay.geojson")
    args = ap.parse_args()

    meta = resolve_city(args.city)
    parquet = ROOT / "data" / "parking" / args.city / meta.parking_filename
    if not parquet.exists():
        raise SystemExit(f"Missing parking parquet: {parquet}")

    backup = parquet.with_suffix(".preregions.parquet")
    if not backup.exists():
        shutil.copy2(parquet, backup)
        log(f"Backed up -> {backup.name}")
    else:
        log(f"Backup already exists -> {backup.name}")

    parking = gpd.read_parquet(parquet)
    if parking.crs is None:
        parking = parking.set_crs("EPSG:4326")
    elif parking.crs.to_epsg() != 4326:
        parking = parking.to_crs("EPSG:4326")
    log(f"Read {parquet.name}: {len(parking):,} lots")

    cent = parking.geometry.to_crs(3857).centroid.to_crs(4326)
    pts = gpd.GeoDataFrame(pd.DataFrame({"_row": range(len(parking))}),
                           geometry=cent.values, crs="EPSG:4326")

    for field in args.fields:
        path = OVERLAY_DIR / f"{args.city}-{field}-overlay.geojson"
        if not path.exists():
            raise SystemExit(f"Missing overlay layer: {path} (build it in augment_<city>_regions.py first)")
        layer = gpd.read_file(path)
        if layer.crs is None:
            layer = layer.set_crs("EPSG:4326")
        elif layer.crs.to_epsg() != 4326:
            layer = layer.to_crs("EPSG:4326")
        layer = layer[layer.geometry.notnull()].copy()
        inv = ~layer.geometry.is_valid
        if inv.any():
            layer.loc[inv, "geometry"] = layer.loc[inv, "geometry"].buffer(0)
        layer = layer[["name", "geometry"]].rename(columns={"name": "_val"})

        tagged = gpd.sjoin(pts, layer, how="left", predicate="within")
        tagged = tagged[~tagged.index.duplicated(keep="first")]
        vals = tagged["_val"].reindex(range(len(parking)))
        parking[field] = vals.map(lambda v: str(v) if pd.notna(v) else NONE).values
        matched = int((parking[field] != NONE).sum())
        vc = parking[field].value_counts(dropna=False)
        log(f"  {field}: {len(vc)} distinct, {matched:,} matched, {int(vc.get(NONE,0)):,} -> {NONE}")

    parking.to_parquet(parquet, index=False)
    log(f"Wrote {parquet.name}. Columns now: {list(parking.columns)}")


if __name__ == "__main__":
    main()
