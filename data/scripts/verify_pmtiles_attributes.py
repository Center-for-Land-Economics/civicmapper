#!/usr/bin/env python3
"""
Verify that a baked city PMTiles carries each feature's OWN attribute values.

tippecanoe < 2.55 can hand a feature another feature's value (string-pool hash collisions;
see the pool guard in parquet_to_pmtiles.py). This decodes the tiles and compares them with
the source parquet:
  - parcels layer at its max zoom, matched on `link`;
  - parcels_low hexes at the finest hex zoom, matched on cell centroid against a fresh
    build_h3_aggregate of the same (remnant-filtered) parcels.
Values may differ by the guard's nudges (a few cents, a few 1e-6 acre); anything larger is
reported. Exit code 1 when mismatches are found.

  python data/scripts/verify_pmtiles_attributes.py --city elpaso [--drop-remnants]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parquet_to_pmtiles import FLOAT_DECIMALS, build_h3_aggregate, resolve_local_path  # noqa: E402

CHECK = ["current_full_land_value", "improvement_value", "land_area_acres", "full_market_value"]


def decode(pmtiles: Path, z: int, layer: str) -> list[dict]:
    out = subprocess.run(["tippecanoe-decode", "-Z", str(z), "-z", str(z), "-l", layer, str(pmtiles)],
                         capture_output=True, text=True, check=True).stdout
    feats = []
    for tile in json.loads(out)["features"]:
        for lyr in tile["features"]:
            feats.extend(lyr["features"])
    return feats


def compare(name: str, got: pd.DataFrame, want: pd.DataFrame) -> int:
    bad = 0
    for c in CHECK:
        if c not in got.columns or c not in want.columns:
            continue
        g = pd.to_numeric(got[c], errors="coerce")
        w = pd.to_numeric(want[c], errors="coerce").round(FLOAT_DECIMALS.get(c, 2))
        tol = 10.0 ** -FLOAT_DECIMALS.get(c, 2) * 10
        diff = (g - w).abs()
        m = (diff > tol) & ~(g.isna() & w.isna())
        if m.any():
            bad += int(m.sum())
            ex = pd.DataFrame({"tile": g[m], "source": w[m]}).head(5)
            print(f"  {name}.{c}: {int(m.sum())} mismatches, e.g.\n{ex.to_string()}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--drop-remnants", action="store_true")
    a = ap.parse_args()
    pq = resolve_local_path(a.city, None)
    pm = pq.with_suffix(".pmtiles")
    gdf = gpd.read_parquet(pq)
    if a.drop_remnants and "likely_remnant" in gdf.columns:
        gdf = gdf[gdf["likely_remnant"].fillna(0).astype(int) != 1]
    meta = subprocess.run(["pmtiles", "show", str(pm)], capture_output=True, text=True).stdout
    maxz = int(next(l.split(":")[1] for l in meta.splitlines() if l.startswith("max zoom")))
    bad = 0

    feats = decode(pm, maxz, "parcels")
    got = pd.DataFrame([f["properties"] for f in feats]).drop_duplicates("link").set_index("link")
    src = pd.DataFrame(gdf.drop(columns="geometry"))
    src = src[~src["link"].duplicated(keep=False)]      # generic search-page links aren't keys
    want = src.set_index("link")
    got = got[got.index.isin(want.index)]
    print(f"parcels z{maxz}: {len(got):,} of {len(want):,} uniquely linked parcels found in tiles")
    bad += compare("parcels", got, want.loc[got.index])

    hexz = 11
    feats = decode(pm, hexz, "parcels_low")
    if feats:
        from shapely.geometry import shape
        rows = []
        for f in feats:
            g = shape(f["geometry"])
            if g.geom_type != "Polygon" or len(g.exterior.coords) != 7:
                continue                                 # clipped at a tile edge
            c = g.centroid
            rows.append({**f["properties"], "k": (round(c.x, 4), round(c.y, 4))})
        got = pd.DataFrame(rows).drop_duplicates("k").set_index("k")
        h = build_h3_aggregate(gdf, resolution=11)
        cen = h.geometry.centroid
        h["k"] = list(zip(cen.x.round(4), cen.y.round(4)))
        want = pd.DataFrame(h.drop(columns="geometry")).drop_duplicates("k").set_index("k")
        got = got[got.index.isin(want.index)]
        print(f"hexes z{hexz} (r11): {len(got):,} of {len(want):,} matched by centroid (unclipped hexes)")
        bad += compare("hexes", got, want.loc[got.index])

    print("OK: every checked value matches its source" if not bad else f"FAIL: {bad} mismatched values")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
