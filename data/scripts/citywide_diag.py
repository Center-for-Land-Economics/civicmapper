#!/usr/bin/env python3
"""
Diagnose a citywide model run at specific labelled squares: is the NAIP there, did the model see it,
what probability did it give, and is the chunk logged as done? Read-only. Paste the output to Claude.

  python data/scripts/citywide_diag.py data/parking-model/laredo/finetune-dinov2-large-targeted
  python data/scripts/citywide_diag.py <run dir> --squares pilot-laredo-0038 pilot-laredo-0105
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ["pilot-laredo-0038", "pilot-laredo-0105", "pilot-laredo-0051", "pilot-laredo-0009",  # missed
           "pilot-laredo-0039", "pilot-laredo-0099"]                                            # found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--campaign", default="pilot-laredo")
    ap.add_argument("--squares", nargs="*", default=DEFAULT)
    a = ap.parse_args()
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window

    d = Path(a.run_dir)
    naip, probs, mask = d / "naip_imagery.tif", d / "parking_probs.tif", d / "parking_mask.tif"
    journal = d / "inference_chunks_done.txt"
    for p in (naip, probs, mask, journal):
        print(f"{p.name:28s} {'exists' if p.exists() else 'MISSING'}"
              + (f"  {p.stat().st_size / 2**20:,.0f} MB  modified {__import__('time').ctime(p.stat().st_mtime)}"
                 if p.exists() else ""))
    done = set()
    if journal.exists():
        lines = journal.read_text(encoding="utf-8").splitlines()
        print(f"journal header: {lines[0] if lines else '(empty)'}; {len(lines) - 1} chunk(s) logged")
        done = {tuple(int(v) for v in ln.split(",")) for ln in lines[1:] if "," in ln}
    tiles = sorted((d / "naip_tiles").glob("*.tif"))
    print(f"cached tiles: {len(tiles)}")

    sq = gpd.read_file(ROOT / "data" / "labels" / a.campaign / "squares.geojson")
    with rasterio.open(naip) as n, rasterio.open(probs) as pr, rasterio.open(mask) as mk:
        print(f"mosaic {n.width}x{n.height}, probs {pr.width}x{pr.height}, mask {mk.width}x{mk.height}")
        sqm = sq.to_crs(n.crs).set_index("id")
        for sid in a.squares:
            if sid not in sqm.index:
                print(f"{sid}: not in campaign")
                continue
            c = sqm.geometry[sid].centroid
            row, col = n.index(c.x, c.y)
            if not (0 <= row < n.height and 0 <= col < n.width):
                print(f"{sid}: outside the mosaic")
                continue
            w = Window(max(0, col - 200), max(0, row - 200), 400, 400)
            img = n.read(window=w)
            p = pr.read(1, window=w).astype(float) / 255
            m = mk.read(1, window=w)
            chunk = ((row // 6144) * 6144, (col // 6144) * 6144)
            tile_hits = []
            for t in tiles:
                with rasterio.open(t) as ts:
                    b = ts.bounds
                    if b.left <= c.x <= b.right and b.bottom <= c.y <= b.top:
                        tr, tc = ts.index(c.x, c.y)
                        tv = ts.read(1, window=Window(max(0, tc - 50), max(0, tr - 50), 100, 100))
                        tile_hits.append(f"{t.name} mean {tv.mean():.0f}")
            print(f"{sid}: px row {row} col {col} | NAIP mean {img[:3].mean():.0f}, empty {(img[0] == 0).mean():.0%} | "
                  f"prob mean {p.mean():.2f} max {p.max():.2f} | mask {m.mean():.0%} | chunk {chunk} "
                  f"{'logged' if chunk in done else 'NOT logged'} | tile: {'; '.join(tile_hits) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
