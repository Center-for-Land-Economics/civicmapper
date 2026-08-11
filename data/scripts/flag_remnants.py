"""Compute geometry-derived area + a `likely_remnant` flag on a shipped parcel parquet.

Singularity towers come from a real account value sitting on a tiny sliver polygon →
value / tiny_area = absurd $/sqft, which drives an absurd 3D extrusion height. The fix is the
standard two-layer remnant treatment: flag `likely_remnant = (area_sqft < --min-area)` so the
frontend (`hideRemnants:true`) filters the detail layer and the bake (`--drop-remnants`) drops them
from the hex aggregate.

Area is computed from the geometry in an estimated local UTM CRS (robust, source-independent).

Two optional repairs for cities whose stored area was wrong (e.g. Morgantown: stored area_sqft was
~8.75x too small → land_value_per_sqft inflated ~9x citywide):
  --recompute-area      overwrite the parcel's area column with the geometry-derived sqft
  --recompute-persqft   recompute *_value_per_sqft = matching *_value / geometry area

Writes the parquet back with index=False. PMTiles cities must be re-baked (with --drop-remnants).

Usage:
    python data/scripts/flag_remnants.py --city bellingham --download
    python data/scripts/flag_remnants.py --city morgantown --download --recompute-area --recompute-persqft
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

BLOB = "https://landeconomics.blob.core.windows.net/parquets-dev"
DATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA))
from parquet_registry import resolve_city  # noqa: E402

AREA_CANDIDATES = ("land_area_sqft", "area_sqft", "parcel_area_sqft", "shape_area")
# value -> per_sqft field pairs to recompute together
PERSQFT_PAIRS = (
    ("current_full_land_value", "land_value_per_sqft"),
    ("land_value", "land_value_per_sqft"),
    ("improvement_value", "improvement_value_per_sqft"),
    ("full_market_value", "full_market_value_per_sqft"),
    ("current_tax", "current_tax_per_sqft"),
)
# Obsolete "smooth" workaround (spatial neighbor smoothing) is superseded by hex baking +
# sliver-dropping. When recomputing per-sqft, collapse smooth twins onto the corrected raw
# values so the columns stay present (dictionary/frontend) but no longer carry stale inflation.
SMOOTH_COLLAPSE = (
    ("smooth_land_value_per_sqft", "land_value_per_sqft"),
    ("smooth_full_land_value", "current_full_land_value"),
)


def _ensure(path: Path, url: str, download: bool) -> Path:
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"{path} not found; pass --download or an explicit path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}\n    -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--min-area", type=float, default=500.0, help="sqft below which a parcel is a remnant")
    ap.add_argument("--remnant-ppsf", type=float, default=None,
                    help="if set, only flag remnants that are ALSO above this land_value_per_sqft "
                         "(targets genuine singularity slivers without nuking legit tiny parcels / condos)")
    ap.add_argument("--parcels", help="explicit parcel parquet path")
    ap.add_argument("--out", help="output path (default: overwrite input)")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--recompute-area", action="store_true", help="overwrite area column with geometry sqft")
    ap.add_argument("--recompute-persqft", action="store_true", help="recompute *_value_per_sqft from geometry area")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import geopandas as gpd
    import numpy as np
    import pandas as pd

    cp = resolve_city(args.city)
    key = args.city.strip().lower()
    parcel_path = Path(args.parcels) if args.parcels else _ensure(
        DATA / "jurisidictions" / "data" / key / cp.canonical_filename,
        f"{BLOB}/{cp.canonical_filename}", args.download)
    out_path = Path(args.out) if args.out else parcel_path

    g = gpd.read_parquet(parcel_path)
    cols = list(g.columns)
    area_sqft = g.to_crs(g.estimate_utm_crs()).geometry.area.to_numpy() * 10.7639104  # m2 -> sqft

    small = area_sqft < args.min_area
    if args.remnant_ppsf is not None and "land_value_per_sqft" in cols:
        ppsf = pd.to_numeric(g["land_value_per_sqft"], errors="coerce").fillna(0).to_numpy(float)
        remnant = (small & (ppsf > args.remnant_ppsf)).astype("int8")
        cond = f"<{args.min_area:g}sqft AND >${args.remnant_ppsf:g}/sqft"
    else:
        remnant = small.astype("int8")
        cond = f"<{args.min_area:g}sqft"
    print(f"[{key}] parcels={len(g)}  geom-area sqft median={np.median(area_sqft):,.0f}  "
          f"small(<{args.min_area:g})={int(small.sum())}  likely_remnant({cond})={int(remnant.sum())}")

    area_col = next((c for c in AREA_CANDIDATES if c in cols), None)
    if args.recompute_area:
        target = area_col or "land_area_sqft"
        if area_col:
            old_med = float(pd.to_numeric(g[area_col], errors="coerce").median())
            print(f"  recompute-area: {area_col} median {old_med:,.0f} -> {np.median(area_sqft):,.0f} "
                  f"(x{np.median(area_sqft)/old_med:.2f})")
        g[target] = area_sqft

    if args.recompute_persqft:
        done = set()
        for vcol, pcol in PERSQFT_PAIRS:
            if vcol in cols and pcol in cols and pcol not in done:
                val = pd.to_numeric(g[vcol], errors="coerce").fillna(0).to_numpy(float)
                g[pcol] = np.divide(val, area_sqft, out=np.zeros(len(g)), where=area_sqft > 0)
                done.add(pcol)
                print(f"  recompute-persqft: {pcol} = {vcol}/area (max now {g[pcol].max():,.0f})")
        for smooth_col, src_col in SMOOTH_COLLAPSE:
            if smooth_col in cols and src_col in g.columns:
                g[smooth_col] = g[src_col]
                print(f"  collapse obsolete smooth: {smooth_col} <- {src_col}")

    g["likely_remnant"] = remnant

    if args.dry_run:
        print("  [dry-run] not writing")
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
