"""Re-use the surface-parking detection to mark V&U "Parking Lot" parcels.

The Vacant & Underused tab reads each parcel's refined category (property_land_use_refined).
Historically "Parking Lot" came only from the assessor land-use text, which badly undercounts
(Fort Collins: 38 categorised parcels vs 893 detected surface lots). This post-step reuses the
*detected* surface parking (the parking-lots GeoParquet shipped for the Parking tab) and tags a
parcel "Parking Lot" when:

    - detected SURFACE lots cover >= --threshold (default 0.50) of the parcel area, AND
    - the parcel is parking-DOMINANT, not substantially developed: land value is >= --land-share
      (default 0.67) of total value, or there is no improvement value at all.

The second guard keeps the tab's "underutilized land" meaning — a store/office with a big lot
(building is the majority of value) is NOT relabelled. (Decision 2026-06-12.)

It operates on the already-shipped parcel + parking parquets (no fragile source/OSM re-run), is
idempotent, and writes the parcel parquet back in place with index=False (also avoids reintroducing
the stray __index_level_0__ int64 column that broke parking rendering). PMTiles cities must be
re-baked afterward; all cities must be re-uploaded.

Usage:
    python data/scripts/apply_parking_category.py --city pueblo [--download] [--dry-run]
    python data/scripts/apply_parking_category.py --city denver --threshold 0.5 --land-share 0.67
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

BLOB = "https://landeconomics.blob.core.windows.net/parquets-dev"
DATA = Path(__file__).resolve().parents[1]  # .../data
sys.path.insert(0, str(DATA))
from parquet_registry import resolve_city  # noqa: E402
REFINED_CANDIDATES = ("property_land_use_refined", "property_category_refined")
LAND_CANDIDATES = ("current_full_land_value", "land_value", "REALLANDVA", "land_val")
IMPR_CANDIDATES = ("improvement_value", "REALIMPROV", "bld_val", "TLLDIMPROV")


def _local_parcel_path(key: str, fname: str) -> Path:
    return DATA / "jurisidictions" / "data" / key / fname


def _local_parking_path(key: str, fname: str) -> Path:
    return DATA / "parking" / key / fname


def _ensure(path: Path, blob_url: str, download: bool) -> Path:
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(
            f"{path} not found. Re-run with --download to fetch from the blob, "
            f"or pass an explicit path."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {blob_url}\n    -> {path}")
    urllib.request.urlretrieve(blob_url, path)
    return path


def _pick(cols, candidates, label):
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(f"none of {candidates} present for {label}; cols={list(cols)[:20]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--threshold", type=float, default=0.50, help="min surface coverage of parcel")
    ap.add_argument("--land-share", type=float, default=0.67,
                    help="min land/(land+impr) to count as parking-dominant (else excluded as developed)")
    ap.add_argument("--parcels", help="explicit parcel parquet path (overrides registry/local)")
    ap.add_argument("--parking", help="explicit parking parquet path")
    ap.add_argument("--out", help="output parcel path (default: overwrite input)")
    ap.add_argument("--download", action="store_true", help="fetch missing inputs from the blob")
    ap.add_argument("--dry-run", action="store_true", help="report only, don't write")
    args = ap.parse_args()

    import geopandas as gpd
    import numpy as np

    cp = resolve_city(args.city)
    key = args.city.strip().lower()
    parcel_fname = cp.canonical_filename
    parking_fname = cp.parking_filename

    parcel_path = Path(args.parcels) if args.parcels else _ensure(
        _local_parcel_path(key, parcel_fname), f"{BLOB}/{parcel_fname}", args.download)
    parking_path = Path(args.parking) if args.parking else _ensure(
        _local_parking_path(key, parking_fname), f"{BLOB}/parking/{parking_fname}", args.download)
    out_path = Path(args.out) if args.out else parcel_path

    print(f"[{key}] parcels={parcel_path.name}  parking={parking_path.name}  "
          f"thresh={args.threshold} land_share={args.land_share}")

    parc = gpd.read_parquet(parcel_path)
    park = gpd.read_parquet(parking_path)
    cols = list(parc.columns)
    refined = _pick(cols, REFINED_CANDIDATES, "refined category")
    land_col = _pick(cols, LAND_CANDIDATES, "land value")
    impr_col = _pick(cols, IMPR_CANDIDATES, "improvement value")

    # surface lots only (legacy parquets without classification => treat all as surface)
    if "parking_type" in park.columns:
        surf = park[park["parking_type"].fillna("surface") == "surface"].copy()
    else:
        surf = park.copy()
    if surf.empty:
        print("  no surface lots — nothing to do")
        return 0

    # equal-area projection for ratios; estimate UTM from the parcels
    mcrs = parc.estimate_utm_crs()
    mp = parc.to_crs(mcrs).reset_index(drop=True)
    mp["__pidx"] = np.arange(len(mp))
    mp["__pa"] = mp.geometry.area
    mp["geometry"] = mp.geometry.apply(lambda g: g if (g is not None and g.is_valid) else (g.buffer(0) if g is not None else g))
    ms = surf.to_crs(mcrs)[["geometry"]].copy()
    ms["geometry"] = ms.geometry.apply(lambda g: g if (g is not None and g.is_valid) else (g.buffer(0) if g is not None else g))

    ov = gpd.overlay(mp[["__pidx", "__pa", "geometry"]], ms, how="intersection", keep_geom_type=True)
    ov["__ia"] = ov.geometry.area
    cov = ov.groupby("__pidx").agg(inter=("__ia", "sum"), pa=("__pa", "first")).reset_index()
    cov["coverage"] = cov["inter"] / cov["pa"].replace(0, np.nan)

    land = np.asarray(mp[land_col].fillna(0) if land_col in mp else 0, float)
    impr = np.asarray(mp[impr_col].fillna(0) if impr_col in mp else 0, float)
    total = land + impr
    land_share = np.divide(land, total, out=np.ones(len(mp)), where=total > 0)  # impr==0 => share 1

    cov_full = np.zeros(len(mp))
    cov_full[cov["__pidx"].to_numpy()] = cov["coverage"].fillna(0).to_numpy()
    parking_dominant = (cov_full >= args.threshold) & (land_share >= args.land_share)
    mark_pidx = mp.loc[parking_dominant, "__pidx"].to_numpy()

    # report
    n_cov = int((cov_full >= args.threshold).sum())
    n_excluded = n_cov - len(mark_pidx)
    before = parc[refined].value_counts(dropna=False).to_dict()
    print(f"  parcels >= {args.threshold} surface coverage: {n_cov}")
    print(f"  excluded as developed (land_share < {args.land_share}): {n_excluded}")
    print(f"  -> tagging 'Parking Lot': {len(mark_pidx)}")

    if args.dry_run:
        print("  [dry-run] not writing")
        return 0

    new = parc[refined].to_numpy(dtype=object, copy=True)
    new[mark_pidx] = "Parking Lot"
    parc[refined] = new
    after = parc[refined].value_counts(dropna=False).to_dict()
    print(f"  refined before: {before}")
    print(f"  refined after:  {after}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    parc.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
