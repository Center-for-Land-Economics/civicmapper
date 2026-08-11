"""
Augment the Tallinn PARKING parquet with the `district` (linnaosa) region column so the
Parking tab offers the identical region show/hide + boundary-overlay treatment the Value
tab does (same as augment_houston_parking_regions.py for Houston).

Each parking lot is tagged by spatially joining its centroid against the SHIPPED overlay
GeoJSON (viz/public/tallinn-district-overlay.geojson) — using the shipped overlay rather
than the raw source guarantees the parking `district` values match the overlay names AND
the cadastre parcels' `district` field exactly (all are the Maa-amet EHAK asustusyksus name).

    district  <-  tallinn-district-overlay.geojson  (property "name", e.g. "Kesklinna linnaosa")

Mirrors augment_houston_parking_regions.py.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import geopandas as gpd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_PARQUET = REPO_ROOT / "data" / "parking" / "tallinn" / "tallinn-harju-ee-parking-lots.parquet"
DEFAULT_OVERLAY_DIR = REPO_ROOT / "viz" / "public"
NONE = "(None)"
LAYERS = [("tallinn-district-overlay.geojson", "district")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    args = ap.parse_args()

    parquet: Path = args.input
    if not parquet.exists():
        raise SystemExit(f"Missing parking parquet: {parquet}")

    backup = parquet.with_suffix(".preregions.parquet")
    if not backup.exists():
        shutil.copy2(parquet, backup)
        print(f"Backed up -> {backup}", flush=True)

    parking = gpd.read_parquet(parquet)
    if parking.crs is None:
        parking = parking.set_crs("EPSG:4326")
    print(f"{len(parking):,} parking lots, crs={parking.crs}", flush=True)

    cent = parking.geometry.to_crs(3857).centroid.to_crs(4326)
    pts = gpd.GeoDataFrame(
        pd.DataFrame({"_row": range(len(parking))}), geometry=cent.values, crs="EPSG:4326"
    )

    for fname, dst_col in LAYERS:
        path = args.overlay_dir / fname
        if not path.exists():
            raise SystemExit(f"Missing overlay layer: {path}")
        layer = gpd.read_file(path)
        if layer.crs is None:
            layer = layer.set_crs("EPSG:4326")
        layer = layer.to_crs("EPSG:4326")
        layer = layer[layer.geometry.notnull()].copy()
        inv = ~layer.geometry.is_valid
        if inv.any():
            layer.loc[inv, "geometry"] = layer.loc[inv, "geometry"].buffer(0)
        layer = layer[["name", "geometry"]].rename(columns={"name": "_val"})

        tagged = gpd.sjoin(pts, layer, how="left", predicate="within")
        tagged = tagged[~tagged.index.duplicated(keep="first")]
        vals = tagged["_val"].reindex(range(len(parking)))
        parking[dst_col] = vals.map(lambda v: str(v) if pd.notna(v) else NONE).values

        vc = parking[dst_col].value_counts(dropna=False)
        non_none = int((parking[dst_col] != NONE).sum())
        print(f"\n{dst_col}: {non_none:,}/{len(parking):,} tagged", flush=True)
        print(vc.to_string(), flush=True)

    parking.to_parquet(parquet, index=False)
    print(f"\nWrote {parquet}", flush=True)


if __name__ == "__main__":
    main()
