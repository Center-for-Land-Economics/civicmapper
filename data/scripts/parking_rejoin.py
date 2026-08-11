#!/usr/bin/env python3
"""
parking_rejoin.py — Re-run the spatial join for existing parking GeoParquets.

Reads each city's existing parking-lots.parquet (which already has ML-detected
polygon geometry from a previous run), re-runs spatial_join_to_parcels() with
the updated exemption-detection logic, recomputes metadata, and overwrites the
parquet + metadata JSON in-place.

No imagery fetch or ML inference is performed — only steps 6 and 7 of the
normal pipeline are executed.

Usage:
    python3 data/scripts/parking_rejoin.py                  # all cities
    python3 data/scripts/parking_rejoin.py --city spokane   # single city
"""

import argparse
import json
import os
import sys
from pathlib import Path

import geopandas as gpd

# ── Project root / import setup ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data.parquet_registry import CITY_PARQUETS
from data.scripts.parking_lot_extraction import (
    spatial_join_to_parcels,
    compute_parking_metadata,
)

JURIS = PROJECT_ROOT / "data" / "jurisidictions" / "data"

# Direct parcel paths per city key (canonical, most-recent file)
PARCEL_PATHS: dict[str, Path] = {
    "cincinnati": JURIS / "cincinnati" / "cincinnati-oh-parcels.parquet",
    "ibx":        JURIS / "nyc-ibx-parcels.parquet",
    "morgantown": JURIS / "morgantown" / "morgantown-wv-parcels.parquet",
    "nyc":        JURIS / "nyc"        / "nyc-ny-parcels.parquet",
    "spokane":    JURIS / "spokane"    / "spokane-wa-parcels.parquet",
    "stpaul":     JURIS / "st_paul"    / "st-paul-mn-parcels.parquet",
    "syracuse":   JURIS / "syracuse"   / "syracuse-ny-parcels.parquet",
}

# Base columns produced by ML step that must be preserved across the rejoin
BASE_COLS = ["geometry", "source", "confidence", "parking_area_sqft", "parking_area_acres"]


def upload_parking_files(
    parquet_path: Path,
    meta_path: Path,
    container: str,
    connection_string: str,
) -> None:
    """Upload parquet + metadata JSON to Azure Blob Storage with correct headers."""
    try:
        from azure.storage.blob import BlobServiceClient, ContentSettings
    except ImportError:
        print("  ✗ azure-storage-blob not installed — skipping upload.")
        return

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container)

    for local_path, content_type, cache_control in [
        (parquet_path, "application/octet-stream", None),
        (meta_path,    "application/json",         "no-cache, must-revalidate"),
    ]:
        if not local_path.exists():
            print(f"  ✗ File not found, skipping upload: {local_path.name}")
            continue

        blob_name = f"parking/{local_path.name}"
        cs = ContentSettings(
            content_type=content_type,
            cache_control=cache_control,
        )
        blob_client = container_client.get_blob_client(blob_name)
        with local_path.open("rb") as fh:
            blob_client.upload_blob(fh, overwrite=True, content_settings=cs)
        print(f"  ↑ Uploaded → {container}/{blob_name}  [{content_type}]"
              + (f"  Cache-Control: {cache_control}" if cache_control else ""))


def rejoin_city(city_key: str, upload: bool = False, container: str = "parquets-dev", connection_string: str = "") -> None:
    parking_dir = PROJECT_ROOT / "data" / "parking" / city_key
    city_meta   = CITY_PARQUETS.get(city_key)
    if city_meta is None:
        print(f"  ✗ Unknown city key: '{city_key}'. Skipping.")
        return

    state = city_meta.state

    # Locate existing parking parquet
    parquet_path = parking_dir / f"{city_key}-{state}-parking-lots.parquet"
    if not parquet_path.exists():
        # Try alt city slug (e.g. stpaul → st-paul)
        alt_slug = city_meta.city  # e.g. "st-paul"
        parquet_path = parking_dir / f"{alt_slug}-{state}-parking-lots.parquet"
    if not parquet_path.exists():
        print(f"  ✗ Parking parquet not found for '{city_key}': {parquet_path}. Skipping.")
        return

    meta_path = parquet_path.with_name(parquet_path.stem + "-metadata.json")

    print(f"\n{'='*60}")
    print(f"  Rejoining {city_key.upper()} ({state.upper()})")
    print(f"  Parquet : {parquet_path}")
    print(f"{'='*60}")

    # ── Load existing parking polygons (base columns only) ────────────────
    parking = gpd.read_parquet(parquet_path)
    present_base = [c for c in BASE_COLS if c in parking.columns]
    parking = parking[present_base].copy()
    print(f"  Loaded {len(parking):,} parking polygons.")

    # ── Load city parcels ─────────────────────────────────────────────────
    parcel_path = PARCEL_PATHS.get(city_key)
    if parcel_path is None or not parcel_path.exists():
        print(f"  ✗ Parcel file not found for '{city_key}': {parcel_path}. Skipping.")
        return
    print(f"  Parcel  : {parcel_path.relative_to(PROJECT_ROOT)}")
    parcels = gpd.read_parquet(parcel_path)
    print(f"  Loaded {len(parcels):,} parcels.")

    # ── Re-run spatial join (new is_exempt logic) ─────────────────────────
    parking = spatial_join_to_parcels(parking, parcels)
    parking["city"]  = city_key
    parking["state"] = state

    # ── Recover imagery metadata from existing JSON (or use defaults) ─────
    naip_date_range   = "2021-01-01/2023-12-31"
    resolution_m      = 0.6
    model_id          = "UTEL-UIUC/SegFormer-large-parking"
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text())
            pt = existing_meta.get("parkingTotals", {})
            src = pt.get("imagery_source", "")
            if src.startswith("naip_"):
                naip_date_range = src[len("naip_"):]
            resolution_m = pt.get("imagery_resolution_m", resolution_m)
            model_id     = pt.get("model", model_id)
        except Exception as e:
            print(f"  Warning: could not read existing metadata ({e}), using defaults.")

    # ── Compute updated metadata ──────────────────────────────────────────
    metadata = compute_parking_metadata(
        parking,
        city_key=city_key,
        state=state,
        resolution_m=resolution_m,
        naip_date_range=naip_date_range,
        model_id=model_id,
    )

    # ── Write outputs (overwrite in-place) ────────────────────────────────
    parking = parking[parking.geometry.notna()].copy()
    if parking.crs is None:
        parking = parking.set_crs(4326)

    parking.to_parquet(parquet_path, index=False)
    meta_path.write_text(json.dumps(metadata, indent=2))

    pt = metadata["parkingTotals"]
    exempt_n   = pt.get("exempt_feature_count",  "?")
    taxable_n  = pt.get("taxable_feature_count", "?")
    total_val  = pt.get("total_estimated_land_value", 0)
    taxable_v  = pt.get("taxable_estimated_land_value", 0)
    exempt_v   = pt.get("exempt_estimated_land_value",  0)

    print(f"  ✓ Written: {parquet_path.name}")
    print(f"  ✓ Written: {meta_path.name}")
    print(f"    Lots  : {taxable_n} taxable + {exempt_n} exempt = {len(parking):,} total")
    print(f"    Value : ${taxable_v:,.0f} taxable  +  ${exempt_v:,.0f} exempt  =  ${total_val:,.0f} total")

    # Upload to Azure Blob Storage if requested
    if upload:
        if not connection_string:
            print("  ✗ --upload requested but AZURE_STORAGE_CONNECTION_STRING is not set. Skipping.")
        else:
            upload_parking_files(parquet_path, meta_path, container, connection_string)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run parking spatial join for existing parquets.")
    parser.add_argument("--city", help="Single city key to rejoin (default: all cities with existing parquets).")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload parquet + metadata JSON to Azure Blob Storage after rejoin.",
    )
    parser.add_argument(
        "--container",
        default=os.getenv("AZURE_DEV_CONTAINER", "parquets-dev"),
        help="Azure Blob container name (default: parquets-dev, or AZURE_DEV_CONTAINER env var).",
    )
    parser.add_argument(
        "--connection-string",
        default=os.getenv("AZURE_STORAGE_CONNECTION_STRING", ""),
        help="Azure Storage connection string (default: AZURE_STORAGE_CONNECTION_STRING env var).",
    )
    args = parser.parse_args()

    if args.city:
        cities = [args.city]
    else:
        parking_root = PROJECT_ROOT / "data" / "parking"
        cities = sorted(d.name for d in parking_root.iterdir() if d.is_dir())

    print(f"Cities to process: {cities}")
    if args.upload:
        print(f"Upload enabled → container: {args.container}")

    for city in cities:
        try:
            rejoin_city(
                city,
                upload=args.upload,
                container=args.container,
                connection_string=args.connection_string,
            )
        except Exception as e:
            print(f"\n  ✗ ERROR processing '{city}': {e}")
            import traceback; traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()
