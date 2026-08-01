#!/usr/bin/env python3
"""
Stage a city's ETL output to Azure Blob Storage for reviewer preview.

Creates a staging slot at  parquets-dev/staging/{staging_id}/
and writes a _manifest.json describing the contents.

The staging ID is an 8-character hex hash of the PR number + timestamp,
giving 4.3 billion combinations — not guessable in practice.

Usage:
    python stage_city.py \
        --city-key portland \
        --state or \
        --parquet data/portland/portland-or-parcels.parquet \
        --pr-number 42 \
        [--pmtiles data/portland/portland-or-parcels.pmtiles] \
        [--parking data/portland/portland-or-parking-lots.parquet] \
        [--validation-json validation_result.json]

Environment variables:
    AZURE_STORAGE_CONNECTION_STRING   required
    AZURE_DEV_CONTAINER               default: parquets-dev
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings


DEV_CONTAINER = os.getenv("AZURE_DEV_CONTAINER", "parquets-dev")


def make_staging_id(pr_number: int) -> str:
    """8-char hex hash of PR number + current timestamp milliseconds."""
    raw = f"{pr_number}:{int(time.time() * 1000)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def upload_file(container_client, local_path: Path, blob_name: str, content_type: str = "application/octet-stream"):
    print(f"  ⬆️  Uploading {local_path.name} → {blob_name} ...")
    with open(local_path, "rb") as f:
        container_client.upload_blob(
            name=blob_name,
            data=f,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
    size = local_path.stat().st_size
    print(f"      ✅ {size / 1_048_576:.1f} MB")
    return size


def main():
    parser = argparse.ArgumentParser(description="Stage city ETL output to Azure Blob Storage")
    parser.add_argument("--city-key",        required=True, help="City key, e.g. portland")
    parser.add_argument("--state",           required=True, help="Two-letter state code, e.g. or")
    parser.add_argument("--parquet",         required=True, help="Path to canonical parquet file")
    parser.add_argument("--pr-number",       required=True, type=int, help="PR number (for staging ID)")
    parser.add_argument("--pmtiles",         default=None,  help="Path to PMTiles file (optional)")
    parser.add_argument("--pmtiles-meta",    default=None,  help="Path to PMTiles metadata JSON (optional)")
    parser.add_argument("--parking",         default=None,  help="Path to parking parquet (optional)")
    parser.add_argument("--parking-meta",    default=None,  help="Path to parking metadata JSON (optional)")
    parser.add_argument("--validation-json", default=None,  help="Path to validation result JSON (optional)")
    parser.add_argument("--etl-script",      default=None,  help="Path to generated ETL script (optional, stored for promotion)")
    parser.add_argument("--output-json",     default=None,  help="Write staging manifest to this path")
    args = parser.parse_args()

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        print("❌ AZURE_STORAGE_CONNECTION_STRING not set")
        sys.exit(1)

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"❌ Parquet file not found: {parquet_path}")
        sys.exit(1)

    staging_id = make_staging_id(args.pr_number)
    staging_prefix = f"staging/{staging_id}"

    print(f"\n🗂️  Staging city: {args.city_key}-{args.state}")
    print(f"   PR number : #{args.pr_number}")
    print(f"   Staging ID: {staging_id}")
    print(f"   Container : {DEV_CONTAINER}/{staging_prefix}/\n")

    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client(DEV_CONTAINER)

    artifacts: dict[str, dict] = {}
    total_bytes = 0

    # ── Map center (feeds viz/src/cities/<key>.json coords) ───────────────────
    # GeoParquet files carry a bbox in the file-level "geo" metadata; its midpoint
    # is a fine map center and costs one pyarrow schema read (no geometry scan).
    center = None
    try:
        import pyarrow.parquet as pq
        geo_meta = pq.read_schema(parquet_path).metadata.get(b"geo")
        if geo_meta:
            geo = json.loads(geo_meta)
            col = geo.get("columns", {}).get(geo.get("primary_column", "geometry"), {})
            bbox = col.get("bbox")
            if bbox and len(bbox) >= 4:
                center = [round((bbox[0] + bbox[2]) / 2, 4), round((bbox[1] + bbox[3]) / 2, 4)]
    except Exception as exc:
        print(f"  ⚠️  Could not derive map center from parquet bbox: {exc}")
    if center is None:
        try:
            import geopandas as gpd
            gdf = gpd.read_parquet(parquet_path, columns=["geometry"])
            b = gdf.to_crs(4326).total_bounds
            center = [round((b[0] + b[2]) / 2, 4), round((b[1] + b[3]) / 2, 4)]
        except Exception as exc:
            print(f"  ⚠️  Could not derive map center from geometry either: {exc}")
    if center:
        print(f"  📍 Map center [lng, lat]: {center}")

    # ── Parquet ────────────────────────────────────────────────────────────────
    parquet_blob = f"{staging_prefix}/{args.city_key}-{args.state}-parcels.parquet"
    size = upload_file(container_client, parquet_path, parquet_blob, "application/octet-stream")
    artifacts["parquet"] = {"blob": parquet_blob, "size_bytes": size}
    total_bytes += size

    # ── PMTiles ────────────────────────────────────────────────────────────────
    if args.pmtiles:
        pmtiles_path = Path(args.pmtiles)
        if pmtiles_path.exists():
            pmtiles_blob = f"{staging_prefix}/{args.city_key}-{args.state}-parcels.pmtiles"
            size = upload_file(container_client, pmtiles_path, pmtiles_blob, "application/octet-stream")
            artifacts["pmtiles"] = {"blob": pmtiles_blob, "size_bytes": size}
            total_bytes += size

            pmtiles_meta_path = Path(args.pmtiles_meta) if args.pmtiles_meta else pmtiles_path.with_name(
                pmtiles_path.name.replace(".pmtiles", "-metadata.json")
            )
            if pmtiles_meta_path.exists():
                meta_blob = f"{staging_prefix}/{args.city_key}-{args.state}-parcels-metadata.json"
                size = upload_file(container_client, pmtiles_meta_path, meta_blob, "application/json")
                artifacts["pmtiles_metadata"] = {"blob": meta_blob, "size_bytes": size}
                total_bytes += size
        else:
            print(f"  ⚠️  PMTiles file not found: {pmtiles_path} — skipping")

    # ── Parking ────────────────────────────────────────────────────────────────
    if args.parking:
        parking_path = Path(args.parking)
        if parking_path.exists():
            parking_blob = f"{staging_prefix}/{args.city_key}-{args.state}-parking-lots.parquet"
            size = upload_file(container_client, parking_path, parking_blob, "application/octet-stream")
            artifacts["parking"] = {"blob": parking_blob, "size_bytes": size}
            total_bytes += size

            parking_meta_path = Path(args.parking_meta) if args.parking_meta else parking_path.with_name(
                parking_path.name.replace(".parquet", "-metadata.json")
            )
            if parking_meta_path.exists():
                parking_meta_blob = f"{staging_prefix}/{args.city_key}-{args.state}-parking-lots-metadata.json"
                size = upload_file(container_client, parking_meta_path, parking_meta_blob, "application/json")
                artifacts["parking_metadata"] = {"blob": parking_meta_blob, "size_bytes": size}
                total_bytes += size
        else:
            print(f"  ⚠️  Parking file not found: {parking_path} — skipping")

    # ── ETL script ────────────────────────────────────────────────────────────
    if args.etl_script:
        etl_path = Path(args.etl_script)
        if etl_path.exists():
            etl_blob = f"{staging_prefix}/generated_etl.py"
            size = upload_file(container_client, etl_path, etl_blob, "text/x-python")
            artifacts["etl_script"] = {"blob": etl_blob, "size_bytes": size}
            total_bytes += size
        else:
            print(f"  ⚠️  ETL script not found: {etl_path} — skipping")

    # ── Validation result ──────────────────────────────────────────────────────
    validation_result = None
    if args.validation_json:
        val_path = Path(args.validation_json)
        if val_path.exists():
            validation_result = json.loads(val_path.read_text(encoding="utf-8"))
            val_blob = f"{staging_prefix}/_validation.json"
            size = upload_file(container_client, val_path, val_blob, "application/json")
            artifacts["validation"] = {"blob": val_blob, "size_bytes": size}
            total_bytes += size

    # ── Manifest ───────────────────────────────────────────────────────────────
    import datetime
    manifest = {
        "staging_id": staging_id,
        "city_key": args.city_key,
        "state": args.state,
        "pr_number": args.pr_number,
        "artifacts": artifacts,
        "total_bytes": total_bytes,
        "has_pmtiles": "pmtiles" in artifacts,
        "has_parking": "parking" in artifacts,
        "has_etl_script": "etl_script" in artifacts,
        "center": center,
        "validation": {
            "passed": validation_result.get("passed") if validation_result else None,
            "summary": validation_result.get("summary") if validation_result else None,
            "row_count": validation_result.get("row_count") if validation_result else None,
        },
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "review_url": f"https://dev.civicmapper.org/app.html?city=staging-{staging_id}",
    }

    manifest_json = json.dumps(manifest, indent=2)
    container_client.upload_blob(
        name=f"{staging_prefix}/_manifest.json",
        data=manifest_json.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )
    print(f"\n  ✅ Manifest written: {staging_prefix}/_manifest.json")

    if args.output_json:
        Path(args.output_json).write_text(manifest_json, encoding="utf-8")
        print(f"  ✅ Local manifest: {args.output_json}")

    print(f"\n✅ Staging complete")
    print(f"   Staging ID  : {staging_id}")
    print(f"   Total size  : {total_bytes / 1_048_576:.1f} MB")
    print(f"   Review URL  : {manifest['review_url']}")

    return manifest


if __name__ == "__main__":
    main()
