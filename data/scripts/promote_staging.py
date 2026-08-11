#!/usr/bin/env python3
"""
Promote a staging slot to the live dev dataset.

Copies all artifacts from  parquets-dev/staging/{staging_id}/
to the top-level (and parking/) paths in parquets-dev, then deletes the staging slot.

Usage:
    python promote_staging.py \
        --staging-id a1b2c3d4 \
        --city-key portland \
        --state or \
        [--output-json /tmp/promote_result.json]

Environment variables:
    AZURE_STORAGE_CONNECTION_STRING   required
    AZURE_DEV_CONTAINER               default: parquets-dev
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings


DEV_CONTAINER = os.getenv("AZURE_DEV_CONTAINER", "parquets-dev")


def _blob_prop(props, key: str, default=None):
    if hasattr(props, key):
        return getattr(props, key)
    if isinstance(props, dict):
        return props.get(key, default)
    return default


def wait_for_copy(container_client, dst_blob: str, timeout_seconds: int = 300, poll_seconds: float = 1.0):
    """Poll Azure until a server-side copy finishes or fails."""
    dst_client = container_client.get_blob_client(dst_blob)
    deadline = time.time() + timeout_seconds

    while True:
        props = dst_client.get_blob_properties()
        copy_props = _blob_prop(props, "copy")
        status = _blob_prop(copy_props, "status", "success") if copy_props else "success"

        if status == "success":
            return props
        if status == "pending":
            if time.time() >= deadline:
                raise TimeoutError(f"Copy timed out for {dst_blob}")
            time.sleep(poll_seconds)
            continue

        description = _blob_prop(copy_props, "status_description", "Unknown copy failure")
        raise RuntimeError(f"Copy failed for {dst_blob}: {status} ({description})")


def copy_blob(container_client, src_blob: str, dst_blob: str, content_type: str = "application/octet-stream") -> int:
    """Copy a blob within the same container. Returns size in bytes."""
    src_client = container_client.get_blob_client(src_blob)
    dst_client = container_client.get_blob_client(dst_blob)

    # Start the server-side copy
    src_url = src_client.url
    print(f"  📋 {src_blob}")
    print(f"     → {dst_blob}")
    dst_client.start_copy_from_url(src_url)

    # Server-side copy is asynchronous for larger blobs; do not proceed until the
    # destination blob is actually readable from the live path.
    props = wait_for_copy(container_client, dst_blob)
    size = _blob_prop(props, "size", 0)
    print(f"     ✅ {size / 1_048_576:.1f} MB")
    return size


def delete_blob(container_client, blob_name: str):
    """Delete a blob, ignoring 404."""
    try:
        container_client.get_blob_client(blob_name).delete_blob()
        print(f"  🗑️  Deleted staging blob: {blob_name}")
    except Exception as exc:
        if "BlobNotFound" in str(exc) or "404" in str(exc):
            pass  # already gone
        else:
            print(f"  ⚠️  Could not delete {blob_name}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Promote staging slot to live dev")
    parser.add_argument("--staging-id", required=True, help="8-char hex staging ID")
    parser.add_argument("--city-key",   required=True, help="City key, e.g. portland")
    parser.add_argument("--state",      required=True, help="Two-letter state code, e.g. or")
    parser.add_argument("--output-json", default=None, help="Write result JSON to this path")
    args = parser.parse_args()

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        print("❌ AZURE_STORAGE_CONNECTION_STRING not set")
        sys.exit(1)

    staging_prefix = f"staging/{args.staging_id}"
    city_state = f"{args.city_key}-{args.state}"

    print(f"\n🚀 Promoting staging slot: {staging_prefix}")
    print(f"   City: {city_state}")
    print(f"   Container: {DEV_CONTAINER}\n")

    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client(DEV_CONTAINER)

    # Read the staging manifest to know what to copy
    manifest_blob = f"{staging_prefix}/_manifest.json"
    try:
        manifest_data = container_client.get_blob_client(manifest_blob).download_blob().readall()
        manifest = json.loads(manifest_data)
    except Exception as exc:
        print(f"❌ Could not read staging manifest: {exc}")
        sys.exit(1)

    artifacts = manifest.get("artifacts", {})
    promoted = {}
    total_bytes = 0

    # ── Parquet ────────────────────────────────────────────────────────────────
    if "parquet" in artifacts:
        src = artifacts["parquet"]["blob"]
        dst = f"{city_state}-parcels.parquet"
        size = copy_blob(container_client, src, dst, "application/octet-stream")
        promoted["parquet"] = {"blob": dst, "size_bytes": size}
        total_bytes += size

    # ── PMTiles ────────────────────────────────────────────────────────────────
    if "pmtiles" in artifacts:
        src = artifacts["pmtiles"]["blob"]
        dst = f"{city_state}-parcels.pmtiles"
        size = copy_blob(container_client, src, dst, "application/octet-stream")
        promoted["pmtiles"] = {"blob": dst, "size_bytes": size}
        total_bytes += size

    if "pmtiles_metadata" in artifacts:
        src = artifacts["pmtiles_metadata"]["blob"]
        dst = f"{city_state}-parcels-metadata.json"
        size = copy_blob(container_client, src, dst, "application/json")
        promoted["pmtiles_metadata"] = {"blob": dst, "size_bytes": size}
        total_bytes += size

    # ── Parking ────────────────────────────────────────────────────────────────
    if "parking" in artifacts:
        src = artifacts["parking"]["blob"]
        dst = f"parking/{city_state}-parking-lots.parquet"
        size = copy_blob(container_client, src, dst, "application/octet-stream")
        promoted["parking"] = {"blob": dst, "size_bytes": size}
        total_bytes += size

    if "parking_metadata" in artifacts:
        src = artifacts["parking_metadata"]["blob"]
        dst = f"parking/{city_state}-parking-lots-metadata.json"
        size = copy_blob(container_client, src, dst, "application/json")
        promoted["parking_metadata"] = {"blob": dst, "size_bytes": size}
        total_bytes += size

    # ── Clean up staging slot ──────────────────────────────────────────────────
    print(f"\n🧹 Cleaning up staging slot: {staging_prefix}/")
    blobs_to_delete = container_client.list_blobs(name_starts_with=staging_prefix + "/")
    for blob in blobs_to_delete:
        delete_blob(container_client, blob.name)

    result = {
        "status": "success",
        "staging_id": args.staging_id,
        "city_key": args.city_key,
        "state": args.state,
        "promoted_artifacts": promoted,
        "total_bytes": total_bytes,
        "has_pmtiles": "pmtiles" in promoted,
        "has_parking": "parking" in promoted,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n✅ Result: {args.output_json}")

    print(f"\n✅ Promotion complete")
    print(f"   Artifacts promoted: {list(promoted.keys())}")
    print(f"   Total size: {total_bytes / 1_048_576:.1f} MB")

    return result


if __name__ == "__main__":
    main()
