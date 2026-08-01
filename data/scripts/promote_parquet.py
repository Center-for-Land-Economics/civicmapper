from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from azure.storage.blob import BlobServiceClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.parquet_registry import resolve_city


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a parquet from dev to prod by copying the blob."
    )
    parser.add_argument(
        "--connection-string",
        default=os.getenv("AZURE_STORAGE_CONNECTION_STRING", ""),
        help="Azure Storage connection string (AZURE_STORAGE_CONNECTION_STRING).",
    )
    parser.add_argument(
        "--dev-container",
        default=os.getenv("AZURE_DEV_CONTAINER", "parquets-dev"),
        help="Dev container name (AZURE_DEV_CONTAINER).",
    )
    parser.add_argument(
        "--prod-container",
        default=os.getenv("AZURE_PROD_CONTAINER", "parquets-prod"),
        help="Prod container name (AZURE_PROD_CONTAINER).",
    )
    parser.add_argument("--city", required=True, help="City key to promote (e.g. southbend).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination blob if it exists.",
    )
    parser.add_argument(
        "--include-pmtiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also promote <city>-<state>-parcels.pmtiles and -metadata.json when present.",
    )
    parser.add_argument(
        "--include-parking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also promote parking parquet and metadata when present.",
    )
    return parser.parse_args()


def ensure_connection_string(connection_string: str) -> None:
    if not connection_string:
        raise SystemExit(
            "Missing connection string. Set AZURE_STORAGE_CONNECTION_STRING or pass --connection-string."
        )


def wait_for_copy(blob_client, timeout_seconds: int = 300, poll_seconds: float = 1.0) -> None:
    deadline = time.time() + timeout_seconds
    while True:
        props = blob_client.get_blob_properties()
        copy_props = getattr(props, "copy", None)
        status = getattr(copy_props, "status", "success") if copy_props else "success"
        if status == "success":
            return
        if status == "pending":
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Copy timed out for {blob_client.container_name}/{blob_client.blob_name}"
                )
            time.sleep(poll_seconds)
            continue

        description = getattr(copy_props, "status_description", "Unknown copy failure")
        raise RuntimeError(
            f"Copy failed for {blob_client.container_name}/{blob_client.blob_name}: "
            f"{status} ({description})"
        )


def copy_blob(
    blob_service: BlobServiceClient,
    dev_container: str,
    prod_container: str,
    src_name: str,
    dst_name: str,
    overwrite: bool,
) -> bool:
    dev_blob = blob_service.get_blob_client(dev_container, src_name)
    prod_blob = blob_service.get_blob_client(prod_container, dst_name)

    if not dev_blob.exists():
        print(f"Skipping missing dev blob: {dev_container}/{src_name}")
        return False

    if prod_blob.exists():
        if not overwrite:
            raise SystemExit(
                f"Prod blob already exists: {prod_container}/{dst_name}. Use --overwrite to replace it."
            )
        prod_blob.delete_blob()

    print(f"Promoting {dev_container}/{src_name} -> {prod_container}/{dst_name}")
    prod_blob.start_copy_from_url(dev_blob.url)
    wait_for_copy(prod_blob)
    return True


def main() -> None:
    args = parse_args()
    ensure_connection_string(args.connection_string)

    meta = resolve_city(args.city)
    blob_service = BlobServiceClient.from_connection_string(args.connection_string)
    promoted: list[str] = []

    required_artifacts = [
        (meta.canonical_filename, meta.canonical_filename),
    ]
    optional_artifacts: list[tuple[str, str]] = []

    if args.include_pmtiles:
        pmtiles_filename = meta.canonical_filename.replace(".parquet", ".pmtiles")
        pmtiles_metadata_filename = meta.canonical_filename.replace(".parquet", "-metadata.json")
        optional_artifacts.extend(
            [
                (pmtiles_filename, pmtiles_filename),
                (pmtiles_metadata_filename, pmtiles_metadata_filename),
            ]
        )

    if args.include_parking:
        optional_artifacts.extend(
            [
                (f"parking/{meta.parking_filename}", f"parking/{meta.parking_filename}"),
                (f"parking/{meta.parking_metadata_filename}", f"parking/{meta.parking_metadata_filename}"),
            ]
        )

    for src_name, dst_name in required_artifacts:
        copied = copy_blob(
            blob_service,
            args.dev_container,
            args.prod_container,
            src_name,
            dst_name,
            args.overwrite,
        )
        if not copied:
            raise SystemExit(f"Required dev blob not found: {args.dev_container}/{src_name}")
        promoted.append(dst_name)

    for src_name, dst_name in optional_artifacts:
        copied = copy_blob(
            blob_service,
            args.dev_container,
            args.prod_container,
            src_name,
            dst_name,
            args.overwrite,
        )
        if copied:
            promoted.append(dst_name)

    print("\nPromotion complete:")
    for blob_name in promoted:
        print(f"  - {args.prod_container}/{blob_name}")

    if not promoted:
        print("No blobs were promoted.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
