from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from azure.storage.blob import BlobServiceClient

from data.parquet_registry import CITY_PARQUETS


@dataclass(frozen=True)
class Containers:
    legacy: str
    dev: str
    prod: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy legacy parquet blobs into dev/prod containers using canonical names."
    )
    parser.add_argument(
        "--connection-string",
        default=os.getenv("AZURE_STORAGE_CONNECTION_STRING", ""),
        help="Azure Storage connection string (AZURE_STORAGE_CONNECTION_STRING).",
    )
    parser.add_argument(
        "--legacy-container",
        default=os.getenv("AZURE_LEGACY_CONTAINER", "public-sharing-cle"),
        help="Legacy container name (AZURE_LEGACY_CONTAINER).",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination blobs if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying blobs.",
    )
    return parser.parse_args()


def ensure_connection_string(connection_string: str) -> None:
    if not connection_string:
        raise SystemExit(
            "Missing connection string. Set AZURE_STORAGE_CONNECTION_STRING or pass --connection-string."
        )


def maybe_delete(blob_client, overwrite: bool, label: str) -> None:
    if not blob_client.exists():
        return
    if not overwrite:
        raise SystemExit(f"{label} already exists. Use --overwrite to replace it.")
    blob_client.delete_blob()


def copy_blob(source_blob, dest_blob, overwrite: bool, dry_run: bool, label: str) -> None:
    if dry_run:
        print(f"[dry-run] {label}: {source_blob.blob_name} -> {dest_blob.container_name}/{dest_blob.blob_name}")
        return
    maybe_delete(dest_blob, overwrite, label)
    dest_blob.start_copy_from_url(source_blob.url)
    print(f"Copied {label}: {source_blob.blob_name} -> {dest_blob.container_name}/{dest_blob.blob_name}")


def main() -> None:
    args = parse_args()
    ensure_connection_string(args.connection_string)

    containers = Containers(
        legacy=args.legacy_container,
        dev=args.dev_container,
        prod=args.prod_container,
    )

    blob_service = BlobServiceClient.from_connection_string(args.connection_string)
    legacy_client = blob_service.get_container_client(containers.legacy)

    for city, meta in CITY_PARQUETS.items():
        source_blob = legacy_client.get_blob_client(meta.legacy_filename)
        if not source_blob.exists():
            print(f"Skipping {city}: legacy blob {meta.legacy_filename} not found.")
            continue

        dev_blob = blob_service.get_blob_client(containers.dev, meta.canonical_filename)
        prod_blob = blob_service.get_blob_client(containers.prod, meta.canonical_filename)

        copy_blob(source_blob, dev_blob, args.overwrite, args.dry_run, f"{city} (dev)")
        copy_blob(source_blob, prod_blob, args.overwrite, args.dry_run, f"{city} (prod)")


if __name__ == "__main__":
    main()
