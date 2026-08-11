from __future__ import annotations

import argparse
import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient

from data.parquet_registry import CITY_PARQUETS, list_cities, resolve_city


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload local parquet files to Azure Blob Storage using canonical names."
    )
    parser.add_argument(
        "--connection-string",
        default=os.getenv("AZURE_STORAGE_CONNECTION_STRING", ""),
        help="Azure Storage connection string (AZURE_STORAGE_CONNECTION_STRING).",
    )
    parser.add_argument(
        "--container",
        default=os.getenv("AZURE_DEV_CONTAINER", "parquets-dev"),
        help="Target container name (defaults to AZURE_DEV_CONTAINER).",
    )
    parser.add_argument(
        "--city",
        help="City key to upload (e.g. southbend). Use --all to upload all cities.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload all known cities using data/output/final/<city>.parquet.",
    )
    parser.add_argument(
        "--file",
        help="Override local parquet path. Defaults to data/output/final/<city>.parquet.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination blob if it exists.",
    )
    return parser.parse_args()


def ensure_connection_string(connection_string: str) -> None:
    if not connection_string:
        raise SystemExit(
            "Missing connection string. Set AZURE_STORAGE_CONNECTION_STRING or pass --connection-string."
        )


def resolve_local_path(city: str, override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    return Path("data/output/final") / f"{city}.parquet"


def upload_file(container_client, local_path: Path, blob_name: str, overwrite: bool) -> None:
    if not local_path.exists():
        raise SystemExit(f"Local parquet not found: {local_path}")
    blob_client = container_client.get_blob_client(blob_name)
    with local_path.open("rb") as handle:
        blob_client.upload_blob(handle, overwrite=overwrite)
    print(f"Uploaded {local_path} -> {container_client.container_name}/{blob_name}")


def main() -> None:
    args = parse_args()
    ensure_connection_string(args.connection_string)

    if not args.city and not args.all:
        cities = ", ".join(list_cities())
        raise SystemExit(f"Provide --city <name> or --all. Available: {cities}")

    blob_service = BlobServiceClient.from_connection_string(args.connection_string)
    container_client = blob_service.get_container_client(args.container)

    if args.all:
        for city in list_cities():
            meta = CITY_PARQUETS[city]
            local_path = resolve_local_path(city, None)
            upload_file(container_client, local_path, meta.canonical_filename, args.overwrite)
        return

    meta = resolve_city(args.city)
    local_path = resolve_local_path(meta.city, args.file)
    upload_file(container_client, local_path, meta.canonical_filename, args.overwrite)


if __name__ == "__main__":
    main()
