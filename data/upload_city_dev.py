#!/usr/bin/env python3
"""Upload a city's dev artifacts (parquet + optional PMTiles + optional parking) to Azure.

Generic version of the old upload_austin_dev.py. Resolves canonical filenames and
local paths from data/parquet_registry.py, then uploads whichever artifacts exist
locally — parcel parquet (required) plus PMTiles, PMTiles metadata, parking parquet,
and parking metadata (each optional, uploaded only if present on disk).

Reads AZURE_STORAGE_CONNECTION_STRING from data/.env (gitignored); the value is never
printed or placed on the command line. Uploads in 4 MB blocks (resilient on slow
uplinks) and skips any blob already present with a matching size (idempotent).

    python data/upload_city_dev.py austin
    python data/upload_city_dev.py rockville
    python data/upload_city_dev.py --list          # show known city keys

Blob layout (matches the API proxy routes):
    parcel + pmtiles + metadata   ->  <container>/<file>
    parking parquet + metadata    ->  <container>/parking/<file>

Prereq:  python -m pip install azure-storage-blob
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / "data" / ".env"

# parquet_registry.py sits next to this script in data/
sys.path.insert(0, str(ROOT / "data"))
from parquet_registry import list_cities, resolve_city  # noqa: E402

MB = 1024 * 1024


def load_env(path: Path) -> None:
    if not path.exists():
        sys.exit(f"Missing {path}\nCreate it with: AZURE_STORAGE_CONNECTION_STRING=...")
    import os
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip().strip('"').strip("'")


def city_dir(subdir: str, city_key: str, meta) -> Path:
    """Resolve a city's local data dir, tolerating key-named or city-named folders.

    ETL output lives under data/jurisidictions/data/<key>/ and data/parking/<key>/,
    but some cities use the registry's `city` value (e.g. st-paul vs stpaul) as the
    folder name, so try both.
    """
    base = ROOT / "data" / subdir
    for name in (city_key, meta.city):
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / city_key  # default; missing artifacts are reported below


def build_artifacts(city_key: str, meta) -> list[tuple[Path, str]]:
    """(local_path, blob_name) pairs. Parcel parquet first; others appended if present."""
    juris = city_dir("jurisidictions/data", city_key, meta)
    parking = city_dir("parking", city_key, meta)
    stem = f"{meta.city}-{meta.state}-parcels"

    candidates: list[tuple[Path, str]] = [
        (juris / meta.canonical_filename, meta.canonical_filename),          # required
        (juris / f"{stem}.pmtiles", f"{stem}.pmtiles"),                      # optional
        (juris / f"{stem}-metadata.json", f"{stem}-metadata.json"),          # optional
        (juris / meta.land_totals_filename, meta.land_totals_filename),       # optional (parking-share denominators)
        (parking / meta.parking_filename, f"parking/{meta.parking_filename}"),
        (parking / meta.parking_metadata_filename, f"parking/{meta.parking_metadata_filename}"),
    ]
    return candidates


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload a city's dev artifacts to Azure Blob Storage.")
    ap.add_argument("city", nargs="?", help="City key (e.g. austin, rockville).")
    ap.add_argument("--list", action="store_true", help="List known city keys and exit.")
    ap.add_argument("--container", default=None, help="Override target container (default: AZURE_DEV_CONTAINER or parquets-dev).")
    args = ap.parse_args()

    if args.list:
        print("Known cities:", ", ".join(list_cities()))
        return 0
    if not args.city:
        ap.error("a city key is required (or use --list)")

    try:
        meta = resolve_city(args.city)
    except ValueError as e:
        sys.exit(str(e))
    city_key = args.city.strip().lower()

    load_env(ENV_FILE)
    import os
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        sys.exit("AZURE_STORAGE_CONNECTION_STRING not found in data/.env")
    try:
        from azure.storage.blob import BlobServiceClient
        from azure.core.exceptions import ResourceNotFoundError
    except ImportError:
        sys.exit("Missing dependency. Run:  python -m pip install azure-storage-blob")

    container = args.container or os.environ.get("AZURE_DEV_CONTAINER", "parquets-dev")
    svc = BlobServiceClient.from_connection_string(
        conn,
        max_single_put_size=4 * MB,
        max_block_size=4 * MB,
        connection_timeout=300,
        read_timeout=600,
        retry_total=8,
    )
    cc = svc.get_container_client(container)

    artifacts = build_artifacts(city_key, meta)
    print(f"Uploading '{city_key}' artifacts to {container} ...")

    uploaded = 0
    parcel_local = artifacts[0][0]
    if not parcel_local.exists():
        sys.exit(f"Required parcel parquet not found: {parcel_local}\n"
                 f"Run the city ETL first.")

    for local, name in artifacts:
        if not local.exists():
            print(f"SKIP  {name}: not found locally")
            continue
        size = local.stat().st_size
        blob = cc.get_blob_client(name)
        try:
            if blob.get_blob_properties().size == size:
                print(f"SKIP  {name}: already in {container} ({size/MB:.1f} MB, same size)")
                continue
        except ResourceNotFoundError:
            pass
        print(f"UP    {name} ({size/MB:.1f} MB) -> {container} (4 MB blocks)...")
        with local.open("rb") as fh:
            blob.upload_blob(fh, overwrite=True, max_concurrency=4)
        print(f"  done -> {container}/{name}")
        uploaded += 1
    print(f"All artifacts processed. ({uploaded} uploaded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
