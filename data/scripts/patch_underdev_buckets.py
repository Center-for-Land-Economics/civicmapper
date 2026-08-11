"""Patch existing PMTiles metadata JSONs with the Underdeveloped improvement-share
breakdown (Underdeveloped_lt10 / _10_25 / _25_50 dollar totals + parcel counts)
without re-baking tiles.

For each *-metadata.json blob in the container that has underutilizedTotals, the
script downloads the sibling parquet (same name minus -metadata.json, plus
.parquet), computes the buckets exactly like parquet_to_pmtiles.py, merges the
new keys into the JSON, and re-uploads it. Data endpoints are NOCACHE, so the
patched metadata is live immediately.

Usage:
  export AZURE_STORAGE_CONNECTION_STRING=...   # UNQUOTED
  python patch_underdev_buckets.py --container parquets-dev [--city houston-tx] [--dry-run]
"""
import argparse
import io
import json
import os
import sys

import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient

CANDIDATE_CATEGORY_FIELDS = ["property_land_use_refined", "refined_category"]
# Same precedence the frontend uses (viz/src/main.ts field mapping).
LAND_FIELDS = ["REALLANDVA", "current_full_land_value", "land_value"]
IMPR_FIELDS = ["REALIMPROV", "improvement_value"]
BUCKETS = [("lt10", 0, 10), ("10_25", 10, 25), ("25_50", 25, 101)]


def compute_buckets(parquet_bytes: bytes, category_field: str, land_field: str, impr_field: str):
    table = pq.read_table(
        io.BytesIO(parquet_bytes),
        columns=[category_field, land_field, impr_field, "exemption_flag"],
    )
    df = table.to_pandas()
    exempt_ok = df["exemption_flag"] == 0
    ud = df[(df[category_field] == "Underdeveloped") & exempt_ok]
    land = ud[land_field].astype(float)
    impr = ud[impr_field].astype(float)
    total = land + impr
    impr_pct = (impr / total.where(total > 0)) * 100
    out = {}
    for key, lo, hi in BUCKETS:
        mask = ((impr_pct >= lo) & (impr_pct < hi)).fillna(False)
        out[f"Underdeveloped_{key}"] = float(land[mask].sum())
        out[f"Underdeveloped_{key}_count"] = int(mask.sum())
    return out


def parquet_schema_fields(parquet_bytes: bytes):
    return set(pq.read_schema(io.BytesIO(parquet_bytes)).names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="parquets-dev")
    ap.add_argument("--city", action="append", help="Blob prefix filter, e.g. houston-tx (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip().strip('"')
    if not conn:
        sys.exit("AZURE_STORAGE_CONNECTION_STRING not set")

    svc = BlobServiceClient.from_connection_string(conn)
    container = svc.get_container_client(args.container)

    metadata_blobs = [
        b.name for b in container.list_blobs()
        if b.name.endswith("-metadata.json") and "/" not in b.name
    ]
    if args.city:
        metadata_blobs = [n for n in metadata_blobs if any(n.startswith(c) for c in args.city)]

    print(f"{len(metadata_blobs)} metadata blob(s) in {args.container}: {metadata_blobs}")

    for meta_name in metadata_blobs:
        parquet_name = meta_name.replace("-metadata.json", ".parquet")
        print(f"\n── {meta_name}")

        meta = json.loads(container.download_blob(meta_name).readall())
        totals = meta.get("underutilizedTotals")
        if not isinstance(totals, dict) or not totals:
            print("   no underutilizedTotals — skipping")
            continue
        if "Underdeveloped_lt10" in totals:
            print("   already patched — skipping")
            continue

        try:
            blob_props = container.get_blob_client(parquet_name).get_blob_properties()
        except Exception:
            print(f"   sibling parquet {parquet_name} not found — skipping")
            continue
        print(f"   downloading {parquet_name} ({blob_props.size / 1e6:.1f} MB)…")
        parquet_bytes = container.download_blob(parquet_name).readall()

        fields = parquet_schema_fields(parquet_bytes)
        category_field = next((f for f in CANDIDATE_CATEGORY_FIELDS if f in fields), None)
        land_field = next((f for f in LAND_FIELDS if f in fields), None)
        impr_field = next((f for f in IMPR_FIELDS if f in fields), None)
        if not category_field or not land_field or not impr_field or "exemption_flag" not in fields:
            print(f"   missing needed columns (category={category_field} land={land_field} impr={impr_field}) — skipping")
            continue

        buckets = compute_buckets(parquet_bytes, category_field, land_field, impr_field)
        check = sum(buckets[f"Underdeveloped_{k}"] for k, _, _ in BUCKETS)
        print(f"   buckets: {buckets}")
        print(f"   bucket sum ${check:,.0f} vs metadata Underdeveloped ${totals.get('Underdeveloped', 0):,.0f}")

        totals.update(buckets)
        if args.dry_run:
            print("   dry-run — not uploading")
            continue
        container.upload_blob(meta_name, json.dumps(meta), overwrite=True, content_type="application/json")
        print("   ✅ uploaded")


if __name__ == "__main__":
    main()
