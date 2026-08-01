#!/usr/bin/env python3
"""
Apply generated city patches to the source files in-place.

Modifies:
  - viz/src/cities/<key>.json — the city registry file (copied from the patches dir;
    the frontend discovers it via import.meta.glob, so no shared files change)
  - data/parquet_registry.py  — adds CITY_PARQUETS entry

This script is called from the etl-promote.yml workflow after generate_frontend_files.py
has produced the patches. It applies changes directly to the checked-out repo files.

Usage:
    python apply_city_patches.py \
        --city-key portland \
        --state or \
        --patches-dir /tmp/frontend-patches
"""
import argparse
import json
import re
import sys
from pathlib import Path


def install_city_json(city_key: str, patches_dir: Path):
    """Copy the generated registry file into viz/src/cities/, validating coords."""
    src = patches_dir / "patch" / f"{city_key}.city.json"
    if not src.exists():
        raise SystemExit(f"❌ Generated city JSON not found: {src}")
    city = json.loads(src.read_text(encoding="utf-8"))
    coords = city.get("coords")
    if not (isinstance(coords, list) and len(coords) == 2
            and all(isinstance(v, (int, float)) for v in coords)):
        raise SystemExit(
            f"❌ {src.name} has no valid coords [lng, lat] — the frontend loader will "
            f"reject it. Re-run stage_city.py (it derives center from the parquet bbox) "
            f"or fill coords in by hand."
        )
    dest = Path("viz/src/cities") / f"{city_key}.json"
    dest.write_text(json.dumps(city, indent=2) + "\n", encoding="utf-8")
    print(f"✅ viz/src/cities/{city_key}.json written")


def patch_parquet_registry(city_key: str, state: str):
    path = Path("data/parquet_registry.py")
    src = path.read_text(encoding="utf-8")

    entry = (
        f'    "{city_key}": CityParquet(\n'
        f'        city="{city_key}", state="{state}", '
        f'legacy_filename="{city_key}-{state}-parcels.parquet"\n'
        f"    ),"
    )

    if f'"{city_key}"' not in src:
        src = re.sub(
            r'(    "baltimore": CityParquet\([\s\S]+?\),)',
            rf'\1\n{entry}',
            src
        )

    path.write_text(src, encoding="utf-8")
    print(f"✅ data/parquet_registry.py patched")


def main():
    parser = argparse.ArgumentParser(description="Apply city patches to source files")
    parser.add_argument("--city-key", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--patches-dir", required=True)
    # Accepted for backwards compatibility with older workflow invocations; unused —
    # everything now comes from the generated <key>.city.json.
    parser.add_argument("--display-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--has-pmtiles", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--has-parking", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--orig-field", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    patches_dir = Path(args.patches_dir)

    print(f"\n🔧 Applying patches for {args.city_key}-{args.state}")

    install_city_json(args.city_key, patches_dir)
    patch_parquet_registry(args.city_key, args.state)

    print(f"\n✅ All patches applied")


if __name__ == "__main__":
    main()
