#!/usr/bin/env python3
"""
Generate frontend patch files for a new city.

Reads a staging manifest + city contribution markdown and generates:
  - patch/{city_key}.city.json    — the complete viz/src/cities/<key>.json registry
                                    file (CityDef fields + coords from manifest center)
  - patch/parquet_registry.patch  — text showing what to add to parquet_registry.py
  - dictionaries/{city_key}.json  — data dictionary JSON (minimal starter)

The city registry and dictionary are discovered by the frontend via
import.meta.glob, so adding a city touches NO shared source files.

These patches are written as ready-to-apply text that the promotion workflow
will commit to a branch and open as a PR.

Usage:
    python generate_frontend_files.py \
        --city-file /path/to/city.md \
        --staging-manifest /path/to/_manifest.json \
        --output-dir /tmp/frontend-patches

Environment variables:
    none required
"""
import argparse
import json
import re
import textwrap
from datetime import datetime
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_city_metadata(markdown: str) -> dict:
    meta = {}
    patterns = {
        "city_key":      r'\*\*City key\*\*\s*\|\s*`([^`]+)`',
        "state":         r'\*\*State code\*\*\s*\|\s*`([^`]+)`',
        "display_name":  r'\*\*Display name\*\*\s*\|\s*`?([^`|\n]+)`?',
        "approx_parcels":r'\*\*Approx\. parcel count\*\*\s*\|\s*([^\n|]+)',
        "pmtiles":       r'\*\*PMTiles recommended\*\*\s*\|\s*([^\n|]+)',
        "parking":       r'\*\*Include parking dataset\?\*\*\s*\|\s*([^\n|]+)',
        "orig_field":    r'\*\*Source land use field\*\*\s*\|\s*`?([^`|\n]+)`?',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, markdown)
        if m:
            val = m.group(1).strip().strip("`")
            if key in ("pmtiles", "parking"):
                meta[key] = "yes" in val.lower()
            else:
                meta[key] = val
    return meta


def city_icon(display_name: str) -> str:
    """Derive a 2-letter icon from the display name."""
    parts = display_name.split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return display_name[:2].upper()


# ── Generators ─────────────────────────────────────────────────────────────────

def gen_city_json(meta: dict, has_pmtiles: bool, has_parking: bool,
                  center: list | None) -> dict:
    """The complete viz/src/cities/<key>.json registry file (CityDef fields + coords).
    The frontend discovers it via import.meta.glob — no shared-file edits needed."""
    city_key = meta["city_key"]
    state = meta["state"]
    display_name = meta.get("display_name", city_key.title())
    orig_field = meta.get("orig_field", "property_land_use_category")

    city: dict = {
        "displayName": display_name,
        "state": state,
        "filename": f"{city_key}-{state}-parcels.parquet",
    }
    if has_pmtiles:
        city["pmtilesFilename"] = f"{city_key}-{state}-parcels.pmtiles"
    if has_parking:
        city["parkingFilename"] = f"{city_key}-{state}-parking-lots.parquet"
    city["devCategoryField"] = "property_land_use_refined"
    city["origCategoryField"] = orig_field
    # coords is REQUIRED by the frontend loader; stage_city.py derives it from the
    # parquet bbox and puts it in the staging manifest as "center".
    city["coords"] = center
    return city


def gen_parquet_registry_entry(city_key: str, state: str) -> str:
    return (
        f'    "{city_key}": CityParquet(\n'
        f'        city="{city_key}", state="{state}", '
        f'legacy_filename="{city_key}-{state}-parcels.parquet"\n'
        f"    ),"
    )


def gen_dictionary_json(meta: dict, has_pmtiles: bool, has_parking: bool) -> dict:
    city_key = meta["city_key"]
    state = meta["state"]
    config: dict = {
        "usePmtiles": has_pmtiles,
        "hasParkingData": has_parking,
    }
    if has_pmtiles:
        config["pmtilesUrl"] = f"{city_key}-{state}-parcels.pmtiles"
        config["pmtilesMetadataUrl"] = f"{city_key}-{state}-parcels-metadata.json"

    return {
        "_config": config,
        "property_land_use_refined": "Property Category",
        "current_full_land_value": "Full Land Value",
        "improvement_value": "Improvement Value",
        "land_value_per_sqft": "Land value/land ft²",
        "improvement_value_per_sqft": "Improvement value/land ft²",
        "exemption_flag": "Tax Exempt",
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate frontend patch files for a new city")
    parser.add_argument("--city-file", required=True, help="Path to the city contribution .md file")
    parser.add_argument("--staging-manifest", required=True, help="Path to staging _manifest.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write patch files")
    args = parser.parse_args()

    city_markdown = Path(args.city_file).read_text(encoding="utf-8")
    staging_manifest = json.loads(Path(args.staging_manifest).read_text(encoding="utf-8"))
    meta = parse_city_metadata(city_markdown)

    city_key = meta.get("city_key")
    state = meta.get("state")
    if not city_key or not state:
        print("❌ Could not extract city_key or state from markdown")
        raise SystemExit(1)

    has_pmtiles = staging_manifest.get("has_pmtiles", False)
    has_parking = staging_manifest.get("has_parking", False)

    output_dir = Path(args.output_dir)
    (output_dir / "patch").mkdir(parents=True, exist_ok=True)
    (output_dir / "dictionaries").mkdir(exist_ok=True)

    print(f"📍 City: {meta.get('display_name', city_key)} ({city_key}-{state})")
    print(f"   PMTiles: {has_pmtiles}  |  Parking: {has_parking}")

    # 1. City registry JSON (viz/src/cities/<key>.json) — replaces the old cities.ts
    # patch; the frontend picks it up via import.meta.glob, no shared-file edits.
    center = staging_manifest.get("center")
    if not center:
        print("⚠️  Staging manifest has no 'center' — the city JSON will fail frontend "
              "validation until coords are filled in (re-run stage_city.py, or edit by hand).")
    city_json = gen_city_json(meta, has_pmtiles, has_parking, center)
    (output_dir / "patch" / f"{city_key}.city.json").write_text(
        json.dumps(city_json, indent=2) + "\n",
        encoding="utf-8"
    )
    print(f"✅ patch/{city_key}.city.json")

    # 2. parquet_registry.py entry
    reg_entry = gen_parquet_registry_entry(city_key, state)
    (output_dir / "patch" / "parquet_registry.patch").write_text(
        f"# Add the following to the CITY_PARQUETS dict in data/parquet_registry.py:\n\n"
        f"{reg_entry}\n",
        encoding="utf-8"
    )
    print("✅ patch/parquet_registry.patch")

    # 3. Dictionary JSON
    dictionary = gen_dictionary_json(meta, has_pmtiles, has_parking)
    dict_path = output_dir / "dictionaries" / f"{city_key}.json"
    dict_path.write_text(json.dumps(dictionary, indent=2), encoding="utf-8")
    print(f"✅ dictionaries/{city_key}.json")

    # 4. Summary manifest
    summary = {
        "city_key": city_key,
        "state": state,
        "display_name": meta.get("display_name", ""),
        "has_pmtiles": has_pmtiles,
        "has_parking": has_parking,
        "patches": [
            f"patch/{city_key}.city.json",
            "patch/parquet_registry.patch",
            f"dictionaries/{city_key}.json",
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (output_dir / "frontend_files_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n✅ Frontend patch files written to: {output_dir}")
    return summary


if __name__ == "__main__":
    main()
