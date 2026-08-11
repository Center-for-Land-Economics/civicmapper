#!/usr/bin/env python3
"""
ETL Regression Runner.

Generates ETL scripts for all cities in the golden directory using the
specified (or current) prompt bundle, then checks that:
  - The script was generated without error
  - The script contains expected field names and patterns

Each city in data/prompts/golden/ must have a {city_key}.md contribution file
and optionally a {city_key}_expected.json with field expectations.

Usage:
    python etl_regression_runner.py \
        --golden-dir data/prompts/golden \
        --output-dir /tmp/regression-output \
        --prompt-version v1.0

Output:
    {output_dir}/regression_result.json
    {output_dir}/{city_key}/generated_etl.py
    {output_dir}/{city_key}/orchestrator_result.json
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Add data/scripts to path so we can import the orchestrator
sys.path.insert(0, str(Path(__file__).parent))
from etl_ai_orchestrator import load_prompt_bundle, call_claude, parse_city_metadata


REQUIRED_PATTERNS = [
    (r"current_full_land_value", "canonical land value field"),
    (r"property_land_use_refined", "canonical refined category field"),
    (r"geometry", "geometry column"),
    (r"EPSG:4326", "WGS84 CRS"),
    (r"\.parquet", "parquet output"),
]


def check_script(script: str, expected: dict) -> list[str]:
    """Return list of failure messages (empty = pass)."""
    failures = []

    # Required patterns
    for pattern, label in REQUIRED_PATTERNS:
        if not re.search(pattern, script):
            failures.append(f"Missing {label} (pattern: {pattern!r})")

    # Optional per-city expectations
    if expected.get("must_contain"):
        for s in expected["must_contain"]:
            if s not in script:
                failures.append(f"Expected string not found: {s!r}")

    if expected.get("must_not_contain"):
        for s in expected["must_not_contain"]:
            if s in script:
                failures.append(f"Unexpected string found: {s!r}")

    # Must be reasonable length
    lines = script.splitlines()
    if len(lines) < 30:
        failures.append(f"Script too short ({len(lines)} lines — expected ≥ 30)")

    return failures


def run_city(city_md_path: Path, output_dir: Path, bundle: dict) -> dict:
    """Generate and check an ETL script for one city. Returns result dict."""
    city_markdown = city_md_path.read_text(encoding="utf-8")
    city_meta = parse_city_metadata(city_markdown)
    city_key = city_meta.get("city_key", city_md_path.stem)

    city_out = output_dir / city_key
    city_out.mkdir(parents=True, exist_ok=True)

    # Load per-city expectations (optional)
    expected_path = city_md_path.parent / f"{city_md_path.stem}_expected.json"
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else {}

    try:
        generated_script, usage = call_claude(bundle, city_markdown)
    except Exception as exc:
        return {
            "city_key": city_key,
            "passed": False,
            "skipped": False,
            "error": str(exc),
            "failed_checks": [f"API call failed: {exc}"],
        }

    # Write script
    (city_out / "generated_etl.py").write_text(generated_script, encoding="utf-8")

    failures = check_script(generated_script, expected)

    result = {
        "city_key": city_key,
        "passed": len(failures) == 0,
        "skipped": False,
        "lines": len(generated_script.splitlines()),
        "api_usage": usage,
        "failed_checks": failures,
    }

    (city_out / "orchestrator_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    status = "✅" if result["passed"] else "❌"
    print(f"  {status} {city_key}  ({result['lines']} lines)"
          + (f"  — {len(failures)} failure(s)" if failures else ""))

    return result


def main():
    parser = argparse.ArgumentParser(description="ETL Regression Runner")
    parser.add_argument("--golden-dir", required=True, help="Directory with golden city .md files")
    parser.add_argument("--output-dir", required=True, help="Directory to write results")
    parser.add_argument("--prompt-version", default="", help="Prompt bundle version (default: CURRENT_VERSION)")
    parser.add_argument("--city-filter", default="", help="Comma-separated city keys to run (default: all)")
    args = parser.parse_args()

    golden_dir = Path(args.golden_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not golden_dir.exists():
        print(f"⚠️  Golden directory not found: {golden_dir}")
        print("  Create golden/{city_key}.md files to enable regression testing.")
        result = {
            "prompt_version": args.prompt_version or "unknown",
            "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0},
            "cities": [],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        (output_dir / "regression_result.json").write_text(json.dumps(result, indent=2))
        return

    # Determine prompt version
    prompts_dir = Path(os.getenv("PROMPTS_DIR", "data/prompts"))
    version = args.prompt_version or os.getenv("PROMPT_VERSION_OVERRIDE", "").strip()
    if not version:
        version_file = prompts_dir / "CURRENT_VERSION"
        if not version_file.exists():
            print(f"❌ CURRENT_VERSION not found at {version_file}")
            sys.exit(1)
        version = version_file.read_text(encoding="utf-8").strip()

    print(f"📦 Prompt version: {version}")
    bundle = load_prompt_bundle(version)

    # Discover city files
    city_filter = {k.strip() for k in args.city_filter.split(",") if k.strip()}
    city_files = sorted(golden_dir.glob("*.md"))
    if city_filter:
        city_files = [f for f in city_files if f.stem in city_filter]

    if not city_files:
        print("⚠️  No golden city files found. Nothing to test.")
        result = {
            "prompt_version": version,
            "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0},
            "cities": [],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        (output_dir / "regression_result.json").write_text(json.dumps(result, indent=2))
        return

    print(f"\n🔁 Running regression for {len(city_files)} city/cities...\n")

    city_results = []
    for city_md in city_files:
        city_results.append(run_city(city_md, output_dir, bundle))

    passed = sum(1 for r in city_results if r["passed"])
    failed = sum(1 for r in city_results if not r["passed"] and not r.get("skipped"))
    skipped = sum(1 for r in city_results if r.get("skipped"))

    result = {
        "prompt_version": version,
        "summary": {
            "total": len(city_results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "cities": city_results,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    result_path = output_dir / "regression_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n📊 Regression complete: {passed}/{len(city_results)} passed")
    print(f"   Results: {result_path}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
