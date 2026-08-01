#!/usr/bin/env python3
"""
CivicMapper canonical parquet validation suite.

Runs 13 checks against a generated city parquet and writes a structured
JSON report. Used both by the ETL pipeline (CI) and locally.

Usage:
    python validate_city_parquet.py <parquet_path> [--output-json result.json]

Exit codes:
    0  all checks passed (warnings allowed)
    1  one or more FAIL checks
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS


# ── Thresholds ─────────────────────────────────────────────────────────────────
GEOMETRY_PRESENT_MIN   = 0.99   # >99% of rows must have geometry
LAND_VALUE_NONZERO_MIN = 0.75   # >75% of rows should have non-zero land value
CATEGORY_COVERAGE_MIN  = 0.85   # >85% of rows must have a mapped category
PARCEL_COUNT_MIN       = 1_000  # at least 1,000 rows

# Bounding box for the contiguous USA + AK/HI (generous)
USA_BOUNDS = {"minx": -180, "miny": 18, "maxx": -60, "maxy": 72}

REFINED_FIELD_CANDIDATES = [
    "property_land_use_refined",
    "property_category_refined",
]


def _check(checks: list, name: str, level: str, passed: bool, detail: str):
    """Append a check result. level: 'FAIL' or 'WARN'."""
    checks.append({"name": name, "level": level, "passed": passed, "detail": detail})
    icon = "✅" if passed else ("⚠️ " if level == "WARN" else "❌")
    print(f"  {icon}  {name}: {detail}")


def validate(parquet_path: Path) -> dict:
    print(f"\n🔍 Validating: {parquet_path}")
    checks: list[dict] = []

    # ── Load ───────────────────────────────────────────────────────────────────
    try:
        gdf = gpd.read_parquet(parquet_path)
    except Exception as e:
        return {
            "passed": False,
            "parquet_path": str(parquet_path),
            "checks": [{"name": "load", "level": "FAIL", "passed": False,
                         "detail": f"Could not read parquet: {e}"}],
            "summary": "FAIL — could not load file",
        }

    total = len(gdf)
    print(f"  Loaded {total:,} rows, {len(gdf.columns)} columns")

    # ── 1. Geometry present ────────────────────────────────────────────────────
    n_with_geom = gdf["geometry"].notna().sum() if "geometry" in gdf.columns else 0
    pct = n_with_geom / total if total else 0
    _check(checks, "geometry_present", "FAIL",
           pct >= GEOMETRY_PRESENT_MIN,
           f"{pct:.1%} of rows have geometry (threshold: {GEOMETRY_PRESENT_MIN:.0%})")

    # ── 2. Geometry is polygon ──────────────────────────────────────────────────
    if "geometry" in gdf.columns:
        non_null = gdf["geometry"].dropna()
        n_poly = non_null.apply(
            lambda g: "polygon" in g.geom_type.lower() if g is not None else False
        ).sum()
        pct_poly = n_poly / len(non_null) if len(non_null) else 0
        _check(checks, "geometry_is_polygon", "FAIL",
               pct_poly == 1.0,
               f"{pct_poly:.1%} of non-null geometries are Polygon/MultiPolygon")
    else:
        _check(checks, "geometry_is_polygon", "FAIL", False, "No geometry column found")

    # ── 3. CRS is WGS84 ────────────────────────────────────────────────────────
    crs_ok = False
    crs_detail = "No CRS set"
    if hasattr(gdf, "crs") and gdf.crs is not None:
        try:
            epsg = CRS(gdf.crs).to_epsg()
            crs_ok = (epsg == 4326)
            crs_detail = f"EPSG:{epsg}"
        except Exception as e:
            crs_detail = f"Could not parse CRS: {e}"
    _check(checks, "crs_wgs84", "FAIL", crs_ok, crs_detail)

    # ── 4. Land value field present ────────────────────────────────────────────
    land_candidates = ["current_full_land_value", "land_value", "REALLANDVA"]
    land_col = next((c for c in land_candidates if c in gdf.columns), None)
    _check(checks, "land_value_field_present", "FAIL",
           land_col is not None,
           f"Found: '{land_col}'" if land_col else f"None of {land_candidates} found")

    # ── 5. Improvement value field present ─────────────────────────────────────
    impr_candidates = ["improvement_value", "REALIMPROV"]
    impr_col = next((c for c in impr_candidates if c in gdf.columns), None)
    _check(checks, "improvement_value_field_present", "FAIL",
           impr_col is not None,
           f"Found: '{impr_col}'" if impr_col else f"None of {impr_candidates} found")

    # ── 6. Refined category field present ──────────────────────────────────────
    cat_col = next((c for c in REFINED_FIELD_CANDIDATES if c in gdf.columns), None)
    _check(checks, "refined_category_field_present", "FAIL",
           cat_col is not None,
           f"Found: '{cat_col}'" if cat_col else f"None of {REFINED_FIELD_CANDIDATES} found")

    # ── 7. Land value non-zero (warn only) ─────────────────────────────────────
    if land_col and land_col in gdf.columns:
        vals = pd.to_numeric(gdf[land_col], errors="coerce")
        n_nonzero = (vals > 0).sum()
        pct_nz = n_nonzero / total if total else 0
        _check(checks, "land_value_nonzero", "WARN",
               pct_nz >= LAND_VALUE_NONZERO_MIN,
               f"{pct_nz:.1%} of parcels have non-zero land value (threshold: {LAND_VALUE_NONZERO_MIN:.0%})")
    else:
        _check(checks, "land_value_nonzero", "WARN", False, "Land value column not found — skipped")

    # ── 8. No negative land/improvement values ──────────────────────────────────
    has_neg = False
    neg_detail = "No negative values"
    for col in [c for c in [land_col, impr_col] if c and c in gdf.columns]:
        vals = pd.to_numeric(gdf[col], errors="coerce").fillna(0)
        n_neg = (vals < 0).sum()
        if n_neg > 0:
            has_neg = True
            neg_detail = f"{n_neg:,} negative values in '{col}'"
            break
    _check(checks, "no_negative_values", "FAIL", not has_neg, neg_detail)

    # ── 9. Bounding box within USA ─────────────────────────────────────────────
    bbox_ok = False
    bbox_detail = "Geometry column not available"
    if "geometry" in gdf.columns and gdf["geometry"].notna().any():
        try:
            b = gdf.total_bounds  # [minx, miny, maxx, maxy]
            bbox_ok = (
                b[0] >= USA_BOUNDS["minx"] and b[2] <= USA_BOUNDS["maxx"] and
                b[1] >= USA_BOUNDS["miny"] and b[3] <= USA_BOUNDS["maxy"]
            )
            bbox_detail = f"bounds=[{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}]"
        except Exception as e:
            bbox_detail = f"Could not compute bounds: {e}"
    _check(checks, "bbox_within_usa", "FAIL", bbox_ok, bbox_detail)

    # ── 10. Parcel count floor ──────────────────────────────────────────────────
    _check(checks, "parcel_count_floor", "FAIL",
           total >= PARCEL_COUNT_MIN,
           f"{total:,} parcels (minimum: {PARCEL_COUNT_MIN:,})")

    # ── 11. Exemption flag present ──────────────────────────────────────────────
    has_exempt = "exemption_flag" in gdf.columns
    _check(checks, "exemption_flag_present", "FAIL", has_exempt,
           "Found 'exemption_flag'" if has_exempt else "Column 'exemption_flag' not found")

    # ── 12. Exempt flag has at least one non-zero (warn) ───────────────────────
    if has_exempt:
        # The exported parquet has exempt parcels REMOVED, so exemption_flag should be all 0
        # This check verifies the column exists and is consistent (all 0 after filtering)
        n_nonzero_exempt = (gdf["exemption_flag"].fillna(0).astype(int) != 0).sum()
        _check(checks, "exemption_flag_all_zero_after_filter", "WARN",
               n_nonzero_exempt == 0,
               f"{n_nonzero_exempt:,} rows have exemption_flag != 0 (should be 0 after filtering)")
    else:
        _check(checks, "exemption_flag_all_zero_after_filter", "WARN", False, "Skipped — field missing")

    # ── 13. Category coverage (warn) ───────────────────────────────────────────
    if cat_col and cat_col in gdf.columns:
        n_mapped = gdf[cat_col].notna().sum()
        pct_mapped = n_mapped / total if total else 0
        top = gdf[cat_col].value_counts(dropna=False).head(8).to_dict()
        _check(checks, "category_coverage", "WARN",
               pct_mapped >= CATEGORY_COVERAGE_MIN,
               f"{pct_mapped:.1%} mapped (threshold: {CATEGORY_COVERAGE_MIN:.0%}) | top: {top}")
    else:
        _check(checks, "category_coverage", "WARN", False, "Refined category column not found — skipped")

    # ── Summary ────────────────────────────────────────────────────────────────
    fails = [c for c in checks if not c["passed"] and c["level"] == "FAIL"]
    warns = [c for c in checks if not c["passed"] and c["level"] == "WARN"]
    passed = len(fails) == 0

    summary = f"{'PASS' if passed else 'FAIL'} — {len(checks)} checks, {len(fails)} fail(s), {len(warns)} warn(s)"
    print(f"\n{'✅' if passed else '❌'} {summary}\n")

    return {
        "passed": passed,
        "parquet_path": str(parquet_path),
        "row_count": total,
        "column_count": len(gdf.columns),
        "land_col": land_col,
        "impr_col": impr_col,
        "cat_col": cat_col,
        "checks": checks,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path", help="Path to the parquet file to validate")
    parser.add_argument("--output-json", default=None, help="Write JSON result to this path")
    args = parser.parse_args()

    result = validate(Path(args.parquet_path))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Result written to: {args.output_json}")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
