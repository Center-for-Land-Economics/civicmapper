#!/usr/bin/env python3
"""Build a tiny per-city land-totals JSON for the Parking page's "parking share of taxable
land value" metric.

The parking page loads ONLY the parking dataset, so it has no idea of the citywide (or
per-neighborhood) total parcel land value. This script reads the parcel parquet and emits the
NON-EXEMPT land-value denominators the parking page needs — both citywide and per region — so
the metric can update live as neighborhood toggles change. Works uniformly for PMTiles cities
and browser-GeoParquet cities (no bake / no metadata.groups dependency).

Output: data/jurisidictions/data/<city>/<city>-<state>-land-totals.json
  {
    "citywide": { "land": <sum land value>, "nonExemptLand": <sum where not exempt> },
    "groups": {
      "<field>": { "<region>": { "land": .., "nonExemptLand": .. }, ... },
      ...
    }
  }

Notes:
- land value column: current_full_land_value (canonical) else REALLANDVA.
- exempt: exemption_flag (0/1). Most WA ETLs DROP exempt parcels entirely, so exemption_flag is
  all 0 and nonExemptLand == land — which is correct (all shipped land is taxable). Cities that
  RETAIN exempt parcels (flag=1) get them excluded from nonExemptLand.
- region fields: the same categorical region columns the bake tags (H3_CATEGORICAL_FIELDS). Only
  those present in the parquet are emitted, keyed by region name exactly as the parcels/overlays
  use them (so the frontend can sum over the widget's visible region set).

Usage: python data/scripts/build_land_totals.py --city vancouver
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parquet_registry import list_cities, resolve_city  # noqa: E402

# Same set the PMTiles bake tags as categorical region groups (H3_CATEGORICAL_FIELDS).
REGION_FIELDS = ["jurisdiction", "council_district", "super_neighborhood", "civic_club",
                 "neighborhood_district", "neighborhood", "borough"]


def resolve_parcel_path(city_key: str, meta) -> Path:
    base = ROOT / "data" / "jurisidictions" / "data" / city_key
    for name in (meta.canonical_filename, meta.legacy_filename):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No parcel parquet for '{city_key}' under {base}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True, help=f"City key. Available: {', '.join(list_cities())}")
    args = ap.parse_args()

    meta = resolve_city(args.city)
    parcel_path = resolve_parcel_path(args.city, meta)
    gdf = gpd.read_parquet(parcel_path)
    print(f"Read {parcel_path.name}: {len(gdf):,} parcels", flush=True)

    land = (pd.to_numeric(gdf["current_full_land_value"], errors="coerce")
            if "current_full_land_value" in gdf.columns
            else pd.to_numeric(gdf.get("REALLANDVA"), errors="coerce")).fillna(0.0)
    if "exemption_flag" in gdf.columns:
        non_exempt = pd.to_numeric(gdf["exemption_flag"], errors="coerce").fillna(0).astype(int) == 0
    else:
        non_exempt = pd.Series(True, index=gdf.index)  # no flag -> exempt already dropped upstream

    out = {
        "citywide": {"land": round(float(land.sum())),
                     "nonExemptLand": round(float(land[non_exempt].sum()))},
        "groups": {},
    }

    for field in REGION_FIELDS:
        if field not in gdf.columns:
            continue
        df = pd.DataFrame({"region": gdf[field].astype(str), "land": land, "ne": non_exempt})
        totals = {}
        for region, sub in df.groupby("region"):
            totals[region] = {"land": round(float(sub["land"].sum())),
                              "nonExemptLand": round(float(sub.loc[sub["ne"], "land"].sum()))}
        out["groups"][field] = totals
        print(f"  {field}: {len(totals)} regions", flush=True)

    out_path = parcel_path.parent / f"{meta.city}-{meta.state}-land-totals.json"
    out_path.write_text(json.dumps(out), encoding="utf-8")
    kb = out_path.stat().st_size / 1024
    print(f"WROTE {out_path.name}: citywide nonExemptLand=${out['citywide']['nonExemptLand']:,} "
          f"({kb:,.1f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
