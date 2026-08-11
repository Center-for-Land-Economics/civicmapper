#!/usr/bin/env python3
"""Augment the Seattle parcel parquet with NMA neighborhood region-grouping columns.

Mirrors augment_houston_regions.py: spatially joins parcel centroids against Seattle's
"Neighborhood Map Atlas" layers and tags each parcel with two categorical region columns
so the frontend can offer per-region show/hide toggles (CityDef.jurisdictionGroups) and the
PMTiles bake can tag each hex with its dominant value per field (H3_CATEGORICAL_FIELDS).

    neighborhood_district  <- nma_nhoods_sub.L_HOOD   (20 districts, e.g. "Ballard")
    neighborhood           <- nma_nhoods_sub.S_HOOD    (94 neighborhoods, e.g. "Loyal Heights")

Both come from the SAME single join against the fine `nma_nhoods_sub` layer (which carries
the parent district L_HOOD alongside the small-hood S_HOOD), so district and neighborhood tags
are always consistent. Parcels outside all neighborhoods (water, edges) -> "(None)".

Also writes the frontend "Overlays" boundary GeoJSONs into viz/public/ (one per field), built
from the dissolved district layer (nma_nhoods_main, 20) and the neighborhood layer (94):
    viz/public/seattle-neighborhood_district-overlay.geojson
    viz/public/seattle-neighborhood-overlay.geojson

Operates in place on data/jurisidictions/data/seattle/seattle-wa-parcels.parquet (backup first).
Re-run parquet_to_pmtiles.py --city seattle --drop-remnants afterwards to rebuild tiles+metadata.

Source layers (City of Seattle ArcGIS, public, no token):
  nma_nhoods_sub  / nma_nhoods_main on services.arcgis.com/ZOyb2t4B0UYuYNYH
"""
from __future__ import annotations

import io
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEATTLE = HERE / "data" / "seattle"
PARQUET = SEATTLE / "seattle-wa-parcels.parquet"
BACKUP = SEATTLE / "seattle-wa-parcels.preregions.parquet"
OVERLAY_DIR = ROOT / "viz" / "public"
NONE = "(None)"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
BASE = "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services"
SUB_URL = f"{BASE}/nma_nhoods_sub/FeatureServer/0/query"     # S_HOOD + L_HOOD, 94
MAIN_URL = f"{BASE}/nma_nhoods_main/FeatureServer/0/query"   # L_HOOD, 20 (dissolved districts)
SIMPLIFY_TOL = 0.00005   # ~5.5 m — plenty for outline overlays
PRECISION = 5


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_layer(url: str, fields: str) -> gpd.GeoDataFrame:
    r = requests.get(url, params={"where": "1=1", "outFields": fields,
                                  "returnGeometry": "true", "outSR": 4326, "f": "geojson"},
                     headers=HEADERS, timeout=120)
    r.raise_for_status()
    g = gpd.read_file(io.BytesIO(r.content))
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    elif g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    g = g[g.geometry.notnull()].copy()
    inv = ~g.geometry.is_valid
    if inv.any():
        g.loc[inv, "geometry"] = g.loc[inv, "geometry"].buffer(0)
    return g[g.geometry.notnull() & ~g.geometry.is_empty].copy()


def augment_parcels(sub: gpd.GeoDataFrame) -> None:
    if not PARQUET.exists():
        raise SystemExit(f"Missing parquet: {PARQUET}")
    if not BACKUP.exists():
        shutil.copy2(PARQUET, BACKUP)
        log(f"Backed up current parquet -> {BACKUP.name}")
    else:
        log(f"Backup already exists, leaving it untouched -> {BACKUP.name}")

    log(f"Reading {PARQUET.name} ...")
    parcel = gpd.read_parquet(PARQUET)
    log(f"  {len(parcel):,} parcels, crs={parcel.crs}")

    cent = parcel.geometry.to_crs(3857).centroid.to_crs(4326)
    pts = gpd.GeoDataFrame(pd.DataFrame({"_row": range(len(parcel))}),
                           geometry=cent.values, crs="EPSG:4326")

    layer = sub[["S_HOOD", "L_HOOD", "geometry"]].copy()
    tagged = gpd.sjoin(pts, layer, how="left", predicate="within")
    tagged = tagged[~tagged.index.duplicated(keep="first")]  # overlapping slivers -> first

    for src_col, dst_col in [("L_HOOD", "neighborhood_district"), ("S_HOOD", "neighborhood")]:
        vals = tagged[src_col].reindex(range(len(parcel)))
        parcel[dst_col] = vals.map(lambda v: str(v) if pd.notna(v) else NONE).values
        vc = parcel[dst_col].value_counts(dropna=False)
        matched = int((parcel[dst_col] != NONE).sum())
        log(f"  {dst_col}: {len(vc)} distinct, {matched:,} matched, "
            f"{int(vc.get(NONE, 0)):,} -> {NONE}; top {dict(list(vc.head(5).items()))}")

    log(f"Writing {PARQUET.name} ...")
    parcel.to_parquet(PARQUET, index=False)
    log(f"Done. Columns now include: neighborhood_district, neighborhood")


def build_overlay(g: gpd.GeoDataFrame, name_col: str, field: str) -> None:
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    out = OVERLAY_DIR / f"seattle-{field}-overlay.geojson"
    # dissolve to one polygon per region name (sub layer can have multiple rows per S_HOOD)
    d = g[[name_col, "geometry"]].rename(columns={name_col: "name"}).dissolve(by="name").reset_index()
    d["geometry"] = d.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    if out.exists():
        out.unlink()
    try:
        d.to_file(out, driver="GeoJSON", COORDINATE_PRECISION=PRECISION)
    except TypeError:
        d.to_file(out, driver="GeoJSON")
    kb = out.stat().st_size / 1024
    log(f"WROTE {out.name}: {len(d)} regions, {kb:,.0f} KB")


def main() -> None:
    log("Fetching NMA neighborhood layers ...")
    sub = fetch_layer(SUB_URL, "S_HOOD,L_HOOD")
    main_layer = fetch_layer(MAIN_URL, "L_HOOD")
    log(f"  sub: {len(sub)} neighborhoods; main: {len(main_layer)} districts")

    augment_parcels(sub)
    build_overlay(main_layer, "L_HOOD", "neighborhood_district")
    build_overlay(sub, "S_HOOD", "neighborhood")


if __name__ == "__main__":
    main()
