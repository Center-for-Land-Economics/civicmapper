#!/usr/bin/env python3
"""Augment the NYC parcel parquet with borough + neighborhood (NTA) region-grouping columns.

Mirrors augment_seattle_regions.py. Spatially joins parcel centroids against NYC DCP's 2020
Neighborhood Tabulation Areas layer (which carries BOTH the NTA neighborhood name and its parent
borough), tagging two categorical columns so the frontend can offer per-region show/hide toggles
(CityDef.jurisdictionGroups) and the PMTiles bake can tag each hex with its dominant value per
field (H3_CATEGORICAL_FIELDS):

    borough       <- NTA2020.BoroName   (5: Manhattan/Brooklyn/Queens/Bronx/Staten Island)
    neighborhood  <- NTA2020.NTAName    (262 NTAs, e.g. "Greenpoint", "Astoria")

Both from one join. Parcels outside all NTAs (water, edges) -> "(None)".

Also writes the frontend "Overlays" boundary GeoJSONs into viz/public/:
    viz/public/nyc-borough-overlay.geojson        (5 boroughs, dissolved)
    viz/public/nyc-neighborhood-overlay.geojson   (262 NTAs)

Operates in place on data/jurisidictions/data/nyc/nyc-ny-parcels.parquet (backup first).
Re-run parquet_to_pmtiles.py --city nyc afterwards to rebuild tiles+metadata.

Source layers (NYC DCP, public, no token), services5.arcgis.com/GfwWNkhOj9bNBqoJ:
  NYC_Neighborhood_Tabulation_Areas_2020 (NTAName + BoroName) ; NYC_Borough_Boundary (BoroName)
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
NYC = HERE / "data" / "nyc"
PARQUET = NYC / "nyc-ny-parcels.parquet"
BACKUP = NYC / "nyc-ny-parcels.preregions.parquet"
OVERLAY_DIR = ROOT / "viz" / "public"
NONE = "(None)"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
BASE = "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services"
NTA_URL = f"{BASE}/NYC_Neighborhood_Tabulation_Areas_2020/FeatureServer/0/query"
BORO_URL = f"{BASE}/NYC_Borough_Boundary/FeatureServer/0/query"
SIMPLIFY_TOL = 0.00005
PRECISION = 5


def log(m: str) -> None:
    print(m, flush=True)


def fetch_layer(url: str, fields: str) -> gpd.GeoDataFrame:
    r = requests.get(url, params={"where": "1=1", "outFields": fields,
                                  "returnGeometry": "true", "outSR": 4326, "f": "geojson"},
                     headers=HEADERS, timeout=180)
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


def build_overlay(g: gpd.GeoDataFrame, name_col: str, field: str) -> None:
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    out = OVERLAY_DIR / f"nyc-{field}-overlay.geojson"
    d = g[[name_col, "geometry"]].rename(columns={name_col: "name"}).dissolve(by="name").reset_index()
    d["geometry"] = d.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    if out.exists():
        out.unlink()
    try:
        d.to_file(out, driver="GeoJSON", COORDINATE_PRECISION=PRECISION)
    except TypeError:
        d.to_file(out, driver="GeoJSON")
    log(f"WROTE {out.name}: {len(d)} regions, {out.stat().st_size/1024:,.0f} KB")


def main() -> None:
    log("Fetching NYC NTA + borough layers ...")
    nta = fetch_layer(NTA_URL, "NTAName,BoroName")
    boro = fetch_layer(BORO_URL, "BoroName")
    log(f"  NTAs: {len(nta)}; boroughs: {len(boro)}")

    if not PARQUET.exists():
        raise SystemExit(f"Missing parquet: {PARQUET}")
    if not BACKUP.exists():
        shutil.copy2(PARQUET, BACKUP)
        log(f"Backed up -> {BACKUP.name}")
    else:
        log(f"Backup already exists -> {BACKUP.name}")

    log(f"Reading {PARQUET.name} ...")
    parcel = gpd.read_parquet(PARQUET)
    log(f"  {len(parcel):,} parcels, crs={parcel.crs}")

    cent = parcel.geometry.to_crs(3857).centroid.to_crs(4326)
    pts = gpd.GeoDataFrame(pd.DataFrame({"_row": range(len(parcel))}),
                           geometry=cent.values, crs="EPSG:4326")

    layer = nta[["NTAName", "BoroName", "geometry"]].copy()
    tagged = gpd.sjoin(pts, layer, how="left", predicate="within")
    tagged = tagged[~tagged.index.duplicated(keep="first")]
    for src_col, dst_col in [("BoroName", "borough"), ("NTAName", "neighborhood")]:
        vals = tagged[src_col].reindex(range(len(parcel)))
        parcel[dst_col] = vals.map(lambda v: str(v) if pd.notna(v) else NONE).values
        vc = parcel[dst_col].value_counts(dropna=False)
        matched = int((parcel[dst_col] != NONE).sum())
        log(f"  {dst_col}: {len(vc)} distinct, {matched:,} matched, "
            f"{int(vc.get(NONE, 0)):,} -> {NONE}; top {dict(list(vc.head(5).items()))}")

    log(f"Writing {PARQUET.name} ...")
    parcel.to_parquet(PARQUET, index=False)
    log("Done. Columns now include: borough, neighborhood")

    build_overlay(boro, "BoroName", "borough")
    build_overlay(nta, "NTAName", "neighborhood")


if __name__ == "__main__":
    main()
