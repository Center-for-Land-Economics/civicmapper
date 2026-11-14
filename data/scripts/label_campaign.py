#!/usr/bin/env python3
"""
Create a parking-labelling campaign: stratified random sample squares for one city, ready for
the dev-only labelling tool (viz/label.html, served by the local API).

Writes data/labels/<campaign>/:
  campaign.json    city, square size, strata, imagery policy, creation time
  squares.geojson  one polygon per square: id, stratum, double (needs a 2nd labeller),
                   NAIP item ids of the most recent year covering it, imagery date
  prefill.geojson  existing (OSM / Overture) parking polygons clipped to the squares, for
                   correction mode (never shown in blind mode)
  parcels.geojson  parcel outlines clipped to the squares (category + exempt flag), context only

Strata: each square is assigned the dominant land use of the parcel land inside it
(commercial, industrial, multifamily, residential, other), or "open" when parcels cover little
of it (rights-of-way, parks, exempt land). Allocation over-samples where parking lives.

Imagery: the most recent NAIP year that covers each square on Microsoft Planetary Computer
(Texas 2024 is partial there; squares fall back to 2022). Tiles are streamed by the browser
from Planetary Computer's tiler, so nothing is downloaded here.

Usage:
  python data/scripts/label_campaign.py --city plano --name pilot-plano --n 150
  python data/scripts/label_campaign.py --city laredo --name pilot-laredo --n 150 --seed 7
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import box, mapping

ROOT = Path(__file__).resolve().parents[2]
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

STRATUM_OF = {
    "Commercial": "commercial", "Vacant Commercial": "commercial",
    "Industrial": "industrial",
    "Multifamily": "multifamily",
    "Single Family": "residential", "Residential Inventory": "residential",
    "Vacant Residential": "residential",
}
# Share of squares per stratum (over-samples where parking lives; residential and open squares
# are still sampled so false positives and driveways are measured).
DEFAULT_WEIGHTS = {"commercial": 0.35, "industrial": 0.15, "multifamily": 0.15,
                   "residential": 0.20, "other": 0.10, "open": 0.05}


def city_paths(city: str, state: str = "tx") -> tuple[Path, Path, Path]:
    d = ROOT / "data" / "jurisidictions" / "data" / city
    return (d / f"{city}-{state}-parcels.parquet", d / f"{city}-{state}-boundary.parquet",
            ROOT / "data" / "parking" / city / f"{city}-{state}-parking-lots.parquet")


def naip_items(bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Every NAIP item over bounds (all years), paging the STAC search."""
    feats, body = [], {"collections": ["naip"], "bbox": list(bounds), "limit": 500}
    url = STAC
    while url:
        r = requests.post(url, json=body, timeout=120)
        r.raise_for_status()
        js = r.json()
        feats += js.get("features", [])
        nxt = next((l for l in js.get("links", []) if l.get("rel") == "next"), None)
        url, body = (nxt["href"], nxt.get("body", body)) if nxt else (None, None)
    rows = [{"id": f["id"], "year": int(f["properties"].get("naip:year") or 0),
             "datetime": f["properties"].get("datetime"), "gsd": f["properties"].get("gsd"),
             "geometry": f["geometry"]} for f in feats]
    g = gpd.GeoDataFrame.from_features(
        [{"type": "Feature", "geometry": r.pop("geometry"), "properties": r} for r in rows], crs=4326)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--city", required=True)
    ap.add_argument("--state", default="tx")
    ap.add_argument("--name", required=True, help="Campaign id, e.g. pilot-plano")
    ap.add_argument("--n", type=int, default=150, help="Number of squares")
    ap.add_argument("--size-m", type=float, default=250.0, help="Square side in metres")
    ap.add_argument("--double-share", type=float, default=0.2,
                    help="Share of squares that need two independent labellers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = ROOT / "data" / "labels" / a.name
    if out.exists() and not a.overwrite and (out / "squares.geojson").exists():
        sys.exit(f"{out} exists; pass --overwrite to rebuild (existing labels.jsonl is kept)")
    out.mkdir(parents=True, exist_ok=True)

    parcels_p, boundary_p, parking_p = city_paths(a.city, a.state)
    parcels = gpd.read_parquet(parcels_p)
    if "likely_remnant" in parcels.columns:
        parcels = parcels[parcels["likely_remnant"].fillna(0).astype(int) != 1]
    boundary = (gpd.read_parquet(boundary_p).union_all() if boundary_p.exists()
                else parcels.union_all().convex_hull)
    utm = gpd.GeoSeries([boundary], crs=4326).estimate_utm_crs()
    b_utm = gpd.GeoSeries([boundary], crs=4326).to_crs(utm).iloc[0]

    # Grid of squares whose centre lies in the city.
    s = a.size_m
    x0, y0, x1, y1 = b_utm.bounds
    cells = [box(x, y, x + s, y + s)
             for x in np.arange(math.floor(x0 / s) * s, x1, s)
             for y in np.arange(math.floor(y0 / s) * s, y1, s)]
    grid = gpd.GeoDataFrame(geometry=cells, crs=utm)
    grid = grid[grid.geometry.centroid.within(b_utm)].reset_index(drop=True)

    # Stratum = dominant parcel land use by area inside the square.
    p = parcels.to_crs(utm)[["property_land_use_category", "geometry"]].copy()
    p["stratum"] = p["property_land_use_category"].map(STRATUM_OF).fillna("other")
    inter = gpd.overlay(grid.reset_index(names="cell"), p[["stratum", "geometry"]], how="intersection",
                        keep_geom_type=True)
    inter["a"] = inter.geometry.area
    share = inter.groupby("cell")["a"].sum() / (s * s)
    dom = inter.groupby(["cell", "stratum"])["a"].sum().reset_index().sort_values("a")\
        .drop_duplicates("cell", keep="last").set_index("cell")["stratum"]
    grid["stratum"] = grid.index.map(dom).fillna("open")
    grid.loc[grid.index.map(share).fillna(0) < 0.2, "stratum"] = "open"

    # Allocate and sample.
    rng = random.Random(a.seed)
    avail = grid["stratum"].value_counts().to_dict()
    want = {k: int(round(a.n * w)) for k, w in DEFAULT_WEIGHTS.items()}
    picks = []
    for k, n in want.items():
        idx = list(grid.index[grid["stratum"] == k])
        rng.shuffle(idx)
        picks += idx[:min(n, len(idx))]
    short = a.n - len(picks)
    if short > 0:
        # A stratum was too small (Laredo: Webb CAD codes warehouses as commercial, so there are
        # few industrial squares). Top up from commercial, then other strata, in weight order.
        taken = set(picks)
        for k in sorted(DEFAULT_WEIGHTS, key=DEFAULT_WEIGHTS.get, reverse=True):
            rest = [i for i in grid.index[grid["stratum"] == k] if i not in taken]
            rng.shuffle(rest)
            add = rest[:short]
            picks += add
            taken.update(add)
            short -= len(add)
            if short <= 0:
                break
    sq = grid.loc[picks].copy()
    sq = sq.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    sq["id"] = [f"{a.name}-{i + 1:04d}" for i in range(len(sq))]
    n_double = int(round(len(sq) * a.double_share))
    sq["double"] = [i < n_double for i in range(len(sq))]
    sq = sq.sample(frac=1, random_state=a.seed + 1).reset_index(drop=True)  # interleave doubles
    sq_ll = sq.to_crs(4326)

    # Most recent NAIP year covering each square.
    items = naip_items(tuple(sq_ll.total_bounds))
    years = []
    for _, r in sq_ll.iterrows():
        cov = items[items.intersects(r.geometry)]
        if cov.empty:
            years.append((None, [], None))
            continue
        yr = int(cov["year"].max())
        sel = cov[cov["year"] == yr]
        years.append((yr, sel["id"].tolist(), str(sel["datetime"].iloc[0])[:10]))
    sq_ll["naip_year"] = [y for y, _, _ in years]
    sq_ll["naip_items"] = [json.dumps(i) for _, i, _ in years]
    sq_ll["imagery_date"] = [d for _, _, d in years]
    cx = sq_ll.geometry.centroid
    sq_ll["center_lng"], sq_ll["center_lat"] = cx.x.round(6), cx.y.round(6)

    def write(gdf: gpd.GeoDataFrame, name: str) -> None:
        feats = [{"type": "Feature", "geometry": mapping(g),
                  "properties": {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                                 for k, v in row.items()}}
                 for g, (_, row) in zip(gdf.geometry, gdf.drop(columns="geometry").iterrows())]
        (out / name).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

    sq_out = sq_ll[["id", "stratum", "double", "naip_year", "naip_items", "imagery_date",
                    "center_lng", "center_lat", "geometry"]].copy()
    sq_out["naip_items"] = sq_out["naip_items"].map(json.loads)
    write(sq_out, "squares.geojson")

    # Context layers clipped to the squares.
    clip = sq_ll[["id", "geometry"]].rename(columns={"id": "square_id"})
    pc = gpd.overlay(parcels[["property_land_use_category", "exemption_flag", "geometry"]]
                     .rename(columns={"property_land_use_category": "category"}),
                     clip, how="intersection", keep_geom_type=True)
    write(pc, "parcels.geojson")
    n_pre = 0
    if parking_p.exists():
        pk = gpd.read_parquet(parking_p)
        keep = [c for c in ["parking_type", "source", "is_exempt"] if c in pk.columns]
        pre = gpd.overlay(pk[keep + ["geometry"]].to_crs(4326), clip, how="intersection",
                          keep_geom_type=True)
        n_pre = len(pre)
        write(pre, "prefill.geojson")
    else:
        write(gpd.GeoDataFrame(geometry=[], crs=4326), "prefill.geojson")

    meta = {"campaign": a.name, "city": a.city, "state": a.state, "square_m": s, "n": len(sq_out),
            "double": n_double, "seed": a.seed, "strata": sq_out["stratum"].value_counts().to_dict(),
            "strata_available": avail, "naip_years": sq_out["naip_year"].value_counts().to_dict(),
            "prefill_polygons": n_pre, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "classes": ["paved", "unpaved", "structure", "unsure", "cutout"]}
    (out / "campaign.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
