#!/usr/bin/env python3
"""
Add "targeted" squares to a labelling campaign: places where the model probably errs.

The fine-tuned model's main error is calling bare ground (caliche, dirt pads, tank farms, vacant
lots) parking. This samples unlabelled cells of the campaign's grid whose land is mostly
"suspect" (vacant, agricultural, industrial, or no parcel at all), runs the model on them, and
appends the cells where it predicts the most parking on suspect land.

Targeted squares are labelled like any other, but they are NOT a random sample:
  - they carry targeted=true and stratum "targeted";
  - label_finetune.py never uses them as test squares and the label page never treats them as
    test squares, so accuracy figures stay unbiased;
  - citywide estimates from the random squares must leave them out.
They get the same context layers (parcels, OSM prefill, buildings, roads) and model guesses.

Usage:
  python data/scripts/label_targets.py pilot-laredo --weights data/labels/pilot-laredo/finetune/model.pt --n 25
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from shapely.geometry import box, mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.scripts.label_buildings import campaign_buildings, campaign_roads  # noqa: E402
from data.scripts.label_campaign import city_paths, naip_items  # noqa: E402
from data.scripts.label_predict import CELL, load_model, mask_polygons, square_prediction  # noqa: E402
from data.scripts import overture_source  # noqa: E402
from data.scripts.classify_parking_surface import OVERTURE_RELEASE  # noqa: E402

LABELS = ROOT / "data" / "labels"
SUSPECT = {"Vacant Residential", "Vacant Commercial", "Agricultural / Rural", "Industrial"}


def clean(v):
    return None if isinstance(v, float) and np.isnan(v) else (v.item() if hasattr(v, "item") else v)


def append_features(path: Path, gdf: gpd.GeoDataFrame) -> None:
    fc = json.loads(path.read_text()) if path.exists() else {"type": "FeatureCollection", "features": []}
    for g, (_, row) in zip(gdf.geometry, gdf.drop(columns="geometry").iterrows()):
        fc["features"].append({"type": "Feature", "geometry": mapping(g),
                               "properties": {k: clean(v) for k, v in row.items()}})
    path.write_text(json.dumps(fc))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaign")
    ap.add_argument("--weights", help="fine-tuned model.pt")
    ap.add_argument("--n", type=int, default=25, help="squares to add")
    ap.add_argument("--candidates", type=int, default=250, help="cells to run the model on")
    ap.add_argument("--min-suspect-share", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    d = LABELS / a.campaign
    meta = json.loads((d / "campaign.json").read_text())
    city, state, s = meta["city"], meta.get("state", "tx"), float(meta["square_m"])
    squares = gpd.read_file(d / "squares.geojson")
    utm = squares.estimate_utm_crs()
    parcels_p, boundary_p, parking_p = city_paths(city, state)
    parcels = gpd.read_parquet(parcels_p)
    if "likely_remnant" in parcels.columns:
        parcels = parcels[parcels["likely_remnant"].fillna(0).astype(int) != 1]
    boundary = gpd.read_parquet(boundary_p).union_all()
    b_utm = gpd.GeoSeries([boundary], crs=4326).to_crs(utm).iloc[0]

    # Same grid as label_campaign.py; drop cells already in the campaign.
    x0, y0, x1, y1 = b_utm.bounds
    cells = [box(x, y, x + s, y + s) for x in np.arange(math.floor(x0 / s) * s, x1, s)
             for y in np.arange(math.floor(y0 / s) * s, y1, s)]
    grid = gpd.GeoDataFrame(geometry=cells, crs=utm)
    grid = grid[grid.geometry.centroid.within(b_utm)].reset_index(drop=True)
    taken = squares.to_crs(utm).union_all()
    grid = grid[~grid.geometry.centroid.within(taken)].reset_index(drop=True)

    # Suspect land share per cell (no parcel counts as suspect: exempt yards, unlinked polygons).
    p = parcels.to_crs(utm)[["property_land_use_category", "geometry"]]
    inter = gpd.overlay(grid.reset_index(names="cell"), p, how="intersection", keep_geom_type=True)
    inter["a"] = inter.geometry.area
    parcel_a = inter.groupby("cell")["a"].sum()
    sus_a = inter[inter.property_land_use_category.isin(SUSPECT)].groupby("cell")["a"].sum()
    share = ((s * s - parcel_a.reindex(grid.index).fillna(0)).clip(lower=0) + sus_a.reindex(grid.index).fillna(0)) / (s * s)
    pool = list(grid.index[share >= a.min_suspect_share])
    random.Random(a.seed).shuffle(pool)
    cand = grid.loc[pool[:a.candidates]].copy()
    print(f"{a.campaign}: {len(grid):,} unlabelled cells, {len(pool):,} mostly suspect land; "
          f"running the model on {len(cand)}", flush=True)

    cand_ll = cand.to_crs(4326)
    items = naip_items(tuple(cand_ll.total_bounds))
    bundle = load_model(a.weights, a.cpu)
    sus_parcels = p[p.property_land_use_category.isin(SUSPECT)]
    results = []
    t0 = time.time()
    for n, (idx, cell) in enumerate(cand.iterrows(), 1):
        cov = items[items.intersects(cand_ll.loc[idx].geometry)]
        if cov.empty:
            continue
        yr = int(cov["year"].max())
        sel = cov[cov["year"] == yr]
        mask, _p, tr = square_prediction(cell.geometry, sel["id"].tolist(), utm, bundle, a.threshold)
        h, w = mask.shape
        here = p[p.intersects(cell.geometry)]
        land = rasterize([(g, 1) for g in here.geometry], out_shape=(h, w), transform=tr, fill=0) == 1 \
            if len(here) else np.zeros((h, w), bool)
        sh = sus_parcels[sus_parcels.intersects(cell.geometry)]
        sus = rasterize([(g, 1) for g in sh.geometry], out_shape=(h, w), transform=tr, fill=0) == 1 \
            if len(sh) else np.zeros((h, w), bool)
        score = float((mask & (sus | ~land)).sum() * CELL * CELL)
        results.append({"cell": idx, "score": score, "pred_m2": float(mask.sum() * CELL * CELL),
                        "naip_year": yr, "naip_items": sel["id"].tolist(),
                        "imagery_date": str(sel["datetime"].iloc[0])[:10], "mask": mask, "tr": tr})
        if n % 25 == 0:
            print(f"  {n}/{len(cand)} cells ({time.time() - t0:.0f}s)", flush=True)
    results.sort(key=lambda r: -r["score"])
    pick = [r for r in results if r["score"] > 0][:a.n]
    if not pick:
        print("No candidate had predicted parking on suspect land; nothing added.")
        return 0

    # Append squares.
    existing = [int(i.rsplit("-", 1)[1]) for i in squares["id"]]
    nxt = max(existing) + 1
    new = gpd.GeoDataFrame(geometry=[cand.loc[r["cell"]].geometry for r in pick], crs=utm).to_crs(4326)
    new["id"] = [f"{a.campaign}-{nxt + k:04d}" for k in range(len(pick))]
    new["stratum"] = "targeted"
    new["double"] = False
    new["targeted"] = True
    new["naip_year"] = [r["naip_year"] for r in pick]
    new["naip_items"] = [r["naip_items"] for r in pick]
    new["imagery_date"] = [r["imagery_date"] for r in pick]
    c = new.to_crs(utm).geometry.centroid.to_crs(4326)
    new["center_lng"], new["center_lat"] = c.x.round(6), c.y.round(6)
    new["target_score_m2"] = [round(r["score"]) for r in pick]
    fc = json.loads((d / "squares.geojson").read_text())
    for g, (_, row) in zip(new.geometry, new.drop(columns="geometry").iterrows()):
        fc["features"].append({"type": "Feature", "geometry": mapping(g),
                               "properties": {k: clean(v) if not isinstance(v, list) else v for k, v in row.items()}})
    (d / "squares.geojson").write_text(json.dumps(fc))

    # Context layers for the new squares.
    clip = new[["id", "geometry"]].rename(columns={"id": "square_id"})
    pc = gpd.overlay(parcels[["property_land_use_category", "exemption_flag", "geometry"]]
                     .rename(columns={"property_land_use_category": "category"}), clip,
                     how="intersection", keep_geom_type=True)
    append_features(d / "parcels.geojson", pc)
    if parking_p.exists():
        pk = gpd.read_parquet(parking_p)
        keep = [k for k in ["parking_type", "source", "is_exempt"] if k in pk.columns]
        append_features(d / "prefill.geojson", gpd.overlay(pk[keep + ["geometry"]].to_crs(4326), clip,
                                                           how="intersection", keep_geom_type=True))
    release = overture_source.resolve_release(OVERTURE_RELEASE)
    sq_for_layers = new[["id", "geometry"]]
    append_features(d / "buildings.geojson", campaign_buildings(sq_for_layers, release))
    append_features(d / "roads.geojson", campaign_roads(sq_for_layers, release))
    mp = d / "model_prefill.geojson"
    mfc = json.loads(mp.read_text()) if mp.exists() else {"type": "FeatureCollection", "features": []}
    for r, sid, geom in zip(pick, new["id"], new.to_crs(utm).geometry):
        for g in gpd.GeoSeries(mask_polygons(r["mask"], r["tr"], utm, geom, 50.0), crs=utm).to_crs(4326):
            mfc["features"].append({"type": "Feature", "geometry": mapping(g),
                                    "properties": {"square_id": sid, "class": "paved", "source": "model"}})
    mp.write_text(json.dumps(mfc))

    meta.setdefault("targeted", []).append({
        "added": time.strftime("%Y-%m-%dT%H:%M:%S"), "squares": new["id"].tolist(), "model": bundle[3],
        "rule": f"top {len(pick)} of {len(results)} sampled cells with >= {a.min_suspect_share:.0%} vacant / "
                "agricultural / industrial / unparcelled land, ranked by predicted parking on that land",
        "note": "not a random sample: excluded from test squares and citywide estimates"})
    (d / "campaign.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Added {len(pick)} targeted squares ({new['id'].iloc[0]} … {new['id'].iloc[-1]}); predicted parking on "
          f"suspect land: {', '.join(str(r) for r in new['target_score_m2'].tolist()[:10])}{' …' if len(pick) > 10 else ''} m²")
    return 0


if __name__ == "__main__":
    sys.exit(main())
