#!/usr/bin/env python3
"""
Score parking models against Parking Reform Network's hand-traced downtown surface parking.

PRN's Parking Lot Map (github.com/ParkingReformNetwork/parking-lot-map, MIT licence) traces the
surface parking lots inside a downtown boundary for ~100 US cities, 9 of them in Texas. Inside the
boundary every pixel is either a PRN lot (parking) or not, so it gives free precision / recall /
IoU in cities we have not labelled. PRN traces off-street parking footprints INCLUDING garages
(checked 2026-11: 22-26% of PRN lot area in El Paso, Dallas and Corpus Christi lies under Overture
buildings, mostly tagged parking). Our definition is surface parking only, so PRN area under an
Overture building is ignored in scoring (neither hit nor miss); predictions on other buildings
still count as false alarms.

Each model is scored twice: as predicted, and with Overture building footprints removed from the
prediction ("-bldg"), the post-processing the citywide pipeline can apply cheaply (rooftops of
flat mid-rise buildings are a common false alarm outside Laredo).

Caveats: downtowns are the hardest terrain (garages, leaning towers, shadows) and not typical of a
whole city; PRN traced from other imagery at other dates, so a little disagreement is change or
misregistration, not model error.

For each city the downtown is cut into 250 m tiles on a 0.6 m grid (NAIP's pixel), NAIP is read
once per tile (latest year on Planetary Computer, cached under data/prn/<city>/naip/), every model
runs on the same imagery with the pipeline's recipe (threshold, NDVI strip, opening), and pixels
inside the boundary are scored. Writes data/prn/<city>/overview_<model>.png (imagery | PRN |
model | errors: green hit, red false alarm, blue miss) and data/prn/results.csv.

Usage:
  python data/scripts/prn_eval.py                                   # all 8 of our Texas cities, original model
  python data/scripts/prn_eval.py --weights data/labels/pilot-laredo/finetune/model.pt
  python data/scripts/prn_eval.py --cities dallas-tx houston-tx --weights a.pt b.pt
  python data/scripts/prn_eval.py --list                            # every PRN city id
Weights: original = the pipeline's model; each extra path is a fine-tuned state dict.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from shapely.geometry import box, shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.scripts.label_campaign import naip_items  # noqa: E402
from data.scripts.label_eval import NDVI_VEG, OPEN_RADIUS, disk, predict, read_naip  # noqa: E402

PRN_RAW = "https://raw.githubusercontent.com/ParkingReformNetwork/parking-lot-map/main/packages/primary/data"
OUT = ROOT / "data" / "prn"
TEXAS_OURS = ["arlington-tx", "austin-tx", "corpus-christi-tx", "dallas-tx", "el-paso-tx", "fort-worth-tx",
              "houston-tx", "san-antonio-tx"]
CELL, TILE_M, MARGIN = 0.6, 250.0, 48


def _closed(geom: dict):
    fix = lambda r: r if r[0] == r[-1] else r + [r[0]]  # noqa: E731  PRN rings are sometimes left open
    if geom["type"] == "Polygon":
        geom["coordinates"] = [fix(r) for r in geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        geom["coordinates"] = [[fix(r) for r in p] for p in geom["coordinates"]]
    return shape(geom).buffer(0)


def fetch(path: str, dest: Path) -> dict:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{PRN_RAW}/{path}", timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return json.loads(dest.read_text())


def prn_city(city: str):
    """-> (boundary, lots) shapely geometries in EPSG:4326."""
    bnd = fetch("city-boundaries.geojson", OUT / "city-boundaries.geojson")
    b = next((_closed(f["geometry"]) for f in bnd["features"] if f["properties"].get("id") == city), None)
    if b is None:
        raise SystemExit(f"{city}: not in PRN's city list (try --list)")
    lots = fetch(f"parking-lots/{city}.geojson", OUT / city / "lots.geojson")
    feats = lots["features"] if lots.get("type") == "FeatureCollection" else [lots]
    return b, gpd.GeoSeries([_closed(f["geometry"]) for f in feats], crs=4326).union_all()


def load_models(weights: list[str], cpu: bool):
    import torch
    from data.scripts.parking_lot_extraction import load_segformer_bundle
    processor, base, device, model_id = load_segformer_bundle(use_cpu=cpu)
    models = [("original", base)]
    for w in weights:
        _p, m, _d, _id = load_segformer_bundle(use_cpu=cpu)
        m.load_state_dict(torch.load(w, map_location=device))
        m.eval()
        models.append((Path(w).parent.parent.name + ":" + Path(w).stem, m))
    return processor, models, device


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cities", nargs="*", default=TEXAS_OURS)
    ap.add_argument("--weights", nargs="*", default=[], help="fine-tuned model.pt files to compare")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        stats = fetch("city-stats.json", OUT / "city-stats.json")
        print(" ".join(sorted(stats)))
        return 0

    from scipy.ndimage import binary_opening
    processor, models, device = load_models(a.weights, a.cpu)
    kernel = disk(OPEN_RADIUS)
    rows = []
    for city in a.cities:
        t0 = time.time()
        bnd, lots = prn_city(city)
        utm = gpd.GeoSeries([bnd], crs=4326).estimate_utm_crs()
        b_utm = gpd.GeoSeries([bnd], crs=4326).to_crs(utm).iloc[0]
        l_utm = gpd.GeoSeries([lots], crs=4326).to_crs(utm).iloc[0]
        x0, y0, x1, y1 = b_utm.bounds
        x0, y1 = math.floor(x0 / CELL) * CELL, math.ceil(y1 / CELL) * CELL
        W, H = math.ceil((x1 - x0) / CELL), math.ceil((y1 - y0) / CELL)
        tr_all = from_origin(x0, y1, CELL, CELL)
        inside = rasterize([(b_utm, 1)], out_shape=(H, W), transform=tr_all, fill=0) == 1
        truth = rasterize([(l_utm, 1)], out_shape=(H, W), transform=tr_all, fill=0) == 1
        bpath = OUT / city / "buildings.parquet"
        if not bpath.exists():
            from data.scripts import overture_source
            from data.scripts.classify_parking_surface import OVERTURE_RELEASE
            bl = overture_source.buildings(tuple(gpd.GeoSeries([bnd], crs=4326).total_bounds),
                                           overture_source.resolve_release(OVERTURE_RELEASE))
            bl[["class", "geometry"]].to_parquet(bpath)
        bl = gpd.read_parquet(bpath).to_crs(utm)
        bl = bl[bl.intersects(b_utm)]
        built = rasterize([(g, 1) for g in bl.geometry], out_shape=(H, W), transform=tr_all, fill=0) == 1 \
            if len(bl) else np.zeros((H, W), bool)
        ignore = truth & built          # PRN garages / covered parking: not surface parking
        truth = truth & ~built
        rgb = np.zeros((H, W, 3), np.uint8)
        preds = {name: np.zeros((H, W), bool) for name, _ in models}
        tpx = int(TILE_M / CELL)
        tiles = [(r, c) for r in range(0, H, tpx) for c in range(0, W, tpx)]
        items = naip_items(tuple(gpd.GeoSeries([bnd], crs=4326).total_bounds))
        cache = OUT / city / "naip"
        cache.mkdir(parents=True, exist_ok=True)
        years = set()
        for k, (r, c) in enumerate(tiles, 1):
            h, w = min(tpx, H - r), min(tpx, W - c)
            if not inside[r:r + h, c:c + w].any():
                continue
            tr = tr_all * Affine.translation(c, r)
            tile_ll = gpd.GeoSeries([box(*(tr * (0, h)), *(tr * (w, 0)))], crs=utm).to_crs(4326).iloc[0]
            cov = items[items.intersects(tile_ll)]
            if cov.empty:
                continue
            yr = int(cov["year"].max())
            years.add(yr)
            cp = cache / f"{r}_{c}_{yr}.npy"
            if cp.exists():
                img = np.load(cp)
            else:
                img = read_naip(cov[cov["year"] == yr]["id"].tolist(), utm,
                                tr * Affine.translation(-MARGIN, -MARGIN), w + 2 * MARGIN, h + 2 * MARGIN)
                np.save(cp, img)
            crop = (slice(MARGIN, MARGIN + h), slice(MARGIN, MARGIN + w))
            nir, red = img[3].astype(np.float32), img[0].astype(np.float32)
            veg = (np.where(nir + red > 0, (nir - red) / np.maximum(nir + red, 1), 0) > NDVI_VEG)[crop]
            rgb[r:r + h, c:c + w] = img[:3][(slice(None),) + crop].transpose(1, 2, 0)
            for name, m in models:
                p = predict(img, processor, m, device)[crop]
                preds[name][r:r + h, c:c + w] = binary_opening((p >= a.threshold) & ~veg, structure=kernel)
            if k % 20 == 0:
                print(f"  {city}: {k}/{len(tiles)} tiles ({time.time() - t0:.0f}s)", flush=True)
        variants = {}
        for name, pred in preds.items():
            variants[name] = pred
            variants[f"{name}-bldg"] = pred & ~built
        for name, pred in variants.items():
            p, t = pred & inside & ~ignore, truth & inside
            tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
            rows.append({"city": city, "model": name, "naip_years": "/".join(map(str, sorted(years))),
                         "prn_surface_ha": round(t.sum() * CELL * CELL / 1e4, 1),
                         "prn_garage_ha_ignored": round((ignore & inside).sum() * CELL * CELL / 1e4, 1), "pred_ha": round(p.sum() * CELL * CELL / 1e4, 1),
                         "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                         "iou": tp / (tp + fp + fn) if tp + fp + fn else None})
            # overview picture, downsampled to <= 1400 px wide per panel
            s = max(1, math.ceil(W / 1400))
            base = rgb[::s, ::s].astype(np.float32)
            dim = ~inside[::s, ::s]
            panels = []
            for ov in (None, (t, (255, 77, 109)), (p, (0, 200, 255))):
                im = base.copy()
                if ov is not None:
                    mk = ov[0][::s, ::s]
                    im[mk] = im[mk] * 0.45 + np.array(ov[1]) * 0.55
                im[dim] *= 0.35
                panels.append(im)
            err = base.copy()
            for mk, col in (((p & t), (40, 200, 90)), ((p & ~t), (230, 40, 40)), ((~p & t), (40, 90, 255))):
                mk = mk[::s, ::s]
                err[mk] = err[mk] * 0.45 + np.array(col) * 0.55
            err[dim] *= 0.35
            panels.append(err)
            gap = np.full((base.shape[0], 8, 3), 255, np.float32)
            strip = np.concatenate(sum([[x, gap] for x in panels], [])[:-1], axis=1)
            Image.fromarray(strip.clip(0, 255).astype(np.uint8)).save(OUT / city / f"overview_{name.replace(':', '_')}.png")
        print(f"{city}: done in {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    res = OUT / "results.csv"
    if res.exists():
        old = pd.read_csv(res)
        old = old[~old.set_index(["city", "model"]).index.isin(df.set_index(["city", "model"]).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(res, index=False)
    show = df[df.city.isin(a.cities)].copy()
    for c in ("precision", "recall", "iou"):
        show[c] = show[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "-")
    print("\nPRN downtowns (surface parking inside the boundary; PRN area under buildings ignored):")
    print(show.to_string(index=False))
    for name in show.model.unique():
        sub = df[(df.model == name) & df.city.isin(a.cities)]
        print(f"  {name}: median IoU {sub.iou.median():.0%}, median precision {sub.precision.median():.0%}, "
              f"median recall {sub.recall.median():.0%}")
    print(f"Pictures: {OUT.relative_to(ROOT)}/<city>/overview_<model>.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
