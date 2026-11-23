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

Each model is scored three ways: as predicted; with Overture building footprints removed
("-bldg"); and with buildings and public streets removed ("-clean", parking_postprocess.py: road
half-widths off-parcel, using our parcels when data/jurisidictions/data/<city>/ has them, else the
road buffer alone).

Caveats: downtowns are the hardest terrain (garages, leaning towers, shadows) and not typical of a
whole city; PRN traced from other imagery at other dates, so a little disagreement is change or
misregistration, not model error.

For each city the downtown is cut into 250 m tiles on a 0.6 m grid (NAIP's pixel), NAIP is read
once per tile (latest year on Planetary Computer, cached under data/prn-eval/<city>/naip/), every model
runs on the same imagery with the pipeline's recipe (threshold, NDVI strip, opening), and pixels
inside the boundary are scored. Writes data/prn-eval/<city>/overview_<model>.png (imagery | PRN |
model | errors: green hit, red false alarm, blue miss) and data/prn-eval/results.csv.

Usage:
  python data/scripts/prn_eval.py                                   # all 8 of our Texas cities, original model
  python data/scripts/prn_eval.py --weights data/labels/pilot-laredo/finetune/model.pt
  python data/scripts/prn_eval.py --cities dallas-tx houston-tx --weights a.pt b.pt
  python data/scripts/prn_eval.py --models segformer vit:data/labels/pilot-laredo/finetune-dinov2-large
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
from data.scripts.label_eval import NDVI_VEG, OPEN_RADIUS, disk, read_naip  # noqa: E402

PRN_RAW = "https://raw.githubusercontent.com/ParkingReformNetwork/parking-lot-map/main/packages/primary/data"
OUT = ROOT / "data" / "prn-eval"  # not "prn": a reserved device name on Windows
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


def prepare_city(city: str) -> dict:
    """Grids for one PRN downtown: boundary, PRN surface truth, ignored garages, buildings and streets
    for the clean-up, the 250 m tiles with their NAIP (cached under data/prn-eval/<city>/naip/)."""
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
    from data.scripts import overture_source
    from data.scripts.classify_parking_surface import OVERTURE_RELEASE
    from data.scripts.parking_postprocess import public_roads, street_mask
    bpath = OUT / city / "buildings.parquet"
    if not bpath.exists():
        bl = overture_source.buildings(tuple(gpd.GeoSeries([bnd], crs=4326).total_bounds),
                                       overture_source.resolve_release(OVERTURE_RELEASE))
        bl[["class", "geometry"]].to_parquet(bpath)
    bl = gpd.read_parquet(bpath).to_crs(utm)
    bl = bl[bl.intersects(b_utm)]
    built = rasterize([(g, 1) for g in bl.geometry], out_shape=(H, W), transform=tr_all, fill=0) == 1 \
        if len(bl) else np.zeros((H, W), bool)
    ignore = truth & built          # PRN garages / covered parking: not surface parking
    truth = truth & ~built
    rpath = OUT / city / "roads.parquet"
    if not rpath.exists():
        x0b, y0b, x1b, y1b = gpd.GeoSeries([bnd], crs=4326).total_bounds
        pad = 0.002  # ~200 m of margin so roads along the boundary are included
        public_roads((x0b - pad, y0b - pad, x1b + pad, y1b + pad),
                     overture_source.resolve_release(OVERTURE_RELEASE)).to_parquet(rpath)
    roads = gpd.read_parquet(rpath).to_crs(utm)
    key = city.replace("-tx", "").replace("-", "")
    ours = ROOT / "data" / "jurisidictions" / "data" / key / f"{key}-tx-parcels.parquet"
    parcel_geoms = None
    if ours.exists():
        pc = gpd.read_parquet(ours, columns=["geometry"]).to_crs(utm)
        parcel_geoms = pc[pc.intersects(b_utm.buffer(50))].geometry
    street = street_mask((H, W), tr_all, roads, parcel_geoms)
    tpx = int(TILE_M / CELL)
    items = naip_items(tuple(gpd.GeoSeries([bnd], crs=4326).total_bounds))
    cache = OUT / city / "naip"
    cache.mkdir(parents=True, exist_ok=True)
    tiles, years = [], set()
    for r in range(0, H, tpx):
        for c in range(0, W, tpx):
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
            tiles.append({"r": r, "c": c, "h": h, "w": w, "tr": tr, "year": yr,
                          "items": cov[cov["year"] == yr]["id"].tolist(), "cache": cache / f"{r}_{c}_{yr}.npy"})
    return {"city": city, "utm": utm, "H": H, "W": W, "inside": inside, "truth": truth, "ignore": ignore,
            "built": built, "street": street, "tiles": tiles, "years": years}


def tile_image(ctx: dict, t: dict) -> np.ndarray:
    if t["cache"].exists():
        return np.load(t["cache"])
    img = read_naip(t["items"], ctx["utm"], t["tr"] * Affine.translation(-MARGIN, -MARGIN),
                    t["w"] + 2 * MARGIN, t["h"] + 2 * MARGIN)
    np.save(t["cache"], img)
    return img


def run_model(ctx: dict, model, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """-> (pipeline-recipe mask over the downtown grid, RGB mosaic, seconds of model inference)."""
    from scipy.ndimage import binary_opening
    thr = model.threshold if threshold is None else threshold
    kernel = disk(OPEN_RADIUS)
    pred = np.zeros((ctx["H"], ctx["W"]), bool)
    rgb = np.zeros((ctx["H"], ctx["W"], 3), np.uint8)
    secs = 0.0
    for t in ctx["tiles"]:
        img = tile_image(ctx, t)
        r, c, h, w = t["r"], t["c"], t["h"], t["w"]
        crop = (slice(MARGIN, MARGIN + h), slice(MARGIN, MARGIN + w))
        nir, red = img[3].astype(np.float32), img[0].astype(np.float32)
        veg = (np.where(nir + red > 0, (nir - red) / np.maximum(nir + red, 1), 0) > NDVI_VEG)[crop]
        rgb[r:r + h, c:c + w] = img[:3][(slice(None),) + crop].transpose(1, 2, 0)
        ts = time.time()
        p = model.predict(img)[crop]
        secs += time.time() - ts
        pred[r:r + h, c:c + w] = binary_opening((p >= thr) & ~veg, structure=kernel)
    return pred, rgb, secs


def score_variants(ctx: dict, name: str, pred: np.ndarray) -> list[dict]:
    """Rows for name (as predicted), name-bldg (buildings removed), name-clean (buildings + streets)."""
    inside, truth, ignore = ctx["inside"], ctx["truth"], ctx["ignore"]
    rows = []
    for vname, vp in ((name, pred), (f"{name}-bldg", pred & ~ctx["built"]),
                      (f"{name}-clean", pred & ~ctx["built"] & ~ctx["street"])):
        p, t = vp & inside & ~ignore, truth & inside
        tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
        rows.append({"city": ctx["city"], "model": vname, "naip_years": "/".join(map(str, sorted(ctx["years"]))),
                     "prn_surface_ha": round(t.sum() * CELL * CELL / 1e4, 1),
                     "prn_garage_ha_ignored": round((ignore & inside).sum() * CELL * CELL / 1e4, 1),
                     "pred_ha": round(p.sum() * CELL * CELL / 1e4, 1),
                     "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                     "iou": tp / (tp + fp + fn) if tp + fp + fn else None})
    return rows


def overview(ctx: dict, rgb: np.ndarray, pred: np.ndarray, path: Path) -> None:
    """imagery | PRN | model | errors (green hit, red false alarm, blue miss), <= 1400 px per panel."""
    inside, ignore = ctx["inside"], ctx["ignore"]
    p, t = pred & inside & ~ignore, ctx["truth"] & inside
    s = max(1, math.ceil(ctx["W"] / 1400))
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
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(sum([[x, gap] for x in panels], [])[:-1], axis=1).clip(0, 255)
                    .astype(np.uint8)).save(path)


def safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cities", nargs="*", default=TEXAS_OURS)
    ap.add_argument("--models", nargs="*", default=["segformer"],
                    help="model specs (parking_models.py): segformer, segformer:<model.pt>, vit:<run dir>, sam2:<spec>")
    ap.add_argument("--weights", nargs="*", default=[], help="shorthand: fine-tuned SegFormer model.pt files")
    ap.add_argument("--threshold", type=float, default=None, help="override every model's own threshold")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        stats = fetch("city-stats.json", OUT / "city-stats.json")
        print(" ".join(sorted(stats)))
        return 0

    from data.scripts.parking_models import load_spec
    specs = list(a.models) + [f"segformer:{w}" for w in a.weights]
    ctxs = {}
    rows = []
    for spec in specs:
        model = load_spec(spec, cpu=a.cpu)
        for city in a.cities:
            t0 = time.time()
            ctx = ctxs.get(city) or ctxs.setdefault(city, prepare_city(city))
            pred, rgb, secs = run_model(ctx, model, a.threshold)
            rows += score_variants(ctx, model.name, pred)
            overview(ctx, rgb, pred & ~ctx["built"] & ~ctx["street"], OUT / city / f"overview_{safe(model.name)}.png")
            print(f"{city}: {model.name} done in {time.time() - t0:.0f}s", flush=True)
        del model

    df = pd.DataFrame(rows)
    res = OUT / "results.csv"
    if res.exists():
        old = pd.read_csv(res)
        old = old[~old.set_index(["city", "model"]).index.isin(df.set_index(["city", "model"]).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(res, index=False)
    names = list(dict.fromkeys(r["model"] for r in rows))
    show = df[df.city.isin(a.cities) & df.model.isin(names)].copy()
    for c in ("precision", "recall", "iou"):
        show[c] = show[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "-")
    print("\nPRN downtowns (surface parking inside the boundary; PRN area under buildings ignored):")
    print(show.to_string(index=False))
    for name in names:
        sub = df[(df.model == name) & df.city.isin(a.cities)]
        print(f"  {name}: median IoU {sub.iou.median():.0%}, median precision {sub.precision.median():.0%}, "
              f"median recall {sub.recall.median():.0%}")
    print(f"Pictures (clean predictions): {OUT.relative_to(ROOT)}/<city>/overview_<model>.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
