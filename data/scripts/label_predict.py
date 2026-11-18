#!/usr/bin/env python3
"""
Model guesses for a labelling campaign: run the parking model over every square and save the
predicted parking as editable shapes (data/labels/<campaign>/model_prefill.geojson).

The label page's "Model" mode pre-loads these so labellers correct outlines instead of drawing
everything. Held-out test squares (the same hash split as label_finetune.py) are never pre-filled
in the tool, so accuracy is always measured on squares drawn from scratch.

Prediction = the pipeline's recipe: probability >= threshold, NDVI > 0.20 vegetation removed,
morphological opening, then Overture buildings and public streets removed (parking_postprocess.py,
--no-clean to skip); polygons are simplified (0.6 m) and pieces under --min-m2 dropped. The
model is two-class, so every guess arrives as "paved"; labellers reclassify gravel.

Usage:
  python data/scripts/label_predict.py pilot-laredo --weights data/labels/pilot-laredo/finetune/model.pt
  python data/scripts/label_predict.py pilot-laredo --weights ... --squares pilot-laredo-0151 pilot-laredo-0152
  python data/scripts/label_predict.py pilot-laredo --model vit:data/labels/pilot-laredo/finetune-dinov2-large
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
from rasterio.features import shapes
from rasterio.transform import from_origin, Affine
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.scripts.label_eval import NDVI_VEG, OPEN_RADIUS, disk, keep_mask, load_context, read_naip  # noqa: E402

LABELS = ROOT / "data" / "labels"
CELL = 0.6
MARGIN = 48


def load_model(weights: str | None, cpu: bool, spec: str | None = None):
    """A parking_models.ParkingModel: --model spec (any architecture), else SegFormer (+ --weights)."""
    from data.scripts.parking_models import load_spec
    m = load_spec(spec or (f"segformer:{weights}" if weights else "segformer"), cpu=cpu)
    m.train(False)
    return m


def square_prediction(geom_utm, items: list[str], utm, model, threshold: float):
    """-> (binary mask on the square's 0.6 m grid, prob, transform). geom_utm: square polygon in UTM;
    model: parking_models.ParkingModel."""
    from scipy.ndimage import binary_opening
    x0, y0, x1, y1 = geom_utm.bounds
    w, h = math.ceil((x1 - x0) / CELL - 1e-6), math.ceil((y1 - y0) / CELL - 1e-6)
    tr = from_origin(x0, y1, CELL, CELL)
    img = read_naip(items, utm, tr * Affine.translation(-MARGIN, -MARGIN), w + 2 * MARGIN, h + 2 * MARGIN)
    prob = model.predict(img)
    nir, red = img[3].astype(np.float32), img[0].astype(np.float32)
    ndvi = np.where(nir + red > 0, (nir - red) / np.maximum(nir + red, 1), 0)
    crop = (slice(MARGIN, MARGIN + h), slice(MARGIN, MARGIN + w))
    p = prob[crop]
    mask = binary_opening((p >= threshold) & ~(ndvi[crop] > NDVI_VEG), structure=disk(OPEN_RADIUS))
    return mask, p, tr


def mask_polygons(mask: np.ndarray, tr, utm, clip_utm, min_m2: float) -> list:
    out = []
    for geom, val in shapes(mask.astype(np.uint8), mask=mask, transform=tr):
        g = shape(geom).simplify(CELL, preserve_topology=True).intersection(clip_utm)
        if g.is_empty or g.area < min_m2:
            continue
        for part in getattr(g, "geoms", [g]):
            if part.geom_type == "Polygon" and part.area >= min_m2:
                out.append(part)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaign")
    ap.add_argument("--weights", help="fine-tuned SegFormer model.pt (default: the original pipeline model)")
    ap.add_argument("--model", help="any parking_models.py spec, e.g. vit:data/labels/pilot-laredo/finetune-dinov2-large")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-m2", type=float, default=50.0)
    ap.add_argument("--squares", nargs="*", help="only these squares (others already in the file are kept)")
    ap.add_argument("--no-clean", action="store_true", help="keep guesses on buildings and public streets")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    d = LABELS / a.campaign
    sq = gpd.read_file(d / "squares.geojson")
    utm = sq.estimate_utm_crs()
    sq_utm = sq.to_crs(utm)
    todo = sq_utm[sq_utm["id"].isin(a.squares)] if a.squares else sq_utm
    bundle = load_model(a.weights, a.cpu, a.model)
    path = d / "model_prefill.geojson"
    keep = []
    if a.squares and path.exists():
        keep = [f for f in json.loads(path.read_text())["features"] if f["properties"]["square_id"] not in set(a.squares)]
    feats = list(keep)
    ctx = None if a.no_clean else load_context(d)
    t0 = time.time()
    for n, r in enumerate(todo.itertuples(), 1):
        items = json.loads(r.naip_items) if isinstance(r.naip_items, str) else list(r.naip_items)
        mask, _p, tr = square_prediction(r.geometry, items, utm, bundle, a.threshold)
        if ctx is not None:
            mask = mask & keep_mask(ctx, r.id, utm, tr, mask.shape)
        polys = mask_polygons(mask, tr, utm, r.geometry, a.min_m2)
        ll = gpd.GeoSeries(polys, crs=utm).to_crs(4326) if polys else []
        for g in ll:
            feats.append({"type": "Feature", "geometry": mapping(g),
                          "properties": {"square_id": r.id, "class": "paved", "source": "model"}})
        if n % 10 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} squares ({time.time() - t0:.0f}s)", flush=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats,
                                "properties": {"model": bundle.name, "threshold": a.threshold}}))
    meta = json.loads((d / "campaign.json").read_text())
    meta["model_prefill"] = {"model": bundle.name, "threshold": a.threshold, "polygons": len(feats),
                             "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / "campaign.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"{a.campaign}: {len(feats):,} model polygons -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
