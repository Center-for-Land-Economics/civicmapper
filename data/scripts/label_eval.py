#!/usr/bin/env python3
"""
Score the parking segmentation model against hand labels.

For each exported label mask (label_export.py -> data/labels/<campaign>/export/masks/*.tif) this
reads the square's NAIP imagery (Planetary Computer, same items the labeller saw) onto the mask's
grid with a context margin, runs the same model and windowing as parking_lot_extraction.py
(UTEL-UIUC/SegFormer-large-parking, 512 px windows), and compares surface parking (labels 1 paved +
2 unpaved) with the prediction. "Unsure" label pixels are ignored.

Variants scored at each threshold:
  raw       probability >= threshold
  pipeline  raw, minus NDVI > 0.20 vegetation, then morphological opening (what the pipeline ships)

Writes data/labels/<campaign>/eval/: <square>__<labeller>.png (imagery | label | prediction |
errors: green hit, red false alarm, blue miss), probs/<...>.tif, and metrics.csv. Prints pooled
precision / recall / IoU, by stratum and by labelled class.

Usage:
  python data/scripts/label_eval.py pilot-laredo
  python data/scripts/label_eval.py pilot-laredo --cpu --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LABELS = ROOT / "data" / "labels"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
TILE = 512
NDVI_VEG = 0.20
OPEN_RADIUS = 3
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
# Behind an intercepting proxy (the cloud sandbox) GDAL needs the proxy's CA bundle; elsewhere
# (e.g. Windows) leave GDAL's own certificate handling alone.
if Path("/root/.ccr/ca-bundle.crt").exists():
    os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")


_token: dict = {}


def signed_href(item_id: str) -> str:
    if not _token or _token["exp"] < time.time():
        r = requests.get(f"{STAC.replace('/stac/v1', '')}/sas/v1/token/naip", timeout=60).json()
        _token.update(tok=r["token"], exp=time.time() + 30 * 60)
    it = requests.get(f"{STAC}/collections/naip/items/{item_id}", timeout=60).json()
    return f"{it['assets']['image']['href']}?{_token['tok']}"


def read_naip(items: list[str], crs, transform: Affine, w: int, h: int) -> np.ndarray:
    """4-band NAIP warped onto the target grid; later items fill gaps left by earlier ones."""
    out = np.zeros((4, h, w), dtype=np.uint8)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2"):
        for item in items:
            with rasterio.open(signed_href(item)) as src, WarpedVRT(
                    src, crs=crs, transform=transform, width=w, height=h,
                    resampling=Resampling.bilinear, nodata=0) as vrt:
                arr = vrt.read()
            empty = out[:3].sum(axis=0) == 0
            out[:, empty] = arr[:, empty]
            if not (out[:3].sum(axis=0) == 0).any():
                break
    return out


def predict(img: np.ndarray, processor, model, device) -> np.ndarray:
    """Parking probability for a 4-band image (any size >= 1 px), 512 px windows with overlap."""
    import torch
    _, h, w = img.shape
    ph, pw = max(h, TILE), max(w, TILE)
    if (ph, pw) != (h, w):
        img = np.pad(img, ((0, 0), (0, ph - h), (0, pw - w)), mode="reflect")
    ys = sorted({0, max(0, ph - TILE)} | set(range(0, max(1, ph - TILE), TILE - 64)))
    xs = sorted({0, max(0, pw - TILE)} | set(range(0, max(1, pw - TILE), TILE - 64)))
    acc = np.zeros((ph, pw), np.float32)
    wt = np.zeros((ph, pw), np.float32)
    patches = [(y, x, img[:3, y:y + TILE, x:x + TILE].transpose(1, 2, 0)) for y in ys for x in xs]
    for k in range(0, len(patches), 4):
        batch = patches[k:k + 4]
        inputs = processor(images=[p[2] for p in batch], return_tensors="pt", do_resize=False).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        up = torch.nn.functional.interpolate(logits, size=(TILE, TILE), mode="bilinear", align_corners=False)
        probs = torch.softmax(up, dim=1)[:, 1].cpu().numpy()
        for (y, x, _), pr in zip(batch, probs):
            acc[y:y + TILE, x:x + TILE] += pr
            wt[y:y + TILE, x:x + TILE] += 1
    return (acc / np.maximum(wt, 1))[:h, :w]


def disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def overlay(rgb: np.ndarray, mask: np.ndarray, colors: dict[int, tuple]) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    for v, c in colors.items():
        m = mask == v
        out[m] = out[m] * 0.45 + np.array(c, np.float32) * 0.55
    return out.clip(0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaign")
    ap.add_argument("--margin-px", type=int, default=48, help="context around the square for the model")
    ap.add_argument("--threshold", type=float, default=0.5, help="threshold for the pictures")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from scipy.ndimage import binary_opening
    from data.scripts.parking_lot_extraction import load_segformer_bundle

    d = LABELS / a.campaign
    exp = d / "export"
    summary = pd.read_csv(exp / "summary.csv")
    squares = {f["properties"]["id"]: f["properties"] for f in json.loads((d / "squares.geojson").read_text())["features"]}
    out = d / "eval"
    (out / "probs").mkdir(parents=True, exist_ok=True)
    processor, model, device, model_id = load_segformer_bundle(use_cpu=a.cpu)

    todo = summary[summary.outcome.isin(["labeled", "no_parking"])]
    if a.limit:
        todo = todo.head(a.limit)
    counts = defaultdict(lambda: np.zeros(3, np.int64))  # key -> [tp, fp, fn]
    rows = []
    kernel = disk(OPEN_RADIUS)
    for n, r in enumerate(todo.itertuples(), 1):
        stem = f"{r.square_id}__{''.join(ch if ch.isalnum() else '-' for ch in r.labeler.lower()).strip('-')}"
        mpath = exp / "masks" / f"{stem}.tif"
        if not mpath.exists():
            cand = sorted((exp / "masks").glob(f"{r.square_id}__*.tif"))
            if not cand:
                print(f"  {r.square_id}: no mask, skipped")
                continue
            mpath = cand[0]
        with rasterio.open(mpath) as m:
            label, tr, crs, w, h = m.read(1), m.transform, m.crs, m.width, m.height
        mg = a.margin_px
        ptr = tr * Affine.translation(-mg, -mg)
        t0 = time.time()
        img = read_naip(squares[r.square_id]["naip_items"], crs, ptr, w + 2 * mg, h + 2 * mg)
        prob = predict(img, processor, model, device)
        nir, red = img[3].astype(np.float32), img[0].astype(np.float32)
        ndvi = np.where(nir + red > 0, (nir - red) / np.maximum(nir + red, 1), 0)
        crop = (slice(mg, mg + h), slice(mg, mg + w))
        p, veg = prob[crop], (ndvi > NDVI_VEG)[crop]
        with rasterio.open(out / "probs" / f"{stem}.tif", "w", driver="GTiff", width=w, height=h, count=1,
                           dtype="uint8", crs=crs, transform=tr, compress="deflate") as ds:
            ds.write(np.rint(p * 255).astype(np.uint8), 1)

        truth = np.isin(label, (1, 2))
        valid = label != 4
        stratum = squares[r.square_id]["stratum"]
        for thr in THRESHOLDS:
            raw = p >= thr
            pipe = binary_opening(raw & ~veg, structure=kernel)
            for name, pred in (("raw", raw), ("pipeline", pipe)):
                tp = int((pred & truth & valid).sum())
                fp = int((pred & ~truth & valid).sum())
                fn = int((~pred & truth & valid).sum())
                for key in ((name, thr, "all"), (name, thr, f"stratum:{stratum}")):
                    counts[key] += (tp, fp, fn)
                for cls, v in (("paved", 1), ("unpaved", 2)):
                    t = label == v
                    counts[(name, thr, f"class:{cls}")] += (int((pred & t).sum()), 0, int((~pred & t).sum()))
                if name == "pipeline" and abs(thr - a.threshold) < 1e-9:
                    rows.append({"square_id": r.square_id, "labeler": r.labeler, "stratum": stratum,
                                 "outcome": r.outcome, "label_m2": round(truth.sum() * 0.36),
                                 "pred_m2": round(pred.sum() * 0.36), "tp_m2": round(tp * 0.36),
                                 "recall": tp / (tp + fn) if tp + fn else None,
                                 "precision": tp / (tp + fp) if tp + fp else None})
        # picture: imagery | label | prediction (pipeline @ threshold) | errors
        rgb = img[:3][(slice(None),) + crop].transpose(1, 2, 0)
        pred = binary_opening((p >= a.threshold) & ~veg, structure=kernel)
        err = np.zeros_like(label)
        err[pred & truth] = 1
        err[pred & ~truth & valid] = 2
        err[~pred & truth] = 3
        panels = [rgb,
                  overlay(rgb, label, {1: (255, 77, 109), 2: (244, 162, 89), 3: (179, 136, 255), 4: (255, 224, 102)}),
                  overlay(rgb, pred.astype(np.uint8), {1: (0, 200, 255)}),
                  overlay(rgb, err, {1: (40, 200, 90), 2: (230, 40, 40), 3: (40, 90, 255)})]
        gap = np.full((h, 6, 3), 255, np.uint8)
        Image.fromarray(np.concatenate(sum([[pn, gap] for pn in panels], [])[:-1], axis=1)).save(out / f"{stem}.png")
        rr = rows[-1] if rows else {}
        print(f"  [{n}/{len(todo)}] {r.square_id} {stratum:<11} label {rr.get('label_m2', 0):>7,} m²  "
              f"pred {rr.get('pred_m2', 0):>7,} m²  recall {rr.get('recall') or 0:.0%}  "
              f"precision {rr.get('precision') or 0:.0%}  ({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    def fmt(c):
        tp, fp, fn = c
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
        return prec, rec, iou

    print(f"\nModel {model_id} on {len(rows)} labelled squares ({a.campaign}); surface parking = paved + unpaved")
    print("variant   thr   precision  recall   IoU")
    for name in ("raw", "pipeline"):
        for thr in THRESHOLDS:
            prec, rec, iou = fmt(counts[(name, thr, "all")])
            print(f"{name:<9} {thr:.1f}   {prec:8.0%}  {rec:6.0%}  {iou:5.0%}")
    print(f"\nPipeline @ {a.threshold}: recall by labelled class / precision-recall by stratum")
    for key in sorted(k for k in counts if k[0] == "pipeline" and abs(k[1] - a.threshold) < 1e-9 and k[2] != "all"):
        prec, rec, iou = fmt(counts[key])
        if key[2].startswith("class:"):
            print(f"  {key[2]:<22} recall {rec:.0%}")
        else:
            print(f"  {key[2]:<22} precision {prec:.0%}  recall {rec:.0%}  IoU {iou:.0%}")
    print(f"\nPictures: {out.relative_to(ROOT)}/<square>__<labeller>.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
