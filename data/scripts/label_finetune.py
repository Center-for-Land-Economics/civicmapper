#!/usr/bin/env python3
"""
Fine-tune the parking SegFormer on hand labels and compare it with the original on held-out squares.

Data: exported masks (label_export.py) + the squares' NAIP (cached under export/naip/). Target is
the pipeline's two classes: 1 = surface parking (labels paved + unpaved), 0 = everything else;
"unsure" pixels are ignored. Squares are split into train / test by a hash of the square id
(--test-share, default 25%), so the split is stable as more squares are labelled.

Starts from the pipeline's checkpoint (UTEL-UIUC/SegFormer-large-parking, B5) so the result drops
into parking_lot_extraction.py unchanged. Augmentation: random 512 px crops, flips, 90° rotations,
small brightness/contrast jitter.

Usage:
  python data/scripts/label_finetune.py pilot-laredo --bench          # time one training step
  python data/scripts/label_finetune.py pilot-laredo --epochs 8       # train + evaluate
  python data/scripts/label_finetune.py pilot-laredo --eval-only      # re-score saved model
Writes data/labels/<campaign>/finetune/: model.pt (state dict), metrics.json, log.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.scripts.label_eval import TILE, predict, read_naip  # noqa: E402

LABELS = ROOT / "data" / "labels"
MARGIN = 48
IGNORE = 255


def is_test(square_id: str, share: float) -> bool:
    return int(hashlib.sha1(square_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def load_examples(campaign: str) -> list[dict]:
    """[{square, img (4,H+2m,W+2m) uint8, target (H+2m,W+2m) uint8 with IGNORE in margin/unsure}]"""
    d = LABELS / campaign
    exp = d / "export"
    summary = pd.read_csv(exp / "summary.csv")
    squares = {f["properties"]["id"]: f["properties"]
               for f in json.loads((d / "squares.geojson").read_text())["features"]}
    cache = exp / "naip"
    cache.mkdir(exist_ok=True)
    out = []
    for r in summary[summary.outcome.isin(["labeled", "no_parking"])].itertuples():
        masks = sorted((exp / "masks").glob(f"{r.square_id}__*.tif"))
        if not masks:
            continue
        with rasterio.open(masks[0]) as m:
            label, tr, crs, w, h = m.read(1), m.transform, m.crs, m.width, m.height
        cpath = cache / f"{r.square_id}.tif"
        if cpath.exists():
            with rasterio.open(cpath) as c:
                img = c.read()
        else:
            ptr = tr * Affine.translation(-MARGIN, -MARGIN)
            img = read_naip(squares[r.square_id]["naip_items"], crs, ptr, w + 2 * MARGIN, h + 2 * MARGIN)
            with rasterio.open(cpath, "w", driver="GTiff", width=img.shape[2], height=img.shape[1], count=4,
                               dtype="uint8", crs=crs, transform=ptr, compress="deflate") as c:
                c.write(img)
        tgt = np.full(img.shape[1:], IGNORE, np.uint8)
        core = np.where(label == 4, IGNORE, np.isin(label, (1, 2)).astype(np.uint8))
        tgt[MARGIN:MARGIN + h, MARGIN:MARGIN + w] = core
        out.append({"square": r.square_id, "stratum": squares[r.square_id]["stratum"], "img": img,
                    "target": tgt, "label": label})
    return out


def augment(img: np.ndarray, tgt: np.ndarray, rng: random.Random):
    _, H, W = img.shape
    y = rng.randint(0, max(0, H - TILE)); x = rng.randint(0, max(0, W - TILE))
    im, t = img[:3, y:y + TILE, x:x + TILE], tgt[y:y + TILE, x:x + TILE]
    k = rng.randint(0, 3)
    im, t = np.rot90(im, k, axes=(1, 2)), np.rot90(t, k)
    if rng.random() < 0.5:
        im, t = im[:, :, ::-1], t[:, ::-1]
    im = im.astype(np.float32)
    im = (im - im.mean()) * rng.uniform(0.85, 1.15) + im.mean() * rng.uniform(0.9, 1.1)
    return np.clip(im, 0, 255).astype(np.uint8).transpose(1, 2, 0).copy(), t.copy()


def score(examples, processor, model, device, thresholds=(0.3, 0.4, 0.5, 0.6)) -> dict:
    """Pooled precision/recall/IoU on surface parking, plus paved/unpaved recall and by stratum."""
    res = {t: np.zeros(3, np.int64) for t in thresholds}
    cls = {t: {"paved": np.zeros(2, np.int64), "unpaved": np.zeros(2, np.int64)} for t in thresholds}
    strata: dict = {}
    model.eval()
    for ex in examples:
        prob = predict(ex["img"], processor, model, device)[MARGIN:-MARGIN, MARGIN:-MARGIN]
        lab = ex["label"]
        truth, valid = np.isin(lab, (1, 2)), lab != 4
        for t in thresholds:
            p = prob >= t
            c = np.array([(p & truth & valid).sum(), (p & ~truth & valid).sum(), (~p & truth & valid).sum()])
            res[t] += c
            for name, v in (("paved", 1), ("unpaved", 2)):
                cls[t][name] += ((p & (lab == v)).sum(), (lab == v).sum())
            if t == 0.5:
                strata.setdefault(ex["stratum"], np.zeros(3, np.int64))
                strata[ex["stratum"]] += c

    def m(c):
        tp, fp, fn = (int(x) for x in c)
        return {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                "iou": tp / (tp + fp + fn) if tp + fp + fn else None}
    return {"by_threshold": {str(t): {**m(res[t]), **{f"recall_{k}": (int(v[0]) / int(v[1]) if v[1] else None)
                                                     for k, v in cls[t].items()}} for t in thresholds},
            "by_stratum_at_0.5": {s: m(c) for s, c in strata.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaign")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--crops-per-square", type=int, default=2, help="random crops per training square per epoch")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--test-share", type=float, default=0.25)
    ap.add_argument("--freeze-stages", type=int, default=2, help="freeze the first N of 4 encoder stages")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    import torch
    from data.scripts.parking_lot_extraction import load_segformer_bundle

    out = LABELS / a.campaign / "finetune"
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    t0 = time.time()
    examples = load_examples(a.campaign)
    test = [e for e in examples if is_test(e["square"], a.test_share)]
    train = [e for e in examples if not is_test(e["square"], a.test_share)]
    say(f"{a.campaign}: {len(examples)} squares -> train {len(train)}, test {len(test)} "
        f"(imagery ready in {time.time() - t0:.0f}s)")
    processor, model, device, model_id = load_segformer_bundle(use_cpu=a.cpu)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    if not a.bench:
        base = score(test, processor, model, device)
        say("original model on test squares: " + json.dumps(base["by_threshold"]["0.5"]))
    if a.eval_only:
        model.load_state_dict(torch.load(out / "model.pt", map_location="cpu"))
        tuned = score(test, processor, model, device)
        say("fine-tuned model on test squares: " + json.dumps(tuned["by_threshold"]["0.5"]))
        return 0

    enc = model.segformer.encoder
    for i in range(a.freeze_stages):
        for mod in (enc.patch_embeddings[i], enc.block[i], enc.layer_norm[i]):
            for p_ in mod.parameters():
                p_.requires_grad = False
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    rng = random.Random(0)
    steps_per_epoch = max(1, (len(train) * a.crops_per_square) // a.batch)
    total = steps_per_epoch * (1 if a.bench else a.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    step = 0
    for epoch in range(1 if a.bench else a.epochs):
        model.train()
        order = [e for e in train for _ in range(a.crops_per_square)]
        rng.shuffle(order)
        losses = []
        for k in range(steps_per_epoch):
            batch = order[k * a.batch:(k + 1) * a.batch] or order[:a.batch]
            crops = [augment(e["img"], e["target"], rng) for e in batch]
            ts = time.time()
            inputs = processor(images=[c[0] for c in crops], return_tensors="pt", do_resize=False).to(device)
            tgt = torch.as_tensor(np.stack([c[1] for c in crops]).astype(np.int64)).to(device)
            logits = model(**inputs).logits
            up = torch.nn.functional.interpolate(logits, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
            loss_px = torch.nn.functional.cross_entropy(up, tgt, ignore_index=IGNORE, reduction="none")
            valid = (tgt != IGNORE).float()
            loss = (loss_px * valid).sum() / valid.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step(); sched.step(); step += 1
            losses.append(float(loss))
            if a.bench:
                say(f"bench: step {k + 1} took {time.time() - ts:.1f}s (batch {a.batch}, "
                    f"{sum(p_.numel() for p_ in params) / 1e6:.0f}M trainable params)")
                if k >= 2:
                    est = (time.time() - ts) * steps_per_epoch
                    say(f"bench: ~{est / 60:.0f} min per epoch ({steps_per_epoch} steps); "
                        f"8 epochs ~{8 * est / 3600:.1f} h on this machine")
                    return 0
        say(f"epoch {epoch + 1}/{a.epochs}: mean loss {np.mean(losses):.4f} ({time.time() - t0:.0f}s elapsed)")
        torch.save(model.state_dict(), out / "model.pt")

    tuned = score(test, processor, model, device)
    metrics = {"campaign": a.campaign, "base_model": model_id, "train_squares": [e["square"] for e in train],
               "test_squares": [e["square"] for e in test], "args": vars(a), "original": base, "fine_tuned": tuned}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    say("\nTest squares, threshold 0.5:      precision  recall  IoU   paved  unpaved")
    for name, r in (("original", base), ("fine-tuned", tuned)):
        b = r["by_threshold"]["0.5"]
        say(f"  {name:<12}                 {b['precision']:8.0%} {b['recall']:6.0%} {b['iou']:5.0%} "
            f"{b['recall_paved'] or 0:6.0%} {b['recall_unpaved'] or 0:7.0%}")
    say(f"Saved {out / 'model.pt'} and metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
