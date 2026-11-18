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

Architectures (parking_models.py): segformer (default; the pipeline's model) or vit (a ViT backbone
such as DINOv3-sat or DINOv2 plus a light head). 15% of the training squares are held back to pick
the threshold (best cleaned IoU), which is saved in config.json and used by bakeoff.py.

Usage:
  python data/scripts/label_finetune.py pilot-laredo --bench          # time one training step
  python data/scripts/label_finetune.py pilot-laredo --epochs 8       # train + evaluate
  python data/scripts/label_finetune.py pilot-laredo --eval-only      # re-score saved model
  python data/scripts/label_finetune.py pilot-laredo --arch vit --backbone facebook/dinov2-large --amp
Writes data/labels/<campaign>/<run>/ (finetune, or finetune-<backbone> for vit): model.pt,
config.json, metrics.json, log.txt.
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
from data.scripts.label_eval import TILE, keep_mask, load_context, read_naip  # noqa: E402

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
    ctx = load_context(d)
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
        # three classes: 0 not parking (incl. garages/rooftops), 1 paved, 2 unpaved; unsure ignored
        tgt3 = np.full(img.shape[1:], IGNORE, np.uint8)
        tgt3[MARGIN:MARGIN + h, MARGIN:MARGIN + w] = np.select(
            [label == 4, label == 1, label == 2], [IGNORE, 1, 2], 0).astype(np.uint8)
        out.append({"square": r.square_id, "stratum": squares[r.square_id]["stratum"], "img": img,
                    "targeted": bool(squares[r.square_id].get("targeted")),
                    "keep": keep_mask(ctx, r.square_id, crs, tr, (h, w)),
                    "target": tgt, "target3": tgt3, "label": label})
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


def score(examples, model, thresholds=(0.3, 0.4, 0.5, 0.6)) -> dict:
    """Pooled precision/recall/IoU on surface parking, plus paved/unpaved recall and by stratum.
    model: parking_models.ParkingModel."""
    res = {t: np.zeros(3, np.int64) for t in thresholds}
    res_clean = {t: np.zeros(3, np.int64) for t in thresholds}
    cls = {t: {"paved": np.zeros(2, np.int64), "unpaved": np.zeros(2, np.int64)} for t in thresholds}
    strata: dict = {}
    three = getattr(model, "n_classes", 2) == 3
    surf = {t: {"paved": np.zeros(3, np.int64), "unpaved": np.zeros(3, np.int64)} for t in thresholds}
    model.train(False)
    for ex in examples:
        if three:
            pc_all = model.predict_classes(ex["img"])[:, MARGIN:-MARGIN, MARGIN:-MARGIN]
            prob = 1.0 - pc_all[0]
            is_unpaved = pc_all[2] > pc_all[1]
        else:
            prob = model.predict(ex["img"])[MARGIN:-MARGIN, MARGIN:-MARGIN]
        lab = ex["label"]
        truth, valid = np.isin(lab, (1, 2)), lab != 4
        for t in thresholds:
            p = prob >= t
            c = np.array([(p & truth & valid).sum(), (p & ~truth & valid).sum(), (~p & truth & valid).sum()])
            res[t] += c
            pc = p & ex["keep"]
            res_clean[t] += np.array([(pc & truth & valid).sum(), (pc & ~truth & valid).sum(),
                                      (~pc & truth & valid).sum()])
            for name, v in (("paved", 1), ("unpaved", 2)):
                cls[t][name] += ((p & (lab == v)).sum(), (lab == v).sum())
            if t == 0.5:
                strata.setdefault(ex["stratum"], np.zeros(3, np.int64))
                strata[ex["stratum"]] += c
            if three:  # surface type among predicted parking (cleaned), vs labelled surface type
                for name, v, sel in (("paved", 1, pc & ~is_unpaved), ("unpaved", 2, pc & is_unpaved)):
                    lt = (lab == v) & valid
                    surf[t][name] += (int((sel & lt).sum()), int((sel & ~lt & valid).sum()), int((~sel & lt).sum()))

    def m(c):
        tp, fp, fn = (int(x) for x in c)
        return {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                "iou": tp / (tp + fp + fn) if tp + fp + fn else None}
    return {"by_threshold": {str(t): {**m(res[t]), **{f"recall_{k}": (int(v[0]) / int(v[1]) if v[1] else None)
                                                     for k, v in cls[t].items()}} for t in thresholds},
            "by_stratum_at_0.5": {s: m(c) for s, c in strata.items()},
            # probability >= threshold minus Overture buildings and public streets (parking_postprocess)
            "clean_by_threshold": {str(t): m(res_clean[t]) for t in thresholds},
            # three-class models only: per-surface precision/recall/IoU (cleaned prediction)
            **({"surface_by_threshold": {str(t): {k: m(v) for k, v in surf[t].items()} for t in thresholds}}
               if three else {})}


# Thresholds picked on a dozen validation squares over-fitted (0.10-0.25) and lost 7-10 points of
# IoU on the PRN downtowns against a plain 0.5 (bake-off, 2026-11-18). Keep 0.5 unless there are
# enough validation squares to trust the choice.
MIN_VAL_FOR_THRESHOLD = 30


def pick_threshold(examples, model) -> tuple[float, float]:
    """Threshold with the best cleaned IoU on validation squares -> (threshold, IoU)."""
    ts = [round(0.10 + 0.05 * k, 2) for k in range(15)]  # 0.10 .. 0.80
    r = score(examples, model, thresholds=ts)["clean_by_threshold"]
    best = max(ts, key=lambda t: r[str(t)]["iou"] or 0)
    return best, r[str(best)]["iou"] or 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaign")
    ap.add_argument("--arch", choices=["segformer", "vit"], default="segformer",
                    help="segformer: the pipeline model (default); vit: a ViT backbone + head (see --backbone)")
    ap.add_argument("--classes", type=int, choices=[2, 3], default=2,
                    help="3 = paved / unpaved / not parking (--arch vit only); parking = paved + unpaved")
    ap.add_argument("--unpaved-weight", type=float, default=2.0,
                    help="--classes 3: loss weight on unpaved pixels (paved outnumbers unpaved ~3.5:1 in Laredo)")
    ap.add_argument("--backbone", default="facebook/dinov3-vitl16-pretrain-sat493m",
                    help="Hugging Face backbone for --arch vit (DINOv3 sat is gated: request access, set HF_TOKEN; "
                         "facebook/dinov2-large is open)")
    ap.add_argument("--unfreeze-last", type=int, default=4,
                    help="--arch vit: train the last N backbone layers plus the head (-1 = whole backbone)")
    ap.add_argument("--name", help="run name (output folder data/labels/<campaign>/<name>); "
                                   "default finetune (segformer) or finetune-<backbone>")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--crops-per-square", type=int, default=2, help="random crops per training square per epoch")
    ap.add_argument("--lr", type=float, default=None, help="default 3e-5 (segformer) / 1e-4 (vit head; backbone x0.1)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--test-share", type=float, default=0.25)
    ap.add_argument("--val-share", type=float, default=0.15,
                    help="share of training squares held back to choose the threshold (not trained on)")
    ap.add_argument("--freeze-stages", type=int, default=2, help="--arch segformer: freeze the first N of 4 stages")
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision on CUDA (about half the memory, faster on RTX cards)")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    import torch
    from data.scripts.parking_models import SegFormerModel, ViTSegModel

    if a.classes == 3 and a.arch != "vit":
        ap.error("--classes 3 needs --arch vit")
    name = a.name or ("finetune" if a.arch == "segformer" else "finetune-" + a.backbone.split("/")[-1]
                      + ("-3class" if a.classes == 3 else ""))
    out = LABELS / a.campaign / name
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    t0 = time.time()
    examples = load_examples(a.campaign)
    # Targeted squares (label_targets.py: picked where the model looked wrong) are never test squares:
    # they are not a random sample, so scoring on them would misstate accuracy.
    test = [e for e in examples if is_test(e["square"], a.test_share) and not e["targeted"]]
    rest = [e for e in examples if e not in test]
    val = [e for e in rest if is_test("val:" + e["square"], a.val_share) and not e["targeted"]] if a.val_share else []
    train = [e for e in rest if e not in val]
    say(f"{a.campaign}: {len(examples)} squares -> train {len(train)}, validation {len(val)}, test {len(test)} "
        f"(imagery ready in {time.time() - t0:.0f}s) | {a.arch} -> {out.relative_to(ROOT)}")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    original = SegFormerModel(cpu=a.cpu)
    if not a.bench:
        base = score(test, original)
        say("original model on test squares: " + json.dumps(base["by_threshold"]["0.5"]))
    if a.eval_only:
        if a.arch == "segformer":
            model = SegFormerModel(str(out / "model.pt"), cpu=a.cpu)
        else:
            model = ViTSegModel.load(out, cpu=a.cpu)
        tuned = score(test, model)
        say("fine-tuned model on test squares: " + json.dumps(tuned["by_threshold"]["0.5"]))
        say("  with buildings and streets removed: " + json.dumps(tuned["clean_by_threshold"]["0.5"]))
        return 0

    if a.arch == "segformer":
        model = original
        model.freeze_stages(a.freeze_stages)
        lr = a.lr or 3e-5
        groups = [{"params": [p_ for p_ in model.parameters() if p_.requires_grad], "lr": lr}]
    else:
        del original
        model = ViTSegModel(a.backbone, cpu=a.cpu, n_classes=a.classes)
        if a.unfreeze_last >= 0:
            model.freeze_backbone(unfreeze_last=a.unfreeze_last)
        lr = a.lr or 1e-4
        groups = [{"params": list(model.head.parameters()), "lr": lr},
                  {"params": [p_ for p_ in model.backbone.parameters() if p_.requires_grad], "lr": lr * 0.1}]
    device = model.device
    params = [p_ for g in groups for p_ in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    rng = random.Random(0)
    steps_per_epoch = max(1, (len(train) * a.crops_per_square) // a.batch)
    total = steps_per_epoch * (1 if a.bench else a.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=total,
                                                pct_start=0.1)
    class_w = torch.tensor([1.0, 1.0, a.unpaved_weight], device=device) if a.classes == 3 else None
    use_amp = a.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device == "cuda":
        say(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB; "
            f"mixed precision {'on' if use_amp else 'off'}")
    for epoch in range(1 if a.bench else a.epochs):
        model.train(True)
        order = [e for e in train for _ in range(a.crops_per_square)]
        rng.shuffle(order)
        losses = []
        for k in range(steps_per_epoch):
            batch = order[k * a.batch:(k + 1) * a.batch] or order[:a.batch]
            crops = [augment(e["img"], e["target3" if a.classes == 3 else "target"], rng) for e in batch]
            ts = time.time()
            tgt = torch.as_tensor(np.stack([c[1] for c in crops]).astype(np.int64)).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model.logits([c[0] for c in crops])
            up = torch.nn.functional.interpolate(logits.float(), size=tgt.shape[-2:], mode="bilinear",
                                                 align_corners=False)
            loss_px = torch.nn.functional.cross_entropy(up, tgt, weight=class_w, ignore_index=IGNORE, reduction="none")
            valid = (tgt != IGNORE).float()
            loss = (loss_px * valid).sum() / valid.sum().clamp(min=1)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss))
            if a.bench:
                say(f"bench: step {k + 1} took {time.time() - ts:.1f}s (batch {a.batch}, "
                    f"{sum(p_.numel() for p_ in params) / 1e6:.0f}M trainable params)")
                if device == "cuda":
                    say(f"bench: peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
                if k >= 2:
                    est = (time.time() - ts) * steps_per_epoch
                    say(f"bench: ~{est / 60:.1f} min per epoch ({steps_per_epoch} steps); "
                        f"{a.epochs} epochs ~{a.epochs * est / 60:.0f} min on this machine")
                    return 0
        say(f"epoch {epoch + 1}/{a.epochs}: mean loss {np.mean(losses):.4f} ({time.time() - t0:.0f}s elapsed)")
        model.save(out, {"campaign": a.campaign})

    if len(val) >= MIN_VAL_FOR_THRESHOLD:
        model.threshold, viou = pick_threshold(val, model)
        say(f"threshold {model.threshold:.2f} chosen on {len(val)} validation squares (clean IoU {viou:.0%})")
    elif val:
        say(f"threshold kept at 0.50: {len(val)} validation squares is too few to choose one "
            f"(needs {MIN_VAL_FOR_THRESHOLD})")
    model.save(out, {"campaign": a.campaign, "train_squares": len(train), "validation_squares": len(val)})
    tuned = score(test, model, thresholds=sorted({0.3, 0.4, 0.5, 0.6, model.threshold}))
    metrics = {"campaign": a.campaign, "arch": a.arch, "model": model.name, "threshold": model.threshold,
               "train_squares": [e["square"] for e in train], "validation_squares": [e["square"] for e in val],
               "test_squares": [e["square"] for e in test], "args": vars(a), "original": base, "fine_tuned": tuned}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    say("\nTest squares, threshold 0.5:      precision  recall  IoU   paved  unpaved")
    for nm, r in (("original", base), ("fine-tuned", tuned)):
        b = r["by_threshold"]["0.5"]
        say(f"  {nm:<12}                 {b['precision']:8.0%} {b['recall']:6.0%} {b['iou']:5.0%} "
            f"{b['recall_paved'] or 0:6.0%} {b['recall_unpaved'] or 0:7.0%}")
    for nm, r in (("original", base), ("fine-tuned", tuned)):
        b = r["clean_by_threshold"]["0.5"]
        say(f"  {nm + ' clean':<28} {b['precision']:8.0%} {b['recall']:6.0%} {b['iou']:5.0%}"
            "   (buildings + streets removed)")
    b = tuned["clean_by_threshold"][str(model.threshold)]
    say(f"  {'fine-tuned clean @' + format(model.threshold, '.2f'):<28} {b['precision']:8.0%} {b['recall']:6.0%} "
        f"{b['iou']:5.0%}   (validation-chosen threshold)")
    if "surface_by_threshold" in tuned:
        sb = tuned["surface_by_threshold"][str(model.threshold)]
        for k in ("paved", "unpaved"):
            say(f"  {k + ' (surface type)':<28} {sb[k]['precision'] or 0:8.0%} {sb[k]['recall'] or 0:6.0%} "
                f"{sb[k]['iou'] or 0:5.0%}   (cleaned prediction, threshold {model.threshold:.2f})")
    say(f"Saved {out.relative_to(ROOT)}/model.pt, config.json and metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
