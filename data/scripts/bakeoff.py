#!/usr/bin/env python3
"""
Parking-model bake-off: every model on the same test sets, same clean-up, one leaderboard.

Test sets
  home       held-out test squares of the training campaign(s) (--campaigns; the label_finetune.py
             split: 25% of random squares by hash, targeted squares never included)
  transfer   every labelled random square of campaigns no model trained on (--transfer, e.g.
             pilot-plano labelled in Blind mode): how a model does in a city it has not seen
  prn        Parking Reform Network downtowns (prn_eval.py; 8 Texas cities by default)
Rules
  each model uses its own threshold (chosen on validation squares by label_finetune.py; 0.5 for
  the original); predictions get the pipeline recipe (NDVI strip, opening) and the clean-up
  (Overture buildings and public streets removed, parking_postprocess.py), as the labels do.
Score
  overall = mean of the test sets' IoU (home, transfer, PRN median); tie-breakers in the table:
  gravel recall, and inference seconds per km² (citywide cost).

Models are parking_models.py specs; with no --models, every trained run under data/labels/*/
(finetune*/config.json) plus the original SegFormer.

Usage:
  python data/scripts/bakeoff.py
  python data/scripts/bakeoff.py --models segformer segformer:data/labels/pilot-laredo/finetune/model.pt \\
      vit:data/labels/pilot-laredo/finetune-dinov3-vitl16-pretrain-sat493m \\
      "sam2:segformer:data/labels/pilot-laredo/finetune/model.pt" --transfer pilot-plano
Writes data/bakeoff/leaderboard.csv / .md and compare_<square>.png (one row per model).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.scripts.label_finetune import MARGIN, is_test, load_examples, score  # noqa: E402

LABELS = ROOT / "data" / "labels"
OUT = ROOT / "data" / "bakeoff"
SQUARE_KM2 = 0.0625


def discover_models() -> list[str]:
    specs = ["segformer"]
    for cfg in sorted(LABELS.glob("*/finetune*/config.json")):
        arch = json.loads(cfg.read_text()).get("arch")
        if arch == "segformer" and (cfg.parent / "model.pt").exists():
            specs.append(f"segformer:{cfg.parent.relative_to(ROOT) / 'model.pt'}")
        elif arch == "vit":
            specs.append(f"vit:{cfg.parent.relative_to(ROOT)}")
    return specs


def example_strip(examples, model, thr) -> dict:
    """square -> RGB strip (imagery | label | clean prediction) for the comparison pictures."""
    out = {}
    for ex in examples:
        rgb = ex["img"][:3, MARGIN:-MARGIN, MARGIN:-MARGIN].transpose(1, 2, 0).astype(np.float32)
        p = (model.predict(ex["img"])[MARGIN:-MARGIN, MARGIN:-MARGIN] >= thr) & ex["keep"]
        lab = np.isin(ex["label"], (1, 2))
        panels = [rgb.copy(), rgb.copy(), rgb.copy()]
        panels[1][lab] = panels[1][lab] * 0.45 + np.array((255, 77, 109)) * 0.55
        err = panels[2]
        for mk, col in ((p & lab, (40, 200, 90)), (p & ~lab, (230, 40, 40)), (~p & lab, (40, 90, 255))):
            err[mk] = err[mk] * 0.45 + np.array(col) * 0.55
        gap = np.full((rgb.shape[0], 6, 3), 255, np.float32)
        out[ex["square"]] = np.concatenate([panels[0], gap, panels[1], gap, err], axis=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="*", help="parking_models.py specs (default: discovered runs + original)")
    ap.add_argument("--campaigns", nargs="*", default=["pilot-laredo"], help="home campaigns (test squares)")
    ap.add_argument("--transfer", nargs="*", default=[], help="campaigns scored on all labelled random squares")
    ap.add_argument("--prn", nargs="*", default=None, help="PRN city ids (default: our 8 Texas cities; 'none' to skip)")
    ap.add_argument("--examples", type=int, default=4, help="home test squares to picture (most parking first)")
    ap.add_argument("--test-share", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=None,
                    help="score every model at this threshold instead of its own (e.g. 0.5 for a like-for-like check)")
    ap.add_argument("--fresh", action="store_true",
                    help="start a new leaderboard instead of merging into data/bakeoff/leaderboard.csv")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    from data.scripts.parking_models import load_spec
    from data.scripts import prn_eval
    specs = a.models or discover_models()
    prn_cities = [] if a.prn == ["none"] else (a.prn or prn_eval.TEXAS_OURS)
    OUT.mkdir(parents=True, exist_ok=True)

    home, transfer = [], {}
    for c in a.campaigns:
        home += [e for e in load_examples(c) if is_test(e["square"], a.test_share) and not e["targeted"]]
    for c in a.transfer:
        transfer[c] = [e for e in load_examples(c) if not e["targeted"]]
    shown = sorted(home, key=lambda e: -int(np.isin(e["label"], (1, 2)).sum()))[:a.examples]
    prn_ctx = {city: prn_eval.prepare_city(city) for city in prn_cities}
    print(f"Test sets: home {len(home)} squares ({', '.join(a.campaigns)}); "
          f"transfer {', '.join(f'{k} {len(v)}' for k, v in transfer.items()) or 'none'}; "
          f"PRN {len(prn_cities)} downtowns. Models: {len(specs)}", flush=True)

    rows, strips, failed = [], {}, []
    for spec in specs:
      try:
        t0 = time.time()
        model = load_spec(spec, cpu=a.cpu)
        thr = model.threshold if a.threshold is None else a.threshold
        row = {"model": model.name + ("" if a.threshold is None else f"@{thr:.2f}"), "spec": spec, "threshold": thr}
        secs, km2 = 0.0, 0.0
        ts = time.time()
        r = score(home, model, thresholds=[thr])
        secs += time.time() - ts
        km2 += len(home) * SQUARE_KM2 * (1 + 2 * MARGIN / 417) ** 2
        c, raw = r["clean_by_threshold"][str(thr)], r["by_threshold"][str(thr)]
        row.update({"home_iou": c["iou"], "home_precision": c["precision"], "home_recall": c["recall"],
                    "home_paved_recall": raw["recall_paved"], "home_gravel_recall": raw["recall_unpaved"]})
        if "surface_by_threshold" in r:  # three-class models: is each surface type right?
            sb = r["surface_by_threshold"][str(thr)]
            row.update({"home_surface_paved_iou": sb["paved"]["iou"], "home_surface_unpaved_iou": sb["unpaved"]["iou"],
                        "home_surface_unpaved_precision": sb["unpaved"]["precision"],
                        "home_surface_unpaved_recall": sb["unpaved"]["recall"]})
        for name, exs in transfer.items():
            ts = time.time()
            tr = score(exs, model, thresholds=[thr])["clean_by_threshold"][str(thr)]
            secs += time.time() - ts
            km2 += len(exs) * SQUARE_KM2
            row.update({f"{name}_iou": tr["iou"], f"{name}_precision": tr["precision"], f"{name}_recall": tr["recall"]})
        prn_rows = []
        for city, ctx in prn_ctx.items():
            pred, rgb, s = prn_eval.run_model(ctx, model, thr)
            secs += s
            km2 += len(ctx["tiles"]) * SQUARE_KM2
            prn_rows += [x for x in prn_eval.score_variants(ctx, model.name, pred) if x["model"].endswith("-clean")]
            prn_eval.overview(ctx, rgb, pred & ~ctx["built"] & ~ctx["street"],
                              OUT / "prn" / city / f"{prn_eval.safe(model.name)}.png")
        if prn_rows:
            pr = pd.DataFrame(prn_rows)
            row.update({"prn_median_iou": pr.iou.median(), "prn_median_precision": pr.precision.median(),
                        "prn_median_recall": pr.recall.median()})
            row.update({f"prn_{x['city']}": x["iou"] for x in prn_rows})
        sets = [row.get("home_iou")] + [row.get(f"{n}_iou") for n in transfer] + [row.get("prn_median_iou")]
        sets = [v for v in sets if v is not None and not pd.isna(v)]
        row["overall_iou"] = float(np.mean(sets)) if sets else None
        row["sec_per_km2"] = secs / km2 if km2 else None
        strips[row["model"]] = example_strip(shown, model, thr)
        rows.append(row)
        print(f"{row['model']}: overall IoU {row['overall_iou']:.0%} (home {row['home_iou']:.0%}"
              + "".join(f", {n} {row[f'{n}_iou']:.0%}" for n in transfer)
              + (f", PRN {row['prn_median_iou']:.0%}" if prn_rows else "")
              + f"); {row['sec_per_km2']:.0f} s/km²; {time.time() - t0:.0f}s", flush=True)
        del model
      except Exception as e:  # one broken model must not lose the others' results
        import traceback
        traceback.print_exc()
        failed.append(spec)
        print(f"!! {spec} failed ({type(e).__name__}: {e}); continuing with the other models", flush=True)

    df = pd.DataFrame(rows)
    lb = OUT / "leaderboard.csv"
    if lb.exists() and not a.fresh:  # merge with earlier runs; a re-scored model replaces its old row
        old = pd.read_csv(lb)
        old = old[~old["model"].isin(df["model"] if len(df) else [])]
        df = pd.concat([old, df], ignore_index=True)
    if df.empty:
        print("No model was scored.")
        return 1
    df = df.sort_values("overall_iou", ascending=False)
    df.to_csv(OUT / "leaderboard.csv", index=False)
    pct = lambda v: f"{v:.0%}" if v is not None and not pd.isna(v) else "-"  # noqa: E731
    cols = ["model", "overall_iou", "home_iou", "home_precision", "home_recall", "home_gravel_recall"] + \
        [f"{n}_iou" for n in transfer] + (["prn_median_iou", "prn_median_precision", "prn_median_recall"]
                                          if prn_cities else []) + \
        (["home_surface_unpaved_iou"] if "home_surface_unpaved_iou" in df.columns else []) + ["threshold", "sec_per_km2"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(
            str(r[c2]) if c2 == "model" else (f"{r[c2]:.2f}" if c2 == "threshold" else
                                              (f"{r[c2]:.0f}" if c2 == "sec_per_km2" else pct(r[c2]))) for c2 in cols) + " |")
    (OUT / "leaderboard.md").write_text("\n".join(md) + "\n")
    for ex in shown:  # one picture per square: a row per model scored in this run (imagery | label | errors)
        rows_img = [strips[m][ex["square"]] for m in df.model if m in strips and ex["square"] in strips[m]]
        if not rows_img:
            continue
        gap = np.full((8, rows_img[0].shape[1], 3), 255, np.float32)
        Image.fromarray(np.concatenate(sum([[x, gap] for x in rows_img], [])[:-1], axis=0)
                        .clip(0, 255).astype(np.uint8)).save(OUT / f"compare_{ex['square']}.png")
    print("\n" + "\n".join(md))
    print(f"\nRows in compare_*.png follow the leaderboard order: {', '.join(df.model)}")
    print(f"Written to {OUT.relative_to(ROOT)}/")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
