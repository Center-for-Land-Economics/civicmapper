#!/usr/bin/env python3
"""
Guided runner for the remaining parking-model work: one step at a time, on the local GPU.

Phase 1 (DINOv2): sync, rebuild masks, train and score the paved/unpaved (3-class) model, pick the
model for the citywide run, run Laredo citywide, back everything up to Azure.
Phase 2 (DINOv3): check access, memory test, train 2-class and 3-class DINOv3, score, and (if it
wins) rerun Laredo citywide with it.

Each step shows why it matters, the exact command and a time estimate, then asks:
  Enter = run it, s = skip, d = mark done (you already did it), q = quit.
Progress is kept in data/labels/.parking_next.json, so you can quit any time and pick up later.

Usage (from the repo root, GPU environment active; parking-next.bat does both for you):
  python data/scripts/parking_next.py            # continue where you left off
  python data/scripts/parking_next.py --list     # show all steps and their status
  python data/scripts/parking_next.py --from v2-citywide   # jump to a step (by id)
  python data/scripts/parking_next.py --dry-run  # print the commands without running anything
  python data/scripts/parking_next.py --reset    # forget progress
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labels"
STATE = LABELS / ".parking_next.json"
ENV_FILE = ROOT / "data" / ".env"
LEADERBOARD = ROOT / "data" / "bakeoff" / "leaderboard.csv"
CAMPAIGN = "pilot-laredo"
CITY = "laredo"
PY = sys.executable
S = "data/scripts/"

V2 = "facebook/dinov2-large"
V3 = "facebook/dinov3-vitl16-pretrain-sat493m"
RUN_V2_TARGETED = f"data/labels/{CAMPAIGN}/finetune-dinov2-large-targeted"
RUN_V2_3CLASS = f"data/labels/{CAMPAIGN}/finetune-dinov2-large-3class"
RUN_V3 = f"data/labels/{CAMPAIGN}/finetune-dinov3-vitl16-pretrain-sat493m"
RUN_V3_3CLASS = f"data/labels/{CAMPAIGN}/finetune-dinov3-vitl16-pretrain-sat493m-3class"
TRAIN = ["--arch", "vit", "--amp", "--batch", "2", "--epochs", "8"]


# ----------------------------------------------------------------------------------------------- state

def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"done": {}, "skipped": {}, "choice": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def child_env() -> dict:
    env = dict(os.environ)
    if ENV_FILE.exists():  # HF_TOKEN / Azure token may live here; never printed
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    env["PYTHONUTF8"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return env


# ----------------------------------------------------------------------------------------------- steps

@dataclass
class Step:
    id: str
    phase: str
    title: str
    why: str
    estimate: str
    cmds: Callable[[dict], list[list[str]]] | None = None  # commands, built when the step runs
    action: Callable[[dict, bool], bool] | None = None      # in-process step (checks, choices)
    after: str = ""                                          # what to do with the output
    optional: bool = False
    done_if: Callable[[], bool] | None = field(default=None)  # hint: output already exists


def has_model(run: str) -> bool:
    return (ROOT / run / "model.pt").exists()


def leaderboard() -> list[dict]:
    if not LEADERBOARD.exists():
        return []
    with open(LEADERBOARD, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def on_leaderboard(run: str) -> bool:
    return any(Path(r.get("spec", "").split(":", 1)[-1]).as_posix().rstrip("/") == run for r in leaderboard())


def pct(v: str) -> str:
    try:
        return f"{float(v):.0%}"
    except (TypeError, ValueError):
        return "-"


# --- in-process actions -----------------------------------------------------------------------------

def check_env(st: dict, dry: bool) -> bool:
    ok = True

    def bad(msg: str, fix: str = "") -> None:
        nonlocal ok
        ok = False
        print(f"  [X] {msg}" + (f"\n      fix: {fix}" if fix else ""))

    print(f"  Python {sys.version.split()[0]} at {PY}")
    if ".venv-gpu" not in sys.prefix:
        print("  [!] not running inside .venv-gpu (fine if this environment has torch with CUDA)")
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"  [ok] GPU {p.name}, {p.total_memory / 2**30:.0f} GB")
        else:
            bad("torch sees no CUDA GPU",
                "pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall")
    except ImportError:
        bad("torch not installed", "pip install torch --index-url https://download.pytorch.org/whl/cu124")
    try:
        import transformers
        if transformers.__version__ == "4.57.6":
            print("  [ok] transformers 4.57.6")
        else:
            bad(f"transformers {transformers.__version__} (needs 4.57.6 for DINOv3/SAM 2.1)",
                'pip install "transformers==4.57.6"')
    except ImportError:
        bad("transformers not installed", 'pip install "transformers==4.57.6"')
    for mod, pkg in [("azure.storage.blob", "azure-storage-blob"), ("rasterio", "rasterio"),
                     ("geopandas", "geopandas"), ("pyarrow", "pyarrow"), ("scipy", "scipy"),
                     ("huggingface_hub", "huggingface_hub")]:
        try:
            __import__(mod)
        except ImportError:
            bad(f"{pkg} missing", f"pip install {pkg}")
    env = child_env()
    if env.get("AZURE_PARKING_LABELS_CONNECTION_STRING"):
        print("  [ok] Azure label-sync token found")
    else:
        bad("no AZURE_PARKING_LABELS_CONNECTION_STRING",
            r"add the line AZURE_PARKING_LABELS_CONNECTION_STRING=... to data\.env")
    print("  " + ("Everything needed is in place." if ok else "Fix the [X] items, then run this again."))
    return ok


def choose_model(phase_runs: list[str], key: str) -> Callable[[dict, bool], bool]:
    def act(st: dict, dry: bool) -> bool:
        rows = [r for r in leaderboard() if r.get("spec", "").startswith("vit:")]
        if not rows:
            print("  No ViT models on data/bakeoff/leaderboard.csv yet: run the bake-off step first.")
            return False
        rows.sort(key=lambda r: float(r.get("overall_iou") or 0), reverse=True)
        print(f"  {'#':>2}  {'overall':>7}  {'home':>5}  {'prec':>5}  {'PRN':>5}  model")
        for i, r in enumerate(rows[:8], 1):
            print(f"  {i:>2}  {pct(r.get('overall_iou')):>7}  {pct(r.get('home_iou')):>5}  "
                  f"{pct(r.get('home_precision')):>5}  {pct(r.get('prn_median_iou')):>5}  "
                  f"{r['spec'].split(':', 1)[1]}")
        best = 1
        top = float(rows[0]["overall_iou"])
        for run in phase_runs:  # in preference order (3-class first): take it if within 1 point of the top
            hit = [i for i, r in enumerate(rows[:8], 1)
                   if r["spec"].rstrip("/").endswith(run) and float(r["overall_iou"]) >= top - 0.01]
            if hit:
                best = hit[0]
                break
        print(f"\n  Suggested: #{best}. A 3-class model also tags lots paved/unpaved; take it if it is within"
              f"\n  about 1 point of the best 2-class model.")
        if dry:
            return True
        pick = input(f"  Model for the citywide run [Enter = #{best}, or a number]: ").strip()
        try:
            r = rows[int(pick) - 1] if pick else rows[best - 1]
        except (ValueError, IndexError):
            print("  Not a number on the list.")
            return False
        st["choice"][key] = r["spec"]
        print(f"  Citywide model: {r['spec']}")
        return True
    return act


def check_dinov3_access(st: dict, dry: bool) -> bool:
    env = child_env()
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(V3, "config.json", token=env.get("HF_TOKEN") or None)
        print(f"  [ok] DINOv3 satellite weights reachable ({Path(p).parent.name})")
        return True
    except Exception as e:  # gated repo, no token, network
        print(f"  [X] cannot read {V3}: {type(e).__name__}")
        print("      1. On https://huggingface.co/settings/tokens make a READ token.")
        print(r"      2. Add the line HF_TOKEN=... to data\.env (or run: hf auth login).")
        print("      3. Make sure the model page says your access request was granted.")
        print("      Never paste the token into chat or commit it.")
        return False


def citywide_cmds(key: str, fallback: str) -> Callable[[dict], list[list[str]]]:
    def build(st: dict) -> list[list[str]]:
        spec = st["choice"].get(key) or f"vit:{fallback}"
        run = Path(spec.split(":", 1)[1]).name
        out = ROOT / "data" / "parking-model" / CITY / run
        cmds = []
        prev = [p for p in sorted((ROOT / "data" / "parking-model" / CITY).glob("*/naip_imagery.tif"))
                if p.parent != out]
        if prev and not (out / "naip_imagery.tif").exists():  # reuse the city's NAIP mosaic (same grid)
            cmds.append([PY, "-c", "import shutil,sys,pathlib; pathlib.Path(sys.argv[2]).parent.mkdir("
                         "parents=True, exist_ok=True); shutil.copy2(sys.argv[1], sys.argv[2])",
                         str(prev[0].relative_to(ROOT)), str((out / "naip_imagery.tif").relative_to(ROOT))])
        return cmds + [[PY, S + "parking_lot_extraction.py", "--city", CITY, "--model", spec,
                 "--output-dir", f"data/parking-model/{CITY}/{run}"]]
    return build


def bakeoff_cmds(runs: list[str], also: list[str]) -> Callable[[dict], list[list[str]]]:
    def build(st: dict) -> list[list[str]]:
        models = [f"vit:{r}" for r in runs if has_model(r)]
        models += [f"vit:{r}" for r in also if has_model(r) and not on_leaderboard(r)]  # reference points
        if not models:
            models = [f"vit:{r}" for r in runs]
        cmd = [PY, S + "bakeoff.py", "--models", *models]
        if any((LABELS / "pilot-plano" / "labels").glob("*.json")):
            cmd += ["--transfer", "pilot-plano"]
        return [cmd]
    return build


def info(lines: list[str]) -> Callable[[dict, bool], bool]:
    def act(st: dict, dry: bool) -> bool:
        for ln in lines:
            print("  " + ln)
        if dry:
            return True
        return input("\n  Done with this? [y = mark done, Enter = come back later]: ").strip().lower() == "y"
    return act


STEPS: list[Step] = [
    # ---------------------------------------------------------------- phase 1: DINOv2
    Step("check", "DINOv2", "Check the GPU environment",
         "Catches a missing package or token now instead of 20 minutes into a run.", "seconds",
         action=check_env),
    Step("git-pull", "DINOv2", "Get the latest scripts",
         "Brings in this runner's companions (label_sync --models, the GDAL-free NAIP mosaic).", "seconds",
         cmds=lambda st: [["git", "pull", "--ff-only"]],
         after="If git refuses because of local edits, run `git stash`, then retry this step."),
    Step("sync-pull", "DINOv2", "Download the newest labels and models from Azure",
         "Anything labelled or trained on your other machine comes down first.", "<1 min (models: a few min once)",
         cmds=lambda st: [[PY, S + "label_sync.py", "--pull", "--models"]]),
    Step("backup-1", "DINOv2", "Back up the models you already trained",
         "model.pt files only existed on one laptop. This copies every finetune*/ run to Azure "
         "(skips ones already there).", "1-5 min (each model ~300-1200 MB)",
         cmds=lambda st: [[PY, S + "label_sync.py", "--models"]]),
    Step("export", "DINOv2", "Rebuild the training masks from the labels",
         "Picks up every label so far, with buildings and streets removed the same way as predictions.",
         "~1 min", cmds=lambda st: [[PY, S + "label_export.py", CAMPAIGN]]),
    Step("v2-3class", "DINOv2", "Train DINOv2 with a separate gravel/unpaved class",
         "One model that finds lots AND says paved vs unpaved, so gravel lots stop being a "
         "separate manual question.", "~10-15 min",
         cmds=lambda st: [[PY, S + "label_finetune.py", CAMPAIGN, *TRAIN, "--backbone", V2, "--classes", "3"]],
         done_if=lambda: has_model(RUN_V2_3CLASS),
         after="Paste the 'Test squares' table to Claude if you want a read on it."),
    Step("v2-bakeoff", "DINOv2", "Score it against the current best (bake-off)",
         "Same test squares + 8 PRN downtowns for every model, so the choice is apples to apples.",
         "~10-20 min",
         cmds=bakeoff_cmds([RUN_V2_3CLASS], [RUN_V2_TARGETED]),
         after="Paste data/bakeoff/leaderboard.md to Claude."),
    Step("v2-choose", "DINOv2", "Pick the model for the citywide run",
         "Reads the leaderboard and suggests one.", "seconds",
         action=choose_model([RUN_V2_3CLASS, RUN_V2_TARGETED], "v2")),
    Step("backup-2", "DINOv2", "Back up the new model",
         "So the citywide run can be redone on any machine.", "1-2 min",
         cmds=lambda st: [[PY, S + "label_sync.py", "--models", CAMPAIGN]]),
    Step("v2-citywide", "DINOv2", "Run all of Laredo with the chosen model",
         "The real product: every surface lot in the city as polygons (with paved/unpaved if 3-class). "
         "Resumable: if it stops, run the step again and finished chunks are kept.",
         "1-3 h (NAIP download ~30-60 min, then inference); laptop on power, sleep off",
         cmds=citywide_cmds("v2", RUN_V2_TARGETED),
         after="Output: data/parking-model/laredo/<model>/. The published OSM layer is not touched."),
    Step("v2-citywide-up", "DINOv2", "Share the citywide result",
         "Uploads the citywide polygons/summary so Claude can calibrate the Laredo total against "
         "your 150 random squares.", "1-2 min",
         cmds=lambda st: [[PY, S + "label_sync.py", "--citywide", CITY]],
         after="Tell Claude 'citywide synced' - next is the calibrated Laredo estimate."),
    # ---------------------------------------------------------------- phase 2: DINOv3
    Step("v3-access", "DINOv3", "Check DINOv3 access",
         "The satellite DINOv3 weights are gated: your Hugging Face token must be visible here.", "seconds",
         action=check_dinov3_access),
    Step("v3-bench", "DINOv3", "Memory/speed test",
         "One minute to learn whether batch 2 fits your GPU before a long run.", "~1-2 min",
         cmds=lambda st: [[PY, S + "label_finetune.py", CAMPAIGN, *TRAIN, "--backbone", V3, "--bench"]],
         after="If it runs out of memory, tell Claude (fix: --batch 1 or fewer unfrozen blocks)."),
    Step("v3-train", "DINOv3", "Train DINOv3 (parking / not parking)",
         "Same recipe as the DINOv2 winner, satellite-pretrained backbone. Expect ~2-6 points better "
         "if the satellite pretraining helps.", "~15-25 min",
         cmds=lambda st: [[PY, S + "label_finetune.py", CAMPAIGN, *TRAIN, "--backbone", V3]],
         done_if=lambda: has_model(RUN_V3)),
    Step("v3-3class", "DINOv3", "Train DINOv3 with the unpaved class",
         "Worth it only if the DINOv2 3-class model kept up with its 2-class twin.", "~15-25 min",
         cmds=lambda st: [[PY, S + "label_finetune.py", CAMPAIGN, *TRAIN, "--backbone", V3, "--classes", "3"]],
         done_if=lambda: has_model(RUN_V3_3CLASS), optional=True),
    Step("v3-bakeoff", "DINOv3", "Bake-off: DINOv3 vs DINOv2",
         "Decides whether DINOv3 replaces DINOv2.", "~10-25 min",
         cmds=bakeoff_cmds([RUN_V3, RUN_V3_3CLASS], [RUN_V2_TARGETED, RUN_V2_3CLASS]),
         after="Paste data/bakeoff/leaderboard.md to Claude."),
    Step("backup-3", "DINOv3", "Back up the DINOv3 models", "Same as before.", "1-3 min",
         cmds=lambda st: [[PY, S + "label_sync.py", "--models", CAMPAIGN]]),
    Step("v3-choose", "DINOv3", "Pick the citywide model again",
         "If a DINOv3 run is now on top, the next step reruns Laredo with it; if DINOv2 still wins, "
         "skip the next two steps.", "seconds",
         action=choose_model([RUN_V3, RUN_V3_3CLASS], "v3")),
    Step("v3-citywide", "DINOv3", "Rerun Laredo citywide with the DINOv3 winner",
         "Only if DINOv3 won. Copies the NAIP mosaic from the first run, so it is mostly inference.", "~30-90 min",
         cmds=citywide_cmds("v3", RUN_V3), optional=True),
    Step("v3-citywide-up", "DINOv3", "Share the new citywide result", "", "1-2 min",
         cmds=lambda st: [[PY, S + "label_sync.py", "--citywide", CITY]], optional=True,
         after="Tell Claude 'DINOv3 citywide synced'."),
    # ---------------------------------------------------------------- whenever
    Step("plano", "Labelling (whenever)", "Label ~30 Plano squares in Blind mode",
         "The only real test of how a Laredo-trained model does in another Texas city. Bake-offs add "
         "the Plano column automatically once labels exist.", "~1-2 h of labelling, any time",
         action=info([
             "1. start-local.bat",
             "2. (optional, Segment tool) in a second window: python data/scripts/sam_server.py",
             "3. http://localhost:5173/label.html -> campaign pilot-plano -> mode Blind",
             "4. afterwards: python data/scripts/label_sync.py",
         ]), optional=True),
]


# ----------------------------------------------------------------------------------------------- runner

def status(st: dict, s: Step) -> str:
    if s.id in st["done"]:
        return "done"
    if s.id in st["skipped"]:
        return "skipped"
    return "todo"


def show_list(st: dict) -> None:
    phase = None
    for i, s in enumerate(STEPS, 1):
        if s.phase != phase:
            phase = s.phase
            print(f"\n== {phase}")
        mark = {"done": "[x]", "skipped": "[-]", "todo": "[ ]"}[status(st, s)]
        print(f"  {mark} {i:>2}. {s.title}{' (optional)' if s.optional else ''}   id: {s.id}")
    print()


def run_cmds(cmds: list[list[str]]) -> bool:
    env = child_env()
    for cmd in cmds:
        print(f"\n  $ {' '.join(cmd)}\n", flush=True)
        t0 = time.time()
        try:
            rc = subprocess.call(cmd, cwd=ROOT, env=env)
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            return False
        except FileNotFoundError as e:
            print(f"  Cannot start: {e}")
            return False
        mins = (time.time() - t0) / 60
        if rc != 0:
            print(f"\n  Step failed (exit code {rc}) after {mins:.0f} min.")
            return False
        print(f"\n  Finished in {mins:.0f} min.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="show the steps and their status")
    ap.add_argument("--from", dest="start", help="start at this step id (marks nothing)")
    ap.add_argument("--dry-run", action="store_true", help="print each step's commands; run nothing")
    ap.add_argument("--reset", action="store_true", help="forget progress")
    a = ap.parse_args()

    st = load_state()
    if a.reset:
        st = {"done": {}, "skipped": {}, "choice": {}}
        save_state(st)
        print("Progress cleared.")
    if a.list:
        show_list(st)
        return 0

    ids = [s.id for s in STEPS]
    if a.start and a.start not in ids:
        print(f"No step '{a.start}'. Steps: {', '.join(ids)}")
        return 2
    i0 = ids.index(a.start) if a.start else 0
    todo = [s for s in STEPS[i0:] if a.start or a.dry_run or status(st, s) == "todo"]
    if not todo:
        print("Everything is done or skipped. --list shows the steps, --from <id> reruns one.")
        return 0

    for s in todo:
        n = ids.index(s.id) + 1
        print("\n" + "=" * 78)
        print(f"Step {n}/{len(STEPS)} [{s.phase}]  {s.title}{'  (optional)' if s.optional else ''}")
        print("=" * 78)
        if s.why:
            print(f"  Why: {s.why}")
        print(f"  Time: {s.estimate}")
        if s.done_if and s.done_if():
            print("  Note: its output already exists. Running again retrains/overwrites it; 'd' keeps it.")
        cmds = s.cmds(st) if s.cmds else []
        for c in cmds:
            print(f"  Command: {' '.join(c)}")
        if a.dry_run:
            if s.action:
                s.action(st, True)
            continue
        while True:
            ans = input("\n  [Enter] run   [s] skip   [d] already done   [q] quit : ").strip().lower()
            if ans == "q":
                save_state(st)
                print("Progress saved. Run the same command to continue.")
                return 0
            if ans == "s":
                st["skipped"][s.id] = time.strftime("%Y-%m-%d %H:%M")
                st["done"].pop(s.id, None)
                break
            if ans == "d":
                st["done"][s.id] = time.strftime("%Y-%m-%d %H:%M")
                st["skipped"].pop(s.id, None)
                break
            if ans:
                continue
            ok = s.action(st, False) if s.action else run_cmds(cmds)
            if ok:
                st["done"][s.id] = time.strftime("%Y-%m-%d %H:%M")
                st["skipped"].pop(s.id, None)
                save_state(st)
                if s.after:
                    print(f"  Next: {s.after}")
                break
            save_state(st)
            print("  Not marked done. Enter retries, s skips, q quits (progress is kept).")
        save_state(st)

    print("\nAll steps visited. `--list` shows what was skipped; Claude can take it from here.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Progress saved; run the same command to continue.")
        sys.exit(130)
