# Parking labels: running things locally

Everything here runs from the repository root on the `claude/amazing-archimedes-ihog6m` branch.
Campaign files and labels live in `data/labels/` (not in git) and sync with the Azure container
`parking-labels-dev` through `data/scripts/label_sync.py`.

## Guided runner (start here)

Double-click `parking-next.bat` (or `python data/scripts/parking_next.py` in `.venv-gpu`). It walks
through everything left, one step at a time: DINOv2 3-class model, bake-off, model choice, Laredo
citywide run and backups, then the DINOv3 steps. Each step shows the command and a time estimate;
Enter runs it, `s` skips, `d` marks it done, `q` quits. Progress lives in
`data/labels/.parking_next.json`; `--list` shows it, `--from <id>` jumps to a step, `--dry-run`
prints commands only. Put `HF_TOKEN=...` for DINOv3 in `data\.env` (never in chat or git).

`label_sync.py --models` backs up trained runs (`finetune*/`) to Azure and fetches missing ones;
`label_sync.py --citywide laredo` shares citywide outputs (`data/parking-model/laredo/`, parquet/json).
The citywide pipeline no longer needs the GDAL command-line tools (it mosaics with rasterio).

## One-time setup

1. `data\.env` with one line (the `parking-labels-sync` SAS token; never paste it anywhere else):
   `AZURE_PARKING_LABELS_CONNECTION_STRING=st=...`
2. Labelling only needs Node.js and `pip install azure-storage-blob`.
3. GPU work (NVIDIA; tested on an RTX 3060 12 GB) in its own environment:

```
python -m venv .venv-gpu
.venv-gpu\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==4.57.6" huggingface_hub rasterio geopandas pyarrow scipy pillow requests azure-storage-blob
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

## Labelling session

```
git pull
python data/scripts/label_sync.py --pull
start-local.bat
```
Open http://localhost:5173/label.html, type your name (same spelling every time), pick the
campaign and **Model fix**. Afterwards:
```
python data/scripts/label_sync.py
```

## Retrain cycle (after every ~25 new labels; ~10 min on the GPU)

| Step | Command | Time on a 3060 |
|---|---|---|
| Rebuild masks from labels | `python data/scripts/label_export.py pilot-laredo` | ~1 min |
| Score the current model on test squares | `python data/scripts/label_finetune.py pilot-laredo --eval-only` | ~1 min |
| Retrain | `python data/scripts/label_finetune.py pilot-laredo --amp --batch 4 --epochs 8` | ~5 min |
| Refresh the model guesses in the label tool | `python data/scripts/label_predict.py pilot-laredo --weights data/labels/pilot-laredo/finetune/model.pt` | ~3 min |
| Share guesses and labels | `python data/scripts/label_sync.py` | <1 min |

`label_finetune.py` overwrites `finetune/model.pt` and `metrics.json`; copy them aside first if you
want to keep the previous model. Paste the "Test squares" table to Claude to track the learning curve.

## Other scripts

| Purpose | Command |
|---|---|
| Score any model on the labelled squares, with pictures | `python data/scripts/label_eval.py pilot-laredo [--weights data/labels/pilot-laredo/finetune/model.pt --test-only]` |
| Score models on Parking Reform Network downtowns (8 Texas cities) | `python data/scripts/prn_eval.py --weights data/labels/pilot-laredo/finetune/model.pt` |
| Add squares where the model probably errs | `python data/scripts/label_targets.py pilot-laredo --weights data/labels/pilot-laredo/finetune/model.pt --n 25` |
| Add Overture buildings and roads to a campaign | `python data/scripts/label_buildings.py pilot-laredo` |
| New campaign for another city | `python data/scripts/label_campaign.py --city plano --name pilot-plano --n 150` |

All model output is cleaned the same way as the labels: Overture buildings and public streets
(off-parcel road half-widths) are removed (`data/scripts/parking_postprocess.py`). Evaluations
report both the raw and the "clean" numbers; the label tool's guesses are cleaned by default.

Outputs: masks and summary in `data/labels/<campaign>/export/`, scoring pictures in
`data/labels/<campaign>/eval*/`, PRN pictures and `results.csv` in `data/prn/`.

## Bake-off (compare model families)

Needs `transformers==4.57.6` (`pip install "transformers==4.57.6"` in `.venv-gpu`; the SegFormer
results are identical) and, for DINOv3, access to `facebook/dinov3-vitl16-pretrain-sat493m`
(request it on the Hugging Face model page, then `set HF_TOKEN=...` in the shell; never commit it).

| Step | Command | Time on a 3060 (est.) |
|---|---|---|
| SegFormer (pipeline model), fine-tuned | `python data/scripts/label_finetune.py pilot-laredo --amp --batch 4 --epochs 8` | ~6 min |
| DINOv3 satellite ViT-L + head | `python data/scripts/label_finetune.py pilot-laredo --arch vit --amp --batch 2 --epochs 8` | ~30-45 min |
| DINOv2-large + head (open stand-in) | `python data/scripts/label_finetune.py pilot-laredo --arch vit --backbone facebook/dinov2-large --amp --batch 2 --epochs 8` | ~30-45 min |
| Check memory/speed first | add `--bench` to any of the above | ~1 min |
| Score everything | `python data/scripts/bakeoff.py --models segformer segformer:data/labels/pilot-laredo/finetune/model.pt vit:data/labels/pilot-laredo/finetune-dinov3-vitl16-pretrain-sat493m "sam2:segformer:data/labels/pilot-laredo/finetune/model.pt"` | ~20-40 min |

Add `--transfer pilot-plano` once some Plano squares are labelled in Blind mode. Results:
`data/bakeoff/leaderboard.md` (paste to Claude) and `compare_<square>.png` (one row per model).
Each run holds back 15% of its training squares to choose its threshold.

## Segment tool (SAM 2.1) in the label page

Run the helper next to `start-local.bat`, in the GPU environment (first run downloads ~320 MB):
```
python data/scripts/sam_server.py
```
In the label page press **A** (Segment), click inside a lot: SAM outlines it. Click more places to
grow it, right- or Alt-click to exclude, **Enter** keeps it as the current class (an ordinary
editable shape), **Esc** clears. Without the helper the page says so and the other tools still work.

## Three-class model (paved / unpaved) and the citywide pipeline

| Step | Command |
|---|---|
| Train DINOv2 with separate gravel class (~8 min) | `python data/scripts/label_finetune.py pilot-laredo --arch vit --backbone facebook/dinov2-large --classes 3 --amp --batch 2 --epochs 8` |
| Score it (adds a gravel-surface column) | `python data/scripts/bakeoff.py --models vit:data/labels/pilot-laredo/finetune-dinov2-large-3class` |
| Citywide run with any model | `python data/scripts/parking_lot_extraction.py --city laredo --model vit:data/labels/pilot-laredo/finetune-dinov2-large-targeted` |

With a three-class model the pipeline also writes `parking_unpaved.tif` and tags every lot polygon
with `unpaved_share` and `surface_material` (paved / unpaved). Without `--model` it runs the original
SegFormer (research/evaluation licence, see docs/TODO.md).

## Memory and speed notes

- Fine-tuning at `--batch 4 --amp` peaks at ~9 GB; use `--batch 2` on 8 GB cards.
- The first run of each script downloads NAIP for its squares (cached afterwards).
- More epochs or unfreezing (`--freeze-stages 0`) did not beat the 8-epoch default at 51 squares;
  more labels did.
