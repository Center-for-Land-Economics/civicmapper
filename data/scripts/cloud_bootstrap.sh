#!/usr/bin/env bash
# Bootstrap a fresh Linux GPU box (RunPod/Vast/Lambda Ubuntu+CUDA image) for
# the satellite parking pipeline (data/scripts/parking_lot_extraction.py).
#
# Usage (from the directory containing the geovizwiz bundle/repo root):
#   bash data/scripts/cloud_bootstrap.sh
#
# Then run a city, e.g.:
#   tmux new -s parking
#   python -u data/scripts/parking_lot_extraction.py \
#       --city houston --use-osm-supplement \
#       --output-dir data/parking/houston-ml-validation
#
# Notes:
# - Inference resumes chunk-by-chunk (inference_chunks_done.txt journal), so
#   spot/interruptible instances are safe.
# - The heavy intermediates (NAIP tiles, mosaic, probs) land under the
#   --output-dir; only the final <city>-parking-lots.parquet + metadata JSON
#   need to come back home.
# - transformers MUST stay <5 (v5 renames SegFormer state-dict keys and the
#   UTEL-UIUC checkpoint fails to load).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends gdal-bin tmux

python -m pip install --upgrade pip

# RunPod/Lambda PyTorch images ship torch+CUDA already; only install if absent.
if ! python -c "import torch" 2>/dev/null; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

pip install \
    "transformers==4.46.3" \
    huggingface_hub \
    rasterio \
    planetary_computer \
    pystac_client \
    pillow \
    geopandas \
    pandas \
    numpy \
    shapely \
    pyproj \
    osmnx \
    requests \
    scipy \
    pyarrow \
    "duckdb>=1.0.0"

python - <<'EOF'
import torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
EOF
gdalwarp --version
echo "bootstrap complete"
