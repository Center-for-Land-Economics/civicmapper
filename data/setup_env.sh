#!/usr/bin/env bash
# setup_env.sh — create venv, install ETL dependencies, register Jupyter kernel
# Run from the repo root:  bash data/setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
KERNEL_NAME="geovizwiz-data"
DISPLAY_NAME="Python (geovizwiz-data)"

echo "==> Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --quiet --upgrade pip

echo "==> Installing dependencies from data/requirements.txt"
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "==> Registering Jupyter kernel as '$KERNEL_NAME'"
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="$DISPLAY_NAME"

echo ""
echo "✅ Setup complete."
echo ""
echo "   Kernel registered: $DISPLAY_NAME"
echo "   To launch Jupyter: bash data/jupyter.sh"
echo "   In the notebook:   Kernel → Change Kernel → $DISPLAY_NAME"
