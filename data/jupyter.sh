#!/usr/bin/env bash
# jupyter.sh — activate venv and launch Jupyter pointed at the notebooks directory
# Run from the repo root:  bash data/jupyter.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
NOTEBOOKS_DIR="$SCRIPT_DIR/jurisidictions"

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run setup first:  bash data/setup_env.sh"
    exit 1
fi

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Launching Jupyter at $NOTEBOOKS_DIR"
echo "    Select kernel: Kernel → Change Kernel → Python (geovizwiz-data)"
echo ""
jupyter notebook "$NOTEBOOKS_DIR"
