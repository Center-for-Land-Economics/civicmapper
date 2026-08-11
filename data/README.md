# ETL Notebooks — Setup Guide

This directory contains the parcel ETL Jupyter notebooks for each city
(`jurisidictions/`) as well as shared Python utilities (`parcel_calculations.py`,
`cloud_utils.py`) and post-processing scripts (`scripts/`).

## Prerequisites

- **Python 3.11+** — [python.org](https://www.python.org/downloads/)
- **tippecanoe** + **pmtiles** — required only for PMTiles generation (the final
  step in each notebook). Run the installer script to set these up:

```bash
python data/scripts/install_tippecanoe.py
```

The script detects your OS and does the right thing:

| Platform | What it does |
|---|---|
| **macOS** | `brew install tippecanoe pmtiles` (requires [Homebrew](https://brew.sh)) |
| **Linux** | `apt-get install tippecanoe pmtiles`; falls back to building from source |
| **Windows** | Installs inside WSL2 via `apt-get` — tippecanoe has no native Windows binary. Requires [WSL2](https://learn.microsoft.com/windows/wsl/install) with a Ubuntu/Debian distribution. |

If you're on Windows and WSL2 isn't set up yet, install it first:
```powershell
# Run in an elevated PowerShell, then restart
wsl --install
```
Then re-run `python data/scripts/install_tippecanoe.py`.

Once installed, the PMTiles cell in each notebook works automatically — on
Windows it routes calls through `wsl -e tippecanoe` transparently.

## Quick start (automated)

Two helper scripts handle the full setup in one shot. Run from the repo root.

**Linux / macOS:**
```bash
bash data/setup_env.sh        # create venv, install deps, register Jupyter kernel
bash data/jupyter.sh          # activate venv and launch Jupyter
```

**Windows (Command Prompt):**
```bat
data\setup_env.bat
data\jupyter.bat
```

---

## Manual setup

### 1. Create a virtual environment

From the **repo root**:

```bash
# Linux / macOS
python3 -m venv data/.venv

# Windows
python -m venv data\.venv
```

### 2. Activate the virtual environment

```bash
# Linux / macOS
source data/.venv/bin/activate

# Windows (Command Prompt)
data\.venv\Scripts\activate.bat

# Windows (PowerShell)
data\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r data/requirements.txt
```

### 4. Register the venv as a Jupyter kernel

```bash
python -m ipykernel install --user --name=geovizwiz-data --display-name "Python (geovizwiz-data)"
```

### 5. Launch Jupyter

```bash
jupyter notebook data/jurisidictions/
```

In the notebook, select **Kernel → Change Kernel → Python (geovizwiz-data)**.

---

## Environment variables

The notebooks use Azure Blob Storage for uploading parquets and PMTiles.
Set the following before running any upload/promote cells:

| Variable | Description |
|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | Connection string from Azure Portal → Storage Account → Access keys |
| `AZURE_DEV_CONTAINER` | Dev blob container name (default: `parquets-dev`) |
| `AZURE_PROD_CONTAINER` | Prod blob container name (default: `parquets-prod`) |

```bash
# Linux / macOS
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."

# Windows (Command Prompt)
set AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...

# Windows (PowerShell)
$env:AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."
```

The notebooks are safe to run without these set — upload and promote cells are
gated behind `upload_dev = True` / `promote_to_prod = False` flags.

---

## Folder structure

```
data/
├── requirements.txt           # Python dependencies for ETL notebooks
├── README.md                  # This file
├── setup_env.sh               # One-shot env setup (Linux/macOS)
├── setup_env.bat              # One-shot env setup (Windows)
├── jupyter.sh                 # Launch Jupyter (Linux/macOS)
├── jupyter.bat                # Launch Jupyter (Windows)
│
├── parcel_calculations.py     # Shared: improvement/land ratio helpers
├── cloud_utils.py             # Shared: ArcGIS download + Azure upload helpers
├── parquet_registry.py        # Registry of exported parquet metadata
│
├── jurisidictions/            # One notebook per city
│   ├── baltimore.ipynb
│   ├── denver.ipynb
│   ├── houston.ipynb
│   └── ...
│
└── scripts/                   # Post-processing, promotion, and setup scripts
    ├── install_tippecanoe.py  # Cross-platform tippecanoe + pmtiles installer
    ├── parquet_to_pmtiles.py
    ├── promote_parquet.py
    └── ...
```

## Removing the kernel

```bash
jupyter kernelspec remove geovizwiz-data
```
