#!/usr/bin/env python3
"""
install_tippecanoe.py — cross-platform installer for tippecanoe and pmtiles.

Usage:
    python data/scripts/install_tippecanoe.py

Platforms:
    macOS   — installs via Homebrew (brew install tippecanoe pmtiles)
    Linux   — installs via apt-get; falls back to building tippecanoe from source
    Windows — installs inside WSL2 via apt-get (tippecanoe has no native Windows binary)

Both binaries are required by parquet_to_pmtiles.py:
    tippecanoe  https://github.com/felt/tippecanoe
    pmtiles     https://github.com/protomaps/go-pmtiles
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def check_native(binary: str) -> bool:
    """Return True and print version if binary is already in PATH."""
    if shutil.which(binary):
        result = subprocess.run([binary, "--version"], capture_output=True, text=True)
        version = (result.stdout or result.stderr).strip().splitlines()[0][:80]
        print(f"  {binary} already installed: {version}")
        return True
    return False


def check_wsl(binary: str) -> bool:
    """Return True if binary is available inside WSL."""
    result = subprocess.run(
        ["wsl", "--", "bash", "-c", f"command -v {binary}"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  {binary} already installed in WSL")
        return True
    return False


# ---------------------------------------------------------------------------
# pmtiles binary download (Linux / WSL fallback when apt doesn't have it)
# ---------------------------------------------------------------------------

def _pmtiles_linux_arch() -> str:
    """Map platform.machine() to the arch string used in go-pmtiles release filenames."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "Linux_x86_64"
    if machine in ("aarch64", "arm64"):
        return "Linux_arm64"
    raise RuntimeError(f"Unsupported machine architecture for pmtiles download: {machine}")


def _fetch_latest_pmtiles_url(arch: str) -> tuple[str, str]:
    """Return (download_url, filename) for the latest pmtiles CLI release."""
    api_url = "https://api.github.com/repos/protomaps/go-pmtiles/releases/latest"
    req = urllib.request.Request(api_url, headers={"User-Agent": "install_tippecanoe"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    assets = data.get("assets", [])
    for asset in assets:
        name = asset["name"]
        # Match case-insensitively; Linux releases are .tar.gz
        if arch.lower() in name.lower() and (name.endswith(".tar.gz") or name.endswith(".zip")):
            return asset["browser_download_url"], name
    raise RuntimeError(
        f"Could not find pmtiles release asset for arch={arch} in:\n"
        + "\n".join(a["name"] for a in assets)
    )


def _install_pmtiles_binary(dest_dir: str = "/usr/local/bin", wsl: bool = False) -> None:
    """Download the latest pmtiles binary and install it to dest_dir."""
    import tarfile
    import zipfile

    arch = _pmtiles_linux_arch()
    print(f"  Fetching latest pmtiles release URL ({arch})...")
    url, filename = _fetch_latest_pmtiles_url(arch)
    print(f"  Downloading: {url}")

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / filename
        urllib.request.urlretrieve(url, archive_path)

        # Extract the pmtiles binary from tar.gz or zip
        if filename.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                members = tf.getnames()
                binary_name = next(
                    (m for m in members if m.split("/")[-1] == "pmtiles"), None
                )
                if not binary_name:
                    raise RuntimeError(f"pmtiles binary not found in archive. Contents: {members}")
                member = tf.getmember(binary_name)
                member.name = "pmtiles"  # extract flat to tmp/
                tf.extract(member, tmp)
        else:
            with zipfile.ZipFile(archive_path) as zf:
                names = zf.namelist()
                binary_name = next((n for n in names if n.split("/")[-1] == "pmtiles"), None)
                if not binary_name:
                    raise RuntimeError(f"pmtiles binary not found in zip. Contents: {names}")
                zf.extract(binary_name, tmp)
                Path(tmp, binary_name).rename(Path(tmp, "pmtiles"))

        dest = Path(tmp) / "pmtiles"
        dest.chmod(0o755)

        if wsl:
            wsl_tmp = f"/tmp/pmtiles_install_{dest.name}"
            run(["wsl", "--", "cp", _win_to_wsl(dest), wsl_tmp])
            run(["wsl", "--", "sudo", "mv", wsl_tmp, f"{dest_dir}/pmtiles"])
            run(["wsl", "--", "sudo", "chmod", "+x", f"{dest_dir}/pmtiles"])
        else:
            run(["sudo", "mv", str(dest), f"{dest_dir}/pmtiles"])
            run(["sudo", "chmod", "+x", f"{dest_dir}/pmtiles"])

    print(f"  pmtiles installed to {dest_dir}/pmtiles")


def _win_to_wsl(path: Path) -> str:
    s = str(path).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


# ---------------------------------------------------------------------------
# Platform installers
# ---------------------------------------------------------------------------

def install_macos() -> None:
    if not shutil.which("brew"):
        print(
            "Homebrew not found. Install it first:\n"
            '  /bin/bash -c "$(curl -fsSL '
            'https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"\n'
            "Then re-run this script."
        )
        sys.exit(1)

    print("\n--- Installing tippecanoe ---")
    if not check_native("tippecanoe"):
        run(["brew", "install", "tippecanoe"])

    print("\n--- Installing pmtiles ---")
    if not check_native("pmtiles"):
        run(["brew", "install", "pmtiles"])


def install_linux() -> None:
    print("\n--- Installing tippecanoe ---")
    if not check_native("tippecanoe"):
        if shutil.which("apt-get"):
            run(["sudo", "apt-get", "update", "-qq"])
            result = run(["sudo", "apt-get", "install", "-y", "tippecanoe"], check=False)
            if result.returncode != 0:
                print("  apt install failed; building tippecanoe from source...")
                _build_tippecanoe_from_source()
        else:
            print("  apt-get not available; building tippecanoe from source...")
            _build_tippecanoe_from_source()

    print("\n--- Installing pmtiles ---")
    if not check_native("pmtiles"):
        if shutil.which("apt-get"):
            result = run(["sudo", "apt-get", "install", "-y", "pmtiles"], check=False)
            if result.returncode != 0:
                print("  apt install failed; downloading pmtiles binary from GitHub...")
                _install_pmtiles_binary()
        else:
            print("  apt-get not available; downloading pmtiles binary from GitHub...")
            _install_pmtiles_binary()


def _build_tippecanoe_from_source() -> None:
    """Clone and build tippecanoe from source (Linux fallback)."""
    print("  Installing build dependencies...")
    run(["sudo", "apt-get", "install", "-y", "build-essential", "libsqlite3-dev", "zlib1g-dev"])
    src = Path(tempfile.mkdtemp()) / "tippecanoe"
    run(["git", "clone", "--depth=1", "https://github.com/felt/tippecanoe.git", str(src)])
    run(["make", "-j4"], cwd=str(src))
    run(["sudo", "make", "install"], cwd=str(src))


def install_windows() -> None:
    if not shutil.which("wsl"):
        print(
            "WSL2 not found. Install it first (run in an elevated PowerShell):\n"
            "  wsl --install\n"
            "Restart your computer, then re-run this script."
        )
        sys.exit(1)

    # Confirm a Linux distro is set up
    probe = subprocess.run(["wsl", "-e", "uname"], capture_output=True, text=True)
    if probe.returncode != 0:
        print(
            "WSL is installed but no Linux distribution is set up.\n"
            "Run:  wsl --install -d Ubuntu\n"
            "Then restart and re-run this script."
        )
        sys.exit(1)

    print("\n--- Installing tippecanoe (via WSL) ---")
    if not check_wsl("tippecanoe"):
        run(["wsl", "--", "sudo", "apt-get", "update", "-qq"])
        result = run(["wsl", "--", "sudo", "apt-get", "install", "-y", "tippecanoe"], check=False)
        if result.returncode != 0:
            print("  apt install failed inside WSL; building from source...")
            run(["wsl", "--", "sudo", "apt-get", "install", "-y",
                 "build-essential", "libsqlite3-dev", "zlib1g-dev"])
            run(["wsl", "--", "bash", "-c",
                 "git clone --depth=1 https://github.com/felt/tippecanoe.git /tmp/tippecanoe"
                 " && cd /tmp/tippecanoe && make -j4 && sudo make install"])

    print("\n--- Installing pmtiles (via WSL) ---")
    if not check_wsl("pmtiles"):
        result = run(["wsl", "--", "sudo", "apt-get", "install", "-y", "pmtiles"], check=False)
        if result.returncode != 0:
            print("  apt install failed inside WSL; downloading pmtiles binary from GitHub...")
            _install_pmtiles_binary(wsl=True)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(system: str) -> None:
    print("\n--- Verification ---")
    binaries = ["tippecanoe", "pmtiles"]
    all_ok = True

    if system == "Windows":
        for b in binaries:
            ok = check_wsl(b)
            if not ok:
                print(f"  {b}: NOT FOUND in WSL")
                all_ok = False
    else:
        for b in binaries:
            ok = check_native(b)
            if not ok:
                print(f"  {b}: NOT FOUND")
                all_ok = False

    if all_ok:
        print("\nAll tools installed successfully.")
    else:
        print("\nSome tools were not found after install. Check the output above.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    system = platform.system()
    print(f"Platform: {system} ({platform.machine()})")

    if system == "Darwin":
        install_macos()
    elif system == "Linux":
        install_linux()
    elif system == "Windows":
        install_windows()
    else:
        print(f"Unsupported platform: {system}")
        sys.exit(1)

    verify(system)


if __name__ == "__main__":
    main()
