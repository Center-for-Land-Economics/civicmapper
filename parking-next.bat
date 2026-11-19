@echo off
rem Guided parking-model steps (DINOv2, then DINOv3). Progress is saved; run again to continue.
cd /d "%~dp0"
if exist ".venv-gpu\Scripts\activate.bat" call ".venv-gpu\Scripts\activate.bat"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
python data\scripts\parking_next.py %*
pause
