@echo off
REM jupyter.bat -- activate venv and launch Jupyter pointed at the notebooks directory
REM Run from the repo root:  data\jupyter.bat

setlocal

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%.venv
set NOTEBOOKS_DIR=%SCRIPT_DIR%jurisidictions

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at %VENV_DIR%
    echo Run setup first:  data\setup_env.bat
    exit /b 1
)

echo =^> Activating virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

echo =^> Launching Jupyter at %NOTEBOOKS_DIR%
echo     Select kernel: Kernel ^> Change Kernel ^> Python (geovizwiz-data)
echo.
"%VENV_DIR%\Scripts\jupyter.exe" notebook "%NOTEBOOKS_DIR%"
