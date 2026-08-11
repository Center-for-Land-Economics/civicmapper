@echo off
REM setup_env.bat -- create venv, install ETL dependencies, register Jupyter kernel
REM Run from the repo root:  data\setup_env.bat

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%.venv
set KERNEL_NAME=geovizwiz-data
set DISPLAY_NAME=Python (geovizwiz-data)

echo =^> Creating virtual environment at %VENV_DIR%
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    echo Make sure Python 3.11+ is installed and on your PATH.
    exit /b 1
)

echo =^> Upgrading pip
"%VENV_DIR%\Scripts\python.exe" -m pip install --quiet --upgrade pip

echo =^> Installing dependencies from data\requirements.txt
"%VENV_DIR%\Scripts\pip.exe" install -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    exit /b 1
)

echo =^> Registering Jupyter kernel as '%KERNEL_NAME%'
"%VENV_DIR%\Scripts\python.exe" -m ipykernel install --user --name="%KERNEL_NAME%" --display-name="%DISPLAY_NAME%"
if errorlevel 1 (
    echo ERROR: Kernel registration failed.
    exit /b 1
)

echo.
echo Setup complete.
echo.
echo    Kernel registered: %DISPLAY_NAME%
echo    To launch Jupyter: data\jupyter.bat
echo    In the notebook:   Kernel ^> Change Kernel ^> %DISPLAY_NAME%
