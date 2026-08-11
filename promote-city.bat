@echo off
setlocal
REM Promote a city's parquet + pmtiles + parking from parquets-dev -> parquets-prod
REM via the "Promote City To Prod" GitHub Action.
REM
REM Runs the workflow from the `develop` branch: the workflow resolves the city against
REM that branch's data/parquet_registry.py, and newly-added cities exist on develop before
REM they're merged to main (running from main fails with "Unknown city ...").
REM
REM Requires: gh installed + authenticated  (verify with: gh auth status)

set /p CITY=Enter city key to promote (e.g. austin):
if "%CITY%"=="" (
  echo No city entered. Aborting.
  exit /b 1
)

echo.
echo About to promote "%CITY%" to PROD ^(overwrite + pmtiles + parking^) from ref=develop.
set /p OK=Type Y to continue:
if /I not "%OK%"=="Y" (
  echo Cancelled.
  exit /b 1
)

echo.
REM No --repo flag: gh resolves the repo from this checkout's origin remote,
REM so the script keeps working wherever the repo lives.
gh workflow run promote-city-prod.yml ^
  -f "city=%CITY%" ^
  -f overwrite=true ^
  -f include_pmtiles=true ^
  -f include_parking=true ^
  --ref develop

if errorlevel 1 (
  echo.
  echo Dispatch failed - is gh installed and authenticated?  Run: gh auth status
  exit /b 1
)

echo.
echo Dispatched. Follow it with:   gh run watch
echo Or list recent runs:          gh run list --workflow=promote-city-prod.yml
endlocal
