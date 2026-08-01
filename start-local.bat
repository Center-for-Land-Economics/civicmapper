@echo off
setlocal
REM start-local.bat  (Windows equivalent of start-local.sh)
REM Starts the API server (port 8080) and the Vite dev server (port 5173)
REM in two separate windows. Close those windows (or Ctrl+C in each) to stop.

cd /d "%~dp0"

echo ========================================
echo   Starting Local Development Environment
echo ========================================

if not exist "api\node_modules" (
  echo Installing API dependencies...
  pushd api && call npm install && popd
)
if not exist "viz\node_modules" (
  echo Installing viz dependencies...
  pushd viz && call npm install && popd
)

REM Inherited by the child windows below.
set "PORT=8080"

echo Starting API server on port 8080...
start "geoviz API (8080)" cmd /k "cd api && node src/server.js"

echo Starting Vite dev server on port 5173...
start "geoviz Vite (5173)" cmd /k "cd viz && npm run dev"

echo.
echo ========================================
echo   Servers starting in two new windows
echo ========================================
echo   App (Austin): http://localhost:5173/app.html?city=austin
echo   City picker:  http://localhost:5173/cities.html
echo   API:          http://localhost:8080
echo ========================================

REM Give Vite a few seconds to boot, then open the browser.
timeout /t 6 /nobreak >nul
start "" "http://localhost:5173/app.html?city=austin"

endlocal
