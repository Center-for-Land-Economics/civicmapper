#!/bin/bash

# start-local.sh
# Starts both the API server (port 8080) and the Vite dev server (port 5173) for local development

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Store PIDs for cleanup
API_PID=""
VITE_PID=""

# Cleanup function
cleanup() {
  echo -e "\n${YELLOW}Shutting down servers...${NC}"
  
  if [ ! -z "$API_PID" ]; then
    echo -e "${BLUE}Stopping API server (PID: $API_PID)${NC}"
    kill $API_PID 2>/dev/null || true
  fi
  
  if [ ! -z "$VITE_PID" ]; then
    echo -e "${BLUE}Stopping Vite dev server (PID: $VITE_PID)${NC}"
    kill $VITE_PID 2>/dev/null || true
  fi
  
  echo -e "${GREEN}✓ Cleanup complete${NC}"
  exit 0
}

# Set up trap for cleanup on exit
trap cleanup EXIT INT TERM

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Starting Local Development Environment${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check if node_modules exist and install if needed
if ! (cd api && node -e "const f=require('fs');process.exit(f.statSync('package-lock.json').mtimeMs>f.statSync('node_modules/.package-lock.json').mtimeMs?1:0)" 2>/dev/null); then
  echo "Installing API dependencies (node_modules missing or package-lock.json changed)..."
  (cd api && npm install)
fi

if ! (cd viz && node -e "const f=require('fs');process.exit(f.statSync('package-lock.json').mtimeMs>f.statSync('node_modules/.package-lock.json').mtimeMs?1:0)" 2>/dev/null); then
  echo "Installing viz dependencies (node_modules missing or package-lock.json changed)..."
  (cd viz && npm install)
fi

echo -e "${GREEN}✓ Dependencies ready${NC}\n"

# Start API server on port 8080
echo -e "${BLUE}Starting API server on port 8080...${NC}"
cd api
PORT=8080 node src/server.js &
API_PID=$!
cd ..
echo -e "${GREEN}✓ API server started (PID: $API_PID)${NC}"
sleep 2

# Start Vite dev server on port 5173
echo -e "${BLUE}Starting Vite dev server on port 5173...${NC}"
cd viz
npm run dev &
VITE_PID=$!
cd ..
echo -e "${GREEN}✓ Vite dev server started (PID: $VITE_PID)${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  🚀 Servers are running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${BLUE}  3D Maps:${NC} http://localhost:5173"
echo -e "${BLUE}  API:${NC}     http://localhost:8080"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}\n"

# Wait for both processes
wait

