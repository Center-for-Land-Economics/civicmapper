#!/usr/bin/env bash
# Helper: run OSM-only surface parking extraction + upload for one or more cities.
# Loads the Azure connection string from data/.env (stripping surrounding quotes,
# which the raw value carries and os.getenv would otherwise keep).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

AZURE_STORAGE_CONNECTION_STRING="$(python3 -c "import sys;[sys.stdout.write(l.split('=',1)[1].strip().strip(chr(34)).strip(chr(39))) for l in open('data/.env') if l.startswith('AZURE_STORAGE_CONNECTION_STRING')]")"
export AZURE_STORAGE_CONNECTION_STRING

for city in "$@"; do
  echo "############ PARKING: $city ############"
  python3 data/scripts/parking_lot_extraction.py --city "$city" --osm-only --upload --overwrite
  echo "############ PARKING DONE: $city (exit $?) ############"
done
echo "ALL PARKING RUNS FINISHED"
