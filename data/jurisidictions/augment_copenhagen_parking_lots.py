"""
Copenhagen parking post-step: region tags on the parking lots + the "Parking Lot" category.

Run AFTER run_copenhagen.py and the parking pipeline, then re-upload both parquets:

    PYTHONUTF8=1 python data/jurisidictions/run_copenhagen.py
    OSMNX_OVERPASS_URL=https://maps.mail.ru/osm/tools/overpass/api \
      python data/scripts/parking_lot_extraction.py --city copenhagen --osm-only \
        --file data/jurisidictions/data/copenhagen/copenhagen-hovedstaden-dk-parcels.parquet \
        --output-dir data/parking/copenhagen
    python data/jurisidictions/augment_copenhagen_parking_lots.py
    python data/upload_city_dev.py copenhagen

1. PARKING: copy each lot's parcel `district` (parish) and `jurisdiction` (municipality) onto the
   lot, so the Parking tab's region panel groups the same way as the Value/Underused tabs.

2. PARCELS: tag `property_land_use_refined = "Parking Lot"` with the repo-standard rule
   (data/scripts/apply_parking_category.py): surface parking covers >= 50% of the parcel AND land
   is >= 67% of the parcel's value. The July version of this script (copied from Tallinn) tagged
   EVERY parcel hosting any surface lot and painted the whole parcel as parking. That turned
   Fælledparken (Eksercerpladsen 188: a 68 ha parcel, 0.4% parking), Kastellet/Churchillparken,
   Botanisk Have, Bellahøjmarken and Rigshospitalet into "Parking Lot" blocks in the Underused tab
   (reported by Victor Mayland Nielsen, 2026-10). The per-parcel footprint columns it added
   (parking_footprint_*) are dropped, so the Underused tab uses the standard whole-parcel
   wording; the Parking tab still carries the footprint-level land value under every lot.

Idempotent: the ETL output carries no "Parking Lot" tags, and step 2 re-derives them each run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_PARCELS = REPO_ROOT / "data" / "jurisidictions" / "data" / "copenhagen" / "copenhagen-hovedstaden-dk-parcels.parquet"
DEFAULT_PARKING = REPO_ROOT / "data" / "parking" / "copenhagen" / "copenhagen-hovedstaden-dk-parking-lots.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parcels", type=Path, default=DEFAULT_PARCELS)
    ap.add_argument("--parking", type=Path, default=DEFAULT_PARKING)
    args = ap.parse_args()
    for p in (args.parcels, args.parking):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    g = gpd.read_parquet(args.parcels)
    pk = gpd.read_parquet(args.parking)  # GeoDataFrame, so the rewrite keeps the GeoParquet "geo" metadata

    # 1. region tags on the parking lots
    pid = g["parcel_id"].astype(str)
    for col in ("district", "jurisdiction"):
        pk[col] = pk["parcel_id"].astype(str).map(dict(zip(pid, g[col])))
    pk.to_parquet(args.parking, index=False)
    print(f"Tagged district on {pk['district'].notna().sum():,}/{len(pk):,} parking features; "
          f"by municipality: {pk['jurisdiction'].value_counts().to_dict()}", flush=True)

    # 2. "Parking Lot" category, standard rule
    drop = [c for c in g.columns if c.startswith("parking_footprint_")]
    if drop or (g["property_land_use_refined"] == "Parking Lot").any():
        raise SystemExit("Parcels already carry parking tags; re-run run_copenhagen.py first "
                         "(the original Vacant/Underdeveloped tag under a Parking Lot is not recoverable here).")
    subprocess.run([sys.executable, str(REPO_ROOT / "data" / "scripts" / "apply_parking_category.py"),
                    "--city", "copenhagen", "--parcels", str(args.parcels), "--parking", str(args.parking)],
                   check=True)


if __name__ == "__main__":
    main()
