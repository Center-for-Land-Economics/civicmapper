"""
Tallinn (Estonia) parcel ETL — first non-US CivicMapper city.

Source: Maa-amet (Estonian Land Board) national cadastre GPKG.
  https://s3.pilw.io/rp-kemit-kataster/ANDMED/20260101_Eesti_KATASTER_GPKG.zip
  Single layer `Eesti`, ~775k parcels, EPSG:3301 (L-EST97). Filter by `ov_nimi`
  (municipality name); Tallinn == 'Tallinn' → ~38.6k parcels.

Key differences from the US cities (see docs/estonia-cadastre-glossary.md):
  * `maks_hind` = maksustamishind = assessed *land* value in EUR (post-2022 mass
    revaluation, the maamaks / land-tax basis). This maps to REALLANDVA.
  * There is NO building/improvement value — Estonia does not tax buildings — so
    REALIMPROV and the improvement/land-ratio "Underdeveloped" heuristic have no
    source. This ships as a land-value MVP.
  * One cadastral unit per land parcel; apartments are korteriomandid in the land
    register, not separate polygons → no US-style per-unit condo stacking.
  * `sihtotstarve` (siht1/2/3) = intended land use → property_land_use_category.
  * `omvorm` = ownership form → a filterable public/private field. Per product
    decision, nothing is dropped/hidden by default (everything visible, filterable).

Parameterised by municipality so Tartu / Pärnu / etc. are one-liners later:
  python run_tallinn.py --municipality "Tartu linn" --city-key tartu --state tartu

Run (using an already-downloaded GPKG):
  PYTHONUTF8=1 python run_tallinn.py --gpkg /path/Eesti_KATASTER_GPKG.gpkg
Otherwise the national GPKG is downloaded + cached under data/<city-key>/cache/.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

GPKG_URL = "https://s3.pilw.io/rp-kemit-kataster/ANDMED/20260101_Eesti_KATASTER_GPKG.zip"
GPKG_LAYER = "Eesti"
SOURCE_CRS = "EPSG:3301"

# sihtotstarve (siht1) → English land-use category used for the map's category filter.
SIHT_CATEGORY = {
    "ELAMUMAA": "Residential",
    "ARIMAA": "Commercial",
    "TOOTMISMAA": "Industrial",
    "TRANSPORDIMAA": "Transport / ROW",
    "ULDKASUTATAV_MAA": "Public / Open Space",
    "UHISKONDLIKE_EHITISTE_MAA": "Civic / Institutional",
    "MAATULUNDUSMAA": "Agricultural",
    "VEEKOGUDE_MAA": "Water",
    "SIHTOTSTARBETA_MAA": "Undesignated",
    "RIIGIKAITSEMAA": "Defense",
    "KAITSEALUNE_MAA": "Protected",
    "MAETOOSTUSMAA": "Mineral / Extraction",
    "JAATMEHOIDLA_MAA": "Waste / Utility",
    "SOTSIAALMAA": "Social",
}

# omvorm (Estonian) → English ownership label.
OWNERSHIP_LABEL = {
    "Eraomand": "Private",
    "Munitsipaalomand": "Municipal",
    "Riigiomand": "State",
    "Avalik-õiguslik omand": "Public-law",
    "Segaomand": "Mixed",
    "Kinnistamata eraomand": "Private (unregistered)",
    "Omandi ulatus selgitamisel": "Under clarification",
}
PUBLIC_OWNERSHIP = {"Munitsipaalomand", "Riigiomand", "Avalik-õiguslik omand"}

SQM_TO_SQFT = 10.76391041671
SQM_TO_ACRES = 1.0 / 4046.8564224


def resolve_gpkg(gpkg_arg: str | None, cache_dir: str) -> str:
    """Return a path to the cadastre GPKG, downloading + extracting if needed."""
    if gpkg_arg and os.path.exists(gpkg_arg):
        print(f"📂 Using provided GPKG: {gpkg_arg}")
        return gpkg_arg
    os.makedirs(cache_dir, exist_ok=True)
    # Reuse any previously extracted .gpkg in the cache.
    for f in os.listdir(cache_dir):
        if f.lower().endswith(".gpkg"):
            path = os.path.join(cache_dir, f)
            print(f"📂 Using cached GPKG: {path}")
            return path
    import urllib.request

    zip_path = os.path.join(cache_dir, "Eesti_KATASTER_GPKG.zip")
    print(f"⬇️  Downloading national cadastre → {zip_path}\n    {GPKG_URL}")
    urllib.request.urlretrieve(GPKG_URL, zip_path)
    print("📦 Extracting…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    for f in os.listdir(cache_dir):
        if f.lower().endswith(".gpkg"):
            return os.path.join(cache_dir, f)
    raise RuntimeError("No .gpkg found after extracting the cadastre zip.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--municipality", default="Tallinn",
                    help="ov_nimi value to filter on (e.g. 'Tallinn', 'Tartu linn').")
    ap.add_argument("--city-key", default="tallinn")
    ap.add_argument("--state", default="harju", help="province/maakond slug")
    ap.add_argument("--country", default="ee")
    ap.add_argument("--gpkg", default=None, help="Path to an already-extracted cadastre GPKG.")
    args = ap.parse_args()

    data_dir = f"data/{args.city_key}"
    os.makedirs(data_dir, exist_ok=True)
    slug = f"{args.city_key}-{args.state}-{args.country}"

    gpkg = resolve_gpkg(args.gpkg, os.path.join(data_dir, "cache"))

    # ── 1. Read the municipality (attribute filter pushed down to OGR) ──────────
    print(f"\n📖 Reading {GPKG_LAYER} where ov_nimi = '{args.municipality}' …")
    where = f"ov_nimi = '{args.municipality.replace(chr(39), chr(39) * 2)}'"
    gdf = gpd.read_file(gpkg, layer=GPKG_LAYER, where=where)
    if len(gdf) == 0:
        raise RuntimeError(f"No parcels for ov_nimi = '{args.municipality}'.")
    if gdf.crs is None:
        gdf = gdf.set_crs(SOURCE_CRS)
    print(f"✅ {len(gdf):,} parcels  CRS={gdf.crs}")

    # ── 2. Reproject 3301 → 4326 + fix geometry validity ────────────────────────
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    # ── 3. Classify land use + ownership ─────────────────────────────────────────
    gdf["siht1"] = gdf["siht1"].astype("string")
    gdf["property_land_use_category"] = (
        gdf["siht1"].map(SIHT_CATEGORY).fillna("Other")
    )
    # Mixed-use flag: a secondary purpose present with the primary under 100%.
    gdf["mixed_use"] = (gdf["siht2"].notna() & (pd.to_numeric(gdf["so_prts1"], errors="coerce") < 100))

    omvorm = gdf["omvorm"].astype("string").str.strip().replace("", pd.NA)
    gdf["ownership_form"] = omvorm.map(OWNERSHIP_LABEL).fillna(omvorm).fillna("Unknown")
    gdf["public_owned"] = gdf["omvorm"].isin(PUBLIC_OWNERSHIP).astype(int)

    # Per product decision: everything visible + filterable, nothing auto-hidden.
    gdf["exemption_flag"] = 0

    print("\nproperty_land_use_category counts:")
    print(gdf["property_land_use_category"].value_counts(dropna=False).to_string())
    print("\nownership_form counts:")
    print(gdf["ownership_form"].value_counts(dropna=False).to_string())

    # ── 4. Values + areas (land value only; no building value in Estonia) ────────
    land = pd.to_numeric(gdf["maks_hind"], errors="coerce")
    area_sqm = pd.to_numeric(gdf["pindala"], errors="coerce")
    area_sqm = area_sqm.where(area_sqm > 0, np.nan)

    gdf["current_full_land_value"] = land            # → REALLANDVA (via loader alias)
    gdf["land_area_sqm"] = area_sqm
    gdf["land_area_sqft"] = area_sqm * SQM_TO_SQFT
    gdf["land_area_acres"] = area_sqm * SQM_TO_ACRES
    gdf["land_value_per_sqm"] = land / area_sqm
    gdf["land_value_per_sqft"] = land / (area_sqm * SQM_TO_SQFT)  # → REALLANDVA_per_sqft

    # ── 5. Refined underutilization (land-value MVP: undesignated land as Vacant) ─
    # No building footprints in this MVP, so the impr/land-ratio "Underdeveloped"
    # heuristic is not computed. `SIHTOTSTARBETA_MAA` (land with no designated
    # purpose) is the one buildings-free vacancy signal available from the cadastre.
    gdf["property_land_use_refined"] = np.where(
        gdf["siht1"] == "SIHTOTSTARBETA_MAA", "Vacant", None
    )

    # ── 6. Link to the official Maa-amet cadastral card ──────────────────────────
    gdf["link"] = "https://xgis.maaamet.ee/ky/" + gdf["tunnus"].astype(str)

    # ── 7. Select + rename canonical columns ─────────────────────────────────────
    out = gdf.rename(columns={
        "tunnus": "parcel_id",
        "l_aadress": "address",
        "ay_nimi": "district",
    })
    keep = [
        "geometry", "parcel_id", "address", "district",
        "property_land_use_category", "property_land_use_refined", "mixed_use",
        "siht1", "siht2", "siht3", "so_prts1",
        "ownership_form", "public_owned", "exemption_flag",
        "current_full_land_value", "land_value_per_sqft", "land_value_per_sqm",
        "land_area_sqm", "land_area_sqft", "land_area_acres",
        "link",
    ]
    out = out[[c for c in keep if c in out.columns]].copy()
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")

    # ── 8. Save ──────────────────────────────────────────────────────────────────
    today = datetime.now().strftime("%Y_%m_%d")
    canonical = os.path.join(data_dir, f"{slug}-parcels.parquet")
    dated = os.path.join(data_dir, f"{slug}-parcels_{today}.parquet")
    out.to_parquet(canonical, index=False)
    out.to_parquet(dated, index=False)

    print(f"\n✅ Saved {len(out):,} parcels → {canonical}")
    print("\nLand value (EUR) describe:")
    print(out["current_full_land_value"].describe().to_string())
    print("\n€/sqft describe:")
    print(out["land_value_per_sqft"].describe().to_string())
    print("\nrefined counts:")
    print(out["property_land_use_refined"].value_counts(dropna=False).to_string())
    print(f"\nbounds: {out.total_bounds}")


if __name__ == "__main__":
    main()
