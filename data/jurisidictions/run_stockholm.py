"""
Stockholm (Sweden) parcel ETL — candidate first Swedish CivicMapper city.

Two-source city (like Copenhagen: open geometry base now, gated value join later).
See memory sweden-stockholm-data-sources.md for the full scouting write-up.

  GEOMETRY  — OPEN (CC0) but ACCOUNT-GATED at the pipe. Lantmäteriet
    "Fastighetsindelning Nedladdning, vektor" became free High-Value-Data on
    2025-02-03 (EU Open Data Directive), licensed CC0. BUT Lantmäteriet exposes
    NO token-free endpoint — you need a free opendata.lantmateriet.se account and
    either (a) download the GeoPackage yourself (filter kommun) or (b) an OAuth2
    client. This script therefore reads a LOCALLY-PROVIDED GeoPackage (same
    manual-download-then-resume pattern as the TCAD/DCAD appraisal exports)
    rather than fetching over the wire.

    How to get the file (free, ~5 min, one time):
      1. Register at https://opendata.lantmateriet.se/ (no cost, no obligation).
      2. Order/download "Fastighetsindelning Nedladdning, vektor" for
         Stockholms kommun (kommunkod 0180) as GeoPackage.
      3. Drop the .gpkg in data/stockholm/cache/ and pass it via --gpkg.

    Layer `registerenhet_yta` (property-area polygons), CRS SWEREF99 TM = EPSG:3006.
    Fields: objekt_id (UUID — Lantmäteriet's id, the value join key), kommunkod,
    kommunnamn, trakt, blockenhet, omrnr, fastighet (full designation), ytkval.

  VALUE     — GATED (agreement + legal-basis review), but IDEAL data. Swedish
    taxeringsvärde splits into delvärden incl. Markvärde / Tomtmarksvärde (LAND
    value) separate from Byggnadsvärde (building) — a real land/building split.
    Two paths, both need an access agreement:
      • Skatteverket "Fastighetstaxering taxeringsuppgifter" Partner API v2 (free,
        bulk async job-uttag). Carries Fastighetsbeteckning + Lantmäteriets id
        (objekt_id) + Typkod + delvärden. Private-actor eligibility is the open
        question (see the scouting memo / access email).
      • Lantmäteriet "Taxering Nedladdning" (fee + fastighetsregisterlagen review).
    Until we have an extract, this builds the GEOMETRY BASE only — NOT app-loadable
    (frontend requires REALLANDVA). Provide an extract keyed by objekt_id or
    fastighetsbeteckning and re-run with --tax.

  LAND-USE  — from Skatteverket Typkod (gated with the value extract). "Unknown"
    until then.

Currency = SEK ("kr"), units = metric (m²) — same frontend support Tallinn added.

Run (geometry base only, once you have the GeoPackage):
  PYTHONUTF8=1 python run_stockholm.py --gpkg data/stockholm/cache/fastighetsindelning_0180.gpkg

Run (full, once a taxering extract exists — parquet/csv with objekt_id or
fastighetsbeteckning + markvarde[/byggnadsvarde/taxeringsvarde]):
  PYTHONUTF8=1 python run_stockholm.py --gpkg <file>.gpkg --tax data/stockholm/cache/tax_extract.parquet
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

# National delivery CRS for Fastighetsindelning. If you ordered a local zone
# (EPSG:3007-3018) the script auto-detects from the GeoPackage CRS instead.
SWEREF99_TM = "EPSG:3006"

SQM_TO_SQFT = 10.76391041671
SQM_TO_ACRES = 1.0 / 4046.8564224

# Fastighetsindelning polygon layer + its fields (confirmed from the product doc).
PARCEL_LAYER_CANDIDATES = ("registerenhet_yta", "registerenhetsomradesyta", "fastighetsyta")


def load_parcels(gpkg_path: str, kommunkod: str) -> gpd.GeoDataFrame:
    """Read the property-area polygons from a locally-provided Fastighetsindelning GeoPackage."""
    if not os.path.exists(gpkg_path):
        raise SystemExit(
            f"\n❌ GeoPackage not found: {gpkg_path}\n"
            "   Lantmäteriet Fastighetsindelning is CC0 but account-gated — there is no\n"
            "   token-free download. Register (free) at https://opendata.lantmateriet.se/,\n"
            "   download 'Fastighetsindelning Nedladdning, vektor' for Stockholms kommun\n"
            "   (kommunkod 0180) as GeoPackage, drop it in data/stockholm/cache/, and pass --gpkg.\n")

    import fiona
    layers = fiona.listlayers(gpkg_path)
    layer = next((l for l in PARCEL_LAYER_CANDIDATES if l in layers), None)
    if layer is None:
        # Fall back to the first polygon layer; print what's there so we can adjust.
        raise SystemExit(f"❌ No known parcel layer in {gpkg_path}. Layers present: {layers}\n"
                         f"   Expected one of {PARCEL_LAYER_CANDIDATES}. Re-run with the right name.")
    print(f"📂 Reading layer '{layer}' from {gpkg_path}")
    gdf = gpd.read_file(gpkg_path, layer=layer)
    print(f"   {len(gdf):,} features, CRS={gdf.crs}, columns={list(gdf.columns)}")

    # Filter to Stockholm municipality if a kommunkod column is present (national files
    # may carry all of Sweden; a kommun-scoped order will already be filtered).
    kk_col = next((c for c in ("kommunkod", "kommun", "KOMMUNKOD") if c in gdf.columns), None)
    if kk_col is not None and kommunkod:
        before = len(gdf)
        gdf = gdf[gdf[kk_col].astype(str).str.zfill(4) == kommunkod].copy()
        print(f"   filtered {kk_col}=={kommunkod}: {before:,} → {len(gdf):,}")
    if len(gdf) == 0:
        raise SystemExit(f"❌ No parcels after kommun filter (kommunkod={kommunkod}).")
    return gdf


def join_valuations(gdf: gpd.GeoDataFrame, tax_path: str) -> gpd.GeoDataFrame:
    """Join taxeringsvärde delvärden (assessed per taxeringsenhet) onto the parcels.

    THE BROADCAST TRAP (playbook §2; present here just like Copenhagen's BFE and
    Oslo's matrikkelenhet): a taxeringsvärde is decided once per *taxeringsenhet*,
    which can span MULTIPLE fastigheter (1..* relation). Broadcasting the value
    onto every parcel then summing = N× inflation. Fix: allocate each
    taxeringsenhet's value across its constituent parcels in proportion to land
    area, so per-sqm is uniform within the unit and the citywide sum is correct.

    Expected extract columns (lower-cased): a join key (objekt_id / lantmaterid /
    fastighetsbeteckning) + at least `markvarde` (land value SEK). Optional:
    `byggnadsvarde`, `taxeringsvarde` (total), `typkod`, `taxeringsenhetsnummer`.
    """
    ext = os.path.splitext(tax_path)[1].lower()
    tax = pd.read_parquet(tax_path) if ext == ".parquet" else pd.read_csv(tax_path)
    tax.columns = [c.lower() for c in tax.columns]

    # Pick the join key present in BOTH sides.
    left_key = next((c for c in ("objekt_id", "objektidentitet", "fastighet") if c in gdf.columns), None)
    key_map = {"objekt_id": ("objekt_id", "lantmaterid", "lantmateriets_id", "objektidentitet"),
               "fastighet": ("fastighetsbeteckning", "fastighet", "beteckning")}
    right_key = None
    for lk, candidates in key_map.items():
        if lk == left_key or (left_key == "objektidentitet" and lk == "objekt_id"):
            right_key = next((c for c in candidates if c in tax.columns), None)
            if right_key:
                left_key = lk
                break
    if right_key is None:
        raise SystemExit(f"❌ No shared join key. gdf has {[c for c in gdf.columns]}, "
                         f"tax has {list(tax.columns)}. Need objekt_id/fastighetsbeteckning on both.")

    value_cols = [c for c in ("markvarde", "byggnadsvarde", "taxeringsvarde", "tomtmarksvarde") if c in tax.columns]
    if "markvarde" not in value_cols and "tomtmarksvarde" not in value_cols:
        raise SystemExit(f"❌ Tax extract has no land-value column (markvarde/tomtmarksvarde); got {list(tax.columns)}")

    # Aggregate delvärden to one value per join key (sum delvärden within a taxeringsenhet
    # that maps to one parcel key; if the extract is already per-key, sum is a no-op).
    tax_by_key = tax.groupby(right_key, as_index=False)[value_cols].sum(min_count=1)
    merged = gdf.merge(tax_by_key, left_on=left_key, right_on=right_key, how="left")

    # Area-allocate across parcels that share a join key (multi-parcel taxeringsenheter).
    grp_area = merged.groupby(left_key)["land_area_sqm"].transform("sum")
    share = (merged["land_area_sqm"] / grp_area).where(grp_area > 0, 0.0)
    dup = merged[left_key].duplicated(keep=False)
    for col in value_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged.loc[dup, col] = merged.loc[dup, col] * share.loc[dup]

    land_col = "markvarde" if "markvarde" in value_cols else "tomtmarksvarde"
    got = merged[land_col].notna().sum()
    print(f"🔗 Joined taxering on {left_key}↔{right_key}: {got:,}/{len(merged):,} parcels got a land value.")
    merged["_land_col"] = land_col
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf.crs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpkg", required=True, help="Path to the Lantmäteriet Fastighetsindelning GeoPackage.")
    ap.add_argument("--kommunkod", default="0180", help="Stockholms kommun = 0180.")
    ap.add_argument("--city-key", default="stockholm")
    ap.add_argument("--state", default="stockholm", help="län slug (Stockholms län).")
    ap.add_argument("--country", default="se")
    ap.add_argument("--tax", default=None,
                    help="Path to a taxering extract (parquet/csv keyed by objekt_id or "
                         "fastighetsbeteckning) with markvarde[/byggnadsvarde]. Omit for geometry base.")
    args = ap.parse_args()

    data_dir = f"data/{args.city_key}"
    os.makedirs(os.path.join(data_dir, "cache"), exist_ok=True)
    slug = f"{args.city_key}-{args.state}-{args.country}"

    # ── 1. Geometry (open, CC0; account-gated download) ──────────────────────────
    gdf = load_parcels(args.gpkg, args.kommunkod)
    src_crs = gdf.crs.to_string() if gdf.crs else SWEREF99_TM
    gdf["geometry"] = gdf["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    # ── 2. Areas (geometry-derived in the projected CRS, then to 4326) ───────────
    geom_area_sqm = gdf.geometry.area  # gdf is still in SWEREF99 (metres) here
    gdf["land_area_sqm"] = geom_area_sqm
    gdf["land_area_sqft"] = geom_area_sqm * SQM_TO_SQFT
    gdf["land_area_acres"] = geom_area_sqm * SQM_TO_ACRES
    gdf = gdf.to_crs("EPSG:4326")

    # ── 3. Values (gated: only present once a taxering extract is joined) ─────────
    if args.tax:
        gdf = join_valuations(gdf, args.tax)
        land_col = gdf["_land_col"].iloc[0]
        land_val = pd.to_numeric(gdf.get(land_col), errors="coerce")
        bld_val = pd.to_numeric(gdf.get("byggnadsvarde"), errors="coerce") if "byggnadsvarde" in gdf else None
        tot_val = pd.to_numeric(gdf.get("taxeringsvarde"), errors="coerce") if "taxeringsvarde" in gdf else None
        gdf["current_full_land_value"] = land_val                 # → REALLANDVA
        if bld_val is not None:
            gdf["improvement_value"] = bld_val
        if tot_val is not None:
            gdf["full_market_value"] = tot_val
        gdf["land_value_per_sqm"] = land_val / gdf["land_area_sqm"]
        gdf["land_value_per_sqft"] = land_val / gdf["land_area_sqft"]
        gdf["property_land_use_refined"] = None  # wire when Typkod-based heuristic lands
    else:
        print("\n⚠️  No --tax extract: building GEOMETRY BASE only (no land value / category).")
        print("    This parquet is NOT app-loadable yet (frontend requires REALLANDVA).")
        print("    Get a taxeringsvärde extract (see sweden-stockholm access email), re-run with --tax.")

    gdf["property_land_use_category"] = "Unknown"  # from Skatteverket Typkod (gated)
    gdf["exemption_flag"] = 0  # non-US product decision: everything visible + filterable.

    # ── 4. Canonical columns ─────────────────────────────────────────────────────
    out = gdf.rename(columns={
        "objekt_id": "parcel_id",       # Lantmäteriet UUID = stable property id + value join key
        "fastighet": "fastighetsbeteckning",
        "trakt": "district",            # trakt → natural region grouping candidate
    })
    keep = [
        "geometry", "parcel_id", "fastighetsbeteckning", "district", "blockenhet",
        "kommunkod", "kommunnamn",
        "property_land_use_category", "property_land_use_refined", "exemption_flag",
        "current_full_land_value", "full_market_value", "improvement_value",
        "land_value_per_sqft", "land_value_per_sqm",
        "land_area_sqm", "land_area_sqft", "land_area_acres",
    ]
    out = out[[c for c in keep if c in out.columns]].copy()
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")

    # ── 5. Save ──────────────────────────────────────────────────────────────────
    suffix = "" if args.tax else "-geometry-base"
    canonical = os.path.join(data_dir, f"{slug}-parcels{suffix}.parquet")
    out.to_parquet(canonical, index=False)
    print(f"\n✅ Saved {len(out):,} parcels → {canonical}  (source CRS {src_crs})")
    print(f"   bounds: {out.total_bounds}")
    if "district" in out.columns:
        print(f"   districts (trakter): {out['district'].nunique()}")
    if args.tax:
        print("\nLand value (SEK) describe:")
        print(out["current_full_land_value"].describe().to_string())


if __name__ == "__main__":
    main()
