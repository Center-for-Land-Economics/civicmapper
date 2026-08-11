#!/usr/bin/env python3
"""Augment the Olympia parcel parquet with a neighborhood region-grouping column.

Mirrors augment_seattle_regions.py, but Olympia is a browser-GeoParquet city (NO PMTiles/bake):
the Land Value (3D) + Parking region widgets scan the loaded features directly, so tagging the
parquet + registering jurisdictionGroups in cities.ts is all that's needed (no metadata.groups).

Spatially joins parcel centroids against the Olympia planning neighborhoods layer (48
neighborhoods), tagging one column:

    neighborhood  <- ASSOC (title-cased)   (e.g. "Bigelow Highlands", "Eastside", "South Capitol")

Source choice: the City's current "Recognized Neighborhood Associations" (25) only covers ~48%
of parcels (downtown/Capitol/commercial areas have no association), so we use the fuller planning
neighborhoods layer (48 areas, ~84% parcel coverage). Names are UPPERCASE in the source -> title-
cased for display. Parcels outside all neighborhoods -> "(None)". Also writes the frontend overlay
viz/public/olympia-neighborhood-overlay.geojson.

Operates in place on data/jurisidictions/data/olympia/olympia-wa-parcels.parquet (backup first).
Data endpoints are NOCACHE so re-uploading the parquet is live immediately (no version bump for
the parcel layer; parking uses parkingVersion).

Source (Olympia planning neighborhoods, public AGOL, no token):
  services3.arcgis.com/0IbpLwS460cn4psv/.../Neighborhoods_Oly/FeatureServer/0 (ASSOC)
"""
from __future__ import annotations

import io
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLY = HERE / "data" / "olympia"
PARQUET = OLY / "olympia-wa-parcels.parquet"
BACKUP = OLY / "olympia-wa-parcels.preregions.parquet"
OVERLAY_DIR = ROOT / "viz" / "public"
NONE = "(None)"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
NBHD_URL = ("https://services3.arcgis.com/0IbpLwS460cn4psv/arcgis/rest/services/"
            "Neighborhoods_Oly/FeatureServer/0/query")
SIMPLIFY_TOL = 0.00005
PRECISION = 5


def log(m: str) -> None:
    print(m, flush=True)


def fetch_neighborhoods() -> gpd.GeoDataFrame:
    r = requests.get(NBHD_URL, params={"where": "1=1", "outFields": "ASSOC",
                                       "returnGeometry": "true", "outSR": 4326, "f": "geojson"},
                     headers=HEADERS, timeout=120)
    r.raise_for_status()
    g = gpd.read_file(io.BytesIO(r.content))
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    elif g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    g = g[g.geometry.notnull()].copy()
    # Source names are UPPERCASE -> title-case for display (e.g. "SOUTH CAPITOL" -> "South Capitol").
    g["ASSOC"] = g["ASSOC"].map(lambda v: str(v).title() if v is not None else v)
    inv = ~g.geometry.is_valid
    if inv.any():
        g.loc[inv, "geometry"] = g.loc[inv, "geometry"].buffer(0)
    return g[g.geometry.notnull() & ~g.geometry.is_empty].copy()


def main() -> None:
    log("Fetching Olympia Recognized Neighborhood Associations ...")
    nbhd = fetch_neighborhoods()
    log(f"  {len(nbhd)} active associations")

    if not PARQUET.exists():
        raise SystemExit(f"Missing parquet: {PARQUET}")
    if not BACKUP.exists():
        shutil.copy2(PARQUET, BACKUP)
        log(f"Backed up -> {BACKUP.name}")
    else:
        log(f"Backup already exists -> {BACKUP.name}")

    parcel = gpd.read_parquet(PARQUET)
    log(f"Read {PARQUET.name}: {len(parcel):,} parcels, crs={parcel.crs}")

    cent = parcel.geometry.to_crs(3857).centroid.to_crs(4326)
    pts = gpd.GeoDataFrame(pd.DataFrame({"_row": range(len(parcel))}),
                           geometry=cent.values, crs="EPSG:4326")
    layer = nbhd[["ASSOC", "geometry"]].copy()
    tagged = gpd.sjoin(pts, layer, how="left", predicate="within")
    tagged = tagged[~tagged.index.duplicated(keep="first")]
    vals = tagged["ASSOC"].reindex(range(len(parcel)))
    parcel["neighborhood"] = vals.map(lambda v: str(v) if pd.notna(v) else NONE).values
    vc = parcel["neighborhood"].value_counts(dropna=False)
    matched = int((parcel["neighborhood"] != NONE).sum())
    log(f"  neighborhood: {len(vc)} distinct, {matched:,} matched, "
        f"{int(vc.get(NONE, 0)):,} -> {NONE}; top {dict(list(vc.head(5).items()))}")

    parcel.to_parquet(PARQUET, index=False)
    log(f"Wrote {PARQUET.name} (col 'neighborhood' added)")

    # overlay
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    out = OVERLAY_DIR / "olympia-neighborhood-overlay.geojson"
    d = nbhd[["ASSOC", "geometry"]].rename(columns={"ASSOC": "name"}).dissolve(by="name").reset_index()
    d["geometry"] = d.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    if out.exists():
        out.unlink()
    try:
        d.to_file(out, driver="GeoJSON", COORDINATE_PRECISION=PRECISION)
    except TypeError:
        d.to_file(out, driver="GeoJSON")
    log(f"WROTE {out.name}: {len(d)} regions, {out.stat().st_size/1024:,.0f} KB")


if __name__ == "__main__":
    main()
