#!/usr/bin/env python3
"""Augment the Spokane parcel parquet with a neighborhood region-grouping column.

Mirrors augment_olympia_regions.py. Spokane is a browser-GeoParquet city (NO PMTiles/bake):
the Land Value (3D) + Parking region widgets scan the loaded features directly, so tagging the
parquet + registering jurisdictionGroups in cities.ts is all that's needed (no re-bake, no
metadata.groups). Data endpoints are NOCACHE so re-uploading the parquet is live immediately.

Spatially joins parcel centroids against the City of Spokane official neighborhoods layer (29
neighborhoods), tagging one column:

    neighborhood  <- Name   (e.g. "Whitman", "Bemiss", "West Central")

Names are already title-cased in the source. Parcels outside all neighborhoods -> "(None)".
Also writes viz/public/spokane-neighborhood-overlay.geojson.

Operates in place on data/jurisidictions/data/spokane/spokane-wa-parcels.parquet (backup first).

Source (City of Spokane official AGOL, public, no token):
  services.arcgis.com/3PDwyTturHqnGCu0/.../Neighborhood/FeatureServer/6 (Name)
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
CITY = HERE / "data" / "spokane"
PARQUET = CITY / "spokane-wa-parcels.parquet"
BACKUP = CITY / "spokane-wa-parcels.preregions.parquet"
OVERLAY_DIR = ROOT / "viz" / "public"
NONE = "(None)"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
NBHD_URL = ("https://services.arcgis.com/3PDwyTturHqnGCu0/arcgis/rest/services/"
            "Neighborhood/FeatureServer/6/query")
NAME_FIELD = "Name"
SIMPLIFY_TOL = 0.00005
PRECISION = 5


def log(m: str) -> None:
    print(m, flush=True)


def fetch_neighborhoods() -> gpd.GeoDataFrame:
    r = requests.get(NBHD_URL, params={"where": "1=1", "outFields": NAME_FIELD,
                                       "returnGeometry": "true", "outSR": 4326, "f": "geojson"},
                     headers=HEADERS, timeout=120)
    r.raise_for_status()
    g = gpd.read_file(io.BytesIO(r.content))
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    elif g.crs.to_epsg() != 4326:
        g = g.to_crs("EPSG:4326")
    g = g[g.geometry.notnull()].copy()
    g = g.rename(columns={NAME_FIELD: "name"})
    g["name"] = g["name"].map(lambda v: str(v).strip() if v is not None else v)
    inv = ~g.geometry.is_valid
    if inv.any():
        g.loc[inv, "geometry"] = g.loc[inv, "geometry"].buffer(0)
    return g[g.geometry.notnull() & ~g.geometry.is_empty].copy()


def main() -> None:
    log("Fetching City of Spokane neighborhoods ...")
    nbhd = fetch_neighborhoods()
    log(f"  {len(nbhd)} neighborhoods")

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
    layer = nbhd[["name", "geometry"]].copy()
    tagged = gpd.sjoin(pts, layer, how="left", predicate="within")
    tagged = tagged[~tagged.index.duplicated(keep="first")]
    vals = tagged["name"].reindex(range(len(parcel)))
    parcel["neighborhood"] = vals.map(lambda v: str(v) if pd.notna(v) else NONE).values
    vc = parcel["neighborhood"].value_counts(dropna=False)
    matched = int((parcel["neighborhood"] != NONE).sum())
    log(f"  neighborhood: {len(vc)} distinct, {matched:,} matched, "
        f"{int(vc.get(NONE, 0)):,} -> {NONE}; top {dict(list(vc.head(5).items()))}")

    parcel.to_parquet(PARQUET, index=False)
    log(f"Wrote {PARQUET.name} (col 'neighborhood' added)")

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    out = OVERLAY_DIR / "spokane-neighborhood-overlay.geojson"
    d = nbhd[["name", "geometry"]].dissolve(by="name").reset_index()
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
