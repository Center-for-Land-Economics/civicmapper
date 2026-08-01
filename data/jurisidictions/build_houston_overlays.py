#!/usr/bin/env python3
"""Build frontend-ready region-boundary overlays for Houston (one GeoJSON per grouping scheme).

The viz "Overlays" control (2D mode) draws the actual region polygons on top of the map. The
source layers live in `data/jurisidictions/data/houston/` — the municipal boundaries plus the
three uploaded shapefiles. This cleans each for the browser:
  - reproject to EPSG:4326 (the Civic Clubs file is mislabeled — coords are Web Mercator)
  - repair self-intersecting polygons (buffer(0); e.g. council districts A/B/E)
  - keep just a `name` property (matches the parcel/hex tag value where relevant)
  - simplify + trim coordinate precision to keep the files small enough to ship in public/

Outputs to viz/public/ as houston-<field>-overlay.geojson (served as a static asset, resolved
via import.meta.env.BASE_URL in the frontend). Mirrors augment_houston_regions.py for the tag
value transforms so overlay names line up with the categorical columns.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
HOUSTON = ROOT / "data" / "jurisidictions" / "data" / "houston"
SHAPES = HOUSTON / "Houston Shape Files"
OUT_DIR = ROOT / "viz" / "public"
SIMPLIFY_TOL = 0.00005     # ~5.5 m in degrees — plenty for outline overlays
PRECISION = 5              # ~1 m coordinate precision

# (source path, source name column, output field key, value transform)
LAYERS = [
    (HOUSTON / "houston-harris-cities.geojson", "jurisdiction", "jurisdiction", lambda v: str(v)),
    (SHAPES / "City Council Districts.geojson", "DISTRICT", "council_district", lambda v: f"District {v}"),
    (SHAPES / "Super Neighborhoods.geojson", "SNBNAME", "super_neighborhood", lambda v: str(v)),
    (SHAPES / "Civic Clubs.geojson", "CivicName", "civic_club", lambda v: str(v)),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src, name_col, field, transform in LAYERS:
        if not src.exists():
            print(f"SKIP {field}: missing {src}", flush=True)
            continue
        g = gpd.read_file(src)
        # Detect mislabeled Web-Mercator coords (declared 4326 but bounds out of lon/lat range).
        minx, miny, maxx, maxy = g.total_bounds
        if max(abs(minx), abs(maxx)) > 180 or max(abs(miny), abs(maxy)) > 90:
            g = g.set_crs("EPSG:3857", allow_override=True)
        elif g.crs is None:
            g = g.set_crs("EPSG:4326")
        g = g.to_crs("EPSG:4326")

        g = g[g.geometry.notnull()].copy()
        inv = ~g.geometry.is_valid
        if inv.any():
            g.loc[inv, "geometry"] = g.loc[inv, "geometry"].buffer(0)
        g = g[g.geometry.notnull() & ~g.geometry.is_empty].copy()

        g["name"] = g[name_col].map(lambda v: transform(v) if v is not None else "(None)")
        g["geometry"] = g.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
        g = g[["name", "geometry"]]

        out = OUT_DIR / f"houston-{field}-overlay.geojson"
        if out.exists():
            out.unlink()
        try:
            g.to_file(out, driver="GeoJSON", COORDINATE_PRECISION=PRECISION)
        except TypeError:
            g.to_file(out, driver="GeoJSON")
        kb = out.stat().st_size / 1024
        print(f"WROTE {out.name}: {len(g)} regions, {kb:,.0f} KB", flush=True)


if __name__ == "__main__":
    main()
