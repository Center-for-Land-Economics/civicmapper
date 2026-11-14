#!/usr/bin/env python3
"""
Add Overture building footprints (buildings.geojson) and public road centrelines (roads.geojson) to
parking-label campaigns (data/labels/<campaign>/).

Roads: Overture transportation segments of subtype road, excluding service roads, parking aisles,
driveways and paths (those are on-site circulation that counts as parking). The export removes
labelled parking that is both outside every parcel and within a road's half-width, i.e. the public
street, without touching parking on land whose parcel is merely missing from our data.

The label page shows them as an overlay (key O), and the export (label_export.py) subtracts them
from labelled parking, so labellers can draw lots straight across mapped buildings and only cut out
what Overture lacks (kiosks, sheds, new buildings, islands).

Usage:
  python data/scripts/label_buildings.py pilot-laredo pilot-plano
  python data/scripts/label_buildings.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.scripts import overture_source  # noqa: E402
from data.scripts.classify_parking_surface import OVERTURE_RELEASE  # noqa: E402

LABELS = ROOT / "data" / "labels"


def campaign_buildings(squares: gpd.GeoDataFrame, release: str | None = None) -> gpd.GeoDataFrame:
    """Overture buildings clipped to each square (square_id, class, subtype, height, geometry)."""
    release = release or overture_source.resolve_release(OVERTURE_RELEASE)
    # One read over all the squares' bbox: each read scans the footers of Overture's global building
    # files, so per-square reads cost minutes each. Clip locally afterwards.
    allb = overture_source.buildings(tuple(squares.total_bounds), release)
    parts = []
    if allb is not None and not allb.empty:
        idx = allb.sindex
        for _, sq in squares.iterrows():
            b = allb.iloc[idx.query(sq.geometry, predicate="intersects")].copy()
            if b.empty:
                continue
            b["geometry"] = b.geometry.intersection(sq.geometry)
            b = b[~b.geometry.is_empty & b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            b["square_id"] = sq["id"]
            parts.append(b)
    cols = ["square_id", "class", "subtype", "height", "geometry"]
    if not parts:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=4326)
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=4326)
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out["release"] = release
    return out[cols + ["release"]]


PUBLIC_ROAD_HALF_WIDTH_M = {"motorway": 15, "trunk": 12, "primary": 12, "secondary": 10, "tertiary": 8,
                            "residential": 6, "unclassified": 6, "living_street": 5}


def campaign_roads(squares: gpd.GeoDataFrame, release: str) -> gpd.GeoDataFrame:
    """Public road centrelines near each square (square_id, class, half_width_m, geometry)."""
    segs = overture_source.read("transportation", "segment", tuple(squares.total_bounds), release=release,
                                columns=["subtype", "class", "geometry"])
    segs = segs[(segs["subtype"] == "road") & segs["class"].isin(PUBLIC_ROAD_HALF_WIDTH_M)]
    parts = []
    if len(segs):
        idx = segs.sindex
        utm = squares.estimate_utm_crs()
        for _, sq in squares.iterrows():
            area = gpd.GeoSeries([sq.geometry], crs=4326).to_crs(utm).buffer(30).to_crs(4326).iloc[0]
            r = segs.iloc[idx.query(area, predicate="intersects")].copy()
            if r.empty:
                continue
            r["geometry"] = r.geometry.intersection(area)
            r["square_id"] = sq["id"]
            parts.append(r)
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=4326) if parts else \
        gpd.GeoDataFrame(columns=["square_id", "class", "geometry"], geometry="geometry", crs=4326)
    out["half_width_m"] = out["class"].map(PUBLIC_ROAD_HALF_WIDTH_M)
    out["release"] = release
    return out[["square_id", "class", "half_width_m", "release", "geometry"]]


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    feats = []
    for geom, (_, row) in zip(gdf.geometry, gdf.drop(columns="geometry").iterrows()):
        props = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}
        feats.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaigns", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    names = sorted(p.name for p in LABELS.glob("*") if (p / "squares.geojson").exists()) if a.all else a.campaigns
    if not names:
        ap.error("name a campaign or pass --all")
    release = overture_source.resolve_release(OVERTURE_RELEASE)
    for name in names:
        d = LABELS / name
        squares = gpd.read_file(d / "squares.geojson")
        b = campaign_buildings(squares, release)
        write_geojson(b, d / "buildings.geojson")
        roads = campaign_roads(squares, release)
        write_geojson(roads, d / "roads.geojson")
        meta = json.loads((d / "campaign.json").read_text())
        meta["buildings"] = {"source": "Overture Maps", "release": release, "count": int(len(b))}
        meta["roads"] = {"source": "Overture Maps", "release": release, "count": int(len(roads))}
        (d / "campaign.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"{name}: {len(b):,} building parts in {b['square_id'].nunique() if len(b) else 0} squares "
              f"and {len(roads):,} public road pieces (Overture {release})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
