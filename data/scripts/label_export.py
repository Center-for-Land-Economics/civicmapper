#!/usr/bin/env python3
"""
Export parking labels to training masks, clean shapes and a summary.

For every square's latest label per labeller (data/labels/<campaign>/labels.jsonl) this builds a
mask on a 0.6 m grid (NAIP's pixel size) in the square's UTM zone, in this order:
  1. polygons of class paved / unpaved / structure / unsure, in the order drawn;
  2. cut-out polygons clear what they cover;
  3. Overture buildings (buildings.geojson, see label_buildings.py) clear paved / unpaved / unsure
     (garage/roof "structure" labels are kept: they mark structured parking to exclude);
  4. public streets are cleared: cells outside every parcel AND within a public road's half-width
     (roads.geojson, see label_buildings.py), so labellers need not stop exactly at the curb.
     Requiring both keeps parking on land whose parcel is merely missing from our data; such
     parking is reported as off_parcel_m2 (a large value points at a parcel-data gap);
  5. paint overrides everything: painted classes set the cell, painted cut-out clears it.

Mask values: 0 not parking, 1 paved, 2 unpaved (gravel/caliche/dirt), 3 structure (garage or
rooftop parking, never surface), 4 unsure (ignore in training/scoring).

Writes data/labels/<campaign>/export/:
  masks/<square>__<labeller>.tif   uint8 GeoTIFF (UTM, 0.6 m)
  parking.geojson                  mask polygonised (EPSG:4326), one feature per class region
  summary.csv                      per label: outcome, areas by class, buildings removed, OSM overlap
and prints campaign totals: parking share of parcel land by land use, OSM coverage, agreement on
double-labelled squares.

Usage:
  python data/scripts/label_export.py pilot-laredo
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labels"
CELL = 0.6
CODES = {"paved": 1, "unpaved": 2, "structure": 3, "unsure": 4}
NAMES = {v: k for k, v in CODES.items()}
PARKING = (1, 2)  # surface parking for shares and scoring
COLORMAP = {0: (0, 0, 0, 0), 1: (255, 77, 109, 255), 2: (244, 162, 89, 255), 3: (179, 136, 255, 255),
            4: (255, 224, 102, 255)}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "anon"


def latest_labels(path: Path) -> list[dict]:
    """Latest submission per (square, labeller); skips dropped."""
    latest: dict[tuple[str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["square_id"], r["labeler"].strip().lower())] = r
    return [r for r in latest.values() if r["outcome"] != "skip"]


def decode_rle(rle: str, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.uint8)
    k = 0
    for part in rle.split(","):
        v, n = (int(x) for x in part.split(":"))
        out[k:k + n] = v
        k += n
    return out


def paint_lookup(paint: dict, lng: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Paint value at each lng/lat, using the same corner affine as the label page."""
    tl, tr, _br, bl = paint["corners"]
    kx = math.cos(math.radians(tl[1]))
    ex = ((tr[0] - tl[0]) * kx, tr[1] - tl[1])
    ey = ((bl[0] - tl[0]) * kx, bl[1] - tl[1])
    det = ex[0] * ey[1] - ey[0] * ex[1]
    dx, dy = (lng - tl[0]) * kx, lat - tl[1]
    u = (ey[1] * dx - ey[0] * dy) / det
    v = (-ex[1] * dx + ex[0] * dy) / det
    w, h = paint["width"], paint["height"]
    i, j = np.floor(u * w).astype(int), np.floor(v * h).astype(int)
    ok = (i >= 0) & (i < w) & (j >= 0) & (j < h)
    cells = decode_rle(paint["rle"], w * h)
    out = np.zeros(lng.shape, dtype=np.uint8)
    out[ok] = cells[j[ok] * w + i[ok]]
    return out


def square_grid(sq_utm) -> tuple[rasterio.Affine, int, int]:
    x0, y0, x1, y1 = sq_utm.bounds
    w, h = math.ceil((x1 - x0) / CELL - 1e-6), math.ceil((y1 - y0) / CELL - 1e-6)
    return from_origin(x0, y1, CELL, CELL), w, h


def build_mask(label: dict, sq_utm, utm, buildings_utm: list, street: np.ndarray | None,
               transform, w: int, h: int) -> tuple[np.ndarray, int, int]:
    """-> (mask, cells removed as Overture buildings, cells removed as public street)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if label["outcome"] != "labeled":
        return mask, 0, 0

    def utm_geom(g):
        return gpd.GeoSeries([shape(g)], crs=4326).to_crs(utm).iloc[0].buffer(0)

    polys = [(utm_geom(f["geometry"]), f["properties"]["class"]) for f in label["polygons"]]
    filled = [(g, CODES[c]) for g, c in polys if c in CODES and not g.is_empty]
    if filled:
        mask = rasterize(filled, out_shape=(h, w), transform=transform, fill=0, dtype="uint8")
    cuts = [g for g, c in polys if c == "cutout" and not g.is_empty]
    if cuts:
        mask[rasterize([(g, 1) for g in cuts], out_shape=(h, w), transform=transform, fill=0, dtype="uint8") == 1] = 0
    removed = 0
    if buildings_utm:
        b = rasterize([(g, 1) for g in buildings_utm], out_shape=(h, w), transform=transform, fill=0, dtype="uint8") == 1
        hit = b & np.isin(mask, (1, 2, 4))
        removed = int(hit.sum())
        mask[hit] = 0
    row_removed = 0
    if street is not None:
        off = street & (mask > 0)
        row_removed = int(off.sum())
        mask[off] = 0
    if label.get("paint"):
        cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        xs, ys = transform * (cols, rows)
        lng, lat = Transformer.from_crs(utm, 4326, always_xy=True).transform(xs, ys)
        p = paint_lookup(label["paint"], np.asarray(lng), np.asarray(lat))
        mask = np.where((p >= 1) & (p <= 4), p, mask)
        mask[p == 5] = 0
    return mask.astype(np.uint8), removed, row_removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaign")
    a = ap.parse_args()
    d = LABELS / a.campaign
    out = d / "export"
    (out / "masks").mkdir(parents=True, exist_ok=True)

    squares = gpd.read_file(d / "squares.geojson")
    utm = squares.estimate_utm_crs()
    sq_utm = squares.to_crs(utm).set_index("id")
    sq_props = squares.set_index("id")
    parcels = gpd.read_file(d / "parcels.geojson").to_crs(utm)
    prefill = gpd.read_file(d / "prefill.geojson").to_crs(utm) if (d / "prefill.geojson").exists() else None
    bpath = d / "buildings.geojson"
    buildings = gpd.read_file(bpath).to_crs(utm) if bpath.exists() else None
    if buildings is None:
        print("! no buildings.geojson: run label_buildings.py first; buildings are NOT subtracted")
    rpath = d / "roads.geojson"
    roads = gpd.read_file(rpath).to_crs(utm) if rpath.exists() else None
    if roads is None:
        print("! no roads.geojson: run label_buildings.py first; public streets are NOT removed")

    labels = latest_labels(d / "labels.jsonl")
    rows, feats = [], []
    share_num: Counter = Counter()
    share_den: Counter = Counter()
    masks_by_square: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    osm_hit = drawn_total = osm_total = 0.0
    for lab in sorted(labels, key=lambda r: (r["square_id"], r["labeler"].lower())):
        sid = lab["square_id"]
        sq = sq_utm.loc[sid].geometry
        transform, w, h = square_grid(sq)
        b_utm = list(buildings[buildings.square_id == sid].geometry) if buildings is not None else []
        pg = parcels[parcels.square_id == sid]
        parcel_land = (rasterize([(g, 1) for g in pg.geometry], out_shape=(h, w), transform=transform, fill=0,
                                 dtype="uint8") == 1) if len(pg) else np.zeros((h, w), bool)
        street = None
        if roads is not None:
            rs = roads[roads.square_id == sid]
            if len(rs):
                road = rasterize([(g.buffer(hw), 1) for g, hw in zip(rs.geometry, rs.half_width_m)],
                                 out_shape=(h, w), transform=transform, fill=0, dtype="uint8") == 1
                street = road & ~parcel_land
        mask, removed, row_removed = build_mask(lab, sq, utm, b_utm, street, transform, w, h)
        off_parcel = int((np.isin(mask, PARKING) & ~parcel_land).sum())
        inside = rasterize([(sq, 1)], out_shape=(h, w), transform=transform, fill=0, dtype="uint8") == 1
        mask[~inside] = 0
        name = f"{sid}__{slug(lab['labeler'])}"
        if lab["outcome"] != "cant_tell":
            masks_by_square[sid].append((lab["labeler"], mask))
            with rasterio.open(out / "masks" / f"{name}.tif", "w", driver="GTiff", width=w, height=h, count=1,
                               dtype="uint8", crs=utm, transform=transform, compress="deflate") as ds:
                ds.write(mask, 1)
                ds.write_colormap(1, COLORMAP)
                ds.update_tags(square_id=sid, labeler=lab["labeler"], outcome=lab["outcome"],
                               imagery=str(lab.get("imagery")), classes="0 none,1 paved,2 unpaved,3 structure,4 unsure")
            for geom, val in shapes(mask, mask=mask > 0, transform=transform):
                g = gpd.GeoSeries([shape(geom)], crs=utm).to_crs(4326).iloc[0]
                feats.append({"type": "Feature", "geometry": mapping(g), "properties": {
                    "square_id": sid, "labeler": lab["labeler"], "class": NAMES[int(val)],
                    "area_m2": round(shape(geom).area, 1), "imagery": lab.get("imagery"),
                    "naip_year": lab.get("naip_year")}})
        cell_m2 = CELL * CELL
        areas = {c: int((mask == v).sum()) * cell_m2 for c, v in CODES.items()}
        # parking vs existing OSM/Overture lots
        park = np.isin(mask, PARKING)
        osm_m2 = hit_m2 = 0.0
        if prefill is not None:
            pf = prefill[prefill.square_id == sid]
            if len(pf):
                pm = rasterize([(g, 1) for g in pf.geometry], out_shape=(h, w), transform=transform, fill=0,
                               dtype="uint8") == 1
                osm_m2, hit_m2 = pm.sum() * cell_m2, (pm & park).sum() * cell_m2
        if lab["outcome"] in ("labeled", "no_parking"):
            osm_total += osm_m2; osm_hit += hit_m2; drawn_total += park.sum() * cell_m2
            # parking share of parcel land, by parcel land use
            for cat, grp in parcels[parcels.square_id == sid].groupby("category"):
                pm = rasterize([(g, 1) for g in grp.geometry], out_shape=(h, w), transform=transform, fill=0,
                               dtype="uint8") == 1
                share_den[cat] += pm.sum() * cell_m2
                share_num[cat] += (pm & park).sum() * cell_m2
        rows.append({"square_id": sid, "labeler": lab["labeler"], "outcome": lab["outcome"],
                     "stratum": sq_props.loc[sid, "stratum"], "mode": lab.get("mode"),
                     **{f"{c}_m2": round(v) for c, v in areas.items()},
                     "buildings_removed_m2": round(removed * cell_m2),
                     "row_removed_m2": round(row_removed * cell_m2),
                     "off_parcel_m2": round(off_parcel * cell_m2), "osm_m2": round(osm_m2),
                     "osm_overlap_m2": round(hit_m2), "seconds": lab.get("seconds"),
                     "flags": ";".join(lab.get("flags") or []), "imagery": lab.get("imagery")})

    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    (out / "parking.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

    print(f"{a.campaign}: {len(df)} labels on {df['square_id'].nunique() if len(df) else 0} squares "
          f"-> {out.relative_to(ROOT)}")
    if not len(df):
        return 0
    print(df["outcome"].value_counts().to_string())
    lab_df = df[df.outcome == "labeled"]
    print(f"\nArea in labelled squares (net of cut-outs and Overture buildings): "
          f"paved {lab_df.paved_m2.sum():,.0f} m², unpaved {lab_df.unpaved_m2.sum():,.0f} m², "
          f"structure {lab_df.structure_m2.sum():,.0f} m², unsure {lab_df.unsure_m2.sum():,.0f} m²; "
          f"buildings removed {df.buildings_removed_m2.sum():,.0f} m²; public streets removed "
          f"{df.row_removed_m2.sum():,.0f} m²")
    gaps = df[df.off_parcel_m2 >= 500]
    if len(gaps):
        print("Parking kept on land with no parcel in our data (possible parcel gaps): " +
              ", ".join(f"{r.square_id} {r.off_parcel_m2:,} m²" for r in gaps.itertuples()))
    if drawn_total:
        print(f"Existing OSM/Overture lots cover {osm_hit / drawn_total:.0%} of labelled surface parking "
              f"({osm_hit:,.0f} of {drawn_total:,.0f} m²); {osm_hit / osm_total if osm_total else float('nan'):.0%} "
              f"of their area is parking in the labels")
    print("\nSurface parking share of parcel land, by land use (labelled + no-parking squares):")
    for cat in sorted(share_den, key=lambda c: -share_den[c]):
        if share_den[cat] > 0:
            print(f"  {cat:<24} {share_num[cat] / share_den[cat]:6.1%}   ({share_num[cat]:,.0f} of {share_den[cat]:,.0f} m²)")
    pairs = {s: m for s, m in masks_by_square.items() if len(m) >= 2}
    if pairs:
        print("\nAgreement on squares with 2+ labellers (IoU of surface parking):")
        for s, ms in pairs.items():
            (n1, a1), (n2, a2) = ms[:2]
            p1, p2 = np.isin(a1, PARKING), np.isin(a2, PARKING)
            union = (p1 | p2).sum()
            print(f"  {s}: {n1} vs {n2}: IoU {(p1 & p2).sum() / union if union else 1:.2f}")
    else:
        print("\nNo square has two labellers yet (agreement needs the double-labelled squares done twice).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
