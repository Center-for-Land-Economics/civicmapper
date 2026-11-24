#!/usr/bin/env python3
"""
Turn a citywide model run into a publishable "satellite-detected parking" layer: every lot carries
the evidence for it, and the metadata carries the validation and the calibrated citywide total.

Per lot (added columns):
  detection_source     "satellite_model" (vs OSM-derived lots in the regular layer)
  detection_method     one-line human description (imagery, model, threshold)
  imagery_date         NAIP acquisition date of the tile under the lot
  model_confidence     mean model probability inside the lot (0-1; lots exist only above the threshold)
  confidence           "high" (>= 0.9) / "medium" (>= 0.8) / "low" (below), from model_confidence
  in_osm               the lot overlaps a parking polygon in OpenStreetMap (independent confirmation)
  osm_overlap_share    share of the lot's area covered by OSM parking
  nearest_building_m   distance to the nearest building footprint (Overture)
  land_use             parcel land-use category under the lot
  validated_precision  measured share of mapped area that is real parking, for that land use, on
                       the hand-labelled random squares (null where the sample was too small)

Needs the run directory (parking_probs.tif, naip_tiles/, the lots parquet). Run after
parking_lot_extraction.py --model ...:
  python data/scripts/parking_evidence.py data/parking-model/laredo/finetune-dinov2-large-targeted --city laredo
Writes <city>-<state>-parking-lots-satellite.parquet and -metadata.json next to the run's output;
label_sync.py --citywide <city> shares them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VALIDATION = Path(__file__).resolve().parent / "parking_validation"
PUBLIC_DEV = "https://landeconomics.blob.core.windows.net/parquets-dev"


def mean_in_polygons(polys: gpd.GeoSeries, raster: Path, scale: float = 1.0) -> np.ndarray:
    """Mean of band 1 inside each polygon (windowed reads), times scale."""
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds
    out = np.full(len(polys), np.nan)
    with rasterio.open(raster) as src:
        g = polys.to_crs(src.crs)
        for k, geom in enumerate(g.values):
            if geom is None or geom.is_empty:
                continue
            win = from_bounds(*geom.bounds, transform=src.transform).round_offsets().round_lengths()
            if win.width < 1 or win.height < 1:
                continue
            a = src.read(1, window=win, boundless=True).astype(np.float32)
            inside = geometry_mask([geom], out_shape=a.shape, transform=src.window_transform(win), invert=True)
            if inside.any():
                out[k] = float(a[inside].mean()) * scale
    return out


def tile_dates(tile_dir: Path) -> gpd.GeoDataFrame:
    """Footprint and acquisition date of each cached NAIP tile (date from the file name)."""
    import rasterio
    from shapely.geometry import box
    rows = []
    for t in sorted(tile_dir.glob("*.tif")):
        m = re.search(r"_(\d{8})(?:_\d{8})?\.tif$", t.name)
        if not m:
            continue
        with rasterio.open(t) as src:
            fp = gpd.GeoSeries([box(*src.bounds)], crs=src.crs).to_crs(4326).iloc[0]
        d = m.group(1)
        rows.append({"imagery_date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "geometry": fp})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326) if rows else gpd.GeoDataFrame(
        {"imagery_date": []}, geometry=[], crs=4326)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--city", required=True)
    ap.add_argument("--state", default="tx")
    ap.add_argument("--validation", help=f"validation JSON (default {VALIDATION}/<city>.json)")
    ap.add_argument("--osm", help="OSM parking layer parquet (default: the published dev layer for the city)")
    a = ap.parse_args()

    run = Path(a.run_dir)
    stem = f"{a.city}-{a.state}-parking-lots"
    lots = gpd.read_parquet(run / f"{stem}.parquet")
    meta = json.loads((run / f"{stem}-metadata.json").read_text(encoding="utf-8"))
    vpath = Path(a.validation) if a.validation else VALIDATION / f"{a.city}.json"
    val = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else None
    if val is None:
        print(f"  no validation file at {vpath}: validated_precision will be empty")
    print(f"{len(lots):,} lots from {run}")

    # Only surface lots go into the satellite layer (structures and gas stations are dropped there).
    lots = lots[lots.get("parking_type", pd.Series("surface", index=lots.index)).fillna("surface") == "surface"]
    lots = lots.reset_index(drop=True)
    utm = lots.estimate_utm_crs()

    print("  model confidence (mean probability per lot)...", flush=True)
    conf = mean_in_polygons(lots.geometry, run / "parking_probs.tif", scale=1 / 255)
    lots["model_confidence"] = np.round(conf, 3)
    lots["confidence"] = np.where(conf >= 0.9, "high", np.where(conf >= 0.8, "medium", "low"))

    print("  imagery dates...", flush=True)
    tiles = tile_dates(run / "naip_tiles")
    if len(tiles):
        pts = gpd.GeoDataFrame(geometry=lots.geometry.representative_point(), crs=lots.crs).to_crs(4326)
        j = gpd.sjoin(pts, tiles, predicate="within", how="left")
        lots["imagery_date"] = j[~j.index.duplicated()]["imagery_date"].reindex(lots.index).values
    else:
        lots["imagery_date"] = None

    print("  OpenStreetMap overlap...", flush=True)
    osm_src = a.osm or f"{PUBLIC_DEV}/parking/{stem}.parquet"
    try:
        osm = gpd.read_parquet(osm_src) if not str(osm_src).startswith("http") else _read_remote(osm_src)
        osm = osm[osm.get("parking_type", pd.Series("surface", index=osm.index)).fillna("surface") != "excluded"]
        ou = osm.to_crs(utm).geometry
        ou = ou[ou.notna() & ~ou.is_empty]
        lu_ = lots.to_crs(utm).geometry
        share = np.zeros(len(lots))
        hits = ou.sindex.query(lu_.values, predicate="intersects")
        for i in np.unique(hits[0]):
            share[i] = lu_.iloc[i].intersection(ou.iloc[hits[1][hits[0] == i]].union_all()).area / lu_.iloc[i].area
        lots["osm_overlap_share"] = np.round(share, 3)
        lots["in_osm"] = share >= 0.2
        print(f"    {int(lots['in_osm'].sum()):,} lots also mapped in OSM ({len(osm):,} OSM lots)")
    except Exception as e:  # noqa: BLE001
        print(f"    skipped ({e})")
        lots["osm_overlap_share"] = np.nan
        lots["in_osm"] = None

    print("  nearest building...", flush=True)
    try:
        from data.scripts import overture_source
        from data.scripts.classify_parking_surface import OVERTURE_RELEASE
        from data.scripts.parking_postprocess import PARKING_BUILDING_CLASSES
        bl = overture_source.buildings(tuple(lots.to_crs(4326).total_bounds),
                                       overture_source.resolve_release(OVERTURE_RELEASE)).to_crs(utm)
        if "class" in bl.columns:
            bl = bl[~bl["class"].fillna("").str.lower().isin(PARKING_BUILDING_CLASSES)]
        idx, dist = bl.sindex.nearest(lots.to_crs(utm).geometry, return_distance=True)
        lots["nearest_building_m"] = pd.Series(dist, index=idx[0]).groupby(level=0).min() \
            .reindex(range(len(lots))).round(1).values
    except Exception as e:  # noqa: BLE001
        print(f"    skipped ({e})")
        lots["nearest_building_m"] = np.nan

    print("  land use and validated precision...", flush=True)
    from data.parquet_registry import CITY_PARQUETS
    cm = CITY_PARQUETS.get(a.city)
    ppath = ROOT / "data" / "jurisidictions" / "data" / a.city / (cm.legacy_filename if cm else f"{a.city}-{a.state}-parcels.parquet")
    if ppath.exists():
        par = gpd.read_parquet(ppath, columns=["property_land_use_category", "geometry"]).to_crs(utm)
        pts = gpd.GeoDataFrame(geometry=lots.to_crs(utm).geometry.representative_point(), crs=utm)
        j = gpd.sjoin(pts, par, predicate="within", how="left")
        lots["land_use"] = j[~j.index.duplicated()]["property_land_use_category"].reindex(lots.index) \
            .fillna("(no parcel)").values  # no parcel in the appraisal data: public / exempt land
    else:
        lots["land_use"] = lots.get("context_category")
    bylu = (val or {}).get("by_land_use", {})
    lots["validated_precision"] = [((bylu.get(u) or {}).get("precision")) for u in lots["land_use"]]

    lots["detection_source"] = "satellite_model"
    method = (f"Detected by a machine-learning model in {val['imagery'] if val else 'NAIP aerial imagery'}; "
              f"model probability threshold {val['threshold'] if val else meta['parkingTotals'].get('model')}.")
    lots["detection_method"] = method
    lots["source"] = "satellite_model"

    out = run / f"{stem}-satellite.parquet"
    lots.to_parquet(out)
    meta["layer"] = "satellite_detected"
    meta["detection"] = {
        "source": "satellite_model",
        "method": method,
        "model": (val or {}).get("model_description") or meta["parkingTotals"].get("model"),
        "threshold": (val or {}).get("threshold"),
        "validation": {k: val[k] for k in ("validated", "validation_set", "overall", "by_land_use", "by_confidence", "notes")
                       if k in val} if val else None,
        "calibration": (val or {}).get("calibration"),
        "lots": int(len(lots)),
        "lots_in_osm": int(lots["in_osm"].fillna(False).sum()) if "in_osm" in lots else None,
        "confidence_counts": lots["confidence"].value_counts().to_dict(),
    }
    (run / f"{stem}-satellite-metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out} ({len(lots):,} lots) and its metadata JSON")
    # Local preview: the dev server serves viz/public/ first (gitignored), so the parking page shows
    # this layer right away, before it is uploaded anywhere.
    import shutil
    pub = ROOT / "viz" / "public" / "parking"
    pub.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, pub / out.name)
    shutil.copy2(run / f"{stem}-satellite-metadata.json", pub / f"{stem}-satellite-metadata.json")
    print(f"  copied to {pub} for local preview (restart the dev server if it was running)")
    print(f"  confidence: {lots['confidence'].value_counts().to_dict()}; "
          f"median model confidence {np.nanmedian(conf):.2f}")
    return 0


def _read_remote(url: str) -> gpd.GeoDataFrame:
    import io
    import requests
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return gpd.read_parquet(io.BytesIO(r.content))


if __name__ == "__main__":
    sys.exit(main())
