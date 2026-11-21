"""
Clean-up applied to parking-model output: remove building footprints and public streets.

The same rules the label export applies to hand labels (label_export.py), so predictions and
labels are measured on equal terms:
  - buildings: Overture footprints are not surface parking. Rooftops of flat mid-rise buildings are
    the model's most common false alarm outside Laredo (PRN downtowns, 2026-11: removing them raised
    the Laredo-tuned model's median IoU from 45% to 51% with no recall loss).
  - streets: cells within a public road's half-width (Overture road classes, no service roads,
    which are parking aisles and driveways) AND outside every parcel. Requiring "outside every
    parcel" keeps lots whose parcel merely touches the road; without parcels the road buffer alone
    is used.

Raster form (masks on a grid) for the label tools and evaluations; vector form for the citywide
pipeline (parking_lot_extraction.py), which keeps buildings tagged as parking structures for its
surface-vs-structure classifier.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

PUBLIC_ROAD_HALF_WIDTH_M = {"motorway": 15, "trunk": 12, "primary": 12, "secondary": 10, "tertiary": 8,
                            "residential": 6, "unclassified": 6, "living_street": 5}
PARKING_BUILDING_CLASSES = {"parking", "carport", "garage", "garages"}


def _raster(geoms, shape, transform) -> np.ndarray:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, bool)
    return rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, fill=0, dtype="uint8") == 1


def street_mask(shape, transform, roads: gpd.GeoDataFrame | None, parcels=None) -> np.ndarray:
    """Public street cells: within a road's half-width and (when parcels are given) outside all parcels.
    roads: geometry + half_width_m (or class) in the grid's CRS. parcels: iterable of geometries or None."""
    if roads is None or len(roads) == 0:
        return np.zeros(shape, bool)
    hw = roads["half_width_m"] if "half_width_m" in roads else roads["class"].map(PUBLIC_ROAD_HALF_WIDTH_M)
    road = _raster([g.buffer(float(w)) for g, w in zip(roads.geometry, hw.fillna(6))], shape, transform)
    if parcels is None:
        return road
    return road & ~_raster(list(parcels), shape, transform)


def clean_mask(mask: np.ndarray, transform, buildings=None, roads: gpd.GeoDataFrame | None = None,
               parcels=None) -> tuple[np.ndarray, int, int]:
    """-> (cleaned mask, cells removed as buildings, cells removed as street)."""
    built = _raster(list(buildings), mask.shape, transform) if buildings is not None else np.zeros(mask.shape, bool)
    street = street_mask(mask.shape, transform, roads, parcels)
    b_hit = mask & built
    s_hit = mask & street & ~built
    return mask & ~built & ~street, int(b_hit.sum()), int(s_hit.sum())


def public_roads(bbox: tuple, release: str) -> gpd.GeoDataFrame:
    """Overture public road centrelines in bbox (EPSG:4326) with half_width_m."""
    from data.scripts import overture_source
    segs = overture_source.read("transportation", "segment", bbox, release=release,
                                columns=["subtype", "class", "geometry"])
    segs = segs[(segs["subtype"] == "road") & segs["class"].isin(PUBLIC_ROAD_HALF_WIDTH_M)].copy()
    segs["half_width_m"] = segs["class"].map(PUBLIC_ROAD_HALF_WIDTH_M)
    return segs[["class", "half_width_m", "geometry"]]


def subtract_buildings(parking: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame,
                       min_area_m2: float = 50.0) -> gpd.GeoDataFrame:
    """Vector form for the pipeline: cut non-parking building footprints out of parking polygons.
    Buildings tagged as parking structures are kept so the surface-vs-structure step still sees them."""
    if parking.empty or buildings is None or buildings.empty:
        return parking
    utm = parking.estimate_utm_crs()
    p = parking.to_crs(utm)
    b = buildings.to_crs(utm)
    if "class" in b.columns:
        b = b[~b["class"].fillna("").str.lower().isin(PARKING_BUILDING_CLASSES)]
    if b.empty:
        return parking
    idx = b.sindex
    geoms = []
    n = len(p)
    for k, g in enumerate(p.geometry, 1):
        hit = b.iloc[idx.query(g, predicate="intersects")]
        geoms.append(g.difference(hit.union_all()) if len(hit) else g)
        if n >= 20000 and k % 10000 == 0:
            print(f"    buildings cut from {k:,}/{n:,} polygons", flush=True)
    p = p.set_geometry(geoms)
    p = p[~p.geometry.is_empty & (p.geometry.area >= min_area_m2)]
    return p.to_crs(parking.crs)
