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
        if len(hit):
            try:
                g = g.difference(hit.union_all())
            except Exception:  # invalid ring from an earlier cut: repair and retry
                import shapely
                g = _polygonal(shapely.make_valid(g)).difference(shapely.make_valid(hit.union_all()))
        geoms.append(g)
        if n >= 20000 and k % 10000 == 0:
            print(f"    buildings cut from {k:,}/{n:,} polygons", flush=True)
    import shapely
    p = p.set_geometry([_polygonal(x) for x in shapely.make_valid(np.array(geoms, dtype=object))])
    p = p[p.geometry.notna() & ~p.geometry.is_empty]
    p = p[p.geometry.area >= min_area_m2]
    return p.to_crs(parking.crs)


# Context rules for model output, chosen on the Laredo random squares (2026-11-22/23). Precision
# first (the client prefers an undercount to false parking); the citywide total is calibrated anyway.
#   far from buildings + single-family driveways: false parking 42.5 -> 20.8 ha, real 133.6 -> 133.2
#   + rural parcels, bare soil, under 0.1 acre:     precision 87.8% -> 91.7%, recall x0.979
MAX_BUILDING_DISTANCE_M = 100.0  # parking serves buildings; bare ground far from any building is not a lot
SMALL_SINGLE_FAMILY_ACRES = 0.1  # small patches on single-family parcels are driveways
SINGLE_FAMILY_USES = {"single family"}
RURAL_USES = {"agricultural / rural"}  # 12% precision on rural parcels: yards, pads, bare ground
BARE_SOIL_NDVI = 0.0             # pavement reads clearly negative (median -0.14 on real lots); soil does not
MIN_LOT_ACRES = 0.1              # small isolated patches were mostly false


def _mean_ndvi(polys: gpd.GeoSeries, naip_path) -> np.ndarray:
    """Mean NDVI (NAIP band 4 = NIR, band 1 = red) inside each polygon; NaN where unreadable."""
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds
    out = np.full(len(polys), np.nan)
    with rasterio.open(naip_path) as src:
        if src.count < 4:
            return out
        g = polys.to_crs(src.crs)
        for k, geom in enumerate(g.values):
            if geom is None or geom.is_empty:
                continue
            win = from_bounds(*geom.bounds, transform=src.transform).round_offsets().round_lengths()
            if win.width < 1 or win.height < 1:
                continue
            red = src.read(1, window=win, boundless=True).astype(np.float32)
            nir = src.read(4, window=win, boundless=True).astype(np.float32)
            inside = geometry_mask([geom], out_shape=red.shape, transform=src.window_transform(win), invert=True)
            den = (nir + red)[inside]
            ok = den > 0
            if ok.any():
                out[k] = float(np.mean((nir[inside][ok] - red[inside][ok]) / den[ok]))
            if (k + 1) % 10000 == 0:
                print(f"    bare-soil check {k + 1:,}/{len(g):,}", flush=True)
    return out


def context_filter(parking: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame | None,
                   parcels: gpd.GeoDataFrame | None, naip_path=None,
                   max_building_m: float = MAX_BUILDING_DISTANCE_M,
                   small_sf_acres: float = SMALL_SINGLE_FAMILY_ACRES,
                   min_acres: float = MIN_LOT_ACRES,
                   bare_soil_ndvi: float | None = BARE_SOIL_NDVI) -> gpd.GeoDataFrame:
    """Drop model polygons that are
      1. more than max_building_m from any non-parking building,
      2. smaller than small_sf_acres and on a single-family parcel (driveways),
      3. on an agricultural / rural parcel,
      4. smaller than min_acres,
      5. bare soil: mean NDVI above bare_soil_ndvi (needs the 4-band NAIP mosaic).
    A rule is skipped when its input is missing. Prints what each rule removed."""
    if parking.empty:
        return parking
    from pathlib import Path
    utm = parking.estimate_utm_crs()
    p = parking.to_crs(utm).reset_index(drop=True)
    acres = p.geometry.area.to_numpy() / 4046.86
    drop = np.zeros(len(p), bool)

    def report(rule, mask, label):
        new = mask & ~drop
        print(f"  {label}: {new.sum():,} polygon(s), {acres[new].sum():,.0f} ac removed", flush=True)
        return drop | mask

    if buildings is not None and not buildings.empty:
        b = buildings.to_crs(utm)
        if "class" in b.columns:
            b = b[~b["class"].fillna("").str.lower().isin(PARKING_BUILDING_CLASSES)]
        if not b.empty:
            near = b.sindex.query(p.geometry.values, predicate="dwithin", distance=max_building_m)
            has_near = np.zeros(len(p), bool)
            has_near[np.unique(near[0])] = True
            drop = report(1, ~has_near, f"No building within {max_building_m:.0f} m")
    if parcels is not None and not parcels.empty and "property_land_use_category" in parcels.columns:
        par = parcels[["property_land_use_category", "geometry"]].to_crs(utm)
        live = np.flatnonzero(~drop)
        pts = gpd.GeoDataFrame(geometry=p.geometry.iloc[live].representative_point().values, crs=utm)
        hit = gpd.sjoin(pts, par, predicate="within", how="left")
        hit = hit[~hit.index.duplicated()]
        use = np.full(len(p), "", dtype=object)
        use[live] = hit["property_land_use_category"].fillna("").str.lower().to_numpy()
        drop = report(2, np.isin(use, list(SINGLE_FAMILY_USES)) & (acres < small_sf_acres),
                      f"Small (<{small_sf_acres} ac) on single-family parcels")
        drop = report(3, np.isin(use, list(RURAL_USES)), "On agricultural / rural parcels")
    if min_acres:
        drop = report(4, acres < min_acres, f"Smaller than {min_acres} ac")
    if bare_soil_ndvi is not None and naip_path is not None and Path(naip_path).exists():
        live = np.flatnonzero(~drop)
        print(f"  Bare-soil check on {len(live):,} polygon(s) (mean NDVI from NAIP)...", flush=True)
        ndvi = np.full(len(p), np.nan)
        ndvi[live] = _mean_ndvi(p.geometry.iloc[live], naip_path)
        drop = report(5, ndvi > bare_soil_ndvi, f"Bare soil (mean NDVI > {bare_soil_ndvi})")
    return p[~drop].to_crs(parking.crs)


def _polygonal(g):
    """Polygonal part of a geometry (None if there is none)."""
    if g is None or g.is_empty:
        return None
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    if g.geom_type == "GeometryCollection":
        from shapely.geometry import MultiPolygon
        parts = [q for x in g.geoms if x.geom_type in ("Polygon", "MultiPolygon")
                 for q in (x.geoms if x.geom_type == "MultiPolygon" else [x])]
        return MultiPolygon(parts) if parts else None
    return None


def subtract_public_streets(parking: gpd.GeoDataFrame, roads: gpd.GeoDataFrame | None,
                            parcels: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    """Vector form of the label rule (street_mask): remove parking that is within a public road's
    half-width AND outside every parcel. Parking aisles, driveways and footpaths are not public roads
    and are left alone, as in the labels."""
    if parking.empty or roads is None or len(roads) == 0:
        return parking
    import shapely
    utm = parking.estimate_utm_crs()
    p = parking.to_crs(utm).reset_index(drop=True)
    r = roads.to_crs(utm)
    hw = r["half_width_m"] if "half_width_m" in r else r["class"].map(PUBLIC_ROAD_HALF_WIDTH_M)
    rbuf = shapely.buffer(r.geometry.values, hw.fillna(6).to_numpy(float))
    rtree = shapely.STRtree(rbuf)
    par = parcels.to_crs(utm).geometry.values if parcels is not None and len(parcels) else None
    ptree = shapely.STRtree(par) if par is not None else None
    geoms = p.geometry.values.copy()
    pi, ri = rtree.query(geoms, predicate="intersects")
    order = np.argsort(pi, kind="stable")
    pi, ri = pi[order], ri[order]
    starts = np.flatnonzero(np.r_[True, pi[1:] != pi[:-1]]) if len(pi) else np.array([], int)
    ends = np.r_[starts[1:], len(pi)] if len(pi) else np.array([], int)
    for k, (a, b) in enumerate(zip(starts, ends), 1):
        i = pi[a]
        street = shapely.intersection(shapely.union_all(rbuf[ri[a:b]]), geoms[i])
        if ptree is not None and not street.is_empty:
            hits = ptree.query(street, predicate="intersects")
            if len(hits):
                street = shapely.difference(street, shapely.union_all(par[hits]))
        if not street.is_empty:
            geoms[i] = shapely.difference(geoms[i], street)
        if k % 10000 == 0:
            print(f"    streets cut from {k:,}/{len(starts):,} polygons", flush=True)
    before = p.geometry.area.sum()
    p = p.set_geometry(geoms)
    # differences can leave self-touching rings; repair, keep the polygonal part
    p["geometry"] = [_polygonal(x) for x in shapely.make_valid(p.geometry.values)]
    p = p[p.geometry.notna() & ~p.geometry.is_empty]
    p = p[p.geometry.area >= 46.5]  # 500 sq ft, as the pipeline's min area
    print(f"  Public streets off-parcel: {(before - p.geometry.area.sum()) / 4046.86:,.0f} ac removed "
          f"({len(roads):,} road segments)")
    return p.to_crs(parking.crs)
