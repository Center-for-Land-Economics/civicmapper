from __future__ import annotations

import numpy as np
import pandas as pd


SQM_TO_SQFT = 10.763910416709722

_GEOD = None


def _geod():
    """Lazily built WGS84 Geod, so importing this module never hard-requires pyproj."""
    global _GEOD
    if _GEOD is None:
        from pyproj import Geod
        _GEOD = Geod(ellps="WGS84")
    return _GEOD


PARKING_STRUCTURE_RE = r"Garage|Deck|Structure|Ramp"


def is_parking_structure_label(category) -> bool:
    """True when a land-use label names a parking STRUCTURE rather than a surface lot.

    House rule (issue #12): a deck/garage/ramp is a building on developed land, so it is judged
    by the ordinary improvement ratio instead of being force-labelled underused. Cities with
    their own row-wise classifier call this so they agree with classify_property_refined.
    """
    import re as _re
    return bool(_re.search(PARKING_STRUCTURE_RE, str(category or ""), _re.I))


def geodesic_area_sqft(geom) -> float:
    """Geodesic (WGS84) area of a lon/lat polygon in SQUARE FEET, interior rings SUBTRACTED.

    Every ETL used to carry its own copy of this, and all of them measured the EXTERIOR ring
    only, so a donut parcel (a lot with another parcel carved out of the middle) had the hole
    counted as part of its own land. Measured against the sources' own GIS area columns the
    overstatement ran ~1.24-1.58x median on affected parcels and up to ~2.8x. Two consequences:
    per-sqft values come out too LOW (so nothing looks visibly wrong), and — worse — an ETL that
    sanity-checks reported acreage against computed acreage will REJECT correct assessor acreage
    because the computed figure is inflated. Newport News shows the second effect plainly: 84%
    of its donut parcels fell back to computed area versus 0% citywide.

    Donut parcels are rare (0.0-0.3% of a typical roll), so this is an accuracy fix rather than
    an emergency; cities pick it up whenever they are next re-run.

    Returns NaN for missing/empty/non-polygonal geometry, matching the old per-ETL helpers.
    """
    if geom is None or getattr(geom, "is_empty", True):
        return np.nan
    gt = getattr(geom, "geom_type", None)
    if gt == "Polygon":
        g = _geod()
        a, _ = g.polygon_area_perimeter(*geom.exterior.coords.xy)
        total = abs(a)
        for ring in geom.interiors:
            ra, _ = g.polygon_area_perimeter(*ring.coords.xy)
            total -= abs(ra)
        return max(total, 0.0) * SQM_TO_SQFT
    if gt == "MultiPolygon":
        parts = [geodesic_area_sqft(sub) for sub in geom.geoms]
        parts = [v for v in parts if v == v]  # drop NaN
        return float(np.sum(parts)) if parts else np.nan
    return np.nan


# Both names exist across the ETLs; keep them interchangeable.
gis_area_sqft = geodesic_area_sqft


def check_area_agreement(computed_sqft, source_sqft, *, label="source area", tol=0.05, log=print):
    """Log how well our computed geodesic area agrees with the source's own area column.

    Individual parcels can legitimately differ; a shifted MEDIAN cannot, so this is the cheap
    tripwire that catches an area routine measuring the wrong thing (an exterior-only area lands
    near 1.25x on a hole-heavy roll, hole-subtracted lands at 1.0000). Logs and returns the
    median ratio; never raises, so an ETL can call it purely as a diagnostic.
    """
    c = pd.to_numeric(pd.Series(computed_sqft).reset_index(drop=True), errors="coerce")
    src = pd.to_numeric(pd.Series(source_sqft).reset_index(drop=True), errors="coerce")
    ratio = (c / src.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    med = ratio.median()
    off = int((ratio.sub(1).abs() > tol).sum())
    log(f"Area cross-check vs {label} — median ratio {med:.4f} (expect ~1.0000), "
        f"rows off by >{tol:.0%}: {off:,}")
    return med


def add_improvement_ratio_fields(
    df: pd.DataFrame,
    *,
    land_col: str,
    improvement_col: str,
    total_col: str = "TLLDIMPROV",
    ratio_col: str = "IMPR_LAND_RATIO",
    pct_col: str = "IMPR_LAND_PCT",
    pct_total_col: str = "IMPR_PCT_TOTAL",
) -> pd.DataFrame:
    """Add derived improvement/land ratio fields when missing."""
    land = pd.to_numeric(df.get(land_col), errors="coerce")
    improvement = pd.to_numeric(df.get(improvement_col), errors="coerce")
    total = land + improvement

    if total_col not in df.columns:
        df[total_col] = total

    if ratio_col not in df.columns:
        df[ratio_col] = np.where(land > 0, improvement / land, np.nan)

    if pct_col not in df.columns:
        df[pct_col] = np.where(land > 0, (improvement / land) * 100, np.nan)

    if pct_total_col not in df.columns:
        df[pct_total_col] = np.where(total > 0, (improvement / total) * 100, np.nan)

    return df


def smooth_land_value_per_sqft(
    gdf,
    *,
    land_value_col: str,
    area_sqft_col: str,
    geometry_col: str = "geometry",
    k: int = 10,
    neighbor_weight: float = 0.75,
    self_weight: float = 0.25,
    distance_power: float = 1.0,
    distance_floor: float = 10.0,
    distance_crs: str | None = None,
) -> pd.Series:
    """Return a smoothed land value per sqft using distance-weighted neighbors."""
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover - optional import
        raise ImportError("geopandas is required for smoothing land values.") from exc

    if not isinstance(gdf, gpd.GeoDataFrame):
        if geometry_col not in gdf.columns:
            raise ValueError(f"Expected '{geometry_col}' column for geometry.")
        gdf = gpd.GeoDataFrame(gdf, geometry=geometry_col)

    if gdf.empty:
        return pd.Series([], dtype="float64", index=gdf.index)

    if geometry_col not in gdf.columns:
        raise ValueError(f"Expected '{geometry_col}' column for geometry.")

    if k < 1:
        raise ValueError("k must be >= 1.")

    weight_total = neighbor_weight + self_weight
    if weight_total <= 0:
        raise ValueError("neighbor_weight + self_weight must be > 0.")
    neighbor_weight /= weight_total
    self_weight /= weight_total

    land_value = pd.to_numeric(gdf[land_value_col], errors="coerce")
    area_sqft = pd.to_numeric(gdf[area_sqft_col], errors="coerce")
    base_ppsf = land_value / area_sqft.replace(0, np.nan)

    if distance_crs:
        gdf_dist = gdf.to_crs(distance_crs)
    else:
        if gdf.crs is None:
            raise ValueError("CRS is required for distance calculations.")
        gdf_dist = gdf

    centroids = gdf_dist.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])

    n = len(gdf_dist)
    k_query = min(k + 1, n)

    distances = None
    indices = None

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=k_query)
    except Exception:
        # Fallback to a slower, spatial-index-based approach.
        sindex = gdf_dist.sindex
        distances = np.full((n, k_query), np.nan)
        indices = np.full((n, k_query), -1, dtype=int)
        for i, geom in enumerate(centroids):
            if geom is None or geom.is_empty:
                continue
            try:
                result = sindex.nearest(geom, return_distance=True)
                idxs, dists = result
                if hasattr(idxs, "shape") and len(idxs.shape) == 2:
                    idxs = idxs[1]
                idxs = np.asarray(idxs, dtype=int)
                dists = np.asarray(dists, dtype=float)
            except TypeError:
                idxs = []
                for pair in sindex.nearest(geom):
                    if isinstance(pair, (tuple, list)) and len(pair) == 2:
                        idxs.append(pair[1])
                    else:
                        idxs.append(pair)
                    if len(idxs) >= k_query:
                        break
                idxs = np.asarray(idxs, dtype=int)
                dists = np.linalg.norm(coords[idxs] - coords[i], axis=1)
            if idxs.size == 0:
                continue
            order = np.argsort(dists)
            idxs = idxs[order][:k_query]
            dists = dists[order][:k_query]
            indices[i, : len(idxs)] = idxs
            distances[i, : len(dists)] = dists

    if distances is None or indices is None:
        raise ValueError("Failed to compute nearest neighbors for smoothing.")

    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
    if indices.ndim == 1:
        indices = indices[:, np.newaxis]

    smoothed = np.full(n, np.nan, dtype="float64")
    base_vals = base_ppsf.to_numpy()

    for i in range(n):
        neighbor_idxs = indices[i]
        neighbor_dists = distances[i]
        valid_mask = neighbor_idxs >= 0
        neighbor_idxs = neighbor_idxs[valid_mask]
        neighbor_dists = neighbor_dists[valid_mask]

        # Drop self-index if present.
        self_mask = neighbor_idxs != i
        neighbor_idxs = neighbor_idxs[self_mask]
        neighbor_dists = neighbor_dists[self_mask]

        if neighbor_idxs.size == 0:
            smoothed[i] = base_vals[i]
            continue

        neighbor_idxs = neighbor_idxs[:k]
        neighbor_dists = neighbor_dists[:k]

        neighbor_vals = base_vals[neighbor_idxs]
        valid_vals = ~np.isnan(neighbor_vals)
        if not np.any(valid_vals):
            smoothed[i] = base_vals[i]
            continue

        neighbor_vals = neighbor_vals[valid_vals]
        neighbor_dists = neighbor_dists[valid_vals]

        weights = 1.0 / np.maximum(neighbor_dists, distance_floor) ** distance_power
        neighbor_avg = np.average(neighbor_vals, weights=weights)

        base_val = base_vals[i]
        if np.isnan(base_val):
            smoothed[i] = neighbor_avg
        else:
            smoothed[i] = self_weight * base_val + neighbor_weight * neighbor_avg

    return pd.Series(smoothed, index=gdf.index, name="smooth_land_value_per_sqft")


def classify_property_refined(
    gdf,
    *,
    sf_cutoff: float = 0.67,
    other_cutoff: float = 0.50,
    exclude_state_class_prefixes: tuple = ("X", "J", "D"),
    exclude_categories: tuple = ("Utility", "Agricultural / Rural"),
    category_col: str = "property_land_use_category",
    land_col: str = "land_value",
    improvement_col: str = "improvement_value",
    state_class_col: str = "state_class",
    bld_ar_col: str = "bld_ar",
    fetch_footprints: bool = True,
) -> pd.Series:
    """Classify each parcel as 'Vacant', 'Underdeveloped', 'Parking Lot', or None.

    Multi-signal rule (defaults calibrated for Houston / HCAD, 2026-05-30):
      - category contains 'Vacant'  -> 'Vacant'  (assessor already says so)
      - category contains 'Parking' -> 'Parking Lot', EXCEPT parking structures
            (garage/deck/ramp), which are buildings and fall through to the ratio test
      - improvement_value == 0      -> 'Vacant' ONLY if the parcel is genuinely empty:
            no building sqft (bld_ar == 0), no Overture building footprint, and not
            exempt/utility/ag (state_class prefix in `exclude_state_class_prefixes`
            or '1D', or category in `exclude_categories`). Otherwise None — this lets
            exempt land (parks, ROW, govt) and unvalued-but-built structures fall out.
      - improvement_value  > 0      -> 'Underdeveloped' if land/(land+improvement) >=
            cutoff, where cutoff = `sf_cutoff` for 'Single Family' else `other_cutoff`;
            else None.

    The Overture footprint check only runs for improvement_value == 0 candidates, and
    needs DuckDB + network access (set `fetch_footprints=False` to skip it). The cutoffs,
    exclusion prefixes and 'Single Family' label are market/assessor specific — pass them
    explicitly for jurisdictions other than Houston.
    """
    n = len(gdf)
    land = pd.to_numeric(gdf[land_col], errors="coerce").fillna(0).to_numpy(float)
    impr = pd.to_numeric(gdf[improvement_col], errors="coerce").fillna(0).to_numpy(float)
    total = land + impr

    cat = gdf[category_col].astype(str)
    is_vacant_cat = cat.str.contains("Vacant", na=False).to_numpy()
    is_parking_cat = cat.str.contains("Parking", na=False).to_numpy()
    # A parking STRUCTURE (deck / garage / ramp) is a building standing on developed land, so it
    # must NOT be force-labelled underused the way a surface lot is. Excluding it here lets it
    # fall through to the ordinary improvement-ratio test below and be judged on its merits.
    # This was the Lynchburg-vs-Baltimore disagreement in issue #12: Baltimore forced decks into
    # 'Parking Lot', Lynchburg judged them by ratio, and Lynchburg's reading is the house rule.
    is_parking_structure = cat.str.contains(
        r"Garage|Deck|Structure|Ramp", case=False, regex=True, na=False).to_numpy()
    is_parking_cat = is_parking_cat & ~is_parking_structure  # new array: to_numpy() view is read-only
    is_sf = (cat == "Single Family").to_numpy()
    cat_arr = cat.to_numpy()

    if state_class_col in gdf.columns:
        sc = gdf[state_class_col].fillna("").astype(str)
        excluded = sc.str[:1].isin(exclude_state_class_prefixes).to_numpy() \
            | sc.str[:2].eq("1D").to_numpy()
    else:
        excluded = np.zeros(n, dtype=bool)
    excluded |= np.isin(cat_arr, list(exclude_categories))

    if bld_ar_col in gdf.columns:
        bld_ar = pd.to_numeric(gdf[bld_ar_col], errors="coerce").fillna(0).to_numpy(float)
    else:
        bld_ar = np.zeros(n)

    # Overture footprint presence — only needed for the no-improvement-value candidates.
    no_value = impr == 0
    has_fp = np.zeros(n, dtype=bool)
    if fetch_footprints and no_value.any():
        import geopandas as gpd
        try:  # data/ on sys.path (notebook)
            from scripts.classify_parking_surface import fetch_overture_buildings
        except ImportError:  # repo root on sys.path (standalone scripts)
            from data.scripts.classify_parking_surface import fetch_overture_buildings
        bldg = fetch_overture_buildings(tuple(gdf.total_bounds))
        geom_col = gdf.geometry.name
        cand = gdf.loc[no_value, [geom_col]].copy()
        cand[geom_col] = cand[geom_col].apply(
            lambda g: g if (g is None or g.is_valid) else g.buffer(0))
        cand = cand.to_crs(3857).reset_index()
        joined = gpd.sjoin(cand, bldg[["geometry"]].to_crs(3857),
                           predicate="intersects", how="left")
        fp_idx = set(joined.loc[joined.index_right.notna(), "index"])
        has_fp = np.isin(np.arange(n), list(fp_idx))

    # Assign lowest precedence first; later writes win (matches the if/elif order above).
    cutoff = np.where(is_sf, sf_cutoff, other_cutoff)
    land_share = np.divide(land, total, out=np.zeros(n), where=total > 0)
    out = np.full(n, None, dtype=object)
    out[(~no_value) & (total > 0) & (land_share >= cutoff)] = "Underdeveloped"
    out[no_value & (bld_ar == 0) & ~has_fp & ~excluded] = "Vacant"
    out[is_parking_cat] = "Parking Lot"
    out[is_vacant_cat] = "Vacant"
    return pd.Series(out, index=gdf.index)
