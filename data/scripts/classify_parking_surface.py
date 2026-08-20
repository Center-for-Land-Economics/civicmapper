"""classify_parking_surface.py  — PROOF OF CONCEPT (Houston)

Differentiate real surface parking lots from parking structures (decks /
garages / podiums) in the OSM-derived parking layer, and pro-rate land value
to the effective *unbuilt* surface area.

Signal waterfall (most authoritative first):
  1. OSM native tags        — parking=multi-storey/underground/rooftop, building=*,
                              building:levels / parking:levels >= 2   (free; we
                              re-fetch the tags the main ETL discards)
  2. Building-footprint     — Overture buildings overlap ratio + height/floors
     overlap                  (the city-agnostic workhorse)
  3. Assessor tiebreaker    — parcel improvement value (ambiguous cases only;
                              optional, only where available)

Outputs per parking polygon:
  parking_type            surface | structure | ambiguous
  confidence              high | medium | low
  classification_source   osm_tag | building_overlap | assessor | none
  building_overlap_ratio  fraction of the polygon covered by building footprints
  surface_area_sqft       effective UNBUILT area (gross - building overlap)
  effective_surface_land_value
                          land_value_per_sqft * surface_area_sqft
                          (== assessed land value pro-rated by unbuilt share)

This is a READ-ONLY proof: it consumes the existing parking parquet + Overture
and writes a NEW classified parquet. It does not touch parking_lot_extraction.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
import shapely.wkb

# Overture prunes old releases from S3, so a stale pin makes the surface/structure
# classification fail outright ("No files found that match the pattern ...") and the
# parking export silently drops its classification columns. Re-pin when that happens;
# list live releases with:
#   curl -s "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/?list-type=2&prefix=release/&delimiter=/"
# Bumped 2026-08-20 from the pruned 2026-05-20.0 (only 2026-07-22.0 and 2026-08-19.0
# remained). Cities baked against an earlier release keep their existing parking
# metadata; only re-runs pick this up.
OVERTURE_RELEASE = "2026-08-19.0"
OVERTURE_BUILDINGS = (
    f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/"
    "theme=buildings/type=building/*"
)

# Downtown Houston validation bbox — contains both of Lars's ground-truth points:
#   808 San Jacinto (structure)  ~ 29.7588, -95.3658
#   surface lot across Walker    ~ 29.75684, -95.36194
VALIDATION_BBOX = (-95.372, 29.752, -95.356, 29.763)  # (minx, miny, maxx, maxy)

# Lars's audited ground truth (lon, lat). Expected verdicts in the labels.
GROUND_TRUTH = {
    "808 San Jacinto      (STRUCTURE)": (-95.36256, 29.75723),
    "across Walker        (SURFACE)  ": (-95.36194, 29.75684),
    "2-layer deck A       (STRUCTURE)": (-95.42720, 29.79996),
    "gas station B        (EXCLUDE)  ": (-95.42961, 29.80134),
    "#2 half-covered      (SURFACE)  ": (-95.38670, 29.73419),
    "#4 canopy lot        (SURFACE)  ": (-95.38158, 29.70696),
    "#5 lot around bldg   (SURFACE)  ": (-95.36694, 29.76861),
    "#6 canopy lot        (SURFACE)  ": (-95.37463, 29.78996),
    "#8 gas station       (EXCLUDE)  ": (-95.31584, 29.77359),
}

# Classification thresholds (tunable; validate before promoting to the pipeline).
# Re-based on REAL-building overlap (canopies excluded). The audit showed height
# does NOT separate canopies (gas-station roof = 7 m) from decks, and the assessor
# improvement value is too noisy to be a tiebreaker — so the middle band is left
# explicitly UNCERTAIN rather than forced either way.
OVERLAP_STRUCTURE = 0.85   # real-building coverage >= this -> structure (confident)
OVERLAP_UNCERTAIN = 0.40   # real-building coverage in [this, STRUCTURE) -> uncertain
OVERLAP_SURFACE = 0.15     # <= this -> surface (high conf); (this, UNCERTAIN) -> surface (carved)
STRUCTURE_CLASS_OVERLAP = 0.30  # Overture class=parking/garage covering >= this -> structure

# Overture building `class` buckets:
#  - parking structures: a direct, authoritative structure signal
#  - canopies: open-air covers (roof/carport) — parking exists UNDERNEATH, so they
#    are NOT real buildings: never carved, never counted toward structure.
STRUCTURE_CLASSES = {"parking", "garage"}
CANOPY_CLASSES = {"roof", "carport", "canopy"}
FUEL_NEAR_M = 25.0         # parking within this distance of an OSM amenity=fuel ...
FORECOURT_MAX_SQFT = 20_000   # ...AND no bigger than this -> excluded (real forecourts
                              # are small; a big lot merely near a station is a real lot)
DECK_MAX_SQFT = 80_000     # partial real-building overlap on a polygon larger than this
                           # is a surface lot AROUND a building, not a deck (decks are
                           # small footprints) -> surface+carve, not uncertain

# --- Parcel-context filter (optional; needs parcel land-use + improvement value) ---
# Even an NIR-trained model can't tell parking pavement from a bare/gravel lot or an
# open materials/rail yard — they're spectrally identical. Land use can: real
# off-street parking serves an *improved* use (buildings present on/under the lot's
# parcels), whereas bare lots, open storage yards, and utility/rail ROW do not. We
# emit a non-destructive `context_confidence` flag rather than dropping anything.
CONTEXT_IMPR_FRAC_MIN = 0.15    # <this fraction of the lot over built parcels -> low conf
CONTEXT_VACANT_FRAC_MAX = 0.50  # utility/vacant parcels need >= this overlap to stay high


def load_parking(path: Path, bbox: tuple | None) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    if bbox:
        gdf = gdf[gdf.intersects(box(*bbox))].copy()
    gdf = gdf.reset_index(drop=True)
    print(f"  Loaded {len(gdf):,} parking polygons"
          + (f" within validation bbox" if bbox else " (full city)"))
    return gdf


def _fetch_osm_tiled(bbox: tuple, tags: dict, label: str,
                     geom_types=("Polygon", "MultiPolygon"),
                     cell_deg: float = 0.12) -> gpd.GeoDataFrame:
    """Fetch OSM features over a tiled grid with per-cell retry.

    A single citywide Overpass query returns too large a response and the
    connection drops mid-stream; small cells are fast and reliable.
    """
    import math
    import time
    import osmnx as ox
    # 90s is ample for a healthy single-cell Overpass response (<1s typical); the
    # lower ceiling means a stale socket to a flapped backend fails fast instead of
    # burning a full 180s before we can drop it and reconnect.
    ox.settings.requests_timeout = 90
    # overpass-api.de's two backends flap independently — when concurrent
    # requests stack up, the load-balanced backend starts refusing TCP
    # connections (errno 61). Mitigations:
    #   1. Brief sleep between cells so we never have more than 1 request in
    #      flight from this script (typical extracted IPs do +/- 2 req/s).
    #   2. More attempts with linear backoff so a 1-2 minute backend wobble
    #      doesn't kill a cell.
    #   3. On connection-refused specifically, clear urllib3's connection pool
    #      so the next request opens a fresh socket (avoids reusing a stale
    #      TCP entry against an IP that just went unhealthy).

    def _flush_urllib3_pools():
        try:
            import requests as _r
            from urllib3 import poolmanager as _pm  # noqa: F401
            sess = getattr(ox._http, "_session", None)
            if sess is not None:
                sess.close()
            # Also clear default pools just in case
            for adp in ((sess.adapters.values() if sess else [])):
                try:
                    adp.poolmanager.clear()
                except Exception:
                    pass
        except Exception:
            pass

    minx, miny, maxx, maxy = bbox
    nx = max(1, math.ceil((maxx - minx) / cell_deg))
    ny = max(1, math.ceil((maxy - miny) / cell_deg))
    print(f"  Fetching OSM {label} over a {nx}x{ny} grid ({nx*ny} cells)...", flush=True)

    parts = []
    MAX_ATTEMPTS = 6
    # Early-exit: if overpass-api.de stays down for a long stretch, don't
    # waste 30-60 minutes burning through every remaining cell. Once we've
    # seen this many CONSECUTIVE cell failures, give up on the tiled fetch
    # for this layer and fall back to whatever cells we got (or to the
    # downstream Overture-only signal). The classifier handles empty OSM
    # gracefully — better degraded classification than no classification.
    MAX_CONSECUTIVE_FAILS = 8
    consecutive_fails = 0
    bailed = False
    for i in range(nx):
        if bailed: break
        for j in range(ny):
            cx0 = minx + i * (maxx - minx) / nx
            cx1 = minx + (i + 1) * (maxx - minx) / nx
            cy0 = miny + j * (maxy - miny) / ny
            cy1 = miny + (j + 1) * (maxy - miny) / ny
            cell = box(cx0, cy0, cx1, cy1)
            cell_ok = False
            for attempt in range(MAX_ATTEMPTS):
                try:
                    g = ox.features_from_polygon(cell, tags=tags)
                    if len(g):
                        parts.append(g)
                    cell_ok = True
                    break
                except Exception as e:
                    # Dump pooled connections on ANY failure (timeouts included) so
                    # the next attempt opens a fresh TCP socket and lets system DNS
                    # rotate to a healthy backend. A stale socket to a flapped
                    # Overpass backend otherwise makes every retry time out
                    # identically — observed hanging the fuel-station fetch for
                    # tens of minutes even though a fresh request succeeds in <1s.
                    _flush_urllib3_pools()
                    if attempt == MAX_ATTEMPTS - 1:
                        print(f"    cell ({i},{j}) failed after {MAX_ATTEMPTS} retries: {e}", flush=True)
                    else:
                        wait = min(60, 5 * (attempt + 1))  # 5,10,15,20,25,30
                        time.sleep(wait)
            if cell_ok:
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                    print(f"  Bailing on tiled fetch: {consecutive_fails} consecutive "
                          f"cell failures suggests overpass-api.de is down. "
                          f"Continuing with {len(parts)} partial cells.", flush=True)
                    bailed = True
                    break
            # Polite pause between cells regardless of success
            time.sleep(1.0)
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    gdf = pd.concat(parts)
    gdf = gdf[~gdf.index.duplicated(keep="first")]
    gdf = gdf[gdf.geometry.type.isin(list(geom_types))].to_crs(4326)
    return gdf


def fetch_osm_tags(bbox: tuple) -> gpd.GeoDataFrame:
    """Re-fetch OSM parking WITH the tags the main ETL drops (parking=multi-storey,
    building=parking, etc.). Only parking features — NOT building=True, which would
    pull every building in Houston."""
    gdf = _fetch_osm_tiled(bbox, {"amenity": "parking", "landuse": "parking", "parking": True},
                           "parking")
    if gdf.empty:
        print("  OSM: no parking features fetched.")
        return gdf
    keep = [c for c in ["parking", "building", "building:levels",
                        "parking:levels", "levels", "layer"] if c in gdf.columns]
    gdf = gpd.GeoDataFrame(gdf[["geometry"] + keep]).reset_index(drop=True)
    print(f"  OSM tagged parking polygons: {len(gdf):,}  (tag cols: {keep})")
    return gdf


def fetch_osm_fuel(bbox: tuple) -> gpd.GeoDataFrame:
    """Fetch OSM amenity=fuel (gas stations) to exclude fuel forecourts from parking."""
    gdf = _fetch_osm_tiled(bbox, {"amenity": "fuel"},
                           "fuel stations", geom_types=("Point", "Polygon", "MultiPolygon"))
    gdf = gpd.GeoDataFrame(gdf[["geometry"]]).reset_index(drop=True) if not gdf.empty else gdf
    print(f"  OSM fuel features: {len(gdf):,}")
    return gdf


def fetch_overture_buildings(bbox: tuple) -> gpd.GeoDataFrame:
    """Pull Overture building footprints (geometry + height + num_floors + class)."""
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    minx, miny, maxx, maxy = bbox
    rows = con.execute(
        f"""
        SELECT geometry AS wkb, height, num_floors, class, subtype
        FROM read_parquet('{OVERTURE_BUILDINGS}')
        WHERE bbox.xmin BETWEEN {minx} AND {maxx}
          AND bbox.ymin BETWEEN {miny} AND {maxy}
        """
    ).fetchall()
    geoms = [shapely.wkb.loads(bytes(r[0])) for r in rows]
    gdf = gpd.GeoDataFrame(
        {"height": [r[1] for r in rows], "num_floors": [r[2] for r in rows],
         "class": [r[3] for r in rows], "subtype": [r[4] for r in rows]},
        geometry=geoms, crs=4326,
    )
    # Bucket each building: parking structure / canopy / real building
    cls = gdf["class"].fillna("").str.lower()
    gdf["bcat"] = "real"
    gdf.loc[cls.isin(STRUCTURE_CLASSES), "bcat"] = "parking_struct"
    gdf.loc[cls.isin(CANOPY_CLASSES), "bcat"] = "canopy"
    print(f"  Overture buildings: {len(gdf):,}  "
          f"({gdf['height'].notna().sum():,} w/ height, "
          f"{(gdf.bcat=='parking_struct').sum():,} parking-class, "
          f"{(gdf.bcat=='canopy').sum():,} canopy-class)")
    return gdf


def attach_osm_tags(parking: gpd.GeoDataFrame, osm: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach OSM tags to each parking polygon via largest-overlap match."""
    if osm.empty:
        for c in ["parking", "building", "building:levels", "parking:levels"]:
            parking[c] = None
        return parking
    p = parking.to_crs(3857).reset_index(drop=True)
    o = osm.to_crs(3857).reset_index(drop=True)
    join = gpd.sjoin(p[["geometry"]], o, predicate="intersects", how="left")
    # for duplicate matches keep the first (sufficient for tag presence)
    join = join[~join.index.duplicated(keep="first")]
    for c in ["parking", "building", "building:levels", "parking:levels", "levels", "layer"]:
        parking[c] = join[c].values if c in join.columns else None
    return parking


def building_overlap(parking: gpd.GeoDataFrame, bldg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Per-polygon overlap, split by building category.

    Tracks separate coverage ratios for REAL buildings, parking-class structures,
    and canopies — because a canopy (parking underneath) must not be carved or
    counted as a structure, while an Overture class=parking footprint is a direct
    structure signal.
    """
    p = parking.to_crs(3857).reset_index(drop=True)
    p["_geom_area"] = p.geometry.area
    for col in ["real_overlap", "struct_overlap", "canopy_overlap"]:
        parking[col] = 0.0
    parking["building_overlap_ratio"] = 0.0  # real + struct (used for carve / structure)
    parking["ov_height"] = None
    parking["ov_floors"] = None
    if bldg.empty:
        return parking
    b = bldg.to_crs(3857).reset_index(drop=True)
    bsindex = b.sindex
    real_r, struct_r, canopy_r, heights, floors = [], [], [], [], []
    for geom, garea in zip(p.geometry, p["_geom_area"]):
        if garea <= 0:
            real_r.append(0.0); struct_r.append(0.0); canopy_r.append(0.0)
            heights.append(None); floors.append(None); continue
        idx = list(bsindex.query(geom, predicate="intersects"))
        if not idx:
            real_r.append(0.0); struct_r.append(0.0); canopy_r.append(0.0)
            heights.append(None); floors.append(None); continue
        sub = b.iloc[idx]
        inter = sub.geometry.intersection(geom)
        a = inter.area
        cat = sub["bcat"].values
        real_a = a[cat == "real"].sum()
        struct_a = a[cat == "parking_struct"].sum()
        canopy_a = a[cat == "canopy"].sum()
        real_r.append(min(1.0, real_a / garea))
        struct_r.append(min(1.0, struct_a / garea))
        canopy_r.append(min(1.0, canopy_a / garea))
        # height/floors of the largest non-canopy building over this polygon
        noncanopy = (cat != "canopy")
        if noncanopy.any():
            i = a[noncanopy].values.argmax()
            top = sub[noncanopy].iloc[i]
            heights.append(top["height"]); floors.append(top["num_floors"])
        else:
            heights.append(None); floors.append(None)
    parking["real_overlap"] = real_r
    parking["struct_overlap"] = struct_r
    parking["canopy_overlap"] = canopy_r
    parking["building_overlap_ratio"] = [min(1.0, r + s) for r, s in zip(real_r, struct_r)]
    parking["ov_height"] = heights
    parking["ov_floors"] = floors
    return parking


def flag_fuel(parking: gpd.GeoDataFrame, fuel: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Flag parking polygons within FUEL_NEAR_M of an OSM amenity=fuel feature."""
    parking["near_fuel"] = False
    if fuel is None or fuel.empty:
        return parking
    p = parking.to_crs(3857).reset_index(drop=True)
    f = fuel.to_crs(3857)
    fbuf = f.geometry.buffer(FUEL_NEAR_M).union_all()
    parking["near_fuel"] = p.geometry.intersects(fbuf).values
    return parking


def _truthy_levels(v) -> float | None:
    try:
        return float(str(v).split(";")[0])
    except (TypeError, ValueError):
        return None


def classify(row) -> tuple[str, str, str]:
    """Return (parking_type, confidence, source). Buckets: surface | structure |
    uncertain | excluded. Based on REAL-building overlap (canopies excluded)."""
    area = row.get("parking_area_sqft") or 0.0
    # --- Gas stations: a forecourt-sized polygon next to a fuel pump. A large lot
    # merely near a station is a real lot, so size-gate the exclusion. ---
    if row.get("near_fuel") and area <= FORECOURT_MAX_SQFT:
        return "excluded", "high", "osm_fuel"

    # --- Tier 1: OSM explicit structure tags ---
    parking_tag = str(row.get("parking") or "").lower()
    if parking_tag in ("multi-storey", "multistorey", "underground", "rooftop"):
        return "structure", "high", "osm_tag"
    bldg_tag = str(row.get("building") or "").lower()
    if bldg_tag in ("parking", "garage", "garages"):
        return "structure", "high", "osm_tag"
    lv = _truthy_levels(row.get("building:levels")) or _truthy_levels(row.get("parking:levels"))
    if lv is not None and lv >= 2:
        return "structure", "high", "osm_tag"

    # --- Tier 1.5: Overture class=parking/garage covering the lot ---
    if (row.get("struct_overlap") or 0) >= STRUCTURE_CLASS_OVERLAP:
        return "structure", "high", "overture_class"

    # --- Tier 2: real-building footprint coverage ---
    real = row.get("real_overlap") or 0.0
    if real >= OVERLAP_STRUCTURE:
        return "structure", "high", "building_overlap"
    if real >= OVERLAP_UNCERTAIN:
        # partial real-building coverage. On a deck-sized footprint this is genuinely
        # unresolvable (deck w/ geometry mismatch vs surface lot w/ a building) -> flag.
        # On a LARGE footprint it can only be a surface lot around a building -> carve.
        if area > DECK_MAX_SQFT:
            return "surface", "medium", "building_overlap"
        return "uncertain", "low", "building_overlap"
    # explicit OSM surface tag, or low/no real coverage -> surface (carved)
    if parking_tag == "surface" or real <= OVERLAP_SURFACE:
        conf = "high" if (parking_tag == "surface" or real < 0.05) else "medium"
        src = "osm_tag" if parking_tag == "surface" else "building_overlap"
        return "surface", conf, src
    return "surface", "medium", "building_overlap"


def parcel_context(parking: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Flag detections whose parcel land use cannot plausibly generate surface parking.

    Adds (non-destructively — parking_type / areas are untouched):
      frac_over_improved   fraction of the lot's area over parcels that have buildings
      context_category     dominant parcel land-use category
      context_confidence   high | low | unknown
      context_reason       why it was flagged low (else "")

    `low` means the pixels look like parking but the land underneath is bare/vacant,
    utility/rail ROW, or an open yard — the false-positive class no imagery model
    (RGB or NIR) can rule out. No-op (`unknown`) when the parcel layer lacks the
    improvement-value / land-use columns (e.g. cities without those attributes).
    """
    broad = "property_land_use_category"
    refined = "property_land_use_refined"
    if "improvement_value" not in parcels.columns or broad not in parcels.columns:
        out = parking.copy()
        out["frac_over_improved"] = np.nan
        out["context_category"] = None
        out["context_confidence"] = "unknown"
        out["context_reason"] = "no parcel land-use/improvement data"
        return out

    area_crs = parcels.estimate_utm_crs()
    pk = parking.to_crs(area_crs).reset_index(drop=True)
    pk["_pid"] = pk.index
    keep = ["geometry", "improvement_value", broad]
    if refined in parcels.columns:
        keep.append(refined)
    pc = parcels.to_crs(area_crs)[keep].copy()

    ov = gpd.overlay(pk[["_pid", "geometry"]], pc, how="intersection", keep_geom_type=True)
    ov["_isect"] = ov.geometry.area
    ov["_impr"] = (ov["improvement_value"].fillna(0) > 0).astype(float)
    grp = ov.groupby("_pid")
    frac = grp.apply(lambda d: (d["_impr"] * d["_isect"]).sum() / max(d["_isect"].sum(), 1e-9),
                     include_groups=False)
    dom = ov.sort_values("_isect").groupby("_pid").tail(1).set_index("_pid")

    out = parking.reset_index(drop=True).copy()
    # frac/dom are indexed by _pid (== out's positional index); reindex to align.
    out["frac_over_improved"] = frac.reindex(out.index).astype(float).fillna(0.0).values
    out["context_category"] = dom[broad].reindex(out.index).values
    refs = (dom[refined].reindex(out.index).astype("object").values
            if refined in parcels.columns else [None] * len(out))
    imprs = dom["improvement_value"].reindex(out.index).fillna(0.0).values

    conf, reason = [], []
    for f, cat, ref, impr in zip(out["frac_over_improved"].values,
                                 out["context_category"].astype("object").values,
                                 refs, imprs):
        cat = str(cat or "")
        if cat == "Utility" and f < CONTEXT_VACANT_FRAC_MAX:
            conf.append("low"); reason.append("utility/rail ROW, no buildings")
        elif f < CONTEXT_IMPR_FRAC_MIN:
            conf.append("low"); reason.append("barely overlaps any built parcel (bare lot/yard)")
        elif str(ref) == "Vacant" and impr == 0 and f < CONTEXT_VACANT_FRAC_MAX:
            conf.append("low"); reason.append("vacant parcel, no improvement")
        else:
            conf.append("high"); reason.append("")
    out["context_confidence"] = conf
    out["context_reason"] = reason
    return out


def classify_parking(parking: gpd.GeoDataFrame, bbox: tuple,
                     parcels: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """Classify each parking polygon surface | structure | uncertain | excluded and
    pro-rate land value to the effective unbuilt surface.

    Adds columns: parking_type, confidence, classification_source, real_overlap,
    struct_overlap, canopy_overlap, building_overlap_ratio, near_fuel, ov_height,
    ov_floors, surface_area_sqft, effective_surface_land_value.

    Reusable by the main ETL pipeline. Requires `parking_area_sqft` and (for value)
    `land_value_per_sqft` columns. Robust to missing OSM tags (e.g. ML-derived
    parking): falls back to footprint signals.
    """
    osm = fetch_osm_tags(bbox)
    fuel = fetch_osm_fuel(bbox)
    bldg = fetch_overture_buildings(bbox)
    parking = attach_osm_tags(parking, osm)
    parking = building_overlap(parking, bldg)
    parking = flag_fuel(parking, fuel)

    cls = parking.apply(classify, axis=1, result_type="expand")
    parking["parking_type"] = cls[0]
    parking["confidence"] = cls[1]
    parking["classification_source"] = cls[2]
    # Effective unbuilt surface area: carve out only REAL buildings (canopies leave
    # the parking intact underneath, so canopy_overlap is NOT subtracted).
    parking["surface_area_sqft"] = (
        parking["parking_area_sqft"] * (1.0 - parking["real_overlap"].clip(0, 1))
    )
    parking.loc[parking.parking_type.isin(["structure", "excluded"]), "surface_area_sqft"] = 0.0
    lvpsf = parking.get("land_value_per_sqft")
    if lvpsf is not None:
        parking["effective_surface_land_value"] = lvpsf.fillna(0) * parking["surface_area_sqft"]
    else:
        parking["effective_surface_land_value"] = None

    # Parcel-context flag: down-rank lots whose land use can't generate parking
    # (bare/vacant lots, open materials yards, utility/rail ROW). Non-destructive.
    if parcels is not None:
        parking = parcel_context(parking, parcels)
    return parking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parking", default="data/parking/houston/houston-tx-parking-lots.parquet")
    ap.add_argument("--out", default="data/parking/houston/houston-tx-parking-classified.parquet")
    ap.add_argument("--full-city", action="store_true",
                    help="Run on the whole city bbox instead of the downtown validation bbox.")
    args = ap.parse_args()

    parking_path = Path(args.parking)
    print("=" * 60); print("Step 1: Load parking polygons"); print("=" * 60)
    parking_all = load_parking(parking_path, None)
    total_bounds = tuple(parking_all.total_bounds)
    bbox = total_bounds if args.full_city else VALIDATION_BBOX
    parking = load_parking(parking_path, bbox)

    print("=" * 60); print("Steps 2-5: Classify"); print("=" * 60)
    parking = classify_parking(parking, bbox)

    # --- Report ---
    print("\nClass distribution:")
    print(parking["parking_type"].value_counts().to_string())
    print("\nBy confidence:")
    print(parking.groupby(["parking_type", "confidence"]).size().to_string())

    lv = parking["effective_surface_land_value"].fillna(0)
    conf_surf = parking.parking_type == "surface"
    uncertain = parking.parking_type == "uncertain"
    gross = (parking["land_value_per_sqft"].fillna(0) * parking["parking_area_sqft"]).sum()
    print(f"\nLand value:")
    print(f"  gross (OSM as-is):              ${gross:,.0f}")
    print(f"  confident surface (headline):   ${lv[conf_surf].sum():,.0f}")
    print(f"  uncertain (reported separately):${lv[uncertain].sum():,.0f}")

    print("\n" + "=" * 60); print("Ground-truth validation"); print("=" * 60)
    from shapely.geometry import Point
    for label, (lon, lat) in GROUND_TRUTH.items():
        hit = parking[parking.contains(Point(lon, lat))]
        if hit.empty:
            print(f"  {label}: NO polygon contains this point")
            continue
        r = hit.iloc[0]
        print(f"  {label}: -> {r['parking_type']} ({r['confidence']}, {r['classification_source']})"
              f"  real_ov={r['real_overlap']:.2f} struct_ov={r['struct_overlap']:.2f}"
              f" canopy_ov={r['canopy_overlap']:.2f} h={r.get('ov_height')}"
              f" tag={r.get('parking')!r} fuel={r.get('near_fuel')}"
              f"  surf={r['surface_area_sqft']:,.0f}sqft ${r['effective_surface_land_value'] or 0:,.0f}")

    out = Path(args.out)
    parking.to_parquet(out)
    print(f"\nWrote {len(parking):,} classified polygons -> {out}")


if __name__ == "__main__":
    main()
