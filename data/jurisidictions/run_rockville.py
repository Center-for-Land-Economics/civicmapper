"""
Run Rockville (MD) parcel ETL pipeline end-to-end.

Source data (all from Maryland iMAP, statewide services):
  - Parcels  : PlanningCadastre/MD_ParcelBoundaries/MapServer/0
               Statewide parcel polygons attributed with SDAT assessment data
               (land/improvement/total value, land use, exemption class, owner).
  - Boundary : Boundaries/MD_PoliticalBoundaries/FeatureServer/5
               "Municipal Boundaries - Detailed" — authoritative, state-maintained
               from each city's own GIS. Rockville is stored as ~19 annexation
               polygons (MUN_NAME='ROCKVILLE', JURSCODE='MONT') that we union.

Rockville's own ArcGIS server (maps.rockvillemd.gov) is not reachable from CI/dev
networks, so the state municipal layer is used for the city limits instead.

Because JURSCODE only isolates Montgomery County, parcels are downloaded by the
Rockville bounding box and then spatially clipped to the exact municipal polygon.

Local-only: this script writes parquet to disk. It does NOT upload to Azure.

Usage:
    python data/jurisidictions/run_rockville.py            # use cached raw scrape if present
    python data/jurisidictions/run_rockville.py --scrape   # force a fresh download
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.ops import unary_union
from pyproj import Geod

# parcel_calculations.py lives in data/ (one level up from this script)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from parcel_calculations import add_improvement_ratio_fields  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "rockville")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Endpoints ────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}
TIMEOUT = 120
PARCELS_URL = (
    "https://mdgeodata.md.gov/imap/rest/services/PlanningCadastre/"
    "MD_ParcelBoundaries/MapServer/0/query"
)
MUNI_URL = (
    "https://mdgeodata.md.gov/imap/rest/services/Boundaries/"
    "MD_PoliticalBoundaries/FeatureServer/5/query"
)
PAGE = 1000  # MD_ParcelBoundaries maxRecordCount


# ── 1. Fetch + union the Rockville municipal boundary ───────────────────────────
def fetch_rockville_boundary():
    print("📍 Fetching Rockville municipal boundary (MD_PoliticalBoundaries/5)...")
    params = {
        "f": "geojson", "where": "MUN_NAME='ROCKVILLE'",
        "outFields": "MUN_NAME", "returnGeometry": "true", "outSR": 4326,
    }
    r = requests.get(MUNI_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    gj = r.json()
    parts = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    boundary = unary_union(parts.geometry.values)
    print(f"   ✅ Unioned {len(parts)} annexation polygons → {boundary.geom_type}")
    print(f"   bounds = {[round(v, 5) for v in boundary.bounds]}")
    return boundary


# ── 2. Download parcels within the boundary bbox (paginated geojson) ────────────
def fetch_parcels_bbox(bounds):
    minx, miny, maxx, maxy = bounds
    env = json.dumps(
        {"xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
         "spatialReference": {"wkid": 4326}}
    )
    feats, offset = [], 0
    while True:
        params = {
            "f": "geojson", "where": "1=1", "outFields": "*", "returnGeometry": "true",
            "geometry": env, "geometryType": "esriGeometryEnvelope", "inSR": 4326,
            "outSR": 4326, "spatialRel": "esriSpatialRelIntersects",
            "resultOffset": offset, "resultRecordCount": PAGE,
        }
        r = requests.get(PARCELS_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        page_feats = r.json().get("features", [])
        if not page_feats:
            break
        feats.extend(page_feats)
        print(f"   fetched {len(feats):,} parcels...")
        if len(page_feats) < PAGE:
            break
        offset += len(page_feats)
        time.sleep(0.3)
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    return gdf


def load_raw(force_scrape: bool) -> gpd.GeoDataFrame:
    cached = sorted(glob.glob(os.path.join(DATA_DIR, "rockville_raw_*.parquet")), reverse=True)
    if cached and not force_scrape:
        print(f"📂 Loading cached raw scrape: {cached[0]}")
        return gpd.read_parquet(cached[0])

    boundary = fetch_rockville_boundary()
    print("⬇️  Downloading parcels in Rockville bbox (MD_ParcelBoundaries/0)...")
    gdf = fetch_parcels_bbox(boundary.bounds)
    print(f"   ✅ Downloaded {len(gdf):,} parcels (pre-clip)")

    # Clip to the exact municipal polygon (not just the bbox).
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:4326")
    clipped = gpd.sjoin(gdf, boundary_gdf, how="inner", predicate="intersects")
    clipped = clipped.drop(columns=[c for c in clipped.columns if c.startswith("index_")])
    print(f"   ✅ {len(clipped):,} parcels intersect the Rockville boundary")

    today = datetime.now().strftime("%Y_%m_%d")
    raw_path = os.path.join(DATA_DIR, f"rockville_raw_{today}.parquet")
    clipped.to_parquet(raw_path, index=False)
    print(f"   ✅ Cached raw → {raw_path}")
    return clipped


# ── 3. Classification ───────────────────────────────────────────────────────────
def categorize_property_type(row):
    """Bucket Maryland DESCLU land-use descriptions into display categories."""
    d = str(row.get("DESCLU") or "").strip().upper()
    if not d or d == "NAN":
        return "Unknown"
    if "EXEMPT" in d:
        return "Exempt / Governmental"
    if "VACANT" in d:
        return "Vacant Land"
    if "AGRICULT" in d:
        return "Agricultural"
    if "APARTMENT" in d:
        return "Apartments"
    if "TOWN" in d and ("HOUSE" in d or "HSE" in d):
        return "Town House"
    if "CONDO" in d:
        return "Condominium"
    if "INDUSTRIAL" in d:
        return "Industrial"
    if "COMMERCIAL" in d:
        return "Commercial"
    if "RESIDENTIAL" in d:
        return "Residential"
    if "MARSH" in d or "WETLAND" in d:
        return "Marsh / Wetland"
    return "Other"


def categorize_property_refined(row):
    cat = str(row["PROPERTY_CATEGORY"])
    desc = str(row.get("DESCLU") or "").upper()
    lv = row["land_value"] if pd.notna(row["land_value"]) else 0
    iv = row["improvement_value"] if pd.notna(row["improvement_value"]) else 0
    total = lv + iv
    if "VACANT" in desc or "Vacant" in cat:
        return "Vacant"
    if "PARKING" in desc:
        return "Parking Lot"
    if total > 0 and iv == 0:
        return "Vacant"  # land value but no structure
    if total > 0 and iv < 0.5 * total:
        return "Underdeveloped"
    return None


# ── 4. Geodesic area helper ─────────────────────────────────────────────────────
_GEOD = Geod(ellps="WGS84")


def geodesic_area_sqft(geom):
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type == "Polygon":
        lon, lat = geom.exterior.coords.xy
        area_m2, _ = _GEOD.polygon_area_perimeter(lon, lat)
        return abs(area_m2) * 10.763910416709722
    if geom.geom_type == "MultiPolygon":
        return sum(geodesic_area_sqft(p) for p in geom.geoms)
    return np.nan


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrape", action="store_true", help="Force a fresh download")
    args = ap.parse_args()

    gdf = load_raw(args.scrape)
    today = datetime.now().strftime("%Y_%m_%d")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print(f"\n✅ Working set: {len(gdf):,} parcels | CRS={gdf.crs}")
    print("\nDESCLU value counts (top 30):")
    print(gdf["DESCLU"].value_counts(dropna=False).head(30).to_string())
    print("\nLU value counts (top 30):")
    print(gdf["LU"].value_counts(dropna=False).head(30).to_string())
    print("\nEXCLASS / DESCEXCL counts:")
    print(gdf["DESCEXCL"].value_counts(dropna=False).head(20).to_string())

    # ── Drop polygons with no joined assessment (ROW / common-area / unmatched) ──
    has_assess = gdf["DESCLU"].notna() | gdf["NFMTTLVL"].notna() | gdf["LU"].notna()
    dropped = (~has_assess).sum()
    gdf = gdf[has_assess].copy()
    print(f"\n🧹 Dropped {dropped:,} parcels with no assessment data → {len(gdf):,} remain")

    # ── Condo cure: collapse duplicate ACCTID ───────────────────────────────────
    if "ACCTID" in gdf.columns:
        n_dupes = gdf.duplicated(subset=["ACCTID"], keep=False).sum()
        print(f"Duplicate rows by ACCTID: {n_dupes:,}")
        if n_dupes > 0:
            num_sum = [c for c in ["NFMLNDVL", "NFMIMPVL", "NFMTTLVL", "LANDAREA", "ACRES"]
                       if c in gdf.columns and np.issubdtype(gdf[c].dtype, np.number)]
            cat_cols = [c for c in gdf.columns if c not in set(num_sum + ["geometry", "ACCTID"])]
            agg = {c: "sum" for c in num_sum}
            agg.update({c: "first" for c in cat_cols})
            collapsed = gdf.groupby("ACCTID", dropna=False).agg(agg).reset_index()
            geom_union = gdf.groupby("ACCTID", dropna=False)["geometry"].apply(
                lambda gs: unary_union([g for g in gs if g is not None])
                if any(g is not None for g in gs) else None
            )
            collapsed["geometry"] = geom_union.values
            gdf = gpd.GeoDataFrame(collapsed, geometry="geometry", crs=gdf.crs)
            print(f"   ✅ Rows after condo-cure collapse: {len(gdf):,}")

    # ── Categorize ──────────────────────────────────────────────────────────────
    gdf["PROPERTY_CATEGORY"] = gdf.apply(categorize_property_type, axis=1)
    print("\nPROPERTY_CATEGORY counts:")
    print(gdf["PROPERTY_CATEGORY"].value_counts(dropna=False).to_string())

    # ── Exemption flag + filter ─────────────────────────────────────────────────
    desclu = gdf["DESCLU"].fillna("").astype(str).str.upper()
    exclass = gdf["EXCLASS"].fillna("").astype(str).str.strip() if "EXCLASS" in gdf.columns else pd.Series("", index=gdf.index)
    exempt_by_cat = desclu.str.contains("EXEMPT", na=False)
    exempt_by_class = exclass.ne("") & exclass.str.upper().ne("NAN")
    gdf["exemption_flag"] = (exempt_by_cat | exempt_by_class).astype(int)
    before = len(gdf)
    gdf = gdf[gdf["exemption_flag"] == 0].copy()
    print(f"\n✅ Removed {before - len(gdf):,} exempt parcels → {len(gdf):,} remaining")

    # ── Canonical value fields ──────────────────────────────────────────────────
    gdf["land_value"] = pd.to_numeric(gdf.get("NFMLNDVL"), errors="coerce")
    gdf["improvement_value"] = pd.to_numeric(gdf.get("NFMIMPVL"), errors="coerce")
    total = pd.to_numeric(gdf.get("NFMTTLVL"), errors="coerce")
    gdf["full_market_value"] = total.fillna(gdf["land_value"].fillna(0) + gdf["improvement_value"].fillna(0))
    gdf["property_land_use_category"] = gdf["PROPERTY_CATEGORY"]
    gdf["property_land_use_refined"] = gdf.apply(categorize_property_refined, axis=1)

    # ── Geometry validity + area + per-sqft ─────────────────────────────────────
    gdf["geometry"] = gdf["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
    print("Computing geodesic areas...")
    gdf["area_sqft"] = gdf["geometry"].apply(geodesic_area_sqft)
    gdf.loc[gdf["area_sqft"] < 1, "area_sqft"] = np.nan
    gdf["full_market_value_per_sqft"] = gdf["full_market_value"] / gdf["area_sqft"]
    gdf["land_value_per_sqft"] = gdf["land_value"] / gdf["area_sqft"]
    gdf["improvement_value_per_sqft"] = gdf["improvement_value"] / gdf["area_sqft"]

    gdf = add_improvement_ratio_fields(gdf, land_col="land_value", improvement_col="improvement_value")

    # ── Link ────────────────────────────────────────────────────────────────────
    if "SDATWEBADR" in gdf.columns:
        gdf["link"] = gdf["SDATWEBADR"].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    else:
        gdf["link"] = np.nan

    # ── Select + export ─────────────────────────────────────────────────────────
    columns_to_export = [
        "geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
        "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
        "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
        "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link",
    ]
    for col in columns_to_export:
        if col not in gdf.columns:
            gdf[col] = np.nan

    export = gdf[columns_to_export].rename(columns={"land_value": "current_full_land_value"})
    export["geometry"] = export["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
    export = gpd.GeoDataFrame(export, geometry="geometry", crs=gdf.crs)
    export = export[export.geometry.notna() & ~export.geometry.is_empty].copy()
    if export.crs is None or export.crs.to_epsg() != 4326:
        export = export.to_crs("EPSG:4326")

    canonical = os.path.join(DATA_DIR, "rockville-md-parcels.parquet")
    dated = os.path.join(DATA_DIR, f"rockville-md-parcels_{today}.parquet")
    export.to_parquet(canonical, index=False)
    export.to_parquet(dated, index=False)

    print(f"\n✅ Saved canonical parquet: {canonical}")
    print(f"✅ Saved dated parquet:     {dated}")
    print(f"   Total rows: {len(export):,}")
    print("\nRefined category counts:")
    print(export["property_land_use_refined"].value_counts(dropna=False).to_string())
    print("\nLand value summary:")
    print(export["current_full_land_value"].describe().to_string())


if __name__ == "__main__":
    main()
