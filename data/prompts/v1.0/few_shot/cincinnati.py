"""
Cincinnati, OH — ETL reference script.
Pattern: County-wide ArcGIS FeatureServer → spatial clip to city boundary via osmnx.
Demonstrates: county-clip, CLASS-code categorization, GRPPCLID condo collapse.
"""
import os, sys, glob, numpy as np, pandas as pd, geopandas as gpd
from datetime import datetime
from shapely.ops import unary_union

sys.path.append("..")
from parcel_calculations import add_improvement_ratio_fields
from cloud_utils import get_feature_data_with_geometry

DATA_DIR = "data/cincinnati"
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://services.arcgis.com/JyZag7oO4NteHGiq/ArcGIS/rest/services"
LAYER_ID = 12   # Hamilton_County_Parcel_Polygons

# ── 1. Load raw parcel data ────────────────────────────────────────────────────
SCRAPE_DATA = int(os.getenv("SCRAPE_DATA", "0"))
if SCRAPE_DATA:
    parcel_gdf = get_feature_data_with_geometry("CAGIS_Open_Data", BASE_URL, LAYER_ID)
    today_str = datetime.now().strftime("%Y_%m_%d")
    raw_path = os.path.join(DATA_DIR, f"cincinnati_parcels_{today_str}.parquet")
    parcel_gdf.to_parquet(raw_path, index=False)
    print(f"✅ Scraped and saved: {raw_path}")
else:
    files = sorted(
        glob.glob(os.path.join(DATA_DIR, "cincinnati_parcels_*.parquet")),
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No raw parcel files in {DATA_DIR}. Set SCRAPE_DATA=1.")
    parcel_gdf = gpd.read_parquet(files[0])
    print(f"✅ Loaded: {files[0]}  rows={len(parcel_gdf):,}")

if parcel_gdf.crs is None:
    parcel_gdf = parcel_gdf.set_crs("EPSG:4326")

# ── 2. Spatial clip to Cincinnati city limits ──────────────────────────────────
import osmnx as ox
boundary_gdf = ox.geocode_to_gdf("Cincinnati, Ohio, USA")
if boundary_gdf.crs != parcel_gdf.crs:
    boundary_gdf = boundary_gdf.to_crs(parcel_gdf.crs)
city_geom = boundary_gdf.geometry.iloc[0]

valid_mask = parcel_gdf["geometry"].notnull() & parcel_gdf["geometry"].is_valid
inside_mask = valid_mask & parcel_gdf["geometry"].within(city_geom)
parcel_gdf = parcel_gdf[inside_mask].copy()
print(f"✅ After city clip: {len(parcel_gdf):,} parcels")

# ── 3. Parcel link ─────────────────────────────────────────────────────────────
parcel_gdf["link"] = (
    "https://wedge.hcauditor.org/view/re/" + parcel_gdf["AUDPCLID"].astype(str) + "/2025/summary"
)

# ── 4. Condo collapse (aggregate on GRPPCLID) ──────────────────────────────────
numeric_sum_cols = [c for c in parcel_gdf.columns if np.issubdtype(parcel_gdf[c].dtype, np.number)]
categorical_cols = [c for c in parcel_gdf.columns if c not in numeric_sum_cols + ["geometry", "GRPPCLID"]]

agg = {c: "sum" for c in numeric_sum_cols}
agg.update({c: "first" for c in categorical_cols})
collapsed = parcel_gdf.groupby("GRPPCLID", dropna=False).agg(agg).reset_index()

geom_union = parcel_gdf.groupby("GRPPCLID", dropna=False)["geometry"].apply(
    lambda gs: unary_union([g for g in gs if g is not None])
)
collapsed["geometry"] = geom_union.values
parcel_gdf = gpd.GeoDataFrame(collapsed, geometry="geometry", crs="EPSG:4326")
print(f"✅ After condo collapse: {len(parcel_gdf):,} parcels")

# ── 5. Categorise ──────────────────────────────────────────────────────────────
CLASS_MAP = {
    # Residential
    "510": "Single Family Residential",
    "517": "Multi-Family Residential",   # 2-family
    "520": "Multi-Family Residential",   # 3-family
    "530": "Condo / PUD", "550": "Condo / PUD", "551": "Condo / PUD",
    "552": "Condo / PUD", "553": "Condo / PUD", "554": "Condo / PUD",
    "555": "Condo / PUD", "556": "Condo / PUD", "558": "Condo / PUD",
    "401": "Multi-Family Residential",   # 4-19 units
    "402": "Multi-Family Residential",   # 20-39 units
    "403": "Multi-Family Residential",   # 40+ units
    "500": "Vacant", "501": "Vacant", "502": "Vacant",
    "503": "Vacant", "504": "Vacant", "505": "Vacant",
    # Commercial
    "420": "Commercial", "421": "Commercial", "422": "Commercial",
    "430": "Commercial", "431": "Commercial", "432": "Commercial",
    "440": "Commercial", "441": "Commercial", "442": "Commercial",
    "455": "Parking Lot", "456": "Parking Lot",
    # Industrial
    "480": "Industrial", "482": "Industrial", "488": "Industrial",
    # Exempt / civic
    "100": "Institutional / Civic", "110": "Institutional / Civic",
    "200": "Institutional / Civic", "300": "Institutional / Civic",
}

def classify(row):
    code = str(int(float(row["CLASS"]))) if pd.notna(row["CLASS"]) else ""
    return CLASS_MAP.get(code, "Other")

parcel_gdf["PROPERTY_CATEGORY"] = parcel_gdf.apply(classify, axis=1)
print("\nPROPERTY_CATEGORY counts:")
print(parcel_gdf["PROPERTY_CATEGORY"].value_counts(dropna=False).to_string())

# ── 6. Exempt parcels ──────────────────────────────────────────────────────────
parcel_gdf["exemption_flag"] = (parcel_gdf["PROPERTY_CATEGORY"] == "Publicly Owned").astype(int)
# Also flag via class codes 100/200/300
pub_codes = {"100", "110", "200", "300"}
class_str = parcel_gdf["CLASS"].astype(str).apply(
    lambda x: str(int(float(x))) if x.replace(".", "").isdigit() else x
)
parcel_gdf["exemption_flag"] = (parcel_gdf["exemption_flag"] | class_str.isin(pub_codes)).astype(int)

before = len(parcel_gdf)
parcel_gdf = parcel_gdf[parcel_gdf["exemption_flag"] == 0].copy()
print(f"\n✅ Removed {before - len(parcel_gdf):,} exempt parcels → {len(parcel_gdf):,} remaining")

# ── 7. Refined categories ──────────────────────────────────────────────────────
def refine(row):
    cat = str(row["PROPERTY_CATEGORY"])
    if "Vacant" in cat:     return "Vacant"
    if "Parking" in cat:    return "Parking Lot"
    impr = float(row.get("improvement_value") or 0)
    land = float(row.get("land_value") or 0)
    total = land + impr
    if total > 0 and impr < 0.5 * total:
        return "Underdeveloped"
    return None

# ── 8. Canonical fields ────────────────────────────────────────────────────────
parcel_gdf["land_value"]        = pd.to_numeric(parcel_gdf.get("LNDVAL"), errors="coerce")
parcel_gdf["improvement_value"] = pd.to_numeric(parcel_gdf.get("BLDVAL"), errors="coerce")
parcel_gdf["full_market_value"] = parcel_gdf["land_value"].fillna(0) + parcel_gdf["improvement_value"].fillna(0)
parcel_gdf["property_land_use_category"] = parcel_gdf["PROPERTY_CATEGORY"]
parcel_gdf["property_land_use_refined"]  = parcel_gdf.apply(refine, axis=1)

from pyproj import Geod
geod = Geod(ellps="WGS84")
def geodesic_area_sqft(geom):
    if geom is None or geom.is_empty: return np.nan
    if geom.geom_type == "Polygon":
        lon, lat = geom.exterior.coords.xy
        area_m2, _ = geod.polygon_area_perimeter(lon, lat)
        return abs(area_m2) * 10.763910416709722
    if geom.geom_type == "MultiPolygon":
        return sum(geodesic_area_sqft(p) for p in geom.geoms)
    return np.nan

parcel_gdf["area_sqft"] = parcel_gdf["geometry"].apply(geodesic_area_sqft)
parcel_gdf["land_value_per_sqft"]        = parcel_gdf["land_value"]        / parcel_gdf["area_sqft"]
parcel_gdf["improvement_value_per_sqft"] = parcel_gdf["improvement_value"] / parcel_gdf["area_sqft"]
parcel_gdf["full_market_value_per_sqft"] = parcel_gdf["full_market_value"] / parcel_gdf["area_sqft"]

parcel_gdf = add_improvement_ratio_fields(parcel_gdf, land_col="land_value", improvement_col="improvement_value")

# ── 9. Export ──────────────────────────────────────────────────────────────────
EXPORT_COLS = [
    "geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
    "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
    "improvement_value", "improvement_value_per_sqft",
    "TLLDIMPROV", "IMPR_LAND_RATIO", "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link",
]
for col in EXPORT_COLS:
    if col not in parcel_gdf.columns:
        parcel_gdf[col] = np.nan

export = parcel_gdf[EXPORT_COLS].rename(columns={"land_value": "current_full_land_value"})
export["geometry"] = export["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
export = gpd.GeoDataFrame(export, geometry="geometry", crs="EPSG:4326")

out = os.path.join(DATA_DIR, "cincinnati-oh-parcels.parquet")
export.to_parquet(out, index=False)
print(f"\n✅ Saved: {out}  rows={len(export):,}")
print("\nRefined category counts:")
print(export["property_land_use_refined"].value_counts(dropna=False).to_string())
