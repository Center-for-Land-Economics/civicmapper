"""
Run Baltimore parcel ETL pipeline end-to-end.
Equivalent to running baltimore.ipynb with SCRAPE_DATA=1 and upload_dev=True.
"""
import os, glob, sys, numpy as np, pandas as pd, geopandas as gpd
from datetime import datetime
from shapely.ops import unary_union
from pyproj import Geod

sys.path.append("..")
from parcel_calculations import add_improvement_ratio_fields
from cloud_utils import get_feature_data_with_geometry, ensure_geodataframe

DATA_DIR = "data/baltimore"
os.makedirs(DATA_DIR, exist_ok=True)

GOVERNMENT_OWNER_PATTERNS = [
    "MAYOR AND CITY COUNCIL",
    "STATE OF MARYLAND",
    "UNITED STATES OF AMERICA",
    "UNITED STATES",
    "US GOVERNMENT",
    "HOUSING AUTHORITY",
    "BALTIMORE CITY BOARD SCHOOL",
    "DEPARTMENT OF",
    "DEPT OF",
]

# ── 1. Load cached raw parquet (skip re-scrape) ────────────────────────────────
today_str = datetime.now().strftime("%Y_%m_%d")
cached_raw = os.path.join(DATA_DIR, "baltimore_parcels_2026_03_13.parquet")
print(f"📂 Loading cached raw parquet: {cached_raw}")
parcel_gdf = gpd.read_parquet(cached_raw)
if parcel_gdf is None or len(parcel_gdf) == 0:
    raise RuntimeError("Cached parquet is empty.")
parcel_gdf = ensure_geodataframe(parcel_gdf)
print(f"✅ Loaded {len(parcel_gdf):,} rows  CRS={parcel_gdf.crs}")

# ── 2. Explore ─────────────────────────────────────────────────────────────────
print("\nUSEGROUP counts:")
print(parcel_gdf["USEGROUP"].value_counts(dropna=False).to_string())
print("\nVACIND counts:")
print(parcel_gdf["VACIND"].value_counts(dropna=False).to_string())

# ── 3. Condo cure ──────────────────────────────────────────────────────────────
n_dupes = parcel_gdf.duplicated(subset=["BLOCKLOT"], keep=False).sum()
print(f"\nDuplicate rows by BLOCKLOT: {n_dupes}")
if n_dupes > 0:
    numeric_sum_candidates = [
        "CURRLAND","CURRIMPR","BFCVLAND","BFCVIMPR","TAXBASE","FULLCASH",
        "LANDEXMP","IMPREXMP","Shape__Area","Shape__Length",
    ]
    numeric_sum_cols = [
        c for c in numeric_sum_candidates
        if c in parcel_gdf.columns and np.issubdtype(parcel_gdf[c].dtype, np.number)
    ]
    categorical_cols = [
        c for c in parcel_gdf.columns
        if c not in set(numeric_sum_cols + ["geometry","BLOCKLOT"])
    ]
    agg_dict = {c: "sum" for c in numeric_sum_cols}
    agg_dict.update({c: "first" for c in categorical_cols})
    collapsed = parcel_gdf.groupby("BLOCKLOT", dropna=False).agg(agg_dict).reset_index()
    geom_union = parcel_gdf.groupby("BLOCKLOT", dropna=False)["geometry"].apply(
        lambda geoms: unary_union([g for g in geoms if g is not None])
        if any(g is not None for g in geoms) else None
    )
    collapsed["geometry"] = geom_union.values
    parcel_gdf = gpd.GeoDataFrame(collapsed, geometry="geometry", crs=parcel_gdf.crs)
    print(f"✅ Rows after condo-cure collapse: {len(parcel_gdf):,}")

# ── 4. Categorise ──────────────────────────────────────────────────────────────
def categorize_property_type(row):
    usegroup = str(row.get("USEGROUP", "") or "").strip().upper()
    vacind   = str(row.get("VACIND",   "") or "").strip().upper()
    sdatcode = str(row.get("SDATCODE", "") or "").strip()
    if vacind == "Y":
        return "Vacant Land"
    if usegroup in ("E","EC"):
        return "Exempt / Governmental"
    if usegroup == "R":
        if sdatcode in ("11110","11115"):           return "Single Family"
        if sdatcode in ("11120","11125"):           return "Semi-Detached"
        if sdatcode in ("11130","11135","11136"):   return "Rowhouse"
        if sdatcode in ("91010","91020","91030"):   return "Condo/PUD"
        if sdatcode in ("11210",):                 return "Two Family"
        if sdatcode in ("11220",):                 return "Three Family"
        if sdatcode in ("11230","11235","11236","11310","11320"): return "Large Multi-Family (4+ units)"
        if sdatcode in ("11140","11150","11160"):   return "Residential Vacant"
        return "Other Residential"
    if usegroup in ("RC","CR"):      return "Mixed Use"
    if usegroup in ("C","CC"):
        if sdatcode in ("46000","46100","46200"):   return "Large Multi-Family (4+ units)"
        if sdatcode in ("44000","44100"):           return "Parking Garage"
        return "Commercial"
    if usegroup == "I":              return "Industrial"
    if usegroup == "U":              return "Utility"
    if usegroup == "M":              return "Open Space / Natural"
    return "Other"

parcel_gdf["PROPERTY_CATEGORY"] = parcel_gdf.apply(categorize_property_type, axis=1)
print("\nPROPERTY_CATEGORY counts:")
print(parcel_gdf["PROPERTY_CATEGORY"].value_counts(dropna=False).to_string())

# ── 5. Filter exempt ───────────────────────────────────────────────────────────
export_gdf = parcel_gdf.copy()
currland = pd.to_numeric(export_gdf.get("CURRLAND"), errors="coerce").fillna(0)
currimpr = pd.to_numeric(export_gdf.get("CURRIMPR"), errors="coerce").fillna(0)
landexmp = pd.to_numeric(export_gdf.get("LANDEXMP"), errors="coerce").fillna(0)
imprexmp = pd.to_numeric(export_gdf.get("IMPREXMP"), errors="coerce").fillna(0)
total_value  = currland + currimpr
total_exempt = landexmp + imprexmp
exempt_by_cat = export_gdf["PROPERTY_CATEGORY"] == "Exempt / Governmental"
exempt_by_val = (total_value > 0) & (total_exempt / total_value >= 0.995)
owner_text = (
    export_gdf[["OWNER_1", "OWNER_2", "OWNER_3"]]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
if "OWNER_ABBR" in export_gdf.columns:
    owner_abbr = export_gdf["OWNER_ABBR"].fillna("").astype(str).str.strip()
else:
    owner_abbr = pd.Series("", index=export_gdf.index, dtype="object")
government_owner_regex = "|".join(GOVERNMENT_OWNER_PATTERNS)
exempt_by_owner = owner_text.str.contains(government_owner_regex, case=False, na=False)
exempt_by_owner |= owner_abbr.eq("MCC")
export_gdf["exemption_flag"] = (exempt_by_cat | exempt_by_val | exempt_by_owner).astype(int)
before = len(export_gdf)
export_gdf = export_gdf[export_gdf["exemption_flag"] == 0].copy()
print(f"\n✅ Removed {before - len(export_gdf):,} fully exempt parcels → {len(export_gdf):,} remaining")

# ── 6. Export block ────────────────────────────────────────────────────────────
export_gdf["land_value"]        = pd.to_numeric(export_gdf.get("CURRLAND"), errors="coerce")
export_gdf["improvement_value"] = pd.to_numeric(export_gdf.get("CURRIMPR"), errors="coerce")
export_gdf["full_market_value"] = export_gdf["land_value"].fillna(0) + export_gdf["improvement_value"].fillna(0)
export_gdf["property_land_use_category"] = export_gdf["PROPERTY_CATEGORY"]

def categorize_property_refined(row):
    cat = str(row["PROPERTY_CATEGORY"])
    if "Vacant" in cat:   return "Vacant"
    if "Parking Garage" in cat: return "Parking Lot"
    if row["improvement_value"] < 0.5 * (row["land_value"] + row["improvement_value"]):
        return "Underdeveloped"
    return None

export_gdf["property_land_use_refined"] = export_gdf.apply(categorize_property_refined, axis=1)

# Area
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

export_gdf["geometry"] = export_gdf["geometry"].apply(
    lambda g: g if g is None or g.is_valid else g.buffer(0)
)
print("Computing geodesic areas...")
export_gdf["area_sqft"] = export_gdf["geometry"].apply(geodesic_area_sqft)
export_gdf.loc[export_gdf["area_sqft"] < 1, "area_sqft"] = np.nan

export_gdf["full_market_value_per_sqft"]  = export_gdf["full_market_value"]  / export_gdf["area_sqft"]
export_gdf["land_value_per_sqft"]         = export_gdf["land_value"]         / export_gdf["area_sqft"]
export_gdf["improvement_value_per_sqft"]  = export_gdf["improvement_value"]  / export_gdf["area_sqft"]

export_gdf = add_improvement_ratio_fields(
    export_gdf, land_col="land_value", improvement_col="improvement_value"
)

# Link
if "SDATLINK" in export_gdf.columns:
    export_gdf["link"] = export_gdf["SDATLINK"].astype(str).str.strip()
else:
    export_gdf["link"] = np.nan

# ── 7. Select columns & save ───────────────────────────────────────────────────
columns_to_export = [
    "geometry","exemption_flag","property_land_use_category","property_land_use_refined",
    "full_market_value","full_market_value_per_sqft","land_value","land_value_per_sqft",
    "improvement_value","improvement_value_per_sqft","TLLDIMPROV","IMPR_LAND_RATIO",
    "IMPR_LAND_PCT","IMPR_PCT_TOTAL","link",
]
for col in columns_to_export:
    if col not in export_gdf.columns:
        export_gdf[col] = np.nan

export_final = export_gdf[columns_to_export].rename(columns={"land_value": "current_full_land_value"})
export_final["geometry"] = export_final["geometry"].apply(
    lambda g: g if g is None or g.is_valid else g.buffer(0)
)
export_final = gpd.GeoDataFrame(export_final, geometry="geometry", crs=export_gdf.crs)
if export_final.crs is None or export_final.crs.to_epsg() != 4326:
    export_final = export_final.to_crs("EPSG:4326")

canonical_path = os.path.join(DATA_DIR, "baltimore-md-parcels.parquet")
dated_path     = os.path.join(DATA_DIR, f"baltimore-md-parcels_{today_str}.parquet")
export_final.to_parquet(canonical_path, index=False)
export_final.to_parquet(dated_path,     index=False)
print(f"\n✅ Saved canonical parquet: {canonical_path}")
print(f"✅ Saved dated parquet:     {dated_path}")
print(f"   Total rows: {len(export_final):,}")
print("\nRefined category counts:")
print(export_final["property_land_use_refined"].value_counts(dropna=False).to_string())

# ── 8. Upload to dev blob ──────────────────────────────────────────────────────
from azure.storage.blob import BlobServiceClient

conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not conn_str:
    print("⚠️  AZURE_STORAGE_CONNECTION_STRING not set — skipping upload")
    sys.exit(0)

container  = os.getenv("AZURE_DEV_CONTAINER", "parquets-dev")
blob_name  = "baltimore-md-parcels.parquet"
local_path = os.path.join(DATA_DIR, blob_name)

blob_service    = BlobServiceClient.from_connection_string(conn_str)
container_client = blob_service.get_container_client(container)
with open(local_path, "rb") as fh:
    container_client.upload_blob(name=blob_name, data=fh, overwrite=True)
print(f"\n✅ Uploaded → {container}/{blob_name}")

# ── 9. Promote to prod ─────────────────────────────────────────────────────────
dev_container  = os.getenv("AZURE_DEV_CONTAINER",  "parquets-dev")
prod_container = os.getenv("AZURE_PROD_CONTAINER", "parquets-prod")
dev_blob  = blob_service.get_blob_client(dev_container,  blob_name)
prod_blob = blob_service.get_blob_client(prod_container, blob_name)
if prod_blob.exists():
    prod_blob.delete_blob()
prod_blob.start_copy_from_url(dev_blob.url)
print(f"✅ Promoted → {prod_container}/{blob_name}")
