"""
Run Baltimore parcel ETL pipeline end-to-end.
Equivalent to running baltimore.ipynb with SCRAPE_DATA=1 and upload_dev=True.
"""
import os, glob, sys, numpy as np, pandas as pd, geopandas as gpd
from datetime import datetime
from shapely.ops import unary_union
from pyproj import Geod

sys.path.append("..")
from parcel_calculations import add_improvement_ratio_fields, geodesic_area_sqft
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

# ── 3. Multi-polygon BLOCKLOTs ─────────────────────────────────────────────────
# Some BLOCKLOTs come as several rows: pieces of one parcel plus exact duplicate rows (993 keys /
# 2,367 rows in the 2026-03-13 pull). Every attribute (CURRLAND, CURRIMPR, owner, SDATCODE, PIN) is
# BROADCAST identically onto each row; only OBJECTID and geometry differ. So values take `first`
# and geometry is the union. Summing them (what this block used to do) multiplied the value by the
# piece count, ~$133M of land. Shipping the rows uncollapsed put the full value on every piece
# (add-city skill §2/§2a). The assert below fails loudly if a future pull ever carries distinct
# per-row values, in which case they are separate records and need a different rule.
n_dupes = parcel_gdf.duplicated(subset=["BLOCKLOT"], keep=False).sum()
print(f"\nDuplicate rows by BLOCKLOT: {n_dupes} "
      f"({parcel_gdf.loc[parcel_gdf['BLOCKLOT'].duplicated(keep=False), 'BLOCKLOT'].nunique():,} keys)")
if n_dupes > 0:
    VALUE_COLS = [c for c in ["CURRLAND", "CURRIMPR", "FULLCASH", "TAXBASE"] if c in parcel_gdf.columns]
    varying = (parcel_gdf.groupby("BLOCKLOT", dropna=False)[VALUE_COLS]
               .nunique(dropna=False).gt(1).any(axis=1))
    assert not varying.any(), (
        f"{int(varying.sum())} BLOCKLOTs carry DIFFERENT values across rows — not a broadcast; "
        f"e.g. {varying[varying].index[:5].tolist()}")
    agg_dict = {c: "first" for c in parcel_gdf.columns if c not in ("geometry", "BLOCKLOT")}
    collapsed = parcel_gdf.groupby("BLOCKLOT", dropna=False).agg(agg_dict).reset_index()
    geom_union = parcel_gdf.groupby("BLOCKLOT", dropna=False)["geometry"].apply(
        lambda geoms: unary_union([g for g in geoms if g is not None])
        if any(g is not None for g in geoms) else None
    )
    collapsed["geometry"] = geom_union.values
    parcel_gdf = gpd.GeoDataFrame(collapsed, geometry="geometry", crs=parcel_gdf.crs)
    print(f"✅ Rows after BLOCKLOT collapse: {len(parcel_gdf):,}")
assert not parcel_gdf["BLOCKLOT"].duplicated().any(), "BLOCKLOT must be unique after collapse"

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
    # NOTE: 'Parking Garage' (SDAT 44000/44100) deliberately gets NO special case — a deck is a
    # built structure, so it is judged by the improvement ratio below like any other building.
    # This matches the repo-wide rule in parcel_calculations.classify_property_refined.
    if row["improvement_value"] < 0.5 * (row["land_value"] + row["improvement_value"]):
        return "Underdeveloped"
    return None

export_gdf["property_land_use_refined"] = export_gdf.apply(categorize_property_refined, axis=1)

# Area
geod = Geod(ellps="WGS84")
export_gdf["geometry"] = export_gdf["geometry"].apply(
    lambda g: g if g is None or g.is_valid else g.buffer(0)
)
print("Computing geodesic areas...")
export_gdf["area_sqft"] = export_gdf["geometry"].apply(geodesic_area_sqft)
export_gdf.loc[export_gdf["area_sqft"] < 1, "area_sqft"] = np.nan

export_gdf["land_area_acres"] = export_gdf["area_sqft"] / 43560.0
# Remnant = a sub-500-sqft sliver carrying an implausible rate. Baltimore rowhouse lots are
# legitimately tiny (2.5k+ parcels under 500 sqft), so area alone would hide real homes; the
# $1,500/sqft gate matches the flag_remnants.py --remnant-ppsf pass the shipped file carried.
_lpsf = pd.to_numeric(export_gdf["land_value"], errors="coerce") / export_gdf["area_sqft"]
export_gdf["likely_remnant"] = ((export_gdf["area_sqft"] < 500) & (_lpsf > 1500)).astype(int)
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
    "IMPR_LAND_PCT","IMPR_PCT_TOTAL","link","land_area_acres","likely_remnant",
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

# ── 8. Upload / promote ────────────────────────────────────────────────────────
# Deliberately NOT done here (this script used to upload to dev AND promote straight to prod on
# every run). Bake + upload + promote are separate, reviewed steps:
#   python data/scripts/parquet_to_pmtiles.py --city baltimore --h3 --drop-remnants
#   python data/upload_city_dev.py baltimore      # then verify on dev.civicmapper.org
#   python data/promote_to_prod.py baltimore-md-parcels
