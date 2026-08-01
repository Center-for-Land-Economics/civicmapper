# Parcel GeoParquet format (adding a new city)

This app reads a **GeoParquet** file and converts it to GeoJSON in `viz/src/main.ts` via `toGeoJson()`.
The expectations below reflect the current loader behavior and the dropdown/popup rules.

## Geometry

- **Type:** `Polygon` or `MultiPolygon` only.
  - The loader accepts any geometry whose type name includes `Polygon` (case-insensitive).
  - Point/Line geometries are **ignored**, and the load will fail if no polygon-like features exist.
- **Column:** Standard GeoParquet geometry column (typically `geometry`).

## CRS

- **Expected CRS:** `EPSG:4326` (WGS84 longitude/latitude).
  - The app does not reproject on the client.
  - If your data is in a local/state plane CRS, reproject to 4326 before exporting.

## Required columns

These must exist on each feature’s properties (and be numeric where noted):

- **Category (refined)**
  - `property_category_refined` (South Bend)
  - `property_land_use_refined` (Syracuse/Spokane/Rochester/Bellingham/Morgantown)
- **Values (numeric)**
  - `REALIMPROV` (improvements assessed value)
  - `REALLANDVA` (land assessed value)

The loader also uses `PROPERTY_CATEGORY` or `property_land_use_category` for **original** category filters if present.

## Derived fields (computed client-side)

If `REALIMPROV` and `REALLANDVA` are present, the app computes:

- `TLLDIMPROV` (total land + improvements)
- `IMPR_LAND_RATIO`
- `IMPR_LAND_PCT`
- `IMPR_PCT_TOTAL`

You may include these in the parquet, but they are not required.

## Dropdown fields (only these appear in the metric dropdowns)

The app uses a fixed allowlist for dropdowns in `viz/src/utils.dictionary.ts`:

- `REALLANDVA_per_sqft` (land value per sqft)
- `TLLDIMPROV_per_sqft` (total value per sqft; only shown if present)
- `IMPR_LAND_PCT` (improvement-to-land ratio percent)
- `current_tax_per_sqft` (current tax per sqft)
- `REALIMPROV_per_sqft` (improvements value per sqft)

If one of these fields is missing, it simply won’t appear in the dropdowns.

## Popup-only fields

Any additional fields you want to show **only in the parcel popup** (and not in dropdowns)
must be:

1. Present in the parquet properties.
2. Included in the **data dictionary** for the city (e.g., `viz/src/dictionaries/<city>.json`).

The loader trims properties to the data dictionary keys, so fields not in the dictionary are removed
before the popup renders.

## Optional input aliases (auto-normalized)

For compatibility with some city datasets, the loader maps these to expected keys **if** the
standard keys are missing:

- `improvement_value` → `REALIMPROV`
- `current_full_land_value` or `land_value` → `REALLANDVA`
- `improvement_value_per_sqft` → `REALIMPROV_per_sqft`
- `land_value_per_sqft` → `REALLANDVA_per_sqft`

## Property category refined

When generating the refined category field, define it similarly to:

```python
# -----------------------------
def categorize_property_refined(row):
    cat = str(row["PROPERTY_CATEGORY"])
    if "Vacant" in cat:
        return "Vacant"
    elif "Parking" in cat:
        return "Parking Lot"
    elif row["improvement_value"] < 0.5 * (row["land_value"] + row["improvement_value"]):
        return "Underdeveloped"
    else:
        return None
```

Notes:
- Store the result in the refined category column required for your city.
- `None` (or empty) is allowed for non-underutilized parcels.
