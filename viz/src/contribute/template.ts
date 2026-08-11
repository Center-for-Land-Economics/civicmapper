/**
 * Generates the city contribution Markdown file from the form state.
 * The output follows the 13-section city-proposal template (submitted as a
 * GitHub issue on the main repo).
 */

export interface ContributeFormState {
  // Step 1 — City Identity
  displayName: string;
  cityKey: string;
  stateCode: string;

  // Step 2 — Data Source
  sourceUrl: string;
  sourceType: string;    // e.g. "ArcGIS FeatureServer — Hamilton County"
  layerPath: string;     // e.g. "Layer 12 — Hamilton_County_Parcel_Polygons"
  authentication: string; // "None" | "API Key" | "Requires login"
  updateFrequency: string;
  dataVintage: string;
  sourceCRS: string;
  geometryIncluded: string; // "Yes" | "No — separate centroid layer"

  // Step 3 — Geographic Scope
  sourceCoverage: string; // "City-only" | "County-wide — {county}"
  clipMethod: string;     // "No clip needed" | "Spatial clip to city boundary" | ...
  filterField: string;    // e.g. "CITYNAME = 'Portland'" or "Spatial intersection"
  boundarySource: string; // e.g. "osmnx.geocode_to_gdf(...)"
  geographyNotes: string;

  // Step 4 — Field Mapping
  parcelIdField: string;
  landValueField: string;
  improvementValueField: string;
  landUseCategoryField: string;
  ownerNameField: string;
  assessorLinkAvailable: boolean;
  assessorLinkPattern: string;
  assessorLinkIdField: string;

  // Step 5 — Scale & Rendering
  approxParcelCount: string;
  pmtilesRecommended: boolean;
  includeParking: boolean;

  // Step 6 — Contributor info
  contributorNotes: string;
}

export const DEFAULT_STATE: ContributeFormState = {
  displayName: '',
  cityKey: '',
  stateCode: '',
  sourceUrl: '',
  sourceType: 'ArcGIS FeatureServer',
  layerPath: '',
  authentication: 'None',
  updateFrequency: 'Annual',
  dataVintage: new Date().getFullYear().toString(),
  sourceCRS: '',
  geometryIncluded: 'Yes',
  sourceCoverage: 'City-only',
  clipMethod: 'No clip needed',
  filterField: '',
  boundarySource: '',
  geographyNotes: '',
  parcelIdField: '',
  landValueField: '',
  improvementValueField: '',
  landUseCategoryField: '',
  ownerNameField: 'not used',
  assessorLinkAvailable: false,
  assessorLinkPattern: '',
  assessorLinkIdField: '',
  approxParcelCount: '',
  pmtilesRecommended: false,
  includeParking: false,
  contributorNotes: '',
};

/** Derive a safe city key from a display name. */
export function deriveCityKey(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .trim()
    .replace(/\s+/g, '')
    .replace(/^(city|the)\s*/i, '');
}

/** Derive a city file path from state and key. */
export function deriveFilePath(stateCode: string, cityKey: string): string {
  return `cities/${stateCode.toLowerCase()}/${cityKey}.md`;
}

/** Suggest whether PMTiles is needed based on parcel count. */
export function suggestPmtiles(approxCount: string): boolean {
  const n = parseInt(approxCount.replace(/[^0-9]/g, ''), 10);
  return !isNaN(n) && n > 100_000;
}

/** Generate the full Markdown contribution file. */
export function generateMarkdown(state: ContributeFormState): string {
  const displayWithState = `${state.displayName}, ${state.stateCode.toUpperCase()}`;
  const pmtilesVal = state.pmtilesRecommended ? 'Yes' : 'No';
  const parkingVal = state.includeParking ? 'Yes' : 'No';

  const clipSection = state.sourceCoverage.startsWith('County')
    ? `| **Source coverage**      | ${state.sourceCoverage}                                                      |
| **Clip method**          | ${state.clipMethod}                                                          |
| **Filter field & value** | ${state.filterField || 'Spatial intersection (no reliable municipal field)'}  |
| **Boundary source**      | ${state.boundarySource || '`osmnx.geocode_to_gdf("' + state.displayName + ', ' + state.stateCode.toUpperCase() + '")`'} |`
    : `| **Source coverage**      | City-only — no clip needed                                                   |
| **Clip method**          | None                                                                          |
| **Filter field & value** | ${state.filterField || 'n/a'}                                                |`;

  const notes = state.geographyNotes || 'Fill in any relevant notes about boundary precision or edge cases.';

  const assessorSection = state.assessorLinkAvailable
    ? `| **Link available** | Yes                                                              |
| **URL pattern** | \`${state.assessorLinkPattern}\`                                               |
| **ID field**    | \`${state.assessorLinkIdField}\`                                               |
| **Notes**       | Update the year component of the URL annually when the pipeline is refreshed. |`
    : `| **Link available** | No                                                               |
| **Notes**       | No per-parcel assessor link available for this data source.      |`;

  return `# City Contribution: ${displayWithState}

---

## 1. City Identity

| Field            | Value             |
|------------------|-------------------|
| **City key**     | \`${state.cityKey}\`      |
| **State code**   | \`${state.stateCode.toLowerCase()}\`              |
| **Display name** | \`${displayWithState}\`  |

---

## 2. Source Data

### 2a. Primary Source

| Field               | Value                                                                                                                 |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|
| **URL**             | \`${state.sourceUrl}\`                                                                                               |
| **Source type**     | ${state.sourceType}                                                                                                   |
| **Layer / path**    | ${state.layerPath}                                                                                                    |
| **Authentication**  | ${state.authentication}                                                                                               |
| **Update frequency** | ${state.updateFrequency}                                                                                             |
| **Data vintage**    | ${state.dataVintage}                                                                                                  |
| **CRS of source**   | ${state.sourceCRS} — reprojected to EPSG:4326 during ETL                                                             |
| **Geometry included** | ${state.geometryIncluded}                                                                                          |

---

## 3. Geographic Scope

| Field                    | Value                                                                              |
|--------------------------|------------------------------------------------------------------------------------|
${clipSection}
| **Notes**                | ${notes} |

---

## 4. Field Mapping

| Purpose                           | Source Column Name | Notes                                                                          |
|-----------------------------------|--------------------|--------------------------------------------------------------------------------|
| **Parcel ID** (aggregation key)   | \`${state.parcelIdField || 'TODO'}\`         | Unique parcel identifier — also used as condo aggregation key if applicable    |
| **Land market value**             | \`${state.landValueField || 'TODO'}\`        | Full market land value, USD                                                    |
| **Improvement market value**      | \`${state.improvementValueField || 'TODO'}\` | Market improvement (structure) value, USD                                      |
| **Parcel area (sq ft)**           | computed from geometry | Geodetic area (WGS84 ellipsoid) after reprojection                        |
| **Original land use / category**  | \`${state.landUseCategoryField || 'TODO'}\`  | Source land use classification field                                           |
| **Owner name**                    | ${state.ownerNameField || 'not used'}         | Used for owner-name-based exemption detection, if available                    |
| **Exemption indicator**           | derived            | Derived from category code or owner name pattern — see Section 8               |

---

## 5. Per-Parcel Assessor Link

| Field           | Value                                                                          |
|-----------------|--------------------------------------------------------------------------------|
${assessorSection}

---

## 6. Property Categories

### 6a. Original → Refined Category Mapping

<!-- TODO: Fill in the mapping from your source land use codes to CivicMapper refined categories. -->
<!-- Available refined categories: Single Family Residential, Multi-Family Residential, Condo / PUD, -->
<!-- Commercial, Industrial, Vacant, Parking Lot, Mixed Use, Institutional / Civic, -->
<!-- Open Space / Natural, Agricultural, Transportation / Utility, Other Residential, Other -->
<!-- Use "(excluded from output)" for government-owned / publicly exempt parcels. -->

| Source Code(s) | Original Label | Refined Category |
|----------------|----------------|-----------------|
| \`TODO\`       | TODO           | TODO            |

### 6b. Vacant & Parking Lot Detection

| Signal          | Detection Logic                                                              |
|-----------------|------------------------------------------------------------------------------|
| **Vacant**      | TODO — describe how vacant parcels are identified in the source data         |
| **Parking Lot** | TODO — describe how surface parking lots are identified in the source data   |

### 6c. Underdeveloped Detection

| Field    | Value                                                      |
|----------|------------------------------------------------------------|
| **Rule** | Use default threshold: improvement value < 50% of total value |
| **Notes** | TODO — note any adjustments needed for this data source   |

---

## 7. Condo & Duplicate Handling

| Field                    | Value                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| **Condos present?**      | TODO — Yes/No                                                                                        |
| **How they appear**      | TODO — describe how condos appear in the source data                                                 |
| **Aggregation key**      | \`${state.parcelIdField || 'TODO'}\`                                                                  |
| **Numeric aggregation**  | Sum land value, improvement value, and area across all rows in group                                 |
| **Category aggregation** | Take \`first\` for category, link ID, and other categorical fields                                   |
| **Geometry aggregation** | Union all footprints in the group                                                                     |

---

## 8. Exempt Parcel Detection

### 8a. Primary Exemption Signal

| Method                 | Details                                                                  |
|------------------------|--------------------------------------------------------------------------|
| **Category-based**     | TODO — which category codes indicate government/exempt ownership?        |
| **Zero value**         | \`land_value = 0 AND improvement_value = 0\` → likely exempt; flag for review |
| **Owner name pattern** | TODO — list any owner name patterns used for exemption detection          |

### 8b. Known Exempt Categories to Double-Check

- [ ] Government / municipal owned
- [ ] Public schools
- [ ] Parks and open space
- [ ] Utilities and right-of-way
- [ ] Nonprofit / religious institutions
- [ ] Public housing authority
- [ ] State and federal property

---

## 9. Scale & Rendering

| Field                           | Value                                                    |
|---------------------------------|----------------------------------------------------------|
| **Approx. parcel count**        | ${state.approxParcelCount || 'TODO'}                     |
| **PMTiles recommended**         | ${pmtilesVal}                                            |
| **Low-zoom cell size (meters)** | ${state.pmtilesRecommended ? '50' : 'n/a'}              |
| **Notes**                       | TODO — describe rendering considerations for this city.  |

---

## 10. Parking Data

| Field                           | Value                                                   |
|---------------------------------|---------------------------------------------------------|
| **Include parking dataset?**    | ${parkingVal}                                           |
| **Preferred extraction method** | OSM (default) — override with NAIP+ML if available      |
| **Separate parking data source** | None — extracted from OSM / imagery                   |
| **Notes**                       | TODO — describe any special parking data considerations  |

---

## 11. Data Quality Notes

| Issue                       | Description                                                                           |
|-----------------------------|-----------------------------------------------------------------------------------------------|
| **TODO**                    | Add any known data quality issues, field inconsistencies, or edge cases here.         |

---

## 12. Validation Checklist

- [ ] All required fields in Section 4 are filled
- [ ] Section 5 assessor link URL tested with a real parcel ID (if available)
- [ ] Section 6 covers all major category codes
- [ ] Section 7 condo handling described accurately
- [ ] Section 8 exempt logic validated against known government properties
- [ ] Parcel count estimated from source data
- [ ] Source URL is publicly accessible (tested in browser)

---

## 13. Contributor Notes

${state.contributorNotes || 'Add any additional context, known issues, or helpful tips for the reviewer here.'}
`;
}
