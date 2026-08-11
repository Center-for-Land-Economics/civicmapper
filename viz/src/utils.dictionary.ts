import { DEV_CATEGORY_FIELD, SELECTED_CITY } from './config';

// Core fields required for app functionality
const CORE_FIELD_LABELS: Record<string, string> = {
  [DEV_CATEGORY_FIELD]: 'Property Category',
  REALIMPROV: 'Improvements Assessed Value',
  REALIMPROV_per_sqft: 'Improvements value/land ft²',
  REALLANDVA: 'Land Assessed Value',
  REALLANDVA_per_sqft: 'Land value/land ft²',
  land_value_per_sqm: 'Land value / m²',
  TLLDIMPROV: 'Total Land & Improvements',
  TLLDIMPROV_per_sqft: 'Total land & improvements/land ft²',
  IMPR_LAND_RATIO: 'Improvement to Land Ratio',
  IMPR_LAND_PCT: 'Improvement to Land Ratio (%)',
  IMPR_PCT_TOTAL: 'Improvement value share (%)',
  LAND_PCT_TOTAL: 'Land value share (%)'
};

// Hover descriptions for the metric picker (set as the <select>'s title for the chosen field).
export const FIELD_TOOLTIPS: Record<string, string> = {
  REALLANDVA_per_sqft: 'The land value of each parcel divided by the land area of the parcel',
  land_value_per_sqft: 'The land value of each parcel divided by the land area of the parcel',
  land_value_per_sqm: 'The land value of each parcel divided by the land area (m²) of the parcel',
  smooth_land_value_per_sqft: 'The land value of each parcel divided by the land area of the parcel',
  REALIMPROV_per_sqft: 'The value of all improvements (such as buildings) on each parcel divided by the land area of the parcel',
  improvement_value_per_sqft: 'The value of all improvements (such as buildings) on each parcel divided by the land area of the parcel',
  TLLDIMPROV_per_sqft: 'The value of (land + improvements) on each parcel divided by the land area of the parcel',
  full_market_value_per_sqft: 'The value of (land + improvements) on each parcel divided by the land area of the parcel',
  LAND_PCT_TOTAL: 'The portion of total value due to land. (Land value / total value)',
  IMPR_PCT_TOTAL: 'The portion of total value due to improvements. (Improvement value / total value)'
};

const CORE_FIELD_KEYS = new Set<string>(Object.keys(CORE_FIELD_LABELS));

export function isCoreField(key: string): boolean {
  return CORE_FIELD_KEYS.has(key);
}

export let FIELD_LABELS: Record<string, string> = { ...CORE_FIELD_LABELS };
export let ALL_FIELDS: string[] = Object.keys(FIELD_LABELS);
export const DROPDOWN_FIELDS: string[] = [
  'REALLANDVA_per_sqft',
  'land_value_per_sqm',
  'TLLDIMPROV_per_sqft',
  'LAND_PCT_TOTAL',
  'IMPR_PCT_TOTAL',
  'current_tax_per_sqft',
  'REALIMPROV_per_sqft',
  'land_value_per_sqft',
  'smooth_land_value_per_sqft',
  'improvement_value_per_sqft',
  'full_market_value_per_sqft'
];

type CityDictionary = Record<string, unknown>;
const cityDictionaryModules = import.meta.glob<{ default: CityDictionary }>('./dictionaries/*.json');

// City configuration (PMTiles settings, etc.)
export let CITY_CONFIG: Record<string, any> | null = null;

// Merge additional labels from a JSON file bundled with the app (no network fetch)
// Filters out keys starting with _ to prevent config from appearing as field labels
export async function loadDataDictionary() {
  FIELD_LABELS = { ...CORE_FIELD_LABELS };
  ALL_FIELDS = Object.keys(FIELD_LABELS);
  CITY_CONFIG = null;

  try {
    const loadModule = cityDictionaryModules[`./dictionaries/${SELECTED_CITY}.json`];
    if (!loadModule) return;
    const extra = (await loadModule()).default ?? {};

    // Extract _config separately before filtering
    CITY_CONFIG = (extra._config as Record<string, any> | null | undefined) ?? null;

    // Filter out keys starting with _ when merging into FIELD_LABELS
    const filtered: Record<string, string> = {};
    for (const [key, value] of Object.entries(extra)) {
      if (!key.startsWith('_') && typeof value === 'string') {
        filtered[key] = value;
      }
    }

    FIELD_LABELS = { ...CORE_FIELD_LABELS, ...filtered };
    ALL_FIELDS = Object.keys(FIELD_LABELS);
  } catch {
    // ignore
  }
}

// Helper functions for city configuration
export function getCityConfig(): Record<string, any> | null {
  return CITY_CONFIG;
}

export function cityUsesPmtiles(): boolean {
  return CITY_CONFIG?.usePmtiles === true;
}
