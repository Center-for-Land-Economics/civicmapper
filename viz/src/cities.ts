/**
 * City registry — data-driven.
 *
 * Each city is a JSON file at ./cities/<key>.json holding its CityDef fields plus
 * `coords` ([lng, lat] map center). To add a city: drop in cities/<key>.json and
 * dictionaries/<key>.json — no shared-file edits needed. The filename (minus .json)
 * is the city key used in ?city=<key> URLs.
 *
 * A private/extra city pack can be injected by copying additional <key>.json files
 * into src/cities/ (+ dictionaries) before build; nothing else needs to change.
 */

export type CityKey = string;

export interface CityDef {
  /** Human-readable display name. */
  displayName: string;
  /** Two-letter state code (US) or country code (non-US, e.g. 'ee'). Used for the picker label. */
  state: string;
  /** Currency symbol for value formatting. Defaults to '$'. Set '€' for euro-denominated cities. */
  currencySymbol?: string;
  /** Overrides the region shown in the picker label ("<displayName>, <region>"). Defaults to the
   *  uppercased two-letter `state` (US style, e.g. "MD"). Set to a full name for non-US cities
   *  (e.g. "Estonia") so the label reads "Tallinn, Estonia" rather than "Tallinn, EE". */
  displayRegion?: string;
  /** Area unit system for value-per-area + area displays. Defaults to 'imperial' (per sqft / acres).
   *  'metric' shows per m² / hectares (non-US cities). */
  unitSystem?: 'imperial' | 'metric';
  /** The source has ONLY a single combined/total assessed value — no land-vs-building split
   *  (e.g. Oslo: the eiendomsskatt `skattegrunnlag` is total property value; Norway publishes no
   *  land valuation). The value still lives in REALLANDVA (the required field), but it is TOTAL
   *  value, so the UI must never call it "land value": the headline blurb says "total assessed
   *  value", and the meaningless land/improvement-split rows + metrics are suppressed. */
  combinedValueOnly?: boolean;
  /** Parquet filename in blob storage (top-level). */
  filename: string;
  /** PMTiles filename in blob storage, if generated. Leave undefined for smaller cities. */
  pmtilesFilename?: string;
  /** Optional cache-busting token appended to PMTiles + its metadata requests.
   *  Bump on every re-bake/re-upload so the API proxy / CDN serves the fresh file. */
  pmtilesVersion?: string;
  /** When true, hide the "Vacant & Underdeveloped" analysis tab for this city. Defaults to enabled. */
  hideUnderutilized?: boolean;
  /** When true, hide tiny sub-500-sqft sliver remnants (likely_remnant=1) whose per-sqft
   *  values are meaningless (a real account value sitting on a fragment polygon). */
  hideRemnants?: boolean;
  /** Dev-only city: never listed in the city picker, and on a deployed host it won't load even
   *  via ?city=<key> (falls back to the default). Reachable only on localhost via ?city=<key>. */
  devOnly?: boolean;
  /** Parking lot parquet filename inside the parking/ subfolder, or undefined if not available. */
  parkingFilename?: string;
  /** Optional cache-busting token appended to parking dataset requests. */
  parkingVersion?: string;
  /** Field holding the refined development category used for primary coloring/filtering. */
  devCategoryField: string;
  /** Field holding the original/raw land use category from the source data. */
  origCategoryField: string;
  /** URL-slug aliases that also resolve to this city (lower-cased). */
  aliases?: string[];
  /** Name of the parcel property holding the municipal jurisdiction tag (e.g. "jurisdiction").
   *  When set (and no jurisdictionGroups), the single-group de-emphasis treatment is enabled. */
  jurisdictionField?: string;
  /** The jurisdiction shown normally by default; all others are de-emphasized. */
  primaryJurisdiction?: string;
  /** Multiple region grouping schemes the user can switch between (e.g. municipal jurisdiction,
   *  council district, neighborhood). Each `field` is a categorical parcel/hex property; the bake
   *  emits a region->count map per field under metadata.groups. Takes precedence over
   *  jurisdictionField when present. `defaultMode` sets the initial selection for that group. */
  jurisdictionGroups?: Array<{
    field: string;
    label: string;
    primary?: string;
    defaultMode?: 'all' | 'primaryOnly';
    /** Static GeoJSON of this group's region boundaries (served from viz/public via
     *  import.meta.env.BASE_URL). Drives the 2D "Overlays" layer. */
    overlayUrl?: string;
  }>;
  /** Optional rail transit overlay (lines colored by route + station markers) drawn on top of
   *  the parcel map, toggled by a "Show <label> stations & lines" checkbox. Both files are static
   *  GeoJSONs in viz/public. See transit.ts. */
  transitOverlay?: {
    lines: string;
    stations: string;
    label?: string;
  };
  /** Free-form provenance/engineering notes carried over from the old inline comments.
   *  Not used by the app — documentation that travels with the city config. */
  notes?: string[];
}

type CityFile = CityDef & { coords: [number, number] };

const cityModules = import.meta.glob<{ default: CityFile }>('./cities/*.json', { eager: true });

export const CITIES: Record<CityKey, CityDef> = {};
/** [lng, lat] map center per city. */
export const CITY_COORDS: Record<CityKey, [number, number]> = {};

for (const [path, mod] of Object.entries(cityModules)) {
  const key = path.replace('./cities/', '').replace('.json', '');
  const { coords, ...def } = mod.default;
  if (!Array.isArray(coords) || coords.length !== 2) {
    throw new Error(`cities/${key}.json is missing a valid coords [lng, lat] pair`);
  }
  CITIES[key] = def;
  CITY_COORDS[key] = coords;
}

/** Full state names keyed by the two-letter code used in CityDef.state. */
export const STATE_NAMES: Record<string, string> = {
  co: 'Colorado', ct: 'Connecticut', dc: 'District of Columbia', il: 'Illinois', in: 'Indiana', md: 'Maryland', mi: 'Michigan',
  mn: 'Minnesota', nm: 'New Mexico', ny: 'New York', oh: 'Ohio', ok: 'Oklahoma',
  or: 'Oregon', tx: 'Texas', va: 'Virginia', wa: 'Washington', wv: 'West Virginia',
  ee: 'Estonia', dk: 'Denmark'
};

/** All valid city keys. */
export const CITY_KEYS = Object.keys(CITIES) as CityKey[];

/** Canonical UI label for a city, e.g. "Fort Collins, CO". */
export function formatCityLabel(cityKey: CityKey): string {
  const city = CITIES[cityKey];
  return `${city.displayName}, ${city.displayRegion ?? city.state.toUpperCase()}`;
}

/**
 * Resolve a raw URL param value to a CityKey.
 * Handles aliases (e.g. "st-paul" → "stpaul") and case-insensitivity.
 * Returns "southbend" if the value is missing or unrecognised.
 */
export function resolveCityKey(raw: string | null | undefined): CityKey {
  const c = (raw ?? '').toLowerCase().trim();
  if (!c) return 'southbend';
  if (c in CITIES) return c as CityKey;
  // Check aliases
  for (const [key, def] of Object.entries(CITIES)) {
    if (def.aliases?.includes(c)) return key as CityKey;
  }
  return 'southbend';
}
