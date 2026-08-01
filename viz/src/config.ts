import { API_BASE as ENV_API_BASE } from './env';
import { CITIES, resolveCityKey } from './cities';

// Repo migrated to Center-for-Land-Economics/civicmapper (2026-07-07).

// API and tiles via env (Vite)
export const API_BASE = ENV_API_BASE;
export const TILE_URL = (import.meta as any).env?.VITE_TILE_API_URL || 'http://localhost:8000/tiles/{z}/{x}/{y}.pbf';
function normalizeBaseUrl(value: string | undefined | null, fallback = ''): string {
  const trimmed = typeof value === 'string' ? value.trim() : '';
  if (!trimmed) return fallback;
  return trimmed.replace(/\/+$/, '') || fallback;
}

// In local dev the browser can't fetch the blob directly (no CORS headers for localhost), so route
// dataset fetches through Vite's same-origin `/data` proxy (vite.config.ts), which forwards to the
// blob server-side. Deployed builds use the API proxy (see DEFAULT_DATASET_URL / getPmtilesUrl), so
// the blob default here is only a fallback. NOTE: do NOT set VITE_PARQUET_BASE_URL=/data to achieve
// this — vite.config.ts uses that same env var as the proxy's UPSTREAM TARGET, so pointing it at
// `/data` makes the proxy forward to itself (HTTP 500). Leave it unset; the default below handles it.
export const PARQUET_BASE_URL = normalizeBaseUrl(
  (import.meta as any).env?.VITE_PARQUET_BASE_URL,
  (import.meta as any).env?.DEV ? '/data' : 'https://landeconomics.blob.core.windows.net/parquets-dev'
);

const PMTILES_BASE_URL_RAW = normalizeBaseUrl(
  (import.meta as any).env?.VITE_PMTILES_BASE_URL,
  PARQUET_BASE_URL
);

// Helper function to get PMTiles file URL.
// In deployed environments, always prefer the API proxy so branch/dev/prod
// hosts do not depend on Azure Blob CORS allowlists.
export function getPmtilesUrl(filename: string): string {
  // Cache-bust: the PMTiles (and its metadata) live at a fixed filename, so when a
  // city's tiles are re-baked and re-uploaded the API proxy / CDN can serve STALE
  // byte-ranges of the file — which corrupts individual tiles (missing/garbled) on
  // the deployed site even though direct-from-blob (local dev) is fine. Appending a
  // version param (bumped in cities.ts on each re-bake) forces a fresh fetch.
  const v = (CITIES[SELECTED_CITY] as any)?.pmtilesVersion as string | undefined;
  // Local override (VITE_PMTILES_BASE_URL, e.g. serving one freshly-baked city's tiles from
  // public/) applies ONLY to files matching VITE_PMTILES_LOCAL_PREFIX when that's set — so other
  // cities still load from the blob instead of 404ing against the local server. With no prefix it
  // applies to everything (legacy behavior).
  const localBase = (import.meta as any).env?.VITE_PMTILES_BASE_URL;
  const localPrefix = (import.meta as any).env?.VITE_PMTILES_LOCAL_PREFIX as string | undefined;
  if (localBase && (!localPrefix || filename.startsWith(localPrefix))) {
    return appendVersionParam(`${PMTILES_BASE_URL_RAW}/${filename}`, v);
  }
  // In deployed environments, use API proxy to avoid CORS issues
  if (isDeployed) {
    return appendVersionParam(`${API_BASE}/data/${filename}`, v);
  }
  // In local dev, fetch from blob storage directly (the default base, not the local override)
  return appendVersionParam(`${PARQUET_BASE_URL}/${filename}`, v);
}

// Keep PMTILES_BASE_URL for backward compatibility (but prefer getPmtilesUrl)
export const PMTILES_BASE_URL = PMTILES_BASE_URL_RAW;

export const DATA_DICTIONARY_BASE_URL = normalizeBaseUrl(
  (import.meta as any).env?.VITE_DATA_DICTIONARY_BASE_URL,
  PARQUET_BASE_URL
);

// Glyphs endpoint so symbol/text layers (e.g. the 2D region-overlay labels) can render over the
// raster basemaps, which otherwise ship no fonts. maplibre demotiles serves "Open Sans *".
const GLYPHS_URL = 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf';

// Base map styles
export const OSM_STYLE: any = {
  version: 8,
  glyphs: GLYPHS_URL,
  sources: { 'osm-tiles': { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256, attribution: '© OpenStreetMap contributors' } },
  layers: [{ id: 'osm-tiles', type: 'raster', source: 'osm-tiles', minzoom: 0, maxzoom: 19 }]
};

// OpenFreeMap tiles; may fail if CORS headers are missing
export const OPENFREEMAP_STYLE: any = {
  version: 8,
  glyphs: GLYPHS_URL,
  sources: {
    'ofm-tiles': {
      type: 'raster',
      tiles: ['https://tile.openfreemap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution: '© OpenFreeMap, © OpenStreetMap contributors'
    }
  },
  layers: [{ id: 'ofm-tiles', type: 'raster', source: 'ofm-tiles', minzoom: 0, maxzoom: 19 }]
};

const TOPO_STYLE: any = {
  version: 8,
  glyphs: GLYPHS_URL,
  sources: { 'topo-tiles': { type: 'raster', tiles: ['https://a.tile.opentopomap.org/{z}/{x}/{y}.png', 'https://b.tile.opentopomap.org/{z}/{x}/{y}.png', 'https://c.tile.opentopomap.org/{z}/{x}/{y}.png'], tileSize: 256, attribution: '© OpenStreetMap contributors, SRTM | Map style © OpenTopoMap (CC-BY-SA)' } },
  layers: [{ id: 'topo-tiles', type: 'raster', source: 'topo-tiles', minzoom: 0, maxzoom: 17 }]
};

// Simple, offline-safe background style
const SIMPLE_STYLE: any = {
  version: 8,
  glyphs: GLYPHS_URL,
  sources: {},
  layers: [
    {
      id: 'background',
      type: 'background',
      paint: { 'background-color': '#e8eaed' }
    }
  ]
};

export const BASEMAP_STYLES: Record<string, any> = {
  'Simple Gray': SIMPLE_STYLE,
  'OpenStreetMap': OSM_STYLE,
  'OpenFreeMap': OPENFREEMAP_STYLE,
  'Topographic': TOPO_STYLE
};

// Parking page basemap: Carto Voyager vector style.
// Cool/neutral palette with building footprints — strong contrast with YlOrRd parking fills.
export const PARKING_BASEMAP_STYLE = 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json';

// Source / layer IDs
export const SOURCE_ID = 'gp-source';
export const LAYER_ID = 'gp-extrusions';
export const LAYER_ID_LOW = 'gp-extrusions-low';
export const LAYER_ID_MID = 'gp-extrusions-mid';
export const ERROR_LAYER_ID = 'gp-error';

// Autoscale caps
export const HEIGHT_CAP_METERS = 1000;
// Per-map default caps (meters). Allows independent tuning per map.
export const HEIGHT_CAPS = {
  main: 1000,
  under: 1000,
  ratio: 1000
} as const;
export const HEIGHT_PCTL = 99;

// Color ramps. Default is Viridis (perceptually uniform, colorblind-safe). The old green→red→
// purple palette ("default"/"GreenRedPurple") was removed; LEGACY_DEFAULT_RAMP_KEY is kept only so
// old saved settings that stored it migrate cleanly to the new default (see normalizeRampKey).
export const DEFAULT_RAMP_KEY = 'Viridis';
export const LEGACY_DEFAULT_RAMP_KEY = 'GreenRedPurple';
export const COLOR_RAMPS: Record<string, string[]> = {
  Viridis: ['#440154','#46327E','#365C8D','#277F8E','#1FA187','#4AC16D','#A0DA39','#FDE725'],
  Magma:   ['#000004','#1B0C41','#4F0A6D','#7A1E6C','#A52C60','#CF4446','#ED6925','#FB9F06','#F7D13D','#FCFDBF'],
  Plasma:  ['#0D0887','#5B02A3','#9A179B','#CB4679','#ED7953','#FB9F3A','#F0F921'],
  Turbo:   ['#30123B','#4145AB','#2CC0F0','#6AE4B4','#C6F86D','#F9DD32','#F28C21','#CB3E1F','#8A0D2C'],
  YlOrRd:  ['#FFFFB2','#FECC5C','#FD8D3C','#F03B20','#BD0026'],
  Blues:   ['#DEEBF7','#9ECAE1','#6BAED6','#3182BD','#08519C'],
  Reds:    ['#FEE5D9','#FCBBA1','#FB6A4A','#DE2D26','#A50F15']
};

// Unit conversion (unchanged)
export const UNIT_TO_METERS = {
  centimeters: 0.01,
  meters: 1,
  inches: 0.0254,
  feet: 0.3048,
  kilometers: 1000,
  miles: 1609.344,
  stories: 3.3
};

// Data field used for vacancy filtering
export const UNDERUTILIZED_DEFAULTS = ['Vacant', 'Parking Lot', 'Underdeveloped'];

// City selection via query string (?city=<key>). Defaults to southbend.
// All city metadata lives in cities.ts — add new cities there.
function getCityFromUrl() {
  try {
    const u = new URL(window.location.href);
    const key = resolveCityKey(u.searchParams.get('city'));
    // Dev-only cities (e.g. 'harris') must not load on a deployed host even via ?city= — fall back
    // to the default. (Computed locally; the exported `isDeployed` const isn't defined yet here.)
    const host = u.hostname;
    const deployed = host !== '' && !host.includes('localhost') && !host.includes('127.0.0.1');
    if (deployed && (CITIES[key] as any)?.devOnly) return 'southbend' as const;
    return key;
  } catch {
    return 'southbend' as const;
  }
}

export const SELECTED_CITY = getCityFromUrl();

// City-specific fields — driven by cities.ts, no more manual if/else chains.
export const DEV_CATEGORY_FIELD = CITIES[SELECTED_CITY].devCategoryField;
export const ORIG_CATEGORY_FIELD = CITIES[SELECTED_CITY].origCategoryField;

const _city = CITIES[SELECTED_CITY];

// Whether the "Vacant & Underdeveloped" analysis tab is available for the current
// city. Enabled by default; set hideUnderutilized: true in cities.ts to turn it off.
export const UNDERUTILIZED_ENABLED: boolean = !_city.hideUnderutilized;

// When true, hide tiny sub-500-sqft sliver remnants (likely_remnant=1) for this city,
// whose per-sqft values are meaningless. Set hideRemnants: true in cities.ts.
export const HIDE_REMNANTS: boolean = (_city as any).hideRemnants === true;

// Optional rail transit overlay (lines + stations) for this city, or null. Points at two
// static GeoJSONs in viz/public (see transit.ts / CityDef.transitOverlay).
export const TRANSIT_OVERLAY: { lines: string; stations: string; label?: string } | null =
  (_city as any).transitOverlay ?? null;

// When true, display value-per-area in €/m² and areas in hectares (non-US cities).
// Defaults to imperial (per sqft / acres). Set unitSystem: 'metric' in cities.ts.
export const METRIC_UNITS: boolean = (_city as any).unitSystem === 'metric';

function appendVersionParam(url: string, version?: string): string {
  if (!version) return url;
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}v=${encodeURIComponent(version)}`;
}

export const REMOTE_DATASET_URL = `${PARQUET_BASE_URL}/${_city.filename}`;

// Construct API proxy URL dynamically
export const API_PROXY_DATASET_URL = `${API_BASE}/data/${_city.filename}`;

// Detect environment by hostname.
// - localhost / 127.0.0.1 : local dev (direct blob, Vite proxy handles CORS)
// - everything else       : deployed app (use API proxy for data)
const hostname = typeof window !== 'undefined' ? window.location.hostname : '';
const isDeployed = hostname !== '' &&
  !hostname.includes('localhost') &&
  !hostname.includes('127.0.0.1');

// Explicit dataset override (VITE_DEFAULT_DATASET_URL) is a single fixed path — e.g. a local Houston
// parquet in public/ used to preview one freshly-baked city without the blob upload. It must be scoped
// the SAME way getPmtilesUrl scopes VITE_PMTILES_BASE_URL: apply it ONLY to cities whose filename
// matches VITE_PMTILES_LOCAL_PREFIX (when set), so pointing it at Houston doesn't silently force
// Houston's parcels onto every OTHER city (which produced a Houston map under, e.g., a Tallinn label).
// With no prefix set it applies to everything (legacy behavior).
const DATASET_OVERRIDE = (import.meta as any).env?.VITE_DEFAULT_DATASET_URL as string | undefined;
const DATASET_OVERRIDE_PREFIX = (import.meta as any).env?.VITE_PMTILES_LOCAL_PREFIX as string | undefined;
const DATASET_OVERRIDE_APPLIES = !!DATASET_OVERRIDE &&
  (!DATASET_OVERRIDE_PREFIX || _city.filename.startsWith(DATASET_OVERRIDE_PREFIX));

// Always prefer the (scoped) explicit override; otherwise use API proxy on deployed
// environments and direct blob locally.
export const DEFAULT_DATASET_URL = DATASET_OVERRIDE_APPLIES
  ? (DATASET_OVERRIDE as string)
  : (isDeployed ? API_PROXY_DATASET_URL : REMOTE_DATASET_URL);

// ---------------------------------------------------------------------------
// Parking lot dataset URLs (isolated from parcel files)
// Parking data lives at parking/ subfolder in blob storage.
// Both exports will be null if parkingFilename is not set for the current city.
// ---------------------------------------------------------------------------

/** Raw parking filename for the current city, or null if not yet available. */
export const PARKING_FILENAME: string | null = _city.parkingFilename ?? null;
export const PARKING_VERSION: string | undefined = _city.parkingVersion;

/** Full URL to the parking GeoParquet for the current city. */
export const PARKING_DATASET_URL: string | null = PARKING_FILENAME
  ? appendVersionParam((isDeployed
      ? `${API_BASE}/data/parking/${PARKING_FILENAME}`
      : `${PARQUET_BASE_URL}/parking/${PARKING_FILENAME}`), PARKING_VERSION)
  : null;

/** Full URL to the parking metadata JSON for the current city. */
export const PARKING_METADATA_URL: string | null = PARKING_FILENAME
  ? appendVersionParam(
      (isDeployed
        ? `${API_BASE}/data/parking/${PARKING_FILENAME.replace('.parquet', '-metadata.json')}`
        : `${PARQUET_BASE_URL}/parking/${PARKING_FILENAME.replace('.parquet', '-metadata.json')}`),
      PARKING_VERSION
    )
  : null;

// ---------------------------------------------------------------------------
// Dev-only local-first dataset resolution
// ---------------------------------------------------------------------------
// In `npm run dev`, prefer a locally-served copy of the CURRENT city's file if one has been
// dropped into viz/public/ (Vite serves public/<path> at `/<path>`); otherwise fall back to the
// normal remote URL. This lets you preview a city that isn't on the blob yet — and sidesteps blob
// CORS on localhost — WITHOUT a global override that forces one file onto every city (the old
// VITE_DEFAULT_DATASET_URL foot-gun). Per-city: cities with a local copy load locally, everything
// else loads from the server. A no-op in production (returns the remote URL unchanged).
const IS_DEV = !!(import.meta as any).env?.DEV;

/** Vite-served path to a local copy of the current city's parcel file (dev only), else null. */
export const LOCAL_DATASET_PATH: string | null = IS_DEV ? `/${_city.filename}` : null;
/** Vite-served path to a local copy of the current city's parking file (dev only), else null. */
export const LOCAL_PARKING_DATASET_PATH: string | null =
  IS_DEV && PARKING_FILENAME ? `/parking/${PARKING_FILENAME}` : null;

/**
 * Return `localPath` if it exists on the dev server (HEAD 200), else `remoteUrl`.
 * No-op outside dev or when localPath is null. Never throws.
 */
export async function resolveLocalFirst(localPath: string | null, remoteUrl: string): Promise<string> {
  if (!IS_DEV || !localPath) return remoteUrl;
  try {
    const r = await fetch(localPath, { method: 'HEAD' });
    // Vite 404s a missing .parquet (no SPA fallback for file-like paths), so 200 = the real file.
    if (r.ok && !/text\/html/i.test(r.headers.get('content-type') || '')) {
      console.log('[Dataset] Using local dev copy:', localPath);
      return localPath;
    }
  } catch { /* fall through to remote */ }
  return remoteUrl;
}

/** Full URL to the per-city land-totals JSON (parking-share-of-taxable-land denominators), or
 *  null. This is a small parcel-derived sidecar (citywide + per-region NON-EXEMPT land value)
 *  the Parking page fetches so it can show parking value as a % of taxable land value without
 *  loading the whole parcel dataset. Top-level data blob (like the parcel parquet). The Parking
 *  page degrades gracefully (hides the % line) when this 404s, so it can roll out per-city. */
const LAND_TOTALS_FILENAME: string | null = _city.filename.includes('-parcels.parquet')
  ? _city.filename.replace('-parcels.parquet', '-land-totals.json')
  : null;
export const LAND_TOTALS_URL: string | null = LAND_TOTALS_FILENAME
  ? appendVersionParam((isDeployed
      ? `${API_BASE}/data/${LAND_TOTALS_FILENAME}`
      : `${PARQUET_BASE_URL}/${LAND_TOTALS_FILENAME}`), _city.pmtilesVersion ?? _city.parkingVersion)
  : null;

/**
 * Master feature flag for the Surface Parking module.
 * Set VITE_PARKING_ENABLED=true in viz/env/.env.production (or local .env) to enable.
 * When false/absent, the parking UI is completely hidden regardless of city data availability.
 */
const PARKING_ENABLED_ENV = (import.meta as any).env?.VITE_PARKING_ENABLED;
export const PARKING_ENABLED: boolean =
  typeof PARKING_ENABLED_ENV === 'string'
    ? PARKING_ENABLED_ENV === 'true'
    : true;

/**
 * Master feature flag for the PDF report button. Global (not per-city). Defaults OFF — the button
 * is hidden everywhere (local dev + shipped builds) unless VITE_REPORT_ENABLED="true" is set. Turn
 * it on later by flipping that flag to "true" in the deploy workflow (or a local .env for testing).
 */
const REPORT_ENABLED_ENV = (import.meta as any).env?.VITE_REPORT_ENABLED;
export const REPORT_ENABLED: boolean = REPORT_ENABLED_ENV === 'true';
