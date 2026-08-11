/**
 * parking.ts — Surface Parking Analysis module for CivicMapper.
 *
 * Standalone module: does NOT import from main.ts.
 * Loads a separate parking-lots GeoParquet (isolated from parcel files),
 * renders a 2D flat MapLibre fill layer colored by land value / sqft
 * and populates the parking summary panel.
 *
 * Feature gate: only runs if PARKING_ENABLED env var is true AND
 * the city dictionary has hasParkingData: true.
 */

import './analytics';
import './design-system.css';
import './components.css';
import 'maplibre-gl/dist/maplibre-gl.css';
import maplibregl from 'maplibre-gl';
import { toGeoJson } from 'geoparquet';
import { compressors } from 'hyparquet-compressors';

import {
  PARKING_BASEMAP_STYLE,
  COLOR_RAMPS,
  PARKING_DATASET_URL,
  LOCAL_PARKING_DATASET_PATH,
  resolveLocalFirst,
  PARKING_ENABLED,
  SELECTED_CITY,
  LAND_TOTALS_URL,
  METRIC_UNITS,
} from './config';
import { formatCityLabel, CITIES } from './cities';
import { createJurisdiction, OVERLAY_PALETTE } from './jurisdiction';
import { loadDataDictionary, getCityConfig } from './utils.dictionary';
import { urlToAsyncBuffer, sanitizeFeaturesInPlace } from './utils.sanitize';
import { bbox, normalizeWindingInPlace } from './utils.geo';
import { quantileBreaks, percentile } from './utils.number';

// ---------------------------------------------------------------------------
// Layer / source IDs
// ---------------------------------------------------------------------------
const PARKING_SOURCE   = 'parking-source';
const FILL_LAYER       = 'parking-fill';
const OUTLINE_LAYER    = 'parking-outline';
const EXEMPT_FILL_LAYER    = 'parking-exempt-fill';
const EXEMPT_OUTLINE_LAYER = 'parking-exempt-outline';

// Region show/hide + boundary-overlay widget (own instance — separate from the parcel map's).
// Only activates when the parking features carry the region field(s) declared in cities.ts.
const parkingJur = createJurisdiction();
const P_OVERLAY_SRC   = 'parking-region-overlays';
const P_OVERLAY_FILL  = 'parking-region-overlays-fill';
const P_OVERLAY_HATCH = 'parking-region-overlays-hatch';
const P_OVERLAY_LINE  = 'parking-region-overlays-line';
const P_OVERLAY_LABEL = 'parking-region-overlays-label';
const overlayCache = new Map<string, any>();
let overlayGen = 0;

// ── Parking category selector ────────────────────────────────────────────
type ParkingCategory = 'all' | 'surface' | 'structure';
// Which parking_type values belong to each category (gas stations / 'excluded'
// are never shown). 'uncertain' rolls up into 'all' only.
const CATEGORY_TYPES: Record<ParkingCategory, string[]> = {
  all: ['surface', 'structure', 'uncertain'],
  surface: ['surface'],
  structure: ['structure'],
};
let selectedCategory: ParkingCategory = 'surface';
let loadedFeatures: GeoJSON.Feature[] = [];
let hasClassification = false;

// Per-city land-value denominators for the "parking share of taxable land value" metric.
// {citywide:{land,nonExemptLand}, groups:{field:{region:{land,nonExemptLand}}}}. Loaded from a
// small parcel-derived sidecar (config.LAND_TOTALS_URL); null when the city has no file yet, in
// which case the % line stays hidden (graceful rollout).
interface LandTotal { land: number; nonExemptLand: number; }
interface LandTotals { citywide: LandTotal; groups: Record<string, Record<string, LandTotal>>; }
let landTotals: LandTotals | null = null;

/** Fetch the land-totals sidecar once per city. Non-fatal: leaves landTotals null on any error. */
async function loadLandTotals(signal: AbortSignal): Promise<void> {
  landTotals = null;
  if (!LAND_TOTALS_URL) return;
  try {
    const res = await fetch(LAND_TOTALS_URL, { signal });
    if (!res.ok) return;
    const j = await res.json();
    if (j && j.citywide) landTotals = j as LandTotals;
  } catch { /* no land-totals file for this city yet -> % line hidden */ }
}

/** Total NON-EXEMPT parcel land value in the current scope: sum over the region widget's visible
 *  regions when active (matching the map + parking numerator), else citywide. 0 when unavailable. */
function nonExemptLandDenominator(): number {
  if (!landTotals) return 0;
  if (parkingJur.isActive()) {
    const field = parkingJur.getActiveField();
    const g = landTotals.groups?.[field];
    if (g) {
      let sum = 0;
      for (const r of parkingJur.getSelected()) sum += g[r]?.nonExemptLand ?? 0;
      return sum;
    }
  }
  return landTotals.citywide?.nonExemptLand ?? 0;
}

function featureType(p: any): string {
  // Legacy parquets without classification: treat every lot as surface.
  return (p?.parking_type as string) || 'surface';
}
function featureInCategory(p: any, cat: ParkingCategory): boolean {
  const t = featureType(p);
  if (t === 'excluded') return false;
  // Parcel-context filter: lots whose land use can't generate parking (bare/vacant
  // lots, materials/rail yards) are flagged `low` and excluded from the headline &
  // default view. Null/absent (legacy parquets, OSM-only) is treated as confident.
  if (p?.context_confidence === 'low') return false;
  return CATEGORY_TYPES[cat].includes(t);
}
// Per-feature display area/value: structures use the gross footprint + land value
// under them; surface/uncertain use the carve-out-adjusted effective surface.
function displayAreaSqft(p: any): number {
  if (featureType(p) === 'structure') return p?.parking_area_sqft ?? 0;
  return p?.surface_area_sqft ?? p?.parking_area_sqft ?? 0;
}
function displayValue(p: any): number {
  if (featureType(p) === 'structure') return p?.estimated_parking_land_value ?? 0;
  const eff = p?.effective_surface_land_value;
  return (eff !== undefined && eff !== null) ? eff : (p?.estimated_parking_land_value ?? 0);
}

interface CategoryStats {
  sqft: number; acres: number; value: number;
  taxableValue: number; exemptValue: number; meanPpsf: number; count: number;
  exemptCount: number;
}
function computeCategoryStats(features: GeoJSON.Feature[], cat: ParkingCategory): CategoryStats {
  let sqft = 0, value = 0, taxableValue = 0, exemptValue = 0, count = 0, exemptCount = 0;
  let ppsfSum = 0, ppsfN = 0;
  // Honor the region widget's visible set so the totals/headline match what the map shows.
  const regionActive = parkingJur.isActive();
  const regionField = regionActive ? parkingJur.getActiveField() : '';
  const visibleRegions = regionActive ? parkingJur.getSelected() : null;
  for (const f of features) {
    const p = f.properties ?? {};
    if (!featureInCategory(p, cat)) continue;
    if (visibleRegions && !visibleRegions.has(p[regionField])) continue;
    const a = displayAreaSqft(p), v = displayValue(p);
    sqft += a; value += v; count++;
    const isEx = p.is_exempt === true || p.is_exempt === 'true';
    if (isEx) { exemptValue += v; exemptCount++; } else taxableValue += v;
    const ppsf = p.land_value_per_sqft;
    if (typeof ppsf === 'number' && ppsf > 0) { ppsfSum += ppsf; ppsfN++; }
  }
  return {
    sqft, acres: sqft / 43560, value, taxableValue, exemptValue,
    meanPpsf: ppsfN ? ppsfSum / ppsfN : 0, count, exemptCount,
  };
}

// YlOrRd ramp (low → high land value)
const RAMP = COLOR_RAMPS['YlOrRd'];   // ['#FFFFB2','#FECC5C','#FD8D3C','#F03B20','#BD0026']

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Per-city currency symbol (e.g. '€' for Tallinn). Defaults to '$'.
const CURRENCY = (CITIES as any)[SELECTED_CITY]?.currencySymbol ?? '$';

// Metric unit helpers (Tallinn etc.). Parking data is denominated per sqft / in acres;
// for metric cities we convert at display: €/m² = €/sqft × 10.7639, ha = ac × 0.404686.
const SQFT_PER_SQM = 10.7639104167;
const AREA_UNIT = METRIC_UNITS ? 'm²' : 'sqft';
/** Format a €/sqft value as €/m² (metric) or €/sqft, with the currency symbol. */
function fmtPpsf(perSqft: number): string {
  const v = METRIC_UNITS ? perSqft * SQFT_PER_SQM : perSqft;
  return `${CURRENCY}${v.toFixed(2)} / ${AREA_UNIT}`;
}

/** Format a currency amount: €1.2M, $450K, €3.2B … */
function fmtCurrency(n: number): string {
  if (!isFinite(n) || n === 0) return `${CURRENCY}0`;
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${CURRENCY}${(n / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${CURRENCY}${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${CURRENCY}${(n / 1e3).toFixed(0)}K`;
  return `${CURRENCY}${n.toFixed(0)}`;
}

/** Format acres with commas and 1 decimal */
function fmtAcres(a: number): string {
  return a.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
}

/** Format a ratio (0–1) as a percent: <1% shows 2 decimals, otherwise 1 decimal. */
function fmtPercent(r: number): string {
  if (!isFinite(r) || r <= 0) return '0%';
  const pct = r * 100;
  return `${pct < 1 ? pct.toFixed(2) : pct.toFixed(1)}%`;
}

// ---------------------------------------------------------------------------
// Loading overlay
// ---------------------------------------------------------------------------
const loadingOverlay = document.getElementById('parking-map-loading-overlay') as HTMLElement | null;
const loadingMessage = document.getElementById('parking-map-loading-message') as HTMLElement | null;
const parkingContent = document.getElementById('parking-content') as HTMLElement | null;
const noDataState = document.getElementById('no-data-state') as HTMLElement | null;
const parkingMapStage = document.getElementById('parking-map')?.parentElement ?? null;
const colorBySelect = document.getElementById('parking-color-by') as HTMLSelectElement | null;
const exemptToggle = document.getElementById('toggle-exempt') as HTMLInputElement | null;

// Metric cities: relabel the static "/ sqft" / "land ft²" copy in the shared markup to metric.
if (METRIC_UNITS) {
  const lvOpt = colorBySelect?.querySelector('option[value="land_value_per_sqft"]');
  if (lvOpt) lvOpt.textContent = 'Land value / m²';
  const ppsfLabel = document.getElementById('pt-ppsf')?.previousElementSibling;
  if (ppsfLabel && /land ft²|sqft/i.test(ppsfLabel.textContent || '')) ppsfLabel.textContent = 'Avg. land value / m²';
}

let parkingMap: maplibregl.Map | null = null;
let parkingInitPromise: Promise<void> | null = null;
let parkingResizeObserver: ResizeObserver | null = null;
let parkingLoadController: AbortController | null = null;
let parkingLoadToken = 0;
let loadedCityKey: string | null = null;
let controlsBound = false;
let currentColorField: ColorField = 'land_value_per_sqft';
let currentColorConfigs: Record<ColorField, { breaks: number[]; min: number; max: number; labelFn: (v: number) => string }> | null = null;

function showLoading(msg = 'Loading parking data…') {
  if (loadingMessage) loadingMessage.textContent = msg;
  loadingOverlay?.classList.add('show');
}

function hideLoading() {
  loadingOverlay?.classList.remove('show');
}

type ColorField = 'land_value_per_sqft' | 'parking_area_sqft' | 'estimated_parking_land_value';

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

function isLoadActive(token: number, controller: AbortController): boolean {
  return parkingLoadToken === token && parkingLoadController === controller && !controller.signal.aborted;
}

function destroyParkingMap() {
  parkingResizeObserver?.disconnect();
  parkingResizeObserver = null;
  parkingMap?.remove();
  parkingMap = null;
}

function applyColorField(field: ColorField) {
  currentColorField = field;
  if (!parkingMap || !currentColorConfigs || !parkingMap.getLayer(FILL_LAYER)) return;
  const cfg = currentColorConfigs[field];
  parkingMap.setPaintProperty(FILL_LAYER, 'fill-color',
    buildColorExpression(field, cfg.breaks, RAMP)
  );
  updateLegend(cfg.breaks, cfg.min, cfg.max, cfg.labelFn);
}

function updateExemptVisibility() {
  if (!parkingMap || !exemptToggle) return;
  const vis = exemptToggle.checked ? 'visible' : 'none';
  if (parkingMap.getLayer(EXEMPT_FILL_LAYER)) parkingMap.setLayoutProperty(EXEMPT_FILL_LAYER, 'visibility', vis);
  if (parkingMap.getLayer(EXEMPT_OUTLINE_LAYER)) parkingMap.setLayoutProperty(EXEMPT_OUTLINE_LAYER, 'visibility', vis);
}

function bindControlsOnce() {
  if (controlsBound) return;
  controlsBound = true;

  if (colorBySelect) {
    currentColorField = colorBySelect.value as ColorField;
    colorBySelect.addEventListener('change', () => {
      applyColorField(colorBySelect.value as ColorField);
    });
  }

  exemptToggle?.addEventListener('change', () => {
    updateExemptVisibility();
  });
}

// ---------------------------------------------------------------------------
// Build MapLibre step-color expression from quantile breaks
// ---------------------------------------------------------------------------
function buildColorExpression(
  field: string,
  breaks: number[],
  ramp: string[],
): maplibregl.ExpressionSpecification {
  // Build a 'step' expression: [ 'step', ['get', field], ramp[0], b0, ramp[1], b1, ... ]
  const expr: any[] = ['step', ['coalesce', ['get', field], 0], ramp[0]];
  for (let i = 0; i < breaks.length; i++) {
    expr.push(breaks[i], ramp[Math.min(i + 1, ramp.length - 1)]);
  }
  return expr as maplibregl.ExpressionSpecification;
}

// ---------------------------------------------------------------------------
// Populate legend bar
// ---------------------------------------------------------------------------
function updateLegend(_breaks: number[], min: number, max: number, labelFn: (v: number) => string) {
  const bar = document.getElementById('legend-bar');
  const minEl = document.getElementById('legend-min');
  const maxEl = document.getElementById('legend-max');
  if (!bar || !minEl || !maxEl) return;

  bar.innerHTML = RAMP.map(color =>
    `<div class="legend-bar-segment" style="background:${color};"></div>`
  ).join('');

  minEl.textContent = labelFn(min);
  maxEl.textContent = labelFn(max);
}

// ---------------------------------------------------------------------------
// Populate totals panel
// ---------------------------------------------------------------------------
const CATEGORY_COPY: Record<ParkingCategory, { areaLabel: string; phrase: string }> = {
  all:       { areaLabel: 'Total parking area',          phrase: 'is devoted to parking' },
  surface:   { areaLabel: 'Total surface parking area',  phrase: 'sits under surface parking' },
  structure: { areaLabel: 'Total parking-structure area', phrase: 'sits under parking structures' },
};

function updateTotalsPanel(s: CategoryStats) {
  const set = (id: string, val: string) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  // The area total and the "value locked in parking" total now live only in the headline blurb
  // ("X acres … worth $Y sits under …"), so they're no longer shown as line-items. The per-sqft
  // label carries its unit ("/ land ft²"), so the value is just the dollar amount.
  // Total value is in the headline blurb and exempt is shown below, so taxable (= total − exempt)
  // is redundant and no longer displayed.
  set('pt-ppsf',          `${CURRENCY}${(METRIC_UNITS ? s.meanPpsf * SQFT_PER_SQM : s.meanPpsf).toFixed(2)}`);
  set('pt-value-exempt',  fmtCurrency(s.exemptValue));

  // "Parking share of taxable land value": non-exempt parking value (current category + visible
  // regions) over the non-exempt parcel land value in the same scope. Hidden when we have no
  // land-totals file for the city or the denominator is unusable.
  const denom = nonExemptLandDenominator();
  const shareLabel = document.getElementById('pt-parking-share-label');
  const shareVal = document.getElementById('pt-parking-share');
  const showShare = landTotals !== null && denom > 0;
  if (shareLabel) shareLabel.style.display = showShare ? '' : 'none';
  if (shareVal) {
    shareVal.style.display = showShare ? '' : 'none';
    if (showShare) shareVal.textContent = fmtPercent(s.taxableValue / denom);
  }

  // Hide the "Show exempt lots" toggle when there's nothing exempt to show — i.e. zero exempt
  // value AND no exempt lots in the current view.
  const noExempt = s.exemptValue === 0 && s.exemptCount === 0;
  const exemptRow = exemptToggle?.closest('.headline-exempt-toggle') as HTMLElement | null;
  if (exemptRow) exemptRow.style.display = noExempt ? 'none' : '';
}

// ---------------------------------------------------------------------------
// Build headline stat text
// ---------------------------------------------------------------------------
function pluralize(s: string): string {
  if (/(s|x|ch|sh)$/i.test(s)) return s + 'es';
  if (/[^aeiou]y$/i.test(s)) return s.replace(/y$/i, 'ies');
  return s + 's';
}

/** Qualifier appended to the blurb when a subset of regions is selected: " in <region>" for one,
 *  " across N <group-noun>" for several. Empty when the widget is off or all regions are shown
 *  (then the headline is the whole-city total, so no qualifier is needed). */
function regionScopeSuffix(): string {
  if (!parkingJur.isActive()) return '';
  const visible = parkingJur.getSelected();
  const total = parkingJur.getRegionCount();
  if (visible.size === total) return '';
  if (visible.size === 1) return ` in ${[...visible][0]}`;
  return ` across ${visible.size} ${pluralize(parkingJur.getActiveLabel().toLowerCase())}`;
}

function updateHeadlineStat(s: CategoryStats, cat: ParkingCategory) {
  const el = document.getElementById('headline-stat-text');
  if (!el) return;
  el.innerHTML =
    `<span class="highlight">${fmtAcres(s.acres)} acres</span> of land ` +
    `worth <span class="highlight">${fmtCurrency(s.value)}</span> ` +
    `${CATEGORY_COPY[cat].phrase}${regionScopeSuffix()}.`;
}

// ---------------------------------------------------------------------------
// Filter the map layers to the selected category + visible regions
// ---------------------------------------------------------------------------
/** Build the combined fill/outline filter: parking-type category (when classified) AND the active
 *  region group's visible regions (when the region widget is active), plus is_exempt for the exempt
 *  layers. Returns null (= no filter) when nothing constrains the layer. */
function buildParkingFilter(cat: ParkingCategory, exemptOnly: boolean): any {
  const parts: any[] = [];
  if (hasClassification) {
    parts.push(['match', ['coalesce', ['get', 'parking_type'], 'surface'],
                CATEGORY_TYPES[cat], true, false]);
    // Drop parcel-context low-confidence lots (bare/vacant/yard FPs); null = confident.
    parts.push(['!=', ['coalesce', ['get', 'context_confidence'], 'high'], 'low']);
  }
  if (parkingJur.isActive()) parts.push(parkingJur.selectedClause());
  if (exemptOnly) parts.push(['==', ['coalesce', ['get', 'is_exempt'], false], true]);
  return parts.length ? ['all', ...parts] : null;
}

function applyParkingFilters() {
  if (!parkingMap) return;
  for (const id of [FILL_LAYER, OUTLINE_LAYER]) {
    if (parkingMap.getLayer(id)) parkingMap.setFilter(id, buildParkingFilter(selectedCategory, false));
  }
  for (const id of [EXEMPT_FILL_LAYER, EXEMPT_OUTLINE_LAYER]) {
    if (parkingMap.getLayer(id)) parkingMap.setFilter(id, buildParkingFilter(selectedCategory, true));
  }
}

// ---------------------------------------------------------------------------
// Region boundary overlays on the parking map: a very light tint + a thick border per region,
// in the region's palette color (no crosshatch — clean outlines read better on the parking map).
// ---------------------------------------------------------------------------
function removeParkingOverlayLayers(map: maplibregl.Map) {
  for (const id of [P_OVERLAY_LABEL, P_OVERLAY_LINE, P_OVERLAY_HATCH, P_OVERLAY_FILL]) {
    if (map.getLayer(id)) map.removeLayer(id);
  }
  if (map.getSource(P_OVERLAY_SRC)) map.removeSource(P_OVERLAY_SRC);
}

async function refreshParkingOverlays() {
  const map = parkingMap;
  if (!map) return;
  const gen = ++overlayGen;
  removeParkingOverlayLayers(map);
  let regions = parkingJur.getOverlayRegions();
  if (!regions.size) return;
  const file = parkingJur.getActiveOverlayUrl();
  if (!file) return;
  const url = `${import.meta.env.BASE_URL}${file}`;
  let gj = overlayCache.get(url);
  if (!gj) {
    try { gj = await (await fetch(url)).json(); } catch { return; }
    // Consistent winding so the inset line-offset goes INWARD for every region (source data has
    // mixed winding) — adjoining regions then each show their own boundary band.
    normalizeWindingInPlace(gj.features || []);
    (gj.features || []).forEach((f: any) => {
      f.properties = f.properties || {};
      const idx = parkingJur.regionColorIndex(f.properties.name);
      f.properties.__ovColor = OVERLAY_PALETTE[idx];
    });
    overlayCache.set(url, gj);
  }
  // Re-read after the fetch — the set / active group may have changed; bail if stale.
  regions = parkingJur.getOverlayRegions();
  if (gen !== overlayGen || !regions.size || parkingJur.getActiveOverlayUrl() !== file) return;
  removeParkingOverlayLayers(map);
  const filter = ['in', ['get', 'name'], ['literal', Array.from(regions)]] as any;
  map.addSource(P_OVERLAY_SRC, { type: 'geojson', data: gj });
  map.addLayer({ id: P_OVERLAY_FILL, type: 'fill', source: P_OVERLAY_SRC, filter,
    paint: { 'fill-color': ['get', '__ovColor'], 'fill-opacity': 0.05 } });
  map.addLayer({ id: P_OVERLAY_LINE, type: 'line', source: P_OVERLAY_SRC, filter,
    layout: { 'line-join': 'round' },
    // Inset by half the width (winding normalized above) so adjoining regions each show their band.
    paint: { 'line-color': ['get', '__ovColor'], 'line-width': 4, 'line-offset': 2, 'line-opacity': 0.9 } });
  if (parkingJur.labelsEnabled()) {
    map.addLayer({ id: P_OVERLAY_LABEL, type: 'symbol', source: P_OVERLAY_SRC, filter,
      layout: { 'text-field': ['get', 'name'], 'text-font': ['Noto Sans Regular'],
        'text-size': 12, 'text-allow-overlap': false, 'text-padding': 2, 'symbol-placement': 'point' },
      paint: { 'text-color': '#1b1b1b', 'text-halo-color': '#ffffff', 'text-halo-width': 1.4 } });
  }
}

/** Configure + mount the region widget for the parking map. No-ops (returns) unless the parking
 *  features carry the region field(s) the active city declares in cities.ts. */
function setupParkingRegionWidget() {
  if (!parkingMap) return;
  const cityDef: any = (CITIES as any)[SELECTED_CITY];
  const groups: any[] = (cityDef?.jurisdictionGroups && cityDef.jurisdictionGroups.length)
    ? cityDef.jurisdictionGroups
    : cityDef?.jurisdictionField
      ? [{ field: cityDef.jurisdictionField, label: 'Region', primary: cityDef.primaryJurisdiction,
           defaultMode: cityDef.primaryJurisdiction ? 'primaryOnly' : 'all' }]
      : [];
  if (!groups.length) return;
  const configured = parkingJur.configure({
    groups,
    features: loadedFeatures,
    // Region visibility affects both the map AND the totals/headline, so recompute the whole UI.
    onChange: refreshCategoryUI,
    onOverlaysChange: refreshParkingOverlays,
  });
  if (!configured) return;
  parkingJur.buildPanel(parkingMap, {
    containerSelector: '.parking-summary',
    cardId: 'parking-jurisdiction-card',
    // The legend/controls card was merged into the hero box, so anchor below the hero.
    insertAfterSelector: '.headline-stat',
  });
  applyParkingFilters();
}

// Recompute every on-screen number + the map filter for the active category.
function refreshCategoryUI() {
  const stats = computeCategoryStats(loadedFeatures, selectedCategory);
  updateTotalsPanel(stats);
  updateHeadlineStat(stats, selectedCategory);
  applyParkingFilters();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function runParkingWorkspace() {
  bindControlsOnce();

  if (loadedCityKey === SELECTED_CITY && parkingMap) {
    parkingContent!.style.display = '';
    noDataState!.style.display = 'none';
    parkingMap.resize();
    return;
  }

  if (parkingInitPromise) {
    return parkingInitPromise;
  }

  const controller = new AbortController();
  parkingLoadController?.abort();
  parkingLoadController = controller;
  const loadToken = ++parkingLoadToken;

  parkingInitPromise = (async () => {
    destroyParkingMap();

    // Load city dictionary (sets CITY_CONFIG)
    await loadDataDictionary();
    if (!isLoadActive(loadToken, controller)) return;

    const cityConfig = getCityConfig();
    const cityName = formatCityLabel(SELECTED_CITY);

    // Update city name in header + headline
    const cityNameEls = document.querySelectorAll('#cityName, #hs-city-name');
    cityNameEls.forEach(el => { el.textContent = cityName; });

    // Update "Back to Map" link to preserve city param
    const backLink = document.getElementById('back-to-map') as HTMLAnchorElement | null;
    if (backLink) {
      backLink.href = `app.html?city=${SELECTED_CITY}`;
    }

    // ── Feature gate check ─────────────────────────────────────────────────
    // Gate 1: VITE_PARKING_ENABLED env var must be true
    // Gate 2: City dictionary must have hasParkingData: true
    const hasParkingData = cityConfig?.hasParkingData === true;

    if (!PARKING_ENABLED || !hasParkingData || !PARKING_DATASET_URL) {
      parkingContent!.style.display = 'none';
      noDataState!.style.display = '';
      loadedCityKey = null;
      return;
    }

    // ── Show content, start loading ────────────────────────────────────────
    parkingContent!.style.display = '';
    noDataState!.style.display = 'none';
    showLoading('Fetching parking lot data…');

    try {
      // Totals are computed client-side from the loaded features (so they can be
      // filtered by category), so the pre-computed metadata JSON is not needed here.

      // ── Load parking GeoParquet → GeoJSON ──────────────────────────────
      showLoading('Parsing parking lot geometries…');
      // Dev: prefer a local copy in viz/public/parking/ for the current city if present.
      const parkingUrl = await resolveLocalFirst(LOCAL_PARKING_DATASET_PATH, PARKING_DATASET_URL);
      const asyncBuffer = await urlToAsyncBuffer(
        parkingUrl,
        {},
        controller.signal
      );
      if (!isLoadActive(loadToken, controller)) return;
      const geojson = await toGeoJson({ file: asyncBuffer, compressors }) as GeoJSON.FeatureCollection;
      if (!isLoadActive(loadToken, controller)) return;

      if (!geojson?.features?.length) {
        throw new Error('Parking dataset is empty or could not be parsed.');
      }

      const features = geojson.features;
      // Coerce BigInt props (int64 parquet columns like a stray `__index_level_0__`)
      // to Number — MapLibre throws "Do not know how to serialize a BigInt" when it
      // hands the GeoJSON to its worker, which silently kills the fill layers while
      // the client-side stats (computed below) still render. Mirrors main.ts:938.
      sanitizeFeaturesInPlace(features);
      console.log(`[parking] Loaded ${features.length} parking lot features`);
      loadedFeatures = features;
      hasClassification = features.some(f => f.properties?.parking_type != null);

      // Land-value denominators for the parking-share metric (non-fatal; % line hides if absent).
      await loadLandTotals(controller.signal);
      if (!isLoadActive(loadToken, controller)) return;

      // ── Compute per-field statistics for color expressions ──────────────
      const lvValues    = features.map(f => f.properties?.land_value_per_sqft).filter(v => v > 0) as number[];
      const areaValues  = features.map(f => f.properties?.parking_area_sqft).filter(v => v > 0) as number[];
      const estValues   = features.map(f => f.properties?.estimated_parking_land_value).filter(v => v > 0) as number[];

      const lvBreaks    = quantileBreaks(lvValues, 4);    // 5 bins → 4 breaks
      const areaBreaks  = quantileBreaks(areaValues, 4);
      const estBreaks   = quantileBreaks(estValues, 4);

      const lvMin  = percentile(lvValues, 1);
      const lvMax  = percentile(lvValues, 99);

      // ── Initialize MapLibre map ─────────────────────────────────────────
      showLoading('Rendering map…');

      // Named hash param (#pmap=…) so shared parking links restore the camera.
      // When present at load, the visitor arrived via a share link — skip the
      // fit-to-data below so their camera survives.
      const restoredParkingCamera = /(^#|&)pmap=/.test(window.location.hash);
      const map = new maplibregl.Map({
        container: 'parking-map',
        style: PARKING_BASEMAP_STYLE,
        center: [0, 0],
        zoom: 2,
        hash: 'pmap',
        canvasContextAttributes: { preserveDrawingBuffer: true },
      });
      parkingMap = map;
      parkingResizeObserver = (typeof ResizeObserver !== 'undefined' && parkingMapStage)
        ? new ResizeObserver(() => map.resize())
        : null;
      if (parkingResizeObserver && parkingMapStage) {
        parkingResizeObserver.observe(parkingMapStage);
      }

      map.addControl(new maplibregl.NavigationControl(), 'top-left');
      map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: 'imperial' }), 'bottom-left');

      await new Promise<void>((resolve, reject) => {
        const handleAbort = () => reject(new DOMException('Parking load aborted', 'AbortError'));
        controller.signal.addEventListener('abort', handleAbort, { once: true });
        map.once('load', () => {
          controller.signal.removeEventListener('abort', handleAbort);
          resolve();
        });
      });
      if (!isLoadActive(loadToken, controller)) return;

      // ── Add GeoJSON source ─────────────────────────────────────────────
      map.addSource(PARKING_SOURCE, {
        type: 'geojson',
        data: geojson,
        generateId: true,
        // Parking classification uses Overture building footprints (ODbL) and OSM
        // parking/fuel tags (ODbL). ODbL requires attribution on display.
        attribution: '© OpenStreetMap contributors · Building footprints © Overture Maps Foundation (ODbL)',
      });

    // ── Fill layer (2D flat — NOT fill-extrusion) ──────────────────────
    map.addLayer({
      id: FILL_LAYER,
      type: 'fill',
      source: PARKING_SOURCE,
      paint: {
        'fill-color': buildColorExpression('land_value_per_sqft', lvBreaks, RAMP),
        'fill-opacity': [
          'case',
          ['boolean', ['feature-state', 'hover'], false], 0.9,
          0.72,
        ],
      },
    });

    // ── Outline layer ──────────────────────────────────────────────────
    map.addLayer({
      id: OUTLINE_LAYER,
      type: 'line',
      source: PARKING_SOURCE,
      paint: {
        'line-color': '#1a1a2e',
        'line-width': [
          'interpolate', ['linear'], ['zoom'],
          10, 0.3,
          14, 1.0,
          17, 2.0,
        ],
        'line-opacity': 0.55,
      },
    });

    // ── Exempt lots: grey fill + outline (shown by default, togglable) ─
    map.addLayer({
      id: EXEMPT_FILL_LAYER,
      type: 'fill',
      source: PARKING_SOURCE,
      filter: ['==', ['coalesce', ['get', 'is_exempt'], false], true],
      paint: {
        'fill-color': '#aaaaaa',
        'fill-opacity': [
          'case',
          ['boolean', ['feature-state', 'hover'], false], 0.7,
          0.45,
        ],
      },
    });

    map.addLayer({
      id: EXEMPT_OUTLINE_LAYER,
      type: 'line',
      source: PARKING_SOURCE,
      filter: ['==', ['coalesce', ['get', 'is_exempt'], false], true],
      paint: {
        'line-color': '#666666',
        'line-width': [
          'interpolate', ['linear'], ['zoom'],
          10, 0.3,
          14, 1.0,
          17, 1.5,
        ],
        'line-opacity': 0.4,
      },
    });

    // ── Fit map to data (unless a shared link already set the camera) ──
    const bounds = bbox(geojson);
    if (bounds && !restoredParkingCamera) {
      map.fitBounds(
        [[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
        { padding: 40 }
      );
    }

    // ── Hover state ────────────────────────────────────────────────────
    let hoveredId: number | null = null;
    map.on('mousemove', FILL_LAYER, e => {
      if (!e.features?.length) return;
      if (hoveredId !== null) {
        map.setFeatureState({ source: PARKING_SOURCE, id: hoveredId }, { hover: false });
      }
      hoveredId = e.features[0].id as number;
      map.setFeatureState({ source: PARKING_SOURCE, id: hoveredId }, { hover: true });
      map.getCanvas().style.cursor = 'pointer';
    });
    map.on('mouseleave', FILL_LAYER, () => {
      if (hoveredId !== null) {
        map.setFeatureState({ source: PARKING_SOURCE, id: hoveredId }, { hover: false });
      }
      hoveredId = null;
      map.getCanvas().style.cursor = '';
    });

    // ── Click popup ────────────────────────────────────────────────────
    map.on('click', FILL_LAYER, e => {
      if (!e.features?.length) return;
      const p = e.features[0].properties ?? {};

      const areaSqftRaw = Number(p.parking_area_sqft ?? 0);
      // Popup area: acres (imperial) or hectares (metric); primary figure in sqft or m².
      const areaBig   = METRIC_UNITS ? (areaSqftRaw / 43560 * 0.404686).toFixed(2) : (areaSqftRaw / 43560).toFixed(2);
      const areaBigUnit = METRIC_UNITS ? 'ha' : 'ac';
      const cat       = p.parcel_category ?? '–';
      const ppsf      = Number(p.land_value_per_sqft ?? 0);
      const sqft      = (METRIC_UNITS ? Math.round(areaSqftRaw / SQFT_PER_SQM) : areaSqftRaw).toLocaleString();
      const isExempt  = p.is_exempt === true || p.is_exempt === 'true';

      // Surface-vs-structure classification (present when the classifier has run).
      // Falls back to the legacy "Surface Parking Lot" view for older parquets.
      const TYPE_LABEL: Record<string, string> = {
        surface: 'Surface Parking Lot',
        structure: 'Parking Structure',
        uncertain: 'Parking — type unverified',
        excluded: 'Gas Station',
      };
      const SOURCE_LABEL: Record<string, string> = {
        osm_tag: 'OpenStreetMap tag',
        overture_class: 'Overture building class',
        building_overlap: 'building-footprint overlap',
        osm_fuel: 'OpenStreetMap gas station',
      };
      const ptype = p.parking_type as string | undefined;
      const title = (ptype && TYPE_LABEL[ptype]) || 'Surface Parking Lot';
      // Value: pro-rated effective surface land value when classified, else legacy.
      const hasEff = p.effective_surface_land_value !== undefined && p.effective_surface_land_value !== null;
      const estValue = fmtCurrency((hasEff ? p.effective_surface_land_value : p.estimated_parking_land_value) ?? 0);

      const exemptBadge = isExempt
        ? `<div style="display:inline-block; background:#eee; color:#555; border-radius:3px; padding:1px 6px; font-size:11px; font-weight:600; margin-bottom:6px;">EXEMPT LAND</div><br>`
        : '';

      // Caption: "high confidence · OpenStreetMap tag"
      const caption = ptype
        ? `<div style="color:#888; font-size:11px; margin-bottom:6px;">${p.confidence ?? ''} confidence · ${SOURCE_LABEL[p.classification_source as string] ?? p.classification_source ?? ''}</div>`
        : '';

      // For structures/gas the surface land is developed — say so instead of "$0".
      const valueRow = (ptype === 'structure' || ptype === 'excluded')
        ? `<tr><td style="color:#666; padding: 2px 8px 2px 0;">Surface land value</td>
             <td style="text-align:right; font-weight:600; color:#888;">none (developed)</td></tr>`
        : `<tr><td style="color:#666; padding: 2px 8px 2px 0;">${hasEff ? 'Surface land value' : 'Est. land value'}</td>
             <td style="text-align:right; font-weight:600; color:${isExempt ? '#888' : '#BD0026'};">${estValue}</td></tr>`;

      new maplibregl.Popup({ maxWidth: '320px', closeButton: true })
        .setLngLat(e.lngLat)
        .setHTML(`
          <div style="font-family: system-ui, sans-serif; font-size: 13px; line-height: 1.5;">
            <div style="font-weight: 600; margin-bottom: 2px; font-size: 14px;">
              ${title}
            </div>
            ${caption}
            ${exemptBadge}
            <table style="border-collapse: collapse; width: 100%;">
              <tr>
                <td style="color:#666; padding: 2px 8px 2px 0;">Parcel category</td>
                <td style="text-align:right; font-weight:500;">${cat}</td>
              </tr>
              <tr>
                <td style="color:#666; padding: 2px 8px 2px 0;">Area</td>
                <td style="text-align:right; font-weight:500;">${sqft} ${AREA_UNIT} (${areaBig} ${areaBigUnit})</td>
              </tr>
              <tr>
                <td style="color:#666; padding: 2px 8px 2px 0;">Land value / ${AREA_UNIT}</td>
                <td style="text-align:right; font-weight:500;">${fmtPpsf(ppsf)}</td>
              </tr>
              ${valueRow}
            </table>
          </div>
        `)
        .addTo(map);
    });

      // ── Color-by radio controls ────────────────────────────────────────
      currentColorConfigs = {
        land_value_per_sqft: {
          breaks: lvBreaks,
          min: lvMin,
          max: lvMax,
          labelFn: v => `${CURRENCY}${(METRIC_UNITS ? v * SQFT_PER_SQM : v).toFixed(0)}/${AREA_UNIT}`,
        },
        parking_area_sqft: {
          breaks: areaBreaks,
          min: percentile(areaValues, 1),
          max: percentile(areaValues, 99),
          labelFn: v => `${(v / 43560).toFixed(1)} ac`,
        },
        estimated_parking_land_value: {
          breaks: estBreaks,
          min: percentile(estValues, 1),
          max: percentile(estValues, 99),
          labelFn: v => fmtCurrency(v),
        },
      };

      // Initialize legend / controls
      applyColorField(currentColorField);
      updateExemptVisibility();

      // ── Category selector + totals panel ────────────────────────────────
      const catSelect = document.getElementById('parking-category-select') as HTMLSelectElement | null;
      if (catSelect) {
        // Hide the selector for legacy datasets that lack classification.
        catSelect.style.display = hasClassification ? '' : 'none';
        if (hasClassification) {
          catSelect.value = selectedCategory;
          catSelect.onchange = () => {
            selectedCategory = (catSelect.value as ParkingCategory) || 'surface';
            refreshCategoryUI();
          };
        }
      }
      refreshCategoryUI();
      setupParkingRegionWidget();

      // Keep the integrated map stable after the surrounding summary/table layout resolves.
      map.resize();
      requestAnimationFrame(() => map.resize());
      window.setTimeout(() => map.resize(), 120);

      loadedCityKey = SELECTED_CITY;
      hideLoading();

    } catch (err) {
      if (isAbortError(err) || !isLoadActive(loadToken, controller)) {
        return;
      }
      hideLoading();
      loadedCityKey = null;
      destroyParkingMap();
      console.error('[parking] Failed to load parking data:', err);
      parkingContent!.style.display = 'none';
      noDataState!.style.display = '';
    }
  })().finally(() => {
    if (parkingLoadController === controller) {
      parkingLoadController = null;
    }
    if (parkingLoadToken === loadToken) {
      parkingInitPromise = null;
      hideLoading();
    }
  });

  return parkingInitPromise;
}

export function getParkingMap(): maplibregl.Map | null {
  return parkingMap;
}

export function resizeParkingWorkspace() {
  parkingMap?.resize();
}

export function cancelParkingWorkspaceLoad() {
  parkingLoadController?.abort();
  parkingLoadController = null;
  parkingInitPromise = null;
  hideLoading();
}

export function initParkingWorkspace() {
  return runParkingWorkspace();
}
