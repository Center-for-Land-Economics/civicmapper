// Imports
import './analytics';
import './design-system.css';
import './components.css';
import 'maplibre-gl/dist/maplibre-gl.css';
import maplibregl from 'maplibre-gl';
import type { Expression } from 'maplibre-gl';
import { toGeoJson } from 'geoparquet';
import { compressors } from 'hyparquet-compressors';
import { Protocol, PMTiles } from 'pmtiles';
import { formatCityLabel } from './cities';

// Local imports
import { BASEMAP_STYLES, SOURCE_ID, LAYER_ID, LAYER_ID_LOW, ERROR_LAYER_ID, HEIGHT_CAP_METERS, HEIGHT_PCTL, COLOR_RAMPS, DEFAULT_RAMP_KEY, LEGACY_DEFAULT_RAMP_KEY, UNIT_TO_METERS, DEV_CATEGORY_FIELD, UNDERUTILIZED_DEFAULTS, ORIG_CATEGORY_FIELD, DEFAULT_DATASET_URL, LOCAL_DATASET_PATH, resolveLocalFirst, HEIGHT_CAPS, getPmtilesUrl, SELECTED_CITY, PARKING_ENABLED, REPORT_ENABLED, PARKING_BASEMAP_STYLE, UNDERUTILIZED_ENABLED, HIDE_REMNANTS, TRANSIT_OVERLAY, METRIC_UNITS } from './config';
import { setupTransitOverlay } from './transit';
import { FIELD_LABELS, FIELD_TOOLTIPS, ALL_FIELDS, DROPDOWN_FIELDS, loadDataDictionary, getCityConfig, cityUsesPmtiles, isCoreField } from './utils.dictionary';
import { CITIES } from './cities';
import { initPerf } from './perf';
import * as jurisdiction from './jurisdiction';
import { init3DPrint, type Print3DContext } from './print3d';
import { sanitizeFeaturesInPlace, urlToAsyncBuffer, type AsyncBuffer } from './utils.sanitize';
import { roundGeometryInPlace, trimPropertiesInPlace, bbox, normalizeWindingInPlace } from './utils.geo';
import { numOrNull, fmt, percentile, quantileBreaks } from './utils.number';

type ParkingModule = typeof import('./parking');

// ---- Diagnostic toggles (OFF by default so dev/prod aren't slowed by always-on instrumentation;
// flip on when investigating). PERF_HUD = the live profiler (perf.ts) — dev-only + opt-in. VERBOSE
// = chatty console logging (incl. the per-render paint log + PMTiles load trace). Enable via URL
// (?perf=1 / ?debug=1) or localStorage (gvw_perf / gvw_debug = '1'). ----
const _qs = (() => { try { return new URLSearchParams(location.search); } catch { return new URLSearchParams(); } })();
const _flag = (urlKey: string, lsKey: string) => {
  try { return _qs.has(urlKey) || localStorage.getItem(lsKey) === '1'; } catch { return _qs.has(urlKey); }
};
const PERF_HUD = import.meta.env.DEV && _flag('perf', 'gvw_perf');
const VERBOSE = _flag('debug', 'gvw_debug');
const vlog: (...a: any[]) => void = VERBOSE ? console.log.bind(console) : () => {};

/* ---------------- PMTiles Protocol Registration ----------------- */

// Register PMTiles protocol handler for MapLibre
const pmtilesProtocol = new Protocol();
maplibregl.addProtocol('pmtiles', pmtilesProtocol.tile);

/* ---------------- Map Bootstrap ----------------- */


// Render at the device's NATIVE resolution. Supersampling (2–3x) tanked pan FPS, and switching
// pixel ratio at runtime forces a framebuffer realloc (flash) that also interrupts the drag — so
// we pick one ratio and keep it. Native is already crisp; this is the standard rendering res.
const HQ_PR = window.devicePixelRatio;

const ratioContainer = document.getElementById('map-ratio') as HTMLElement | null;

// Cameras restored from a shared URL. Each map writes a named hash param
// (#map=…&umap=…); when one is present at page load the visitor arrived via a
// share link, so the automatic fit-to-data / cross-map camera syncs must not
// stomp that camera for the rest of the session.
//
// Exception: a hash camera at the hardcoded init center (Houston,
// 29.7604/-95.3698) isn't a real share target — MapLibre writes the default
// camera into the hash on the first resize/moveend, so a reload during the
// initial data load would otherwise pin the map to Houston forever.
function hasRealHashCamera(name: string): boolean {
  const m = window.location.hash.match(new RegExp(`(?:^#|&)${name}=([^&]*)`));
  if (!m) return false;
  return !m[1].includes('29.7604/-95.3698');
}
const restoredCameras = {
  map: hasRealHashCamera('map'),
  umap: hasRealHashCamera('umap'),
};

const map = new maplibregl.Map({
  container: 'map',
  // Default to OpenStreetMap; fallback style handled elsewhere
  style: BASEMAP_STYLES['OpenStreetMap'],
  center: [-95.3698, 29.7604],
  zoom: 10,
  pitch: 45,
  bearing: -20,
  // Named hash so the Underused map can carry its own camera in the same URL
  // (share links restore both). URL format: #map=zoom/lat/lng/bearing/pitch.
  hash: 'map',
  // Keep the WebGL back buffer so the PDF report can snapshot the canvas.
  canvasContextAttributes: { preserveDrawingBuffer: true },

  // supersample: render at higher internal resolution (smooth lines)
  pixelRatio: HQ_PR
});
map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-left');
map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');

// Dev-only: expose the map + a goto helper for console debugging (e.g. flying to a
// specific parcel). Stripped from production builds.
if (import.meta.env.DEV) {
  (window as any).map = map;
  (window as any).goto = (lon: number, lat: number, zoom = 19) =>
    map.flyTo({ center: [lon, lat], zoom });
}

// Secondary maps (Underutilized, Ratio)
const mapUnder = new maplibregl.Map({
  container: 'map-under',
  style: PARKING_BASEMAP_STYLE,
  center: [-95.3698, 29.7604],
  zoom: 10,
  pitch: 0,
  bearing: 0,
  hash: 'umap',
  canvasContextAttributes: { preserveDrawingBuffer: true },
  pixelRatio: HQ_PR
});
mapUnder.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), 'top-left');
mapUnder.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');
mapUnder.dragRotate.disable();
mapUnder.touchZoomRotate.disableRotation();

// The Vacant & Underdeveloped view shouldn't show building footprints or greenspace — both
// distract from the vacancy signal. The Voyager basemap is a vector style, so we can drop
// those layers outright (raster basemaps bake them into the tile image and can't be cleaned
// up this way). Buildings live on the "building" source-layer (fill + "building-top" depth);
// greenspace is the green park/vegetation fill on the "park" and "landcover" source-layers
// plus the green cemetery/stadium "landuse" layer (residential landuse is beige — keep it).
mapUnder.on('load', () => {
  for (const layer of mapUnder.getStyle().layers ?? []) {
    const srcLayer = (layer as { 'source-layer'?: string })['source-layer'];
    if (
      srcLayer === 'building' ||
      srcLayer === 'park' ||
      srcLayer === 'landcover' ||
      layer.id === 'landuse'
    ) {
      mapUnder.removeLayer(layer.id);
    }
  }
});

let mapRatio: maplibregl.Map | null = null;
if (ratioContainer) {
  mapRatio = new maplibregl.Map({
    container: ratioContainer,
    style: BASEMAP_STYLES['OpenStreetMap'],
    center: [-95.3698, 29.7604],
    zoom: 10,
    pitch: 45,
    bearing: -20,
    hash: false,
    pixelRatio: HQ_PR
  });
  mapRatio.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-left');
  mapRatio.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');
}


/* ---------------- UI elements ---------------- */


const fieldSelect = document.getElementById('field') as HTMLSelectElement;
const rampSelect = document.getElementById('ramp') as HTMLSelectElement;
const multInput = document.getElementById('mult') as HTMLInputElement;
const unitsSelect = document.getElementById('units') as HTMLSelectElement;
const opacityInput = document.getElementById('opacity') as HTMLInputElement;
const opacityOut = document.getElementById('opacityVal') as HTMLOutputElement
const legendEl = document.getElementById('legend') as HTMLFieldSetElement;
const controlsEl = document.getElementById('controls') as HTMLDivElement;
const settingsBtn = document.getElementById('settingsBtn') as HTMLButtonElement;
const closeControls = document.getElementById('closeControls') as HTMLButtonElement;
const smoothToggle = document.getElementById('smoothToggle') as HTMLButtonElement | null;
const mapDescription = document.getElementById('mapDescription') as HTMLParagraphElement | null;

const expandBtn = document.getElementById('expandBtn') as HTMLButtonElement;
const mapBox = document.getElementById('mapBox') as HTMLDivElement;
const mainHolder = document.getElementById('mapHolder-main') as HTMLDivElement;
const underHolder = document.getElementById('mapHolder-under') as HTMLDivElement;
const parkingSection = document.getElementById('parkingSection') as HTMLElement | null;
const ratioHolder = document.getElementById('mapHolder-ratio') as HTMLDivElement | null;
const mainSection = document.getElementById('mainSection') as HTMLElement;
const underSection = document.getElementById('underSection') as HTMLElement;
// @ts-expect-error TS6133: reserved for future use
const _ratioSection = document.getElementById('ratioSection') as HTMLElement | null;
const analysisShellMain = document.getElementById('analysisShell-main') as HTMLDivElement | null;
const analysisShellUnder = document.getElementById('analysisShell-under') as HTMLDivElement | null;
const tabLandBtn = document.getElementById('tab-land') as HTMLButtonElement | null;
const tabUnderBtn = document.getElementById('tab-underutilized') as HTMLButtonElement | null;
const tabParkingBtn = document.getElementById('tab-parking') as HTMLButtonElement | null;

// The .view-header (city title + tabs) overlays the top of the left menu column; publish its
// measured height as --view-header-h so the sidebars pad below it (robust to the tabs wrapping).
const viewHeaderEl = document.querySelector('.view-header') as HTMLElement | null;
if (viewHeaderEl && typeof ResizeObserver !== 'undefined') {
  const syncViewHeaderH = () => document.documentElement.style.setProperty(
    '--view-header-h', `${Math.ceil(viewHeaderEl.getBoundingClientRect().height)}px`);
  new ResizeObserver(syncViewHeaderH).observe(viewHeaderEl);
  syncViewHeaderH();
}
const categoryFieldset = document.getElementById('categoryFieldset') as HTMLFieldSetElement | null;
const categoryContainer = document.getElementById('categoryFilter') as HTMLDivElement | null;
const scaleFiltered = document.getElementById('scaleFiltered') as HTMLInputElement | null;
const invertHeights = document.getElementById('invertHeights') as HTMLInputElement | null;
const smoothHeights = document.getElementById('smoothHeights') as HTMLInputElement | null;
const smoothHeightsScaleGroup = document.getElementById('smoothHeightsScaleGroup') as HTMLDivElement | null;
const smoothHeightsStrictness = document.getElementById('smoothHeightsStrictness') as HTMLInputElement | null;
const underTotals = document.getElementById('underTotals') as HTMLDivElement;
// Height sliders (bottom-right)
const heightScaleMain = document.getElementById('heightScale') as HTMLInputElement | null;
const heightScaleRatio = document.getElementById('ratioHeightScale') as HTMLInputElement | null;

// 3D/2D mode for the main Land value map. 3D = tilted extrusions (current behavior). 2D = flat,
// north-up overhead (extrusion heights forced to 0, tilt/rotate gestures locked). The choice is
// persisted; `savedPitch` remembers the tilt to restore when 3D is re-enabled (pitch only).
const MODE_3D_KEY = 'gvw_land_mode_3d';
const toggle3DBtn = document.getElementById('toggle3D') as HTMLButtonElement | null;
let is3D = true;
try { is3D = localStorage.getItem(MODE_3D_KEY) !== '2d'; } catch { is3D = true; }
let savedPitch = 45;  // default tilt to restore if none was captured

// Under map controls
const underSettingsBtn = document.getElementById('underSettingsBtn') as HTMLButtonElement;
const underControlsEl = document.getElementById('underControls') as HTMLDivElement;
const underCloseControls = document.getElementById('underCloseControls') as HTMLButtonElement;
const underExpandBtn = document.getElementById('underExpandBtn') as HTMLButtonElement;
const underOpacityInput = document.getElementById('under-opacity') as HTMLInputElement;
const underOpacityOut = document.getElementById('underOpacityVal') as HTMLOutputElement;
const underLegendEl = document.getElementById('underLegend') as HTMLFieldSetElement;
const underCategoryContainer = document.getElementById('underCategoryFilter') as HTMLDivElement;
const origCategorySelect = document.getElementById('origCategorySelect') as HTMLSelectElement | null;
const underOrigCategorySelect = document.getElementById('underOrigCategorySelect') as HTMLSelectElement | null;
const cityNameEl = document.getElementById('cityName') as HTMLSpanElement | null;

// Ratio map controls
const ratioSettingsBtn = document.getElementById('ratioSettingsBtn') as HTMLButtonElement | null;
const ratioControlsEl = document.getElementById('ratioControls') as HTMLDivElement | null;
const ratioCloseControls = document.getElementById('ratioCloseControls') as HTMLButtonElement | null;
const ratioExpandBtn = document.getElementById('ratioExpandBtn') as HTMLButtonElement | null;
const ratioRampSelect = document.getElementById('ratio-ramp') as HTMLSelectElement | null;
const ratioOpacityInput = document.getElementById('ratio-opacity') as HTMLInputElement | null;
const ratioOpacityOut = document.getElementById('ratioOpacityVal') as HTMLOutputElement | null;
const ratioInvertHeights = document.getElementById('ratioInvertHeights') as HTMLInputElement | null;
const ratioMultInput = document.getElementById('ratio-mult') as HTMLInputElement | null;
const ratioLegendEl = document.getElementById('ratioLegend') as HTMLFieldSetElement | null;
const ratioFieldSelect = document.getElementById('ratio-field') as HTMLSelectElement | null;
const ratioOrigCategorySelect = document.getElementById('ratioOrigCategorySelect') as HTMLSelectElement | null;

const mapHelpEls = Array.from(document.querySelectorAll<HTMLDivElement>('.map-help'));
const orbitTipDismissKey = 'gvw_hide_orbit_tip';
const hideOrbitTips = () => {
  for (const el of mapHelpEls) {
    el.style.display = 'none';
  }
};

let orbitTipDismissed = false;
try {
  orbitTipDismissed = localStorage.getItem(orbitTipDismissKey) === '1';
} catch {
  orbitTipDismissed = false;
}

if (orbitTipDismissed) {
  hideOrbitTips();
} else {
  for (const el of mapHelpEls) {
    const closeBtn = el.querySelector<HTMLButtonElement>('.map-help-close');
    if (!closeBtn) continue;
    closeBtn.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      hideOrbitTips();
      try { localStorage.setItem(orbitTipDismissKey, '1'); } catch {}
    });
  }
}

const panelSizeWatchers: Array<{ panel: HTMLDivElement; mapBox: HTMLDivElement }> = [];
const activeFullScreens = new Set<HTMLDivElement>();
const expandRegistry: Array<{ mapBox: HTMLDivElement; toggle: (expanded: boolean) => void }> = [];
let parkingModulePromise: Promise<ParkingModule> | null = null;

function updateBodyScrollLock() {
  if (activeFullScreens.size > 0) {
    document.body.style.overflow = 'hidden';
  } else {
    document.body.style.overflow = '';
  }
}

function updatePanelMaxHeight(panel: HTMLDivElement, mapBoxEl: HTMLDivElement) {
  const { height } = mapBoxEl.getBoundingClientRect();
  if (!height) return;
  const target = Math.max(240, Math.round(height * 0.75));
  panel.style.maxHeight = `${target}px`;
}

window.addEventListener('resize', () => {
  for (const watcher of panelSizeWatchers) {
    if (watcher.panel.style.display !== 'none') {
      updatePanelMaxHeight(watcher.panel, watcher.mapBox);
    }
  }
});
function categoryInputs() {
  return Array.from((categoryContainer || document.createElement('div')).querySelectorAll<HTMLInputElement>('input[type="checkbox"]'));
}
function categoryInputsUnder() {
  return Array.from((underCategoryContainer || document.createElement('div')).querySelectorAll<HTMLInputElement>('input[type="checkbox"]'));
}
categoryContainer?.addEventListener('change', () => {
  applyFilterAndScaling();
  saveSettings(currentTab);
  renderUnderNow();
});
// Apply filters when toggling checkboxes in the Underutilized panel
underCategoryContainer?.addEventListener('change', () => {
  renderUnderNow();
});
origCategorySelect?.addEventListener('change', () => {
  applyFilterAndScaling();
  renderUnderNow();
  renderRatioNow();
});
underOrigCategorySelect?.addEventListener('change', () => { renderUnderNow(); });
ratioOrigCategorySelect?.addEventListener('change', () => { renderRatioNow(); });
scaleFiltered?.addEventListener('change', () => { applyFilterAndScaling(); saveSettings(currentTab); });
invertHeights?.addEventListener('change', () => {
  if (currentTab === 'under') applyFilterAndScaling();
  else computeAndApplyAutoMultiplier('auto', HEIGHT_CAPS.main, HEIGHT_PCTL);
  saveSettings(currentTab);
});
function updateSmoothHeightsUI() {
  const enabled = !!smoothHeights?.checked;
  if (smoothHeightsScaleGroup) smoothHeightsScaleGroup.style.display = enabled ? 'flex' : 'none';
  if (smoothHeightsStrictness) smoothHeightsStrictness.disabled = !enabled;
}
function updateSmoothToggleUI() {
  if (!smoothToggle) return;
  const canShow = !!smoothLandField && !!assessedLandField;
  smoothToggle.style.display = canShow ? 'inline-flex' : 'none';
  smoothToggle.classList.toggle('btn-primary', smoothLandEnabled);
  smoothToggle.classList.toggle('btn-ghost', !smoothLandEnabled);
  smoothToggle.setAttribute('aria-pressed', smoothLandEnabled ? 'true' : 'false');
}
function updateSmoothToggleAvailability(available: string[]) {
  if (!smoothToggle) return;
  const hasSmooth = !!smoothLandField && available.includes(smoothLandField);
  const hasAssessed = !!assessedLandField && available.includes(assessedLandField);
  if (!hasSmooth || !hasAssessed) {
    smoothLandEnabled = false;
    smoothToggle.style.display = 'none';
    return;
  }
  updateSmoothToggleUI();
}
function updateSmoothToggleStateFromField(field: string | null) {
  if (!smoothToggle || !smoothLandField || !assessedLandField) return;
  smoothLandEnabled = field === smoothLandField;
  if (!smoothLandEnabled && field) {
    previousFieldBeforeSmooth = field;
  }
  updateSmoothToggleUI();
}
function updateMapDescriptionText(config: Record<string, any> | null) {
  if (!mapDescription) return;
  let text = 'Map values reflect assessor valuations. By default this map shows land value per square foot. Use Settings to switch to full assessed value or improvements, and to adjust the color ramp and units.';
  if (config?.smoothLandField) {
    text += ' Smoothed land values blend a distance-weighted average of the 10 nearest parcels (75%) with each parcel\'s assessed land ppsf (25%).';
  }
  mapDescription.textContent = text;
}
function handleSmoothThresholdChange() {
  const raw = Number(smoothHeightsStrictness?.value ?? heightSmoothingThreshold);
  const clamped = clampSmoothingThreshold(raw);
  heightSmoothingThreshold = clamped;
  if (smoothHeightsStrictness) smoothHeightsStrictness.value = String(clamped);
  if (heightSmoothingEnabled) {
    computeAndApplyAutoMultiplier('auto', HEIGHT_CAPS.main, HEIGHT_PCTL);
    renderUnderNow();
    renderRatioNow();
  }
  saveSettings(currentTab);
}
smoothHeights?.addEventListener('change', () => {
  heightSmoothingEnabled = !!smoothHeights.checked;
  updateSmoothHeightsUI();
  computeAndApplyAutoMultiplier('auto', HEIGHT_CAPS.main, HEIGHT_PCTL);
  renderUnderNow();
  renderRatioNow();
  saveSettings(currentTab);
});
smoothHeightsStrictness?.addEventListener('input', handleSmoothThresholdChange);
smoothHeightsStrictness?.addEventListener('change', handleSmoothThresholdChange);
updateSmoothHeightsUI();

// The "Original category filter" works per-parcel. On PMTiles cities the low-zoom
// view shows H3 hex aggregates (each hex spans many categories), so the filter
// can't apply there and silently appeared broken. Disable it with a hint until the
// user zooms into real parcels (at/above the city's parcelMinZoom handoff).
function updateOrigCategoryFilterAvailability() {
  // Every control in the Filter box (category filter + height smoothing) only takes effect
  // on real parcels (z >= parcelMinZoom); at hex zoom they're inert. So when zoomed out,
  // hide the whole control group and show ONLY the warning instead of dead controls.
  const parcelMinZoom = Number(getCityConfig()?.parcelMinZoom);
  const tooFarOut = cityUsesPmtiles() && Number.isFinite(parcelMinZoom) && map.getZoom() < parcelMinZoom;
  if (origCategorySelect) origCategorySelect.disabled = tooFarOut;
  const controls = document.getElementById('filterControls');
  const warning = document.getElementById('filterZoomWarning');
  if (controls) controls.style.display = tooFarOut ? 'none' : '';
  if (warning) warning.style.display = tooFarOut ? 'block' : 'none';
}
map.on('zoom', updateOrigCategoryFilterAvailability);
updateOrigCategoryFilterAvailability();

// "Rendering geometry…" notice for the zoom-handoff (and any tile-load) gap. Debounced:
// checkRenderingGap runs queryRenderedFeatures (costly), and `zoom`/`data` fire rapidly while
// panning/zooming — calling it per event was a real source of pan jank.
let _gapTimer: ReturnType<typeof setTimeout> | undefined;
function scheduleRenderingGapCheck() {
  if (_gapTimer) clearTimeout(_gapTimer);
  _gapTimer = setTimeout(() => { _gapTimer = undefined; checkRenderingGap(); }, 200);
}
map.on('zoom', scheduleRenderingGapCheck);
map.on('data', (e: any) => { if (e.sourceId === SOURCE_ID) scheduleRenderingGapCheck(); });
map.on('idle', hideRenderingToast);
smoothToggle?.addEventListener('click', () => {
  if (!smoothLandField || !assessedLandField) return;
  if (!smoothLandEnabled) {
    previousFieldBeforeSmooth = currentField;
    smoothLandEnabled = true;
  } else {
    smoothLandEnabled = false;
  }
  let nextField = smoothLandEnabled ? smoothLandField : previousFieldBeforeSmooth;
  if (!nextField || (!smoothLandEnabled && nextField === smoothLandField)) {
    nextField = assessedLandField;
  }
  if (nextField && fieldSelect.value !== nextField) {
    fieldSelect.value = nextField;
    currentField = nextField;
    if (currentGeoJSON || (cityUsesPmtiles() && pmtilesMetadata)) {
      scheduleUpdate('recomputeAndAutoScale', /*refreshLegend*/ true);
    }
    saveSettings(currentTab);
  }
  updateSmoothToggleUI();
});

function initMapControls(opts: {
  settingsBtn: HTMLButtonElement,
  panelEl: HTMLDivElement,
  closeBtn: HTMLButtonElement,
  expandBtn?: HTMLButtonElement,
  mapBoxEl: HTMLDivElement,
  map: maplibregl.Map,
  getParent?: () => HTMLElement
}) {
  const { settingsBtn, panelEl, closeBtn, expandBtn, mapBoxEl, map } = opts;
  const parent = mapBoxEl.parentElement as HTMLElement;
  const parentGetter = opts.getParent ?? (() => parent);
  const shell = mapBoxEl.closest('.analysis-shell') as HTMLDivElement | null;

  panelSizeWatchers.push({ panel: panelEl, mapBox: mapBoxEl });

  settingsBtn.onclick = () => {
    shell?.classList.add('sidebar-open');
    updatePanelMaxHeight(panelEl, mapBoxEl);
    window.setTimeout(() => map.resize(), 230);
  };
  closeBtn.onclick = () => {
    shell?.classList.remove('sidebar-open');
    window.setTimeout(() => map.resize(), 230);
  };
  if (expandBtn) {
    let expanded = false;
    const updateExpandButton = () => {
      expandBtn.textContent = expanded ? '⤡' : '⤢';
      expandBtn.setAttribute('aria-label', expanded ? 'Exit full screen' : 'Enter full screen');
      expandBtn.title = expanded ? 'Exit full screen (Esc)' : 'Full screen';
      expandBtn.dataset.expanded = expanded ? 'true' : 'false';
    };
    const applyExpandedState = (next: boolean) => {
      if (expanded === next) return;
      expanded = next;
      mapBoxEl.classList.toggle('map-box-expanded', expanded);
      if (expanded) {
        document.body.appendChild(mapBoxEl);
        activeFullScreens.add(mapBoxEl);
      } else {
        parentGetter().appendChild(mapBoxEl);
        activeFullScreens.delete(mapBoxEl);
      }
      updateBodyScrollLock();
      updateExpandButton();
      updatePanelMaxHeight(panelEl, mapBoxEl);
      map.resize();
    };
    expandBtn.onclick = () => {
      applyExpandedState(!expanded);
    };
    updateExpandButton();
    expandRegistry.push({ mapBox: mapBoxEl, toggle: applyExpandedState });
  }
}

// Initialize map control components for each map
initMapControls({ settingsBtn, panelEl: controlsEl, closeBtn: closeControls, expandBtn, mapBoxEl: mapBox, map, getParent: () => mainHolder });
initMapControls({ settingsBtn: underSettingsBtn, panelEl: underControlsEl, closeBtn: underCloseControls, expandBtn: underExpandBtn, mapBoxEl: underHolder.querySelector('.map-box') as HTMLDivElement, map: mapUnder, getParent: () => underHolder });
if (ratioSettingsBtn && ratioControlsEl && ratioCloseControls && ratioExpandBtn && ratioHolder && mapRatio) {
  initMapControls({ settingsBtn: ratioSettingsBtn, panelEl: ratioControlsEl, closeBtn: ratioCloseControls, expandBtn: ratioExpandBtn, mapBoxEl: ratioHolder.querySelector('.map-box') as HTMLDivElement, map: mapRatio });
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    const expandedEntry = expandRegistry.find(({ mapBox }) => mapBox.classList.contains('map-box-expanded'));
    expandedEntry?.toggle(false);
  }
});

type TabKey = 'main' | 'under' | 'ratio' | 'parking';
let currentTab: TabKey = 'main';
let reverseColors = false; // ratio tab uses reversed colors so darkest = tallest
let smoothLandField: string | null = null;
let assessedLandField: string | null = null;
let smoothLandEnabled = false;
let previousFieldBeforeSmooth: string | null = null;

cityNameEl && (cityNameEl.textContent = formatCityLabel(SELECTED_CITY));

function normalizeRampKey(key: string | null | undefined): string {
  if (!key) return DEFAULT_RAMP_KEY;
  if (key === LEGACY_DEFAULT_RAMP_KEY) return DEFAULT_RAMP_KEY;
  return COLOR_RAMPS[key] ? key : DEFAULT_RAMP_KEY;
}

function getRampQuantileBreaks(values: number[], rampKey: string, colorCount: number): number[] {
  const normalized = normalizeRampKey(rampKey);
  if (normalized === DEFAULT_RAMP_KEY && colorCount === 9) {
    const pctStops = [8, 20, 35, 50, 65, 80, 93, 99.5];
    return pctStops.map(p => percentile(values, p)).filter(v => Number.isFinite(v));
  }
  return quantileBreaks(values, colorCount, 1, 99);
}

function getPmtilesFallbackBreaks(lo: number, hi: number, rampKey: string, colorCount: number): number[] {
  const classes = Math.max(2, colorCount);
  if (!(hi > lo)) return [];
  const normalized = normalizeRampKey(rampKey);
  if (normalized === DEFAULT_RAMP_KEY && classes === 9) {
    const ratios = [0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.93, 0.995];
    return ratios.map(r => lo + (hi - lo) * r);
  }
  const step = (hi - lo) / classes;
  const breaks: number[] = [];
  for (let i = 1; i < classes; i++) breaks.push(lo + step * i);
  return breaks;
}

// @ts-expect-error TS6133: reserved for future use
function _holderForTab(tab: TabKey): HTMLDivElement {
  if (tab === 'main') return mainHolder;
  if (tab === 'under') return underHolder;
  return ratioHolder ?? mainHolder;
}

function getUrlView(): 'land' | 'underutilized' | 'parking' {
  const raw = new URLSearchParams(window.location.search).get('view');
  if (raw === 'underutilized') return 'underutilized';
  if (raw === 'parking') return 'parking';
  return 'land';
}

function setUrlView(view: 'land' | 'underutilized' | 'parking') {
  const url = new URL(window.location.href);
  if (view === 'land') url.searchParams.delete('view');
  else url.searchParams.set('view', view);
  window.history.replaceState({}, '', url);
}

/* ---- Share: copy a link that reproduces this view ---- */
// The URL already carries city (?city), tab (?view), and each map's camera
// (named #map= / #umap= hash params), so the current href IS the share link.
function initShareButton() {
  const btn = document.getElementById('btnShare') as HTMLButtonElement | null;
  if (!btn) return;
  const defaultLabel = btn.textContent || 'Share';
  btn.addEventListener('click', async () => {
    const url = window.location.href;
    const cityLabel = (document.getElementById('cityName')?.textContent || 'city').trim();
    const title = `Civic Mapper — ${cityLabel}`;
    if (navigator.share) {
      try {
        await navigator.share({ title, url });
        return;
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        // fall through to clipboard
      }
    }
    try {
      await navigator.clipboard.writeText(url);
      btn.textContent = 'Link copied!';
    } catch {
      btn.textContent = 'Copy failed';
    }
    window.setTimeout(() => { btn.textContent = defaultLabel; }, 1800);
  });
}

function initFunnelCards() {
  const buttons = document.querySelectorAll('.cle-funnel-subscribe');
  if (!buttons.length) return;
  void import('./cle-footer').then(({ openSubscribeModal }) => {
    buttons.forEach((b) => b.addEventListener('click', openSubscribeModal));
  });
}

function initPdfReportButton() {
  const btn = document.getElementById('btnPdfReport') as HTMLButtonElement | null;
  if (!btn) return;

  // Global feature flag: hide the report button entirely when disabled (ship without it, flip
  // VITE_REPORT_ENABLED=true in the deploy workflow to turn it on later).
  if (!REPORT_ENABLED) { btn.style.display = 'none'; return; }

  // Modal elements (see #reportModal in app.html). If any are missing, fall back to running
  // the report directly (button-label progress) so the feature still works.
  const modal = document.getElementById('reportModal');
  const titleEl = document.getElementById('reportModalTitle');
  const descEl = document.getElementById('reportModalDesc');
  const progressEl = document.getElementById('reportModalProgress');
  const spinnerEl = document.getElementById('reportModalSpinner');
  const statusEl = document.getElementById('reportModalStatus');
  const goBtn = document.getElementById('reportGoBtn') as HTMLButtonElement | null;
  const cancelBtn = document.getElementById('reportCancelBtn') as HTMLButtonElement | null;

  let busy = false;  // true while a report is actually generating (locks the screen)

  const ctx = () => {
    let parkingMapGetter: (() => maplibregl.Map | null) = () => null;
    return {
      needParking: PARKING_ENABLED,
      build: async () => {
        const { generateCityReport } = await import('./report-pdf');
        if (PARKING_ENABLED) {
          const parkingModule = await getParkingModule();
          parkingMapGetter = () => parkingModule.getParkingMap();
        }
        return { generateCityReport, opts: {
          map,
          mapUnder,
          getParkingMap: parkingMapGetter,
          setTab: (t: string) => setTab(t === 'main' ? 'main' : t === 'under' ? 'under' : 'parking'),
          getTab: () => currentTab,
          cityLabel: () => (document.getElementById('cityName')?.textContent || formatCityLabel(SELECTED_CITY)).trim(),
          tabsAvailable: () => ({
            under: UNDERUTILIZED_ENABLED && tabUnderBtn?.style.display !== 'none',
            parking: PARKING_ENABLED && tabParkingBtn?.style.display !== 'none',
          }),
        } };
      },
    };
  };

  // ── Fallback: no modal markup → original inline behavior ──────────────────
  if (!modal || !titleEl || !descEl || !progressEl || !statusEl || !goBtn || !cancelBtn) {
    const defaultLabel = btn.textContent || 'Report PDF';
    btn.addEventListener('click', async () => {
      if (busy) return;
      busy = true; btn.disabled = true;
      try {
        const { generateCityReport, opts } = await ctx().build();
        await generateCityReport(opts as any, (msg) => { btn.textContent = msg; });
        btn.textContent = 'Downloaded!';
      } catch (err) {
        console.error('[Report] PDF generation failed:', err);
        btn.textContent = 'Failed — retry?';
      } finally {
        busy = false; btn.disabled = false;
        window.setTimeout(() => { btn.textContent = defaultLabel; }, 2500);
      }
    });
    return;
  }

  // ── Modal-driven flow: inform → block-with-progress → done/error ──────────
  const showIntro = () => {
    titleEl.textContent = 'Generate PDF report';
    descEl.style.display = '';
    progressEl.style.display = 'none';
    goBtn.style.display = ''; goBtn.disabled = false; goBtn.textContent = 'Generate report';
    cancelBtn.style.display = ''; cancelBtn.disabled = false; cancelBtn.textContent = 'Cancel';
  };
  const open = () => { showIntro(); modal.classList.add('show'); };
  const close = () => { if (!busy) modal.classList.remove('show'); };

  btn.addEventListener('click', () => { if (!busy) open(); });
  cancelBtn.addEventListener('click', () => close());
  // Backdrop click closes only when idle (never mid-generation).
  modal.addEventListener('click', (e) => { if (e.target === modal && !busy) close(); });

  goBtn.addEventListener('click', async () => {
    if (busy) return;
    busy = true;
    // → generating: lock the screen, hide the action buttons, show progress.
    titleEl.textContent = 'Generating report…';
    descEl.style.display = 'none';
    progressEl.style.display = 'flex';
    if (spinnerEl) spinnerEl.style.display = '';
    statusEl.textContent = 'Starting…';
    goBtn.style.display = 'none';
    cancelBtn.style.display = 'none';
    try {
      const { generateCityReport, opts } = await ctx().build();
      await generateCityReport(opts as any, (msg) => { statusEl.textContent = msg; });
      // → done
      busy = false;
      titleEl.textContent = 'Report downloaded ✓';
      if (spinnerEl) spinnerEl.style.display = 'none';
      statusEl.textContent = 'Saved to your downloads.';
      cancelBtn.style.display = ''; cancelBtn.textContent = 'Close'; cancelBtn.disabled = false;
      window.setTimeout(() => { if (!busy) modal.classList.remove('show'); }, 2000);
    } catch (err) {
      console.error('[Report] PDF generation failed:', err);
      busy = false;
      titleEl.textContent = 'Report failed';
      if (spinnerEl) spinnerEl.style.display = 'none';
      statusEl.textContent = 'Something went wrong. Please try again.';
      cancelBtn.style.display = ''; cancelBtn.textContent = 'Close'; cancelBtn.disabled = false;
      goBtn.style.display = ''; goBtn.textContent = 'Try again'; goBtn.disabled = false;
    }
  });
}

function getParkingModule(): Promise<ParkingModule> {
  if (!parkingModulePromise) {
    parkingModulePromise = import('./parking');
  }
  return parkingModulePromise;
}

function cancelParkingWorkspaceIfNeeded() {
  if (!parkingModulePromise) return;
  void parkingModulePromise.then((parkingModule) => {
    parkingModule.cancelParkingWorkspaceLoad();
  }).catch(() => {});
}

async function ensureParkingWorkspace() {
  const parkingModule = await getParkingModule();
  await parkingModule.initParkingWorkspace();
  parkingModule.resizeParkingWorkspace();
}

function saveSettings(tab: TabKey) {
  const obj: any = {
    field: fieldSelect.value,
    ramp: rampSelect.value,
    mult: multInput.value,
    units: unitsSelect.value,
    invert: !!invertHeights?.checked,
    smoothHeights: !!smoothHeights?.checked,
    smoothThreshold: heightSmoothingThreshold,
    colorMode: colorScaleSelect?.value,
    colorInvert,                 // "Invert colors" toggle (distinct from `invert` = invertHeights)
    manualFractions,             // Manual scaling handle positions (fractions of the colour domain)
    scaleFiltered: !!scaleFiltered?.checked
    // `categories` deliberately NOT persisted — see loadSettings for why (silent cross-load filter footgun).
  };
  localStorage.setItem(`gvw_settings_${tab}`, JSON.stringify(obj));
}

function loadSettings(tab: TabKey) {
  const raw = localStorage.getItem(`gvw_settings_${tab}`);
  if (!raw) return;
  try {
    const obj = JSON.parse(raw);
    if (obj.field) { fieldSelect.value = obj.field; currentField = obj.field; }
    if (obj.ramp) rampSelect.value = normalizeRampKey(obj.ramp);
    if (obj.mult) multInput.value = obj.mult;
    if (obj.units) unitsSelect.value = obj.units;
    // Opacity intentionally NOT restored — it always defaults to 100% (full). Anything < 100
    // forces depth-sorted alpha blending on the extrusions, a real FPS cost on large cities.
    if (invertHeights) invertHeights.checked = !!obj.invert;
    if (smoothHeights) smoothHeights.checked = !!obj.smoothHeights;
    if (obj.smoothThreshold != null && smoothHeightsStrictness) {
      const clamped = clampSmoothingThreshold(Number(obj.smoothThreshold));
      heightSmoothingThreshold = clamped;
      smoothHeightsStrictness.value = String(clamped);
    }
    if (obj.colorMode) {
      // Migrate legacy scaling values to the new set (linear/quantiles/log).
      let m = obj.colorMode as string;
      if (m === 'continuous') m = 'linear';
      if (m === 'manual') m = 'quantiles';
      if (m === 'linear' || m === 'quantiles' || m === 'log') {
        colorMode = m as ColorMode;
        if (colorScaleSelect) colorScaleSelect.value = m;   // sync the global + the select
      }
    }
    colorInvert = !!obj.colorInvert;
    manualFractions = Array.isArray(obj.manualFractions) && obj.manualFractions.length
      ? obj.manualFractions.map(Number).filter((n: number) => Number.isFinite(n))
      : null;
    if (scaleFiltered) scaleFiltered.checked = !!obj.scaleFiltered;
    // Category filter intentionally NOT restored — like opacity above, it's a transient exploration
    // control, and persisting it is a silent footgun: a saved narrow selection (e.g. ["Vacant"]) hides
    // nearly every parcel on the NEXT load, and across cities the category names don't even match. It
    // always defaults to all-categories-shown (populateCategoryCheckboxes checks every box by default).
    // Unlike `field`/`mult`, which self-correct via auto-selection, a stale category list had no such
    // sanitization and rendered Tallinn as ~537 "Vacant"-only parcels. See saveSettings (no longer
    // writes `categories`).
    heightSmoothingEnabled = smoothHeights?.checked || false;
    updateSmoothHeightsUI();
    updateSmoothToggleStateFromField(currentField);
  } catch {}
}

// With three concurrent maps, we no longer switch sections.

// Loading overlay
const loadingOverlay = document.getElementById('loadingOverlay')!;
const progressEl = document.getElementById('progress')!;
const progressBar = document.getElementById('progressBar') as HTMLDivElement;
const progressMsg = document.getElementById('progressMsg') as HTMLDivElement;

// Color scaling radios
const colorScaleSelect = document.getElementById('colorScale') as HTMLSelectElement | null;

// Color ramp choices
for (const key of Object.keys(COLOR_RAMPS)) {
  if (key === LEGACY_DEFAULT_RAMP_KEY) continue;
  const opt = document.createElement('option'); opt.value = key; opt.textContent = key; rampSelect.appendChild(opt);
}
rampSelect.value = DEFAULT_RAMP_KEY;

// Populate ratio ramp selects
if (ratioRampSelect) {
  for (const key of Object.keys(COLOR_RAMPS)) {
    if (key === LEGACY_DEFAULT_RAMP_KEY) continue;
    const opt = document.createElement('option'); opt.value = key; opt.textContent = key; ratioRampSelect.appendChild(opt);
  }
  ratioRampSelect.value = COLOR_RAMPS['Reds'] ? 'Reds' : 'Magma';
}


/* ---------------- Constants ---------------- */

const FAST_PR = window.devicePixelRatio;                  // normal speed
const HIGH_PR = Math.min(3, window.devicePixelRatio * 2); // 2–3x is a good HQ target


/* ---------------- State ---------------- */


let currentGeoJSON: GeoJSON.FeatureCollection | null = null;
let currentField: string | null = null;
let currentStats: { min: number; max: number } | null = null;
let preferredLandValuePpsfField: string | null = null;
let currentUnderMetadataTotals: NormalizedUnderTotals | null = null;

let normalizationMode: 'asis' | 'perLand' | 'perBuilding' = 'asis';
// Scaling METHOD = the axis that maps a handle's fraction (0..1) to a break value. Handles are
// ALWAYS available; the method just changes the mapping + the natural (even-seed) spacing.
type ColorMode = 'linear' | 'quantiles' | 'log';
let colorMode: ColorMode = 'quantiles';   // default

// User "Invert colors" toggle (distinct from invertHeights and from the ratio tab's reverseColors).
let colorInvert = false;
// Break LEVEL positions as fractions (0..1) on the current scaling axis, sorted ascending, length =
// ramp.length-1. Seeded evenly (= the method's natural spacing); dragging a legend handle rewrites
// them. Because the strip axis IS the method's axis, even fractions are always evenly-spread
// (grabbable) handles regardless of how skewed the data is.
let manualFractions: number[] | null = null;
// Current metric values, needed to map fractions→values on the Quantiles axis (percentile lookup).
// null on PMTiles (metadata only has p1/p99) → Quantiles there uses the baked breaks, else log.
let colorVals: number[] | null = null;

// The ramp with reverse (ratio tab) + user invert applied — the single source of truth so every
// paint/legend path stays consistent. XOR: inverting an already-reversed ratio ramp un-reverses it.
function activeRamp(): string[] {
  let ramp = COLOR_RAMPS[rampSelect.value] || COLOR_RAMPS['Viridis'];
  if ((reverseColors !== colorInvert) && ramp.length) ramp = ramp.slice().reverse();
  return ramp;
}

// Evenly-spaced boundary fractions = the current method's natural scaling (grabbable handles).
function seedManualFractions(need: number): number[] {
  return Array.from({ length: need }, (_, i) => (i + 1) / (need + 1));
}
function isEvenFractions(fr: number[]): boolean {
  return fr.length > 0 && fr.every((v, i) => Math.abs(v - (i + 1) / (fr.length + 1)) < 1e-6);
}

// Map level fractions → break VALUES using the chosen scaling method as the AXIS:
//   linear    — value = lo + f·(hi-lo)                 (equal value spacing)
//   log       — value = exp(lerp(ln lo, ln hi, f))     (geometric; both ends reachable when skewed)
//   quantiles — value at the f-th percentile           (equal parcel-share spacing; needs the data)
// Even fractions reproduce the method's standard scaling; the handles adjust levels within it.
function breaksFromFractions(fractions: number[], mode: ColorMode): number[] {
  const hi = Math.max(colorDomain?.hi ?? 1, 1e-6);
  const lo = colorDomain?.lo ?? 0;
  const logBreaks = () => {
    const loPos = Math.max(lo, hi * 1e-3, 1e-6);
    const lgLo = Math.log(loPos), lgHi = Math.log(Math.max(hi, loPos * 1.0001));
    return fractions.map(f => Math.exp(lgLo + f * (lgHi - lgLo)));
  };
  if (mode === 'log') return logBreaks();
  if (mode === 'quantiles') {
    if (colorVals && colorVals.length) return fractions.map(f => percentile(colorVals as number[], f * 100));
    // PMTiles (no distribution): use the baked quantile breaks for the default even spacing so the
    // shipped look is unchanged; once the user drags, approximate with a log axis.
    const pre = pmtilesMetadata?.quantileBreaks?.[currentField ?? ''];
    if (pre && pre.length === fractions.length && isEvenFractions(fractions)) return pre.slice();
    return logBreaks();
  }
  return fractions.map(f => lo + f * (hi - lo)); // linear
}

// Seed fractions for the current ramp size (if needed), then map to break values for colorMode.
function currentColorBreaks(colorCount: number): number[] {
  const need = Math.max(1, colorCount - 1);
  if (!manualFractions || manualFractions.length !== need) manualFractions = seedManualFractions(need);
  return breaksFromFractions(manualFractions, colorMode);
}

// For continuous mode we may still show a domain label; optional
let colorDomain: { lo: number; hi: number; label: string } | null = null;

// For quantiles: thresholds between classes
let colorBreaks: number[] | null = null;
// For inverted heights: ranking (quintile) breaks on the raw metric
let heightRankBreaks: number[] | null = null;
const HEIGHT_RANK_BINS = 5; // quintiles for inverted-height ranking
// @ts-expect-error TS6133: reserved for future use
const _HEIGHT_SMOOTH_PCTL = 99.5; // cap rare spikes when smoothing is enabled
const LOW_ZOOM_FADE_START = 9;
const LOW_ZOOM_FADE_END = 12;
const UNDER_FILL_LAYER = 'gp-under-fill';
const UNDER_OUTLINE_LAYER = 'gp-under-outline';
// Lightest stop is the visibility FLOOR: an under-category parcel must never blend into the
// basemap. The old ramps started near-white (#fff3e8 etc.), so the lowest value bucket — and the
// PMTiles path's empty-breaks case, where makeStepColorExpression paints EVERYTHING colors[0] —
// disappeared on light basemaps (the Tulsa symptom). Floors are now clearly saturated tones while
// the upper stops keep the high-value gradation.
const UNDER_CATEGORY_RAMPS: Record<string, string[]> = {
  Vacant: ['#fdba74', '#f97316', '#ea580c', '#9a3412'],
  'Parking Lot': ['#5eead4', '#14b8a6', '#0d9488', '#115e59'],
  Underdeveloped: ['#93c5fd', '#3b82f6', '#2563eb', '#1e3a8a']
};
let availableUnderCategories: string[] = [];

function getLowZoomFadeConfig() {
  const config = getCityConfig();
  const fadeStart = Number(config?.lowZoomFadeStart);
  const fadeEnd = Number(config?.lowZoomFadeEnd);
  const opacityMultiplier = Number(config?.lowZoomOpacityMultiplier);

  return {
    fadeStart: Number.isFinite(fadeStart) ? fadeStart : LOW_ZOOM_FADE_START,
    fadeEnd: Number.isFinite(fadeEnd) ? fadeEnd : LOW_ZOOM_FADE_END,
    opacityMultiplier: Number.isFinite(opacityMultiplier) ? opacityMultiplier : 1
  };
}

// staged loading
let lastAsyncBuffer: AsyncBuffer | null = null;
let cancelRequested = false;

// size identification
let landSizeField: string | null = null;
let bldgSizeField: string | null = null;

// Non-blocking "Geometry is rendering..." toast
let renderToastEl: HTMLDivElement | null = null;
let dotsTimer: number | null = null;

type QualityMode = 'fast' | 'high';
let _qualityMode: QualityMode = 'fast';   // current render quality (toggled by #btn-quality)


// --- popup state ---
let activePopup: maplibregl.Popup | null = null;
let lastPicked: { props: Record<string, any>, lngLat: maplibregl.LngLatLike } | null = null;

type UpdateMode = 'applyOnly' | 'recomputeAndAutoScale';

let _updTimer: number | null = null;
let _pendingMode: UpdateMode = 'applyOnly';
let _pendingRefreshLegend = false;

type MetricUnitKey = 'centimeters' | 'meters' | 'kilometers';

// moved above with TabKey declaration

// Additional height scale factors (0..1), controlled by sliders
const SMOOTH_THRESHOLD_MIN = 2;
const SMOOTH_THRESHOLD_MAX = 15;
const SMOOTH_THRESHOLD_DEFAULT = 6;

let heightFactorMain = 1;
let heightFactorRatio = 1;
let heightSmoothingEnabled = smoothHeights?.checked || false;
let heightSmoothingThreshold = clampSmoothingThreshold(Number(smoothHeightsStrictness?.value ?? SMOOTH_THRESHOLD_DEFAULT));
let heightSmoothingCap: number | null = null;

if (smoothHeightsStrictness) smoothHeightsStrictness.value = String(heightSmoothingThreshold);

/* ---------------- FUNCTIONS ----------------- */


function ensureRenderToast() {
  if (renderToastEl) return;
  renderToastEl = document.createElement('div');
  renderToastEl.style.cssText = `
    position:absolute; top:12px; left:50%; transform:translateX(-50%);
    background:#111; color:#fff; padding:6px 10px; border-radius:999px;
    font-size:12px; opacity:0; transition:opacity .2s; z-index:25; pointer-events:none;
  `;
  renderToastEl.textContent = 'Geometry is rendering...';
  document.body.append(renderToastEl);
}

let renderingToastShown = false;
function showRenderingToast(msg = 'Rendering geometry') {
  if (renderingToastShown) return;  // idempotent — don't restart the dots animation
  renderingToastShown = true;
  ensureRenderToast();
  let i = 0;
  if (dotsTimer) { clearInterval(dotsTimer); dotsTimer = null; }
  renderToastEl!.style.opacity = '0.92';
  renderToastEl!.textContent = msg;
  dotsTimer = window.setInterval(() => {
    i = (i + 1) % 4;
    renderToastEl!.textContent = `${msg}${'.'.repeat(i)}`;
  }, 400);
}

function hideRenderingToast() {
  renderingToastShown = false;
  if (dotsTimer) { clearInterval(dotsTimer); dotsTimer = null; }
  if (renderToastEl) renderToastEl.style.opacity = '0';
}

// Detect the zoom-handoff gap: at parcelMinZoom the hexes are cut (maxzoom) and the real
// parcel tiles (minzoom) may not have loaded yet, leaving a blank frame. More generally,
// whenever the layer that SHOULD be visible at this zoom has no rendered features while
// tiles are still loading, flag it with a "Rendering geometry…" notice.
function checkRenderingGap() {
  if (!cityUsesPmtiles()) { hideRenderingToast(); return; }
  const pmz = Number(getCityConfig()?.parcelMinZoom);
  const activeLayer = (Number.isFinite(pmz) && map.getZoom() >= pmz) ? LAYER_ID : LAYER_ID_LOW;
  if (!map.getLayer(activeLayer)) { hideRenderingToast(); return; }
  let has = 0;
  try { has = map.queryRenderedFeatures({ layers: [activeLayer] }).length; } catch {}
  // areTilesLoaded() guards against false positives in genuinely empty areas (nothing to load).
  if (has === 0 && !map.areTilesLoaded()) showRenderingToast();
  else hideRenderingToast();
}

// @ts-expect-error TS6133: reserved for future use
function _awaitFirstRenderedFeature() {
  // poll one frame at a time; hide toast when the first extrusion is visible
  let tries = 0;
  const maxTries = 600; // ~10s at 60fps
  const tick = () => {
    tries++;
    if (!map.getLayer(LAYER_ID)) { if (tries < maxTries) return requestAnimationFrame(tick); else return hideRenderingToast(); }
    const feats = map.queryRenderedFeatures({ layers: [LAYER_ID] });
    if (feats && feats.length > 0) {
      hideRenderingToast();
    } else if (tries < maxTries) {
      requestAnimationFrame(tick);
    } else {
      hideRenderingToast();
    }
  };
  requestAnimationFrame(tick);
}



function tokenizeName(name: string): string[] {
  return name.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}

// score lower = better
export function scoreValueField(name: string): number {
  const tokens = tokenizeName(name);

  // Category ranking (lower is better)
  const has = (re: RegExp) => tokens.some(t => re.test(t));

  const isLand     = has(/^land$/);
  const isPropLike = has(/^property$/) || has(/^market$/) || has(/^total$/);
  const isBldgLike = has(/^building$/) || has(/^bldg$/) || has(/^impr/) || has(/^improve/);

  let catRank = 3;                // default "other"
  if (isLand)        catRank = 0; // best
  else if (isPropLike) catRank = 1;
  else if (isBldgLike) catRank = 2;

  // Start with category weight
  let score = catRank * 100;

  // Bonus for containing "valu" (as in "value" or "valuation")
  const hasValue = tokens.includes('valu') || /valu/i.test(name);
  if (hasValue) score -= 20;

  // Gentle tie-breakers (keep small so they don't swamp category/bonus)
  // Fewer tokens and shorter total name are better.
  score += tokens.length * 0.5;
  score += Math.min(20, name.length / 50); // tiny nudge for very long names

  return score;
}

function autoPickMainField(fields: string[]): string | undefined {
  let best: string | undefined = undefined;
  let bestScore = Number.POSITIVE_INFINITY;
  for (const f of fields) {
    const s = scoreValueField(f);
    if (s < bestScore) {
      bestScore = s;
      best = f;
    }
  }
  return best;
}

/* ---------------- Loading overlay helpers ---------------- */
function showLoading(msg = 'Parsing GeoParquet…', determinate = false) {
  cancelRequested = false;
  progressMsg.textContent = msg;
  progressEl.classList.toggle('indeterminate', !determinate);
  progressBar.style.width = determinate ? '0%' : '30%';
  loadingOverlay.classList.add('show');
}
function hideLoading() { loadingOverlay.classList.remove('show'); }
// After a large GeoJSON source is handed to the map, geojson-vt indexes it in a worker and the
// extrusions paint only once that finishes — several seconds for a big raw-GeoParquet city (Oslo
// ~72k parcels). Keep the loading overlay up (with a "Rendering…" message) until the map actually
// goes idle (all tiles painted) instead of hiding it the instant parsing returns, which left a
// blank map during indexing. A timeout fallback guards against an 'idle' that never fires.
function hideLoadingAfterRender(msg = 'Rendering parcels…') {
  progressMsg.textContent = msg;
  progressEl.classList.add('indeterminate');
  progressBar.style.width = '75%';
  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    map.off('idle', finish);
    clearTimeout(timer);
    hideLoading();
  };
  const timer = setTimeout(finish, 15000);
  map.on('idle', finish);
}
(document.getElementById('btnCancelLoading') as HTMLButtonElement).onclick = () => {
  cancelRequested = true;
  hideLoading();
  clearData();
};

/* ---------------- Load selected columns (+ geometry) ---------------- */
async function loadSelectedColumns() {
  if (!lastAsyncBuffer) return;
  showLoading('Reading geometry + fields…');

  try {
    const result: any = await toGeoJson({ file: lastAsyncBuffer, compressors });
    if (cancelRequested) return;

    const fc: GeoJSON.FeatureCollection | undefined =
      result?.type === 'FeatureCollection' ? result : result?.geojson;
    if (!fc?.features) throw new Error('Parser returned no FeatureCollection.');

    // Log geometry type distribution for debugging
    try {
      const typeCounts: Record<string, number> = {};
      for (const f of fc.features) {
        const t = (f as any)?.geometry?.type ?? 'null';
        typeCounts[t] = (typeCounts[t] || 0) + 1;
      }
      console.log('[GeoParquet] Parsed FeatureCollection:', {
        totalFeatures: fc.features?.length ?? 0,
        geometryTypes: typeCounts
      });
    } catch {}

    // Accept any geometry whose type name includes 'Polygon' (case-insensitive),
    // to handle potential Z/M variants.
    let features = fc.features.filter(f => {
      const t = (f as any)?.geometry?.type as string | undefined;
      return !!t && /polygon/i.test(t);
    });
    console.log('[GeoParquet] Polygon-like feature count:', { polygonFeatures: features.length });
    if (features.length === 0) throw new Error('No Polygon-like features found (expect Polygon/MultiPolygon).');

    // Normalize field names across cities before validations
    // Map Syracuse/Bellingham fields to expected South Bend-style keys if needed
    try {
      const hasRealImprov = features[0]?.properties?.hasOwnProperty('REALIMPROV');
      const hasRealLand = features[0]?.properties?.hasOwnProperty('REALLANDVA');
      const hasImprov = features[0]?.properties?.hasOwnProperty('improvement_value');
      const hasLand = features[0]?.properties?.hasOwnProperty('current_full_land_value') || features[0]?.properties?.hasOwnProperty('land_value');
      const hasImprovSqft = features[0]?.properties?.hasOwnProperty('improvement_value_per_sqft');
      const hasLandSqft = features[0]?.properties?.hasOwnProperty('land_value_per_sqft');
      if ((!hasRealImprov && hasImprov) || (!hasRealLand && hasLand)) {
        for (const f of features) {
          const p = (f.properties || {}) as Record<string, any>;
          if (hasImprov && !p.hasOwnProperty('REALIMPROV') && p.improvement_value !== undefined) p.REALIMPROV = p.improvement_value;

          const landVal = p.current_full_land_value ?? p.land_value;
          if (hasLand && !p.hasOwnProperty('REALLANDVA') && landVal !== undefined) p.REALLANDVA = landVal;

          if (hasImprovSqft && !p.hasOwnProperty('REALIMPROV_per_sqft') && p.improvement_value_per_sqft !== undefined) p.REALIMPROV_per_sqft = p.improvement_value_per_sqft;
          if (hasLandSqft && !p.hasOwnProperty('REALLANDVA_per_sqft') && p.land_value_per_sqft !== undefined) p.REALLANDVA_per_sqft = p.land_value_per_sqft;
        }
      }
    } catch {}

    // Check for required fields with fallback names. REALLANDVA (land value) is the only
    // hard requirement — it drives the default land-value metric. REALIMPROV (improvement/
    // building value) is optional: some jurisdictions assess land only and ship no building
    // value (e.g. Estonia/Tallinn, where the cadastre has no building assessment). Downstream
    // reads of REALIMPROV are null-safe (JS `Number(...)||0`, MapLibre `to-number` → 0) and the
    // metric dropdown is built from present fields, so improvement metrics simply don't appear.
    const required = ['REALLANDVA'];
    for (const key of required) {
      if (!features[0]?.properties?.hasOwnProperty(key)) {
        throw new Error(`Required field missing: ${key}`);
      }
    }
    
    // Check for category field (with fallback names for different cities)
    const categoryField = DEV_CATEGORY_FIELD;
    const categoryAlternatives = [
      'property_land_use_refined',
      'property_category_refined',
      'PROPERTY_CATEGORY_REFINED',
      'property_land_use_category',
      'PROPERTY_CATEGORY'
    ];
    
    let hasCategoryField = features[0]?.properties?.hasOwnProperty(categoryField);
    if (!hasCategoryField) {
      // Try to find an alternative category field
      for (const alt of categoryAlternatives) {
        if (features[0]?.properties?.hasOwnProperty(alt)) {
          // Normalize to the expected field name
          for (const f of features) {
            const p = (f.properties || {}) as Record<string, any>;
            if (!p.hasOwnProperty(categoryField) && p[alt] !== undefined) {
              p[categoryField] = p[alt];
            }
          }
          hasCategoryField = true;
          break;
        }
      }
    }
    
    if (!hasCategoryField) {
      throw new Error(`Required field missing: ${categoryField} (also checked alternatives: ${categoryAlternatives.join(', ')})`);
    }

    sanitizeFeaturesInPlace(features);
    const derivedFields = ['TLLDIMPROV', 'IMPR_LAND_RATIO', 'IMPR_LAND_PCT', 'IMPR_PCT_TOTAL', 'LAND_PCT_TOTAL'] as const;
    const shouldComputeDerived = derivedFields.some(
      field => !features[0]?.properties?.hasOwnProperty(field)
    );
    if (shouldComputeDerived) {
      for (const f of features) {
        const p = (f.properties || {}) as Record<string, any>;
        const land = Number(p.REALLANDVA);
        const impr = Number(p.REALIMPROV);
        if (!Number.isFinite(land) || !Number.isFinite(impr)) continue;

        const total = land + impr;
        if (!p.hasOwnProperty('TLLDIMPROV')) p.TLLDIMPROV = total;
        if (land > 0) {
          if (!p.hasOwnProperty('IMPR_LAND_RATIO')) p.IMPR_LAND_RATIO = impr / land;
          if (!p.hasOwnProperty('IMPR_LAND_PCT')) p.IMPR_LAND_PCT = (impr / land) * 100;
        }
        if (total > 0 && !p.hasOwnProperty('IMPR_PCT_TOTAL')) {
          p.IMPR_PCT_TOTAL = (impr / total) * 100;
        }
        if (total > 0 && !p.hasOwnProperty('LAND_PCT_TOTAL')) {
          p.LAND_PCT_TOTAL = (land / total) * 100;
        }
      }
    }

    const keep = new Set<string>([
      'id','ID','fid','FID','name','NAME',
      DEV_CATEGORY_FIELD,
      ORIG_CATEGORY_FIELD,
      // System flags the render logic keys off (remnant hiding, exempt handling) but that are
      // not display fields — they live in HIDDEN_METRIC_FIELDS, not every city's dictionary, so
      // they must be kept explicitly or trimming silently disables the hideRemnants filter.
      'likely_remnant','exemption_flag',
      ...ALL_FIELDS,
      bldgSizeField || '',
      landSizeField || '',
      // Per-parcel surface-parking footprint value/area (Tallinn): the Underused "Parking Lot"
      // total sums these so it matches the footprint-precise Parking-tab number. Kept even though
      // they're not dictionary popup fields.
      'parking_footprint_land_value',
      'parking_footprint_area_sqft',
      // Keep every region-group tag (jurisdiction, council_district, …) so the show/hide/dim
      // treatment works for whichever group the user activates.
      ...(((CITIES[SELECTED_CITY] as any).jurisdictionGroups || []).map((g: any) => g.field)),
      (CITIES[SELECTED_CITY] as any).jurisdictionField || ''
    ]);
    trimPropertiesInPlace(features, keep);

    for (const f of features) roundGeometryInPlace(f);

    if (cancelRequested) return;
    currentGeoJSON = { type: 'FeatureCollection', features };
    currentUnderMetadataTotals = null;
    populateCategoryOptions(currentGeoJSON);
    populateOriginalCategoryOptions(currentGeoJSON);
    updateUnderTotals(currentGeoJSON);
    loadSettings(currentTab);

    // Region de-emphasis with switchable grouping schemes (GeoParquet path: scan features).
    const cityDef = CITIES[SELECTED_CITY] as any;
    if (jurisdiction.configure({
      groups: buildRegionGroupConfigs(cityDef),
      features,
      onChange: () => { applyFilterAndScaling(); applyExtrusion(); updateLandBlurb(); },
      onOverlaysChange: refreshOverlays,
    })) {
      jurisdiction.buildPanel(map);
    }

    // dropdown = predetermined fields (ensure they exist)
    // Build the available list with de-duplication
    const availableSet = new Set<string>();
    for (const k of DROPDOWN_FIELDS) {
      if (features[0]?.properties?.hasOwnProperty(k)) availableSet.add(k);
    }
    let available = Array.from(availableSet);

    // Metric cities show €/m² metrics only — drop the imperial per-sqft variants from the picker
    // (the per_sqm field carries the same information in metric units).
    if (METRIC_UNITS) {
      available = available.filter(f => !/_per_sqft$/.test(f) && f !== 'current_tax_per_sqft');
    }

    // For Denver, filter to only show specific fields (like Spokane)
    if (SELECTED_CITY === 'denver') {
      const denverFields = [
        'land_value_per_sqft',
        'IMPR_LAND_PCT',
        'improvement_value_per_sqft',
        'full_market_value_per_sqft'
      ];
      available = available.filter(f => denverFields.includes(f));
      console.log('[Dataset] Filtered fields for Denver:', available);
    }
    
    populateFieldDropdownFromList(available);
    // Populate per-map field selects with the same list
    if (ratioFieldSelect) {
      ratioFieldSelect.replaceChildren();
      if (!available.length) ratioFieldSelect.append(new Option('— no data —', ''));
      else {
        ratioFieldSelect.append(new Option('— choose —', ''));
        for (const n of available) ratioFieldSelect.append(new Option(FIELD_LABELS[n] ?? n, n));
      }
    }

    // Always auto-select land price per square foot (prefer assessed land field if configured)
    const preferredLandField = assessedLandField && available.includes(assessedLandField)
      ? assessedLandField
      : null;
    const landPricePerSqftField = preferredLandField
      || (METRIC_UNITS ? available.find(f => f === 'land_value_per_sqm') : null)
      || available.find(f => f === 'REALLANDVA_per_sqft' || f === 'land_value_per_sqft')
      || available.find(f => f.includes('land') && (f.includes('per_sqft') || f.includes('per_sqm')));
    preferredLandValuePpsfField = landPricePerSqftField || null;
    currentField = landPricePerSqftField || (autoPickMainField(available) ?? null);
    if (currentField) {
      fieldSelect.value = currentField;
      requestAnimationFrame(fitFieldSelectFont);
      currentStats = computeStatsNormalized(currentGeoJSON, currentField, normalizationMode);
      console.log(`[Dataset] Auto-selected land price per sqft: ${currentField}`);
    }
    updateSmoothToggleAvailability(available);
    updateSmoothToggleStateFromField(currentField);

    // Defaults for per-map field selections
    if (ratioFieldSelect) {
      const hasRatio = !!features[0]?.properties?.hasOwnProperty('IMPR_LAND_RATIO');
      const hasPctRatio = !!features[0]?.properties?.hasOwnProperty('IMPR_LAND_PCT');
      ratioFieldSelect.value = hasPctRatio ? 'IMPR_LAND_PCT' : (hasRatio ? 'IMPR_LAND_RATIO' : (currentField || ''));
    }

    addOrUpdateSourceFor(map, /*withClick*/ true);
    addOrUpdateSourceWhenReady(mapUnder, /*withClick*/ true);
    if (mapRatio) addOrUpdateSourceWhenReady(mapRatio, /*withClick*/ true);

    // PROTOTYPE: the de-emph layer + jurisdiction filter split are established by applyExtrusion
    // (triggered next via scheduleUpdate), which is the single authority for both — see there.

    // auto-multiplier for current normalization mode → p99 = 2km (centimeters)
    scheduleUpdate('recomputeAndAutoScale', /*refreshLegend*/ true);
    renderUnderNow();
    renderRatioNow();

    updateLegend();
    updateLandBlurb();
    fitToDataAll(currentGeoJSON);
    // Keep the overlay up through geojson-vt indexing + first paint (not just parsing) so there's
    // no blank-map gap; it hides once the map goes idle. Cancel already hides it via cancelLoad().
    if (!cancelRequested) hideLoadingAfterRender();
    else hideLoading();
  } catch (err: any) {
    // Bubble up so caller can try the next candidate or present a final error
    console.error('GeoParquet load failed:', err);
    hideLoading();
    throw err;
  }
}

/* ---------------- Map helpers ---------------- */
function ensureErrorLayerFor(m: maplibregl.Map) {
  if (m.getLayer(ERROR_LAYER_ID)) return;
  
  // Check if source is vector (PMTiles) or geojson
  const source = m.getSource(SOURCE_ID);
  const isVectorSource = source && (source as any).type === 'vector';
  
  const layerDef: any = {
    id: ERROR_LAYER_ID,
    type: 'line',
    source: SOURCE_ID,
    paint: {
      'line-color': '#ff3b30',          // red outline
      'line-width': 1.5,
      'line-dasharray': [1, 1.3],
      'line-opacity': 0.9
    }
  };
  
  // Vector sources (PMTiles) require source-layer
  if (isVectorSource) {
    layerDef['source-layer'] = 'parcels';
  }
  
  m.addLayer(layerDef);
  // keep it above extrusions for visibility
  try { m.moveLayer(ERROR_LAYER_ID); } catch {}
}

function updateErrorLayer() {
  if (!map.getSource(SOURCE_ID)) return;
  ensureErrorLayerFor(map);

  let filter: any = ['==', ['literal', 1], 2]; // matches nothing by default

  if (currentField === 'IMPR_LAND_RATIO') {
    filter = ['<=', ['to-number', ['get', 'REALLANDVA']], 0];
  } else if (currentField === 'IMPR_LAND_PCT') {
    filter = ['<=', ['to-number', ['get', 'REALLANDVA']], 0];
  } else if (currentField === 'IMPR_PCT_TOTAL' || currentField === 'LAND_PCT_TOTAL') {
    filter = ['<=', ['+', ['to-number', ['get', 'REALIMPROV']], ['to-number', ['get', 'REALLANDVA']]], 0];
  } else if (normalizationMode === 'perLand' && landSizeField) {
    // land invalid when ≤ 0  (zero not allowed)
    filter = ['<=', ['to-number', ['get', landSizeField]], 0];
  } else if (normalizationMode === 'perBuilding' && bldgSizeField) {
    // building invalid when negative (zero is allowed and not flagged)
    filter = ['<', ['to-number', ['get', bldgSizeField]], 0];
  }

  map.setFilter(ERROR_LAYER_ID, filter);
}
function clearData() {
  if (map.getLayer(LAYER_ID)) map.removeLayer(LAYER_ID);
  if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID);
  currentGeoJSON = null; currentField = null; currentStats = null;
  fieldSelect.replaceChildren(new Option('— no data —', ''));
  updateLegend();
  hideRenderingToast();
}
function addOrUpdateSourceFor(m: maplibregl.Map, withClick = false) {
  const existing = m.getSource(SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
  if (existing) existing.setData(currentGeoJSON as any);
  else {
    // tolerance: 0 disables geojson-vt's Douglas-Peucker simplification. With the default
    // (0.375), small parcels collapse below the per-zoom tolerance at citywide zoom and get
    // dropped entirely — so Tallinn (and other raw-GeoParquet cities) rendered only a handful of
    // large polygons (parks/big lots) when zoomed out. Keeping full geometry is cheap here: these
    // are the <~40k-parcel cities on the GeoJSON path (PMTiles cities use a vector source and are
    // unaffected), and geometry is already coarsened by roundGeometryInPlace on load. buffer: 64
    // trims tile overdraw a bit to offset the extra vertices.
    m.addSource(SOURCE_ID, { type: 'geojson', data: currentGeoJSON as any, tolerance: 0, buffer: 64 });
    if (m === mapUnder) addUnderLayerFor(m, withClick);
    else addExtrusionLayerFor(m, withClick);
  }
  // Ensure initial render occurs once layer exists
  if (m === mapUnder ? m.getLayer(UNDER_FILL_LAYER) : m.getLayer(LAYER_ID)) {
    if (m === mapUnder) renderUnderNow();
    else if (m === mapRatio) renderRatioNow();
  }
}

function addOrUpdateSourceWhenReady(m: maplibregl.Map, withClick = false) {
  if (!currentGeoJSON) return;
  if ((m as any).isStyleLoaded && (m as any).isStyleLoaded()) {
    addOrUpdateSourceFor(m, withClick);
  } else {
    m.once('load', () => addOrUpdateSourceFor(m, withClick));
  }
}

function addExtrusionLayerFor(m: maplibregl.Map, withClick = false) {
  if (m.getLayer(LAYER_ID)) return;
  m.addLayer({
    id: LAYER_ID, type: 'fill-extrusion', source: SOURCE_ID,
    paint: {
      'fill-extrusion-color': '#888',
      'fill-extrusion-height': 0,
      'fill-extrusion-opacity': 1,
      'fill-extrusion-vertical-gradient': false
    }
  });
  // Apply the remnant/category filter at creation. The first GeoParquet paint happens before
  // any applyFilterAndScaling call, so without this a hideRemnants city renders (and lets you
  // click) its sub-500-sqft slivers until the first interaction. withRemnantFilter is a no-op
  // when the city hasn't opted in, and applyFilterAndScaling re-derives the same filter later.
  m.setFilter(LAYER_ID, withRemnantFilter(computeCategoryFilter()) as any);
  if (withClick) {
    m.on('click', LAYER_ID, (e) => {
      const f = e.features?.[0];
      if (!f) return;
      const props = (f.properties || {}) as Record<string, any>;
      showPopup(m, props, e.lngLat);
    });
    m.on('mouseenter', LAYER_ID, () => { m.getCanvas().style.cursor = 'pointer'; });
    m.on('mouseleave', LAYER_ID, () => { m.getCanvas().style.cursor = ''; });
    // Show red dashed error outlines only on the main map
    if (m === map) ensureErrorLayerFor(m);
  }
}

function addUnderLayerFor(m: maplibregl.Map, withClick = false) {
  if (m.getLayer(UNDER_FILL_LAYER)) return;
  m.addLayer({
    id: UNDER_FILL_LAYER,
    type: 'fill',
    source: SOURCE_ID,
    paint: {
      'fill-color': '#94a3b8',
      'fill-opacity': 0.92
    }
  });
  m.addLayer({
    id: UNDER_OUTLINE_LAYER,
    type: 'line',
    source: SOURCE_ID,
    paint: {
      'line-color': 'rgba(15, 23, 42, 0.55)',
      'line-width': ['interpolate', ['linear'], ['zoom'], 9, 0.4, 13, 1.0, 16, 1.6] as any,
      'line-opacity': 0.7
    }
  });
  if (withClick) {
    m.on('click', UNDER_FILL_LAYER, (e) => {
      const f = e.features?.[0];
      if (!f) return;
      const props = (f.properties || {}) as Record<string, any>;
      showPopup(m, props, e.lngLat);
    });
    m.on('mouseenter', UNDER_FILL_LAYER, () => { m.getCanvas().style.cursor = 'pointer'; });
    m.on('mouseleave', UNDER_FILL_LAYER, () => { m.getCanvas().style.cursor = ''; });
  }
}

function showPopup(m: maplibregl.Map, props: Record<string, any>, lngLat: maplibregl.LngLatLike) {
  if (activePopup) activePopup.remove();
  activePopup = new maplibregl.Popup({
    closeButton: true,
    closeOnClick: true,
    maxWidth: '460px'          // ← wider than default 240px
  })
    .setLngLat(lngLat)
    .setHTML(buildPopupHTML(props))
    .addTo(m);
  lastPicked = { props, lngLat };
}

/* --- value expression builder (handles normalization) --- */
// Per-sqft metrics are no longer baked into the tiles (dropped to shrink them) — compute them from
// the raw value field ÷ the carried land_area_acres denominator. The SAME expression works on
// parcels (raw value + parcel area) and hexes (Σvalue + Σarea), giving Σvalue/Σarea per hex.
const PER_SQFT_SRC: Record<string, string> = {
  land_value_per_sqft: 'current_full_land_value',
  improvement_value_per_sqft: 'improvement_value',
  full_market_value_per_sqft: 'full_market_value',
};
function perSqftExpr(field: string): Expression | null {
  const raw = PER_SQFT_SRC[field];
  if (!raw) return null;
  const sqft: Expression = ['*', ['to-number', ['get', 'land_area_acres']], 43560] as any;
  return ['case', ['<=', sqft, 0], 0, ['/', ['to-number', ['get', raw]], sqft]] as any;
}

function buildValueExpression(): Expression {
  if (!currentField) return ['literal', 0] as any;
  let base: Expression;
  if (currentField === 'IMPR_LAND_RATIO') {
    const num: Expression = ['to-number', ['get', 'REALIMPROV']] as any;
    const den: Expression = ['to-number', ['get', 'REALLANDVA']] as any;
    base = ['case', ['<=', den, 0], 0, ['/', num, den]] as any;
  } else if (currentField === 'IMPR_LAND_PCT') {
    const num: Expression = ['to-number', ['get', 'REALIMPROV']] as any;
    const den: Expression = ['to-number', ['get', 'REALLANDVA']] as any;
    // percent 0..∞, but typically 0..several hundred
    base = ['case', ['<=', den, 0], 0, ['*', ['/', num, den], 100]] as any;
  } else if (currentField === 'IMPR_PCT_TOTAL') {
    const num: Expression = ['to-number', ['get', 'REALIMPROV']] as any;
    const land: Expression = ['to-number', ['get', 'REALLANDVA']] as any;
    const den: Expression = ['+', num, land] as any;
    // percent 0..100
    base = ['case', ['<=', den, 0], 0, ['*', ['/', num, den], 100]] as any;
  } else if (currentField === 'LAND_PCT_TOTAL') {
    const land: Expression = ['to-number', ['get', 'REALLANDVA']] as any;
    const impr: Expression = ['to-number', ['get', 'REALIMPROV']] as any;
    const den: Expression = ['+', land, impr] as any;
    // land's share of total value, percent 0..100
    base = ['case', ['<=', den, 0], 0, ['*', ['/', land, den], 100]] as any;
  } else if (PER_SQFT_SRC[currentField]) {
    base = perSqftExpr(currentField)!;
  } else {
    base = ['to-number', ['get', currentField]] as any;
  }

  if (normalizationMode === 'perLand' && landSizeField) {
    const den: Expression = ['to-number', ['get', landSizeField]] as any;
    // Land invalid when ≤ 0 ⇒ height 0 (flat); outline layer will flag it.
    return ['case',
      ['<=', den, 0], 0,
      ['/', base, den]
    ] as any;
  }

  if (normalizationMode === 'perBuilding' && bldgSizeField) {
    const den: Expression = ['to-number', ['get', bldgSizeField]] as any;
    // Building invalid when < 0 ⇒ height 0 (flat) and flagged.
    // Building == 0 is allowed conceptually (no building) but we can't divide by 0 ⇒ also 0 height (not flagged).
    return ['case',
      ['<', den, 0], 0,
      ['==', den, 0], 0,
      ['/', base, den]
    ] as any;
  }

  return base;
}

function buildValueExpressionFor(field: string, mode: 'asis'|'perLand'|'perBuilding'): Expression {
  let base: Expression;
  if (field === 'IMPR_LAND_RATIO') {
    const num: Expression = ['to-number', ['get', 'REALIMPROV']] as any;
    const den: Expression = ['to-number', ['get', 'REALLANDVA']] as any;
    base = ['case', ['<=', den, 0], 0, ['/', num, den]] as any;
  } else if (field === 'IMPR_LAND_PCT') {
    const num: Expression = ['to-number', ['get', 'REALIMPROV']] as any;
    const den: Expression = ['to-number', ['get', 'REALLANDVA']] as any;
    base = ['case', ['<=', den, 0], 0, ['*', ['/', num, den], 100]] as any;
  } else if (field === 'IMPR_PCT_TOTAL') {
    const num: Expression = ['to-number', ['get', 'REALIMPROV']] as any;
    const land: Expression = ['to-number', ['get', 'REALLANDVA']] as any;
    const den: Expression = ['+', num, land] as any;
    base = ['case', ['<=', den, 0], 0, ['*', ['/', num, den], 100]] as any;
  } else if (field === 'LAND_PCT_TOTAL') {
    const land: Expression = ['to-number', ['get', 'REALLANDVA']] as any;
    const impr: Expression = ['to-number', ['get', 'REALIMPROV']] as any;
    const den: Expression = ['+', land, impr] as any;
    base = ['case', ['<=', den, 0], 0, ['*', ['/', land, den], 100]] as any;
  } else if (PER_SQFT_SRC[field]) {
    base = perSqftExpr(field)!;
  } else {
    base = ['to-number', ['get', field]] as any;
  }
  if (mode === 'perLand' && landSizeField) {
    const den: Expression = ['to-number', ['get', landSizeField]] as any;
    return ['case', ['<=', den, 0], 0, ['/', base, den]] as any;
  }
  if (mode === 'perBuilding' && bldgSizeField) {
    const den: Expression = ['to-number', ['get', bldgSizeField]] as any;
    return ['case', ['<', den, 0], 0, ['==', den, 0], 0, ['/', base, den]] as any;
  }
  return base;
}

function buildLowZoomOpacity(baseOpacity: number): Expression {
  const { fadeStart, fadeEnd, opacityMultiplier } = getLowZoomFadeConfig();
  const lowZoomOpacity = Math.max(0, Math.min(1, baseOpacity * opacityMultiplier));
  return ['interpolate', ['linear'], ['zoom'],
    fadeStart, lowZoomOpacity,
    fadeEnd, 0
  ] as any;
}


function applyExtrusion() {
  // For PMTiles, currentGeoJSON is null but we still need to apply extrusion
  const hasData = currentGeoJSON || (cityUsesPmtiles() && pmtilesMetadata);
  if (!hasData || !currentField || !currentStats) return;
  if (!map.getLayer(LAYER_ID)) return;

  const ramp = activeRamp();
  const valueExpr = buildValueExpression();
  const cappedHeightExpr = (heightSmoothingEnabled && heightSmoothingCap != null)
    ? (['min', heightSmoothingCap, valueExpr] as unknown as Expression)
    : valueExpr;

  let colorExpr: Expression;
  if (colorBreaks && colorBreaks.length) {
    // Every scaling method (linear/quantiles/log) produces discrete break levels now.
    colorExpr = makeStepColorExpression(valueExpr, ramp, colorBreaks);
  } else {
    // continuous (keep your existing function or clamped version)
    const nmin = currentStats.min;
    const nmax = currentStats.max;
    const cmin = colorDomain?.lo ?? nmin;
    const cmax = colorDomain?.hi ?? nmax;
    colorExpr = makeColorExpressionFromExpr(valueExpr, ramp, cmin, cmax);
  }


  const rawMult = Number(multInput.value);
  const multiplier = (Number.isFinite(rawMult) ? rawMult : 0) * heightFactorMain;
  const unitFactor = UNIT_TO_METERS[unitsSelect.value as keyof typeof UNIT_TO_METERS] ?? 1;
  // Build the height expression from a given value base. Called twice: with the
  // smoothing-capped value (parcel layer) and with the raw value (low-zoom hexes).
  const buildHeightExpr = (base: Expression): Expression => {
    if (invertHeights?.checked && currentField === 'IMPR_PCT_TOTAL') {
      // simple invert within 0..100 domain
      return ['*', ['-', 100, base] as any, multiplier * unitFactor] as any;
    }
    if (invertHeights?.checked && heightRankBreaks && heightRankBreaks.length) {
      // Use inverted rank (quintiles): highest values => smallest height
      const k = Math.max(2, HEIGHT_RANK_BINS);
      const idxExpr = makeStepIndexExpression(base, heightRankBreaks);
      const denom = (k - 1);
      const rankInv: Expression = ['/', ['-', denom, idxExpr as any], denom] as any; // (k-1 - idx)/(k-1)
      return ['*', rankInv, multiplier * unitFactor] as any;
    }
    if (invertHeights?.checked && currentStats) {
      // Fallback: linear invert when rank breaks not available
      return ['*', ['-', currentStats.max, base] as any, multiplier * unitFactor] as any;
    }
    return ['*', base, multiplier * unitFactor] as any;
  };
  const heightExpr = buildHeightExpr(cappedHeightExpr);
  // 2D mode: flatten every extrusion to ground (height 0). Color/opacity are untouched, so the
  // same parcels/hexes render — just flat. h0() wraps each height expression below.
  const h0 = (h: any): any => (is3D ? h : 0);

  vlog('[applyExtrusion] Setting paint properties:', {
    hasData,
    currentField,
    currentStats,
    multiplier: (Number.isFinite(Number(multInput.value)) ? Number(multInput.value) : 0) * heightFactorMain,
    unitFactor: UNIT_TO_METERS[unitsSelect.value as keyof typeof UNIT_TO_METERS] ?? 1,
    heightExpr: heightExpr,
    valueExpr: valueExpr
  });
  
  map.setPaintProperty(LAYER_ID, 'fill-extrusion-color', colorExpr);
  map.setPaintProperty(LAYER_ID, 'fill-extrusion-height', h0(heightExpr));
  const baseOpacity = parseInt(opacityInput.value) / 100;
  map.setPaintProperty(LAYER_ID, 'fill-extrusion-opacity', baseOpacity);

  if (map.getLayer(LAYER_ID_LOW)) {
    map.setPaintProperty(LAYER_ID_LOW, 'fill-extrusion-color', colorExpr);
    // Extreme-height smoothing is only meaningful at parcel zoom. The low-zoom H3
    // hexes are area-weighted aggregates, so they always use the UNCAPPED height —
    // this is what makes smoothing "kick in" only once you zoom into real parcels.
    map.setPaintProperty(LAYER_ID_LOW, 'fill-extrusion-height',
      h0(cityUsesPmtiles() ? buildHeightExpr(valueExpr) : heightExpr));
    // With a parcel handoff (H3 hexes), the hexes are cut hard at `parcelMinZoom`
    // (layer maxzoom) exactly where real parcels switch on — so they must stay at
    // FULL opacity right up to that cut. The old zoom fade (used by the legacy
    // square-grid aggregate) faded them out early and left a dim gap before the
    // parcels appeared. Only fade when there is no hard handoff.
    const hasHandoff = Number.isFinite(Number(getCityConfig()?.parcelMinZoom));
    map.setPaintProperty(LAYER_ID_LOW, 'fill-extrusion-opacity',
      hasHandoff ? baseOpacity : buildLowZoomOpacity(baseOpacity));
  }

  // Region show/hide: filter the lit layers to the SELECTED regions (non-selected simply don't
  // render — no de-emphasis layer). applyExtrusion is the authority because it runs on every
  // render path including the async PMTiles convergence; setParcelLayerFilter skips these while
  // the treatment is active. The hex layer carries only the dominant region per hex (no category
  // fields), so it gets the region clause alone; the parcel layer ANDs in the category/remnant.
  if (jurisdiction.isActive()) {
    const sel = jurisdiction.selectedClause();
    const catFilter = withRemnantFilter(computeCategoryFilter());
    map.setFilter(LAYER_ID, (catFilter ? ['all', catFilter, sel] : sel) as any);
    if (cityUsesPmtiles() && map.getLayer(LAYER_ID_LOW)) {
      map.setFilter(LAYER_ID_LOW, sel as any);
    }
  }

  // refresh which features are flagged as erroneous for current mode
  updateErrorLayer();

  if (activePopup && lastPicked) {
    activePopup.setHTML(buildPopupHTML(lastPicked.props)).setLngLat(lastPicked.lngLat);
  }
}

function renderMapFor(
  m: maplibregl.Map,
  field: string,
  mode: 'asis'|'perLand'|'perBuilding',
  invert: boolean,
  rampKey: string,
  reverse: boolean,
  opacityOverride?: number,
  multiplierFactor: number = 1,
  legendEl?: HTMLFieldSetElement | null,
  filteredFc?: GeoJSON.FeatureCollection,
  capMeters: number = HEIGHT_CAP_METERS
) {
  const hasData = currentGeoJSON || cityUsesPmtiles();
  if (!hasData) return;
  if (!m.getLayer(LAYER_ID)) return;
  
  // For PMTiles, use metadata statistics
  let min: number, max: number;
  let vals: number[] = [];
  let heightVals: number[] = [];
  let cap: number | null = null;
  
  if (cityUsesPmtiles() && pmtilesMetadata) {
    const fieldStats = pmtilesMetadata.statistics[field];
    if (fieldStats) {
      min = fieldStats.min;
      max = fieldStats.max;
      // Create synthetic values for height smoothing (use min/max range)
      heightVals = [min, max];
      vals = [min, max];
    } else {
      min = 0;
      max = 1;
    }
  } else {
    const src = filteredFc || currentGeoJSON!;
    vals = getNumericValuesNormalized(src, field, mode);
    const smoothing = applyHeightSmoothing(vals);
    heightVals = smoothing.heightVals;
    cap = smoothing.cap;
    if (!vals.length) return;
    min = Infinity;
    max = -Infinity;
    for (const v of vals) { if (v < min) min = v; if (v > max) max = v; }
    if (!(max > min)) { min = 0; max = 1; }
  }

  // For inverted heights, use rank (quintiles) on raw values
  const usePctInvert = invert && field === 'IMPR_PCT_TOTAL';
  const rankBreaks = invert && !usePctInvert ? quantileBreaks(heightVals, HEIGHT_RANK_BINS, 1, 99) : [];
  const k = Math.max(2, HEIGHT_RANK_BINS);
  const denom = (k - 1);
  const toIdx = (v: number) => {
    let i = 0;
    while (i < rankBreaks.length && v >= rankBreaks[i]) i++;
    return i;
  };
  // Height autoscale uses simple invert for percentage, rank-based otherwise
  const scaleValsForHeight = invert
    ? (usePctInvert ? heightVals.map(v => 100 - v) : heightVals.map(v => (denom - toIdx(v)) / denom))
    : heightVals;
  let pVal = percentile(scaleValsForHeight, HEIGHT_PCTL);
  if (!Number.isFinite(pVal) || pVal <= 0) {
    // Fallback: try non-inverted values; if still invalid, use 1
    const alt = percentile(heightVals, HEIGHT_PCTL);
    pVal = (Number.isFinite(alt) && alt > 0) ? alt : 1;
  }
  const heightScale = (capMeters / pVal) * (Number.isFinite(multiplierFactor) ? multiplierFactor : 1);

  let ramp = COLOR_RAMPS[rampKey] || COLOR_RAMPS['Viridis'];
  if (reverse && ramp) ramp = ramp.slice().reverse();

  const valueExpr = buildValueExpressionFor(field, mode);
  const cappedHeightExpr = (heightSmoothingEnabled && cap != null)
    ? (['min', cap, valueExpr] as unknown as Expression)
    : valueExpr;
  let colorExpr: Expression;
  let legendText = '';
  const colorBaseExpr: Expression = invert ? (['-', max, valueExpr] as any) : valueExpr;
  if (colorMode === 'quantiles') {
    // Color breaks follow raw/inverted-linear values to preserve expected ramp
    const colorScaleVals = invert
      ? (usePctInvert ? vals.map(v => 100 - v) : vals.map(v => (max - v)))
      : vals;
    const breaksOnScale = getRampQuantileBreaks(colorScaleVals, rampKey, ramp.length);
    const breaks = breaksOnScale;
    colorExpr = makeStepColorExpression(colorBaseExpr, ramp, breaks);
    const lo = percentile(colorScaleVals, 1);
    const hi = percentile(colorScaleVals, 99);
    const edges = [lo, ...breaksOnScale, hi].map(v => Number(v).toLocaleString()).join(' | ');
    legendText = `Quantiles (p1–p99): ${edges}`;
  } else {
    const colorScaleVals = invert ? vals.map(v => (max - v)) : vals;
    const pLow = percentile(colorScaleVals, 1);
    const pHigh = percentile(colorScaleVals, 99);
    const lo = Number.isFinite(pLow) ? pLow : min;
    const hi = Number.isFinite(pHigh) ? pHigh : max;
    colorExpr = makeColorExpressionFromExpr(colorBaseExpr, ramp, lo, hi);
    legendText = `${lo.toLocaleString()} → ${hi.toLocaleString()}`;
  }

  let heightExpr: Expression;
  if (invert) {
    if (usePctInvert) {
      heightExpr = ['*', ['-', 100, cappedHeightExpr] as any, heightScale] as any;
    } else {
      const idxExpr = makeStepIndexExpression(cappedHeightExpr, rankBreaks);
      const rankInv: Expression = ['/', ['-', denom, idxExpr as any], denom] as any; // (k-1 - idx)/(k-1)
      heightExpr = ['*', rankInv, heightScale] as any;
    }
  } else {
    const heightBase: Expression = cappedHeightExpr;
    heightExpr = ['*', heightBase, heightScale] as any;
  }

  m.setPaintProperty(LAYER_ID, 'fill-extrusion-color', colorExpr);
  m.setPaintProperty(LAYER_ID, 'fill-extrusion-height', heightExpr);
  const op = (typeof opacityOverride === 'number') ? opacityOverride : (parseInt(opacityInput.value) / 100);
  m.setPaintProperty(LAYER_ID, 'fill-extrusion-opacity', op);
  if (m.getLayer(LAYER_ID_LOW)) {
    m.setPaintProperty(LAYER_ID_LOW, 'fill-extrusion-color', colorExpr);
    m.setPaintProperty(LAYER_ID_LOW, 'fill-extrusion-height', heightExpr);
    m.setPaintProperty(LAYER_ID_LOW, 'fill-extrusion-opacity', buildLowZoomOpacity(op));
  }

  if (legendEl) {
    legendEl.replaceChildren();
    if (legendText) {
      const row = document.createElement('div');
      row.style.display = 'flex'; row.style.gap = '6px'; row.style.alignItems = 'center'; row.style.flexWrap = 'wrap';
      const label = document.createElement('div'); label.textContent = 'Legend:'; label.style.fontSize = '12px';
      const meta = document.createElement('div'); meta.className = 'muted'; meta.textContent = legendText;
      row.appendChild(label); row.appendChild(meta);
      legendEl.appendChild(row);
      legendEl.style.display = 'grid';
    } else {
      legendEl.style.display = 'none';
    }
  }
}
function setTab(tab: TabKey) {
  if (tab !== 'parking') {
    cancelParkingWorkspaceIfNeeded();
  }

  currentTab = tab;
  mainSection.classList.toggle('is-active', tab === 'main');
  underSection.classList.toggle('is-active', tab === 'under');
  parkingSection?.classList.toggle('is-active', tab === 'parking');
  tabLandBtn?.classList.toggle('is-active', tab === 'main');
  tabUnderBtn?.classList.toggle('is-active', tab === 'under');
  tabParkingBtn?.classList.toggle('is-active', tab === 'parking');

  if (tab === 'main') {
    setUrlView('land');
    categoryFieldset && (categoryFieldset.style.display = 'none');
    reverseColors = false;
    setParcelLayerFilter(map, null);
    computeAndApplyAutoMultiplier('auto', HEIGHT_CAPS.main, HEIGHT_PCTL);
    applyExtrusion();
    analysisShellMain?.classList.remove('sidebar-open');
    map.resize();
  } else if (tab === 'under') {
    if (!restoredCameras.umap) {
      syncMapView(map, mapUnder);
      mapUnder.setPitch(0);
      mapUnder.setBearing(0);
    }
    setUrlView('underutilized');
    // Do not surface the main map's category filter in this view
    if (categoryFieldset) categoryFieldset.style.display = 'none';
    reverseColors = false;
    const inputs = categoryInputs();
    if (inputs.length && !inputs.some(i => i.checked)) {
      inputs.forEach(i => (i.checked = true));
    }
    applyFilterAndScaling();
    // applyFilterAndScaling only paints the main map; the underutilized map needs its
    // own render or it stays grey with no legend until a category checkbox is toggled.
    renderUnderNow();
    analysisShellUnder?.classList.remove('sidebar-open');
    mapUnder.resize();
  } else if (tab === 'parking') {
    setUrlView('parking');
    analysisShellMain?.classList.remove('sidebar-open');
    analysisShellUnder?.classList.remove('sidebar-open');
    void ensureParkingWorkspace();
  } else if (tab === 'ratio') {
    syncMapView(map, mapRatio);
    if (categoryFieldset) categoryFieldset.style.display = 'none';
    reverseColors = true;            // darkest = tallest when inverted
    if (COLOR_RAMPS['Reds']) rampSelect.value = 'Reds';
    setParcelLayerFilter(map, null);
    computeAndApplyAutoMultiplier('auto', HEIGHT_CAPS.main, HEIGHT_PCTL);
    applyExtrusion();
    // Same as the under tab: the ratio map needs its own render, not just the main map's.
    renderRatioNow();
  }
}

function populateCategoryOptions(fc: GeoJSON.FeatureCollection) {
  if (!categoryContainer) return;
  const vals = new Set<string>();
  for (const f of fc.features) {
    const v = String((f.properties as any)?.[DEV_CATEGORY_FIELD] ?? '').trim();
    if (v) vals.add(v);
  }
  const list = Array.from(vals).sort();
  populateCategoryCheckboxes(categoryContainer, list);
  populateUnderCategoryOptions(list);
}

function populateOriginalCategoryOptions(fc: GeoJSON.FeatureCollection) {
  const vals = new Set<string>();
  for (const f of fc.features) {
    const v = String((f.properties as any)?.[ORIG_CATEGORY_FIELD] ?? '').trim();
    if (v) vals.add(v);
  }
  const list = Array.from(vals).sort();
  const fill = (sel: HTMLSelectElement | null) => {
    if (!sel) return;
    sel.replaceChildren();
    sel.append(new Option('All categories', ''));
    for (const v of list) sel.append(new Option(v, v));
    sel.value = '';
  };
  fill(origCategorySelect);
  fill(underOrigCategorySelect);
  fill(ratioOrigCategorySelect);
}

// Per-city currency symbol (e.g. '€' for Tallinn). Defaults to '$'. Resolved once at
// module load from the selected city so all value formatters render the right symbol.
const CURRENCY = CITIES[SELECTED_CITY]?.currencySymbol ?? '$';

// Metric cities: rewrite the static "per-square-foot" methodology prose to metric. (The
// interactive units — picker, legend, blurb, popups — are handled by the formatters/config.)
if (METRIC_UNITS) {
  document.querySelectorAll('.sidebar-section-copy, .under-shell-copy').forEach(el => {
    const t = el.textContent || '';
    if (/square foot/i.test(t))
      el.textContent = t.replace(/per-square-foot/gi, 'per-square-meter').replace(/square foot/gi, 'square meter');
  });
  // Shared parking-view markup (hidden on the value/underused tabs) carries a static
  // "/ sqft" option label — relabel to metric so the DOM is unit-consistent everywhere.
  const lvOpt = document.querySelector('#parking-color-by option[value="land_value_per_sqft"]');
  if (lvOpt) lvOpt.textContent = 'Land value / m²';
}

function fmtCurrencyRounded(n: number): string {
  const magnitude = Math.abs(n);
  if (magnitude >= 1_000_000_000) return `~${CURRENCY}${(n / 1_000_000_000).toFixed(1).replace(/\.0$/, '')} billion`;
  if (magnitude >= 1_000_000) return `~${CURRENCY}${(n / 1_000_000).toFixed(1).replace(/\.0$/, '')} million`;
  const v = Math.round(n / 1000) * 1000;
  return `~${CURRENCY}${v.toLocaleString()}`;
}

// Metric-unit conversion: 1 m² = 10.7639 ft², so €/m² = €/sqft × 10.7639.
const SQFT_PER_SQM = 10.7639104167;

function fmtPerArea(value: number, unit: string): string {
  if (Math.abs(value) >= 100) return `${CURRENCY}${value.toFixed(0)} / ${unit}`;
  if (Math.abs(value) >= 10) return `${CURRENCY}${value.toFixed(1)} / ${unit}`;
  return `${CURRENCY}${value.toFixed(2)} / ${unit}`;
}

// Format a value already denominated per m² (e.g. land_value_per_sqm).
function fmtCurrencyPerSqm(n: number | null | undefined): string {
  if (!Number.isFinite(Number(n))) return '—';
  return fmtPerArea(Number(n), 'm²');
}

// Format a value denominated per sqft. In metric cities, convert to €/m² for display so
// stray per-sqft displays (underused avg, parking popups) stay consistent with the map.
function fmtCurrencyPerSqft(n: number | null | undefined): string {
  if (!Number.isFinite(Number(n))) return '—';
  const value = Number(n);
  if (METRIC_UNITS) return fmtPerArea(value * SQFT_PER_SQM, 'm²');
  return fmtPerArea(value, 'sqft');
}

// ---- Land-value headline blurb -------------------------------------------------------------
// Totals for the selected metric over the visible region scope. PMTiles cities read per-region
// {acres,land,impr,total} from the bake metadata (groups[field].totals); GeoParquet cities sum
// the in-memory parcels. Blurb stays hidden until totals are available (e.g. before a metadata
// re-upload), so it degrades gracefully.
type ScopeTotals = { acres: number; land: number; impr: number; total: number };

function fmtCompactDollars(n: number): string {
  const a = Math.abs(n);
  if (a >= 1e9) return `${CURRENCY}${(n / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `${CURRENCY}${(n / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${CURRENCY}${Math.round(n / 1e3)}K`;
  return `${CURRENCY}${Math.round(n)}`;
}
function fmtAcresBlurb(a: number): string {
  return a >= 1000 ? Math.round(a).toLocaleString() : a.toLocaleString(undefined, { maximumFractionDigits: 1 });
}
function pluralizeNoun(s: string): string {
  if (/(s|x|ch|sh)$/i.test(s)) return s + 'es';
  if (/[^aeiou]y$/i.test(s)) return s.replace(/y$/i, 'ies');
  return s + 's';
}

function sumFeatureTotals(features: GeoJSON.Feature[]): ScopeTotals {
  let acres = 0, land = 0, impr = 0, total = 0;
  for (const f of features) {
    const p: any = f.properties || {};
    const lv = Number(p.REALLANDVA) || 0;
    const iv = Number(p.REALIMPROV) || 0;
    land += lv; impr += iv;
    total += Number(p.full_market_value) || (lv + iv);
    // Land AREA for the blurb total. Metric cities ship land_area_sqm = physical *geometry* area,
    // which is the correct area to AGGREGATE. Do NOT derive it from value ÷ per-m² for these: e.g.
    // Copenhagen's per-m² uses the ASSESSED area (vurderetAreal, the right per-m² denominator), so
    // value/per-m² = assessed area — which overcounts the physical municipality when summed (roads +
    // fragment properties). Non-metric (US) cities lack land_area_sqm → use the exact per-sqft
    // relationship (sqft = land value / land-value-per-sqft); fall back to a stored acres column.
    const sqm = Number(p.land_area_sqm);
    const ppsf = Number(p.REALLANDVA_per_sqft);
    if (METRIC_UNITS && Number.isFinite(sqm) && sqm > 0) acres += sqm / 4046.8564224;
    else if (ppsf > 0 && lv > 0) acres += lv / ppsf / 43560;
    else { const la = Number(p.land_area_acres); if (Number.isFinite(la)) acres += la; }
  }
  return { acres, land, impr, total };
}

function computeScopeTotals(): ScopeTotals | null {
  if (jurisdiction.isActive()) {
    const field = jurisdiction.getActiveField();
    const visible = jurisdiction.getSelected();
    const totals = pmtilesMetadata?.groups?.[field]?.totals as Record<string, ScopeTotals> | undefined;
    if (cityUsesPmtiles() && totals) {
      const acc: ScopeTotals = { acres: 0, land: 0, impr: 0, total: 0 };
      for (const r of visible) {
        const t = totals[r];
        if (t) { acc.acres += t.acres; acc.land += t.land; acc.impr += t.impr; acc.total += t.total; }
      }
      return acc;
    }
    if (currentGeoJSON) return sumFeatureTotals(currentGeoJSON.features.filter(f => visible.has((f.properties as any)?.[field])));
    return null;
  }
  if (currentGeoJSON) return sumFeatureTotals(currentGeoJSON.features);
  // PMTiles city with no region widget: parcels aren't in browser memory, so use the baked
  // citywide totals — same blurb as if every region were selected.
  if (cityUsesPmtiles() && pmtilesMetadata?.cityTotals) return pmtilesMetadata.cityTotals;
  return null;
}

/** Location clause: "in <region>" for one selected region, "across N <group-noun>" for several,
 *  "in <city>" when no region widget is active. */
function landScopeLabel(): string {
  if (jurisdiction.isActive()) {
    const visible = jurisdiction.getSelected();
    if (visible.size === 1) return `in ${[...visible][0]}`;
    return `across ${visible.size} ${pluralizeNoun(jurisdiction.getActiveLabel().toLowerCase())}`;
  }
  return `in ${formatCityLabel(SELECTED_CITY)}`;
}

// Cities whose source carries only a single combined/total value (no land/building split).
// The value lives in REALLANDVA but is TOTAL value, so the blurb/labels must not say "land".
const COMBINED_VALUE_ONLY = !!(CITIES[SELECTED_CITY] as any)?.combinedValueOnly;

function landBlurbHtml(): string {
  const t = computeScopeTotals();
  if (!t) return '';
  const where = landScopeLabel();
  // ScopeTotals.acres is unit-agnostic; render as hectares for metric cities (1 ha = 2.47105 ac).
  const areaVal = METRIC_UNITS ? t.acres / 2.4710538 : t.acres;
  const areaLabel = METRIC_UNITS ? 'hectares' : 'acres';
  const acres = `<span class="highlight">${fmtAcresBlurb(areaVal)} ${areaLabel}</span>`;
  const dollars = (v: number) => `<span class="highlight">${fmtCompactDollars(v)}</span>`;
  const f = currentField || '';
  // Combined-value city: REALLANDVA / land_value_per_sqm hold TOTAL value, not land value.
  if (COMBINED_VALUE_ONLY)
    return `${dollars(t.land)} of total assessed value over ${acres} ${where}`;
  if (/^(land_value_per_sqm|land_value_per_sqft|REALLANDVA_per_sqft|smooth_land_value_per_sqft|REALLANDVA)$/.test(f))
    return `${dollars(t.land)} of land value over ${acres} ${where}`;
  if (/^(improvement_value_per_sqft|REALIMPROV_per_sqft|REALIMPROV)$/.test(f))
    return `${dollars(t.impr)} of improvement value over ${acres} ${where}`;
  if (/^(full_market_value_per_sqft|TLLDIMPROV_per_sqft|TLLDIMPROV)$/.test(f))
    return `${dollars(t.total)} of total value over ${acres} ${where}`;
  return `${acres} of land worth ${dollars(t.total)} ${where}`;
}

function updateLandBlurb() {
  // Metric picker hover tooltip describes the selected field (runs on every field change + load).
  fieldSelect.title = (currentField && FIELD_TOOLTIPS[currentField]) || '';
  const el = document.getElementById('land-value-blurb');
  if (!el) return;
  const html = landBlurbHtml();
  el.innerHTML = html;
  el.style.display = html ? '' : 'none';
}

type UnderSummaryRow = {
  label: string;
  total: number;
  count?: number;
  avgPpsf?: number | null;
  // Underdeveloped only: dollar totals split by improvement share of total
  // parcel value (<10%, 10–25%, 25–50%). Only rendered when present.
  buckets?: { label: string; total: number; count?: number }[];
};

// Improvement-share buckets for the Underdeveloped breakdown. A parcel is
// Underdeveloped when improvements are <50% of total value (<33% for single
// family — see data/parcel_calculations.py), so the three bins cover the
// whole category.
const UNDERDEV_BUCKETS = [
  { key: 'lt10', label: 'Improvements <10% of value', min: 0, max: 10 },
  { key: '10_25', label: 'Improvements 10–25%', min: 10, max: 25 },
  { key: '25_50', label: 'Improvements 25–50%', min: 25, max: 101 },
] as const;

function hasMeaningfulUnderSummaryRow(row: UnderSummaryRow) {
  if (!availableUnderCategories.includes(row.label)) return false;
  if (typeof row.count === 'number') return row.count > 0;
  return row.total > 0;
}

function renderUnderSummary(rows: UnderSummaryRow[], totalNonExempt: number, parkingFootprint = false) {
  const sumUnder = rows.reduce((acc, row) => acc + row.total, 0);
  const pct = (v: number) => totalNonExempt ? ((v / totalNonExempt) * 100).toFixed(1) + '%' : '0%';
  const parkingNote = parkingFootprint
    ? '“Parking Lot” is the land value under surface parking — the same total as the Parking tab, summed onto the parcels that host it.'
    : '“Parking Lot” here tags whole parcels whose primary use is parking — distinct from the per-lot footprints on the Parking tab.';

  underTotals.innerHTML = `
    <div class="totals-card">
      <div class="totals-title">Opportunity Land Value</div>
      <p class="sidebar-section-copy">${parkingNote}</p>
      <div class="under-stats-grid">
        ${rows.map((row) => `
          <div class="under-stat">
            <div class="under-stat-label">${row.label}</div>
            <div class="under-stat-value">${fmtCurrencyRounded(row.total)}</div>
            <div class="under-stat-meta">
              ${pct(row.total)} of nonexempt total
              ${Number.isFinite(row.avgPpsf ?? NaN) ? `<span class="under-stat-note">Avg ${fmtCurrencyPerSqft(row.avgPpsf)}</span>` : ''}
              ${typeof row.count === 'number' ? `<span class="under-stat-note">${row.count.toLocaleString()} parcels</span>` : ''}
            </div>
            ${row.buckets && row.buckets.some(b => b.total > 0) ? `
              <div class="under-bucket-list">
                ${row.buckets.map(b => `
                  <div class="under-bucket-row">
                    <span class="under-bucket-label">${b.label}</span>
                    <span class="under-bucket-value">${fmtCurrencyRounded(b.total)}${typeof b.count === 'number' ? ` <span class="under-bucket-count">(${b.count.toLocaleString()})</span>` : ''}</span>
                  </div>
                `).join('')}
              </div>
            ` : ''}
          </div>
        `).join('')}
        <div class="under-stat">
          <div class="under-stat-label">Combined Total</div>
          <div class="under-stat-value">${fmtCurrencyRounded(sumUnder)}</div>
          <div class="under-stat-meta">${pct(sumUnder)} of ${fmtCurrencyRounded(totalNonExempt)}</div>
        </div>
      </div>
    </div>`;
}

// IMPORTANT: `fc` must be the full citywide feature collection (e.g.
// `currentGeoJSON`), NOT a filtered subset. The denominator
// (`totalNonExempt`) needs to span all non-exempt parcels in the
// geographic area of interest so the displayed percentages reflect a
// share of citywide non-exempt land. Passing a filtered FC will make
// the per-category percentages add up to 100%.
//
// `selectedCategories` only controls which rows render (mirrors
// `updateUnderTotalsFromMetadata`); it does NOT narrow the denominator.
function updateUnderTotals(fc: GeoJSON.FeatureCollection, selectedCategories?: string[]) {
  const totals: Record<string, number> = { Vacant: 0, 'Parking Lot': 0, Underdeveloped: 0 };
  const counts: Record<string, number> = { Vacant: 0, 'Parking Lot': 0, Underdeveloped: 0 };
  const ppsfSums: Record<string, number> = { Vacant: 0, 'Parking Lot': 0, Underdeveloped: 0 };
  const ppsfCounts: Record<string, number> = { Vacant: 0, 'Parking Lot': 0, Underdeveloped: 0 };
  const bucketTotals = UNDERDEV_BUCKETS.map(() => ({ total: 0, count: 0 }));
  // For "Parking Lot", carry the surface-parking FOOTPRINT value/area (when the parcel provides
  // them, e.g. Tallinn) so the total matches the footprint-precise Parking tab rather than summing
  // whole-parcel values. avgPpsf for the category is then footprint value ÷ footprint area.
  let pkFootValue = 0, pkFootAreaSqft = 0, pkFoot = false;
  let totalNonExempt = 0;
  for (const f of fc.features) {
    const p = f.properties as any;
    const land = Number(p?.REALLANDVA);
    if (!Number.isFinite(land)) continue;
    const exempt = p?.exemption_flag != null && Number(p.exemption_flag) !== 0;
    if (!exempt) totalNonExempt += land;
    const cat = String(p?.[DEV_CATEGORY_FIELD] ?? '');
    if (!exempt && totals.hasOwnProperty(cat)) {
      const foot = cat === 'Parking Lot' ? Number(p?.parking_footprint_land_value) : NaN;
      const useFoot = cat === 'Parking Lot' && Number.isFinite(foot);
      totals[cat] += useFoot ? foot : land;
      counts[cat] += 1;
      if (cat === 'Underdeveloped') {
        // Improvement share of total value; recompute from raw values when the
        // derived field is missing so no parcel silently drops out.
        let imprPct = Number(p?.IMPR_PCT_TOTAL);
        if (!Number.isFinite(imprPct)) {
          const impr = Number(p?.REALIMPROV);
          const total = land + impr;
          imprPct = Number.isFinite(impr) && total > 0 ? (impr / total) * 100 : NaN;
        }
        if (Number.isFinite(imprPct)) {
          const idx = UNDERDEV_BUCKETS.findIndex(b => imprPct >= b.min && imprPct < b.max);
          if (idx >= 0) { bucketTotals[idx].total += land; bucketTotals[idx].count += 1; }
        }
      }
      if (useFoot) {
        pkFoot = true;
        pkFootValue += foot;
        pkFootAreaSqft += Number(p?.parking_footprint_area_sqft) || 0;
      } else {
        const ppsf = preferredLandValuePpsfField ? Number(p?.[preferredLandValuePpsfField]) : NaN;
        if (Number.isFinite(ppsf)) {
          ppsfSums[cat] += ppsf;
          ppsfCounts[cat] += 1;
        }
      }
    }
  }
  // Footprint-based avg €/sqft for Parking Lot (fmtCurrencyPerSqft converts to €/m² for metric).
  const pkFootPpsf = pkFoot && pkFootAreaSqft > 0 ? pkFootValue / pkFootAreaSqft : null;
  // An explicit empty selection (all boxes unchecked) shows no rows; a truly-absent
  // selection (undefined) still shows all rows.
  const selectedSet = selectedCategories ? new Set(selectedCategories) : null;
  const rows = [
    { label: 'Vacant', total: totals.Vacant, count: counts.Vacant, avgPpsf: ppsfCounts.Vacant ? (ppsfSums.Vacant / ppsfCounts.Vacant) : null },
    {
      label: 'Underdeveloped', total: totals.Underdeveloped, count: counts.Underdeveloped,
      avgPpsf: ppsfCounts.Underdeveloped ? (ppsfSums.Underdeveloped / ppsfCounts.Underdeveloped) : null,
      buckets: UNDERDEV_BUCKETS.map((b, i) => ({ label: b.label, total: bucketTotals[i].total, count: bucketTotals[i].count })),
    },
    { label: 'Parking Lot', total: totals['Parking Lot'], count: counts['Parking Lot'], avgPpsf: pkFoot ? pkFootPpsf : (ppsfCounts['Parking Lot'] ? (ppsfSums['Parking Lot'] / ppsfCounts['Parking Lot']) : null) }
  ].filter((row) => hasMeaningfulUnderSummaryRow(row) && (!selectedSet || selectedSet.has(row.label)));
  renderUnderSummary(rows, totalNonExempt, pkFoot);
}

function populateCategoryCheckboxes(container: HTMLDivElement | null, values: string[], selectedValues?: string[]) {
  if (!container) return;
  container.innerHTML = '';
  const selectedSet = selectedValues ? new Set(selectedValues) : null;
  for (const v of values) {
    const label = document.createElement('label');
    label.style.display = 'flex';
    label.style.gap = '8px';
    label.style.alignItems = 'center';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = v;
    input.checked = selectedSet ? selectedSet.has(v) : true;
    label.appendChild(input);
    label.appendChild(document.createTextNode(v));
    container.appendChild(label);
  }
}

function resolveUnderutilizedCategoryValues(values: string[]): string[] {
  const byLower = new Map<string, string>();
  for (const value of values) {
    const trimmed = value.trim();
    if (!trimmed) continue;
    const key = trimmed.toLowerCase();
    if (!byLower.has(key)) byLower.set(key, trimmed);
  }
  return UNDERUTILIZED_DEFAULTS
    .map(value => byLower.get(value.toLowerCase()) ?? null)
    .filter((value): value is string => !!value);
}

function populateUnderCategoryOptions(values: string[]) {
  const resolved = resolveUnderutilizedCategoryValues(values);
  availableUnderCategories = resolved.slice();
  populateCategoryCheckboxes(underCategoryContainer, resolved, resolved);
}

function updateUnderLegendCategoryMode(selectedCategories: string[]) {
  if (!underLegendEl) return;
  const labels = selectedCategories.map((category) => {
    const ramp = UNDER_CATEGORY_RAMPS[category] ?? ['#94a3b8'];
    const swatches = ramp.map((color) => `<span class="under-legend-chip" style="background:${color};"></span>`).join('');
    return `
      <div class="under-legend-item">
        <span>${category}</span>
        <span style="display:inline-flex; gap:4px; margin-left:auto;">${swatches}</span>
      </div>`;
  }).join('');
  underLegendEl.innerHTML = `
    <div class="under-legend-swatches">${labels}</div>
    <div class="muted" style="font-size:12px;">Within each type, darker shades indicate higher land value per ${METRIC_UNITS ? 'm²' : 'sqft'}.</div>
  `;
  underLegendEl.style.display = 'grid';
}

function getUnderValueField(): string {
  return preferredLandValuePpsfField || currentField || 'REALLANDVA';
}

function getUnderFilteredFeatureCollection(selectedCategories: string[], origVal: string): GeoJSON.FeatureCollection | undefined {
  if (!currentGeoJSON) return undefined;
  const hasCategoryFilter = selectedCategories.length > 0;
  return {
    type: 'FeatureCollection',
    features: currentGeoJSON.features.filter((f) => {
      const p = (f.properties as any) || {};
      const catOk = !hasCategoryFilter || selectedCategories.includes(String(p?.[DEV_CATEGORY_FIELD] ?? ''));
      const origOk = !origVal || String(p?.[ORIG_CATEGORY_FIELD] ?? '') === origVal;
      return catOk && origOk;
    })
  };
}

function buildUnderCategoryColorExpression(field: string, filteredFc?: GeoJSON.FeatureCollection): Expression {
  const fieldExpr: Expression = ['coalesce', ['to-number', ['get', field]], 0] as any;
  const fallbackRamp = COLOR_RAMPS.YlOrRd ?? ['#fff7bc', '#fec44f', '#fe9929', '#d95f0e'];
  const categories = ['Vacant', 'Parking Lot', 'Underdeveloped'].filter((category) => availableUnderCategories.includes(category));
  // A `match` needs input + ≥1 label/output pair + fallback. When the city has none of the
  // under-categories (e.g. a source with no vacant/underdeveloped signal), a bare
  // ['match', input, fallback] is invalid ("Expected at least 4 arguments") — return a
  // constant color instead. (The Underused tab is typically hidden for such cities anyway.)
  if (categories.length === 0) return '#94a3b8' as any;
  const expression: any[] = ['match', ['get', DEV_CATEGORY_FIELD]];

  for (const category of categories) {
    let colors = UNDER_CATEGORY_RAMPS[category] ?? fallbackRamp;
    let breaks: number[] = [];

    if (filteredFc && filteredFc.features.length) {
      const values = filteredFc.features
        .filter((f) => String((f.properties as any)?.[DEV_CATEGORY_FIELD] ?? '') === category)
        .map((f) => Number((f.properties as any)?.[field]))
        .filter((v) => Number.isFinite(v) && v >= 0);
      if (values.length >= 2) {
        breaks = getRampQuantileBreaks(values, DEFAULT_RAMP_KEY, colors.length);
      }
    } else if (pmtilesMetadata?.statistics?.[field]) {
      const { min, max } = pmtilesMetadata.statistics[field];
      breaks = getPmtilesFallbackBreaks(min, max, DEFAULT_RAMP_KEY, colors.length);
    }

    expression.push(category, makeStepColorExpression(fieldExpr, colors, breaks));
  }

  expression.push('#94a3b8');
  return expression as any;
}

// Tiny sub-500-sqft sliver remnants carry a real account value on a fragment polygon,
// so their per-sqft is meaningless. When the city opts in (hideRemnants), drop them from
// the per-parcel layers. The low-zoom hex aggregate has no likely_remnant field, so it is
// left alone (its averages already absorb these slivers harmlessly).
const REMNANT_OK_FILTER: Expression = ['!=', ['to-number', ['get', 'likely_remnant']], 1] as any;
function withRemnantFilter(filter: Expression | null): Expression | null {
  if (!HIDE_REMNANTS) return filter;
  return (filter ? ['all', filter, REMNANT_OK_FILTER] : REMNANT_OK_FILTER) as any;
}

function setParcelLayerFilter(targetMap: maplibregl.Map | null | undefined, filter: Expression | null) {
  if (!targetMap) return;
  if (targetMap === mapUnder) {
    let f = withRemnantFilter(filter);
    // Region show/hide: the panel lives on the Value tab but its selection is app-wide, so the
    // Underused map (same parcels, carrying `district`) honours it too — matching how the value
    // parcel/hex layers are filtered. When all regions are visible the clause is a no-op.
    if (jurisdiction.isActive()) {
      const sel = jurisdiction.selectedClause();
      f = (f ? ['all', f, sel] : sel) as any;
    }
    for (const layerId of [UNDER_FILL_LAYER, UNDER_OUTLINE_LAYER]) {
      if (targetMap.getLayer(layerId)) {
        targetMap.setFilter(layerId, f as any);
      }
    }
    return;
  }

  // The PMTiles low-zoom aggregate layer does not carry the refined/original
  // category fields used by the UI filters, so applying those filters there
  // makes the map appear empty until users zoom into parcel tiles.
  // When the region treatment owns the main map, applyExtrusion is the sole authority for the
  // parcel + hex layer filters (it ANDs the selected-regions clause onto the category/remnant
  // filter, and runs on every render incl. the async PMTiles convergence). Skip them here.
  const jurisdictionOwns = targetMap === map && jurisdiction.isActive();
  if (targetMap.getLayer(LAYER_ID_LOW) && !jurisdictionOwns) {
    const lowZoomFilter = cityUsesPmtiles() && filter ? null : filter;
    targetMap.setFilter(LAYER_ID_LOW, lowZoomFilter as any);
  }
  if (targetMap.getLayer(LAYER_ID) && !jurisdictionOwns) {
    targetMap.setFilter(LAYER_ID, withRemnantFilter(filter) as any);
  }
}

function syncMapView(sourceMap: maplibregl.Map, targetMap: maplibregl.Map | null | undefined) {
  if (!targetMap) return;
  targetMap.jumpTo({
    center: sourceMap.getCenter(),
    zoom: sourceMap.getZoom(),
    bearing: sourceMap.getBearing(),
    pitch: sourceMap.getPitch()
  });
}
// The category (refined + original) filter for the main map, from the current UI state.
// Shared by applyFilterAndScaling (the authority) and applyExtrusion (which re-derives it to
// AND in the jurisdiction clause on the de-emph/main layer pair). Does NOT include remnant.
function computeCategoryFilter(): Expression | null {
  const inputs = categoryInputs();
  const selected = inputs.filter(i => i.checked).map(i => i.value);
  const hasCategoryFilter = inputs.length > 0 && selected.length > 0 && selected.length < inputs.length;
  const refinedFilter = hasCategoryFilter ? (['in', ['get', DEV_CATEGORY_FIELD], ['literal', selected]] as any) : null;
  const origVal = origCategorySelect?.value || '';
  const origFilter = origVal ? (['==', ['get', ORIG_CATEGORY_FIELD], origVal] as any) : null;
  if (refinedFilter && origFilter) return ['all', refinedFilter, origFilter] as any;
  if (refinedFilter) return refinedFilter;
  if (origFilter) return origFilter;
  return null;
}
function applyFilterAndScaling() {
  // For PMTiles, filtering works on the vector source, but we don't have currentGeoJSON
  const hasData = currentGeoJSON || cityUsesPmtiles();
  if (!hasData) return;

  const inputs = categoryInputs();
  const selected = inputs.filter(i => i.checked).map(i => i.value);
  const hasCategoryFilter = inputs.length > 0 && selected.length > 0 && selected.length < inputs.length;

  const filter: Expression | null = computeCategoryFilter();
  setParcelLayerFilter(map, filter);

  // For PMTiles, always use full-dataset statistics from metadata (scale filtered not supported)
  if (cityUsesPmtiles()) {
    computeAndApplyAutoMultiplier('auto', HEIGHT_CAPS.main, HEIGHT_PCTL);
  } else {
    const shouldScaleFiltered = scaleFiltered && scaleFiltered.checked && (hasCategoryFilter || !!origCategorySelect?.value);
    if (shouldScaleFiltered) {
      const filtered: GeoJSON.Feature[] = currentGeoJSON!.features.filter(f => {
        const p = (f.properties as any) || {};
        const catOk = !hasCategoryFilter || selected.includes(String(p?.[DEV_CATEGORY_FIELD] ?? ''));
        const origOk = !origCategorySelect?.value || String(p?.[ORIG_CATEGORY_FIELD] ?? '') === origCategorySelect.value;
        return catOk && origOk;
      });
      const fc: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: filtered };
      computeAndApplyAutoMultiplier('auto', HEIGHT_CAPS.main, HEIGHT_PCTL, fc);
    } else {
      computeAndApplyAutoMultiplier('auto', HEIGHT_CAPS.main, HEIGHT_PCTL);
    }
  }
  applyExtrusion();
}

function renderUnderNow() {
  const hasData = currentGeoJSON || cityUsesPmtiles();
  if (!hasData || !mapUnder.getLayer(UNDER_FILL_LAYER)) return;
  const inputs = categoryInputsUnder();
  const selected = inputs.filter(i => i.checked).map(i => i.value);
  // Treat the category checkboxes as an explicit allow-list. Build the `in` filter
  // whenever the checkboxes exist — even when nothing is checked, so an all-unchecked
  // state matches NOTHING (empty map). Falling through to a null filter would clear the
  // layer filter entirely and render every parcel, which looked like "all three checked".
  let filter: any = null;
  const refinedFilter = inputs.length > 0 ? (['in', ['get', DEV_CATEGORY_FIELD], ['literal', selected]] as any) : null;
  const origVal = underOrigCategorySelect?.value || '';
  const origFilter = origVal ? (['==', ['get', ORIG_CATEGORY_FIELD], origVal] as any) : null;
  if (refinedFilter && origFilter) filter = ['all', refinedFilter, origFilter] as any;
  else if (refinedFilter) filter = refinedFilter;
  else if (origFilter) filter = origFilter;
  setParcelLayerFilter(mapUnder, filter);
  const filteredFc = getUnderFilteredFeatureCollection(selected, origVal);
  const field = getUnderValueField();
  const opacity = (parseInt(underOpacityInput?.value || '92') / 100);
  const colorExpr = buildUnderCategoryColorExpression(field, filteredFc);
  mapUnder.setPaintProperty(UNDER_FILL_LAYER, 'fill-color', colorExpr);
  mapUnder.setPaintProperty(UNDER_FILL_LAYER, 'fill-opacity', opacity);
  // Always feed updateUnderTotals the full citywide FC so the
  // denominator is non-exempt totals across the whole jurisdiction,
  // independent of the refined/original category filters applied to
  // the map. `selected` only narrows which rows are rendered.
  if (currentGeoJSON) {
    updateUnderTotals(currentGeoJSON, selected);
  } else if (currentUnderMetadataTotals) {
    updateUnderTotalsFromMetadata(currentUnderMetadataTotals, selected);
  }
  updateUnderLegendCategoryMode(selected);
}

function renderRatioNow() {
  const hasData = currentGeoJSON || cityUsesPmtiles();
  if (!hasData || !mapRatio?.getLayer(LAYER_ID)) return;
  const ratioField = (ratioFieldSelect?.value || 'IMPR_LAND_RATIO');
  // Apply original category filter to the ratio map
  let filter: any = null;
  const origVal = ratioOrigCategorySelect?.value || '';
  if (origVal) filter = ['==', ['get', ORIG_CATEGORY_FIELD], origVal] as any;
  setParcelLayerFilter(mapRatio, filter);
  renderMapFor(
    mapRatio!,
    ratioField,
    'asis',
    /*invert*/ !!(ratioInvertHeights?.checked),
    ratioRampSelect?.value || 'Reds',
    /*reverse*/ true,
    (parseInt(ratioOpacityInput?.value || '0') / 100),
    (parseFloat(ratioMultInput?.value || '1') || 1) * heightFactorRatio,
    ratioLegendEl,
    undefined,
    HEIGHT_CAPS.ratio
  );
}


// @ts-expect-error TS6133: reserved for future use
function _fitToData(fc: GeoJSON.FeatureCollection) {
  const b = bbox(fc); if (!b) return;
  map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 40, duration: 800 });
}

function fitToDataAll(fc: GeoJSON.FeatureCollection) {
  const b = bbox(fc); if (!b) return;
  const bounds: [[number, number], [number, number]] = [[b[0], b[1]], [b[2], b[3]]];
  // duration: 0 (instant) on initial load. An ANIMATED fit here caused a MapLibre geojson race: the
  // camera animates for 800ms while the geojson-vt worker is still indexing, orphaning the tile
  // requests for the final viewport so the map rendered empty until the user nudged it (Tallinn
  // "only a few sparse parcels" bug, ~60% of loads). An instant jump requests the final-view tiles
  // exactly once, after the worker is ready — verified 5/5 dense loads vs. ~2/5 with the animation.
  // Skip maps whose camera came from a shared URL (restoredCameras) — the link's view wins.
  if (!restoredCameras.map) map.fitBounds(bounds, { padding: 40, duration: 0 });
  if (!restoredCameras.umap) mapUnder.fitBounds(bounds, { padding: 40, duration: 0 });
  mapRatio?.fitBounds(bounds, { padding: 40, duration: 0 });
}

// ---- Quality toggle (runtime supersampling) ----
let _appliedPR = HQ_PR;     // pixel ratio currently applied to the main map (native by default)
let _prBusy = false;        // reentrancy guard (setPixelRatio internally resizes + fires events)
function applyPixelRatio(pr: number) {
  const anyMap = map as any;
  if (typeof anyMap.setPixelRatio !== 'function') return;
  if (_prBusy) return;                              // already mid-change
  if (Math.abs(_appliedPR - pr) < 0.001) return;   // no-op
  // setPixelRatio() internally calls map.resize(), which constrains the transform and FIRES
  // camera events (movestart/zoom) — the same events our downshift handler listens to. So set
  // _appliedPR first and guard reentrancy, or the resize-fired event recurses into setPixelRatio
  // forever ("too much recursion"). No explicit resize() needed — setPixelRatio does it.
  _appliedPR = pr;
  _prBusy = true;
  try { anyMap.setPixelRatio(pr); } finally { _prBusy = false; }
}
function setQuality(mode: QualityMode) {
  _qualityMode = mode;
  applyPixelRatio(mode === 'high' ? HIGH_PR : FAST_PR);
  const btn = document.getElementById('btn-quality') as HTMLButtonElement | null;
  if (btn) btn.textContent = (mode === 'high') ? 'Quality: High' : 'Quality: Fast';
}
// Manual High (supersampled, crisp) ↔ Fast (native, smooth) toggle. High costs pan FPS, so it's
// opt-in; Fast is the default. The resize/flash on switch is acceptable for an explicit click.
const qualityBtn = document.getElementById('btn-quality') as HTMLButtonElement | null;
qualityBtn?.addEventListener('click', () => setQuality(_qualityMode === 'high' ? 'fast' : 'high'));

// NOTE: we intentionally do NOT switch pixel ratio during interaction. setPixelRatio() reallocates
// the framebuffer (a visible flash) and the resize it triggers interrupts the active pan gesture,
// so adaptive supersampling is more disruptive than it's worth. We render at native res (HQ_PR)
// throughout; setQuality remains for a possible future manual High/Fast button.

// Live performance profiler (see perf.ts) — OPT-IN even in dev so its rAF loop + Long Tasks
// observer + idle feature-counts never slow normal browsing. Enable with ?perf=1 (or
// localStorage gvw_perf=1). Production strips it entirely (PERF_HUD folds to false via DEV).
if (PERF_HUD) {
  initPerf({
    map,
    city: SELECTED_CITY,
    activeLayer: () => {
      const pmz = Number(getCityConfig()?.parcelMinZoom);
      const onParcels = Number.isFinite(pmz) && map.getZoom() >= pmz;
      return onParcels ? { id: LAYER_ID, label: 'parcels' } : { id: LAYER_ID_LOW, label: 'hexes' };
    },
  });
} else if (import.meta.env.DEV) {
  console.info('[perf] HUD off — enable with ?perf=1 (or localStorage.gvw_perf=1)');
}

/* ---------------- Helpers ---------------- */
function clampSmoothingThreshold(value: number | null | undefined): number {
  const num = Number(value);
  if (!Number.isFinite(num)) return SMOOTH_THRESHOLD_DEFAULT;
  return Math.min(SMOOTH_THRESHOLD_MAX, Math.max(SMOOTH_THRESHOLD_MIN, num));
}

function computeDisplayedMetricFromProps(props: Record<string, any>): number | null {
  if (!currentField) return null;
  let base: number | null;
  if (currentField === 'IMPR_LAND_RATIO') {
    const num = numOrNull(props.REALIMPROV);
    const den = numOrNull(props.REALLANDVA);
    base = (num != null && den != null && den > 0) ? num / den : null;
  } else if (currentField === 'IMPR_PCT_TOTAL') {
    const impr = numOrNull(props.REALIMPROV);
    const land = numOrNull(props.REALLANDVA);
    if (impr == null || land == null) return null;
    const total = impr + land;
    base = total > 0 ? (impr / total) * 100 : null;
  } else if (currentField === 'LAND_PCT_TOTAL') {
    const impr = numOrNull(props.REALIMPROV);
    const land = numOrNull(props.REALLANDVA);
    if (impr == null || land == null) return null;
    const total = impr + land;
    base = total > 0 ? (land / total) * 100 : null;
  } else if (PER_SQFT_SRC[currentField]) {
    const v = numOrNull(props[PER_SQFT_SRC[currentField]]);
    const acres = numOrNull(props.land_area_acres);
    base = (v != null && acres != null && acres > 0) ? v / (acres * 43560) : null;
  } else {
    base = numOrNull(props[currentField]);
  }
  if (base == null) return null;

  if (normalizationMode === 'perLand' && landSizeField) {
    const d = numOrNull(props[landSizeField]);
    if (d == null || d <= 0) return null;
    base = base / d;
  } else if (normalizationMode === 'perBuilding' && bldgSizeField) {
    const d = numOrNull(props[bldgSizeField]);
    if (d == null || d <= 0) return null;
    base = base / d;
  }
  return base;
}

function computeExtrusionHeightMeters(metricValue: number): number {
  const capped = (heightSmoothingEnabled && heightSmoothingCap != null)
    ? Math.min(metricValue, heightSmoothingCap)
    : metricValue;
  const unitFactor = UNIT_TO_METERS[unitsSelect.value as keyof typeof UNIT_TO_METERS] ?? 1;
  const mult = Number(multInput.value);
  const multiplier = (Number.isFinite(mult) ? mult : 0) * heightFactorMain;
  if (invertHeights?.checked && currentField === 'IMPR_PCT_TOTAL') {
    return (100 - capped) * multiplier * unitFactor;
  }
  if (invertHeights?.checked && heightRankBreaks && heightRankBreaks.length) {
    const k = Math.max(2, HEIGHT_RANK_BINS);
    let i = 0;
    while (i < heightRankBreaks.length && capped >= heightRankBreaks[i]) i++;
    const denom = (k - 1);
    const rankInv = (denom - i) / denom;
    return rankInv * multiplier * unitFactor;
  }
  return capped * multiplier * unitFactor;
}

// Queue an update; newer calls replace older ones.
function scheduleUpdate(mode: UpdateMode, refreshLegend = false, debounceMs = 80) {
  // For PMTiles, currentGeoJSON is null but we still have data
  const hasData = currentGeoJSON || (cityUsesPmtiles() && pmtilesMetadata);
  if (!hasData) return;   // <- hard stop until data exists

  _pendingMode = mode;
  _pendingRefreshLegend = refreshLegend;
  if (_updTimer) clearTimeout(_updTimer);
  _updTimer = window.setTimeout(() => {
    _updTimer = null;
    if (_pendingMode === 'recomputeAndAutoScale') {
      computeAndApplyAutoMultiplier('auto', HEIGHT_CAPS.main, HEIGHT_PCTL);
      if (_pendingRefreshLegend) updateLegend();
    } else {
      applyExtrusion();
      if (_pendingRefreshLegend) updateLegend();
    }
  }, debounceMs);
}

function chooseBestMetricUnitForMultiplier(p99: number, capMeters = 1000): { unit: MetricUnitKey; multiplier: number } {
  const candidates: MetricUnitKey[] = ['centimeters', 'meters', 'kilometers'];
  const RANGE_MIN = 1, RANGE_MAX = 100;

  let best = { unit: 'centimeters' as MetricUnitKey, multiplier: Infinity, score: Infinity };

  for (const u of candidates) {
    const unitFactor = UNIT_TO_METERS[u]; // meters per unit
    const mult = capMeters / (unitFactor * p99);

    const inRange = mult >= RANGE_MIN && mult <= RANGE_MAX;
    const distToRange = inRange ? 0 : Math.min(Math.abs(mult - RANGE_MIN), Math.abs(mult - RANGE_MAX));
    const tieBias = Math.abs(Math.log10(Math.max(1e-12, mult)) - 1); // prefer closer to ~10 if inside

    // Primary: be inside [1,100]; Secondary: closer to the band; Tertiary: closer to 10 within the band
    const score = (inRange ? 0 : 1) * 1e6 + distToRange * 1e3 + (inRange ? tieBias : 0);

    if (score < best.score) best = { unit: u, multiplier: mult, score };
  }
  return { unit: best.unit, multiplier: best.multiplier };
}

// Internal/flag fields that are numeric but not meaningful to visualize — kept out of the
// metric picker.
const HIDDEN_METRIC_FIELDS = new Set(['exemption_flag', 'likely_remnant']);
function populateFieldDropdownFromList(list: string[]) {
  const fields = list.filter(n => !HIDDEN_METRIC_FIELDS.has(n));
  fieldSelect.replaceChildren();
  if (!fields.length) fieldSelect.append(new Option('No numeric fields available', ''));
  else {
    fieldSelect.append(new Option('— choose —', ''));
    for (const n of fields) fieldSelect.append(new Option(FIELD_LABELS[n] ?? n, n));
  }
  fitFieldSelectFont();
}

// The metric picker is a native <select> with a fixed font; long labels otherwise get
// ellipsis-truncated. Shrink the font just enough that the selected label fits on one line.
// Width scales linearly with font-size, so we measure once at the CSS default size and scale.
let _fieldFitCanvas: HTMLCanvasElement | null = null;
function fitFieldSelectFont() {
  const text = fieldSelect.options[fieldSelect.selectedIndex]?.text ?? '';
  fieldSelect.style.fontSize = '';            // reset to CSS default (1.3rem) before measuring
  if (!text) return;
  const cs = getComputedStyle(fieldSelect);
  const basePx = parseFloat(cs.fontSize) || 20.8;
  const avail = fieldSelect.clientWidth
    - parseFloat(cs.paddingLeft || '0')
    - parseFloat(cs.paddingRight || '0');     // right padding holds the dropdown arrow
  if (avail <= 0) return;                      // not laid out yet; a later resize/change refits
  const ctx = (_fieldFitCanvas ||= document.createElement('canvas')).getContext('2d');
  if (!ctx) return;
  ctx.font = `${cs.fontWeight} ${basePx}px ${cs.fontFamily}`;
  const w = ctx.measureText(text).width;
  if (w > avail) fieldSelect.style.fontSize = `${Math.max(11, basePx * (avail / w))}px`;
}

// @ts-expect-error TS6133: reserved for future use
function _polygonsOnly(fc: GeoJSON.FeatureCollection) {
  return fc.features.filter(
    f => f.geometry && (f.geometry.type === 'Polygon' || f.geometry.type === 'MultiPolygon')
  );
}

function getNumericValuesNormalized(fc: GeoJSON.FeatureCollection, field: string, mode: 'asis'|'perLand'|'perBuilding'): number[] {
  const vals: number[] = [];
  for (const f of fc.features) {
    const p = (f.properties as any) || {};
    let base: number;
    if (field === 'IMPR_LAND_RATIO') {
      const num = Number(p?.REALIMPROV);
      const den = Number(p?.REALLANDVA);
      if (!Number.isFinite(num) || !Number.isFinite(den) || den <= 0) continue;
      base = num / den;
    } else if (field === 'IMPR_LAND_PCT') {
      const num = Number(p?.REALIMPROV);
      const den = Number(p?.REALLANDVA);
      if (!Number.isFinite(num) || !Number.isFinite(den) || den <= 0) continue;
      base = (num / den) * 100;
    } else if (field === 'IMPR_PCT_TOTAL') {
      const impr = Number(p?.REALIMPROV);
      const land = Number(p?.REALLANDVA);
      const total = impr + land;
      if (!Number.isFinite(impr) || !Number.isFinite(land) || total <= 0) continue;
      base = (impr / total) * 100;
    } else if (field === 'LAND_PCT_TOTAL') {
      const impr = Number(p?.REALIMPROV);
      const land = Number(p?.REALLANDVA);
      const total = impr + land;
      if (!Number.isFinite(impr) || !Number.isFinite(land) || total <= 0) continue;
      base = (land / total) * 100;
    } else {
      base = Number(p?.[field]);
      if (!Number.isFinite(base)) continue;
    }

    if (mode === 'perLand' && landSizeField) {
      const d = Number(p?.[landSizeField]);
      if (!Number.isFinite(d) || d <= 0) continue;
      base = base / d;
    } else if (mode === 'perBuilding' && bldgSizeField) {
      const d = Number(p?.[bldgSizeField]);
      if (!Number.isFinite(d) || d <= 0) continue;
      base = base / d;
    }
    vals.push(base);
  }
  return vals;
}
function applyHeightSmoothing(vals: number[]): { heightVals: number[]; cap: number | null } {
  if (!heightSmoothingEnabled) return { heightVals: vals, cap: null };
  if (!vals.length) return { heightVals: vals, cap: null };

  // Work in log space for robust outlier detection, but keep heights linear.
  // Use log1p so zeros are safe: log1p(v) = log(1 + v).
  const logVals = vals.map(v => Math.log1p(Math.max(v, 0)));

  // Robust z-score using median and MAD (scaled by 1.4826) in log space.
  const medianLog = percentile(logVals, 50);
  const deviationsLog = logVals.map(v => Math.abs(v - medianLog));
  const madLog = percentile(deviationsLog, 50);
  const scale = madLog * 1.4826;

  // If scale is zero or invalid, nothing to smooth.
  if (!(scale > 0)) return { heightVals: vals, cap: null };

  const threshold = clampSmoothingThreshold(heightSmoothingThreshold);
  heightSmoothingThreshold = threshold;
  if (smoothHeightsStrictness) smoothHeightsStrictness.value = String(threshold);

  // Determine non-outliers based on log-space robust z-score,
  // but keep the values themselves in the original domain.
  const nonOutliers: number[] = [];
  for (let i = 0; i < vals.length; i++) {
    const z = Math.abs(logVals[i] - medianLog) / scale;
    if (z <= threshold) nonOutliers.push(vals[i]);
  }

  if (!nonOutliers.length) return { heightVals: vals, cap: null };

  const cap = Math.max(...nonOutliers);
  if (!Number.isFinite(cap)) return { heightVals: vals, cap: null };

  // Apply capping in the original (non-log) scale.
  return { heightVals: vals.map(v => Math.min(v, cap)), cap };
}


function computeStatsNormalized(fc: GeoJSON.FeatureCollection, field: string, mode: 'asis'|'perLand'|'perBuilding') {
  const vals = getNumericValuesNormalized(fc, field, mode);
  let min = Infinity, max = -Infinity;
  for (const v of vals) { if (v < min) min = v; if (v > max) max = v; }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) { min = 0; max = min + 1; }
  return { min, max };
}

// Build a step expression: first color is < break1, then each break raises the color.
function makeStepColorExpression(valueExpr: Expression, colors: string[], breaks: number[]): Expression {
  const c = colors.slice();                 // copy
  const b = breaks.slice();                 // copy
  if (b.length === 0) return ['step', valueExpr, c[0]] as any;

  const out: (string | number | Expression)[] = ['step', valueExpr, c[0]];
  // pair up thresholds with subsequent colors
  for (let i = 0; i < b.length && i + 1 < c.length; i++) {
    out.push(b[i], c[i + 1]);
  }
  return out as any;
}

// Build a step expression that returns the 0-based bin index for the given breaks
// Example: step(value, 0, b1, 1, b2, 2, ...)
function makeStepIndexExpression(valueExpr: Expression, breaks: number[]): Expression {
  const b = breaks.slice();
  const out: any[] = ['step', valueExpr, 0];
  for (let i = 0; i < b.length; i++) {
    out.push(b[i], i + 1);
  }
  return out as any;
}

// Auto-multiplier so p-th percentile reaches capMeters, in given units
function computeAndApplyAutoMultiplier(
  unitsKeyOrAuto: 'auto' | keyof typeof UNIT_TO_METERS = 'auto',
  capMeters = 1000,
  p = 99,
  fcOverride?: GeoJSON.FeatureCollection
) {
  // Check if using PMTiles with metadata
  if (cityUsesPmtiles() && pmtilesMetadata && currentField) {
    const fieldStats = pmtilesMetadata.statistics[currentField];
    if (fieldStats) {
      // Use metadata statistics
      currentStats = { min: fieldStats.min, max: fieldStats.max };
      const fieldPcts = pmtilesMetadata.percentiles?.[currentField];
      
      // For height scaling, prefer robust p99 from metadata to avoid outliers
      let pVal = fieldPcts?.p99;
      if (!Number.isFinite(pVal) || (pVal as number) <= 0) {
        // Fallback: conservative estimate from max
        pVal = fieldStats.max * 0.95;
      }
      if (!Number.isFinite(pVal) || (pVal as number) <= 0) {
        pVal = 1;
      }
      
      // Robust p1–p99 range. Skewed data (e.g. NYC land values) has min≈0 and max
      // in the millions, which would cluster most parcels into the lowest color bin;
      // p1–p99 gives a stable range for both color and the smoothing cap below.
      const p1v = fieldPcts?.p1;
      const p99v = fieldPcts?.p99;
      const lo = (Number.isFinite(p1v) && (p1v as number) >= fieldStats.min) ? (p1v as number) : fieldStats.min;
      const hi = (Number.isFinite(p99v) && (p99v as number) > lo) ? (p99v as number) : fieldStats.max;

      const ramp = activeRamp();

      // Unified with the geoparquet path: colorMode is the fraction→value axis, breaks come from
      // the (always-adjustable) handle fractions. colorVals is null here (PMTiles ships only p1/p99),
      // so breaksFromFractions() uses the baked quantile breaks for the even default and log if dragged.
      colorVals = null;
      // Colour domain tops at a ROBUST high so handles reach the expensive top end without a lone
      // sliver outlier wasting the range: prefer the baked p99.9 (parquet_to_pmtiles.py), then the
      // raw max, then the p99 `hi`. (`hi` above stays p99 for height/smoothing.)
      const p999 = (fieldPcts as any)?.p999 as number | undefined;
      const colorHi = (Number.isFinite(p999) && (p999 as number) > lo) ? (p999 as number)
        : (Number.isFinite(fieldStats.max) && fieldStats.max > lo) ? fieldStats.max : hi;
      colorDomain = { lo, hi: colorHi, label: 'p1–p99.9' };
      colorBreaks = currentColorBreaks(ramp.length);

      // "Smooth extreme heights" on PMTiles: cap the displayed metric so the top
      // outliers don't tower over everything. This is only meaningful at parcel
      // zoom — applyExtrusion() leaves the low-zoom H3 hexes (area-weighted
      // aggregates) uncapped. The strictness slider interpolates the cap from p99
      // (stricter, more squashing) toward the field max (gentler, less squashing).
      if (heightSmoothingEnabled && Number.isFinite(hi) && hi > 0) {
        const t = (clampSmoothingThreshold(heightSmoothingThreshold) - SMOOTH_THRESHOLD_MIN) /
                  (SMOOTH_THRESHOLD_MAX - SMOOTH_THRESHOLD_MIN);
        const tClamped = Math.max(0, Math.min(1, t));
        const capTop = hi + tClamped * Math.max(0, fieldStats.max - hi);
        heightSmoothingCap = Number.isFinite(capTop) && capTop > 0 ? capTop : null;
      } else {
        heightSmoothingCap = null;
      }
      
      // Height autoscale
      let unitKey: keyof typeof UNIT_TO_METERS;
      let multiplier: number;
      if (unitsKeyOrAuto === 'auto') {
        const best = chooseBestMetricUnitForMultiplier(pVal!, capMeters);
        unitKey = best.unit;
        multiplier = best.multiplier;
      } else {
        unitKey = unitsKeyOrAuto;
        const unitFactor = UNIT_TO_METERS[unitKey];
        multiplier = capMeters / (unitFactor * pVal!);
      }
      
      unitsSelect.value = unitKey;
      multInput.value = String(multiplier);
      
      vlog('[PMTiles] Auto-scale computed:', {
        mode: normalizationMode,
        field: currentField,
        pctl: p,
        pVal,
        unit: unitKey,
        multiplier,
        colorMode,
        colorBreaks: colorBreaks?.length || 0,
        colorDomain,
        stats: currentStats,
        multInputValue: multInput.value,
        unitsSelectValue: unitsSelect.value
      });

      applyExtrusion();

      // Debug: what actually landed on the layer (the getPaintProperty calls are only worth it
      // when we're logging, so they're behind VERBOSE too).
      if (VERBOSE) {
        const layer = map.getLayer(LAYER_ID);
        if (layer) {
          const heightProp = map.getPaintProperty(LAYER_ID, 'fill-extrusion-height');
          const colorProp = map.getPaintProperty(LAYER_ID, 'fill-extrusion-color');
          vlog('[PMTiles] Layer paint properties:', {
            height: heightProp,
            color: Array.isArray(colorProp) ? colorProp.slice(0, 5) : colorProp,
            opacity: map.getPaintProperty(LAYER_ID, 'fill-extrusion-opacity')
          });
        }
        vlog('[PMTiles] Extrusions applied after auto-scale');
      }
      return;
    }
  }
  
  // Original parquet-based logic
  const src = fcOverride || currentGeoJSON;
  if (!src || !currentField) return;

  // values for the CURRENT normalization mode
  const vals = getNumericValuesNormalized(src, currentField, normalizationMode);
  const { heightVals, cap } = applyHeightSmoothing(vals);
  heightSmoothingCap = cap;
  let scaleVals = heightVals;
  // For inverted heights, prefer simple invert for percentage metric; otherwise use rank-based fallback
  if (invertHeights?.checked) {
    if (currentField === 'IMPR_PCT_TOTAL') {
      heightRankBreaks = null;
      scaleVals = heightVals.map(v => 100 - v);
    } else {
      heightRankBreaks = quantileBreaks(heightVals, HEIGHT_RANK_BINS, 1, 99);
      const k = Math.max(2, HEIGHT_RANK_BINS);
      const denom = (k - 1);
      const toIdx = (v: number) => {
        let i = 0;
        while (i < (heightRankBreaks?.length || 0) && heightRankBreaks && v >= heightRankBreaks[i]) i++;
        return i;
      };
      scaleVals = heightVals.map(v => (denom - toIdx(v)) / denom);
    }
  } else {
    heightRankBreaks = null;
  }
  // Use p-th percentile of the active values; fallback to non-inverted; final fallback to 1
  let pVal = percentile(scaleVals, p);
  if (!Number.isFinite(pVal) || pVal <= 0) {
    const alt = percentile(heightVals, p);
    pVal = (Number.isFinite(alt) && alt > 0) ? alt : 1;
  }

  // ---- Color domain / breaks ----
  // Unified: every scaling method shares the p1–p99 domain and is adjustable via the legend
  // handles. colorMode (linear/quantiles/log) only changes how the handle fractions map to values.
  const ramp = activeRamp();
  colorVals = vals;
  const pLow = percentile(vals, 1);
  let lo = Number.isFinite(pLow) ? pLow : 0;
  // Colour domain tops at p99.9 (a ROBUST high — not the raw max, so a lone sliver-inflated outlier
  // doesn't waste the range, and not p99, which clamped the expensive top out of reach). Height
  // scaling still uses p99 (pVal) separately to avoid a lone tower.
  let hi = percentile(vals, 99.9);
  if (!Number.isFinite(hi) || !(hi > lo)) hi = vals.reduce((m, v) => (Number.isFinite(v) && v > m ? v : m), -Infinity);
  if (!Number.isFinite(hi) || !(hi > lo)) { lo = 0; hi = 1; }
  colorDomain = { lo, hi, label: 'p1–p99.9' };
  colorBreaks = currentColorBreaks(ramp.length);

  // ---- Height autoscale: anchor p-th percentile to capMeters ----
  let unitKey: keyof typeof UNIT_TO_METERS;
  let multiplier: number;
  if (unitsKeyOrAuto === 'auto') {
    const best = chooseBestMetricUnitForMultiplier(pVal, capMeters);
    unitKey = best.unit;
    multiplier = best.multiplier;
  } else {
    unitKey = unitsKeyOrAuto;
    const unitFactor = UNIT_TO_METERS[unitKey];
    multiplier = capMeters / (unitFactor * pVal);
  }

  unitsSelect.value = unitKey;
  multInput.value = String(multiplier);

  // stats for legend fallback
  currentStats = computeStatsNormalized(src, currentField, normalizationMode);

  console.debug('autoScale', {
    mode: normalizationMode,
    field: currentField,
    pctl: p,
    pVal,
    unit: unitKey,
    multiplier,
    colorMode,
    colorBreaks,
    colorDomain,
    stats: currentStats
  });

  applyExtrusion();
}

function makeColorExpressionFromExpr(valueExpr: Expression, colors: string[], min: number, max: number): Expression {
  const n = colors.length - 1;
  const stops: (number | string)[] = [];
  for (let i = 0; i < colors.length; i++) {
    const t = i / n;
    stops.push(min + t * (max - min), colors[i]);
  }
  // Clamp value into [min,max] to avoid outliers crushing the ramp
  const clamped: Expression = ['max', min, ['min', max, valueExpr]] as any;
  return ['interpolate', ['linear'], clamped, ...stops] as any;
}

// Format a legend value using the displayed field's natural units so the numbers
// read as money / percent / ratio rather than bare figures.
function formatLegendValue(field: string | null, v: number | null | undefined): string {
  if (v == null || !Number.isFinite(Number(v))) return '—';
  const n = Number(v);
  const f = (field || '').toLowerCase();
  if (f.includes('per_sqm')) return fmtCurrencyPerSqm(n);
  if (f.includes('per_sqft') || f.includes('ppsf')) return fmtCurrencyPerSqft(n);
  if (f.includes('pct')) return `${fmt(n)}%`;
  if (f.includes('ratio')) return fmt(n);
  if (f.includes('value') || f.includes('val') || f.includes('tax') || f.includes('improv') || f.includes('land')) return `${CURRENCY}${fmt(n)}`;
  return fmt(n);
}

// Shared, human-readable legend: title (what's being shown), a color swatch strip,
// the low→high value range in natural units, and a plain-language note explaining
// that both color AND height encode the metric. Replaces the old cryptic
// "Quantiles (p1–p99): 15 | 25 | …" string that meant nothing to most users.
function renderColorLegend(
  targetEl: HTMLFieldSetElement | null,
  opts: {
    field: string | null; ramp: string[]; lo: number | null; hi: number | null; invert: boolean;
    // Always-on draggable boundary handles: place where each colour transition falls along the
    // current scaling axis. fractions are 0..1 (length ramp.length-1). onChange fires with
    // committed=false during a drag (remember positions only) and committed=true on release
    // (recompute + repaint). formatValue renders the live value tooltip while dragging.
    handles?: {
      fractions: number[];
      onChange: (fractions: number[], committed: boolean) => void;
      formatValue: (fraction: number) => string;
    } | null;
  }
) {
  if (!targetEl) return;
  targetEl.replaceChildren();
  const { field, ramp, lo, hi, handles } = opts;
  if (!field || !ramp.length) { targetEl.style.display = 'none'; return; }

  const wrap = document.createElement('div');
  wrap.style.cssText = 'display:grid; gap:4px;';

  const interactive = !!handles && ramp.length > 1;
  const need = ramp.length - 1;
  // Band widths follow the handle fractions, so the strip visually shows the current scaling.
  const fr = interactive && handles!.fractions.length === need
    ? handles!.fractions.slice()
    : Array.from({ length: need }, (_, i) => (i + 1) / ramp.length);

  const stripWrap = document.createElement('div');
  stripWrap.style.cssText = 'position:relative;';   // hosts the swatch strip + the drag value tooltip

  const swatches = document.createElement('div');
  swatches.style.cssText = `position:relative; display:flex; height:${interactive ? 20 : 12}px; border-radius:3px; overflow:hidden; border:1px solid rgba(255,255,255,0.25);` + (interactive ? ' cursor:ew-resize;' : '');
  const cells: HTMLDivElement[] = [];
  for (let i = 0; i < ramp.length; i++) {
    const cell = document.createElement('div');
    const w = interactive ? ((i === 0 ? fr[0] : (i === ramp.length - 1 ? 1 - fr[i - 1] : fr[i] - fr[i - 1])) * 100) : 0;
    cell.style.cssText = interactive ? `width:${w}%; background:${ramp[i]};` : `flex:1 1 0; background:${ramp[i]};`;
    cells.push(cell);
    swatches.appendChild(cell);
  }
  stripWrap.appendChild(swatches);

  if (interactive) {
    // Live value tooltip under the dragged handle (hidden when settled).
    const tip = document.createElement('div');
    tip.style.cssText = 'position:absolute; top:100%; margin-top:3px; transform:translateX(-50%); padding:1px 6px; font-size:11px; font-weight:600; white-space:nowrap; background:rgba(15,23,42,0.92); color:#fff; border-radius:4px; pointer-events:none; display:none; z-index:5;';
    stripWrap.appendChild(tip);

    const applyWidths = () => {
      for (let i = 0; i < ramp.length; i++) {
        const w = (i === 0 ? fr[0] : (i === ramp.length - 1 ? 1 - fr[i - 1] : fr[i] - fr[i - 1])) * 100;
        cells[i].style.width = `${Math.max(0, w)}%`;
      }
    };
    const GAP = 0.015; // min spacing between adjacent handles
    for (let i = 0; i < need; i++) {
      const h = document.createElement('div');
      h.style.cssText = `position:absolute; top:-2px; bottom:-2px; width:9px; margin-left:-4.5px; left:${fr[i] * 100}%; background:rgba(255,255,255,0.9); border:1px solid rgba(15,23,42,0.55); border-radius:2px; cursor:ew-resize; box-shadow:0 1px 3px rgba(0,0,0,0.4);`;
      h.title = 'Drag to move this colour boundary';
      const idx = i;
      const showTip = (f: number) => { tip.textContent = handles!.formatValue(f); tip.style.left = `${f * 100}%`; tip.style.display = 'block'; };
      h.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        h.setPointerCapture(e.pointerId);
        showTip(fr[idx]);
        const onMove = (ev: PointerEvent) => {
          const rect = swatches.getBoundingClientRect();
          if (!rect.width) return;
          let f = (ev.clientX - rect.left) / rect.width;
          const loB = (idx === 0 ? GAP : fr[idx - 1] + GAP);
          const hiB = (idx === need - 1 ? 1 - GAP : fr[idx + 1] - GAP);
          f = Math.min(hiB, Math.max(loB, f));
          fr[idx] = f;
          h.style.left = `${f * 100}%`;
          applyWidths();                         // live legend feedback only (cheap, in place)
          showTip(f);                            // live value readout
          handles!.onChange(fr.slice(), false);  // remember positions; map recompute deferred to release
        };
        const onUp = () => {
          try { h.releasePointerCapture(e.pointerId); } catch { /* already released */ }
          h.removeEventListener('pointermove', onMove);
          h.removeEventListener('pointerup', onUp);
          tip.style.display = 'none';             // hide the readout once settled
          handles!.onChange(fr.slice(), true);   // commit → recompute + repaint once
        };
        h.addEventListener('pointermove', onMove);
        h.addEventListener('pointerup', onUp);
      });
      swatches.appendChild(h);
    }
  }
  wrap.appendChild(stripWrap);

  // Low → high value labels in the field's units.
  const scale = document.createElement('div');
  scale.style.cssText = 'display:flex; justify-content:space-between; font-size:11px; color:rgba(255,255,255,0.72);';
  const loEl = document.createElement('span'); loEl.textContent = formatLegendValue(field, lo);
  const hiEl = document.createElement('span'); hiEl.textContent = formatLegendValue(field, hi);
  scale.appendChild(loEl); scale.appendChild(hiEl);
  wrap.appendChild(scale);

  targetEl.appendChild(wrap);
  targetEl.style.display = 'grid';
}

function updateLegend() {
  const ramp = activeRamp();
  const lo = colorDomain?.lo ?? currentStats?.min ?? null;
  const hi = colorDomain?.hi ?? currentStats?.max ?? null;
  renderColorLegend(legendEl, {
    field: currentField,
    ramp,
    lo,
    hi,
    invert: !!invertHeights?.checked,
    // Handles are always available now; the tooltip formats the break value at a fraction using
    // the current scaling method's axis.
    handles: {
      fractions: manualFractions ?? [],
      onChange: onManualDrag,
      formatValue: (f) => formatLegendValue(currentField, breaksFromFractions([f], colorMode)[0]),
    },
  });
}

// Live-drag handler for the manual break handles. During a drag (committed=false) we only
// recompute breaks + repaint the map (the legend DOM is mutated in place by the handle itself,
// so we must NOT re-render it here or we'd kill the active pointer drag). On release we persist.
function onManualDrag(fractions: number[], committed: boolean) {
  const fr = fractions.slice();
  manualFractions = fr;
  // During the drag (committed=false) do NOTHING heavy — the handle moves the legend strip in
  // place; grab + move stay pure-DOM and snappy. On RELEASE, run the recompute + repaint inside a
  // rAF so the pointerup gesture returns immediately and the (chuggy) recolour never blocks input.
  if (!committed) return;
  requestAnimationFrame(() => {
    colorBreaks = breaksFromFractions(fr, colorMode);
    applyExtrusion();
    updateLegend();
    saveSettings(currentTab);
  });
}

function currentModeErrorMessage(props: Record<string, any>): string | null {
  if (normalizationMode === 'perLand' && landSizeField) {
    const v = Number((props as any)[landSizeField]);
    if (!Number.isFinite(v) || v <= 0) return '⚠ Invalid land size (≤ 0 or missing)';
  } else if (normalizationMode === 'perBuilding' && bldgSizeField) {
    const v = Number((props as any)[bldgSizeField]);
    if (Number.isFinite(v) && v < 0) return '⚠ Negative building size';
    if (v === 0) return 'ℹ Building size is 0 — shown flat (not an error)';
  }
  return null;
}

// HCAD's parcel-detail page (public.hcad.org/records/details.asp?crypt=…) requires an
// ENCRYPTED token, so the Houston tiles' baked links (…?crypt=<13-digit account>) 404.
// The raw account number is still recoverable from that URL, so we rebuild a working
// link here — fixing the already-deployed tiles with NO re-bake. (houston.ipynb is also
// corrected so future bakes emit the right URL directly.)
function buildHcadParcelUrl(acct: string): string {
  // HCAD's search portal, with the account carried in the URL so the user can look it
  // up directly. (A one-click deep-link to the detail record isn't publicly available.)
  return `https://search.hcad.org/?account=${encodeURIComponent(acct)}`;
}

function normalizeParcelLink(link: string): string {
  if (!link) return '';
  const brokenHcad = link.match(/details\.asp\?crypt=(\d{6,20})\s*$/i);
  if (brokenHcad) return buildHcadParcelUrl(brokenHcad[1]);
  return link;
}

function buildPopupHTML(props: Record<string, any>): string {
  const title = props.name ?? props.NAME ?? props.id ?? props.ID ?? '';
  const metric = computeDisplayedMetricFromProps(props);
  const heightM = metric != null ? computeExtrusionHeightMeters(metric) : null;
  const parcelLink = normalizeParcelLink(typeof props.link === 'string' ? props.link.trim() : '');
  const opportunityType = String(props?.[DEV_CATEGORY_FIELD] ?? '').trim();
  const landValuePerSqft = preferredLandValuePpsfField ? numOrNull(props?.[preferredLandValuePpsfField]) : null;

  const unitKey = unitsSelect.value as keyof typeof UNIT_TO_METERS;
  const unitText = (unitsSelect.options[unitsSelect.selectedIndex]?.text || unitKey);

  const fieldsToShow = ALL_FIELDS;
  const fieldKeysByLabel = new Map<string, string>();

  for (const k of fieldsToShow) {
    const label = FIELD_LABELS[k] || k;
    const value = (props as any)[k];
    const hasValue = value !== undefined && value !== null && value !== '';
    const existing = fieldKeysByLabel.get(label);

    if (!existing) {
      fieldKeysByLabel.set(label, k);
      continue;
    }

    const existingValue = (props as any)[existing];
    const existingHasValue = existingValue !== undefined && existingValue !== null && existingValue !== '';
    if (existingHasValue && !hasValue) continue;
    if (!existingHasValue && hasValue) {
      fieldKeysByLabel.set(label, k);
      continue;
    }

    if (existingHasValue && hasValue && isCoreField(existing) && !isCoreField(k)) {
      fieldKeysByLabel.set(label, k);
    }
  }

  const rowHtml = (label: string, printable: string) => `
      <tr>
        <td style="padding:2px 6px; overflow-wrap:anywhere;">
          <code style="white-space:normal;">${label}</code>
        </td>
        <td style="padding:2px 6px; text-align:right; white-space:nowrap;">
          ${printable}
        </td>
      </tr>`;

  // Derive parcel LAND size on demand: land area = land value ÷ land-value-per-sqft. Keyed off the
  // CANONICAL field keys (display labels vary per city — "Assessed" vs "Appraised", "/land ft²" vs
  // "per Sqft" — which is why the old label-based lookup matched nothing and no size ever showed).
  // Building floor area is NOT derivable here: every per-sqft metric divides by LAND area, so
  // improvement_value ÷ its per-sqft would just yield land area again. Shown after the land row.
  const LAND_VALUE_KEYS = ['current_full_land_value', 'REALLANDVA', 'land_value'];
  const numFromKeys = (keys: string[]): number | null => {
    for (const k of keys) { const v = numOrNull((props as any)[k]); if (v != null) return v; }
    return null;
  };
  const landVal = numFromKeys(LAND_VALUE_KEYS);
  const landPpsf = numFromKeys(['land_value_per_sqft', 'REALLANDVA_per_sqft']);
  const landSize = (landVal != null && landPpsf != null && landPpsf > 0) ? landVal / landPpsf : null;

  // Combined-value cities have no land/building split, so the improvement/ratio/share fields are
  // structurally absent — hide them instead of rendering a column of meaningless "—" rows.
  const HIDE_FOR_COMBINED = new Set(['REALIMPROV', 'REALIMPROV_per_sqft', 'improvement_value_per_sqft',
    'TLLDIMPROV', 'TLLDIMPROV_per_sqft', 'full_market_value_per_sqft',
    'IMPR_LAND_RATIO', 'IMPR_LAND_PCT', 'IMPR_PCT_TOTAL', 'LAND_PCT_TOTAL']);
  const rows = Array.from(fieldKeysByLabel.entries()).filter(([, k]) =>
    !(COMBINED_VALUE_ONLY && HIDE_FOR_COMBINED.has(k))).map(([label, k]) => {
    const v = (props as any)[k];
    // Jurisdiction group fields ship int-encoded in PMTiles (id, not name) — decode to the
    // region name so the popup shows "West Hartford", not "0". No-op for non-group fields.
    const decoded = jurisdiction.nameForId(k, v);
    const printable = decoded ?? ((typeof v === 'number') ? fmt(v) : (v ?? '—'));
    let html = rowHtml(label, printable);
    if (LAND_VALUE_KEYS.includes(k) && landSize != null)
      html += rowHtml('Land Size', `${fmt(Math.round(landSize))} sq ft`);
    return html;
  }).join('');

  const modeLabel =
    normalizationMode === 'perLand' ? `per ${landSizeField || 'land size'}` :
    normalizationMode === 'perBuilding' ? `per ${bldgSizeField || 'building size'}` :
    'as-is';

  // @ts-expect-error TS6133: reserved for future use
  const _metricRow = (metric != null)
    ? `<div><strong>Display metric (${modeLabel})</strong>: ${fmt(metric)}</div>`
    : `<div><strong>Display metric</strong>: —</div>`;

  // @ts-expect-error TS6133: reserved for future use
  const _heightRow = (heightM != null)
    ? `<div><strong>Extrusion height</strong>: ${fmt(heightM / (UNIT_TO_METERS[unitKey] || 1))} ${unitText} (${fmt(heightM)} m)</div>`
    : `<div><strong>Extrusion height</strong>: —</div>`;

  const errMsg = currentModeErrorMessage(props);
  const errRow = errMsg ? `<div style="margin-top:4px;color:#b00020;">${errMsg}</div>` : '';

  const linkButton = parcelLink
    ? `<div style="display:flex; justify-content:flex-end; margin-bottom:6px;">
        <a href="${parcelLink}" target="_blank" rel="noopener"
           style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px;
                  background:#eef3ff; color:#1b4dd8; text-decoration:none; border-radius:6px;
                  font-weight:600; font-size:12px;">
          <span>View Parcel</span><span aria-hidden="true" style="font-size:12px;">↗</span>
        </a>
       </div>`
    : '';

  const underSummary = currentTab === 'under'
    ? `<div style="display:grid; gap:4px; margin-bottom:8px;">
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; color:#64748b;">Opportunity parcel</div>
        <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
          ${opportunityType ? `<span style="display:inline-flex; align-items:center; padding:4px 8px; border-radius:999px; background:#eef2ff; color:#1e3a8a; font-weight:700;">${opportunityType}</span>` : ''}
          ${landValuePerSqft != null ? `<span style="font-weight:700; color:#0f172a;">${fmtCurrencyPerSqft(landValuePerSqft)}</span>` : ''}
        </div>
      </div>`
    : '';

  return `
    <div class="gvw-pop" style="max-width:min(92vw, 460px); font-size:12.5px; line-height:1.35;">
      ${linkButton}
      ${title ? `<div style="font-weight:600;margin-bottom:4px; overflow-wrap:anywhere;">${title}</div>` : ''}
      ${underSummary}
      ${errRow}
      <div style="height:1px;background:#eee;margin:6px 0"></div>
      <div style="font-weight:600;margin-bottom:2px">Loaded fields</div>
      <div style="overflow:auto;">
        <table style="width:100%; border-collapse:collapse; font-size:12px; table-layout:fixed;">
          <colgroup>
            <col span="1" style="width:65%">
            <col span="1" style="width:35%">
          </colgroup>
          ${rows}
        </table>
      </div>
    </div>`;
}

function onMultInput() {
  const v = Number(multInput.value);
  if (!Number.isFinite(v)) return; // ignore interim typing states
  scheduleUpdate('applyOnly');
}

/* ---------------- Events ---------------- */

// Only recompute after data is loaded
colorScaleSelect?.addEventListener('change', () => {
  // PMTiles cities have no currentGeoJSON but still render from metadata — don't
  // bail early or the Linear/Quantiles dropdown silently does nothing for them.
  if (!currentGeoJSON && !(cityUsesPmtiles() && pmtilesMetadata)) return;
  const val = colorScaleSelect.value;
  if (val === 'linear' || val === 'quantiles' || val === 'log') {
    colorMode = val;
    // Switching method reseeds the handles evenly = the new method's natural scaling; the user
    // then adjusts levels within it. (Reset does the same.)
    manualFractions = null;
    scheduleUpdate('recomputeAndAutoScale', /*refreshLegend*/ true);
    saveSettings(currentTab);
  }
});

// 3D-orbit help pill: the "?" circle collapses/expands the floating tip over the map.
const orbitTip = document.getElementById('orbitTip');
document.getElementById('orbitTipToggle')?.addEventListener('click', () => {
  const collapsed = orbitTip?.getAttribute('data-collapsed') === 'true';
  orbitTip?.setAttribute('data-collapsed', collapsed ? 'false' : 'true');
});

// ── 3D/2D mode toggle (main Land value map) ──────────────────────────────────
function lockMapRotation(lock: boolean) {
  const touchPitch = (map as any).touchPitch;   // present in maplibre-gl v2+
  if (lock) { map.dragRotate.disable(); map.touchZoomRotate.disableRotation(); touchPitch?.disable?.(); }
  else { map.dragRotate.enable(); map.touchZoomRotate.enableRotation(); touchPitch?.enable?.(); }
}
/** Reflect the current mode on the button label, the Height slider, and the orbit tip. */
function refresh3DUI() {
  if (toggle3DBtn) toggle3DBtn.textContent = is3D ? 'Disable 3D' : 'Enable 3D';
  if (heightScaleMain) {
    heightScaleMain.disabled = !is3D;
    const grp = heightScaleMain.closest('.control-group') as HTMLElement | null;
    if (grp) { grp.style.opacity = is3D ? '' : '0.45'; grp.style.pointerEvents = is3D ? '' : 'none'; }
  }
  // The "3D orbit" hint is meaningless when flat; hide it in 2D (respecting the user dismiss).
  if (orbitTip) orbitTip.style.display = !is3D ? 'none' : (orbitTipDismissed ? 'none' : '');
}
function setMode3D(threeD: boolean, animate = true) {
  is3D = threeD;
  try { localStorage.setItem(MODE_3D_KEY, threeD ? '3d' : '2d'); } catch {}
  if (!threeD) {
    if (map.getPitch() > 0) savedPitch = map.getPitch();   // remember the tilt to restore
    lockMapRotation(true);
    map[animate ? 'easeTo' : 'jumpTo']({ pitch: 0, bearing: 0, ...(animate ? { duration: 300 } : {}) });
  } else {
    lockMapRotation(false);
    map[animate ? 'easeTo' : 'jumpTo']({ pitch: savedPitch, ...(animate ? { duration: 300 } : {}) });
  }
  refresh3DUI();
  applyExtrusion();   // re-set extrusion heights (flattened to 0 in 2D)
}
toggle3DBtn?.addEventListener('click', () => setMode3D(!is3D));
// Apply the persisted mode on load (snap without animation). 3D needs no camera change.
refresh3DUI();
if (!is3D) setMode3D(false, false);

// ── region boundary overlays: outlines for the regions whose square toggle is on ────────────
const OVERLAY_SRC = 'region-overlays';
const OVERLAY_FILL = 'region-overlays-fill';
const OVERLAY_HATCH = 'region-overlays-hatch';
const OVERLAY_LINE = 'region-overlays-line';
const OVERLAY_LABEL = 'region-overlays-label';
// Palette lives in jurisdiction.ts so the list swatch and the map overlay use the SAME region→
// color mapping (by region order, not feature order).
const OVERLAY_PALETTE = jurisdiction.OVERLAY_PALETTE;
const overlayCache = new Map<string, any>();
let overlayGen = 0;

/** Add a thin diagonal crosshatch image (in each palette color) for the overlay fill pattern.
 *  Drawn slightly past the tile edges so the lines tile seamlessly. */
function removeOverlayLayers() {
  for (const id of [OVERLAY_LABEL, OVERLAY_LINE, OVERLAY_HATCH, OVERLAY_FILL]) {
    if (map.getLayer(id)) map.removeLayer(id);
  }
  if (map.getSource(OVERLAY_SRC)) map.removeSource(OVERLAY_SRC);
}

/** Draw (or clear) boundary overlays for the active group's overlay-toggled regions: a thick inset
 *  border in the region's categorical color (no fill/tint or hatch — outline only). Driven by the
 *  per-row square toggles via jurisdiction.getOverlayRegions(). */
async function refreshOverlays() {
  const gen = ++overlayGen;
  removeOverlayLayers();
  let regions = jurisdiction.getOverlayRegions();
  if (!regions.size) return;
  const file = jurisdiction.getActiveOverlayUrl();
  if (!file) return;
  const url = `${import.meta.env.BASE_URL}${file}`;
  let gj = overlayCache.get(url);
  if (!gj) {
    try { gj = await (await fetch(url)).json(); } catch { return; }
    // Consistent winding so the inset line-offset goes INWARD for every region (the source data
    // has mixed winding) — lets adjoining regions each show their own boundary band.
    normalizeWindingInPlace(gj.features || []);
    // Color each region by the SAME region→palette index the list uses, so the on-map outline
    // matches the swatch next to it in the picker.
    (gj.features || []).forEach((f: any) => {
      f.properties = f.properties || {};
      const idx = jurisdiction.regionColorIndex(f.properties.name);
      f.properties.__ovColor = OVERLAY_PALETTE[idx];
    });
    overlayCache.set(url, gj);
  }
  // The set / active group may have changed during the fetch — re-read and bail if stale.
  regions = jurisdiction.getOverlayRegions();
  if (gen !== overlayGen || !regions.size || jurisdiction.getActiveOverlayUrl() !== file) return;
  removeOverlayLayers();
  const filter = ['in', ['get', 'name'], ['literal', Array.from(regions)]] as any;
  map.addSource(OVERLAY_SRC, { type: 'geojson', data: gj });
  map.addLayer({
    id: OVERLAY_LINE, type: 'line', source: OVERLAY_SRC, filter,
    layout: { 'line-join': 'round' },
    // Inset the stroke by half its width (winding normalized above) so each region's outline hugs
    // the inside of its own boundary; on a shared edge both colors show side by side.
    paint: { 'line-color': ['get', '__ovColor'], 'line-width': 4, 'line-offset': 2, 'line-opacity': 1 },
  });
  // Region-name labels for the same overlaid regions, when "Overlay labels" is on.
  if (jurisdiction.labelsEnabled()) {
    map.addLayer({
      id: OVERLAY_LABEL, type: 'symbol', source: OVERLAY_SRC, filter,
      layout: {
        'text-field': ['get', 'name'], 'text-font': ['Noto Sans Regular'],
        'text-size': 12, 'text-allow-overlap': false, 'text-padding': 2, 'symbol-placement': 'point',
      },
      paint: { 'text-color': '#1b1b1b', 'text-halo-color': '#ffffff', 'text-halo-width': 1.4 },
    });
  }
}

rampSelect.addEventListener('change', () => {
  // quantiles + manual depend on ramp length (break count) → recompute; continuous re-interpolates.
  // Every method derives colorBreaks from ramp.length, so a ramp change always needs a recompute.
  scheduleUpdate('recomputeAndAutoScale', /*refreshLegend*/ true);
  saveSettings(currentTab);
  renderUnderNow();
  renderRatioNow();
});

// ── Invert colors + Reset levels — the balanced two-button row under the legend ──
const invertColorsBtn = document.getElementById('invertColorsBtn') as HTMLButtonElement | null;
const resetScaleBtn = document.getElementById('resetScaleBtn') as HTMLButtonElement | null;

function updateInvertColorsBtn() {
  invertColorsBtn?.classList.toggle('btn-primary', colorInvert);  // lit when active
  invertColorsBtn?.classList.toggle('btn-ghost', !colorInvert);
}

invertColorsBtn?.addEventListener('click', () => {
  colorInvert = !colorInvert;
  updateInvertColorsBtn();
  // Only the ramp DIRECTION changes; break values are unchanged, so repaint + refresh legend.
  applyExtrusion();
  updateLegend();
  saveSettings(currentTab);
});

// Reset re-spaces the handles evenly → the current method's natural scaling (always available now).
resetScaleBtn?.addEventListener('click', () => {
  const ramp = activeRamp();
  manualFractions = seedManualFractions(Math.max(1, ramp.length - 1));
  colorBreaks = breaksFromFractions(manualFractions, colorMode);
  applyExtrusion();
  updateLegend();
  saveSettings(currentTab);
});

// Initial state from restored settings.
updateInvertColorsBtn();

multInput.addEventListener('input', () => { onMultInput(); saveSettings(currentTab); });

multInput.addEventListener('change', () => { onMultInput(); saveSettings(currentTab); });

unitsSelect.addEventListener('change', () => { scheduleUpdate('applyOnly'); saveSettings(currentTab); });

opacityInput.addEventListener('input', () => {
  if (opacityOut) opacityOut.value = `${parseInt(opacityInput.value).toFixed(0)}%`;
  scheduleUpdate('applyOnly');
  saveSettings(currentTab);
  renderUnderNow();
  renderRatioNow();
});

// Under map listeners
underOpacityInput?.addEventListener('input', () => { if (underOpacityOut) underOpacityOut.value = `${parseInt(underOpacityInput.value).toFixed(0)}%`; renderUnderNow(); });

// Ratio map listeners
ratioRampSelect?.addEventListener('change', () => { renderRatioNow(); });
ratioOpacityInput?.addEventListener('input', () => { if (ratioOpacityOut) ratioOpacityOut.value = `${parseInt(ratioOpacityInput.value).toFixed(0)}%`; renderRatioNow(); });
ratioInvertHeights?.addEventListener('change', () => { renderRatioNow(); });
ratioMultInput?.addEventListener('input', () => { renderRatioNow(); });
ratioFieldSelect?.addEventListener('change', () => { renderRatioNow(); });

// Height slider listeners (bottom-right)
heightScaleMain?.addEventListener('input', () => {
  const v = parseInt(heightScaleMain.value, 10);
  // Slider range 0–200; /100 means 100 (default, halfway) = 1× and 200 = 2× the prior max.
  heightFactorMain = Number.isFinite(v) ? Math.max(0, Math.min(200, v)) / 100 : 1;
  scheduleUpdate('applyOnly');
});
heightScaleRatio?.addEventListener('input', () => {
  const v = parseInt(heightScaleRatio.value, 10);
  heightFactorRatio = Number.isFinite(v) ? Math.max(0, Math.min(100, v)) / 100 : 1;
  renderRatioNow();
});

fieldSelect.addEventListener('change', () => {
  fitFieldSelectFont();
  currentField = fieldSelect.value || null;
  if (!currentField) return;
  updateSmoothToggleStateFromField(currentField);
  updateLandBlurb();
  if (currentGeoJSON || (cityUsesPmtiles() && pmtilesMetadata)) {
    scheduleUpdate('recomputeAndAutoScale', /*refreshLegend*/ true);
  }
  saveSettings(currentTab);
});

// Refit the picker label when the sidebar/viewport width changes.
let _fieldFitRaf = 0;
window.addEventListener('resize', () => {
  if (_fieldFitRaf) cancelAnimationFrame(_fieldFitRaf);
  _fieldFitRaf = requestAnimationFrame(() => { _fieldFitRaf = 0; fitFieldSelectFont(); });
});

// PMTiles metadata type
type PmtilesMetadata = {
  statistics: Record<string, { min: number; max: number }>;
  categories: {
    refined: string[];
    original: string[];
  };
  underutilizedTotals?: Record<string, number>;
  quantileBreaks?: Record<string, number[]>;
  percentiles?: Record<string, { p1: number; p99: number; p999?: number }>;
  /** Geographic extent of the dataset as [minLon, minLat, maxLon, maxLat] (EPSG:4326). */
  bounds?: [number, number, number, number];
  /** PMTiles source-layer for the underutilized tab. When the bake ships the all-zoom
   *  underutilized subset it's 'parcels_under'; cities baked before that lack the key,
   *  so the underutilized tab falls back to the z13+ 'parcels' layer. */
  underutilizedSourceLayer?: string;
  /** Hex->parcel handoff zoom decided by the bake (plan_h3_ladder prunes hex bands finer
   *  than the city's median parcel and hands off earlier). Baked+uploaded together with the
   *  tiles, so it overrides the dictionary's static parcelMinZoom; tiles baked before this
   *  existed lack the key and keep the dictionary value (13). */
  parcelMinZoom?: number;
  /** PROTOTYPE: jurisdiction -> parcel count, used to populate the enclave dropdown.
   *  Back-compat alias for groups.jurisdiction (kept for tiles baked before `groups`). */
  jurisdictions?: Record<string, number>;
  /** Region grouping schemes, one entry per categorical grouping field the bake tagged.
   *  New (int-encoded) shape: { ids: [name_for_id_0, ...], counts: {name: parcelCount} } — tiles
   *  store the integer id, ids[i] recovers the name. Legacy tiles used a flat {name: count}.
   *  Handled loosely in buildRegionGroupConfigs. */
  groups?: Record<string, any>;
  /** Citywide {acres, land, impr, total} — feeds the Land Value blurb when no region widget is
   *  active (PMTiles parcels aren't in browser memory). Metadata baked/patched before this
   *  existed lacks the key; the blurb then stays hidden for region-less cities as before. */
  cityTotals?: ScopeTotals;
};

let pmtilesMetadata: PmtilesMetadata | null = null;

/** Build the region-group configs for jurisdiction.configure() from a city def.
 *  - `counts` (region -> parcel count) populates the list; from PMTiles metadata.groups when
 *    available, else configure() scans the GeoParquet features.
 *  - `ids` (id-ordered names) is present only for int-encoded tiles; it lets jurisdiction map
 *    selected names -> integer ids for the layer filter (tiles store ids, not strings). */
function buildRegionGroupConfigs(cityDef: any, groupsMeta?: Record<string, any>, jurisdictionsFallback?: Record<string, number>) {
  const defs: any[] = (cityDef.jurisdictionGroups && cityDef.jurisdictionGroups.length)
    ? cityDef.jurisdictionGroups
    : cityDef.jurisdictionField
      ? [{ field: cityDef.jurisdictionField, label: 'Region', primary: cityDef.primaryJurisdiction,
           defaultMode: cityDef.primaryJurisdiction ? 'primaryOnly' : 'all' }]
      : [];
  return defs.map(g => {
    const gm = groupsMeta?.[g.field];
    let counts: Record<string, number> | undefined;
    let ids: string[] | undefined;
    if (gm && Array.isArray(gm.ids)) { ids = gm.ids; counts = gm.counts; }              // int-encoded
    else if (gm) { counts = gm as Record<string, number>; }                             // legacy flat
    else if (g.field === cityDef.jurisdictionField && jurisdictionsFallback) counts = jurisdictionsFallback;
    return { field: g.field, label: g.label, primary: g.primary, defaultMode: g.defaultMode, overlayUrl: g.overlayUrl, counts, ids };
  });
}

// PROTOTYPE: PMTiles context for the localhost-only 3D-print export (reads hex tiles directly).
let print3dArchive: PMTiles | null = null;
function get3DPrintContext(): Print3DContext | null {
  const cfg = getCityConfig();
  if (!cityUsesPmtiles() || !pmtilesMetadata?.bounds || !cfg?.pmtilesUrl) return null;
  const parcelMinZoom = Number(cfg.parcelMinZoom);
  if (!Number.isFinite(parcelMinZoom)) return null;
  if (!print3dArchive) print3dArchive = new PMTiles(getPmtilesUrl(cfg.pmtilesUrl));
  return {
    archive: print3dArchive,
    bounds: pmtilesMetadata.bounds as [number, number, number, number],
    sourceLayer: 'parcels_low',
    parcelMinZoom,
  };
}

type NormalizedUnderTotals = {
  Vacant: number;
  'Parking Lot': number;
  Underdeveloped: number;
  totalNonExempt: number;
  // Present when the bake emitted per-bucket keys (Underdeveloped_lt10, …).
  underdevelopedBuckets?: { label: string; total: number; count?: number }[];
};

function normalizeUnderutilizedTotals(totals?: Record<string, number>): NormalizedUnderTotals | null {
  if (!totals || typeof totals !== 'object') return null;
  const pick = (...keys: string[]) => {
    for (const key of keys) {
      const value = Number(totals[key]);
      if (Number.isFinite(value)) return value;
    }
    return 0;
  };

  const normalized: NormalizedUnderTotals = {
    Vacant: pick('Vacant', 'vacant'),
    'Parking Lot': pick('Parking Lot', 'parking_lot', 'parkingLot', 'ParkingLot'),
    Underdeveloped: pick('Underdeveloped', 'underdeveloped', 'Under Developed', 'under_developed'),
    totalNonExempt: pick('totalNonExempt', 'total_non_exempt', 'TotalNonExempt')
  };

  // Improvement-share breakdown, when baked into the metadata.
  const buckets = UNDERDEV_BUCKETS.map((b) => ({
    label: b.label,
    total: pick(`Underdeveloped_${b.key}`),
    count: Number(totals[`Underdeveloped_${b.key}_count`]) || undefined,
  }));
  if (buckets.some(b => b.total > 0)) normalized.underdevelopedBuckets = buckets;

  const hasAnyValue = [normalized.Vacant, normalized['Parking Lot'], normalized.Underdeveloped, normalized.totalNonExempt].some(v => v > 0);
  return hasAnyValue ? normalized : null;
}

function dedupeFieldListByLabel(fields: string[]): string[] {
  const seenLabels = new Set<string>();
  const result: string[] = [];
  for (const field of fields) {
    const label = (FIELD_LABELS[field] ?? field).trim().toLowerCase();
    if (seenLabels.has(label)) continue;
    seenLabels.add(label);
    result.push(field);
  }
  return result;
}

async function loadPmtilesMetadata(): Promise<PmtilesMetadata | null> {
  const config = getCityConfig();
  if (!config?.pmtilesMetadataUrl) {
    console.warn('[PMTiles] No metadata URL in city config', config);
    return null;
  }

  try {
    const metadataUrl = getPmtilesUrl(config.pmtilesMetadataUrl);
    console.log('[PMTiles] Loading metadata from:', metadataUrl);
    showLoading('Loading PMTiles metadata…');
    const response = await fetch(metadataUrl, {});
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText} - URL: ${metadataUrl}`);
    }
    const metadata = await response.json() as PmtilesMetadata;
    // LAND_PCT_TOTAL (land's share of total value) is the complement of IMPR_PCT_TOTAL. The bake
    // doesn't emit it, so synthesize its statistics from IMPR_PCT_TOTAL's — the PMTiles color/scale
    // machinery + the dropdown allowlist both key off metadata.statistics, so this enables the field
    // without a re-bake. (Value/legend are still computed from the raw REALLANDVA/REALIMPROV tiles.)
    if (metadata.statistics && !metadata.statistics.LAND_PCT_TOTAL) {
      const impr = metadata.statistics.IMPR_PCT_TOTAL;
      metadata.statistics.LAND_PCT_TOTAL = impr
        ? { min: Math.max(0, 100 - impr.max), max: Math.min(100, 100 - impr.min) }
        : { min: 0, max: 100 };
    }
    // The bake decides where hexes end and real parcels begin (it may prune hex bands
    // finer than the city's median parcel and hand off before z13). That zoom ships in
    // this metadata JSON alongside the tiles it was baked with, so it must win over the
    // dictionary's static parcelMinZoom. Mutate the shared config object in place —
    // every consumer reads it live via getCityConfig(), and this runs before the
    // PMTiles source/layers are added.
    const bakedPmz = Number(metadata.parcelMinZoom);
    if (Number.isFinite(bakedPmz) && config) {
      config.parcelMinZoom = bakedPmz;
    }
    pmtilesMetadata = metadata;
    console.log('[PMTiles] Loaded metadata:', {
      fieldCount: Object.keys(metadata.statistics).length,
      refinedCategories: metadata.categories.refined.length,
      originalCategories: metadata.categories.original.length
    });
    return metadata;
  } catch (err) {
    console.error('[PMTiles] Failed to load metadata:', err);
    throw err; // Re-throw so caller can handle it
  }
}

async function loadPmtilesDataset() {
  const config = getCityConfig();
  console.log('[PMTiles] Loading dataset with config:', config);
  if (!config?.pmtilesUrl) {
    throw new Error(`PMTiles URL not configured. Config: ${JSON.stringify(config)}`);
  }

  showLoading('Loading PMTiles dataset…');

  try {
    // Load metadata first
    const metadata = await loadPmtilesMetadata();
    if (!metadata) {
      throw new Error('Failed to load PMTiles metadata');
    }

    // Construct PMTiles URL (use API proxy URL for the pmtiles:// protocol)
    const pmtilesFileUrl = getPmtilesUrl(config.pmtilesUrl);
    const pmtilesUrl = `pmtiles://${pmtilesFileUrl}`;
    console.log('[PMTiles] Loading from:', pmtilesUrl);

    // Initial bounds for the current city. We must NOT hard-code a fallback to any
    // specific city here — doing so previously meant that when the authoritative
    // bounds weren't ready in time, EVERY PMTiles city (Houston, NYC, Baltimore…)
    // would fit to that hard-coded extent. Instead, read the dataset's true extent
    // from the PMTiles header (present in every baked file, no re-bake required),
    // falling back to the bake metadata, then to the vector source's own bounds.
    let bounds: [[number, number], [number, number]] | null = null;

    const boundsFromExtent = (b: ArrayLike<number> | undefined | null):
      [[number, number], [number, number]] | null => {
      if (!b || b.length !== 4) return null;
      const [minLon, minLat, maxLon, maxLat] = [b[0], b[1], b[2], b[3]];
      if (![minLon, minLat, maxLon, maxLat].every(Number.isFinite)) return null;
      if (minLon === maxLon || minLat === maxLat) return null;
      return [[minLon, minLat], [maxLon, maxLat]];
    };

    // 1) Authoritative: the PMTiles header carries the dataset's geographic extent.
    try {
      const header = await new PMTiles(pmtilesFileUrl).getHeader();
      bounds = boundsFromExtent([header.minLon, header.minLat, header.maxLon, header.maxLat]);
      if (bounds) console.log('[PMTiles] Bounds from header:', bounds);
    } catch (e) {
      console.warn('[PMTiles] Could not read header bounds:', e);
    }
    // 2) Fallback: bounds baked into the metadata JSON.
    if (!bounds) bounds = boundsFromExtent(metadata.bounds);

    // Add PMTiles as vector source to all maps
    const addPmtilesSource = (m: maplibregl.Map) => {
      if (m.getSource(SOURCE_ID)) {
        m.removeSource(SOURCE_ID);
      }
      m.addSource(SOURCE_ID, {
        type: 'vector',
        url: pmtilesUrl
      });
    };

    // Wait for maps to be ready and source to load
    const addSourceWhenReady = (m: maplibregl.Map, withClick = false, isMainMap = false) => {
      const addSourceAndLayer = () => {
        addPmtilesSource(m);
        if (m === mapUnder) addPmtilesUnderLayer(m, withClick);
        else addPmtilesExtrusionLayer(m, withClick);

        if (!isMainMap) {
          if (!(m === mapUnder && restoredCameras.umap)) syncMapView(map, m);
          // Guaranteed initial paint for the secondary (under / ratio) maps. Their
          // layers exist now, but the vector source and the category metadata may not
          // be ready yet, so a synchronous render would early-return grey. Repaint when
          // the source finishes loading AND once the map first goes idle (covers loading
          // directly onto this tab). renderUnder/RatioNow are idempotent + cheap, and
          // each early-returns until its layer exists, so firing repeatedly is safe.
          const repaintSecondary = () => {
            if (m === mapUnder) renderUnderNow();
            else if (m === mapRatio) renderRatioNow();
          };
          m.on('sourcedata', (e: any) => {
            if (e.sourceId === SOURCE_ID && e.isSourceLoaded) repaintSecondary();
          });
          m.once('idle', repaintSecondary);
        }

        // Wait for source to load, then fit bounds and apply extrusions
        if (isMainMap) {
          let boundsFitted = false;
          let extrusionsApplied = false;
          let applyInFlight = false;
          // Verbose per-retry/per-tile load logs are a real CPU drag (Chrome serializes each arg)
          // and flooded profiling — off unless the VERBOSE diagnostic flag is set (?debug=1).
          const dlog = vlog;

          const applyExtrusionsWhenReady = (retryCount = 0) => {
            if (extrusionsApplied) return;
            
            // Check if source and layer exist
            const source = m.getSource(SOURCE_ID);
            const layer = m.getLayer(LAYER_ID);
            
            if (!source || !layer) {
              dlog('[PMTiles] Waiting for source/layer...', {
                hasSource: !!source,
                hasLayer: !!layer
              });
              if (retryCount < 10) {
                setTimeout(() => applyExtrusionsWhenReady(retryCount + 1), 200);
              }
              return;
            }
            
            // Check if source is loaded (for vector sources, check if tiles are available)
            const sourceState = (source as any).state;
            if (sourceState === 'unavailable' || sourceState === 'loading') {
              dlog('[PMTiles] Source still loading, state:', sourceState);
              if (retryCount < 10) {
                setTimeout(() => applyExtrusionsWhenReady(retryCount + 1), 200);
              }
              return;
            }
            
            // 3) Last resort: the vector source's own bounds (only if header +
            //    metadata both came up empty — avoids ever fitting to stale/default extent).
            if (!boundsFitted) {
              if (!bounds) {
                const srcBounds = boundsFromExtent((source as any).bounds);
                if (srcBounds) {
                  bounds = srcBounds;
                  dlog('[PMTiles] Got bounds from source:', bounds);
                }
              }

              // Fit to bounds
              if (bounds) {
                if (!restoredCameras.map) m.fitBounds(bounds, { padding: 40, duration: 800 });
                dlog('[PMTiles] Fitted map to bounds');
                boundsFitted = true;
                window.setTimeout(() => {
                  if (!restoredCameras.umap) syncMapView(map, mapUnder);
                  if (mapRatio) syncMapView(map, mapRatio);
                  if (currentTab === 'under') renderUnderNow();
                  if (currentTab === 'ratio') renderRatioNow();
                }, 900);
                // Wait for fitBounds to complete before checking for features
                setTimeout(() => applyExtrusionsWhenReady(retryCount), 900);
                return;
              }
            }
            
            // Apply extrusions once the source is ready AND ≥1 feature is queryable. The two
            // diagnostic queries that used to run here every retry (querySourceFeatures + a
            // vectorLayers scan) were dropped — they're expensive and ran on every chain.
            if (currentField && currentStats) {
              let queryResult: any[] = [];
              try { queryResult = m.queryRenderedFeatures({ layers: [LAYER_ID] }); } catch { /* tiles not ready */ }
              dlog('[PMTiles] queryable features:', queryResult.length, 'retry:', retryCount);

              // Not ready yet → keep polling within THIS chain (single-flight; see requestApply).
              if (queryResult.length === 0 && retryCount < 20) {
                setTimeout(() => applyExtrusionsWhenReady(retryCount + 1), 200);
                return;
              }
              if (queryResult.length === 0) {
                console.warn('[PMTiles] No features queryable after retries; applying anyway (they appear as tiles load).');
              }

              computeAndApplyAutoMultiplier('auto', HEIGHT_CAPS.main, HEIGHT_PCTL);
              updateLegend();
              const finalMultiplier = Number(multInput.value);
              if (!Number.isFinite(finalMultiplier) || finalMultiplier === 0) {
                console.error('[PMTiles] WARNING: multiplier is 0/invalid — extrusions will not be visible.');
              }
              dlog('[PMTiles] auto-scale + extrusions applied', { multiplier: finalMultiplier, field: currentField });
              extrusionsApplied = true;
              finishApply();
            } else if (retryCount < 40) {
              // field/stats are assigned later in load; keep polling rather than locking in grey.
              setTimeout(() => applyExtrusionsWhenReady(retryCount + 1), 250);
            } else {
              console.warn('[PMTiles] field/stats never became ready; applying without auto-scale');
              applyExtrusion();
              extrusionsApplied = true;
              finishApply();
            }
          };
          
          // SINGLE-FLIGHT: external triggers (sourcedata/data/load) route through requestApply so
          // only ONE retry chain ever runs — the bug before was that every tile's `data` event
          // spawned its own chain, each polling queryRenderedFeatures every 200ms, saturating the
          // main thread (blocked ~768 ms/s on load). Internal setTimeout recursion continues the
          // active chain; once applied we detach the listeners so they can't restart it.
          function finishApply() {
            applyInFlight = false;
            try { m.off('sourcedata', onSourceData); m.off('data', onData); } catch { /* noop */ }
          }
          function requestApply() {
            if (extrusionsApplied || applyInFlight) return;
            applyInFlight = true;
            applyExtrusionsWhenReady(0);
          }
          function onSourceData(e: any) {
            if (e.sourceId !== SOURCE_ID || !e.isSourceLoaded) return;
            requestApply();
          }
          function onData(e: any) {
            if (e.sourceId !== SOURCE_ID || extrusionsApplied) return;
            requestApply();
          }
          m.on('sourcedata', onSourceData);
          m.on('data', onData);

          if (m.loaded()) requestApply();
          else m.once('load', () => requestApply());

          // Guaranteed initial paint for the main map: the data/sourcedata handlers
          // can run before currentField/currentStats are assigned (they're set later
          // in the load), leaving the layer grey + flat until a manual zoom. 'idle'
          // fires once the fit animation + tiles have settled, by which point those
          // are ready — so recompute the auto-scale and repaint right then.
          if (m === map) {
            m.once('idle', () => {
              if (currentField && currentStats) {
                computeAndApplyAutoMultiplier('auto', HEIGHT_CAPS.main, HEIGHT_PCTL);
                updateLegend();
              }
            });
          }
        }
      };
      
      if ((m as any).isStyleLoaded && (m as any).isStyleLoaded()) {
        addSourceAndLayer();
      } else {
        m.once('load', addSourceAndLayer);
      }
    };

    addSourceWhenReady(map, true, true);
    addSourceWhenReady(mapUnder, true, false);
    if (mapRatio) addSourceWhenReady(mapRatio, true, false);

    // Use metadata for statistics and categories
    // Set currentField from available fields in metadata.
    // Restrict to the same picker allowlist the GeoParquet path uses (DROPDOWN_FIELDS) so the
    // PMTiles dropdown only shows per-land-sqft metrics + the ratio — NOT base value fields
    // (land/improvement/full_market value, TLLDIMPROV), raw codes, or carried denominators like
    // land_area_acres. Those stay in the tiles/metadata for popups & client-side math; just hidden here.
    let availableFields = dedupeFieldListByLabel(
      Object.keys(metadata.statistics).filter(k => DROPDOWN_FIELDS.includes(k))
    );

    // For Denver, filter to only show specific fields (like Spokane)
    if (SELECTED_CITY === 'denver') {
      const denverFields = [
        'land_value_per_sqft',
        'IMPR_LAND_PCT',
        'improvement_value_per_sqft',
        'full_market_value_per_sqft'
      ];
      availableFields = availableFields.filter(f => denverFields.includes(f));
      console.log('[PMTiles] Filtered fields for Denver:', availableFields);
    }
    
    // Populate field dropdown first (needed for loadSettings validation)
    populateFieldDropdownFromList(availableFields);
    
    // Always auto-select land price per square foot (override saved settings)
    // Check for both naming conventions: land_value_per_sqft and REALLANDVA_per_sqft
    const landPricePerSqftField = availableFields.find(f => 
      f === 'land_value_per_sqft' || f === 'REALLANDVA_per_sqft'
    ) || availableFields.find(f => f.includes('land') && f.includes('per_sqft'));
    preferredLandValuePpsfField = landPricePerSqftField || null;
    if (landPricePerSqftField) {
      currentField = landPricePerSqftField;
      fieldSelect.value = landPricePerSqftField;
      const stats = metadata.statistics[landPricePerSqftField];
      if (stats) {
        currentStats = { min: stats.min, max: stats.max };
      }
      console.log(`[PMTiles] Auto-selected land price per sqft: ${landPricePerSqftField}`, currentStats);
    } else {
      // Fallback: use first available field if land price per sqft not found
      console.warn('[PMTiles] Land price per sqft field not found, using first available field');
      if (availableFields.length > 0) {
        currentField = availableFields[0];
        fieldSelect.value = availableFields[0];
        const stats = metadata.statistics[availableFields[0]];
        if (stats) {
          currentStats = { min: stats.min, max: stats.max };
        }
      }
    }
    
    // Load saved settings after setting the field (to preserve other settings like color ramp, etc.)
    loadSettings(currentTab);
    
    // Override the field selection again after loadSettings to ensure land price per sqft is always used
    if (landPricePerSqftField && currentField !== landPricePerSqftField) {
      currentField = landPricePerSqftField;
      fieldSelect.value = landPricePerSqftField;
      const stats = metadata.statistics[landPricePerSqftField];
      if (stats) {
        currentStats = { min: stats.min, max: stats.max };
      }
      console.log(`[PMTiles] Overrode saved field to use land price per sqft: ${landPricePerSqftField}`);
    }
    requestAnimationFrame(fitFieldSelectFont);

    // Populate category options from metadata
    if (metadata.categories) {
      populateCategoryOptionsFromMetadata(metadata.categories);
      populateOriginalCategoryOptionsFromMetadata(metadata.categories);
    }

    // Region show/hide on the PMTiles path. Per-group region lists + the int id<->name maps come
    // from the bake metadata.groups (jurisdictions kept as a legacy name->count fallback).
    const pmCityDef = CITIES[SELECTED_CITY] as any;
    if (jurisdiction.configure({
      groups: buildRegionGroupConfigs(pmCityDef, metadata.groups, metadata.jurisdictions),
      onChange: () => { applyFilterAndScaling(); applyExtrusion(); updateLandBlurb(); },
      onOverlaysChange: refreshOverlays,
    })) {
      jurisdiction.buildPanel(map);
    }
    updateLandBlurb();   // initial fill (now that the region widget is configured)

    // PROTOTYPE (localhost-only): 3D-print export of the current hex resolution. Wired here
    // because it needs the loaded PMTiles metadata (bounds) + a hex layer. print3d hides the
    // panel unless on localhost and getContext() returns a PMTiles/hex context.
    init3DPrint({
      map,
      getField: () => currentField,
      computeMetric: computeDisplayedMetricFromProps,
      makeSelectedTest: jurisdiction.makeVisibleTest,
      getRegionField: () => jurisdiction.isActive() ? jurisdiction.getActiveField() : null,
      getContext: get3DPrintContext,
    });

    // Update underutilized totals from metadata
    const normalizedTotals = normalizeUnderutilizedTotals(metadata.underutilizedTotals);
    if (normalizedTotals) {
      currentUnderMetadataTotals = normalizedTotals;
      updateUnderTotalsFromMetadata(normalizedTotals);
    } else {
      currentUnderMetadataTotals = null;
    }

    // Auto-scale multiplier (will be applied after source loads)
    if (currentField && currentStats) {
      // Don't call scheduleUpdate here - wait for source to load first
      // It will be called in the onSourceData handler
    }

    hideLoading();
  } catch (err) {
    hideLoading();
    throw err;
  }
}

function addPmtilesExtrusionLayer(m: maplibregl.Map, withClick = false) {
  if (m.getLayer(LAYER_ID)) return;
  // When the aggregate layer is an H3 hex pyramid (multi-resolution, gated by
  // zoom inside the tiles), hand off to real parcels at `parcelMinZoom` so the
  // coalesced low-zoom parcels never show — hexes own everything below it.
  const cfg = getCityConfig();
  const parcelMinZoom = Number(cfg?.parcelMinZoom);
  const hasHandoff = Number.isFinite(parcelMinZoom);
  try {
    m.addLayer({
      id: LAYER_ID_LOW,
      type: 'fill-extrusion',
      source: SOURCE_ID,
      'source-layer': 'parcels_low',
      ...(hasHandoff ? { maxzoom: parcelMinZoom } : {}),
      paint: {
        'fill-extrusion-color': '#888',
        'fill-extrusion-height': 0,
        'fill-extrusion-opacity': 1,
        'fill-extrusion-vertical-gradient': false
      }
    });
  } catch (err) {
    console.warn('[PMTiles] Low-zoom layer unavailable, skipping.', err);
  }
  m.addLayer({
    id: LAYER_ID,
    type: 'fill-extrusion',
    source: SOURCE_ID,
    'source-layer': 'parcels',  // Must match tippecanoe -l option
    ...(hasHandoff ? { minzoom: parcelMinZoom } : {}),
    paint: {
      'fill-extrusion-color': '#888',
      'fill-extrusion-height': 0,
      'fill-extrusion-opacity': 1,
      'fill-extrusion-vertical-gradient': false
    }
  });
  if (withClick) {
    m.on('click', LAYER_ID, (e) => {
      const f = e.features?.[0];
      if (!f) return;
      const props = (f.properties || {}) as Record<string, any>;
      showPopup(m, props, e.lngLat);
    });
    m.on('mouseenter', LAYER_ID, () => { m.getCanvas().style.cursor = 'pointer'; });
    m.on('mouseleave', LAYER_ID, () => { m.getCanvas().style.cursor = ''; });
    if (m === map) ensureErrorLayerFor(m);
  }
}

function addPmtilesUnderLayer(m: maplibregl.Map, withClick = false) {
  if (m.getLayer(UNDER_FILL_LAYER)) return;
  // The underutilized tab only ever shows the Vacant/Parking/Underdeveloped subset, so
  // the bake gives those their own `parcels_under` layer tiled across ALL zooms — that
  // subset is small enough to tile low-zoom cleanly (the full ~300k parcel layer is
  // not, which is why it stays z13+). Use it when the bake advertises it; cities baked
  // before this fall back to the z13+ `parcels` layer (empty until you zoom in).
  const underSourceLayer = pmtilesMetadata?.underutilizedSourceLayer || 'parcels';
  m.addLayer({
    id: UNDER_FILL_LAYER,
    type: 'fill',
    source: SOURCE_ID,
    'source-layer': underSourceLayer,
    paint: {
      'fill-color': '#94a3b8',
      'fill-opacity': 0.92
    }
  });
  m.addLayer({
    id: UNDER_OUTLINE_LAYER,
    type: 'line',
    source: SOURCE_ID,
    'source-layer': underSourceLayer,
    paint: {
      'line-color': 'rgba(15, 23, 42, 0.55)',
      'line-width': ['interpolate', ['linear'], ['zoom'], 9, 0.4, 13, 1.0, 16, 1.6] as any,
      'line-opacity': 0.7
    }
  });
  if (withClick) {
    m.on('click', UNDER_FILL_LAYER, (e) => {
      const f = e.features?.[0];
      if (!f) return;
      const props = (f.properties || {}) as Record<string, any>;
      showPopup(m, props, e.lngLat);
    });
    m.on('mouseenter', UNDER_FILL_LAYER, () => { m.getCanvas().style.cursor = 'pointer'; });
    m.on('mouseleave', UNDER_FILL_LAYER, () => { m.getCanvas().style.cursor = ''; });
  }
}

function populateCategoryOptionsFromMetadata(categories: { refined: string[]; original: string[] }) {
  const list = categories.refined.sort();
  populateCategoryCheckboxes(categoryContainer, list);
  populateUnderCategoryOptions(list);
}

function populateOriginalCategoryOptionsFromMetadata(categories: { refined: string[]; original: string[] }) {
  const list = categories.original.sort();
  const fill = (sel: HTMLSelectElement | null) => {
    if (!sel) return;
    sel.replaceChildren();
    sel.append(new Option('All categories', ''));
    for (const v of list) sel.append(new Option(v, v));
    sel.value = '';
  };
  fill(origCategorySelect);
  fill(underOrigCategorySelect);
  fill(ratioOrigCategorySelect);
}

function updateUnderTotalsFromMetadata(
  totals: NormalizedUnderTotals,
  selectedCategories?: string[]
) {
  // An explicit empty selection (all boxes unchecked) shows no rows; a truly-absent
  // selection (undefined) still shows all rows.
  const selectedSet = selectedCategories ? new Set(selectedCategories) : null;
  const rows = [
    { label: 'Vacant', total: totals.Vacant },
    { label: 'Underdeveloped', total: totals.Underdeveloped, buckets: totals.underdevelopedBuckets },
    { label: 'Parking Lot', total: totals['Parking Lot'] }
  ].filter((row) => hasMeaningfulUnderSummaryRow(row) && (!selectedSet || selectedSet.has(row.label)));
  renderUnderSummary(rows, totals.totalNonExempt);
}

async function loadDefaultDataset() {
  // Check if city uses PMTiles
  const usesPmtiles = cityUsesPmtiles();
  const config = getCityConfig();
  console.log('[Dataset] City config:', config);
  console.log('[Dataset] Uses PMTiles:', usesPmtiles);
  
  if (usesPmtiles) {
    try {
      await loadPmtilesDataset();
      return;
    } catch (err) {
      console.error('[PMTiles] Failed to load PMTiles dataset:', err);
      if (!cancelRequested) {
        alert(getDatasetLoadErrorMessage(err, 'pmtiles'));
      }
      return;
    }
  }

  // Fall back to parquet loading (only if PMTiles is not configured)
  console.log('[Dataset] Falling back to Parquet loading');
  // Dev: prefer a local copy in viz/public/ for the current city if present, else the remote URL.
  const url = await resolveLocalFirst(LOCAL_DATASET_PATH, DEFAULT_DATASET_URL);
  try {
    lastAsyncBuffer = await urlToAsyncBuffer(url);
    try {
      console.log('[GeoParquet] Fetched dataset:', {
        url,
        byteLength: lastAsyncBuffer?.byteLength ?? null
      });
    } catch {}
    await loadSelectedColumns();
    return;
  } catch (err) {
    console.warn('Dataset load failed for', url, err);
    if (!cancelRequested) alert(getDatasetLoadErrorMessage(err, 'parquet'));
    return;
  }
}

function getDatasetLoadErrorMessage(err: unknown, kind: 'pmtiles' | 'parquet'): string {
  const message = err instanceof Error ? err.message : '';

  if (/404\b/.test(message)) {
    return kind === 'pmtiles'
      ? 'Failed to load PMTiles dataset. The dataset is not available at the live dev path yet.'
      : 'Failed to load dataset. The dataset is not available at the live dev path yet.';
  }

  if (/\b(502|503|504)\b|timeout/i.test(message)) {
    return kind === 'pmtiles'
      ? 'Failed to load PMTiles dataset. The data proxy or blob promotion is still in progress. Please retry shortly.'
      : 'Failed to load dataset. The data proxy or blob promotion is still in progress. Please retry shortly.';
  }

  return kind === 'pmtiles'
    ? 'Failed to load PMTiles dataset. The dataset or metadata is temporarily unavailable.'
    : 'Failed to load dataset. The dataset is temporarily unavailable.';
}

/* ---------------- Main ---------------- */

document.querySelectorAll<HTMLInputElement>('input[name="normMode"]').forEach(r => {
  r.addEventListener('change', () => {
    normalizationMode = (document.querySelector('input[name="normMode"]:checked') as HTMLInputElement)?.value as any;
    if (!currentGeoJSON || !currentField) return;
    scheduleUpdate('recomputeAndAutoScale', /*refreshLegend*/ true);
    saveSettings(currentTab);
  });
});

async function init() {
  const initialView = getUrlView();
  const backToCitiesLink = document.querySelector<HTMLAnchorElement>('.workspace-back-link');

  backToCitiesLink?.addEventListener('click', () => {
    cancelParkingWorkspaceIfNeeded();
  });
  window.addEventListener('pagehide', () => {
    cancelParkingWorkspaceIfNeeded();
  });

  tabLandBtn?.addEventListener('click', () => setTab('main'));
  tabUnderBtn?.addEventListener('click', () => setTab('under'));
  tabParkingBtn?.addEventListener('click', () => setTab('parking'));

  initShareButton();
  initFunnelCards();
  initPdfReportButton();

  await loadDataDictionary();
  // Debug: Log the loaded config
  const config = getCityConfig();
  smoothLandField = config?.smoothLandField ?? null;
  assessedLandField = config?.assessedLandField ?? null;
  updateMapDescriptionText(config);
  updateSmoothToggleUI();
  console.log('[Init] Loaded city config:', config);
  console.log('[Init] City uses PMTiles:', cityUsesPmtiles());

  if (!PARKING_ENABLED && tabParkingBtn) {
    tabParkingBtn.style.display = 'none';
    if (initialView === 'parking') setUrlView('land');
  }

  if (!UNDERUTILIZED_ENABLED && tabUnderBtn) {
    tabUnderBtn.style.display = 'none';
    if (initialView === 'underutilized') setUrlView('land');
  }

  // Shrink the tabs to one row when all three are shown (see .workspace-tabs.is-three CSS).
  const visibleTabCount = [tabLandBtn, tabUnderBtn, tabParkingBtn]
    .filter(b => b && b.style.display !== 'none').length;
  document.querySelector('.view-header .workspace-tabs')?.classList.toggle('is-three', visibleTabCount >= 3);

  if (initialView === 'underutilized' && UNDERUTILIZED_ENABLED) setTab('under');
  else if (initialView === 'parking' && PARKING_ENABLED) setTab('parking');
  else setTab('main');
  
  unitsSelect.value = 'centimeters';
  // Defer all map-mutating actions until the style is fully loaded.
  map.once('load', async () => {
    setQuality('fast');   // render at native resolution (see HQ_PR / quality notes above)
    // Apply any saved settings before loading data.
    loadSettings('main');
    await loadDefaultDataset();
    // Rail transit overlay (e.g. WMATA Metro for the DMV) on top of the parcel layers.
    if (TRANSIT_OVERLAY) {
      void setupTransitOverlay(map, TRANSIT_OVERLAY, import.meta.env.BASE_URL);
    }
  });
}
init();
