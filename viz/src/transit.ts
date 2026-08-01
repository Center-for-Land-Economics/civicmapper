/**
 * Transit overlay — draws a rail transit network (lines colored by route + station markers)
 * on top of the parcel map. Generic and city-gated: a city opts in via CityDef.transitOverlay
 * = { stations, lines, label }, pointing at two static GeoJSONs in viz/public.
 *
 * First consumer: the DMV city, using WMATA Metrorail (viz/public/dmv-metro-{lines,stations}.geojson).
 * Lines GeoJSON: one LineString feature per color, property NAME ∈ {red,orange,blue,silver,green,yellow}.
 * Stations GeoJSON: Point features with NAME (station) + LINE (", "-joined color list).
 */
import type maplibregl from 'maplibre-gl';

export interface TransitOverlayConfig {
  /** viz/public filename of the line-network GeoJSON (LineStrings, one per route color). */
  lines: string;
  /** viz/public filename of the station-points GeoJSON. */
  stations: string;
  /** UI label for the toggle, e.g. "Metro". Defaults to "Transit". */
  label?: string;
}

// WMATA Metrorail official-ish line colors (lowercase route name -> hex).
const LINE_COLORS: Record<string, string> = {
  red: '#E51937', orange: '#F7941E', blue: '#0077C0',
  silver: '#A1A3A6', green: '#00A94F', yellow: '#FDBB30',
};

const SRC_LINES = 'transit-lines-src';
const SRC_STATIONS = 'transit-stations-src';
const LAYER_LINES = 'transit-lines';
const LAYER_STATIONS = 'transit-stations';
const LAYER_LABELS = 'transit-station-labels';
const ALL_LAYERS = [LAYER_LINES, LAYER_STATIONS, LAYER_LABELS];

/** MapLibre paint expression mapping the line NAME field to its color. */
function lineColorExpr(): any {
  const match: any[] = ['match', ['downcase', ['coalesce', ['get', 'NAME'], '']]];
  for (const [name, hex] of Object.entries(LINE_COLORS)) match.push(name, hex);
  match.push('#666'); // fallback
  return match;
}

async function fetchJson(url: string): Promise<any | null> {
  try { return await (await fetch(url)).json(); } catch { return null; }
}

/**
 * Load the transit GeoJSONs and add the line + station + label layers to `map` (on top of the
 * current layer stack). Wires the #showTransit checkbox to toggle their visibility, and reveals
 * the #transitControlGroup control. Safe to call once the map style is loaded. Idempotent.
 */
export async function setupTransitOverlay(
  map: maplibregl.Map,
  cfg: TransitOverlayConfig,
  baseUrl: string,
): Promise<void> {
  const label = cfg.label || 'Transit';
  const [lines, stations] = await Promise.all([
    fetchJson(`${baseUrl}${cfg.lines}`),
    fetchJson(`${baseUrl}${cfg.stations}`),
  ]);
  if (!lines && !stations) return;

  const addAll = () => {
    for (const id of ALL_LAYERS) if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(SRC_LINES)) map.removeSource(SRC_LINES);
    if (map.getSource(SRC_STATIONS)) map.removeSource(SRC_STATIONS);

    if (lines?.features?.length) {
      map.addSource(SRC_LINES, { type: 'geojson', data: lines });
      map.addLayer({
        id: LAYER_LINES, type: 'line', source: SRC_LINES,
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': lineColorExpr(),
          'line-width': ['interpolate', ['linear'], ['zoom'], 8, 1.6, 12, 3, 15, 5],
          'line-opacity': 0.9,
        },
      });
    }
    if (stations?.features?.length) {
      map.addSource(SRC_STATIONS, { type: 'geojson', data: stations });
      map.addLayer({
        id: LAYER_STATIONS, type: 'circle', source: SRC_STATIONS,
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 9, 2.5, 12, 4.5, 15, 7],
          'circle-color': '#111827',
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': ['interpolate', ['linear'], ['zoom'], 9, 1, 13, 2],
          'circle-opacity': 0.95,
        },
      });
      map.addLayer({
        id: LAYER_LABELS, type: 'symbol', source: SRC_STATIONS,
        minzoom: 12,
        layout: {
          'text-field': ['coalesce', ['get', 'NAME'], ''],
          'text-font': ['Noto Sans Regular'],
          'text-size': ['interpolate', ['linear'], ['zoom'], 12, 10, 15, 13],
          'text-offset': [0, 1.1], 'text-anchor': 'top',
          'text-allow-overlap': false, 'text-optional': true, 'text-padding': 4,
        },
        paint: { 'text-color': '#111827', 'text-halo-color': '#ffffff', 'text-halo-width': 1.5 },
      });
    }
    applyVisibility(map, visible);
  };

  let visible = true;
  const checkbox = document.getElementById('showTransit') as HTMLInputElement | null;
  const group = document.getElementById('transitControlGroup');
  const labelSpan = document.getElementById('transitControlLabel');
  if (group) group.style.display = '';
  if (labelSpan) labelSpan.textContent = `Show ${label} stations & lines`;
  if (checkbox) {
    visible = checkbox.checked;
    checkbox.addEventListener('change', () => {
      visible = checkbox.checked;
      applyVisibility(map, visible);
    });
  }

  addAll();
}

function applyVisibility(map: maplibregl.Map, visible: boolean): void {
  const v = visible ? 'visible' : 'none';
  for (const id of ALL_LAYERS) {
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', v);
  }
}
