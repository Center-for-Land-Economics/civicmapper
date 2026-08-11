/**
 * City-picker page: a contained MapLibre map (OpenStreetMap basemap) with a dot per
 * supported city, plus a searchable, state-grouped list. Hovering a list row (or the
 * dot itself) pops a name pill above the dot; a list-row hover also makes the dot hop,
 * turn red, and rise above its neighbours. Dots and rows jump to app.html?city=<key>.
 * City metadata (names, states, coords) all comes from the registry in cities.ts.
 */
import 'maplibre-gl/dist/maplibre-gl.css';
import maplibregl from 'maplibre-gl';
import { CITIES, CITY_KEYS, CITY_COORDS, STATE_NAMES, type CityKey } from './cities';

// The IBX dataset is an alternate NYC layer that overlaps the nyc dot — keep it out of the picker.
// devOnly cities (e.g. 'harris') are never listed; they're reachable only via ?city= on localhost.
const PICKER_KEYS = CITY_KEYS.filter((k) => k !== 'ibx' && !CITIES[k].devOnly);

interface PickerCity {
  key: CityKey;
  name: string;
  state: string;
  lng: number;
  lat: number;
}

const PICKER_CITIES: PickerCity[] = PICKER_KEYS.map((key) => {
  const [lng, lat] = CITY_COORDS[key];
  return { key, name: CITIES[key].displayName, state: CITIES[key].state, lng, lat };
});

const initials = (name: string): string =>
  name
    .replace(/[^A-Za-z /]/g, '')
    .split(/[ /]/)
    .filter(Boolean)
    .slice(0, name.includes('/') ? 1 : 2)
    .map((w) => w[0])
    .join('')
    .toUpperCase()
    .slice(0, 3);

const normalize = (s: string): string => s.toLowerCase().replace(/[^a-z0-9]/g, '');

const goTo = (key: string): void => {
  window.location.href = 'app.html?city=' + key;
};

// ---- Map (Carto Voyager vector basemap, with roads stripped) ----
const VOYAGER_STYLE_URL = 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json';

const map = new maplibregl.Map({
  container: 'cityMap',
  style: VOYAGER_STYLE_URL,
  center: [-96.5, 38.5],
  zoom: 3.4,
  attributionControl: { compact: true }
});
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');

// Strip roads (and rail) from the Voyager vector basemap: all road geometry/cases/bridges/
// tunnels/rail live on the "transportation" source-layer, and road labels on
// "transportation_name". Aeroway runways are a separate source-layer and are left intact.
map.on('load', () => {
  for (const layer of map.getStyle().layers ?? []) {
    const srcLayer = (layer as { 'source-layer'?: string })['source-layer'];
    if (srcLayer === 'transportation' || srcLayer === 'transportation_name') {
      map.removeLayer(layer.id);
    }
  }
});

// Dot marker: gradient circle with a white stroke. The .shape class lets the highlight
// rule recolor it red; the .mk-fx wrapper is what hops; the .mk-pill rides inside it.
const dotSVG =
  '<svg viewBox="0 0 26 26" xmlns="http://www.w3.org/2000/svg"><defs>' +
  '<linearGradient id="mkg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2563eb"/><stop offset="1" stop-color="#0f766e"/></linearGradient>' +
  '</defs><circle class="shape" cx="13" cy="13" r="8" fill="url(#mkg)" stroke="#fff" stroke-width="2.2"/></svg>';

const markers: Record<string, { marker: maplibregl.Marker; el: HTMLElement }> = {};
const bounds = new maplibregl.LngLatBounds();

function buildMarkers(): void {
  for (const c of PICKER_CITIES) {
    const el = document.createElement('div');
    el.className = 'mk';
    el.innerHTML = `<div class="mk-fx">${dotSVG}<span class="mk-pill"></span></div>`;
    (el.querySelector('.mk-pill') as HTMLElement).textContent = c.name;

    const marker = new maplibregl.Marker({ element: el, anchor: 'center' })
      .setLngLat([c.lng, c.lat])
      .addTo(map);

    el.addEventListener('click', (e) => { e.stopPropagation(); goTo(c.key); });

    markers[c.key] = { marker, el };
    bounds.extend([c.lng, c.lat]);
  }
  map.fitBounds(bounds, { padding: 56, maxZoom: 6 });

  // Painter's algorithm: stack dots by screen position so lower-right sits on top of
  // upper-left. Recompute as the map pans/zooms.
  updateMarkerOrder();
  map.on('move', updateMarkerOrder);
}
map.on('load', buildMarkers);

function updateMarkerOrder(): void {
  Object.values(markers)
    .map((m) => {
      const p = map.project(m.marker.getLngLat());
      return { m, score: p.y * 2000 + p.x };
    })
    .sort((a, b) => a.score - b.score)
    .forEach((e, i) => { e.m.el.style.zIndex = String(i + 1); });
}

// Highlight a dot from the list: adds .hl, which hops it, reddens it, and (via CSS) lifts
// it above its neighbours. Pass null to clear.
function highlight(key: string | null): void {
  for (const [k, { el }] of Object.entries(markers)) el.classList.toggle('hl', k === key);
}

// ---- Searchable, state-grouped list ----
function byStateGroups(): { st: string; cities: PickerCity[] }[] {
  const groups: Record<string, PickerCity[]> = {};
  for (const c of PICKER_CITIES) (groups[c.state] ||= []).push(c);
  return Object.keys(groups)
    .sort((a, b) => STATE_NAMES[a].localeCompare(STATE_NAMES[b]))
    .map((st) => ({ st, cities: groups[st].sort((a, b) => a.name.localeCompare(b.name)) }));
}

function buildList(): void {
  const list = document.getElementById('cityList');
  const noResults = document.getElementById('noResults');
  const search = document.getElementById('citySearch') as HTMLInputElement | null;
  const count = document.getElementById('cityCount');
  if (!list || !noResults || !search) return;

  for (const { st, cities } of byStateGroups()) {
    const wrap = document.createElement('div');
    wrap.className = 'state-group';
    const head = document.createElement('div');
    head.className = 'state-head';
    head.textContent = STATE_NAMES[st];
    wrap.appendChild(head);
    for (const c of cities) {
      const row = document.createElement('button');
      row.className = 'city-row';
      row.dataset.key = c.key;
      row.dataset.search = normalize(c.name + c.key + c.state + STATE_NAMES[c.state]);
      row.innerHTML = `<span class="chip">${initials(c.name)}</span><span class="label">${c.name}</span>`;
      row.addEventListener('click', () => goTo(c.key));
      row.addEventListener('mouseenter', () => highlight(c.key));
      row.addEventListener('mouseleave', () => highlight(null));
      wrap.appendChild(row);
    }
    list.appendChild(wrap);
  }

  const groupEls = [...list.querySelectorAll<HTMLElement>('.state-group')];

  function applyFilter(): void {
    const q = normalize(search!.value);
    let total = 0;
    for (const group of groupEls) {
      let visible = 0;
      for (const row of group.querySelectorAll<HTMLElement>('.city-row')) {
        const match = !q || (row.dataset.search ?? '').includes(q);
        row.hidden = !match;
        if (match) visible++;
      }
      group.hidden = visible === 0;
      total += visible;
    }
    noResults!.hidden = total > 0;
  }

  search.addEventListener('input', applyFilter);
  search.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    const first = list.querySelector<HTMLElement>('.city-row:not([hidden])');
    if (first?.dataset.key) goTo(first.dataset.key);
  });

  if (count) count.textContent = String(PICKER_CITIES.length);
}

buildList();
