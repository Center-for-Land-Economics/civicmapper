/**
 * DEV ONLY — parking-label tool (viz/label.html).
 *
 * Labellers outline surface parking on NAIP imagery inside sample squares made by
 * data/scripts/label_campaign.py; submissions go to the local API (/api/labels, loopback only,
 * api/src/labels.js) which appends them to data/labels/<campaign>/labels.jsonl.
 * Gated like the 3D print panel: the page does nothing off localhost, and label.html is not a
 * Vite build input, so it never ships.
 */
import 'maplibre-gl/dist/maplibre-gl.css';
import * as maplibregl from 'maplibre-gl';
import '../maplibre-worker';
import {
  TerraDraw, TerraDrawAngledRectangleMode, TerraDrawPolygonMode, TerraDrawSelectMode,
  type GeoJSONStoreFeatures, type HexColor
} from 'terra-draw';
import { TerraDrawMapLibreGLAdapter } from 'terra-draw-maplibre-gl-adapter';
import { is3DPrintEnabled as isLocalDev } from '../print3d';

type Cls = 'paved' | 'unpaved' | 'structure' | 'unsure';
type Outcome = 'labeled' | 'no_parking' | 'cant_tell' | 'skip';
type Mode = 'blind' | 'correction';
interface SquareProps {
  id: string; stratum: string; double: boolean; naip_year: number | null; naip_items: string[];
  imagery_date: string | null; center_lng: number; center_lat: number;
  labels: number; labelers: string[]; complete: boolean;
}
type Square = GeoJSON.Feature<GeoJSON.Polygon, SquareProps>;

const API = '/api/labels';
const TILER = 'https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{z}/{x}/{y}@1x';
const MAX_ITEMS = 4;
const COLORS: Record<Cls, HexColor> = { paved: '#ff4d6d', unpaved: '#f4a259', structure: '#b388ff', unsure: '#ffe066' };
const CLASS_KEYS: Record<string, Cls> = { '1': 'paved', '2': 'unpaved', '3': 'structure', '4': 'unsure' };

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

function store(key: string, value?: string): string | null {
  try {
    if (value !== undefined) localStorage.setItem(key, value);
    return localStorage.getItem(key);
  } catch { return null; }
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API}${path}`, init);
  const js = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((js as any).errors?.join(', ') || (js as any).error || `HTTP ${r.status}`);
  return js as T;
}

function toast(msg: string, ms = 2200) {
  const t = $('toast');
  t.textContent = msg;
  t.style.display = 'block';
  window.clearTimeout((toast as any).timer);
  (toast as any).timer = window.setTimeout(() => (t.style.display = 'none'), ms);
}

function naipTiles(item: string, bands: string): string[] {
  const q = `collection=naip&item=${encodeURIComponent(item)}&assets=image&asset_bidx=${encodeURIComponent(`image|${bands}`)}&format=png`;
  return [`${TILER}?${q}`];
}

function bboxOf(f: GeoJSON.Feature<GeoJSON.Polygon>): [number, number, number, number] {
  const xs = f.geometry.coordinates[0].map((c) => c[0]);
  const ys = f.geometry.coordinates[0].map((c) => c[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

const round = (v: number) => Math.round(v * 1e7) / 1e7;

function closedRing(ring: number[][]): number[][] {
  const r = ring.map((c) => [round(c[0]), round(c[1])]);
  const dedup = r.filter((c, i) => i === 0 || c[0] !== r[i - 1][0] || c[1] !== r[i - 1][1]);
  const [a, b] = [dedup[0], dedup[dedup.length - 1]];
  if (a[0] !== b[0] || a[1] !== b[1]) dedup.push([a[0], a[1]]);
  return dedup;
}

async function main() {
  if (!isLocalDev()) { $('blocked').hidden = false; return; }
  $('app').hidden = false;

  const state = {
    campaign: '' as string,
    labeler: store('label.labeler') || '',
    mode: (store('label.mode') as Mode) || 'blind',
    cls: 'paved' as Cls,
    square: null as Square | null,
    startedAt: 0,
    cir: false,
    parcels: true,
    shapesHidden: false,
    busy: false
  };

  // ---- map --------------------------------------------------------------------------------
  const map = new maplibregl.Map({
    container: 'map',
    style: { version: 8, sources: {}, layers: [{ id: 'bg', type: 'background', paint: { 'background-color': '#0b0c0e' } }] },
    center: [-96.7, 33.0], zoom: 17, maxZoom: 21, minZoom: 14,
    attributionControl: { compact: true, customAttribution: 'USDA NAIP · Microsoft Planetary Computer' },
    dragRotate: false, pitchWithRotate: false
  });
  map.touchZoomRotate.disableRotation();
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-left');
  map.addControl(new maplibregl.ScaleControl({ unit: 'imperial' }), 'bottom-left');
  await new Promise<void>((res) => map.once('load', () => res()));

  for (const kind of ['rgb', 'cir'] as const) {
    for (let i = 0; i < MAX_ITEMS; i++) {
      map.addSource(`naip-${kind}-${i}`, { type: 'raster', tiles: naipTiles('none', '1,2,3'), tileSize: 256, maxzoom: 19 });
      map.addLayer({ id: `naip-${kind}-${i}`, type: 'raster', source: `naip-${kind}-${i}`, layout: { visibility: 'none' } });
    }
  }
  const empty: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] };
  map.addSource('mask', { type: 'geojson', data: empty });
  map.addLayer({ id: 'mask', type: 'fill', source: 'mask', paint: { 'fill-color': '#000', 'fill-opacity': 0.5 } });
  map.addSource('parcels', { type: 'geojson', data: empty });
  map.addLayer({ id: 'parcels', type: 'line', source: 'parcels',
    paint: { 'line-color': '#7fdbff', 'line-width': 1, 'line-opacity': 0.75, 'line-dasharray': [3, 2] } });
  map.addSource('square', { type: 'geojson', data: empty });
  map.addLayer({ id: 'square', type: 'line', source: 'square', paint: { 'line-color': '#fff', 'line-width': 2 } });
  map.on('error', (e) => {
    // A square spanning two NAIP items asks each item for tiles outside it (HTTP 424); harmless.
    if (!String((e as any).error?.message || '').includes('424')) console.warn(e.error);
  });

  // ---- drawing ----------------------------------------------------------------------------
  const clsOf = (f: GeoJSONStoreFeatures): Cls => ((f.properties?.class as Cls) || 'paved');
  const polyStyles = {
    fillColor: (f: GeoJSONStoreFeatures) => COLORS[clsOf(f)],
    outlineColor: (f: GeoJSONStoreFeatures) => COLORS[clsOf(f)],
    fillOpacity: () => (state.shapesHidden ? 0 : 0.28),
    outlineOpacity: () => (state.shapesHidden ? 0 : 1),
    outlineWidth: 2
  };
  const editable = {
    feature: {
      draggable: true,
      coordinates: { midpoints: true, draggable: true, deletable: true }
    }
  };
  const draw = new TerraDraw({
    adapter: new TerraDrawMapLibreGLAdapter({ map }),
    modes: [
      new TerraDrawPolygonMode({ styles: polyStyles }),
      new TerraDrawAngledRectangleMode({ styles: polyStyles }),
      new TerraDrawSelectMode({
        flags: { polygon: editable, 'angled-rectangle': editable },
        styles: {
          selectedPolygonColor: (f: GeoJSONStoreFeatures) => COLORS[clsOf(f)],
          selectedPolygonOutlineColor: '#ffffff',
          selectedPolygonOutlineWidth: 3,
          selectedPolygonFillOpacity: 0.35
        }
      })
    ]
  });
  draw.start();
  draw.on('finish', (id, ctx) => {
    if ((ctx as any)?.action === 'draw') draw.updateFeatureProperties(id, { class: state.cls });
  });

  const polygons = () => draw.getSnapshot().filter((f) => f.geometry.type === 'Polygon');
  const selected = () => polygons().filter((f) => f.properties?.selected);

  function setTool(mode: 'polygon' | 'angled-rectangle' | 'select') {
    draw.setMode(mode);
    $('tool-polygon').classList.toggle('on', mode === 'polygon');
    $('tool-rect').classList.toggle('on', mode === 'angled-rectangle');
    $('tool-select').classList.toggle('on', mode === 'select');
  }

  function setClass(c: Cls) {
    state.cls = c;
    document.querySelectorAll<HTMLButtonElement>('.cls').forEach((b) => b.classList.toggle('on', b.dataset.c === c));
    for (const f of selected()) draw.updateFeatureProperties(f.id as string, { class: c });
  }

  function loadShapes(features: GeoJSON.Feature[], fromPrefill: boolean) {
    const feats: GeoJSONStoreFeatures[] = [];
    for (const f of features) {
      const g = f.geometry;
      const polys = g.type === 'Polygon' ? [g.coordinates] : g.type === 'MultiPolygon' ? g.coordinates : [];
      for (const p of polys) {
        // Terra Draw edits single-ring polygons; holes are dropped (rare in lot outlines).
        const ring = closedRing(p[0] as number[][]);
        if (ring.length < 4) continue;
        feats.push({
          id: crypto.randomUUID(), type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [ring] },
          properties: { mode: 'polygon', class: (f.properties as any)?.class || 'paved',
            ...(fromPrefill || (f.properties as any)?.from_prefill ? { from_prefill: true } : {}) }
        } as GeoJSONStoreFeatures);
      }
    }
    const res = draw.addFeatures(feats);
    const bad = res.filter((r) => !r.valid).length;
    if (bad) toast(`${bad} shape(s) could not be loaded for editing`);
  }

  // ---- squares ----------------------------------------------------------------------------
  function showSquare(sq: Square) {
    state.square = sq;
    const p = sq.properties;
    const items = (p.naip_items || []).slice(0, MAX_ITEMS);
    for (const kind of ['rgb', 'cir'] as const) {
      for (let i = 0; i < MAX_ITEMS; i++) {
        const src = map.getSource(`naip-${kind}-${i}`) as maplibregl.RasterTileSource;
        if (items[i]) src.setTiles(naipTiles(items[i], kind === 'rgb' ? '1,2,3' : '4,1,2'));
        const on = !!items[i] && (kind === 'cir') === state.cir;
        map.setLayoutProperty(`naip-${kind}-${i}`, 'visibility', on ? 'visible' : 'none');
      }
    }
    const ring = sq.geometry.coordinates[0];
    (map.getSource('square') as maplibregl.GeoJSONSource).setData(sq);
    (map.getSource('mask') as maplibregl.GeoJSONSource).setData({
      type: 'Feature', properties: {},
      geometry: { type: 'Polygon', coordinates: [[[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]], [...ring].reverse()] }
    });
    const [x0, y0, x1, y1] = bboxOf(sq);
    const pad = Math.max(x1 - x0, y1 - y0);
    map.setMaxBounds(null);
    map.fitBounds([[x0, y0], [x1, y1]], { padding: 30, animate: false });
    map.setMaxBounds([[x0 - pad, y0 - pad], [x1 + pad, y1 + pad]]);
    const done = p.labelers?.length ? ` · labelled by ${p.labelers.join(', ')}` : '';
    $('sqmeta').innerHTML =
      `<b>${p.id}</b> · ${p.stratum}${p.double ? ' · <span class="warn">double-labelled</span>' : ''}<br>` +
      `Imagery: <b>NAIP ${p.naip_year ?? '—'}</b>${p.imagery_date ? ` (flown ${p.imagery_date})` : ''}` +
      `${p.naip_year && p.naip_year < 2024 ? ' <span class="warn">older imagery</span>' : ''}${done}`;
    draw.clear();
    (document.querySelectorAll<HTMLInputElement>('#flags input')).forEach((i) => (i.checked = false));
    ($('note') as HTMLTextAreaElement).value = '';
    state.startedAt = Date.now();
    const url = new URL(location.href);
    url.searchParams.set('campaign', state.campaign);
    url.searchParams.set('square', p.id);
    history.replaceState(null, '', url);
  }

  async function loadContext(sq: Square) {
    const id = encodeURIComponent(sq.properties.id);
    const c = encodeURIComponent(state.campaign);
    const [parcels, mine] = await Promise.all([
      api<GeoJSON.FeatureCollection>(`/${c}/layer/parcels?square=${id}`),
      api<{ labels: any[] }>(`/${c}/labels?square=${id}&labeler=${encodeURIComponent(state.labeler)}`)
    ]);
    (map.getSource('parcels') as maplibregl.GeoJSONSource).setData(parcels);
    const last = mine.labels.filter((l) => l.outcome !== 'skip').at(-1);
    if (last) {
      loadShapes(last.polygons, false);
      document.querySelectorAll<HTMLInputElement>('#flags input').forEach((i) => (i.checked = last.flags.includes(i.value)));
      ($('note') as HTMLTextAreaElement).value = last.note || '';
      toast(`Loaded your earlier label (${last.outcome}); resubmitting replaces it`, 3500);
    } else if (state.mode === 'correction') {
      const pre = await api<GeoJSON.FeatureCollection>(`/${c}/layer/prefill?square=${id}`);
      loadShapes(pre.features, true);
      toast(`${pre.features.length} existing lot(s) pre-loaded — fix, delete or add`);
    }
  }

  async function refreshProgress() {
    const { campaigns } = await api<{ campaigns: any[] }>('/campaigns');
    const c = campaigns.find((x) => x.campaign === state.campaign);
    if (c) $('progress').textContent = `${c.progress.complete} / ${c.progress.squares} squares complete · ` +
      `${c.progress.labels} / ${c.progress.needed} labels`;
    return campaigns;
  }

  async function next(after?: string) {
    if (!state.labeler) { toast('Enter your name first'); $('labeler').focus(); return; }
    const q = `labeler=${encodeURIComponent(state.labeler)}${after ? `&after=${encodeURIComponent(after)}` : ''}`;
    const r = await api<{ square: Square | null; remaining: number }>(`/${encodeURIComponent(state.campaign)}/next?${q}`);
    if (!r.square) {
      draw.clear();
      $('sqmeta').textContent = 'Nothing left for you in this campaign. Thank you!';
      return;
    }
    showSquare(r.square);
    await loadContext(r.square);
  }

  async function open(id: string) {
    const r = await api<{ square: Square }>(`/${encodeURIComponent(state.campaign)}/square/${encodeURIComponent(id)}`);
    showSquare(r.square);
    await loadContext(r.square);
  }

  async function submit(outcome: Outcome) {
    if (state.busy || !state.square) return;
    if (!state.labeler) { toast('Enter your name first'); $('labeler').focus(); return; }
    const shapes = outcome === 'labeled' ? polygons() : [];
    if (outcome === 'labeled' && !shapes.length) {
      toast('No shapes drawn — use “No parking” (N) if the square has none');
      return;
    }
    if (outcome !== 'labeled' && polygons().length &&
        !confirm(`Discard the ${polygons().length} shape(s) and submit “${outcome.replace('_', ' ')}”?`)) return;
    const p = state.square.properties;
    const body = {
      labeler: state.labeler, square_id: p.id, outcome, mode: state.mode,
      polygons: shapes.map((f) => ({
        type: 'Feature',
        geometry: { type: 'Polygon', coordinates: (f.geometry.coordinates as number[][][]).map(closedRing) },
        properties: { class: clsOf(f), ...(f.properties?.from_prefill ? { from_prefill: true } : {}) }
      })),
      flags: [...document.querySelectorAll<HTMLInputElement>('#flags input:checked')].map((i) => i.value),
      note: ($('note') as HTMLTextAreaElement).value.trim(),
      seconds: Math.round((Date.now() - state.startedAt) / 1000),
      naip_year: p.naip_year ?? null,
      imagery: p.imagery_date ?? null
    };
    state.busy = true;
    try {
      await api(`/${encodeURIComponent(state.campaign)}/labels`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
      });
      toast(outcome === 'skip' ? 'Skipped' : `Saved ${p.id}`);
      await refreshProgress();
      await next(p.id);
    } catch (e) {
      toast(`Not saved: ${(e as Error).message}`, 5000);
    } finally {
      state.busy = false;
    }
  }

  // ---- controls ---------------------------------------------------------------------------
  const setMode = (m: Mode) => {
    state.mode = m;
    store('label.mode', m);
    $('mode-blind').classList.toggle('on', m === 'blind');
    $('mode-correction').classList.toggle('on', m === 'correction');
  };
  const toggleCir = () => {
    state.cir = !state.cir;
    $('view-cir').classList.toggle('on', state.cir);
    if (state.square) showSquareImagery();
  };
  const showSquareImagery = () => {
    const n = (state.square?.properties.naip_items || []).length;
    for (const kind of ['rgb', 'cir'] as const) {
      for (let i = 0; i < MAX_ITEMS; i++) {
        const on = i < n && (kind === 'cir') === state.cir;
        map.setLayoutProperty(`naip-${kind}-${i}`, 'visibility', on ? 'visible' : 'none');
      }
    }
  };
  const toggleParcels = () => {
    state.parcels = !state.parcels;
    $('view-parcels').classList.toggle('on', state.parcels);
    map.setLayoutProperty('parcels', 'visibility', state.parcels ? 'visible' : 'none');
  };
  const toggleShapes = () => {
    state.shapesHidden = !state.shapesHidden;
    $('view-shapes').classList.toggle('on', state.shapesHidden);
    // Re-add the features so Terra Draw restyles them.
    const snap = polygons().map((f) => {
      const { selected: _s, ...props } = f.properties;
      return { ...f, properties: props } as GeoJSONStoreFeatures;
    });
    draw.clear();
    draw.addFeatures(snap);
  };
  const deleteSelected = () => {
    const ids = selected().map((f) => f.id as string);
    if (ids.length) draw.removeFeatures(ids);
  };
  const fit = () => {
    if (!state.square) return;
    const [x0, y0, x1, y1] = bboxOf(state.square);
    map.fitBounds([[x0, y0], [x1, y1]], { padding: 30 });
  };

  $('mode-blind').onclick = () => setMode('blind');
  $('mode-correction').onclick = () => setMode('correction');
  $('tool-polygon').onclick = () => setTool('polygon');
  $('tool-rect').onclick = () => setTool('angled-rectangle');
  $('tool-select').onclick = () => setTool('select');
  $('tool-delete').onclick = deleteSelected;
  document.querySelectorAll<HTMLButtonElement>('.cls').forEach((b) => (b.onclick = () => setClass(b.dataset.c as Cls)));
  $('view-cir').onclick = toggleCir;
  $('view-parcels').onclick = toggleParcels;
  $('view-shapes').onclick = toggleShapes;
  $('view-fit').onclick = fit;
  $('submit').onclick = () => submit('labeled');
  $('no-parking').onclick = () => submit('no_parking');
  $('cant-tell').onclick = () => submit('cant_tell');
  $('skip').onclick = () => submit('skip');
  $('guide-btn').onclick = () => ($('guide') as HTMLDialogElement).showModal();

  const nameInput = $('labeler') as HTMLInputElement;
  nameInput.value = state.labeler;
  nameInput.onchange = () => {
    state.labeler = nameInput.value.trim();
    store('label.labeler', state.labeler);
    if (state.labeler && !state.square) void next();
  };

  window.addEventListener('keydown', (e) => {
    const t = e.target as HTMLElement;
    if (t.closest('input, textarea, select') || ($('guide') as HTMLDialogElement).open) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (CLASS_KEYS[k]) setClass(CLASS_KEYS[k]);
    else if (k === 'd' || k === 'D') setTool('polygon');
    else if (k === 'r' || k === 'R') setTool('angled-rectangle');
    else if (k === 'e' || k === 'E') setTool('select');
    else if (k === 'Delete' || k === 'Backspace') { deleteSelected(); e.preventDefault(); }
    else if (k === 'c' || k === 'C') toggleCir();
    else if (k === 'p' || k === 'P') toggleParcels();
    else if (k === 'h' || k === 'H') toggleShapes();
    else if (k === 'f' || k === 'F') fit();
    else if (k === 'Enter') submit('labeled');
    else if (k === 'n' || k === 'N') submit('no_parking');
    else if (k === 't' || k === 'T') submit('cant_tell');
    else if (k === 'k' || k === 'K') submit('skip');
    else if (k === '?') ($('guide') as HTMLDialogElement).showModal();
  });

  // ---- start ------------------------------------------------------------------------------
  setMode(state.mode);
  setClass('paved');
  setTool('polygon');
  $('view-parcels').classList.add('on');

  const campaigns = await refreshProgress().catch((e) => {
    $('sqmeta').innerHTML = `<span class="warn">Label API not reachable (${(e as Error).message}). ` +
      'Start the API (node api/src/server.js) and create a campaign with data/scripts/label_campaign.py.</span>';
    return [] as any[];
  });
  if (!campaigns.length) return;
  const sel = $('campaign') as HTMLSelectElement;
  sel.innerHTML = campaigns.map((c) => `<option value="${c.campaign}">${c.campaign} — ${c.city}, ${c.n} squares</option>`).join('');
  const params = new URLSearchParams(location.search);
  state.campaign = params.get('campaign') && campaigns.some((c) => c.campaign === params.get('campaign'))
    ? params.get('campaign')! : (store('label.campaign') || campaigns[0].campaign);
  if (!campaigns.some((c) => c.campaign === state.campaign)) state.campaign = campaigns[0].campaign;
  sel.value = state.campaign;
  sel.onchange = async () => {
    state.campaign = sel.value;
    store('label.campaign', state.campaign);
    state.square = null;
    await refreshProgress();
    await next();
  };
  await refreshProgress();
  if (!state.labeler) {
    $('sqmeta').textContent = 'Enter your name to start. Read the guidelines (?) first.';
    ($('guide') as HTMLDialogElement).showModal();
    return;
  }
  if (params.get('square')) await open(params.get('square')!).catch(() => next());
  else await next();
}

void main();
