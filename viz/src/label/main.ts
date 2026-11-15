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
  TerraDraw, TerraDrawAngledRectangleMode, TerraDrawPolygonMode, TerraDrawRenderMode, TerraDrawSelectMode,
  type GeoJSONStoreFeatures, type HexColor
} from 'terra-draw';
import { TerraDrawMapLibreGLAdapter } from 'terra-draw-maplibre-gl-adapter';
import { is3DPrintEnabled as isLocalDev } from '../print3d';

// 'cutout' = a hole: subtracted from the parking shapes it overlaps when labels are exported
// (Terra Draw cannot draw polygons with holes).
type Cls = 'paved' | 'unpaved' | 'structure' | 'unsure' | 'cutout';
type Outcome = 'labeled' | 'no_parking' | 'cant_tell' | 'skip';
type Mode = 'blind' | 'correction' | 'model';
interface SquareProps {
  id: string; stratum: string; double: boolean; naip_year: number | null; naip_items: string[];
  imagery_date: string | null; center_lng: number; center_lat: number;
  labels: number; labelers: string[]; complete: boolean; holdout?: boolean; targeted?: boolean;
}
type Square = GeoJSON.Feature<GeoJSON.Polygon, SquareProps>;

const API = '/api/labels';
const TILER = 'https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{z}/{x}/{y}@1x';
const MAX_ITEMS = 4;
const COLORS: Record<Cls, HexColor> = { paved: '#ff4d6d', unpaved: '#f4a259', structure: '#b388ff', unsure: '#ffe066',
  cutout: '#ffffff' };
const CELL_M = 0.6; // paint grid cell = one NAIP pixel
const SCALE = 2;    // canvas pixels per grid cell (smoother polygon edges)
const CODES: Record<Cls, number> = { paved: 1, unpaved: 2, structure: 3, unsure: 4, cutout: 5 };
const CUTOUT_CODE = 5;
const hexRgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const CODE_RGB: Record<number, number[]> = Object.fromEntries(
  (Object.keys(CODES) as Cls[]).map((c) => [CODES[c], hexRgb(COLORS[c])]));
const CLASS_KEYS: Record<string, Cls> = { '1': 'paved', '2': 'unpaved', '3': 'structure', '4': 'unsure', '5': 'cutout',
  x: 'cutout', X: 'cutout' };

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

type Pt = [number, number];
type Tool = 'polygon' | 'angled-rectangle' | 'select' | 'parcel' | 'paint';
function pointInRing(pt: Pt, ring: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function ringArea(ring: number[][]): number {
  let a = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) a += (ring[j][0] + ring[i][0]) * (ring[j][1] - ring[i][1]);
  return Math.abs(a / 2);
}
/** Closest point on segment ab to p (all screen pixels). */
function nearestOnSegment(p: Pt, a: Pt, b: Pt): Pt {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len = dx * dx + dy * dy;
  const t = len ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len)) : 0;
  return [a[0] + t * dx, a[1] + t * dy];
}
const dist = (a: Pt, b: Pt) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const SNAP_VERTEX_PX = 12;
const SNAP_EDGE_PX = 8;

async function main() {
  (window as any).__labelBooted = true;
  if (!isLocalDev()) { $('blocked').hidden = false; return; }
  $('app').hidden = false;

  const state = {
    campaign: '' as string,
    labeler: store('label.labeler') || '',
    mode: (store('label.mode') as Mode) || 'blind',
    squareMode: 'blind' as Mode, // mode actually used on this square (test squares are always blind)
    cls: 'paved' as Cls,
    square: null as Square | null,
    startedAt: 0,
    cir: false,
    parcels: true,
    shapesHidden: false,
    busy: false,
    tool: 'polygon' as Tool,
    brushM: 1.5,
    snap: true,
    parcels_fc: { type: 'FeatureCollection', features: [] } as GeoJSON.FeatureCollection
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
  // Overture buildings: subtracted from parking at export, so lots can be drawn straight across them.
  map.addSource('buildings', { type: 'geojson', data: empty });
  map.addLayer({ id: 'buildings', type: 'line', source: 'buildings',
    paint: { 'line-color': '#ffd166', 'line-width': 1.5, 'line-opacity': 0.9 } });
  map.addSource('square', { type: 'geojson', data: empty });
  map.addLayer({ id: 'square', type: 'line', source: 'square', paint: { 'line-color': '#fff', 'line-width': 2 } });
  map.on('error', (e) => {
    // A square spanning two NAIP items asks each item for tiles outside it (HTTP 424); harmless.
    if (!String((e as any).error?.message || '').includes('424')) console.warn(e.error);
  });

  // ---- drawing ----------------------------------------------------------------------------
  const clsOf = (f: GeoJSONStoreFeatures): Cls => ((f.properties?.class as Cls) || 'paved');
  // Terra Draw handles editing and draws outlines only; fills come from the 'label-fill' layer
  // below, where each shape is drawn minus the cut-outs that overlap it.
  const polyStyles = {
    fillColor: (f: GeoJSONStoreFeatures) => COLORS[clsOf(f)],
    outlineColor: (f: GeoJSONStoreFeatures) => COLORS[clsOf(f)],
    fillOpacity: 0,
    outlineOpacity: () => (state.shapesHidden ? 0 : 1),
    outlineWidth: 2
  };
  // Fills and paint share one canvas laid over the sample square: one grid cell per NAIP pixel
  // (0.6 m), drawn SCALE canvas pixels wide. Polygons are rasterised into it (cut-outs erase),
  // then painted cells override them (paint cut-out clears).
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 2;
  canvas.style.display = 'none';
  document.body.appendChild(canvas);
  const ctx2d = canvas.getContext('2d', { willReadFrequently: true })!;
  map.addSource('label-canvas', { type: 'canvas', canvas, animate: false,
    coordinates: [[-96.71, 33.01], [-96.69, 33.01], [-96.69, 32.99], [-96.71, 32.99]] });
  map.addLayer({ id: 'label-canvas', type: 'raster', source: 'label-canvas',
    paint: { 'raster-opacity': 0.35, 'raster-resampling': 'nearest', 'raster-fade-duration': 0 } });
  const canvasSource = () => map.getSource('label-canvas') as maplibregl.CanvasSource;

  const grid = { n: 0, cells: new Uint8Array(0), tl: [0, 0] as Pt, kx: 1, inv: [0, 0, 0, 0], corners: [] as Pt[],
    painted: false };
  function setupGrid(sq: Square) {
    const ring = sq.geometry.coordinates[0].slice(0, 4) as Pt[];
    const byLat = [...ring].sort((a, b) => b[1] - a[1]);
    const [tl, tr] = byLat.slice(0, 2).sort((a, b) => a[0] - b[0]);
    const [bl, br] = byLat.slice(2).sort((a, b) => a[0] - b[0]);
    const kx = Math.cos((tl[1] * Math.PI) / 180);
    const ex = [(tr[0] - tl[0]) * kx, tr[1] - tl[1]], ey = [(bl[0] - tl[0]) * kx, bl[1] - tl[1]];
    const det = ex[0] * ey[1] - ey[0] * ex[1];
    const n = Math.max(1, Math.round((Math.hypot(ex[0], ex[1]) * 111320) / CELL_M));
    Object.assign(grid, { n, cells: new Uint8Array(n * n), tl, kx, corners: [tl, tr, br, bl], painted: false,
      inv: [ey[1] / det, -ey[0] / det, -ex[1] / det, ex[0] / det] });
    canvas.width = canvas.height = n * SCALE;
    canvasSource().setCoordinates([tl, tr, br, bl] as any);
  }
  /** lng/lat -> (u, v) in [0, 1] across the square (u east-ish, v south-ish). */
  function toUV(lng: number, lat: number): Pt {
    const dx = (lng - grid.tl[0]) * grid.kx, dy = lat - grid.tl[1];
    return [grid.inv[0] * dx + grid.inv[1] * dy, grid.inv[2] * dx + grid.inv[3] * dy];
  }
  function stamp(u: number, v: number, val: number) {
    const n = grid.n, r = state.brushM / CELL_M;
    const ci = u * n, cj = v * n;
    for (let j = Math.max(0, Math.floor(cj - r)); j <= Math.min(n - 1, Math.ceil(cj + r)); j++) {
      for (let i = Math.max(0, Math.floor(ci - r)); i <= Math.min(n - 1, Math.ceil(ci + r)); i++) {
        if ((i + 0.5 - ci) ** 2 + (j + 0.5 - cj) ** 2 <= r * r) grid.cells[j * n + i] = val;
      }
    }
    if (val) grid.painted = true;
  }
  function paintRle(): string | null {
    const c = grid.cells;
    if (!c.some((x) => x)) return null;
    const out: string[] = [];
    let cur = c[0], run = 0;
    for (let k = 0; k < c.length; k++) {
      if (c[k] === cur) run++;
      else { out.push(`${cur}:${run}`); cur = c[k]; run = 1; }
    }
    out.push(`${cur}:${run}`);
    return out.join(',');
  }
  function loadRle(rle: string | null | undefined) {
    grid.cells.fill(0);
    grid.painted = false;
    if (!rle) return;
    let k = 0;
    for (const part of rle.split(',')) {
      const [v, nRun] = part.split(':').map(Number);
      grid.cells.fill(v, k, Math.min(grid.cells.length, k + nRun));
      k += nRun;
    }
    grid.painted = grid.cells.some((x) => x);
  }
  // Snapping (polygon drawing and corner dragging): vertices first, then edges, of the other
  // shapes, the parcels and the sample square. Toggle with S; hold Alt to place freely.
  const rings = (g: GeoJSON.Geometry | null | undefined): number[][][] =>
    !g ? [] : g.type === 'Polygon' ? g.coordinates : g.type === 'MultiPolygon' ? g.coordinates.flat() : [];
  function snapTargets(excludeId?: string | number): number[][][] {
    const out: number[][][] = [];
    for (const f of draw.getSnapshot()) {
      if (f.geometry.type !== 'Polygon' || f.id === excludeId || f.properties?.currentlyDrawing) continue;
      out.push(...(f.geometry.coordinates as number[][][]));
    }
    for (const f of state.parcels_fc.features) out.push(...rings(f.geometry));
    if (state.square) out.push(...state.square.geometry.coordinates);
    return out;
  }
  function snapPoint(e: { containerX: number; containerY: number; heldKeys: string[] },
                     excludeId?: string | number): number[] | undefined {
    if (!state.snap || e.heldKeys.includes('Alt')) return undefined;
    const p: Pt = [e.containerX, e.containerY];
    const targets = snapTargets(excludeId).map((r) => r.map((c) => { const q = map.project(c as [number, number]); return [q.x, q.y] as Pt; }));
    let best: Pt | null = null, bestD = SNAP_VERTEX_PX;
    for (const r of targets) for (const v of r) { const d = dist(p, v); if (d < bestD) { best = v; bestD = d; } }
    if (!best) {
      bestD = SNAP_EDGE_PX;
      for (const r of targets) for (let i = 1; i < r.length; i++) {
        const q = nearestOnSegment(p, r[i - 1], r[i]);
        const d = dist(p, q);
        if (d < bestD) { best = q; bestD = d; }
      }
    }
    if (!best) return undefined;
    const ll = map.unproject(best);
    return [round(ll.lng), round(ll.lat)];
  }
  const snapping = { toCustom: (e: any, ctx: any) => snapPoint(e, ctx?.currentId), showSnappingPoint: true };
  const editable = {
    feature: {
      draggable: true,
      coordinates: { midpoints: true, draggable: true, deletable: true, snappable: snapping }
    }
  };
  const draw = new TerraDraw({
    // Terra Draw's default treats >1 px of pointer movement before the first corner as a drag, so
    // an ordinary click with a little mouse/trackpad jitter pans the map instead of placing a point.
    adapter: new TerraDrawMapLibreGLAdapter({
      map, minPixelDragDistance: 8, minPixelDragDistanceDrawing: 10, minPixelDragDistanceSelecting: 6
    }),
    modes: [
      new TerraDrawPolygonMode({ styles: polyStyles, snapping }),
      // Idle mode while painting: shapes stay visible but ignore the pointer.
      new TerraDrawRenderMode({ modeName: 'idle', styles: {} }),
      new TerraDrawAngledRectangleMode({ styles: polyStyles }),
      new TerraDrawSelectMode({
        // Selection is ours (map click below): Terra Draw picks the top shape, which made a cut-out
        // drawn before its lot unreachable.
        allowManualSelection: false,
        allowManualDeselection: false,
        flags: { polygon: editable, 'angled-rectangle': editable },
        styles: {
          selectedPolygonColor: (f: GeoJSONStoreFeatures) => COLORS[clsOf(f)],
          selectedPolygonOutlineColor: '#ffffff',
          selectedPolygonOutlineWidth: 3,
          selectedPolygonFillOpacity: 0
        }
      })
    ]
  });
  draw.start();
  (window as any).__label = { map, draw }; // dev page: handy for debugging and headless tests
  draw.on('finish', (id, ctx) => {
    if ((ctx as any)?.action === 'draw') draw.updateFeatureProperties(id, { class: state.cls });
    queueFills();
    record();
  });

  // ---- undo / redo: snapshots of the finished shapes (geometry + class), one per edit ---------
  const undoStack = { past: [] as string[], future: [] as string[], restoring: false };
  function snapshotShapes(): string {
    return JSON.stringify({ p: paintRle(), s: shapeList() });
  }
  function shapeList() {
    return (draw.getSnapshot()
      .filter((f) => f.geometry.type === 'Polygon' && !f.properties?.currentlyDrawing)
      .map((f) => {
        const { selected: _s, ...props } = f.properties;
        return { id: f.id, type: 'Feature', geometry: f.geometry, properties: props };
      }));
  }
  function updateUndoButtons() {
    ($('undo') as HTMLButtonElement).disabled = undoStack.past.length < 2;
    ($('redo') as HTMLButtonElement).disabled = undoStack.future.length === 0;
  }
  function record() {
    if (undoStack.restoring || draw.getModeState() === 'drawing') return;
    const snap = snapshotShapes();
    if (undoStack.past.at(-1) === snap) return;
    undoStack.past.push(snap);
    if (undoStack.past.length > 300) undoStack.past.shift();
    undoStack.future = [];
    updateUndoButtons();
  }
  function resetHistory() {
    undoStack.past = [snapshotShapes()];
    undoStack.future = [];
    updateUndoButtons();
  }
  function restore(snap: string) {
    undoStack.restoring = true;
    for (const f of draw.getSnapshot().filter((x) => x.properties?.selected)) draw.deselectFeature(f.id as string);
    draw.clear();
    const { s: shapesSnap, p: paintSnap } = JSON.parse(snap);
    draw.addFeatures(shapesSnap);
    loadRle(paintSnap);
    undoStack.restoring = false;
    queueFills();
    updateUndoButtons();
  }
  function undo() {
    if (draw.getModeState() === 'drawing') { toast('Finish or cancel (Esc) the shape in progress first'); return; }
    if (undoStack.past.length < 2) { toast('Nothing to undo'); return; }
    undoStack.future.push(undoStack.past.pop()!);
    restore(undoStack.past.at(-1)!);
  }
  function redo() {
    if (draw.getModeState() === 'drawing') return;
    const snap = undoStack.future.pop();
    if (!snap) { toast('Nothing to redo'); return; }
    undoStack.past.push(snap);
    restore(snap);
  }
  // Edits in Edit mode (corner drags, midpoint inserts) end on pointer-up; deletions arrive as
  // change events. Record after either; identical snapshots are ignored.
  window.addEventListener('pointerup', () => setTimeout(record, 0));
  draw.on('change', (_ids, type) => { if (type === 'delete') setTimeout(record, 0); });

  const polygons = () => draw.getSnapshot().filter((f) => f.geometry.type === 'Polygon');

  // Canvas = polygons (cut-outs erase) + painted cells on top; redrawn once per frame on change.
  let fillQueued = false;
  function renderFills() {
    fillQueued = false;
    if (!grid.n) return;
    const W = grid.n * SCALE;
    ctx2d.globalCompositeOperation = 'source-over';
    ctx2d.clearRect(0, 0, W, W);
    const shapes = polygons().map((f) => ({
      cls: (f.properties?.class as Cls) || (f.properties?.currentlyDrawing ? state.cls : 'paved'),
      ring: (f.geometry.coordinates as number[][][])[0]
    })).filter((x) => x.ring?.length >= 4);
    const fillRing = (ring: number[][]) => {
      ctx2d.beginPath();
      ring.forEach((c, i) => { const [u, v] = toUV(c[0], c[1]); i ? ctx2d.lineTo(u * W, v * W) : ctx2d.moveTo(u * W, v * W); });
      ctx2d.closePath();
      ctx2d.fill();
    };
    for (const x of shapes) if (x.cls !== 'cutout') { ctx2d.fillStyle = COLORS[x.cls]; fillRing(x.ring); }
    ctx2d.globalCompositeOperation = 'destination-out';
    ctx2d.fillStyle = '#000000';
    for (const x of shapes) if (x.cls === 'cutout') fillRing(x.ring);
    ctx2d.globalCompositeOperation = 'source-over';
    if (grid.painted) {
      const img = ctx2d.getImageData(0, 0, W, W);
      const d = img.data, n = grid.n;
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const v = grid.cells[j * n + i];
        if (!v) continue;
        const rgb = CODE_RGB[v];
        for (let b = 0; b < SCALE; b++) for (let a = 0; a < SCALE; a++) {
          const o = ((j * SCALE + b) * W + (i * SCALE + a)) * 4;
          if (v === CUTOUT_CODE) d[o + 3] = 0;
          else { d[o] = rgb[0]; d[o + 1] = rgb[1]; d[o + 2] = rgb[2]; d[o + 3] = 255; }
        }
      }
      ctx2d.putImageData(img, 0, 0);
    }
    // A static canvas source is only re-read while playing: play for two frames, then pause.
    const src = canvasSource();
    src.play();
    requestAnimationFrame(() => requestAnimationFrame(() => src.pause()));
  }
  const queueFills = () => {
    if (!fillQueued) { fillQueued = true; requestAnimationFrame(renderFills); }
  };
  draw.on('change', queueFills);
  const selected = () => polygons().filter((f) => f.properties?.selected);

  function setTool(mode: Tool) {
    if (state.tool === 'paint' && mode !== 'paint') { map.dragPan.enable(); map.boxZoom.enable(); }
    state.tool = mode;
    draw.setMode(mode === 'parcel' ? 'select' : mode === 'paint' ? 'idle' : mode);
    if (mode === 'paint') { map.dragPan.disable(); map.boxZoom.disable(); }
    $('tool-paint').classList.toggle('on', mode === 'paint');
    $('brush').hidden = mode !== 'paint';
    $('brush-size').hidden = mode !== 'paint';
    if (mode === 'parcel') for (const f of selected()) draw.deselectFeature(f.id as string);
    $('tool-polygon').classList.toggle('on', mode === 'polygon');
    $('tool-rect').classList.toggle('on', mode === 'angled-rectangle');
    $('tool-select').classList.toggle('on', mode === 'select');
    $('tool-parcel').classList.toggle('on', mode === 'parcel');
    map.setPaintProperty('parcels', 'line-width', mode === 'parcel' ? 2 : 1);
    map.setPaintProperty('parcels', 'line-opacity', mode === 'parcel' ? 1 : 0.75);
  }

  // Edit-mode selection: the smallest shape under the pointer wins (so a cut-out inside a lot is
  // reachable); clicking again in the same place cycles to the next larger shape underneath.
  function shapesAt(px: Pt, tolPx: number) {
    const ll = map.unproject(px);
    const pt: Pt = [ll.lng, ll.lat];
    return polygons().filter((f) => {
      const r = (f.geometry.coordinates as number[][][])[0];
      if (pointInRing(pt, r)) return true;
      const s = r.map((c) => { const q = map.project(c as [number, number]); return [q.x, q.y] as Pt; });
      return s.some((_, i) => i > 0 && dist(px, nearestOnSegment(px, s[i - 1], s[i])) < tolPx);
    }).sort((a, b) => ringArea((a.geometry.coordinates as number[][][])[0]) - ringArea((b.geometry.coordinates as number[][][])[0]));
  }
  function nearOutline(f: GeoJSONStoreFeatures, px: Pt, tolPx: number) {
    const s = (f.geometry.coordinates as number[][][])[0].map((c) => { const q = map.project(c as [number, number]); return [q.x, q.y] as Pt; });
    return s.some((_, i) => i > 0 && dist(px, nearestOnSegment(px, s[i - 1], s[i])) < tolPx);
  }
  function selectAt(px: Pt) {
    const cur = selected()[0];
    // Clicks on the selected shape's corners/midpoints belong to Terra Draw (insert, drag).
    if (cur && nearOutline(cur, px, 10)) return;
    const cands = shapesAt(px, 6);
    for (const f of selected()) draw.deselectFeature(f.id as string);
    if (!cands.length) return;
    const i = cur ? cands.findIndex((f) => f.id === cur.id) : -1;
    const next = cands[(i + 1) % cands.length];
    draw.selectFeature(next.id as string);
    setClassButtons(clsOf(next));
  }

  // "From parcel": click inside a parcel outline to turn it into a shape of the current class.
  function parcelToShape(px: Pt) {
    const ll = map.unproject(px);
    const pt: Pt = [ll.lng, ll.lat];
    const hits: number[][][][] = [];
    for (const f of state.parcels_fc.features) {
      const g = f.geometry;
      const polys = g.type === 'Polygon' ? [g.coordinates] : g.type === 'MultiPolygon' ? g.coordinates : [];
      for (const poly of polys) {
        if (pointInRing(pt, poly[0] as number[][]) && !poly.slice(1).some((h) => pointInRing(pt, h as number[][]))) {
          hits.push(poly as number[][][]);
        }
      }
    }
    if (!hits.length) { toast('No parcel here'); return; }
    hits.sort((a, b) => ringArea(a[0]) - ringArea(b[0]));
    loadShapes([{ type: 'Feature', properties: { class: state.cls },
      geometry: { type: 'Polygon', coordinates: [hits[0][0]] } }], false);
    queueFills();
    toast(`Parcel added as ${state.cls === 'cutout' ? 'a cut-out' : state.cls}`);
  }

  // ---- paint tool: left-drag paints the current class, right- or Alt-drag erases, Space+drag pans,
  // [ and ] change the brush. Each stroke is one undo step.
  let painting: { val: number; last: Pt } | null = null;
  let spaceHeld = false;
  const metresPerPx = () => (40075016.686 * Math.cos((map.getCenter().lat * Math.PI) / 180)) / (512 * 2 ** map.getZoom());
  function moveBrush(px?: Pt) {
    const b = $('brush');
    const d = (2 * state.brushM) / metresPerPx();
    b.style.width = b.style.height = `${d}px`;
    if (px) { b.style.left = `${px[0] - d / 2}px`; b.style.top = `${px[1] - d / 2}px`; }
    $('brush-size').textContent = `Brush ${state.brushM < 1 ? state.brushM.toFixed(1) : Math.round(state.brushM * 10) / 10} m · [ ]`;
  }
  function strokeTo(lng: number, lat: number) {
    if (!painting || !grid.n) return;
    const [u, v] = toUV(lng, lat);
    const [u0, v0] = painting.last;
    const steps = Math.max(1, Math.ceil((Math.hypot(u - u0, v - v0) * grid.n) / 0.5));
    for (let k = 1; k <= steps; k++) stamp(u0 + ((u - u0) * k) / steps, v0 + ((v - v0) * k) / steps, painting.val);
    painting.last = [u, v];
    queueFills();
  }
  // Raw pointer events, captured before Terra Draw and MapLibre: a drag that starts on a shape
  // would otherwise be claimed by Terra Draw and stop reporting moves.
  const container = map.getCanvasContainer();
  const pointAt = (e: PointerEvent): Pt => {
    const r = container.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  container.addEventListener('pointerdown', (e) => {
    if (state.tool !== 'paint' || spaceHeld || !grid.n) return;
    if (e.button !== 0 && e.button !== 2) return;
    e.stopPropagation(); e.preventDefault();
    container.setPointerCapture(e.pointerId);
    const ll = map.unproject(pointAt(e));
    const uv = toUV(ll.lng, ll.lat);
    painting = { val: e.button === 2 || e.altKey ? 0 : CODES[state.cls], last: uv };
    stamp(uv[0], uv[1], painting.val);
    queueFills();
  }, true);
  container.addEventListener('pointermove', (e) => {
    if (state.tool !== 'paint') return;
    const px = pointAt(e);
    moveBrush(px);
    if (!painting) return;
    e.stopPropagation();
    const ll = map.unproject(px);
    strokeTo(ll.lng, ll.lat);
  }, true);
  const endStroke = (e: PointerEvent) => {
    if (!painting) return;
    e.stopPropagation();
    painting = null;
    if (container.hasPointerCapture(e.pointerId)) container.releasePointerCapture(e.pointerId);
    record();
  };
  container.addEventListener('pointerup', endStroke, true);
  container.addEventListener('pointercancel', endStroke, true);
  container.addEventListener('pointerleave', () => { if (state.tool === 'paint') moveBrush([-999, -999]); });
  map.on('zoom', () => { if (state.tool === 'paint') moveBrush(); });
  container.addEventListener('contextmenu', (e) => { if (state.tool === 'paint') e.preventDefault(); });
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && state.tool === 'paint' && !e.repeat && !(e.target as HTMLElement).closest('input, textarea')) {
      spaceHeld = true; map.dragPan.enable(); e.preventDefault();
    }
  });
  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && spaceHeld) { spaceHeld = false; if (state.tool === 'paint') map.dragPan.disable(); }
  });
  const setBrush = (m: number) => { state.brushM = Math.min(15, Math.max(0.3, m)); moveBrush(); };

  map.on('click', (e) => {
    const px: Pt = [e.point.x, e.point.y];
    if (state.tool === 'select') selectAt(px);
    else if (state.tool === 'parcel') parcelToShape(px);
  });

  function setClassButtons(c: Cls) {
    state.cls = c;
    document.querySelectorAll<HTMLButtonElement>('.cls').forEach((b) => b.classList.toggle('on', b.dataset.c === c));
  }
  function setClass(c: Cls) {
    setClassButtons(c);
    const sel = selected();
    for (const f of sel) draw.updateFeatureProperties(f.id as string, { class: c });
    queueFills();
    if (sel.length) record();
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
    record();
    const bad = res.filter((r) => !r.valid).length;
    if (bad) toast(`${bad} shape(s) could not be loaded for editing`);
  }

  // ---- squares ----------------------------------------------------------------------------
  function showSquare(sq: Square) {
    state.square = sq;
    $('map-hint').hidden = true;
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
    setupGrid(sq);
    draw.clear();
    queueFills();
    (document.querySelectorAll<HTMLInputElement>('#flags input')).forEach((i) => (i.checked = false));
    ($('note') as HTMLTextAreaElement).value = '';
    state.startedAt = Date.now();
    const url = new URL(location.href);
    url.searchParams.set('campaign', state.campaign);
    url.searchParams.set('square', p.id);
    history.replaceState(null, '', url);
  }

  async function loadContext(sq: Square) {
    state.squareMode = state.mode;
    const id = encodeURIComponent(sq.properties.id);
    const c = encodeURIComponent(state.campaign);
    const [parcels, mine, buildings] = await Promise.all([
      api<GeoJSON.FeatureCollection>(`/${c}/layer/parcels?square=${id}`),
      api<{ labels: any[] }>(`/${c}/labels?square=${id}&labeler=${encodeURIComponent(state.labeler)}`),
      api<GeoJSON.FeatureCollection>(`/${c}/layer/buildings?square=${id}`)
    ]);
    (map.getSource('buildings') as maplibregl.GeoJSONSource).setData(buildings);
    state.parcels_fc = parcels;
    (map.getSource('parcels') as maplibregl.GeoJSONSource).setData(parcels);
    const last = mine.labels.filter((l) => l.outcome !== 'skip').at(-1);
    if (last) {
      loadShapes(last.polygons, false);
      if (last.paint && last.paint.width === grid.n) { loadRle(last.paint.rle); queueFills(); }
      document.querySelectorAll<HTMLInputElement>('#flags input').forEach((i) => (i.checked = last.flags.includes(i.value)));
      ($('note') as HTMLTextAreaElement).value = last.note || '';
      toast(`Loaded your earlier label (${last.outcome}); resubmitting replaces it`, 3500);
    } else if (state.mode === 'correction') {
      const pre = await api<GeoJSON.FeatureCollection>(`/${c}/layer/prefill?square=${id}`);
      loadShapes(pre.features, true);
      toast(`${pre.features.length} existing lot(s) pre-loaded — fix, delete or add`);
    } else if (state.mode === 'model') {
      if (sq.properties.holdout) {
        state.squareMode = 'blind';
        toast('Test square: draw it from scratch (model guesses are hidden on test squares)', 4500);
      } else {
        const pre = await api<GeoJSON.FeatureCollection>(`/${c}/layer/model_prefill?square=${id}`);
        loadShapes(pre.features.map((f) => ({ ...f, properties: { ...f.properties, class: 'paved' } })), true);
        toast(pre.features.length
          ? `${pre.features.length} model guess(es) pre-loaded as paved — fix edges, reclassify gravel, delete misses`
          : 'The model found no parking here — add any it missed', 4000);
      }
    }
    resetHistory(); // undo never reaches back into the previous square or past the loaded shapes
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
      state.square = null;
      $('map-hint').textContent = 'Nothing left for you in this campaign. Thank you!';
      $('map-hint').hidden = false;
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
    const rle = outcome === 'labeled' ? paintRle() : null;
    const paintsClass = grid.cells.some((v) => v > 0 && v !== CUTOUT_CODE);
    if (outcome === 'labeled' && !shapes.some((f) => clsOf(f) !== 'cutout') && !paintsClass) {
      toast('Nothing drawn or painted — use “No parking” (N) if the square has none');
      return;
    }
    const work = polygons().length + (grid.painted ? 1 : 0);
    if (outcome !== 'labeled' && work &&
        !confirm(`Discard your shapes/paint and submit “${outcome.replace('_', ' ')}”?`)) return;
    const p = state.square.properties;
    const body = {
      labeler: state.labeler, square_id: p.id, outcome, mode: state.squareMode,
      polygons: shapes.map((f) => ({
        type: 'Feature',
        geometry: { type: 'Polygon', coordinates: (f.geometry.coordinates as number[][][]).map(closedRing) },
        properties: { class: clsOf(f), ...(f.properties?.from_prefill ? { from_prefill: true } : {}) }
      })),
      flags: [...document.querySelectorAll<HTMLInputElement>('#flags input:checked')].map((i) => i.value),
      note: ($('note') as HTMLTextAreaElement).value.trim(),
      seconds: Math.round((Date.now() - state.startedAt) / 1000),
      naip_year: p.naip_year ?? null,
      imagery: p.imagery_date ?? null,
      // Paint grid: one cell per NAIP pixel across the square, row-major from the top-left corner;
      // values 0 none, 1 paved, 2 unpaved, 3 structure, 4 unsure, 5 cut-out; run-length "v:count,…".
      paint: rle ? { width: grid.n, height: grid.n, cell_m: CELL_M, corners: grid.corners, rle } : null
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
    $('mode-model').classList.toggle('on', m === 'model');
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
  let buildingsOn = true;
  const toggleBuildings = () => {
    buildingsOn = !buildingsOn;
    $('view-buildings').classList.toggle('on', buildingsOn);
    map.setLayoutProperty('buildings', 'visibility', buildingsOn ? 'visible' : 'none');
  };
  const toggleShapes = () => {
    if (draw.getModeState() === 'drawing') return; // re-adding features would drop a shape in progress
    state.shapesHidden = !state.shapesHidden;
    $('view-shapes').classList.toggle('on', state.shapesHidden);
    map.setLayoutProperty('label-canvas', 'visibility', state.shapesHidden ? 'none' : 'visible');
    // Re-add the features so Terra Draw restyles them.
    const snap = polygons().map((f) => {
      const { selected: _s, ...props } = f.properties;
      return { ...f, properties: props } as GeoJSONStoreFeatures;
    });
    undoStack.restoring = true;
    draw.clear();
    draw.addFeatures(snap);
    undoStack.restoring = false;
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
  $('mode-model').onclick = () => setMode('model');
  $('tool-polygon').onclick = () => setTool('polygon');
  $('tool-rect').onclick = () => setTool('angled-rectangle');
  $('tool-select').onclick = () => setTool('select');
  $('tool-parcel').onclick = () => setTool('parcel');
  $('tool-paint').onclick = () => setTool('paint');
  $('undo').onclick = undo;
  $('redo').onclick = redo;
  // Reference only: shows where you are on Google's newer imagery. Never trace from it (licence).
  const openGoogleMaps = () => {
    const c = map.getCenter();
    const z = Math.min(21, Math.round(map.getZoom() + 1)); // MapLibre's 512 px tiles sit one level below Google's
    window.open(`https://www.google.com/maps/@${c.lat.toFixed(6)},${c.lng.toFixed(6)},${z}z/data=!3m1!1e3`, '_blank', 'noopener');
  };
  $('gmaps').onclick = openGoogleMaps;
  const toggleSnap = () => {
    state.snap = !state.snap;
    $('tool-snap').classList.toggle('on', state.snap);
    toast(state.snap ? 'Snapping on (hold Alt to place freely)' : 'Snapping off');
  };
  $('tool-snap').onclick = toggleSnap;
  $('tool-delete').onclick = deleteSelected;
  document.querySelectorAll<HTMLButtonElement>('.cls').forEach((b) => (b.onclick = () => setClass(b.dataset.c as Cls)));
  $('view-cir').onclick = toggleCir;
  $('view-parcels').onclick = toggleParcels;
  $('view-buildings').onclick = toggleBuildings;
  $('view-buildings').classList.add('on');
  $('view-shapes').onclick = toggleShapes;
  $('view-fit').onclick = fit;
  $('submit').onclick = () => submit('labeled');
  $('no-parking').onclick = () => submit('no_parking');
  $('cant-tell').onclick = () => submit('cant_tell');
  $('skip').onclick = () => submit('skip');
  $('guide-btn').onclick = () => ($('guide') as HTMLDialogElement).showModal();

  const nameInput = $('labeler') as HTMLInputElement;
  nameInput.value = state.labeler;
  const commitName = () => {
    const v = nameInput.value.trim();
    if (v === state.labeler && state.square) return;
    state.labeler = v;
    store('label.labeler', state.labeler);
    if (state.labeler && !state.square) void next();
  };
  nameInput.onchange = commitName;
  nameInput.onkeydown = (e) => { if (e.key === 'Enter') { commitName(); nameInput.blur(); } };
  ($('guide') as HTMLDialogElement).addEventListener('close', () => { if (!state.labeler) nameInput.focus(); });

  // H held = peek under the shapes; released = show them again. The button still toggles.
  let heldHide = false;
  window.addEventListener('keyup', (e) => {
    if ((e.key === 'h' || e.key === 'H') && heldHide) { heldHide = false; if (state.shapesHidden) toggleShapes(); }
  });
  window.addEventListener('keydown', (e) => {
    const t = e.target as HTMLElement;
    if (t.closest('input, textarea, select') || ($('guide') as HTMLDialogElement).open) return;
    if ((e.ctrlKey || e.metaKey) && !e.altKey) {
      const k = e.key.toLowerCase();
      if (k === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
      else if (k === 'y') { e.preventDefault(); redo(); }
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (k === '?') { ($('guide') as HTMLDialogElement).showModal(); return; }
    // No shortcuts until a square is loaded (stray keys while finding the name box would
    // otherwise toggle views).
    if (!state.square) {
      if (!state.labeler) nameInput.focus();
      return;
    }
    if (CLASS_KEYS[k]) setClass(CLASS_KEYS[k]);
    else if (k === 'd' || k === 'D') setTool('polygon');
    else if (k === 'r' || k === 'R') setTool('angled-rectangle');
    else if (k === 'e' || k === 'E') setTool('select');
    else if (k === 'g' || k === 'G') setTool('parcel');
    else if (k === 'b' || k === 'B') setTool('paint');
    else if (k === '[') setBrush(state.brushM * 0.75);
    else if (k === ']') setBrush(state.brushM / 0.75);
    else if (k === 'm' || k === 'M') openGoogleMaps();
    else if (k === 's' || k === 'S') toggleSnap();
    else if (k === 'Delete' || k === 'Backspace') { deleteSelected(); e.preventDefault(); }
    else if (k === 'c' || k === 'C') toggleCir();
    else if (k === 'p' || k === 'P') toggleParcels();
    else if (k === 'o' || k === 'O') toggleBuildings();
    else if ((k === 'h' || k === 'H') && !e.repeat) { if (!state.shapesHidden) { toggleShapes(); heldHide = true; } }
    else if (k === 'f' || k === 'F') fit();
    else if (k === 'Enter') {
      // While a shape is being drawn, Enter closes it (Terra Draw); submit only when idle.
      if (draw.getModeState() !== 'drawing') submit('labeled');
    }
    else if (k === 'n' || k === 'N') submit('no_parking');
    else if (k === 't' || k === 'T') submit('cant_tell');
    else if (k === 'k' || k === 'K') submit('skip');
  });

  // ---- start ------------------------------------------------------------------------------
  setMode(state.mode);
  setClass('paved');
  setTool('polygon');
  $('tool-snap').classList.add('on');
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
