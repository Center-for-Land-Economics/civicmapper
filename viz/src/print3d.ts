/**
 * PROTOTYPE (localhost-only) — export the currently-visualized hexagon surface, at the H3
 * resolution currently being depicted, as a 3D-printable STL. Ports the pure-TS H3 mesh
 * builder from PUTITONAMAP (see viz/src/mesh/*, copied from C:\git\geovizwiz\viz).
 *
 * Scope: ALL hexes at the current resolution (whole county). queryRenderedFeatures is
 * viewport-only, so we read the `parcels_low` hex tiles straight from the PMTiles archive
 * at the current zoom, decode the MVT, recover each hex's H3 id from its centroid
 * (latLngToCell — this also dedupes tile-seam fragments), and feed the source mesh builder.
 *
 * Gated to localhost; the whole panel is hidden otherwise.
 */
import type maplibregl from 'maplibre-gl';
import { PMTiles } from 'pmtiles';
import { VectorTile } from '@mapbox/vector-tile';
import Pbf from 'pbf';
import { latLngToCell, getHexagonAreaAvg } from 'h3-js';
import { requestMeshExport, type MeshExportResult } from './mesh/mesh.client';

export interface Print3DContext {
  archive: PMTiles;                                  // the city's PMTiles archive
  bounds: [number, number, number, number];          // [minLon, minLat, maxLon, maxLat]
  sourceLayer: string;                               // 'parcels_low'
  parcelMinZoom: number;                             // hex→parcel handoff zoom
}

export interface Init3DPrintOpts {
  map: maplibregl.Map;
  getField: () => string | null;                     // current value field (height driver)
  computeMetric: (props: Record<string, any>) => number | null; // on-screen metric per feature
  makeSelectedTest: () => (v: any) => boolean;       // predicate: is this region value currently lit
  getRegionField: () => string | null;               // active group's field, e.g. 'jurisdiction'
  /** Returns the PMTiles context, or null if the city isn't a PMTiles/hex city. */
  getContext: () => Print3DContext | null;
}

export function is3DPrintEnabled(): boolean {
  const h = typeof window !== 'undefined' ? window.location.hostname : '';
  return h === 'localhost' || h === '127.0.0.1' || h.endsWith('.localhost');
}

// --- slippy tile math ---
const lon2x = (lon: number, z: number) => Math.floor(((lon + 180) / 360) * 2 ** z);
const lat2y = (lat: number, z: number) => {
  const r = (lat * Math.PI) / 180;
  return Math.floor(((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * 2 ** z);
};
const clampTile = (v: number, z: number) => Math.max(0, Math.min(2 ** z - 1, v));

/** Average of a polygon's outer-ring vertices (excluding the closing duplicate). */
function ringCentroid(geom: any): { lng: number; lat: number; ring: number[][] } | null {
  const ring: number[][] | undefined =
    geom?.type === 'Polygon' ? geom.coordinates?.[0]
      : geom?.type === 'MultiPolygon' ? geom.coordinates?.[0]?.[0] : undefined;
  if (!ring || ring.length < 3) return null;
  const closed = ring.length > 1 &&
    ring[0][0] === ring[ring.length - 1][0] && ring[0][1] === ring[ring.length - 1][1];
  const pts = closed ? ring.slice(0, -1) : ring;
  let sx = 0, sy = 0;
  for (const [x, y] of pts) { sx += x; sy += y; }
  return { lng: sx / pts.length, lat: sy / pts.length, ring: pts };
}

/** Detect the H3 resolution from sample hexes by area. Uses the MAX sample area (a full,
 * un-clipped hex; tile-edge fragments are only ever smaller) matched to the avg hex area at
 * each resolution. Area changes ~7× per level, so the match is unambiguous (unlike edge
 * length / circumradius, which were noisy and mislabeled by one level). */
function detectResolution(samples: { lng: number; lat: number; ring: number[][] }[]): number {
  let maxArea = 0;
  for (const s of samples) {
    const ring = s.ring;
    if (ring.length < 3) continue;
    const mPerDegLat = 111320, mPerDegLng = 111320 * Math.cos((s.lat * Math.PI) / 180);
    let a = 0; // shoelace, in m²
    for (let i = 0; i < ring.length; i++) {
      const j = (i + 1) % ring.length;
      a += (ring[i][0] * mPerDegLng) * (ring[j][1] * mPerDegLat)
         - (ring[j][0] * mPerDegLng) * (ring[i][1] * mPerDegLat);
    }
    const area = Math.abs(a) / 2;
    if (area > maxArea) maxArea = area;
  }
  if (maxArea <= 0) return 9;
  let best = 9, bestErr = Infinity;
  for (let r = 0; r <= 15; r++) {
    const err = Math.abs(getHexagonAreaAvg(r, 'm2') - maxArea);
    if (err < bestErr) { bestErr = err; best = r; }
  }
  return best;
}

/** Read every hex of the current resolution from the PMTiles archive → {h3, metric}[]. */
async function gatherCells(
  ctx: Print3DContext, map: maplibregl.Map,
  computeMetric: (p: any) => number | null, field: string | null,
  selectedOnly: boolean, selectedTest: ((v: any) => boolean) | null, regionField: string | null,
): Promise<{ cells: { h3: string; metric: number }[]; res: number | null; rawCount: number }> {
  // maplibre draws vector tiles at floor(zoom) for a 512px source, so read tiles at the SAME
  // zoom the screen is showing (round() would jump to the next resolution band near .5).
  const Z = Math.max(0, Math.min(Math.floor(map.getZoom()), ctx.parcelMinZoom - 1));
  const [w, s, e, n] = ctx.bounds;
  const xMin = clampTile(lon2x(w, Z), Z), xMax = clampTile(lon2x(e, Z), Z);
  const yMin = clampTile(lat2y(n, Z), Z), yMax = clampTile(lat2y(s, Z), Z);

  const feats: { props: any; centroid: { lng: number; lat: number; ring: number[][] } }[] = [];
  for (let x = xMin; x <= xMax; x++) {
    for (let y = yMin; y <= yMax; y++) {
      let resp: any;
      try { resp = await ctx.archive.getZxy(Z, x, y); } catch { resp = null; }
      if (!resp?.data) continue;
      const layer = new VectorTile(new Pbf(new Uint8Array(resp.data))).layers[ctx.sourceLayer];
      if (!layer) continue;
      for (let i = 0; i < layer.length; i++) {
        const gj = layer.feature(i).toGeoJSON(x, y, Z);
        const c = ringCentroid(gj.geometry);
        if (c) feats.push({ props: gj.properties, centroid: c });
      }
    }
  }
  if (!feats.length) return { cells: [], res: null, rawCount: 0 };

  // Detect resolution from the decoded full-county set (lots of full hexes → reliable max-area).
  // Because tiles are read at floor(zoom) — maplibre's on-screen tile zoom — this is exactly
  // the resolution being depicted.
  const res = detectResolution(feats.slice(0, 500).map(f => f.centroid));
  const test = selectedOnly ? selectedTest : null;
  const m = new Map<string, number>();
  for (const f of feats) {
    // "Selected only": keep hexes whose ACTIVE-GROUP region is currently lit (not the metric).
    if (test && regionField) { if (!test(f.props?.[regionField])) continue; }
    const metric = computeMetric(f.props) ?? (field ? Number(f.props?.[field]) : NaN);
    if (!Number.isFinite(metric)) continue;
    m.set(latLngToCell(f.centroid.lat, f.centroid.lng, res), metric as number);
  }
  return { cells: [...m].map(([h3, metric]) => ({ h3, metric })), res, rawCount: feats.length };
}

function download(filename: string, data: BlobPart, mime: string) {
  const blob = new Blob([data], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

export function init3DPrint(opts: Init3DPrintOpts) {
  const card = document.getElementById('print3dCard') as HTMLDivElement | null;
  if (!card) return;
  // Localhost-only + needs a PMTiles (hex) city; otherwise the panel never appears.
  if (!is3DPrintEnabled() || !opts.getContext()) { card.style.display = 'none'; return; }
  card.style.display = '';

  const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T | null;
  const btn = $<HTMLButtonElement>('btn3DPrint');
  const status = $<HTMLParagraphElement>('print3DStatus');
  const info = $<HTMLParagraphElement>('print3dInfo');
  const footprint = $<HTMLInputElement>('print3dFootprint');
  const maxHeight = $<HTMLInputElement>('print3dMaxHeight');
  const base = $<HTMLInputElement>('print3dBase');
  const regionsGroup = $<HTMLDivElement>('print3dRegionsGroup');
  const regions = $<HTMLSelectElement>('print3dRegions');

  // Region filter only meaningful when the city has a region grouping.
  if (regionsGroup && !opts.getRegionField()) regionsGroup.style.display = 'none';

  const setStatus = (msg: string) => { if (status) status.textContent = msg; };

  // Reflect zoom state: disable at parcel zoom, otherwise show the on-screen resolution.
  const updateState = () => {
    const ctx = opts.getContext();
    if (!ctx) return;
    const atParcelZoom = opts.map.getZoom() >= ctx.parcelMinZoom;
    if (btn) btn.disabled = atParcelZoom;
    if (info) {
      info.textContent = atParcelZoom
        ? 'Zoom out to a hex view to 3D print.'
        : 'Exports every hex at the resolution currently shown (whole county).';
    }
  };
  opts.map.on('zoom', updateState);
  updateState();

  btn?.addEventListener('click', async () => {
    const ctx = opts.getContext();
    if (!ctx || btn.disabled) return;
    const field = opts.getField();
    if (!field) { setStatus('No metric selected.'); return; }

    const prevLabel = btn.textContent;
    btn.disabled = true; btn.textContent = 'Reading hexes…'; setStatus('');
    try {
      const selectedOnly = regions?.value === 'selected';
      const { cells, res, rawCount } = await gatherCells(
        ctx, opts.map, opts.computeMetric, field, selectedOnly, opts.makeSelectedTest(), opts.getRegionField());
      if (!cells.length) {
        setStatus('No hexes found at this view — zoom out to a hex view.');
        return;
      }
      btn.textContent = 'Building mesh…';
      setStatus(`${cells.length.toLocaleString()} hexes (r${res}) · ${rawCount.toLocaleString()} tile features`);

      const meshOpts = {
        footprintMm: Number(footprint?.value) || 180,
        maxHeightMm: Number(maxHeight?.value) || 30,
        baseThicknessMm: Number(base?.value) || 2,
      };
      await new Promise<void>((resolve) => {
        requestMeshExport(cells, meshOpts, { stl: true, obj: false }, {
          onProgress: (f) => { btn.textContent = `Building mesh… ${Math.round(f * 100)}%`; },
          onError: (m) => { setStatus(`Export failed: ${m}`); resolve(); },
          onResult: (r: MeshExportResult) => {
            if (r.stl) {
              download(`hexes-r${res}.stl`, r.stl, 'model/stl');
              const d = r.dims, rep = r.report;
              // boundaryEdges === 0 means no holes (printable). nonManifoldEdges > 0 is
              // expected here — adjacent hex columns share walls — and slicers union them.
              setStatus(`Exported r${res} · ${r.triangleCount.toLocaleString()} tris · `
                + `${d.x.toFixed(0)}×${d.y.toFixed(0)}×${d.z.toFixed(0)} mm · `
                + (rep.boundaryEdges === 0 ? 'closed ✓ (no holes)' : `⚠ ${rep.boundaryEdges} open edges`));
            } else {
              setStatus('Export produced no geometry.');
            }
            resolve();
          },
        });
      });
    } catch (err) {
      setStatus(`Export failed: ${String(err)}`);
    } finally {
      btn.disabled = false; btn.textContent = prevLabel || 'Export STL';
      updateState();
    }
  });
}
