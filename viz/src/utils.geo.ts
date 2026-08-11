export function roundGeometryInPlace(f: GeoJSON.Feature, decimals = 6) {
  const factor = Math.pow(10, decimals);
  const round = (n: number) => Math.round(n * factor) / factor;
  const walk = (coords: any) => {
    if (!Array.isArray(coords)) return;
    if (typeof coords[0] === 'number') { coords[0] = round(coords[0]); coords[1] = round(coords[1]); }
    else for (const c of coords) walk(c);
  };
  if (f.geometry) walk((f.geometry as any).coordinates);
}

// Shoelace-ish signed area (sign only). >0 and <0 distinguish the two ring orientations.
function ringSignedArea(ring: number[][]): number {
  let s = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    s += (ring[i + 1][0] - ring[i][0]) * (ring[i + 1][1] + ring[i][1]);
  }
  return s;
}

// Normalize polygon ring winding in place so every exterior ring has the SAME orientation (and
// holes the opposite). Region-boundary GeoJSONs come with mixed winding; consistent winding lets a
// single `line-offset` inset every outline toward its own interior, so adjoining regions each show
// their own boundary band instead of one overdrawing the other. Exterior → CCW (signed area < 0
// in this convention), holes → CW.
export function normalizeWindingInPlace(features: GeoJSON.Feature[]) {
  for (const f of features) {
    const g: any = f.geometry;
    if (!g) continue;
    const polys: number[][][][] =
      g.type === 'Polygon' ? [g.coordinates] : g.type === 'MultiPolygon' ? g.coordinates : [];
    for (const poly of polys) {
      poly.forEach((ring, i) => {
        if (!Array.isArray(ring) || ring.length < 4) return;
        const ccw = ringSignedArea(ring) < 0;
        if ((i === 0 && !ccw) || (i > 0 && ccw)) ring.reverse();
      });
    }
  }
}

export function trimPropertiesInPlace(features: GeoJSON.Feature[], keep: Set<string>) {
  for (const feat of features) {
    const p = (feat.properties ||= {});
    for (const k of Object.keys(p as any)) { if (!keep.has(k)) delete (p as any)[k]; }
  }
}

export function bbox(fc: GeoJSON.FeatureCollection): [number, number, number, number] | null {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const add = (x: number, y: number) => { if (x < minX) minX = x; if (y < minY) minY = y; if (x > maxX) maxX = x; if (y > maxY) maxY = y; };
  const walk = (coords: any) => Array.isArray(coords[0]) ? coords.forEach(walk) : add(coords[0], coords[1]);
  for (const f of fc.features) {
    if (!f.geometry) continue;
    const g = f.geometry;
    if (g.type === 'Polygon' || g.type === 'MultiPolygon' || g.type === 'LineString' || g.type === 'MultiLineString') walk(g.coordinates);
    if (g.type === 'Point') add(g.coordinates[0], g.coordinates[1]);
    if (g.type === 'MultiPoint') (g.coordinates as any[]).forEach((c: number[]) => add(c[0], c[1]));
  }
  return (Number.isFinite(minX) && Number.isFinite(minY) && Number.isFinite(maxX) && Number.isFinite(maxY)) ? [minX, minY, maxX, maxY] : null;
}
