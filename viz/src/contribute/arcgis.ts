/**
 * ArcGIS FeatureServer probing utilities.
 *
 * Queries `{url}?f=json` to discover the layer schema:
 * field names, types, geometry type, spatial reference, and record count.
 *
 * CORS note: many ArcGIS servers support CORS, but not all. If the request
 * fails, the caller should fall back to manual field entry.
 */

export interface ArcGISField {
  name: string;
  alias: string;
  type: string; // esriFieldTypeString, esriFieldTypeInteger, esriFieldTypeDouble, ...
}

export interface ArcGISLayerInfo {
  name: string;
  geometryType: string; // esriGeometryPolygon, etc.
  spatialReference?: { wkid?: number; latestWkid?: number };
  fields: ArcGISField[];
  maxRecordCount?: number;
  supportsStatistics?: boolean;
  // approximate count from a statistics query
  approxCount?: number;
}

/** Normalize a raw ArcGIS FeatureServer URL to remove trailing slashes and query strings. */
export function normalizeArcGISUrl(url: string): string {
  try {
    const u = new URL(url.trim());
    u.search = '';
    return u.toString().replace(/\/+$/, '');
  } catch {
    return url.trim();
  }
}

/** Returns true if the URL looks like a FeatureServer or MapServer layer URL. */
export function looksLikeArcGISUrl(url: string): boolean {
  return /\/(Feature|Map)Server\/\d+/i.test(url);
}

/**
 * Probe an ArcGIS FeatureServer layer.
 * Returns null if the server doesn't support CORS or the URL is invalid.
 */
export async function probeArcGISLayer(baseUrl: string): Promise<ArcGISLayerInfo | null> {
  const url = normalizeArcGISUrl(baseUrl);
  const endpoint = `${url}?f=json`;

  try {
    const resp = await fetch(endpoint, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(10_000),
    });
    if (!resp.ok) return null;
    const data = await resp.json();

    if (!data?.fields) return null;

    const info: ArcGISLayerInfo = {
      name: data.name ?? 'Unknown Layer',
      geometryType: data.geometryType ?? '',
      spatialReference: data.extent?.spatialReference ?? data.sourceSpatialReference ?? undefined,
      fields: (data.fields ?? []).map((f: any) => ({
        name: f.name,
        alias: f.alias ?? f.name,
        type: f.type ?? '',
      })),
      maxRecordCount: data.maxRecordCount,
      supportsStatistics: data.supportsStatistics ?? false,
    };

    // Try to get an approximate count
    if (info.supportsStatistics) {
      try {
        const countUrl = `${url}/query?where=1%3D1&returnCountOnly=true&f=json`;
        const countResp = await fetch(countUrl, { signal: AbortSignal.timeout(8_000) });
        if (countResp.ok) {
          const countData = await countResp.json();
          if (typeof countData?.count === 'number') {
            info.approxCount = countData.count;
          }
        }
      } catch {
        // Non-fatal — count stays undefined
      }
    }

    return info;
  } catch {
    return null;
  }
}

/** Suggest which fields likely hold the land value, improvement value, and land use category. */
export function suggestFields(fields: ArcGISField[]): {
  landValue?: string;
  improvementValue?: string;
  landUseCategory?: string;
  parcelId?: string;
} {
  const names = fields.map(f => ({ name: f.name, lower: f.name.toLowerCase() + ' ' + f.alias.toLowerCase() }));

  function match(patterns: string[]): string | undefined {
    for (const p of patterns) {
      const f = names.find(n => n.lower.includes(p));
      if (f) return f.name;
    }
    return undefined;
  }

  return {
    landValue: match(['landval', 'land_val', 'landmkt', 'mktlnd', 'land_value', 'land_market', 'curr_land', 'total_land']),
    improvementValue: match(['imprval', 'impr_val', 'mktimp', 'improvement_value', 'impr_market', 'curr_impr', 'total_impr']),
    landUseCategory: match(['class', 'landuse', 'land_use', 'usecode', 'use_code', 'lutype', 'lu_type', 'proptype', 'prop_type', 'usegroup']),
    parcelId: match(['parcelid', 'parcel_id', 'pin', 'parid', 'apn', 'blockfaceid', 'taxid']),
  };
}

/** Human-readable description of an ArcGIS geometry type. */
export function describeGeometryType(gt: string): string {
  const map: Record<string, string> = {
    esriGeometryPolygon: 'Polygon (parcel footprints)',
    esriGeometryPoint: 'Point (centroids only)',
    esriGeometryPolyline: 'Polyline',
    esriGeometryMultipoint: 'Multipoint',
  };
  return map[gt] ?? gt;
}

/** Human-readable CRS description from WKID. */
export function describeCRS(wkid?: number): string {
  if (!wkid) return 'Unknown';
  if (wkid === 4326) return 'EPSG:4326 (WGS84)';
  if (wkid === 4269) return 'EPSG:4269 (NAD83)';
  if (wkid === 3857) return 'EPSG:3857 (Web Mercator)';
  // State Plane ranges
  if (wkid >= 2224 && wkid <= 2313) return `EPSG:${wkid} (State Plane)`;
  if (wkid >= 26700 && wkid <= 26999) return `EPSG:${wkid} (UTM)`;
  return `EPSG:${wkid}`;
}
