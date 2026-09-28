/**
 * PMTiles byte source that lets the browser's HTTP cache actually cache tiles.
 *
 * Why: pmtiles' stock FetchSource sends every tile as a Range request against ONE URL. Chrome's
 * HTTP cache locks a URL's cache entry while one range is being written, and concurrent range
 * requests for the same URL bypass the cache entirely — and a map view fires dozens at once. So in
 * practice nothing was cached: a repeat visit to Seattle re-downloaded 12.9 MB of tiles, NYC 5.8 MB
 * (measured on prod, 2026-09-28). Sequential range requests to one URL DO cache; concurrent ones
 * never do (verified in headless Chromium with a disk-backed profile).
 *
 * Fix: give each byte range its own URL by appending `r=<offset>-<length>` to the query string.
 * Each range is then an independent cache entry (no lock contention), while the request still
 * carries the same Range header, so the server returns exactly the same bytes — the data proxy
 * builds the blob path from the filename only and ignores the query string. `v=<pmtilesVersion>`
 * stays in the URL, so a re-bake (version bump) still gets a fresh set of cache keys.
 *
 * Behaviour otherwise mirrors pmtiles' FetchSource (v3.2): ETag-mismatch detection + reload,
 * the 416 short-file retry, the "server ignored Range" guard, and no-store on Chromium/Windows
 * (pmtiles disables caching there to dodge a Chromium range-cache bug; we keep that).
 */
import { EtagMismatch, type RangeResponse, type Source } from 'pmtiles';

const CHROMIUM_WINDOWS = (() => {
  const ua = (globalThis.navigator && globalThis.navigator.userAgent) || '';
  return ua.includes('Windows') && /Chrome|Chromium|Edg|OPR|Brave/.test(ua);
})();

export class PerRangeUrlSource implements Source {
  private mustReload = false;

  constructor(private readonly url: string) {}

  /** The archive key (what the maplibre pmtiles:// protocol looks instances up by): the plain URL. */
  getKey(): string {
    return this.url;
  }

  private rangeUrl(offset: number, length: number): string {
    return `${this.url}${this.url.includes('?') ? '&' : '?'}r=${offset}-${length}`;
  }

  async getBytes(offset: number, length: number, signal?: AbortSignal, etag?: string): Promise<RangeResponse> {
    const cache: RequestCache | undefined = this.mustReload ? 'reload' : (CHROMIUM_WINDOWS ? 'no-store' : undefined);
    let resp = await fetch(this.rangeUrl(offset, length), {
      signal,
      cache,
      headers: { range: `bytes=${offset}-${offset + length - 1}` },
    });

    // Archive shorter than the initial 16 KiB header read: re-request exactly what exists.
    if (offset === 0 && resp.status === 416) {
      const contentRange = resp.headers.get('Content-Range');
      if (!contentRange || !contentRange.startsWith('bytes */')) {
        throw new Error('Missing content-length on 416 response');
      }
      const actualLength = +contentRange.substr(8);
      resp = await fetch(this.rangeUrl(0, actualLength), {
        signal,
        cache: 'reload',
        headers: { range: `bytes=0-${actualLength - 1}` },
      });
    }

    let newEtag = resp.headers.get('ETag');
    if (newEtag?.startsWith('W/')) newEtag = null;
    if (resp.status === 416 || (etag && newEtag && newEtag !== etag)) {
      // The file changed under us (re-upload mid-session, or a stale cached range): bypass the
      // cache from now on; pmtiles catches this, drops its header/directory caches and retries.
      this.mustReload = true;
      throw new EtagMismatch(
        `Server returned non-matching ETag ${etag} after one retry. Check browser extensions and servers for issues that may affect correct ETag headers.`
      );
    }
    if (resp.status >= 300) {
      throw new Error(`Bad response code: ${resp.status}`);
    }
    const contentLength = resp.headers.get('Content-Length');
    if (resp.status === 200 && (!contentLength || +contentLength > length)) {
      throw new Error(
        'Server returned no content-length header or content-length exceeding request. Check that your storage backend supports HTTP Byte Serving.'
      );
    }
    return {
      data: await resp.arrayBuffer(),
      etag: newEtag || undefined,
      cacheControl: resp.headers.get('Cache-Control') || undefined,
      expires: resp.headers.get('Expires') || undefined,
    };
  }
}
