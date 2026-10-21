// maplibre-gl 6 is ESM-only and finds its worker next to its own module via import.meta.url,
// which doesn't survive Vite's bundling (the worker file is never emitted). Hand it Vite's bundled
// worker URL instead. Import this before constructing any map.
import { setWorkerUrl } from 'maplibre-gl';
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';

setWorkerUrl(workerUrl);
