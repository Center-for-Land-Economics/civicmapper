import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig(() => {
  return {
    // Serve from domain root in production deployments (Azure SWA)
    // If you later host under a subpath, set base accordingly
    base: '/',
    // Point env loading at a sandbox-safe folder to avoid .env permission issues
    envDir: resolve(__dirname, 'env'),
    build: {
      rollupOptions: {
        output: {
          manualChunks(id: string) {
            if (id.includes('node_modules/maplibre-gl')) return 'maplibre';
            if (id.includes('node_modules/pmtiles')) return 'pmtiles';
            if (
              id.includes('node_modules/geoparquet') ||
              id.includes('node_modules/hyparquet')
            ) {
              return 'parquet';
            }
            return undefined;
          }
        },
        input: {
          main: resolve(__dirname, 'index.html'),
          app: resolve(__dirname, 'app.html'),
          cities: resolve(__dirname, 'cities.html'),
          parking: resolve(__dirname, 'parking.html'),
          contribute: resolve(__dirname, 'contribute.html')
        }
      }
    },
    // Dev server proxies
    server: {
      proxy: {
        // Proxy API requests to local API server on port 8080
        '/api': {
          target: 'http://localhost:8080',
          changeOrigin: true,
          secure: false
        },

        // Proxy remote GeoParquet to avoid browser CORS (dev only)
        '/data': {
          target: process.env.VITE_PARQUET_BASE_URL || 'https://landeconomics.blob.core.windows.net/parquets-dev',
          changeOrigin: true,
          secure: true,
          rewrite: (p: string) => p.replace(/^\/data/, '')
        }
      }
    }
  };
});
