import { API_BASE, ENV_LABEL } from './env';

type TelemetryPayload = Record<string, unknown>;

const TELEMETRY_ENDPOINT = `${API_BASE}/telemetry`;

declare global {
  interface Window {
    __gvwTelemetryInstalled?: boolean;
  }
}

function buildTelemetryEvent(name: string, payload: TelemetryPayload = {}) {
  return {
    name,
    env: ENV_LABEL,
    href: window.location.href,
    path: window.location.pathname,
    pageTitle: document.title,
    referrer: document.referrer || '',
    userAgent: navigator.userAgent,
    ts: new Date().toISOString(),
    ...payload
  };
}

function sendTelemetry(event: ReturnType<typeof buildTelemetryEvent>) {
  const body = JSON.stringify(event);

  try {
    if (navigator.sendBeacon) {
      const blob = new Blob([body], { type: 'application/json' });
      if (navigator.sendBeacon(TELEMETRY_ENDPOINT, blob)) return;
    }
  } catch {}

  fetch(TELEMETRY_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
    keepalive: true
  }).catch(() => {});
}

export function trackEvent(name: string, payload: TelemetryPayload = {}) {
  sendTelemetry(buildTelemetryEvent(name, payload));
}

export function trackPageReady(page: string, startedAt: number, payload: TelemetryPayload = {}) {
  trackEvent('page_ready', {
    page,
    duration_ms: Math.round(performance.now() - startedAt),
    ...payload
  });
}

export function installGlobalTelemetry(page: string) {
  if (window.__gvwTelemetryInstalled) return;
  window.__gvwTelemetryInstalled = true;

  const emitPageLoad = () => {
    const navigationEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined;
    trackEvent('page_load', {
      page,
      duration_ms: navigationEntry ? Math.round(navigationEntry.duration) : null,
      dom_content_loaded_ms: navigationEntry ? Math.round(navigationEntry.domContentLoadedEventEnd) : null,
      load_event_ms: navigationEntry ? Math.round(navigationEntry.loadEventEnd) : null
    });
  };

  if (document.readyState === 'complete') {
    emitPageLoad();
  } else {
    window.addEventListener('load', emitPageLoad, { once: true });
  }

  window.addEventListener('error', (event) => {
    trackEvent('browser_error', {
      page,
      message: event.message,
      source: event.filename || '',
      lineno: event.lineno,
      colno: event.colno
    });
  });

  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason instanceof Error ? event.reason.message : String(event.reason ?? 'unknown');
    trackEvent('browser_unhandled_rejection', {
      page,
      reason
    });
  });
}

export function installFetchTelemetry(options?: {
  context: string;
  augmentInit?: (url: string, init?: RequestInit) => RequestInit | undefined;
}) {
  const originalFetch = window.fetch.bind(window);

  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input);
    const requestInit = options?.augmentInit ? options.augmentInit(url, init) : init;
    const startedAt = performance.now();

    try {
      const response = await originalFetch(input, requestInit);
      if (response.status >= 500 || response.status === 408 || response.status === 504) {
        trackEvent('browser_fetch_failure', {
          context: options?.context,
          url,
          status: response.status,
          duration_ms: Math.round(performance.now() - startedAt)
        });
      }
      return response;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      trackEvent('browser_fetch_failure', {
        context: options?.context,
        url,
        status: null,
        duration_ms: Math.round(performance.now() - startedAt),
        error: message
      });
      throw error;
    }
  };
}
