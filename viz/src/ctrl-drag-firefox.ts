import type { Map as MaplibreMap } from 'maplibre-gl';

/**
 * Ctrl + drag (3D orbit) for Firefox.
 *
 * Firefox on macOS reports a Ctrl+click as a RIGHT-button event that still carries ctrlKey
 * (Mozilla bug 1504210), and can flip the reported button mid-drag if Ctrl is released.
 * MapLibre's rotate handler only accepts left+Ctrl or right-without-Ctrl, so in Firefox the
 * gesture never rotates. We take over Ctrl+mousedown before MapLibre sees it and drive
 * bearing/pitch ourselves, ending on any mouseup regardless of which button it reports.
 * Rates match MapLibre's defaults (0.8°/px bearing, -0.5°/px pitch).
 */
export function enableFirefoxCtrlDrag(map: MaplibreMap) {
  if (!/firefox/i.test(navigator.userAgent)) return;

  const container = map.getContainer();
  let last: { x: number; y: number } | null = null;

  const onMove = (e: MouseEvent) => {
    if (!last) return;
    e.preventDefault();
    e.stopPropagation();
    const dx = e.clientX - last.x;
    const dy = e.clientY - last.y;
    last = { x: e.clientX, y: e.clientY };
    if (!dx && !dy) return;
    // Like MapLibre's aroundCenter rotation: dragging above the map centre spins the other way.
    const rect = container.getBoundingClientRect();
    const aboveCenter = e.clientY - rect.top < rect.height / 2;
    map.jumpTo({
      bearing: map.getBearing() + dx * 0.8 * (aboveCenter ? -1 : 1),
      pitch: map.getPitch() - dy * 0.5,
    });
  };

  const onUp = () => {
    last = null;
    document.removeEventListener('mousemove', onMove, true);
    document.removeEventListener('mouseup', onUp, true);
  };

  // Capture on the outer container so this runs before MapLibre's canvas-container listeners.
  container.addEventListener('mousedown', (e) => {
    if (!e.ctrlKey || (e.button !== 0 && e.button !== 2)) return;
    if (!map.dragRotate.isEnabled()) return;
    e.preventDefault();
    e.stopPropagation();
    last = { x: e.clientX, y: e.clientY };
    document.addEventListener('mousemove', onMove, true);
    document.addEventListener('mouseup', onUp, true);
  }, true);

  // Ctrl+click is a context-menu click on macOS; keep the menu from popping over the map.
  container.addEventListener('contextmenu', (e) => {
    if (e.ctrlKey && map.dragRotate.isEnabled()) e.preventDefault();
  }, true);
}
