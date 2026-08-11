/**
 * Region show/hide + overlay outlines, with switchable grouping schemes (Houston).
 *
 * Parcels (and the dominant value per low-zoom hex) are tagged with several different ways to
 * partition the city — municipal jurisdiction, city council district, super neighborhood, civic
 * club. Each is a "group". The user picks ONE active group; the region list is always open.
 *
 * Each region row has two icon toggles:
 *   - eye    → VISIBILITY: whether the region renders in the value surface (parcels/hexes).
 *   - square → OVERLAY: whether the region's boundary outline is drawn on the map.
 * A header row above the list has a global eye and global square that toggle all regions at once.
 *
 * main ANDs `selectedClause()` (active field ∈ visible regions) into the parcel + hex layer
 * filters, so hidden regions don't render; and reads `getOverlayRegions()` to draw the boundary
 * overlays for the regions whose square is on. Switching group switches the whole context.
 *
 * Gated on the active city declaring `jurisdictionField` (or `jurisdictionGroups`) in cities.ts;
 * other cities unaffected.
 *
 * INSTANCING: the module exposes a `createJurisdiction()` factory so each map (the parcel map in
 * main.ts and the separate parking map in parking.ts) can own an independent instance — its own
 * state and its own onChange/onOverlaysChange callbacks. A default instance is created here and
 * its methods are re-exported as the historical named exports, so `import * as jurisdiction` in
 * main.ts keeps working unchanged.
 */
import type maplibregl from 'maplibre-gl';
import eyeOpenUrl from './svg/eye.svg';
import eyeClosedUrl from './svg/eye_closed.svg';

type Expr = any;

const NONE_VALUE = '(None)';

// Visibility uses the open/closed eye illustrations (self-describing — no tint).
const eyeImg = (visible: boolean) =>
  `<img class="jur-eye-img" src="${visible ? eyeOpenUrl : eyeClosedUrl}" width="18" height="18" alt="" draggable="false">`;

// Qualitative palette (Tableau/d3 category20) — assigned to regions BY ORDER so the list swatch
// and the map overlay agree. main reads regionColor()/regionColorIndex() so the two never drift.
export const OVERLAY_PALETTE = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
  '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
  '#f7b6d2', '#dbdb8d', '#9edae5', '#393b79',
];

// Per-row overlay square: a swatch in the region's overlay color (outline when off, filled when on)
// so it matches the boundary drawn on the map. The global square stays monochrome (currentColor).
const squareSvg = (color: string, on: boolean) =>
  `<svg viewBox="0 0 24 24" width="14" height="14"><rect x="3.5" y="3.5" width="17" height="17" rx="2.5" fill="${on ? color : 'none'}" stroke="${color}" stroke-width="2.4"/></svg>`;
const SQUARE_MONO = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4"><rect x="3.5" y="3.5" width="17" height="17" rx="2.5"/></svg>`;

/** One grouping scheme (e.g. municipal jurisdiction, or council district). */
export interface GroupConfig {
  field: string;                       // parcel/hex property name to filter on
  label: string;                       // human label shown in the group selector
  primary?: string;                    // region pinned to top + used by "primary only" default
  defaultMode?: 'all' | 'primaryOnly'; // initial visibility when this group is active
  counts?: Record<string, number>;     // region -> parcel count (PMTiles path)
  ids?: string[];                      // id-ordered names: tiles store integer id i, ids[i]=name.
                                       // Present only for int-encoded tiles → filter maps name->id.
  overlayUrl?: string;                 // GeoJSON of region boundaries for the overlay layer
}

interface GroupState {
  field: string;
  label: string;
  primary: string;                     // '' if the group has no primary
  defaultMode: 'all' | 'primaryOnly';
  regions: string[];                   // ordered: primary first, rest by count, "(None)" last
  counts: Record<string, number>;
  visible: Set<string>;                // eye on — renders in the value surface
  overlays: Set<string>;               // square on — boundary outline drawn
  overlayUrl?: string;
  nameToId: Map<string, number> | null; // non-null when tiles store integer ids for this field
}

interface State {
  active: boolean;
  groups: GroupState[];
  activeIdx: number;
}

const EMPTY_GROUP: GroupState = { field: '', label: '', primary: '', defaultMode: 'all', regions: [], counts: {}, visible: new Set(), overlays: new Set(), nameToId: null };

/** Where buildPanel inserts its card. Defaults match the land-value tab (main.ts). */
export interface PanelPlacement {
  containerSelector?: string;     // host element to insert the card into
  cardId?: string;                // unique id for the card (dedupe guard)
  insertBeforeSelector?: string;  // insert before the first match within the container, if any
  insertAfterSelector?: string;   // else insert after the first match within the container, if any
}

/** Order a group's regions: primary first, then descending count, with "(None)" pushed last. */
function orderRegions(counts: Record<string, number>, primary: string): string[] {
  const byCount = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  const head = primary && counts[primary] ? [primary] : [];
  const mid = byCount.filter(r => r !== primary && r !== NONE_VALUE);
  const tail = counts[NONE_VALUE] ? [NONE_VALUE] : [];
  return [...head, ...mid, ...tail];
}

function ensureStyle() {
  if (document.getElementById('jur-style')) return;
  const st = document.createElement('style');
  st.id = 'jur-style';
  st.textContent = `
    .jur-box { border:1px solid var(--border, #d8dee9); border-radius:8px; padding:6px 8px; }
    .jur-head { display:flex; align-items:center; gap:1px; padding-bottom:5px; margin-bottom:4px;
                border-bottom:1px solid var(--border, #d8dee9); }
    .jur-list { max-height:187px; overflow-y:auto; padding-right:6px; }
    .jur-row { display:flex; align-items:center; gap:1px; padding:1px 0; }
    .jur-row .jur-name { flex:1; margin-left:6px; font-size:var(--text-sm); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .jur-row .jur-count { font-size:var(--text-xs); }
    .jur-ico { display:inline-flex; align-items:center; justify-content:center; flex:none;
               width:20px; height:24px; padding:0; border:none; background:none; cursor:pointer;
               color:var(--text-muted, #90a4ae); border-radius:5px; transition:color .12s, background .12s; }
    .jur-ico:hover { background:rgba(127,127,127,.16); }
    .jur-ico.is-on { color:var(--accent-primary, #5b8def); }
    .jur-eye-img { width:18px; height:18px; display:block; pointer-events:none; }`;
  document.head.appendChild(st);
}

const setOn = (btn: HTMLElement | null, on: boolean) => {
  if (!btn) return;
  btn.classList.toggle('is-on', on);
  btn.setAttribute('aria-pressed', String(on));
};
// Visibility buttons swap the open/closed eye image (the icon conveys state, not a tint).
const updateEye = (btn: HTMLElement | null, visible: boolean) => {
  if (!btn) return;
  const img = btn.querySelector('img');
  if (img) img.setAttribute('src', visible ? eyeOpenUrl : eyeClosedUrl);
  btn.setAttribute('aria-pressed', String(visible));
};

function escapeHtml(s: string) { return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]!)); }

/** Create an independent region widget instance (own state + callbacks). */
export function createJurisdiction() {
  const S: State = { active: false, groups: [], activeIdx: 0 };
  let onChange: () => void = () => {};          // visibility changed → re-filter the map
  let onOverlaysChange: () => void = () => {};  // overlay set / labels changed → re-draw overlays
  let labelsOn = false;                         // "Overlay labels" checkbox (global to the panel)

  /** The active group (safe when inactive). */
  function ag(): GroupState { return S.groups[S.activeIdx] || EMPTY_GROUP; }

  /** Palette index for a region in the active group (by its position in the ordered list). */
  function regionColorIndex(region: string): number {
    const i = ag().regions.indexOf(region);
    return (i >= 0 ? i : 0) % OVERLAY_PALETTE.length;
  }
  function regionColor(region: string): string { return OVERLAY_PALETTE[regionColorIndex(region)]; }

  function isActive() { return S.active; }
  /** The set of regions currently VISIBLE in the active group (by name). */
  function getSelected(): Set<string> { return new Set(ag().visible); }
  /** A predicate testing whether a raw tile value for the active group's field is visible. Handles
   *  int-encoded tiles (maps the visible names to their ids). Used by the 3D-print "Selected only". */
  function makeVisibleTest(): (v: any) => boolean {
    const g = ag();
    if (g.nameToId) {
      const ids = new Set<number>();
      for (const n of g.visible) { const id = g.nameToId.get(n); if (id !== undefined) ids.add(id); }
      return (v) => ids.has(v as number);
    }
    const names = new Set(g.visible);
    return (v) => names.has(v);
  }
  /** Decode a raw tile value for a group `field` to its region name. Returns the name for
   *  int-encoded tiles (tiles store id i, ids[i]=name), or null when `field` isn't a group /
   *  isn't int-encoded / the id is unknown. Lets the popup show "West Hartford", not "0". */
  function nameForId(field: string, value: any): string | null {
    const g = S.groups.find(x => x.field === field);
    if (!g || !g.nameToId || typeof value !== 'number') return null;
    for (const [name, id] of g.nameToId) if (id === value) return name;
    return null;
  }
  /** The set of regions whose boundary OVERLAY is on (main draws these). */
  function getOverlayRegions(): Set<string> { return new Set(ag().overlays); }
  /** Whether overlay region-name labels should be drawn (only for regions whose overlay is on). */
  function labelsEnabled(): boolean { return labelsOn; }
  /** The parcel/hex property the active group filters on (3D-print region filter). */
  function getActiveField(): string { return ag().field; }
  /** The active group's human label (e.g. "City Council District"). */
  function getActiveLabel(): string { return ag().label; }
  /** Total number of regions in the active group (visible + hidden). */
  function getRegionCount(): number { return ag().regions.length; }
  /** The active group's region-boundary GeoJSON URL (for the overlay layer), or null. */
  function getActiveOverlayUrl(): string | null { return ag().overlayUrl || null; }

  /**
   * Called once the city + data are known. Supply `groups` (multi-group) with per-group `counts`
   * (PMTiles path) and/or `features` to scan (GeoParquet path). A single-field city may instead
   * pass the legacy `field`/`primary`/`counts`. Returns false if no group ends up with at least
   * two regions (nothing to toggle).
   */
  function configure(opts: {
    groups?: GroupConfig[];
    features?: GeoJSON.Feature[];
    onChange: () => void;
    onOverlaysChange?: () => void;                   // (re)draw the region boundary overlays
    // legacy single-group:
    field?: string; primary?: string; counts?: Record<string, number>;
  }): boolean {
    const defs: GroupConfig[] = opts.groups && opts.groups.length
      ? opts.groups
      : opts.field
        ? [{ field: opts.field, label: 'Region', primary: opts.primary,
             defaultMode: opts.primary ? 'primaryOnly' : 'all', counts: opts.counts }]
        : [];
    if (!defs.length) { S.active = false; return false; }

    const built: GroupState[] = [];
    for (const def of defs) {
      let counts: Record<string, number> = {};
      if (def.counts && Object.keys(def.counts).length) {
        counts = { ...def.counts };
      } else {
        for (const f of opts.features || []) {
          const v = (f.properties || {})[def.field];
          if (v == null || v === '') continue;
          counts[v] = (counts[v] || 0) + 1;
        }
      }
      // Need ≥2 real (non-"(None)") regions to be worth a group. A lone region plus a degenerate
      // "(None)" bucket — or a single region — is not a meaningful partition, so skip the group.
      if (Object.keys(counts).filter(k => k !== NONE_VALUE).length < 2) continue;

      const primary = def.primary && counts[def.primary] ? def.primary : '';
      const defaultMode: 'all' | 'primaryOnly' = def.defaultMode || (primary ? 'primaryOnly' : 'all');
      const regions = orderRegions(counts, primary);
      const visible = defaultMode === 'primaryOnly' && primary ? new Set([primary]) : new Set(regions);
      // Int-encoded tiles ship an id-ordered name list (ids[i] = name); build name->id so the layer
      // filter can use the integer ids the tiles actually store.
      const nameToId = def.ids && def.ids.length
        ? new Map(def.ids.map((n, i) => [n, i] as [string, number]))
        : null;
      built.push({ field: def.field, label: def.label, primary, defaultMode, regions, counts, visible, overlays: new Set(), overlayUrl: def.overlayUrl, nameToId });
    }

    if (!built.length) { S.active = false; return false; }
    S.active = true;
    S.groups = built;
    S.activeIdx = 0;
    onChange = opts.onChange;
    onOverlaysChange = opts.onOverlaysChange || (() => {});
    return true;
  }

  // ── filter clause ──────────────────────────────────────────────────────────────
  /** Matches the VISIBLE regions of the active group. main ANDs this into the parcel + hex layer
   *  filters; hidden regions are simply not rendered. When the tiles are int-encoded, map the
   *  selected names to their integer ids (which is what the tiles store). */
  function selectedClause(): Expr {
    const g = ag();
    const sel = Array.from(g.visible);
    const vals: any[] = g.nameToId
      ? sel.map(n => g.nameToId!.get(n)).filter(v => v !== undefined)
      : sel;
    return ['in', ['get', g.field], ['literal', vals]];
  }

  // ── control panel (group selector + always-open region list with per-row toggles) ──
  function buildPanel(_map: maplibregl.Map, placement?: PanelPlacement) {
    if (!S.active) return;
    const containerSelector = placement?.containerSelector ?? '#controls .land-sidebar-stack';
    const cardId = placement?.cardId ?? 'jurisdiction-card';
    const insertBeforeSelector = placement?.insertBeforeSelector
      ?? (placement ? undefined : '.parking-controls-card:not(#jurisdiction-card)');
    const stack = document.querySelector(containerSelector);
    if (!stack || document.getElementById(cardId)) return;
    ensureStyle();

    const groupSelect = S.groups.length > 1 ? `
        <div class="control-group">
          <select id="${cardId}-group" class="input select" style="width:100%;" aria-label="Region grouping">
            ${S.groups.map((g, i) => `<option value="${i}" ${i === S.activeIdx ? 'selected' : ''}>${escapeHtml(g.label)}</option>`).join('')}
          </select>
        </div>` : '';

    const card = document.createElement('div');
    card.className = 'parking-controls-card';
    card.id = cardId;
    card.innerHTML = `
      <div class="parking-controls-stack">
        ${groupSelect}
        <div class="control-group">
          <div class="jur-region-body jur-box"></div>
        </div>
      </div>`;
    // Placement: insert-before (land: above the Filter box) → insert-after (parking: below the
    // legend card) → append.
    const before = insertBeforeSelector ? stack.querySelector(insertBeforeSelector) : null;
    const after = placement?.insertAfterSelector ? stack.querySelector(placement.insertAfterSelector) : null;
    if (before) stack.insertBefore(card, before);
    else if (after && after.parentElement === stack) stack.insertBefore(card, after.nextSibling);
    else stack.appendChild(card);

    renderGroupBody(card);

    // Switching the group switches the whole region context (visibility + overlays).
    card.querySelector<HTMLSelectElement>(`#${cardId}-group`)?.addEventListener('change', (e) => {
      S.activeIdx = Number((e.target as HTMLSelectElement).value) || 0;
      renderGroupBody(card);
      onChange();
      onOverlaysChange();
    });
  }

  // Per-row overlay squares re-render their swatch (fill on/off) in the region's overlay color.
  const updateSquare = (btn: HTMLElement | null, region: string, on: boolean) => {
    if (!btn) return;
    btn.innerHTML = squareSvg(regionColor(region), on);
    btn.setAttribute('aria-pressed', String(on));
  };

  /** (Re)build the active group's region list: filter box, global toggles header, scrollable rows. */
  function renderGroupBody(card: HTMLElement) {
    const body = card.querySelector<HTMLElement>('.jur-region-body');
    if (!body) return;
    const g = ag();
    const allVisible = () => g.regions.length > 0 && g.visible.size === g.regions.length;
    const allOverlaid = () => g.regions.length > 0 && g.overlays.size === g.regions.length;

    // Filter box only worth it for long lists (e.g. 470 civic clubs).
    const filterInput = g.regions.length > 14
      ? `<input type="text" placeholder="Filter…" class="jur-filter input"
                style="width:100%; margin-bottom:6px; font-size:var(--text-sm);"/>`
      : '';
    const rows = g.regions.map(r => `
      <div class="jur-row" data-region="${escapeHtml(r)}">
        <button class="jur-ico jur-eye" data-region="${escapeHtml(r)}" type="button"
                title="Show/hide this region" aria-label="Show/hide this region: ${escapeHtml(r)}" aria-pressed="${g.visible.has(r)}">${eyeImg(g.visible.has(r))}</button>
        <button class="jur-ico jur-sq" data-region="${escapeHtml(r)}" type="button"
                title="Show/hide this overlay" aria-label="Show/hide this overlay: ${escapeHtml(r)}" aria-pressed="${g.overlays.has(r)}">${squareSvg(regionColor(r), g.overlays.has(r))}</button>
        <span class="jur-name">${escapeHtml(r)}</span>
        <span class="muted jur-count">${(g.counts[r] || 0).toLocaleString()}</span>
      </div>`).join('');

    body.innerHTML = `
      ${filterInput}
      <div class="jur-head">
        <button class="jur-eye-all jur-ico jur-eye" type="button"
                title="Show/hide all regions" aria-label="Show/hide all regions">${eyeImg(allVisible())}</button>
        <button class="jur-sq-all jur-ico ${allOverlaid() ? 'is-on' : ''}" type="button"
                title="Show/hide all overlays" aria-label="Show/hide all overlays">${SQUARE_MONO}</button>
      </div>
      <div class="jur-list">${rows}</div>
      <label class="checkbox-item" style="display:flex;gap:6px;align-items:center;margin-top:6px;padding-top:6px;border-top:1px solid var(--border, #d8dee9);">
        <input type="checkbox" class="jur-labels-cb checkbox" ${labelsOn ? 'checked' : ''}/>
        <span>Overlay labels</span>
      </label>`;

    const eyeAll = body.querySelector<HTMLButtonElement>('.jur-eye-all');
    const sqAll = body.querySelector<HTMLButtonElement>('.jur-sq-all');
    // The global eye is also a per-row eye visually (.jur-eye); exclude it from the row sweeps.
    const rowEyes = () => Array.from(body.querySelectorAll<HTMLButtonElement>('.jur-row .jur-eye'));
    const syncGlobals = () => { updateEye(eyeAll, allVisible()); setOn(sqAll, allOverlaid()); };

    // Per-row eye: toggle visibility for that region.
    rowEyes().forEach(btn => btn.addEventListener('click', () => {
      const r = btn.dataset.region as string;
      if (g.visible.has(r)) g.visible.delete(r); else g.visible.add(r);
      updateEye(btn, g.visible.has(r)); syncGlobals(); onChange();
    }));
    // Per-row square: toggle the boundary overlay for that region.
    body.querySelectorAll<HTMLButtonElement>('.jur-sq').forEach(btn => btn.addEventListener('click', () => {
      const r = btn.dataset.region as string;
      if (g.overlays.has(r)) g.overlays.delete(r); else g.overlays.add(r);
      updateSquare(btn, r, g.overlays.has(r)); syncGlobals(); onOverlaysChange();
    }));
    // Global eye: show all, or (when already all on) hide all.
    eyeAll?.addEventListener('click', () => {
      const turnOn = !allVisible();
      g.visible = turnOn ? new Set(g.regions) : new Set();
      rowEyes().forEach(b => updateEye(b, turnOn));
      syncGlobals(); onChange();
    });
    // Global square: outline all, or (when already all on) clear all.
    sqAll?.addEventListener('click', () => {
      const turnOn = !allOverlaid();
      g.overlays = turnOn ? new Set(g.regions) : new Set();
      body.querySelectorAll<HTMLButtonElement>('.jur-sq').forEach(b => updateSquare(b, b.dataset.region as string, turnOn));
      syncGlobals(); onOverlaysChange();
    });
    // "Overlay labels": draw region-name labels for whichever regions have their overlay on.
    body.querySelector<HTMLInputElement>('.jur-labels-cb')?.addEventListener('change', (e) => {
      labelsOn = (e.target as HTMLInputElement).checked;
      onOverlaysChange();
    });
    // Client-side filter: show/hide rows whose region name matches the query.
    body.querySelector<HTMLInputElement>('.jur-filter')?.addEventListener('input', (e) => {
      const q = (e.target as HTMLInputElement).value.trim().toLowerCase();
      body.querySelectorAll<HTMLElement>('.jur-row').forEach(row => {
        const name = (row.dataset.region || '').toLowerCase();
        row.style.display = !q || name.includes(q) ? '' : 'none';
      });
    });
  }

  return {
    configure, buildPanel, selectedClause, getSelected, getOverlayRegions, getActiveOverlayUrl,
    getActiveField, getActiveLabel, getRegionCount, labelsEnabled, isActive, makeVisibleTest,
    regionColor, regionColorIndex, nameForId,
  };
}

export type Jurisdiction = ReturnType<typeof createJurisdiction>;

// Default instance — the parcel map (main.ts) uses these named exports unchanged.
const _default = createJurisdiction();
export const configure = _default.configure;
export const buildPanel = _default.buildPanel;
export const selectedClause = _default.selectedClause;
export const getSelected = _default.getSelected;
export const getOverlayRegions = _default.getOverlayRegions;
export const getActiveOverlayUrl = _default.getActiveOverlayUrl;
export const getActiveField = _default.getActiveField;
export const getActiveLabel = _default.getActiveLabel;
export const getRegionCount = _default.getRegionCount;
export const labelsEnabled = _default.labelsEnabled;
export const isActive = _default.isActive;
export const makeVisibleTest = _default.makeVisibleTest;
export const regionColor = _default.regionColor;
export const regionColorIndex = _default.regionColorIndex;
export const nameForId = _default.nameForId;
