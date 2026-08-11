/**
 * Dev-only live performance profiler + session recorder (stripped from production via the
 * import.meta.env.DEV gate at the call site).
 *
 * Workflow: browse the map, and whenever it feels laggy press **m** to mark the moment. When done,
 * press **p** (or call window.copyPerf()) to copy a compact report to the clipboard — paste it back
 * for analysis. The report aggregates by zoom band + layer and flags whether jank is CPU-bound
 * (high "blocked ms/s" from the Long Tasks API) or GPU-bound (low fps with no long tasks).
 *
 * Live HUD signals: fps / worst / avg frame, blocked ms/s (main-thread blocking), paints/s,
 * in-flight tiles, and feature density of the active layer.
 */
import type maplibregl from 'maplibre-gl';

interface PerfOpts {
  map: maplibregl.Map;
  /** Active layer (+ label) for the current zoom — used for feature counts and per-band grouping. */
  activeLayer: () => { id: string; label: string } | null;
  /** City label for the report header. */
  city?: string;
}

interface Snap {
  t: number; fps: number; worstMs: number; avgMs: number;
  blockedMsPerSec: number; longTasks: number; longWorstMs: number;
  paintsPerSec: number; tilesLoading: number; features: number; layer: string; zoom: number; pr: number;
}

interface ChokeEp {
  fromSec: number; durSec: number; worstFps: number; worstFrameMs: number; peakBlocked: number;
  zMin: number; zMax: number; layers: string; featsMax: number; tilesMax: number; windows: number;
}

const median = (xs: number[]) => {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};
const pct = (xs: number[], p: number) => {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.max(0, Math.min(s.length - 1, Math.floor((p / 100) * s.length)))];
};

export function initPerf({ map, activeLayer, city }: PerfOpts) {
  const el = document.createElement('div');
  el.id = 'perf-meter';
  el.style.cssText = 'position:fixed;bottom:8px;right:8px;z-index:9999;'
    + 'font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
    + 'background:rgba(0,0,0,.74);color:#cfd8dc;padding:5px 8px;border-radius:6px;'
    + 'pointer-events:none;white-space:pre;text-align:right;min-width:184px;'
    + 'border:1px solid transparent;transition:border-color .15s;';
  document.body.appendChild(el);

  // --- Long Tasks (main-thread blocks > 50ms): the clearest "scripting jank" signal ---
  let longTotal = 0, longCount = 0, longWorst = 0;
  let longTasksSupported = false;
  if (typeof PerformanceObserver !== 'undefined') {
    try {
      const obs = new PerformanceObserver(list => {
        for (const e of list.getEntries()) { longTotal += e.duration; longCount++; longWorst = Math.max(longWorst, e.duration); }
      });
      obs.observe({ entryTypes: ['longtask'] });
      longTasksSupported = true;
    } catch { /* not Chromium */ }
  }

  // --- Tile activity + repaint rate via map events ---
  let tilesLoading = 0, renders = 0;
  map.on('dataloading', (e: any) => { if (e.tile) tilesLoading++; });
  map.on('data', (e: any) => { if (e.tile) tilesLoading = Math.max(0, tilesLoading - 1); });
  map.on('render', () => { renders++; });
  map.on('idle', () => { tilesLoading = 0; });

  // --- Feature density of the active layer, sampled only when settled (query is expensive) ---
  let featNum = 0, featLbl = '';
  map.on('idle', () => {
    const a = activeLayer();
    featLbl = a?.label ?? '';
    try { featNum = (a && map.getLayer(a.id)) ? map.queryRenderedFeatures({ layers: [a.id] }).length : 0; }
    catch { featNum = 0; }
  });

  // --- Session recording ---
  const history: Snap[] = [];     // every 500ms window (cap ~20 min)
  const marks: Snap[] = [];       // moments the user flagged as laggy (press "m")
  const startedAt = performance.now();
  let latest: Snap | null = null;

  const flash = (color: string) => { el.style.borderColor = color; setTimeout(() => { el.style.borderColor = 'transparent'; }, 250); };

  // --- Auto choke detection: flag sustained lag episodes live (no manual marking needed) ---
  const FPS_BAD = 30, FRAME_BAD = 100, BLOCK_BAD = 200;   // a window is "bad" if any is breached
  const chokes: ChokeEp[] = [];
  let episode: Snap[] | null = null;
  let recover = 0;
  function closeEpisode() {
    const g = episode;
    episode = null;
    if (!g || !g.length) return;
    const ep: ChokeEp = {
      fromSec: +(g[0].t / 1000).toFixed(0),
      durSec: +(((g[g.length - 1].t - g[0].t) + 500) / 1000).toFixed(1),
      worstFps: Math.min(...g.map(s => s.fps)),
      worstFrameMs: +Math.max(...g.map(s => s.worstMs)).toFixed(0),
      peakBlocked: Math.max(...g.map(s => s.blockedMsPerSec)),
      zMin: Math.min(...g.map(s => s.zoom)), zMax: Math.max(...g.map(s => s.zoom)),
      layers: [...new Set(g.map(s => s.layer).filter(Boolean))].join('/') || '?',
      featsMax: Math.max(...g.map(s => s.features)),
      tilesMax: Math.max(...g.map(s => s.tilesLoading)),
      windows: g.length,
    };
    chokes.push(ep);
    console.warn(`[perf] CHOKE #${chokes.length}: ${ep.durSec}s · ${ep.worstFps}fps · worst-frame ${ep.worstFrameMs}ms · blocked ${ep.peakBlocked}ms/s · z${ep.zMin}-${ep.zMax} ${ep.layers} · ${ep.featsMax.toLocaleString()} feats · ${ep.tilesMax} tiles`);
    flash('#ff8f8f');
  }

  // --- Frame timing (rAF) + 500ms reporting window ---
  let frames = 0, worst = 0, sum = 0, prev = performance.now(), windowStart = prev;
  const loop = (now: number) => {
    const dt = now - prev; prev = now;
    // Discard tab-backgrounded / machine-asleep gaps (rAF pauses): a multi-second "frame" is not a
    // render stall. Reset the window so it doesn't register as a giant false choke.
    if (dt > 3000) {
      frames = 0; worst = 0; sum = 0; renders = 0; longTotal = 0; longCount = 0; longWorst = 0;
      windowStart = now;
      if (episode) closeEpisode();
      requestAnimationFrame(loop);
      return;
    }
    worst = Math.max(worst, dt); sum += dt; frames++;
    const span = now - windowStart;
    if (span >= 500) {
      const a = activeLayer();
      const snap: Snap = {
        t: +(now - startedAt).toFixed(0),
        fps: Math.round((frames * 1000) / span),
        worstMs: +worst.toFixed(1),
        avgMs: +(sum / frames).toFixed(1),
        // Clamp to 1000: it's ms-blocked per second, so it can't exceed wall time. Values above
        // come from the Long Tasks observer flushing a backlog into one short window after a stall.
        blockedMsPerSec: Math.min(1000, Math.round((longTotal * 1000) / span)),
        longTasks: longCount,
        longWorstMs: +longWorst.toFixed(1),
        paintsPerSec: Math.round((renders * 1000) / span),
        tilesLoading,
        features: featNum,
        layer: a?.label ?? featLbl,
        zoom: +map.getZoom().toFixed(2),
        pr: (map as any).getPixelRatio?.() ?? 0,
      };
      latest = snap;
      history.push(snap); if (history.length > 2400) history.shift();
      (window as any).__perf = snap;

      // Auto-flag: open/extend a choke episode on bad windows; close it after ~1s recovery.
      const bad = snap.fps < FPS_BAD || snap.worstMs > FRAME_BAD || (longTasksSupported && snap.blockedMsPerSec > BLOCK_BAD);
      if (bad) { (episode ||= []).push(snap); recover = 0; }
      else if (episode && ++recover >= 2) closeEpisode();

      const diag = !longTasksSupported ? ''
        : snap.blockedMsPerSec > 150 ? '  ⚠ CPU-bound'
        : (snap.fps < 40 && snap.blockedMsPerSec < 60) ? '  ⚠ GPU-bound'
        : '';
      el.textContent =
        `${snap.fps} fps · worst ${snap.worstMs.toFixed(0)} · avg ${snap.avgMs}ms\n`
        + (longTasksSupported ? `blocked ${snap.blockedMsPerSec}ms/s · ${snap.paintsPerSec} paints/s\n` : `${snap.paintsPerSec} paints/s\n`)
        + `${snap.tilesLoading} tiles · ${snap.features.toLocaleString()} ${snap.layer}\n`
        + `z ${snap.zoom} · ${chokes.length}⚡chokes · ${marks.length}m${diag}`;
      el.style.color = snap.fps >= 50 ? '#8fffa6' : snap.fps >= 30 ? '#ffd479' : '#ff8f8f';

      frames = 0; worst = 0; sum = 0; renders = 0; longTotal = 0; longCount = 0; longWorst = 0;
      windowStart = now;
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  // --- Report builder: aggregate by zoom band + layer, list worst windows + marks ---
  function buildReport(): string {
    const dur = ((performance.now() - startedAt) / 1000).toFixed(0);
    const all = history;
    const L = (n: number) => n.toLocaleString();
    const lines: string[] = [];
    lines.push(`=== PERF REPORT · ${city || 'city'} · ${dur}s · ${all.length} windows ===`);
    if (!all.length) { lines.push('(no data captured yet)'); return lines.join('\n'); }

    const fpsAll = all.map(s => s.fps);
    const blkAll = all.map(s => s.blockedMsPerSec);
    lines.push(`Frame:  fps median ${median(fpsAll)} · p5 ${pct(fpsAll, 5)} · worst-frame ${Math.max(...all.map(s => s.worstMs)).toFixed(0)}ms`);
    lines.push(`Main thread: blocked median ${median(blkAll)}ms/s · peak ${Math.max(...blkAll)}ms/s · worst long-task ${Math.max(...all.map(s => s.longWorstMs)).toFixed(0)}ms`);
    lines.push(`Paints: median ${median(all.map(s => s.paintsPerSec))}/s`);

    // group by "z<floor> <layer>"
    const groups = new Map<string, Snap[]>();
    for (const s of all) {
      const k = `z${Math.floor(s.zoom)} ${s.layer || '?'}`;
      (groups.get(k) || groups.set(k, []).get(k)!).push(s);
    }
    lines.push('By zoom band (median):');
    [...groups.entries()].sort((a, b) => median(a[1].map(s => s.fps)) - median(b[1].map(s => s.fps)))
      .forEach(([k, g]) => {
        lines.push(`  ${k.padEnd(12)} fps ${String(median(g.map(s => s.fps))).padStart(3)} · blocked ${String(median(g.map(s => s.blockedMsPerSec))).padStart(4)}ms/s · feats ${L(median(g.map(s => s.features)))} · tiles ${median(g.map(s => s.tilesLoading))} · n=${g.length}`);
      });

    lines.push(`Choke episodes (auto, ${chokes.length}) [fps<${FPS_BAD} or frame>${FRAME_BAD}ms or blocked>${BLOCK_BAD}ms/s]:`);
    if (!chokes.length) lines.push('  (none — smooth)');
    chokes.slice(-15).forEach((c, i) => lines.push(
      `  #${chokes.length - Math.min(15, chokes.length) + i + 1} @${c.fromSec}s · ${c.durSec}s · ${c.worstFps}fps · worst ${c.worstFrameMs}ms · blocked ${c.peakBlocked}ms/s · z${c.zMin}-${c.zMax} ${c.layers} · ${L(c.featsMax)} feats · ${c.tilesMax} tiles`));

    if (marks.length) {
      lines.push(`Marked laggy moments (${marks.length}):`);
      marks.forEach(s => lines.push(`  [mark] ${s.fps} fps · blocked ${s.blockedMsPerSec}ms/s · z${s.zoom} ${s.layer} · ${L(s.features)} feats · ${s.tilesLoading} tiles`));
    }
    lines.push('=== END ===');
    return lines.join('\n');
  }

  function copyReport() {
    if (episode) closeEpisode();   // fold in any in-progress lag so the report is complete
    const txt = buildReport();
    console.log(txt);
    try { navigator.clipboard?.writeText(txt).then(() => flash('#8fffa6'), () => {}); } catch { /* no gesture */ }
    flash('#8fffa6');
    return txt;
  }
  function markLag() {
    if (latest) { marks.push({ ...latest }); flash('#ff8f8f'); console.log(`[perf] marked laggy moment #${marks.length}:`, latest); }
  }

  // Hotkeys: "m" = mark laggy moment, "p" = copy report. Ignored while typing in a field.
  window.addEventListener('keydown', (e) => {
    const tag = (e.target as HTMLElement)?.tagName;
    if (e.metaKey || e.ctrlKey || e.altKey || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (e.key === 'm') markLag();
    else if (e.key === 'p') copyReport();
  });

  (window as any).copyPerf = copyReport;
  (window as any).markLag = markLag;
  (window as any).dumpPerf = () => { console.table(history); return history[history.length - 1]; };
  console.info('[perf] live HUD on. Choke episodes auto-log as you browse; press "p" (or window.copyPerf()) to copy the report. Press "m" to also hand-mark a moment.');
}
