# Low-zoom PMTiles as H3 hexagons (and tippecanoe/WSL gotchas)

Large cities (e.g. Houston, 606k parcels) ship parcels as **PMTiles** instead of a
client-loaded GeoParquet. The problem at low zoom: tippecanoe coalesces the dense
parcel layer down to ~nothing, so zoomed out you got a flat green blob, and the
underutilized view went blank. The fix: pre-aggregate parcels into **H3 hexagons**
at several resolutions, render those when zoomed out, and **hand off to real
parcels** once zoomed in.

Pipeline: `python data/scripts/parquet_to_pmtiles.py --city <x> --h3 --wsl`
(`--wsl` on Windows; tippecanoe/pmtiles/tile-join live in WSL2 — see
`data/scripts/install_tippecanoe.py`).

## The layer design

| Zoom | Layer (`source-layer`) | Content |
|---|---|---|
| z0–6 | `parcels_low` | H3 **r7** hexes |
| z7–8 | `parcels_low` | r8 |
| z9 | `parcels_low` | r10 |
| z10–11 | `parcels_low` | r11 |
| z12 | `parcels_low` or `parcels` | r12 **only if it still aggregates** (see prune below); otherwise real parcels start here |
| z13–14 | `parcels` | **real parcels** (full detail) |

- `H3_RESOLUTIONS` / `H3_ZOOM_BANDS` in `parquet_to_pmtiles.py` define this. Each
  resolution is merged into ONE `parcels_low` layer; a per-feature
  `tippecanoe:{minzoom,maxzoom}` member gates each resolution to its zoom band.
- H3 steps **~7× per level** (no 2× option) — "finer" means the next integer
  resolution. Below r11 (~2,150 m²) hexes stop aggregating for most cities.
- **Per-city ladder + handoff (`plan_h3_ladder`, 2026-07-13):** any band whose hex is
  smaller than the city's **median parcel** is pruned and `parcelMinZoom` moves down to
  that band's zoom — hexes that subdivide single lots aren't aggregation, just false hex
  texture, and cost MORE features than the parcels they proxy (r12 was ~3–4× the parcel
  count). A 2026-07 sweep (`audit_hex_parcel_handoff.py`) found r12 finer than the median
  parcel in 23 of 26 PMTiles cities (typical lots 450–900 m² vs ~300 m² r12 hexes); only
  NYC/Baltimore rowhouse lots (~175–240 m²) genuinely benefit from r12. The count caps
  (`H3_R12_CAP_PARCELS`=600k drops r12, `H3_FINE_CAP_PARCELS`=1M drops r11) are separate:
  those stretch the coarsest survivor over the remaining hex zooms and KEEP `parcelMinZoom`
  at 13, because county-scale parcels at z12 are too heavy. The decided handoff ships as
  `parcelMinZoom` in the metadata JSON; the frontend prefers it over the dictionary value.

## Areal aggregation (`build_h3_aggregate`)

Each parcel is **intersected with the hex grid** (not assigned by centroid), so a
parcel bigger than a hex fills **all** the hexes it covers — no "lonely hex in an
empty spot." Per overlapping hex, weighted by intersection area:

- **rates** ($/sqft, ratios) → **area-weighted mean**
- **totals** ($) → **proportional area split** (`total × piece_area / parcel_area`)
- hexes with **no parcel overlap are not emitted** (water/ROW stay empty)
- the emitted geometry is the **full hexagon**; its value reflects its covered area

Hexes carry the **same field names** as parcels, so the viz's existing
color/extrusion expressions work on them unchanged.

## Baking (`build_pmtiles_h3_wsl_native`) — two runs + tile-join

Two separate tippecanoe runs, joined:

1. **hexes** z0–12: `--no-tile-size-limit --no-feature-limit` (small; never drop)
2. **parcels** z13–14 **only**: `--no-tile-size-limit --no-feature-limit` (full
   detail, nothing dropped). Tiling parcels *only* at the handoff zoom avoids
   generating huge low-zoom parcel tiles.
3. `tile-join -f -pk` merges them (`-pk` = don't re-impose the 500 KB tile cap).
4. `pmtiles convert`.

## ⚠️ Tippecanoe / WSL gotchas (each cost hours — read before re-baking)

1. **WSL `/mnt/c` 9p I/O is ~40× slower than native ext4.** tippecanoe does heavy
   random access on its input and its SQLite MBTiles output; over the Windows-mount
   bridge a Houston bake took **>1 hour vs ~90 s** native. **Fix:** stage inputs +
   scratch (`-t`) + MBTiles on WSL-native ext4 (`$(mktemp -d)`), copy only the final
   `.pmtiles` back to `/mnt/c`. Both `build_pmtiles_via_wsl_native` and
   `build_pmtiles_h3_wsl_native` do this.
2. **`--drop-densest-as-needed` DELETES features** to fit the tile-size cap — it kept
   only **~2% of parcels at z11**, 22% at z13. Never use it for parcels you want
   complete. Use `--no-tile-size-limit` and tile parcels only at the handoff zoom.
3. **`--coalesce-densest-as-needed` is pathologically slow** merging 600k parcels at
   low zoom (got stuck at z1 for an hour). Only the legacy non-H3 square-grid path
   still uses it; the H3 path avoids it entirely (hexes own low zoom).
4. **`PYTHONIOENCODING=utf-8`** — the script prints `✅`/`⚠️`; on Windows cp1252
   stdout (background runs) it crashes without this.
5. **Background runs lack `AZURE_STORAGE_CONNECTION_STRING`**, so `--upload` fails
   there. Bake without `--upload`, then upload via a `load_dotenv()` + azure-blob
   snippet.

## Viz integration (`viz/src/main.ts`, `viz/src/dictionaries/<city>.json`)

- `_config.parcelMinZoom` (Houston = 13) drives the handoff:
  `LAYER_ID_LOW` (hexes) gets `maxzoom = parcelMinZoom`; `LAYER_ID` (parcels) gets
  `minzoom = parcelMinZoom`. Since 2026-07-13 the bake writes the decided handoff into
  the metadata JSON (`metadata.parcelMinZoom`) and `loadPmtilesMetadata` overrides the
  dictionary value with it — tiles and handoff zoom deploy as one unit, so the dictionary's
  static 13 is only the fallback for pre-rule bakes.
- **No-gap transition:** for handoff cities the hexes hold **full opacity** to the
  z13 cut (the old `buildLowZoomOpacity` fade — needed by the legacy square-grid
  aggregate — faded hexes out early and left a dim gap before parcels appeared). A
  true cross-fade would require re-baking hexes into the z13 tiles (hexes live in
  z0–12 tiles, parcels in z13–14 — no shared-data zoom to dissolve across); the hard
  cut reads as clean.
- **Grey/flat-on-load bug + fix:** the source `data`/`sourcedata` handlers fire
  *before* `currentField`/`currentStats` are assigned (set later in the load), so the
  layer painted grey and set `extrusionsApplied = true`, locking out retries; only a
  manual zoom repainted it. **Fix:** a one-shot `map.once('idle', …)` on the main map
  that recomputes the auto-scale + repaints once the map settles.

## Cross-platform

The H3 bake runs on both platforms, via two implementations of the same command
sequence (`parquet_to_pmtiles.py` dispatches on `--wsl`):
- **Windows** (`--wsl`, `build_pmtiles_h3_wsl_native`): stages inputs + scratch +
  output on WSL-native ext4 and copies only the final PMTiles back to `/mnt/c`
  (avoids the slow `/mnt/c` 9p bridge — see gotcha #1).
- **macOS/Linux** (no `--wsl`, `build_pmtiles_h3_native`): runs tippecanoe /
  tile-join / pmtiles **directly** in a local temp dir — no staging needed since
  the filesystem is already native. Requires the binaries on PATH:
  `brew install tippecanoe pmtiles` (tippecanoe ships `tile-join`).

⚠️ **The native (Mac) path mirrors the proven WSL command sequence exactly but has
not been run on a Mac yet** — flag if the first Mac bake misbehaves. The non-H3
square-grid path (`create_mbtiles`) has always been cross-platform.

## Notes / tradeoffs

- Houston combined PMTiles ≈ **132 MB** as of 2026-06-17 (full Harris minus Unincorporated
  Harris County, r7–r11 hexes, per-sqft fields dropped — see Performance below). PMTiles serves
  tiles on demand via range requests, so it isn't downloaded whole, but the densest z13 downtown
  tiles still carry thousands of parcels (several MB each).
- To shrink further: drop the r11 band (hexes to r10). Parcels can't start before z13
  though — a z12 downtown tile would carry ~4× the parcels of a z13 one (too heavy).
- The non-H3 square-grid aggregate path (`build_low_zoom_aggregate`, used by smaller
  cities) is untouched and still uses coalesce.

## Performance — profiling + what actually moves the needle

Hard-won from a Houston (large-city) profiling pass (2026-06-17). **Profile before optimizing**:
`viz/src/perf.ts` is a live HUD — fps / worst-frame / **blocked ms/s** (Long Tasks API, Chromium
only) / tiles-loading / features-in-view, auto-flagged "choke" episodes, press **p** to copy a
by-zoom-band report. Use Chrome for the `blocked` number.

**The instrumentation is OFF by default** (so dev/prod aren't slowed by an always-on rAF loop +
observers + console spam) and **opt-in**: append **`?perf=1`** (or `localStorage.gvw_perf='1'`) to
show the HUD — dev-only, stripped from prod builds. Verbose console logging (the per-render paint
log + PMTiles load trace) is behind **`?debug=1`** (or `localStorage.gvw_debug='1'`), works in
dev+prod. Both gate to no-ops when off (see `PERF_HUD` / `VERBOSE` / `vlog` in `viz/src/main.ts`).

**Diagnosis that held up:** Houston is **CPU / main-thread bound during tile loading** (decode →
fill-extrusion bucket build → GPU upload), *not* GPU draw-bound. Steady-state (settled) is
60–120 fps; every choke coincides with tiles streaming in. So the levers are about main-thread
work and tile weight, not draw cost.

**Load path (biggest win):** the PMTiles apply loop must be **single-flight**. The original bug —
every tile's `data` event spawned its own `applyExtrusionsWhenReady` chain, each polling
`queryRenderedFeatures`/`querySourceFeatures` every 200ms — saturated the main thread
(`blocked 768ms/s` on load). Route external triggers through one guarded `requestApply()`, detach
the `data`/`sourcedata` listeners once applied, and gate verbose `[PMTiles]` logs behind a debug
flag (console spam with DevTools open is itself a real cost).

**Render:** render at **native `devicePixelRatio`** — supersampling (2–3×) tanks pan FPS. Do NOT
switch pixel ratio at runtime: `setPixelRatio()` reallocates the framebuffer (visible flash) and
its `resize()` interrupts the active pan. `fill-extrusion-vertical-gradient: false` and keeping
opacity at 100% (anything <1 forces depth-sorted alpha blending) are minor wins.

**Tile weight — the counterintuitive part:**
- **MVT already deduplicates attribute values into a per-tile pool**, so **integer-encoding
  repeated category STRINGS gives ≈0 size win** (measured: 0%). Don't bother for perf. (Houston
  *does* int-encode the region fields — `metadata.groups[field] = {ids, counts}`, frontend maps
  name↔id in `selectedClause` — but that was a no-op for size; keep it or not, it's neutral.)
- **Continuous NUMERIC fields are the real attribute weight** (they don't pool-dedup). Dropping
  the 3 derived per-sqft fields cut Houston **159 → 132 MB (~17%)**. Pattern: don't bake derived
  metrics — store the raw numerator(s) + a shared denominator and compute client-side. Per hex,
  carry the value sums + an apportioned **`land_area_acres`** sum (`H3_TOTAL_FIELDS`), drop the
  per-sqft rates from `H3_RATE_FIELDS` and from the parcel/under GeoJSON writes (keep them in the
  in-memory gdf so `compute_metadata` still derives stats/breaks). Frontend `perSqftExpr()` does
  `value / (land_area_acres·43560)` — one expression for both layers. Consequence: hex per-sqft
  becomes Σvalue/Σarea (was an area-weighted mean of rates) — arguably *more* correct; parcels
  stay bit-exact. The `IMPR_*` ratios were already computed client-side from REALIMPROV/REALLANDVA.
- **Hexes are the cheap proxy only while they aggregate** — don't lower `parcelMinZoom` past a band
  whose hexes are coarser than the parcels. A parcel is ~5–10× a hex's vertices, and a lower zoom
  shows ~4× the area, so parcels at z12 would cost ~10–40× the r11 hexes they'd replace. The
  inverse also holds and is why `plan_h3_ladder` prunes bands finer than the median parcel: r12
  hexes ran ~3–4× the parcel count, so for those cities parcels at z12 are the CHEAPER layer.

Net: the single-flight fix + native pixel ratio were the felt wins; per-sqft was the meaningful
tile trim. Beyond that it's diminishing returns (some load cost is inherent to a whole-county
parcel layer).
