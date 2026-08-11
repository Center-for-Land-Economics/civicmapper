# Parking: surface lots vs. structures vs. gas stations

The parking layer is sourced from OpenStreetMap, where **everything** tagged
`amenity=parking` comes back as one undifferentiated "parking" polygon — surface
lots, multi-level decks/garages, podium parking, even gas-station forecourts. For
land-value analysis that conflation is wrong: a **surface lot is underutilized
land**, but a **parking structure is a developed improvement** (not underutilized),
and a **gas station isn't a parking lot at all**. This classifier separates them so
only real surface lots count toward the underutilized-parking totals.

Built/validated on Houston (2026-05). Output for Houston: **9,484 surface · 680
structure · 100 excluded (gas) · 7 uncertain**, and the headline underutilized
surface-parking land value dropped from a naive **$5.40B → $3.59B** (≈34% was
structures, gas stations, and building-covered slices that shouldn't count).

## Where it runs

- `data/scripts/classify_parking_surface.py` — the classifier (also a standalone
  CLI for ad-hoc runs / verification on a city's parking parquet).
- `data/scripts/parking_lot_extraction.py` — calls it as **Step 6b** after the
  parcel join. Gate with `--no-classify` to skip. A classifier failure never aborts
  the ETL (it falls back to writing parking without the classification columns).
- Per-feature columns added: `parking_type` (`surface|structure|uncertain|excluded`),
  `confidence` (`high|medium|low`), `classification_source`, `real_overlap`,
  `struct_overlap`, `canopy_overlap`, `near_fuel`, `surface_area_sqft`,
  `effective_surface_land_value`. Metadata gets a `classificationTotals` block.

## Inputs / signals (waterfall, most authoritative first)

1. **OSM tags** (free — the ETL used to *fetch then discard* them). `parking=multi-storey/underground/rooftop`, `building=parking/garage`, or `building:levels/parking:levels ≥ 2` → **structure**.
2. **Overture building `class`** (pulled via DuckDB; release pinned in `OVERTURE_RELEASE`). `class ∈ {parking, garage}` covering the lot → **structure**. `class ∈ {roof, carport, canopy}` → a **canopy**, i.e. NOT a real building (parking continues underneath — never carved, never counted as structure).
3. **Real-building footprint overlap** (the workhorse for untagged lots): fraction of the parking polygon covered by *non-canopy* Overture buildings.
4. **OSM `amenity=fuel`** proximity → gas station → **excluded**.
5. **Assessor improvement value** — *tried and rejected*: too noisy to be a tiebreaker (see below).

## Thresholds (in `classify_parking_surface.py`, tune there)

| Constant | Value | Meaning |
|---|---|---|
| `OVERLAP_STRUCTURE` | 0.85 | real-building coverage ≥ this → structure |
| `OVERLAP_UNCERTAIN` | 0.40 | coverage in [0.40, 0.85) on a deck-sized lot → uncertain |
| `OVERLAP_SURFACE` | 0.15 | ≤ this → surface (high confidence) |
| `STRUCTURE_CLASS_OVERLAP` | 0.30 | Overture parking-class coverage ≥ this → structure |
| `FUEL_NEAR_M` | 25 | within this of an `amenity=fuel` feature… |
| `FORECOURT_MAX_SQFT` | 20,000 | …AND no larger than this → excluded (gas) |
| `DECK_MAX_SQFT` | 80,000 | partial overlap on a polygon larger than this → surface (too big to be a deck) |

## Value handling (confirmed product decisions)

- **Rates** (`land_value_per_sqft`, ratios): each hex/lot shows the **area-weighted
  mean** of overlapping parcels' rates.
- **Totals** ($): **proportional area split** — a hex/slice covering 20% of a parcel
  gets 20% of its value.
- **Carve-out:** `surface_area_sqft = parking_area_sqft × (1 − real_overlap)` and
  `effective_surface_land_value = land_value_per_sqft × surface_area_sqft`. Only
  **real buildings** are carved; **canopies are not** (the parking exists under
  them). Structures/excluded contribute **0** surface value.
- Only confident `surface` feeds the headline total; `uncertain` is reported
  separately; `structure`/`excluded` contribute nothing.

## Lessons from the Google-Earth audit (why it's built this way)

- **`amenity=parking` pulls in everything**, incl. decks — and the ETL was
  **dropping the OSM subtags** right after fetching. Retaining them is a free first
  pass.
- **Height does NOT separate a canopy from a deck.** A gas-station canopy is ~7 m —
  taller than many one-story buildings. **Overlap *fraction* is the signal**: a real
  deck covers ~100% of its footprint; a surface lot with stuff on it covers part.
- **Overture `class` is gold:** it labels parking structures (`parking`) directly and
  canopies (`roof`/`carport`) — resolving most hard cases that footprint+height
  can't.
- **Assessor improvement value fails as a tiebreaker** — two real decks landed at
  *opposite extremes* (downtown deck looks "land-heavy" because land is so valuable;
  a cheap-land deck looks "building-heavy"). Confounded by land-value spread, shared
  parcels, and valuation quirks. Don't use it.
- **Gas exclusion must be size-gated.** Proximity-only swept in big lots merely near
  a station (a 57k-sqft lot, a Chick-fil-A). Real forecourts are small (median ~6.3k
  sqft) → only exclude `near_fuel AND area ≤ FORECOURT_MAX_SQFT`.
- **Size disambiguates the middle band.** A 12-acre lot with a building in the middle
  is a surface lot around a building, never a deck → partial overlap on a polygon >
  `DECK_MAX_SQFT` is surface (carved). Only *deck-sized* partial-overlap lots stay
  `uncertain`.
- **The `uncertain` band is irreducible** (~0.5% of lots): a small untagged lot ~50%
  covered by an untagged building is genuinely indistinguishable from a deck with a
  geometry mismatch. Resolving it would need LiDAR/parking-surface heights. Flag it
  honestly rather than guess.

## Viz integration

- Click popup (`viz/src/parking.ts`) shows **type · confidence · source** + the
  pro-rated surface value (structures/gas read "none (developed)").
- Category dropdown in the hero (`viz/app.html` + `parking.ts`): **All parking /
  Surface parking (default) / Parking structures** — recomputes every on-screen
  number client-side from the features and filters the map. Gas stations are never
  shown; `uncertain` rolls into "All" only.
- **Attribution:** the parking source carries
  `© OpenStreetMap contributors · Building footprints © Overture Maps Foundation (ODbL)`.

## Licensing

Overture's **Buildings theme is ODbL** — the *same* license as the OSM parking
polygons already in use, so it introduces **no new license category**. Obligations:
**attribution** (the string above) and **share-alike on the database** if the
underlying data is publicly distributed. Rendering it on the map is a "Produced
Work" → **attribution only**. (Source: docs.overturemaps.org/attribution.)

## Dependencies

`duckdb` (Overture query) and `osmnx` in `data/requirements.txt`. Overture buildings
are read straight from the public S3 GeoParquet via DuckDB `httpfs`+`spatial` — no
download step.
