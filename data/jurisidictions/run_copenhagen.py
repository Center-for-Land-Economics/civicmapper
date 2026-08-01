"""
Copenhagen (Denmark) parcel ETL — second non-US CivicMapper city.

Two-source city (like the US "geometry + separate value roll" pattern):

  GEOMETRY  — OPEN, no auth. DAWA / Dataforsyningen jordstykker (matrikel):
    https://api.dataforsyningen.dk/jordstykker?kommunekode=0101&format=geojson
    København == kommunekode 0101 → ~38,475 parcels, already EPSG:4326.
    Each parcel carries `bfenummer` (BFE — the join key to valuations),
    `registreretareal` (land m²), `vejareal` (road m²), matrikelnr+ejerlav,
    sfeejendomsnr, and sognenavn (parish → a natural region grouping).

  VALUE     — Datafordeler Ejendomsvurdering (VUR) GraphQL, keyed by BFE.
    grundværdi (land value) + ejendomsværdi (property value). Needs a free
    Datafordeler API key (Administration → IT-system → API-key) in data/.env as
    DATAFORDELER_API_KEY. Endpoint POST https://graphql.datafordeler.dk/VUR/v2.
    Each BFE has ~23 yearly valuations across two systems (old ESR, frozen ~2013;
    new Vurderingsstyrelsen 2020/2022). We take the NEWEST available per BFE
    (max assessment year) — ~95% coverage, overwhelmingly the 2020/2022 revaluation.

  LAND-USE  — VUR `benyttelseKode` (use code) → category; the refined
    Vacant/Parking/Underdeveloped field is derived from the improvement/land ratio
    (ejendomsværdi − grundværdi) plus the parking dataset. No BBR needed.

Currency = DKK ("kr."), units = metric (m²) — same frontend support Tallinn added.

Run (geometry base only, no credentials):
  PYTHONUTF8=1 python run_copenhagen.py

Run (full — pull live values from VUR GraphQL; ~38 min, caches the raw pull):
  PYTHONUTF8=1 python run_copenhagen.py --graphql

Run (full from a pre-fetched VUR extract keyed by bfenummer):
  PYTHONUTF8=1 python run_copenhagen.py --vur data/.../vur_values.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

DAWA_BASE = "https://api.dataforsyningen.dk/jordstykker"
SOURCE_CRS = "EPSG:4326"  # DAWA already serves WGS84

VUR_GRAPHQL = "https://graphql.datafordeler.dk/VUR/v2"  # endpoint path is v2 (verified)
BATCH = 100  # DafLong `in`-filter cap (MaxListSize 100)

SQM_TO_SQFT = 10.76391041671
SQM_TO_ACRES = 1.0 / 4046.8564224

# VUR benyttelseKode (ejendommens benyttelse) → English land-use category.
# Authoritative ESR/BENYTKOD list (Danmarks Statistik v1:1992).
BENYTTELSE_CATEGORY = {
    "00": "Exempt",          # Undtaget fra vurdering
    "01": "Residential",     # Beboelse
    "02": "Mixed Use",       # Beboelse og forretning
    "03": "Commercial",      # Forretning
    "04": "Industrial",      # Fabrik og lager
    "05": "Agricultural",    # Landbrug, bebygget
    "06": "Agricultural",    # Skov og plantage
    "07": "Agricultural",    # Frugtplantage, gartneri, planteskole
    "08": "Residential",     # Sommerhus
    "09": "Vacant",          # Ubebygget areal
    "10": "Institutional",   # Statsejendom
    "11": "Institutional",   # Kommunal beboelses- og forretningsejendom
    "12": "Institutional",   # Andre bebyggede kommunale ejendomme
    "13": "Other",           # Anden vurdering
    "14": "Other",           # Ejendom vurderet til 0
    "20": "Residential",     # Moderejendom for ejerlejligheder (apartment-building land)
    "33": "Institutional",   # Private institutions- og serviceejendomme
    "34": "Commercial",      # Visse erhvervsejendomme
    "45": "Other",           # Andre bygninger på fremmed grund
    "49": "Other",           # Arealer med bygning på fremmed grund
}


def fetch_jordstykker(kommunekode: str, cache_path: str) -> gpd.GeoDataFrame:
    """Fetch all matrikel parcels for a kommune from DAWA (open, no auth), cached to parquet.

    DAWA rejects deep pagination (`side` beyond ~2 pages → HTTP 400) but happily
    returns the whole kommune in one GeoJSON response (~50 MB for Copenhagen), so
    we stream a single unpaginated request to disk and read it. Attributes
    (bfenummer, registreretareal, …) ride along in each feature's `properties`.
    """
    if os.path.exists(cache_path):
        print(f"📂 Using cached geometry: {cache_path}")
        return gpd.read_parquet(cache_path)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    raw_geojson = cache_path.replace(".parquet", ".geojson")
    url = f"{DAWA_BASE}?kommunekode={kommunekode}&format=geojson"
    print(f"⬇️  DAWA (single request) → {raw_geojson}\n    {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "civicmapper-etl"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(raw_geojson, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)

    gdf = gpd.read_file(raw_geojson)
    if gdf.crs is None:
        gdf = gdf.set_crs(SOURCE_CRS)
    if len(gdf) == 0:
        raise RuntimeError(f"DAWA returned no jordstykker for kommunekode={kommunekode}.")
    print(f"✅ {len(gdf):,} parcels from DAWA  CRS={gdf.crs}")
    gdf.to_parquet(cache_path, index=False)
    print(f"💾 Cached geometry → {cache_path}")
    return gdf


def load_api_key() -> str:
    key = os.environ.get("DATAFORDELER_API_KEY")
    if key:
        return key
    # Walk up from cwd AND this file's dir looking for data/.env or .env (script is
    # run from data/jurisidictions/, but the key lives in the repo-root data/.env).
    seen = set()
    for start in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        d = start
        while d not in seen:
            seen.add(d)
            for cand in (os.path.join(d, "data", ".env"), os.path.join(d, ".env")):
                if os.path.exists(cand):
                    for line in open(cand, encoding="utf-8", errors="ignore"):
                        m = re.match(r'\s*DATAFORDELER_API_KEY\s*=\s*(.+)\s*$', line)
                        if m:
                            return m.group(1).strip().strip('"').strip("'")
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    raise RuntimeError("DATAFORDELER_API_KEY not found in env or any data/.env up the tree")


def _gql(query: str, key: str, tries: int = 6) -> dict:
    """POST a GraphQL query with retry/backoff on throttle (429 / cost errors)."""
    url = f"{VUR_GRAPHQL}?apiKey={key}"
    for t in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps({"query": query}).encode(),
                headers={"Content-Type": "application/json", "User-Agent": "civicmapper"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            if "errors" in d:
                raise RuntimeError(str(d["errors"])[:200])
            return d
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
            throttled = getattr(e, "code", None) == 429 or "429" in str(e) or "cost" in str(e).lower()
            if t == tries - 1:
                raise
            time.sleep((2 * (t + 1)) if throttled else 1)
    raise RuntimeError("unreachable")


def fetch_vur_values(bfes: list[int], key: str, cache_path: str) -> pd.DataFrame:
    """Pull the NEWEST-available valuation per BFE from the VUR GraphQL service.

    Two stages (both `in`-batched at 100, the DafLong filter cap):
      1. VUR_BFEKrydsreference: BFEnummer → fkEjendomsvurderingID (cursor-paginated;
         a BFE has ~23 yearly valuation ids so a 100-BFE batch exceeds the 1000/page cap).
      2. VUR_Ejendomsvurdering by id → aar, grundvaerdiBeloeb, ejendomvaerdiBeloeb,
         benyttelseKode, vurderetAreal.
    Newest per BFE = max `aar`; on a tie, max grundværdi (never sum — avoids
    double-counting a re-valued parcel). Returns one row per BFE:
    [bfenummer, grundvaerdi, ejendomsvaerdi, benyttelse_kode, vurderet_areal, aar].
    Caches the result so re-runs (bake tweaks) skip the ~38-min pull.
    """
    if os.path.exists(cache_path):
        print(f"📂 Using cached VUR values: {cache_path}")
        return pd.read_parquet(cache_path)

    def chunks(xs, n):
        for i in range(0, len(xs), n):
            yield xs[i:i + n]

    # Stage 1 — BFE → valuation ids
    bfe_to_ids: dict[int, set] = {}
    t0 = time.time()
    for bi, batch in enumerate(chunks(bfes, BATCH)):
        after = None
        while True:
            ac = f', after:"{after}"' if after else ""
            q = (f'{{ VUR_BFEKrydsreference(first:1000{ac}, where:{{BFEnummer:{{in:{json.dumps(batch)}}}}}) '
                 f'{{ nodes {{ BFEnummer fkEjendomsvurderingID }} pageInfo {{ hasNextPage endCursor }} }} }}')
            conn = _gql(q, key)["data"]["VUR_BFEKrydsreference"]
            for n in conn["nodes"]:
                bfe_to_ids.setdefault(n["BFEnummer"], set()).add(n["fkEjendomsvurderingID"])
            pi = conn["pageInfo"]
            if pi["hasNextPage"]:
                after = pi["endCursor"]
            else:
                break
        if bi % 25 == 0:
            print(f"  stage1 batch {bi}: {len(bfe_to_ids):,} BFEs, {time.time()-t0:.0f}s", flush=True)
    all_ids = sorted({i for s in bfe_to_ids.values() for i in s})
    print(f"stage1 done: {len(bfe_to_ids):,} BFEs w/ valuations, {len(all_ids):,} valuation ids, {time.time()-t0:.0f}s")

    # Stage 2 — valuation records
    val: dict[int, dict] = {}
    t1 = time.time()
    for ci, batch in enumerate(chunks(all_ids, BATCH)):
        q = (f'{{ VUR_Ejendomsvurdering(first:100, where:{{id:{{in:{json.dumps(batch)}}}}}) '
             f'{{ nodes {{ id aar grundvaerdiBeloeb ejendomvaerdiBeloeb benyttelseKode vurderetAreal }} }} }}')
        for n in _gql(q, key)["data"]["VUR_Ejendomsvurdering"]["nodes"]:
            val[n["id"]] = n
        if ci % 100 == 0:
            print(f"  stage2 batch {ci}/{len(all_ids)//BATCH}: {time.time()-t1:.0f}s", flush=True)
    print(f"stage2 done: {len(val):,} valuations, {time.time()-t1:.0f}s")

    # Newest per BFE
    rows = []
    for bfe, ids in bfe_to_ids.items():
        recs = [val[i] for i in ids if i in val]
        if not recs:
            continue
        myr = max(r["aar"] for r in recs)
        best = max((r for r in recs if r["aar"] == myr), key=lambda r: r["grundvaerdiBeloeb"])
        rows.append({"bfenummer": bfe, "grundvaerdi": best["grundvaerdiBeloeb"],
                     "ejendomsvaerdi": best["ejendomvaerdiBeloeb"],
                     "benyttelse_kode": best["benyttelseKode"],
                     "vurderet_areal": best["vurderetAreal"], "aar": myr})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"💾 Cached {len(df):,} per-BFE values → {cache_path}")
    return df


def allocate_values(gdf: gpd.GeoDataFrame, vur_by_bfe: pd.DataFrame) -> gpd.GeoDataFrame:
    """Attach per-BFE values to parcels, area-allocating across multi-polygon BFEs.

    THE BROADCAST TRAP (playbook §2 — 435 Copenhagen BFEs span up to 15 jordstykker):
    a value is assessed once per BFE covering ALL its polygons. A plain merge copies
    the full value onto every polygon → each renders at full value AND a citywide sum
    counts it N×. Fix: allocate the BFE value across its polygons in proportion to
    land area. Per-sqft is then uniform within a property and summing recovers the
    true per-BFE value. (Validated: 0-kr reconstruction error, uniform per-sqft.)
    """
    vur_by_bfe = vur_by_bfe.copy()
    vur_by_bfe.columns = [c.lower() for c in vur_by_bfe.columns]
    if "bfenummer" not in vur_by_bfe.columns:
        raise RuntimeError(f"VUR table needs a 'bfenummer' column; got {list(vur_by_bfe.columns)}")
    carry = [c for c in ("grundvaerdi", "ejendomsvaerdi", "vurderet_areal", "benyttelse_kode", "aar")
             if c in vur_by_bfe.columns]
    vur_by_bfe = vur_by_bfe.groupby("bfenummer", as_index=False)[carry].first()
    merged = gdf.merge(vur_by_bfe, on="bfenummer", how="left")

    # Allocate by GEOMETRY-area share (physical footprint). grundværdi, ejendomsværdi AND the
    # assessed area (vurderet_areal) are all per-BFE quantities → apportion each across the BFE's
    # polygons so summing rows recovers the true per-BFE total; codes/year are broadcast as-is.
    bfe_area = merged.groupby("bfenummer")["geom_area_sqm"].transform("sum")
    share = (merged["geom_area_sqm"] / bfe_area).where(bfe_area > 0, 0.0)
    for col in ("grundvaerdi", "ejendomsvaerdi", "vurderet_areal"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce") * share

    got = merged["grundvaerdi"].notna().sum() if "grundvaerdi" in merged else 0
    n_multi = int((gdf["bfenummer"].value_counts() > 1).sum())
    print(f"🔗 Values joined: {got:,}/{len(merged):,} parcels ({got/len(merged):.1%}); "
          f"area-allocated across {n_multi:,} multi-polygon BFEs.")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf.crs)


def join_valuations(gdf: gpd.GeoDataFrame, vur_path: str) -> gpd.GeoDataFrame:
    """File-based value join (parquet/csv keyed by bfenummer) → area-allocated."""
    ext = os.path.splitext(vur_path)[1].lower()
    vur = pd.read_parquet(vur_path) if ext == ".parquet" else pd.read_csv(vur_path)
    return allocate_values(gdf, vur)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kommunekode", default="0101", help="DAWA kommunekode (København = 0101).")
    ap.add_argument("--city-key", default="copenhagen")
    ap.add_argument("--state", default="hovedstaden", help="region slug (Region Hovedstaden).")
    ap.add_argument("--country", default="dk")
    ap.add_argument("--graphql", action="store_true",
                    help="Pull live values from the VUR GraphQL service (needs DATAFORDELER_API_KEY).")
    ap.add_argument("--vur", default=None,
                    help="Path to a pre-fetched VUR extract (parquet/csv keyed by bfenummer). "
                         "Omit both --graphql and --vur to build the geometry base only.")
    args = ap.parse_args()

    data_dir = f"data/{args.city_key}"
    os.makedirs(data_dir, exist_ok=True)
    slug = f"{args.city_key}-{args.state}-{args.country}"

    # ── 1. Geometry (open) ───────────────────────────────────────────────────────
    gdf = fetch_jordstykker(args.kommunekode,
                            os.path.join(data_dir, "cache", "copenhagen-geometry.parquet"))

    # Validity + drop empties.
    gdf["geometry"] = gdf["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    # ── 2. Geometry area (for RENDERING + parking spatial ops, NOT value metrics) ─
    # registreretareal is the cadastral polygon area; fall back to computed geometry area.
    # NOTE: this is the *drawn polygon's* area. It is deliberately NOT used as the land-value
    # denominator — see §3. For ~2,625 BFEs the DAWA polygon is a tiny registration fragment
    # (e.g. a 2 m² point) while the property's real assessed land (vurderetAreal) is thousands
    # of m²; dividing land value by the fragment area produced the pathological per-m² spikes.
    reg_area = pd.to_numeric(gdf.get("registreretareal"), errors="coerce")
    reg_area = reg_area.where(reg_area > 0, np.nan)
    geom_calc = gdf.to_crs("EPSG:25832").geometry.area  # ETRS89 / UTM 32N (DK)
    gdf["geom_area_sqm"] = reg_area.fillna(geom_calc)
    gdf["likely_remnant"] = (gdf["geom_area_sqm"] < 5).astype(int)  # soft flag; nothing dropped
    print(f"geom area: median {gdf['geom_area_sqm'].median():.0f} m²; "
          f"likely_remnant (<5 m²): {int(gdf['likely_remnant'].sum()):,}")

    # ── 3. Values (VUR grundværdi/ejendomsværdi, newest per BFE) ─────────────────
    have_values = bool(args.graphql or args.vur)
    if args.graphql:
        bfes = [int(x) for x in gdf["bfenummer"].dropna().unique()]
        vur_by_bfe = fetch_vur_values(bfes, load_api_key(),
                                      os.path.join(data_dir, "cache", "copenhagen-vur-values.parquet"))
        gdf = allocate_values(gdf, vur_by_bfe)
    elif args.vur:
        gdf = join_valuations(gdf, args.vur)

    if have_values:
        land_val = pd.to_numeric(gdf.get("grundvaerdi"), errors="coerce")
        prop_val = pd.to_numeric(gdf.get("ejendomsvaerdi"), errors="coerce")
        gdf["current_full_land_value"] = land_val            # → REALLANDVA
        gdf["full_market_value"] = prop_val
        gdf["improvement_value"] = (prop_val - land_val).clip(lower=0)

        # TWO areas, each used where it is correct:
        #  • land_area_sqm = GEOMETRY area → the physically-correct area for AGGREGATION (the
        #    hectare total; summing assessed area overcounts to ~2.2× the municipality because
        #    fragment properties carry their full off-map extent + road area).
        #  • per-m² = grundværdi ÷ ASSESSED area (vurderetAreal) → the correct RATE; using the
        #    drawn-polygon area caused the pathological spikes (a 234M-kr, 44,589 m² property on a
        #    2 m² fragment read 117M kr/m²). Both grund and vurderet_areal are area-allocated by
        #    the same geometry share, so per-m² = grund_BFE / vurderet_BFE (uniform, correct).
        #  The popup DERIVES displayed land size as value ÷ per-m² (main.ts) → recovers the
        #  truthful assessed area per parcel, so display stays honest while the total stays right.
        geom = gdf["geom_area_sqm"]
        gdf["land_area_sqm"] = geom
        gdf["land_area_sqft"] = geom * SQM_TO_SQFT
        gdf["land_area_acres"] = geom * SQM_TO_ACRES
        va = pd.to_numeric(gdf.get("vurderet_areal"), errors="coerce")
        has_va = va.notna() & (va > 0)
        psm = np.where(has_va, land_val / va.where(has_va, np.nan), np.nan)
        gdf["land_value_per_sqm"] = psm
        gdf["land_value_per_sqft"] = np.asarray(psm) / SQM_TO_SQFT

        # Category from VUR benyttelseKode; unmapped → "Other", missing value → "Unknown".
        code = gdf["benyttelse_kode"].astype("string") if "benyttelse_kode" in gdf else pd.Series(pd.NA, index=gdf.index, dtype="string")
        cat = code.map(BENYTTELSE_CATEGORY)
        cat = cat.mask(cat.isna() & code.notna(), "Other")
        cat = cat.mask(cat.isna(), "Unknown")
        gdf["property_land_use_category"] = cat

        # Refined underutilization. Vacant = authoritative code 09 (ubebygget areal).
        # Underdeveloped = improvement < 50% of total value, EXCLUDING: vacant land, exempt/
        # zero-valued records (00/14), and condo master parcels (20 — apartment-building land
        # whose units are valued separately, so their improvement ratio is misleading).
        # Parking is layered on later by the parking augment step.
        total = prop_val
        impr = gdf["improvement_value"]
        impr_share = impr / total.where(total > 0, np.nan)
        code_f = code.fillna("").astype(str)
        is_vacant = (code_f == "09")
        excluded = code_f.isin(["00", "09", "14", "20"])
        under = (land_val.notna() & ~excluded & (impr > 0) & (impr_share < 0.5)).fillna(False)
        refined = np.where(is_vacant, "Vacant", np.where(under, "Underdeveloped", None))
        gdf["property_land_use_refined"] = refined
    else:
        print("\n⚠️  No --graphql/--vur: building GEOMETRY BASE only (no land value / category).")
        print("    NOT app-loadable yet (frontend requires REALLANDVA). Re-run with --graphql.")
        gdf["property_land_use_category"] = "Unknown"
        # No assessed area available → land area = geometry area (display only).
        gdf["land_area_sqm"] = gdf["geom_area_sqm"]
        gdf["land_area_sqft"] = gdf["geom_area_sqm"] * SQM_TO_SQFT
        gdf["land_area_acres"] = gdf["geom_area_sqm"] * SQM_TO_ACRES

    gdf["exemption_flag"] = 0  # per non-US product decision: everything visible + filterable.

    # ── 4. Link to a public parcel record ────────────────────────────────────────
    gdf["link"] = ("https://api.dataforsyningen.dk/jordstykker/"
                   + gdf["ejerlavkode"].astype("Int64").astype(str) + "/"
                   + gdf["matrikelnr"].astype(str))

    # ── 5. Select + rename canonical columns ─────────────────────────────────────
    out = gdf.rename(columns={
        "bfenummer": "parcel_id",       # BFE is the stable property id
        "ejerlavnavn": "ejerlav",
        "sognenavn": "district",        # parish → region grouping (cf Tallinn linnaosa)
    })
    keep = [
        "geometry", "parcel_id", "matrikelnr", "ejerlav", "district",
        "sfeejendomsnr", "esrejendomsnr",
        "property_land_use_category", "property_land_use_refined",
        "exemption_flag", "likely_remnant",
        "current_full_land_value", "full_market_value", "improvement_value",
        "land_value_per_sqft", "land_value_per_sqm",
        "land_area_sqm", "land_area_sqft", "land_area_acres", "geom_area_sqm",
        "vejareal", "link",
    ]
    out = out[[c for c in keep if c in out.columns]].copy()
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")

    # ── 6. Save ──────────────────────────────────────────────────────────────────
    suffix = "" if have_values else "-geometry-base"
    canonical = os.path.join(data_dir, f"{slug}-parcels{suffix}.parquet")
    out.to_parquet(canonical, index=False)
    print(f"\n✅ Saved {len(out):,} parcels → {canonical}")
    print(f"   bounds: {out.total_bounds}")
    print(f"   districts (parishes): {out['district'].nunique()}")
    if have_values:
        print("\ncategory counts:")
        print(out["property_land_use_category"].value_counts(dropna=False).to_string())
        print("\nrefined counts:")
        print(out["property_land_use_refined"].value_counts(dropna=False).to_string())
        print("\nLand value (DKK) describe:")
        print(out["current_full_land_value"].describe().to_string())
        tot = out["current_full_land_value"].sum()
        print(f"\ncitywide grundværdi total: {tot:,.0f} kr ({tot/1e9:.1f} mia. kr)")


if __name__ == "__main__":
    main()
