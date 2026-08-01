#!/usr/bin/env python3
"""
Build Chicago's canonical parcel parquet from Cook County's open-data (Socrata) portal.

The City of Chicago has no parcel data of its own; assessment + geometry are Cook County's.
Cook County's authoritative GIS host (gis.cookcountyil.gov) is firewalled from this
environment, but the county mirrors everything we need on its Socrata portal
(datacatalog.cookcountyil.gov), which is reachable. Three datasets, joined on the 14-digit PIN:

- Parcel geometry: "ccgisdata - Parcel 2021" (77tz-riq7). Multipolygon `the_geom` + the PIN
  components (pina/pinsa/pinb/pinp/pinu) + `assessorbldgclass` + a `municipality` field.
  `municipality='Chicago'` is the authoritative Cook County jurisdiction assignment and
  isolates exactly the City of Chicago (612,202 parcels) server-side — no boundary clip
  needed. Paginated GeoJSON, cached to chicago-il-geometry.parquet (the multi-minute pull).
- Assessed values: "Assessor - Assessed Values" (uzyt-m557). certified/board/mailed
  land+building+total assessed value, per PIN per year. These are ASSESSED values, i.e.
  market * level-of-assessment; we divide by the LOA (below) to recover market value so
  Chicago's $/sqft is comparable to every other city in the app.
- Class is taken from the geometry's assessorbldgclass, falling back to the AV `class`.

Cook County level of assessment (per the classification ordinance):
  classes 1 (vacant), 2 (residential), 3 (apartments 7+), 6/7/8 (incentive), 9 = 10%;
  class 4 (not-for-profit) = 20%; class 5 (commercial 5-00..5-49 / industrial 5-50..5-99)
  = 25%. Exempt parcels carry a non-numeric class ('EX') and have no assessed value.

Chicago spans eight Cook County assessor townships, codes 70-77 (Hyde Park, Jefferson,
Lake, Lake View, North Chicago, Rogers Park, South Chicago, West Chicago) = the "City"
reassessment triad. We pull AV for those township codes only (keeps the value pull bounded)
and coalesce the most recent year with real values per PIN. The script reports the
geometry<->value PIN match rate so a wrong township set would be obvious.

Outputs:
- data/jurisidictions/data/chicago/chicago-il-parcels.parquet
- data/jurisidictions/data/chicago/chicago-il-parcels_YYYY_MM_DD.parquet

Chicago is large (~600k parcels) -> PMTiles. Bake + upload are separate steps:
    python data/scripts/parquet_to_pmtiles.py --city chicago --h3 --wsl --upload
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Geod
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.parcel_calculations import add_improvement_ratio_fields  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "chicago"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "chicago-il-geometry.parquet"
AV_CACHE = DATA_DIR / "chicago-il-assessed-values.parquet"

SOC = "https://datacatalog.cookcountyil.gov/resource"
PARCEL_DS = f"{SOC}/77tz-riq7"   # ccgisdata - Parcel 2021 (geometry)
AV_DS = f"{SOC}/uzyt-m557"       # Assessor - Assessed Values
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36"}

CHICAGO_TOWNSHIPS = ["70", "71", "72", "73", "74", "75", "76", "77"]
# Newest first; per PIN we keep the most recent year that actually carries a value.
# 2024 is the City-of-Chicago triennial reassessment (full revaluation); 2025 is the
# interim update. The in-progress current year is intentionally excluded (incomplete).
AV_YEARS = ["2025", "2024"]
PAGE = 25000
SQFT_PER_ACRE = 43560.0
geod = Geod(ellps="WGS84")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# Optional Socrata app token (env) lifts the anonymous throttle; works fine without one.
APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "").strip()


def soc_get(ds, params, *, fmt="json", tries=8, timeout=300):
    """Resilient Socrata GET. The portal throttles anonymous clients by request VOLUME:
    after a burst it accepts the connection but never responds (read timeout). Long
    backoff between retries lets the throttle window clear."""
    headers = dict(HEADERS)
    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN
    last = None
    for i in range(tries):
        try:
            r = requests.get(f"{ds}.{fmt}", params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"http {r.status_code}: {r.text[:200]}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)[:160]
        log(f"  retry {i+1}/{tries} ({last})")
        time.sleep(min(20 + 20 * i, 90))
    raise RuntimeError(f"Socrata GET failed after {tries} tries: {last}")


# ── 1. Chicago parcel geometry (cached) ───────────────────────────────────────
def fetch_geometry():
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    total = int(soc_get(PARCEL_DS, {"$select": "count(1)",
                "$where": "municipality='Chicago'"}).json()[0]["count_1"])
    log(f"Pulling {total:,} Chicago parcel polygons (paginated GeoJSON)...")
    pages, off = [], 0
    while off < total:
        r = soc_get(PARCEL_DS, {
            "$where": "municipality='Chicago'",
            "$select": ("pin10,pina,pinsa,pinb,pinp,pinu,assessorbldgclass,"
                        "assessornbhd,politicaltownship,tifdistrict,the_geom"),
            "$order": "pin10,pinu",   # stable order so $offset paging doesn't skip/dup
            "$limit": PAGE, "$offset": off,
        }, fmt="geojson")
        feats = r.json().get("features", [])
        if not feats:
            break
        gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
        if not len(gdf):
            break
        pages.append(gdf)
        off += len(gdf)
        log(f"  fetched {off:,}/{total:,}")
        if len(gdf) < PAGE:
            break
    geom = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    geom.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached geometry -> {GEOM_CACHE.name} ({len(geom):,} rows)")
    return geom


def pin14_from_parts(df):
    def pad(col, w):
        return (pd.to_numeric(df[col], errors="coerce").fillna(0)
                .astype("int64").astype(str).str.zfill(w))
    return pad("pina", 2) + pad("pinsa", 2) + pad("pinb", 3) + pad("pinp", 3) + pad("pinu", 4)


# ── 2. Assessed values for the Chicago townships (cached) ──────────────────────
AV_SEL = ("pin,year,class,township_code,certified_land,certified_bldg,certified_tot,"
          "board_land,board_bldg,board_tot,mailed_land,mailed_bldg,mailed_tot")
AV_LIMIT = 250000   # each township-year is ~110k rows; pull in ONE request, not paged


def fetch_values():
    if AV_CACHE.exists():
        log(f"Using cached assessed values: {AV_CACHE.name}")
        return pd.read_parquet(AV_CACHE)
    # Socrata throttles anonymous clients by request VOLUME, so we make as few requests as
    # possible: ONE request per (year, township) (~16 total) at a high $limit, with a polite
    # gap between them. Each chunk is cached so a stall only re-pulls the one missing chunk.
    chunk_dir = DATA_DIR / "av_chunks"
    chunk_dir.mkdir(exist_ok=True)
    for yr in AV_YEARS:
        for tc in CHICAGO_TOWNSHIPS:
            cf = chunk_dir / f"av_{yr}_{tc}.parquet"
            if cf.exists():
                log(f"  AV {yr} township {tc}: cached ({len(pd.read_parquet(cf)):,})")
                continue
            rows = soc_get(AV_DS, {"$where": f"township_code='{tc}' AND year='{yr}'",
                                   "$select": AV_SEL, "$order": "pin", "$limit": AV_LIMIT}).json()
            df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=AV_SEL.split(","))
            if len(df) >= AV_LIMIT:
                raise RuntimeError(f"AV {yr} tc{tc} hit the {AV_LIMIT} cap; raise AV_LIMIT.")
            df.to_parquet(cf, index=False)
            log(f"  AV {yr} township {tc}: {len(df):,} rows")
            time.sleep(8)   # stay under the anonymous throttle
    frames = [pd.read_parquet(f) for f in sorted(chunk_dir.glob("av_*.parquet"))]
    av = pd.concat(frames, ignore_index=True)
    av.to_parquet(AV_CACHE, index=False)
    log(f"  cached assessed values -> {AV_CACHE.name} ({len(av):,} rows)")
    return av


def coalesce_values(av):
    """Reduce the assessed-value rows to one per 14-digit PIN, then aggregate to the
    10-digit PIN (the geometry footprint level).

    Per PIN we take the most recent year (AV_YEARS order) that carries a real value,
    preferring certified -> board -> mailed within that year. We then group to pin10 and
    SUM the per-unit values: the parcel geometry has a single footprint per condo building
    (pinu=0), while assessed values live on the individual unit PINs (pinu=1001..), so the
    per-unit sum is the building's total value sitting on that one footprint. (This is a
    correct many-values->one-footprint collapse, the opposite of the Dallas broadcast bug.)
    """
    for stage in ["certified", "board", "mailed"]:
        for part in ["land", "bldg", "tot"]:
            av[f"{stage}_{part}"] = pd.to_numeric(av.get(f"{stage}_{part}"), errors="coerce")
    land, bldg, tot = av["certified_land"], av["certified_bldg"], av["certified_tot"]
    for stage in ["board", "mailed"]:   # fall back to board, then mailed, when total is 0/NaN
        use = ~(tot > 0)
        land = land.where(~use, av[f"{stage}_land"])
        bldg = bldg.where(~use, av[f"{stage}_bldg"])
        tot = tot.where(~use, av[f"{stage}_tot"])
    av = av.assign(av_land=land, av_bldg=bldg, av_tot=tot)
    av["year_rank"] = av["year"].map({y: i for i, y in enumerate(AV_YEARS)}).fillna(99)
    av = av[av["av_tot"] > 0].sort_values(["pin", "year_rank"]).drop_duplicates("pin", keep="first")
    # Convert assessed -> market value PER UNIT using that unit's OWN class level-of-assessment
    # BEFORE summing, so mixed-class buildings (e.g. ground-floor commercial @25% + condo units
    # @10%) are handled correctly. The per-unit class is the *current* assessor class and is the
    # authoritative source of truth — far more reliable than the stale 2021 GIS assessorbldgclass.
    loa = av["class"].map(level_of_assessment)
    av["mkt_land"] = pd.to_numeric(av["av_land"], errors="coerce") / loa
    av["mkt_bldg"] = pd.to_numeric(av["av_bldg"], errors="coerce") / loa
    av["mkt_tot"] = pd.to_numeric(av["av_tot"], errors="coerce") / loa
    av["pin10"] = av["pin"].str[:10]

    def modal_class(s):   # most common current class among the units (value-weighted tie-break)
        m = s.mode()
        return m.iloc[0] if len(m) else s.iloc[0]

    by10 = (av.groupby("pin10")
              .agg(mkt_land=("mkt_land", "sum"), mkt_bldg=("mkt_bldg", "sum"),
                   mkt_tot=("mkt_tot", "sum"), av_class=("class", modal_class),
                   av_units=("pin", "size"), av_year=("year", "max"))
              .reset_index())
    return by10


# ── 3. Classification (Cook County 3-digit class) ──────────────────────────────
def level_of_assessment(cls):
    """Fraction of market value that the assessed value represents, by major class."""
    c = str(cls or "").strip().upper()
    if not c or not c[0].isdigit():
        return np.nan        # 'EX' / blank -> exempt, no usable value
    d = c[0]
    if d in ("1", "2", "3", "6", "7", "8", "9"):
        return 0.10
    if d == "4":
        return 0.20
    if d == "5":
        return 0.25
    return np.nan


def categorize(cls):
    """Cook County major/minor class -> coarse property category."""
    c = str(cls or "").strip().upper()
    if not c or not c[0].isdigit():
        return "Exempt"
    d = c[0]
    try:
        n = int(c[:3]) if len(c) >= 3 and c[:3].isdigit() else int(d) * 100
    except ValueError:
        n = int(d) * 100
    if d == "1":
        return "Vacant Land"
    if d == "2":
        if n == 200:
            return "Vacant Land"
        if n == 201:
            return "Garage"
        if n in (295, 297, 298, 299) or 290 <= n <= 299:
            return "Condominium"
        if 202 <= n <= 209 or n in (234, 278):
            return "Single Family"
        if 210 <= n <= 225:
            return "Small Multifamily (2-6 units)"
        return "Residential"
    if d == "3":
        return "Multifamily (7+ units)"
    if d == "4":
        return "Not-for-Profit"
    if d == "5":
        return "Industrial" if 550 <= n <= 599 else "Commercial"
    if d in ("6", "7", "8"):
        return "Industrial" if d == "6" else "Commercial"   # development-incentive classes
    if d == "9":
        return "Multifamily (7+ units)"
    return "Other"


def categorize_refined(row):
    cat = str(row["property_land_use_category"])
    if "Vacant" in cat:
        return "Vacant"
    land = float(pd.to_numeric(row.get("current_full_land_value"), errors="coerce") or 0.0)
    impr = float(pd.to_numeric(row.get("improvement_value"), errors="coerce") or 0.0)
    total = land + impr
    if total <= 0:
        return None
    # Underdeveloped = improvement is a small share of total. Single Family needs to be
    # more clearly land-dominant (land >= 67%) before we flag it; everything else >= 50%.
    threshold = 0.33 if cat == "Single Family" else 0.50
    if impr < threshold * total:
        return "Underdeveloped"
    return None


# ── 4. Geometry helpers ────────────────────────────────────────────────────────
def geodesic_area_sqft(geom):
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type == "Polygon":
        lon, lat = geom.exterior.coords.xy
        area_m2, _ = geod.polygon_area_perimeter(lon, lat)
        hole = 0.0
        for ring in geom.interiors:
            lon_h, lat_h = ring.coords.xy
            a, _ = geod.polygon_area_perimeter(lon_h, lat_h)
            hole += abs(a)
        return max(abs(area_m2) - hole, 0.0) * 10.763910416709722
    if geom.geom_type == "MultiPolygon":
        return sum(geodesic_area_sqft(p) for p in geom.geoms)
    return np.nan


def main():
    geom = fetch_geometry()
    geom["pin"] = pin14_from_parts(geom)
    geom["pin10"] = geom["pin10"].astype(str).str.zfill(10)
    # One footprint per pin10 (keep the base parcel, pinu=0); the ~2k pinu>0 rows are
    # air-rights/leasehold duplicates that would otherwise broadcast a pin10 total twice.
    geom["pinu_n"] = pd.to_numeric(geom["pinu"], errors="coerce").fillna(0)
    geom = geom.sort_values("pinu_n").drop_duplicates("pin10", keep="first")
    log(f"Geometry footprints (unique pin10): {len(geom):,}")

    av10 = coalesce_values(fetch_values())
    log(f"Value rows aggregated to pin10: {len(av10):,}")

    parcel = geom.merge(av10, on="pin10", how="left")
    matched = int(parcel["mkt_tot"].notna().sum())
    log(f"Joined {len(parcel):,} | matched values {matched:,} "
        f"({100*matched/max(len(parcel),1):.1f}%)")

    # Class for category: the CURRENT assessor class from the value rows (authoritative),
    # falling back to the stale 2021 GIS assessorbldgclass only when there's no value match.
    # Using the GIS class as primary mis-prices/mis-labels reclassified parcels (e.g. a class
    # 591 industrial parcel still coded vacant in the 2021 GIS got a 10% LOA -> 2.5x inflation).
    parcel["state_class"] = (parcel["av_class"].astype(str).str.strip()
                             .replace({"": np.nan, "nan": np.nan, "None": np.nan})
                             .fillna(parcel["assessorbldgclass"].astype(str).str.strip()))
    parcel["state_class"] = parcel["state_class"].astype(str).str.strip().str.upper()

    # Market values are already computed per-unit-class and summed to pin10 in coalesce_values.
    parcel["land_value"] = pd.to_numeric(parcel["mkt_land"], errors="coerce")
    parcel["improvement_value"] = pd.to_numeric(parcel["mkt_bldg"], errors="coerce")
    parcel["full_market_value"] = pd.to_numeric(parcel["mkt_tot"], errors="coerce")

    parcel["PROPERTY_CATEGORY"] = parcel["state_class"].apply(categorize)

    # ── exempt + no-value drop ──────────────────────────────────────────────
    ex = parcel.copy()
    is_exempt_class = ex["state_class"].apply(lambda c: not str(c)[:1].isdigit())
    no_value = ~(pd.to_numeric(ex["full_market_value"], errors="coerce") > 0)
    ex["exemption_flag"] = (is_exempt_class | no_value).astype(int)
    before = len(ex)
    ex = ex[ex["exemption_flag"] == 0].copy()
    log(f"Removed {before - len(ex):,} exempt/no-value parcels -> {len(ex):,}")

    ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
    ex["current_full_land_value"] = ex["land_value"]
    ex["property_land_use_refined"] = ex.apply(categorize_refined, axis=1)

    # ── area + per-sqft ─────────────────────────────────────────────────────
    ex["geometry"] = ex["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
    log("Computing geodesic areas...")
    ex["area_sqft"] = ex["geometry"].apply(geodesic_area_sqft)
    ex.loc[ex["area_sqft"] < 1, "area_sqft"] = np.nan
    ex["land_area_acres"] = ex["area_sqft"] / SQFT_PER_ACRE
    ex["likely_remnant"] = (ex["area_sqft"] < 500).astype(int)

    den = ex["area_sqft"].replace(0, np.nan)
    ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
    ex["land_value_per_sqft"] = ex["land_value"] / den
    ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
    ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

    ex["link"] = ("https://www.cookcountyassessor.com/pin/" + ex["pin"].astype(str))

    # ── export ──────────────────────────────────────────────────────────────
    COLUMNS = ["geometry", "pin", "exemption_flag", "property_land_use_category",
               "property_land_use_refined", "full_market_value", "full_market_value_per_sqft",
               "land_value", "land_value_per_sqft", "improvement_value",
               "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO", "IMPR_LAND_PCT",
               "IMPR_PCT_TOTAL", "link", "land_area_acres", "likely_remnant"]
    for c in COLUMNS:
        if c not in ex.columns:
            ex[c] = np.nan
    final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
    final["geometry"] = final["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
    final = gpd.GeoDataFrame(final, geometry="geometry", crs=ex.crs)
    if final.crs is None or final.crs.to_epsg() != 4326:
        final = final.to_crs("EPSG:4326")

    out = DATA_DIR / "chicago-il-parcels.parquet"
    final.to_parquet(out, index=False)
    final.to_parquet(DATA_DIR / f"chicago-il-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet", index=False)
    log(f"SAVED {out} | rows {len(final):,}")
    log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
    log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
    lps = final["land_value_per_sqft"]
    log(f"land_value_per_sqft: p50=${lps.median():.0f} p99=${lps.quantile(.99):.0f} "
        f"p999=${lps.quantile(.999):.0f} max=${lps.max():.0f}")
    log("DONE")


if __name__ == "__main__":
    main()
