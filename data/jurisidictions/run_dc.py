#!/usr/bin/env python3
"""
Build Washington, DC's canonical parcel parquet (standalone `washington` city).

Sources (all Open Data DC, CC0 public domain; endpoints verified live 2026-07-14):
- Geometry + conflated tax roll (one-stop): "Common Ownership Lots" (Owner Polygons),
  ArcGIS FeatureServer layer 40. Carries NEWLAND / NEWIMPR / NEWTOTAL (assessed
  land / building / total), PROPTYPE, USECODE, LANDAREA, OWNERNAME, MIX1TXTYPE,
  CONDO_REGIME_NUM etc., conflated weekly from ITS by SSL. 137k lots, ~96% valued.
    https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Property_and_Land_WebMercator/FeatureServer/40
- ITSPE (Integrated Tax System Public Extract): per-SSL tax roll TABLE, needed only
  for condo UNIT rows (units are separate 2xxx-series SSLs with their own allocated
  NEWLAND/NEWIMPR; the underlying footprint lot in layer 40 carries NULL values).
  Hosted view name rotates (e.g. OCFO_ITSPE_view_05212026) -> resolved via AGOL
  search at runtime, hardcoded fallback.
- CONDORELATE table (layer 52 on the same service): one row PER CONDO UNIT with
  SQUARE/SUFFIX/REGIME + MAT_SSL (the underlying record lot) -> deterministic
  unit -> footprint mapping. No spatial condo heuristics needed.

Condo rollup: footprints flagged UNDERLIES_CONDO/CONDOLOT carry no values; we sum
their units' NEWLAND/NEWIMPR/NEWTOTAL from ITSPE onto the footprint (join on
(SQUARE,SUFFIX,REGIME) primary, MAT_SSL fallback), category from the dominant unit
PROPTYPE. Exempt units are excluded from the sums, so a fully-exempt building
drops out via the no-value filter like any other exempt lot.

Exemption: MIX1TXTYPE codes (US=federal, DC=district, E0-E9=exempt classes,
CE=common element) + government owner keywords + Embassy PROPTYPE. Federal
reservations / ROW mostly carry no value and fall out of the value filter anyway.

No city-limits clipping: the dataset IS the jurisdiction.

Outputs:
- data/jurisidictions/data/washington/washington-dc-parcels.parquet (+ dated snapshot)

Upload + PMTiles are separate steps (137k parcels -> PMTiles):
    python data/scripts/parquet_to_pmtiles.py --city washington --wsl --upload
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.ops import unary_union
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "washington"
RAW_DIR = DATA_DIR / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
}
SQFT_PER_ACRE = 43560.0
SQM_TO_SQFT = 10.763910416709722
GEOD = Geod(ellps="WGS84")

SVC = ("https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
       "Property_and_Land_WebMercator/FeatureServer")
COL_URL = f"{SVC}/40/query"          # Common Ownership Lots (Owner Polygons)
CONDORELATE_URL = f"{SVC}/52/query"  # one row per condo unit SSL -> regime + MAT_SSL

# The ITSPE hosted view name rotates with refreshes; resolve current via AGOL search.
ITSPE_FALLBACK = ("https://services.arcgis.com/neT9SoYxizqTHZPH/arcgis/rest/services/"
                  "OCFO_ITSPE_view_05212026/FeatureServer/53")

COL_FIELDS = ("SSL,SQUARE,SUFFIX,PROPTYPE,USECODE,LANDAREA,OWNERNAME,PREMISEADD,"
              "NEWLAND,NEWIMPR,NEWTOTAL,MIX1TXTYPE,CLASSTYPE,LOT_TYPE,"
              "UNDERLIES_CONDO,CONDOLOT,CONDO_REGIME_NUM")
ITSPE_FIELDS = "SSL,SQUARE,SUFFIX,LOT,PROPTYPE,NEWLAND,NEWIMPR,NEWTOTAL,MIX1TXTYPE,OWNERNAME"
CONDORELATE_FIELDS = "SSL,SQUARE,SUFFIX,LOT,REGIME,MAT_SSL,NAME"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def resolve_itspe_url() -> str:
    """The ITSPE hosted feature-service view gets republished under a dated name.
    Find the current one via the AGOL search API; fall back to the known URL."""
    try:
        r = requests.get("https://www.arcgis.com/sharing/rest/search", params={
            "q": '"Integrated Tax System Public Extract" type:"Feature Service"',
            "f": "json", "num": 20}, headers=HEADERS, timeout=30)
        for it in r.json().get("results", []):
            if it.get("title") == "Integrated Tax System Public Extract" and it.get("url"):
                url = it["url"].rstrip("/")
                # the ITSPE lives as TABLE id 53 on the service (root lists it under
                # `tables`, not `layers`)
                j = requests.get(url, params={"f": "json"}, headers=HEADERS, timeout=30).json()
                tabs = j.get("tables", []) or []
                if tabs:
                    return f"{url}/{tabs[0]['id']}"
    except Exception as e:  # noqa: BLE001
        log(f"  ITSPE AGOL resolve failed ({type(e).__name__}); using fallback URL")
    return ITSPE_FALLBACK


# ─────────────────────────────────────────────────────────────────────────────
# Fetchers (paginated; geometry via f=geojson, tables via f=json attributes)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_geo_page(query_url: str, out_fields: str, off: int, count: int,
                    timeout: int, retries: int = 5) -> gpd.GeoDataFrame | None:
    for attempt in range(retries):
        try:
            r = requests.get(query_url, params={
                "where": "1=1", "outFields": out_fields, "returnGeometry": "true",
                "resultOffset": off, "resultRecordCount": count, "outSR": 4326,
                "orderByFields": "OBJECTID", "f": "geojson",
            }, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return gpd.read_file(io.BytesIO(r.content))
        except Exception as e:  # noqa: BLE001
            log(f"      retry {attempt+1} @off {off}: {type(e).__name__}: {str(e)[:90]}")
            time.sleep(3 * (attempt + 1))
    return None


def fetch_geometry(query_url: str, out_fields: str, page: int = 2000,
                   timeout: int = 240) -> gpd.GeoDataFrame:
    total = requests.get(query_url, params={"where": "1=1", "returnCountOnly": "true",
                                            "f": "json"},
                         headers=HEADERS, timeout=120).json().get("count")
    log(f"    total={total:,}")
    pages, off = [], 0
    while True:
        gdf = _fetch_geo_page(query_url, out_fields, off, page, timeout)
        if gdf is None:
            raise RuntimeError(f"geometry fetch failed at offset {off}")
        n = len(gdf)
        if n == 0:
            break
        pages.append(gdf)
        off += n
        if off % (page * 10) < page:
            log(f"      fetched {off:,}/{total:,}")
        if n < page:
            break
    g = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    log(f"    pulled {len(g):,} features")
    return g


def fetch_table(query_url: str, out_fields: str, page: int = 2000,
                timeout: int = 120) -> pd.DataFrame:
    total = requests.get(query_url, params={"where": "1=1", "returnCountOnly": "true",
                                            "f": "json"},
                         headers=HEADERS, timeout=120).json().get("count")
    log(f"    total={total:,}")
    rows, off = [], 0
    while True:
        got = None
        for attempt in range(5):
            try:
                r = requests.get(query_url, params={
                    "where": "1=1", "outFields": out_fields, "returnGeometry": "false",
                    "resultOffset": off, "resultRecordCount": page,
                    "orderByFields": "OBJECTID", "f": "json",
                }, headers=HEADERS, timeout=timeout)
                r.raise_for_status()
                j = r.json()
                if "error" in j:
                    raise RuntimeError(str(j["error"])[:120])
                got = [f["attributes"] for f in j.get("features", [])]
                break
            except Exception as e:  # noqa: BLE001
                log(f"      retry {attempt+1} @off {off}: {type(e).__name__}: {str(e)[:90]}")
                time.sleep(3 * (attempt + 1))
        if got is None:
            raise RuntimeError(f"table fetch failed at offset {off}")
        if not got:
            break
        rows.extend(got)
        off += len(got)
        if off % (page * 20) < page:
            log(f"      fetched {off:,}/{total:,}")
        if len(got) < page:
            break
    df = pd.DataFrame(rows)
    log(f"    pulled {len(df):,} rows")
    return df


def cached(key: str, fetch_fn, force: bool, geo: bool):
    raw = RAW_DIR / f"{key}.parquet"
    if raw.exists() and not force:
        d = gpd.read_parquet(raw) if geo else pd.read_parquet(raw)
        log(f"  [{key}] cached raw: {len(d):,} rows")
        return d
    log(f"  [{key}] fetching fresh...")
    d = fetch_fn()
    d.to_parquet(raw, index=False)
    log(f"  [{key}] cached -> {raw.name} ({len(d):,} rows)")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def norm_ssl(s: pd.Series) -> pd.Series:
    """SSLs are 'SQUARE(4) SUFFIX(4-as-spaces) LOT(4)' with literal space padding that
    is inconsistent across tables — collapse runs of whitespace for a robust join key."""
    return s.fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def norm_code(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def geodesic_area_sqft(geom) -> float:
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type == "Polygon":
        lon, lat = geom.exterior.coords.xy
        a, _ = GEOD.polygon_area_perimeter(lon, lat)
        return abs(a) * SQM_TO_SQFT
    if geom.geom_type == "MultiPolygon":
        return sum(geodesic_area_sqft(p) for p in geom.geoms)
    return np.nan


EXEMPT_TXTYPES = {"US", "DC", "CE"} | {f"E{i}" for i in range(10)}
EXEMPT_OWNER_RE = re.compile(
    "|".join([
        r"^UNITED STATES", r"UNITED STATES OF AMERICA", r"^US GOVT", r"^U S GOVT",
        r"^USA\b", r"GOVT OF THE USA", r"GOVT OF THE UNITED STATES",
        r"^DISTRICT OF COLUMBIA", r"^GOVERNMENT OF THE DISTRICT",
        r"WASHINGTON METROPOLITAN AREA TRANSIT", r"\bWMATA\b",
        r"NATIONAL PARK SERVICE", r"GENERAL SERVICES ADMIN", r"SMITHSONIAN",
        r"^DC HOUSING AUTH", r"HOUSING AUTHORITY",
    ]))


def is_exempt(txtype: pd.Series, owner: pd.Series, proptype: pd.Series) -> pd.Series:
    tx = norm_code(txtype).str.upper().isin(EXEMPT_TXTYPES)
    own = norm_code(owner).str.upper().str.contains(EXEMPT_OWNER_RE, na=False)
    emb = norm_code(proptype).str.upper().str.startswith("EMBASSY")
    return tx | own | emb


def dc_categorize(proptype) -> str:
    """OTR PROPTYPE (human-readable use description, truncated ~30 chars) -> coarse
    category. Mapping covers all 81 distinct live values (probed 2026-07-14)."""
    p = str(proptype or "").strip().upper()
    if not p or p == "NAN" or p == "NONE":
        return "Other"
    if p.startswith("VACANT"):
        return "Parking" if "PARKING" in p else "Vacant Land"
    if p.startswith("PARKING LOT"):
        return "Parking"
    if p.startswith(("RESIDENTIAL-CONDOMINIUM", "RESIDENTIAL-COOPERATIVE", "COOPERATIVE")):
        return "Condominium"
    if p.startswith("RESIDENTIAL-SINGLE FAMILY"):
        return "Single Family"
    if p.startswith(("RESIDENTIAL-FLATS", "RESIDENTIAL-CONVERSION", "RESIDENTIAL-MULTI",
                     "RESIDENTIAL-APARTMENT", "RESIDENTIAL-MIXED USE",
                     "RESIDENTIAL-TRANSIENT")):
        return "Multifamily"
    if p.startswith(("RESIDENTIAL-GARAGE", "GARAGE-MULTIFAMILY")):
        return "Other"
    if p.startswith(("COMMERCIAL", "OFFICE", "STORE", "HOTEL", "MOTEL", "INN",
                     "RESTAURANT", "FAST FOOD", "THEATERS", "CLUB", "MEDICAL",
                     "TOURIST", "VEHICLE SERVICE")):
        return "Commercial"
    if p.startswith("INDUSTRIAL"):
        return "Industrial"
    if p.startswith(("EDUCATIONAL", "RELIGIOUS", "MUSEUMS", "SPECIAL PURPOSE",
                     "PUBLIC SERVICE", "HEALTH CARE", "DORMITORY", "RECREATIONAL",
                     "FRATERNITY", "EMBASSY")):
        return "Institutional"
    return "Other"


def refined_category(cat: str, land: float, impr: float) -> str | None:
    """DC assessments are land-heavy (citywide median improvement share ~0.49), so the
    generic 0.5 land-share cutoff would flag ~half the city Underdeveloped. Use the
    repo-standard thresholds (parcel_calculations.classify_property_refined): land
    share >= 0.67 for Single Family, >= 0.50 for everything else."""
    c = str(cat or "")
    total = (land or 0) + (impr or 0)
    if "Vacant" in c:
        return "Vacant"
    if "Parking" in c:
        return "Parking Lot"
    if total > 0 and (impr or 0) == 0:
        return "Vacant"
    cutoff = 0.67 if c == "Single Family" else 0.50
    if total > 0 and (land or 0) >= cutoff * total:
        return "Underdeveloped"
    return None


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrape", action="store_true", help="force fresh pulls (ignore raw cache)")
    args = ap.parse_args()

    log("=== Washington, DC ===")
    lots = cached("col_lots", lambda: fetch_geometry(COL_URL, COL_FIELDS), args.scrape, geo=True)
    itspe_url = resolve_itspe_url()
    log(f"  ITSPE endpoint: {itspe_url}")
    itspe = cached("itspe", lambda: fetch_table(itspe_url + "/query", ITSPE_FIELDS),
                   args.scrape, geo=False)
    relate = cached("condorelate", lambda: fetch_table(CONDORELATE_URL, CONDORELATE_FIELDS),
                    args.scrape, geo=False)

    for c in ["NEWLAND", "NEWIMPR", "NEWTOTAL", "LANDAREA"]:
        lots[c] = pd.to_numeric(lots.get(c), errors="coerce")
    for c in ["NEWLAND", "NEWIMPR", "NEWTOTAL"]:
        itspe[c] = pd.to_numeric(itspe.get(c), errors="coerce")

    lots["ssl_n"] = norm_ssl(lots["SSL"])
    itspe["ssl_n"] = norm_ssl(itspe["SSL"])
    relate["ssl_n"] = norm_ssl(relate["SSL"])          # unit SSL
    relate["mat_ssl_n"] = norm_ssl(relate["MAT_SSL"])  # underlying record lot SSL

    lots = lots[lots.geometry.notna() & ~lots.geometry.is_empty].copy()
    log(f"  lots with geometry: {len(lots):,}")
    log(f"  LOT_TYPE mix: {lots['LOT_TYPE'].value_counts(dropna=False).to_dict()}")

    # Dedup: one row per SSL (multi-sheet squares can emit >1 polygon; values are
    # SSL-level and broadcast identically -> first, geometry union).
    if lots.duplicated(subset=["ssl_n"], keep=False).any():
        n0 = len(lots)
        first_cols = [c for c in lots.columns if c not in ("geometry", "ssl_n")]
        coll = lots.groupby("ssl_n", dropna=False).agg({c: "first" for c in first_cols})
        gu = lots.groupby("ssl_n", dropna=False)["geometry"].apply(
            lambda gs: unary_union([x for x in gs if x is not None]))
        coll["geometry"] = gu
        lots = gpd.GeoDataFrame(coll.reset_index(), geometry="geometry", crs=lots.crs)
        log(f"  SSL dedup: {n0:,} -> {len(lots):,}")

    # ── Condo rollup ─────────────────────────────────────────────────────────
    # Exempt units excluded from sums so fully-exempt buildings drop out later.
    itspe["unit_exempt"] = is_exempt(itspe.get("MIX1TXTYPE"), itspe.get("OWNERNAME"),
                                     itspe.get("PROPTYPE"))
    units = relate.merge(itspe[["ssl_n", "PROPTYPE", "NEWLAND", "NEWIMPR", "NEWTOTAL",
                                "unit_exempt"]],
                         on="ssl_n", how="inner", suffixes=("_rel", ""))
    log(f"  condo units: relate={len(relate):,}, matched in ITSPE={len(units):,}")

    units["sq_key"] = (norm_code(units["SQUARE"]) + "|" + norm_code(units["SUFFIX"]) + "|"
                       + norm_code(units["REGIME"]))
    tax_units = units[~units["unit_exempt"]].copy()
    by_regime = tax_units.groupby("sq_key").agg(
        u_land=("NEWLAND", "sum"), u_impr=("NEWIMPR", "sum"), u_total=("NEWTOTAL", "sum"),
        u_count=("ssl_n", "size"),
        u_proptype=("PROPTYPE", lambda s: s.mode().iat[0] if len(s.mode()) else None),
    )
    by_matssl = tax_units.groupby("mat_ssl_n").agg(
        u_land=("NEWLAND", "sum"), u_impr=("NEWIMPR", "sum"), u_total=("NEWTOTAL", "sum"),
        u_count=("ssl_n", "size"),
        u_proptype=("PROPTYPE", lambda s: s.mode().iat[0] if len(s.mode()) else None),
    )

    is_condo_fp = ((norm_code(lots["CONDOLOT"]).str.upper() == "Y")
                   | (pd.to_numeric(lots["UNDERLIES_CONDO"], errors="coerce") == 1))
    no_value = lots["NEWTOTAL"].fillna(0) <= 0
    target = is_condo_fp & no_value
    log(f"  condo footprints: {int(is_condo_fp.sum()):,} ({int(target.sum()):,} valueless -> rollup)")

    lots["sq_key"] = (norm_code(lots["SQUARE"]) + "|" + norm_code(lots["SUFFIX"]) + "|"
                      + norm_code(lots["CONDO_REGIME_NUM"]))
    lots["condo_units"] = 0

    # primary: (SQUARE, SUFFIX, REGIME); fallback: footprint SSL == MAT_SSL
    for src, key in ((by_regime, "sq_key"), (by_matssl, "ssl_n")):
        todo = target & (lots["NEWTOTAL"].fillna(0) <= 0)
        m = lots.loc[todo, key].map(src["u_total"])
        hit = todo & m.reindex(lots.index).notna() & (m.reindex(lots.index) > 0)
        if not hit.any():
            continue
        idx = lots.index[hit]
        keyvals = lots.loc[idx, key]
        lots.loc[idx, "NEWLAND"] = keyvals.map(src["u_land"]).values
        lots.loc[idx, "NEWIMPR"] = keyvals.map(src["u_impr"]).values
        lots.loc[idx, "NEWTOTAL"] = keyvals.map(src["u_total"]).values
        lots.loc[idx, "PROPTYPE"] = keyvals.map(src["u_proptype"]).values
        lots.loc[idx, "condo_units"] = keyvals.map(src["u_count"]).values
        log(f"    rollup via {key}: filled {int(hit.sum()):,} footprints")
    unresolved = int((target & (lots["NEWTOTAL"].fillna(0) <= 0)).sum())
    log(f"    condo footprints left valueless (exempt buildings / no units): {unresolved:,}")

    # A regime whose land spans several record lots yields SEVERAL footprints that each
    # just received the FULL regime sum (N x value inflation). Merge them into one
    # parcel per regime: values FIRST (regime-level, broadcast identically), geometry union.
    rolled_dup = (lots["condo_units"] > 0) & lots.duplicated(subset=["sq_key"], keep=False) \
        & (norm_code(lots["CONDO_REGIME_NUM"]) != "")
    if rolled_dup.any():
        dup = lots[rolled_dup].copy()
        keep = lots[~rolled_dup]
        first_cols = [c for c in dup.columns if c not in ("geometry", "sq_key")]
        coll = dup.groupby("sq_key", dropna=False).agg({c: "first" for c in first_cols})
        coll["geometry"] = dup.groupby("sq_key", dropna=False)["geometry"].apply(
            lambda gs: unary_union([x for x in gs if x is not None]))
        lots = gpd.GeoDataFrame(pd.concat([keep, coll.reset_index()], ignore_index=True),
                                geometry="geometry", crs=lots.crs)
        log(f"    regime-split merge: {int(rolled_dup.sum()):,} footprints -> "
            f"{dup['sq_key'].nunique():,} regime parcels")

    # ── Exemption + category ─────────────────────────────────────────────────
    rolled = lots["condo_units"] > 0
    lots["exemption_flag"] = np.where(
        rolled, 0,
        is_exempt(lots.get("MIX1TXTYPE"), lots.get("OWNERNAME"), lots.get("PROPTYPE"))
        .astype(int))
    lots["property_land_use_category"] = lots["PROPTYPE"].apply(dc_categorize)
    log(f"  categories: {lots['property_land_use_category'].value_counts().to_dict()}")
    log(f"  exempt: {int(lots['exemption_flag'].sum()):,}")

    ex = lots[lots["exemption_flag"] == 0].copy()
    # keep only assessed parcels — drops federal reservations, ROW, valueless remainders
    ex = ex[(ex["NEWLAND"].fillna(0) > 0) | (ex["NEWIMPR"].fillna(0) > 0)].copy()
    log(f"  after exempt + value filter: {len(ex):,}")

    # Stacked footprints that remain (air-rights lots over their ground lot etc.) are
    # DISTINCT assessments on the same land: SUM values onto one unioned polygon
    # (Newport News pattern). LANDAREA describes the same ground -> first.
    rp = ex.geometry.representative_point()
    ex["_rpkey"] = rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)
    vc = ex["_rpkey"].value_counts()
    stacked = vc[vc > 1].index
    if len(stacked):
        log(f"  stacked footprints: {len(stacked):,} clusters (max {int(vc.max())}) -> sum-collapse")
        is_st = ex["_rpkey"].isin(stacked)
        multi = ex[is_st].copy()
        sum_cols = ["NEWLAND", "NEWIMPR", "NEWTOTAL", "condo_units"]
        first_cols = [c for c in multi.columns if c not in (["geometry", "_rpkey"] + sum_cols)]
        agg = {c: "sum" for c in sum_cols}
        agg.update({c: "first" for c in first_cols})
        coll = multi.groupby("_rpkey", dropna=False).agg(agg)
        coll["geometry"] = multi.groupby("_rpkey", dropna=False)["geometry"].apply(
            lambda gs: unary_union([x for x in gs if x is not None]))
        ex = gpd.GeoDataFrame(pd.concat([ex[~is_st], coll.reset_index(drop=True)],
                                        ignore_index=True),
                              geometry="geometry", crs=ex.crs)
    ex = ex.drop(columns=["_rpkey"], errors="ignore")
    log(f"  after stacked collapse: {len(ex):,}")

    # ── Canonical fields ─────────────────────────────────────────────────────
    ex["geometry"] = ex["geometry"].apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
    ex["gis_area_sqft"] = pd.Series([geodesic_area_sqft(g) for g in ex.geometry],
                                    index=ex.index, dtype="float64")
    ex.loc[ex["gis_area_sqft"] < 1, "gis_area_sqft"] = np.nan
    # LANDAREA is OTR's stated lot sqft; null on condo footprints -> GIS fallback.
    # Rolled-up condo parcels ALWAYS use GIS area: their LANDAREA (when present) is a
    # single record lot's / unit share's area, not the merged development ground.
    reported = ex["LANDAREA"].where(ex["LANDAREA"] >= 1, np.nan)
    reported = reported.where(ex["condo_units"].fillna(0) == 0, np.nan)
    use_reported = reported > 0
    ex["land_area_sqft"] = np.where(use_reported, reported, ex["gis_area_sqft"])
    ex["area_source"] = np.where(use_reported, "reported", "gis")
    ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
    ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

    ex["land_value"] = ex["NEWLAND"]
    ex["improvement_value"] = ex["NEWIMPR"]
    ex["full_market_value"] = ex["NEWTOTAL"].fillna(
        ex["NEWLAND"].fillna(0) + ex["NEWIMPR"].fillna(0))
    den = ex["land_area_sqft"].replace(0, np.nan)
    ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
    ex["land_value_per_sqft"] = ex["land_value"] / den
    ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
    ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")

    ex["property_land_use_refined"] = [
        refined_category(c, l, i) for c, l, i in
        zip(ex["property_land_use_category"], ex["land_value"].fillna(0),
            ex["improvement_value"].fillna(0))
    ]
    ex["link"] = np.nan  # PropertyQuest / MyTaxDC are not deep-linkable by SSL

    COLUMNS = ["geometry", "exemption_flag", "property_land_use_category",
               "property_land_use_refined", "full_market_value", "full_market_value_per_sqft",
               "land_value", "land_value_per_sqft", "improvement_value",
               "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
               "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres",
               "area_source", "likely_remnant", "condo_units"]
    for c in COLUMNS:
        if c not in ex.columns:
            ex[c] = np.nan
    final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
    final = gpd.GeoDataFrame(final, geometry="geometry", crs=ex.crs)
    if final.crs is None or final.crs.to_epsg() != 4326:
        final = final.to_crs("EPSG:4326")

    out = DATA_DIR / "washington-dc-parcels.parquet"
    final.to_parquet(out, index=False)
    final.to_parquet(DATA_DIR / f"washington-dc-parcels_{datetime.now().strftime('%Y_%m_%d')}.parquet",
                     index=False)
    log(f"SAVED {out} | rows {len(final):,}")
    log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
    lv = final["land_value_per_sqft"]
    log(f"land $/sqft: p50=${lv.median():.0f} p99=${lv.quantile(.99):.0f} "
        f"p999=${lv.quantile(.999):.0f} max=${lv.max():.0f}")
    log("DONE")


if __name__ == "__main__":
    main()
