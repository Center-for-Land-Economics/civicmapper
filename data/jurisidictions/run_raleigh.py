#!/usr/bin/env python3
"""
Build the City of Raleigh, NC canonical parcel parquet.

Raleigh sits almost entirely in Wake County (a small RTP-area sliver lies in Durham County;
Wake's data does not cover it and it is dropped). Values come from Wake County Tax
Administration; both sources are public and need no token.

Sources:
- Parcels (geometry + REID/PIN + land/building value + type-and-use + land class + billing
  class + exemption), Wake County GIS, refreshed nightly:
    https://maps.wake.gov/arcgis/rest/services/Property/Parcels/MapServer/0
  (maps.wakegov.com, the older host that AGOL search results point at, is blocked from the
  cloud dev environment; maps.wake.gov is the same server and is reachable.)
  Pulled where CITY='RAL' OR PLANNING_JURISDICTION='RA', cached to raleigh-nc-geometry.parquet.
- City limits: Wake "Dissolved Corporate Limits", SHORT_NAME='RAL' (centroid-within clip):
    https://maps.wake.gov/arcgis/rest/services/Jurisdictions/DissolvedCorporateLimits/MapServer/0
- Full real-estate roll (fixed-width text, layout PDF linked from wake.gov's "Real Estate
  Property Data Files" page), date-stamped and refreshed daily:
    https://services.wake.gov/realdata_extracts/RealEstData<MMDDYYYY>.zip
  The newest one in the directory listing is downloaded automatically, or pass --roll <path>.

Why the roll is needed (skill §6c / §6d):
  The GIS layer holds ONE polygon per PIN. Condominium units and leasehold improvements
  (buildings on someone else's land) are separate REIDs with no polygon: in the Nov 2026 roll,
  ~13.6k Raleigh REIDs (~$8.0B building value) are missing from the GIS layer. The 641 condo
  "MASTER CARD" parcels (land class COX, billing class HOME ASSOC) carry $0 land and $0
  building. Every missing record's PIN (first 10 digits of the 19-char roll PIN) is the host
  parcel's PIN_NUM, so they are summed onto it: an exact key, no spatial heuristics.

  Wake assesses a condo unit's WHOLE value as building (unit land = $0), like MA/RI/CT (§6d).
  There is no land figure to recover. Condos are ~6% of Raleigh's taxable value, so the
  merged master parcels ship with the assessor's (near-)$0 land and the gp-error layer flags
  them, as for Providence/Hartford. No land is imputed.

Values: one row per REID in both sources. A multi-card REID repeats the same REID-level
values on every card in the roll; this is asserted, then `first` is taken (never summed, §2).

Exemption (`exemption_flag = 1`, excluded):
  - BILLCLASS 3 "Exempt" (city/county/state/federal, schools, religious, PHA, LIHTC, ...).
  - BILLCLASS 6 "HOME ASSOC" common areas, EXCEPT condo master parcels that received units.
    NC assesses HOA common area at ~$0 (G.S. 105-277.8); left in, it maps as acres of "Vacant".
  - BILLCLASS 4 "Public" (state-assessed utilities/railroads) is categorised "Utility" and
    dropped with the other non-land categories.
  Unmapped (unit/leasehold) records that are themselves exempt are not summed.
  Taxable leaseholds on an exempt host (e.g. RDU, state land) are dropped with the host.

Outputs:
- data/jurisidictions/data/raleigh/raleigh-nc-parcels.parquet
- data/jurisidictions/data/raleigh/raleigh-nc-parcels_YYYY_MM_DD.parquet

PMTiles (~135k parcels -> PMTiles + H3, hideRemnants):
    python data/scripts/parquet_to_pmtiles.py --city raleigh \
      --file data/jurisidictions/data/raleigh/raleigh-nc-parcels.parquet --drop-remnants
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import add_improvement_ratio_fields, classify_property_refined, gis_area_sqft  # noqa: E402

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "raleigh"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GEOM_CACHE = DATA_DIR / "raleigh-nc-geometry.parquet"
ROLL_DIR = DATA_DIR / "roll"

PARCELS_URL = "https://maps.wake.gov/arcgis/rest/services/Property/Parcels/MapServer/0/query"
PARCELS_WHERE = "CITY='RAL' OR PLANNING_JURISDICTION='RA'"
BOUNDARY_URL = ("https://maps.wake.gov/arcgis/rest/services/Jurisdictions/"
                "DissolvedCorporateLimits/MapServer/0/query")
ROLL_INDEX = "https://services.wake.gov/realdata_extracts/"
LINK = "https://services.wake.gov/realestate/Account.asp?id="
HEADERS = {"User-Agent": "Mozilla/5.0 (civicmapper ETL)", "Accept": "application/json, */*"}
SQFT_PER_ACRE = 43560.0
UTM = 32617  # UTM 17N
PAGE = 1000

# Wake roll record layout (1-based start, width), from "Record Layout.pdf" (wake.gov).
ROLL_SPEC = [("REID", 176, 7), ("CARD", 183, 3), ("BLDG", 295, 11), ("LAND", 306, 11),
             ("PIN", 317, 19), ("BILLCLASS", 345, 1), ("PROPDESC", 346, 40),
             ("LANDCLASS", 386, 1), ("CITY", 612, 3), ("TU", 654, 3)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get(url, **kw):
    for attempt in range(5):
        try:
            r = requests.get(url, headers=HEADERS, timeout=600, **kw)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            log(f"  retry {attempt + 1}: {type(e).__name__} {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}")


# ── 1. Parcels (geometry + values), cached ───────────────────────────────────
def fetch_parcels() -> gpd.GeoDataFrame:
    if GEOM_CACHE.exists():
        log(f"Using cached geometry: {GEOM_CACHE.name}")
        return gpd.read_parquet(GEOM_CACHE)
    ids = sorted(get(PARCELS_URL, params={"where": PARCELS_WHERE, "returnIdsOnly": "true",
                                          "f": "json"}).json()["objectIds"])
    log(f"Pulling {len(ids):,} Wake parcels (by OBJECTID, GeoJSON)...")
    pages = []
    for i in range(0, len(ids), PAGE):
        chunk = ",".join(map(str, ids[i:i + PAGE]))
        r = None
        for attempt in range(5):
            try:
                r = requests.post(PARCELS_URL, headers=HEADERS, timeout=300, data={
                    "objectIds": chunk, "outFields": "*", "returnGeometry": "true",
                    "outSR": 4326, "f": "geojson"})
                r.raise_for_status()
                pages.append(gpd.read_file(io.BytesIO(r.content)))
                break
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt + 1} @ {i}: {type(e).__name__}")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"parcel pull failed at {i}")
        if i % 20000 == 0:
            log(f"  fetched {i + PAGE:,}/{len(ids):,}")
    g = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    g.to_parquet(GEOM_CACHE, index=False)
    log(f"  cached -> {GEOM_CACHE.name} ({len(g):,})")
    return g


# ── 2. Roll (fixed width), cached ────────────────────────────────────────────
def fetch_roll(path: str | None) -> pd.DataFrame:
    if path:
        src = Path(path)
    else:
        ROLL_DIR.mkdir(exist_ok=True)
        cached = sorted(ROLL_DIR.glob("RealEstData*.txt"))
        if cached:
            src = cached[-1]
        else:
            names = re.findall(r"RealEstData(\d{8})\.zip", get(ROLL_INDEX).text)
            if not names:
                raise RuntimeError(f"no RealEstData*.zip listed at {ROLL_INDEX}")
            latest = max(names, key=lambda s: (s[4:], s[:4]))  # MMDDYYYY -> sort by YYYY, MMDD
            log(f"Downloading roll RealEstData{latest}.zip ...")
            z = zipfile.ZipFile(io.BytesIO(get(f"{ROLL_INDEX}RealEstData{latest}.zip").content))
            member = next(n for n in z.namelist() if n.lower().endswith(".txt"))
            src = ROLL_DIR / Path(member).name
            src.write_bytes(z.read(member))
    log(f"Parsing roll {src.name} ...")
    df = pd.read_fwf(src, colspecs=[(s - 1, s - 1 + w) for _, s, w in ROLL_SPEC],
                     names=[n for n, _, _ in ROLL_SPEC], dtype=str, encoding="latin-1")
    for c in ("LAND", "BLDG"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # REID-level values are repeated on every card: assert, then take first (never sum).
    varies = df.groupby("REID")[["LAND", "BLDG"]].nunique().gt(1).any(axis=1).sum()
    log(f"Roll: {len(df):,} card rows, {df['REID'].nunique():,} REIDs; "
        f"REIDs whose values differ across cards: {varies}")
    assert varies == 0, "roll values vary across cards; summing/first logic must be revisited"
    df = df.drop_duplicates("REID")
    # '1714.09 05 9993 000' -> PIN_NUM '1714059993' + suffix '000'
    parts = df["PIN"].fillna("").str.split()
    df["pin10"] = parts.str[0].str[:4] + parts.str[1].fillna("") + parts.str[2].fillna("")
    df["pin_sfx"] = parts.str[3]
    return df


def categorize(tu: str, landclass: str, bld: float) -> str:
    """Wake TYPE_AND_USE code + LAND_CLASS -> coarse category."""
    tu = tu.strip().lstrip("0") if isinstance(tu, str) else ""
    lc = landclass.strip() if isinstance(landclass, str) else ""
    if lc == "STA":
        return "Utility"
    if lc == "COX":
        return "Condominium"
    if lc in ("AGR", "FOR", "HRT"):
        return "Agricultural / Rural"
    if lc == "VAC" or (not tu and bld <= 0):
        return "Vacant"
    if lc == "WSS":
        return "Utility"
    if lc == "CEM":
        return "Cemetery"
    if lc == "MHP" or tu == "72":
        return "Mobile Home Park"
    if not tu:
        return "Rural / Acreage" if lc in ("AWI", "AHS") else "Other"
    n = int(tu) if tu.isdigit() else -1
    if n == 1:
        return "Single Family"
    if n in (2, 3, 4, 5):
        return "Small Multifamily (2-4 units)"
    if n in (7, 8, 9, 100, 67, 104, 105):
        return "Multifamily"
    if n in (21, 210):
        return "Parking Garage / Deck"
    if n in (28, 29, 30, 31, 32):
        return "Hotel / Motel"
    if n in (11, 12, 34, 36, 37, 38):
        return "Office"
    if n in (79, 80, 83, 84, 85, 86, 88, 90, 92, 94, 95, 96):
        return "Industrial / Warehouse"
    if n in (66, 68, 70, 71, 73, 74, 76, 77, 102, 106, 107, 108, 111, 113, 114):
        return "Institutional"
    if n in (6, 35, 57, 58, 59):
        return "Mixed Use"
    return "Commercial"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", help="path to an extracted RealEstData*.txt (default: download newest)")
    args = ap.parse_args()

    geom = fetch_parcels()
    for c in ("LAND_VAL", "BLDG_VAL", "TOTAL_VALUE_ASSD", "BILLCLASS"):
        geom[c] = pd.to_numeric(geom[c], errors="coerce")

    # City-limits clip (centroid-within on the authoritative dissolved boundary).
    b = gpd.read_file(io.BytesIO(get(BOUNDARY_URL, params={
        "where": "SHORT_NAME='RAL'", "outFields": "SHORT_NAME", "outSR": 4326,
        "f": "geojson"}).content))
    boundary = unary_union(list(b.geometry))
    geom["geometry"] = geom.geometry.make_valid()
    pts = geom.to_crs(UTM).representative_point().to_crs(4326)
    geom = geom[pts.within(boundary).to_numpy()].copy()
    log(f"City-limits clip -> {len(geom):,}; CITY codes: {geom['CITY'].value_counts().to_dict()}")

    # Geometry grain preflight (§2a): one row per REID and per PIN expected.
    for key in ("REID", "PIN_NUM"):
        d = geom[geom[key].duplicated(keep=False)]
        log(f"  {key} with >1 geometry row: {d[key].nunique()}")
    if geom["REID"].duplicated().any():
        geom = geom.dissolve(by="REID", aggfunc="first", as_index=False)

    # ── 3. Merge polygon-less roll records (condo units, leaseholds) onto host ──
    roll = fetch_roll(args.roll)
    mapped = set(geom["REID"])
    extra = roll[(roll["CITY"] == "RAL") & ~roll["REID"].isin(mapped)].copy()
    log(f"Roll RAL REIDs without a polygon: {len(extra):,} "
        f"(land ${extra['LAND'].sum() / 1e6:,.1f}M, bldg ${extra['BLDG'].sum() / 1e9:,.2f}B); "
        f"land class: {extra['LANDCLASS'].value_counts().head(6).to_dict()}")
    ex_extra = extra["BILLCLASS"].eq("3")
    log(f"  of which exempt (not merged): {int(ex_extra.sum()):,}")
    extra = extra[~ex_extra]
    host = extra["pin10"].isin(set(geom["PIN_NUM"]))
    log(f"  host PIN found: {int(host.sum()):,}; orphan (dropped): {int((~host).sum()):,} "
        f"(${extra.loc[~host, 'BLDG'].sum() / 1e6:,.1f}M bldg)")
    extra = extra[host]
    extra["is_unit"] = extra["LANDCLASS"].eq("N")
    agg = extra.groupby("pin10").agg(x_land=("LAND", "sum"), x_bldg=("BLDG", "sum"),
                                     x_n=("REID", "size"), x_units=("is_unit", "sum"),
                                     x_tu=("TU", lambda s: s.dropna().mode().iloc[0]
                                           if s.notna().any() else None))
    geom = geom.merge(agg, how="left", left_on="PIN_NUM", right_index=True)
    for c in ("x_land", "x_bldg", "x_n", "x_units"):
        geom[c] = geom[c].fillna(0)
    geom["land_val"] = geom["LAND_VAL"].fillna(0) + geom["x_land"]
    geom["bld_val"] = geom["BLDG_VAL"].fillna(0) + geom["x_bldg"]
    geom["n_condo_units"] = geom["x_units"].astype(int)
    log(f"  hosts receiving records: {int((geom['x_n'] > 0).sum()):,} "
        f"(condo hosts {int((geom['x_units'] > 0).sum()):,})")

    # ── 4. Classification + exemption ────────────────────────────────────────
    geom["PROPERTY_CATEGORY"] = [categorize(t, lc, b) for t, lc, b in
                                 zip(geom["TYPE_AND_USE"], geom["LAND_CLASS"], geom["bld_val"])]
    # A condo host's category follows its units' type-and-use (residential vs office/retail).
    is_condo = geom["n_condo_units"] > 0
    unit_cat = geom["x_tu"].fillna("").astype(str).str.lstrip("0").map(
        lambda t: "Condominium" if t in ("", "1", "2", "3", "4", "5", "7", "8", "9")
        else "Commercial Condominium")
    geom.loc[is_condo, "PROPERTY_CATEGORY"] = unit_cat[is_condo]

    hoa = geom["BILLCLASS"].eq(6) & ~is_condo
    exempt = geom["BILLCLASS"].eq(3)
    geom["exemption_flag"] = (exempt | hoa).astype(int)
    log(f"Exempt: billing-class Exempt {int(exempt.sum()):,}, HOA common area {int(hoa.sum()):,}; "
        f"taxable leasehold/unit value lost on exempt hosts: "
        f"${geom.loc[geom['exemption_flag'].eq(1), 'x_bldg'].sum() / 1e6:,.1f}M")
    ex = geom[geom["exemption_flag"] == 0].copy()
    ex = ex[~ex["PROPERTY_CATEGORY"].isin(["Utility"])].copy()
    ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
    ex["land_value"] = ex["land_val"]
    ex["improvement_value"] = ex["bld_val"]
    ex["bld_ar"] = pd.to_numeric(ex["HEATEDAREA"], errors="coerce").fillna(0)
    ex["property_land_use_refined"] = classify_property_refined(
        ex, fetch_footprints=False, exclude_state_class_prefixes=(),
        exclude_categories=("Utility", "Agricultural / Rural", "Cemetery",
                            "Condominium", "Commercial Condominium"))
    log(f"After exempt/refine -> {len(ex):,}")

    # ── 5. Canonical fields ──────────────────────────────────────────────────
    log("Computing geodesic areas...")
    ex["land_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
    ex.loc[ex["land_area_sqft"] < 1, "land_area_sqft"] = np.nan
    ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
    ex["area_source"] = "gis"
    ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)
    ex["full_market_value"] = ex["land_value"] + ex["improvement_value"]
    den = ex["land_area_sqft"].replace(0, np.nan)
    ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
    ex["land_value_per_sqft"] = ex["land_value"] / den
    ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
    ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")
    ex["link"] = LINK + ex["REID"].astype(str)
    ex["parcel_id"] = ex["PIN_NUM"]
    ex["address"] = ex["SITE_ADDRESS"].fillna("").str.strip()

    COLUMNS = ["geometry", "parcel_id", "exemption_flag", "property_land_use_category",
               "property_land_use_refined", "full_market_value", "full_market_value_per_sqft",
               "land_value", "land_value_per_sqft", "improvement_value",
               "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO", "IMPR_LAND_PCT",
               "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source", "likely_remnant",
               "address", "n_condo_units"]
    final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
    final = gpd.GeoDataFrame(final, geometry="geometry", crs="EPSG:4326")

    # ── 6. Post-flight checks (§2a, §6a) ────────────────────────────────────
    assert not final["parcel_id"].duplicated().any(), "duplicate parcel_id in output"
    gp = final.to_crs(UTM)
    a = gp.geometry.area * 10.7639
    lv = final["land_value_per_sqft"]
    log(f"footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
    log(f"sub-500 / sub-1000 sqft: {int((a < 500).sum())} / {int((a < 1000).sum())}")
    log(f"land $/sqft p50/p99/max: {lv.median():.0f} / {lv.quantile(.99):.0f} / {lv.max():.0f}")
    c = gp.geometry.representative_point()
    vc = (c.x.round(1).astype(str) + "," + c.y.round(1).astype(str)).value_counts()
    log(f"stacked clusters: {int((vc > 1).sum())}, max stack {int(vc.max())}")
    log(f"zero-land parcels with value: "
        f"{int(((final['current_full_land_value'] <= 0) & (final['improvement_value'] > 0)).sum())}")

    out = DATA_DIR / "raleigh-nc-parcels.parquet"
    final.to_parquet(out, index=False)
    final.to_parquet(DATA_DIR / f"raleigh-nc-parcels_{datetime.now():%Y_%m_%d}.parquet", index=False)
    log(f"SAVED {out} | rows {len(final):,}")
    log(f"totals: land ${final['current_full_land_value'].sum() / 1e9:,.2f}B, "
        f"impr ${final['improvement_value'].sum() / 1e9:,.2f}B")
    log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
    log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
    log("DONE")


if __name__ == "__main__":
    main()
