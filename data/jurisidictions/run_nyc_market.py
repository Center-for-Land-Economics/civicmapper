#!/usr/bin/env python3
"""NYC parcels on DOF MARKET values (replaces the AssessLand/AssessTot build in nyc.ipynb).

WHY
---
nyc.ipynb mapped MapPLUTO `AssessLand` / `AssessTot`. Those are NYC's ASSESSED values, and NYC
assesses by tax class at very different fractions of market value:
    Class 1 (1-3 family homes)                 6% of market
    Class 2 (apartments, co-ops, condos), 4    45% of market
So a Park Slope brownstone showed ~$22/sqft of land while the 4-unit walkup next door (class 2A)
showed $139/sqft, although DOF's own MARKET land values for the two are ~$320 and ~$309/sqft.
Every other CivicMapper city maps market/appraised value, so NYC now does too.

SOURCE
------
Geometry / land use / building class: MapPLUTO (cached raw scrape, data/nyc/mappluto_raw_*.parquet;
`--scrape-pluto` is not implemented here: run nyc.ipynb's scrape cell to refresh it).

Values: NYC DOF "Property Valuation and Assessment Data Tax Classes 1,2,3,4"
(NYC Open Data, Socrata id 8y4t-faws), one row per BBL per roll (year, period):
    period 1 = tentative roll (January), 2 = change by notice, 3 = FINAL roll (May/June).
    pymkt*  = prior year,  ten* = tentative,  cbn* = change-by-notice,  fin* = final (period 3 only)
    cur*    = the value CURRENT as of that period (== ten* in period 1, == fin* in period 3).
We use `curmktland` / `curmkttot` from the latest roll year, preferring its FINAL period (3) when
published, so the map shows "the market value DOF is taxing on" for that fiscal year. `cur*` is
used rather than `fin*` so the same code also works on a tentative-only year.
Only rectype 1 (real-property lots) is kept; rectype 3 are utility/special-franchise records
(class U/T on synthetic blocks) that have no PLUTO polygon.

CONDOS
------
PLUTO maps a condominium as its BILLING lot (lot 7501-7599) with the building's footprint. DOF lists
the billing lot with $0 and every unit lot (1001+) separately; both carry `condo_number`.
Unit values are summed onto that condo's billing lot; a condo spanning several tax blocks has one
billing lot per block, and units go to the billing lot on their own block (pooled + split by lot
area only when their block has none). Each unit is a DISTINCT value record (not a broadcast), so
summing is correct (SKILL §2a); the roll-up asserts every value column is conserved.
Easement records (parid = BBL + letter, class Z7/U7, ~$0.2B) have no polygon and are dropped.
Verification (`--verify`): PLUTO's AssessTot for a billing lot is DOF's sum of unit `curacttot` on
the roll PLUTO was built from (FY2027 tentative, 2027/1, for the Feb-2026 scrape); the script prints
the ratio distribution. 2026-10 run: 10,734 billing lots, 99.6% within 0.5% (multi-billing condos
99.4%), median ratio 1.0000.

EXEMPT
------
Same rule nyc.ipynb applied to PLUTO, re-expressed on DOF fields at the joined (condo-summed) level:
fully exempt = (curactextot >= curacttot AND curacttot > 0) OR (curacttot == 0 AND curactland == 0),
plus any lot with no market value at all. Fully exempt parcels are dropped (exemption_flag = 0).

OUTPUT: data/nyc/nyc-ny-parcels.parquet (+ dated copy). Then:
    python data/jurisidictions/augment_nyc_regions.py         # borough + NTA neighborhood columns
    python data/scripts/build_land_totals.py --city nyc       # parking-share denominators
    python data/scripts/parquet_to_pmtiles.py --city nyc ...  # PMTiles + H3 + metadata
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
from parcel_calculations import add_improvement_ratio_fields  # noqa: E402

DATA_DIR = HERE / "data" / "nyc"
OUT = DATA_DIR / "nyc-ny-parcels.parquet"
SOCRATA = "https://data.cityofnewyork.us/resource/8y4t-faws.csv"
DOF_FIELDS = [
    "parid", "boro", "block", "lot", "curmktland", "curmkttot", "curactland", "curacttot",
    "curactextot", "curtaxclass", "bldg_class", "land_area", "condo_number",
]
NUM = ["curmktland", "curmkttot", "curactland", "curacttot", "curactextot", "land_area"]
BORO_CODE = {"MN": 1, "BX": 2, "BK": 3, "QN": 4, "SI": 5}
PAGE = 50_000


def log(m: str) -> None:
    print(m, flush=True)


# ── DOF roll ─────────────────────────────────────────────────────────────────────────────────

def socrata_csv(params: dict, tries: int = 6) -> pd.DataFrame:
    for i in range(tries):
        try:
            r = requests.get(SOCRATA, params=params, timeout=300)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), dtype=str)
        except Exception as e:  # noqa: BLE001
            wait = 5 * (i + 1)
            log(f"    retry {i + 1}/{tries} after {e!r} (sleep {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Socrata request failed: {params}")


def latest_roll() -> tuple[int, int]:
    """Latest (year, period) with market values; prefer the final period of the latest year."""
    df = socrata_csv({"$select": "year,period,count(*) AS n,sum(curmktland) AS land",
                      "$group": "year,period", "$order": "year,period", "$limit": 100})
    df["land"] = pd.to_numeric(df["land"], errors="coerce").fillna(0)
    df = df[df["land"] > 0]
    df["year"] = df["year"].astype(int)
    df["period"] = df["period"].astype(int)
    log("  rolls available:\n" + df.to_string(index=False))
    y = int(df["year"].max())
    p = int(df.loc[df["year"] == y, "period"].max())
    return y, p


def _fetch_where(where: str, label: str) -> pd.DataFrame:
    frames, off = [], 0
    while True:
        d = socrata_csv({"$select": ",".join(DOF_FIELDS), "$where": where,
                         "$order": "parid", "$limit": PAGE, "$offset": off})
        frames.append(d)
        off += len(d)
        log(f"    {label}: {off:,} rows")
        if len(d) < PAGE:
            break
    return pd.concat(frames, ignore_index=True)


def fetch_roll(year: int, period: int, where_extra: str = "", tag: str = "") -> pd.DataFrame:
    raw_cache = DATA_DIR / f"dof_pvad_{year}_{period}{tag}.raw.parquet"
    if raw_cache.exists():
        log(f"  cached DOF roll: {raw_cache.name}")
        df = pd.read_parquet(raw_cache)
    else:
        from concurrent.futures import ThreadPoolExecutor
        base = f"year='{year}' AND period='{period}' AND rectype='1'" + where_extra
        log(f"  fetching DOF 8y4t-faws {year}/{period} {where_extra.strip()} (5 boroughs in parallel) ...")
        with ThreadPoolExecutor(5) as ex:
            parts = list(ex.map(lambda b: _fetch_where(f"{base} AND boro='{b}'", f"boro {b}"), "12345"))
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(raw_cache, index=False)  # cache BEFORE any validation
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ("boro", "block", "lot"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["condo_number"] = pd.to_numeric(df["condo_number"], errors="coerce").fillna(0).astype(int)
    df["bbl"] = df["boro"].astype("int64") * 1_000_000_000 + df["block"].astype("int64") * 10_000 \
        + df["lot"].astype("int64")
    # parid is unique per (year, period); a repeated parid would be an API paging artefact.
    n0 = len(df)
    df = df.drop_duplicates()
    if len(df) != n0:
        log(f"    dropped {n0 - len(df):,} exact duplicate rows (paging overlap)")
    # Easement records (parid = BBL + letter suffix, e.g. 1000440001E, class Z7/U7) value an
    # easement ON a lot separately. They have no polygon of their own and are not the lot's fee
    # value, so they're dropped (logged for the record).
    eas = df["parid"].astype(str).str.len() != 10
    if eas.any():
        log(f"    easement records dropped: {int(eas.sum()):,}  market land "
            f"${df.loc[eas, 'curmktland'].sum() / 1e9:,.2f}B  total ${df.loc[eas, 'curmkttot'].sum() / 1e9:,.2f}B")
        df = df[~eas].copy()
    dup = df["bbl"].duplicated(keep=False)
    if dup.any():
        log(f"    WARNING {int(dup.sum()):,} rows share a BBL with another parid, e.g.\n"
            + df.loc[dup, ["parid", "bbl", "curmktland", "curmkttot", "bldg_class"]].head(12).to_string())
    return df


def _share(df: pd.DataFrame, keys: list[str], area: pd.Series) -> pd.Series:
    """Each row's share within its `keys` group: by area when the group has any, else equal."""
    a = area.where(area > 0).fillna(0.0)
    tot = a.groupby([df[k] for k in keys]).transform("sum")
    n = df.groupby(keys)[keys[0]].transform("size")
    return pd.Series(np.where(tot > 0, a / tot.where(tot > 0, 1), 1.0 / n), index=df.index)


def roll_to_bbl_values(dof: pd.DataFrame, lot_area: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """One value row per PLUTO-addressable BBL: plain lots as-is, condo units summed onto billing lots.

    A condo can have several billing lots (one per tax block it spans, or per phase). Units go to
    the billing lot on THEIR OWN block (which is how DOF/PLUTO apportion them, verified against
    PLUTO AssessTot); units on a block with no billing lot of their condo are pooled and split over
    all its billing lots by lot area. Every unit dollar lands on exactly one billing lot.
    `lot_area` (BBL -> sqft, e.g. PLUTO LotArea) drives the area splits; DOF land_area otherwise.
    """
    is_billing = (dof["condo_number"] > 0) & dof["lot"].between(7501, 7599)
    is_unit = (dof["condo_number"] > 0) & ~is_billing
    vals = ["curmktland", "curmkttot", "curactland", "curacttot", "curactextot"]
    plain = dof[~is_billing & ~is_unit].copy()
    billing = dof[is_billing].copy()
    units = dof[is_unit].copy()

    area = billing["land_area"].astype(float)
    if lot_area is not None:
        area = billing["bbl"].map(lot_area).fillna(area)
    ck, bk = ["boro", "condo_number"], ["boro", "condo_number", "block"]

    # Dominant tax class of a condo = class holding the most market value.
    cls = (units.groupby(ck + ["curtaxclass"])["curmkttot"].sum().reset_index()
           .sort_values("curmkttot", ascending=False).drop_duplicates(ck)
           .set_index(ck)["curtaxclass"])

    ublk = units.groupby(bk)[vals].sum()
    ublk["n_units"] = units.groupby(bk).size()
    billing_blocks = billing.set_index(bk).index.unique()
    same_blk = ublk.index.isin(billing_blocks)
    pooled = ublk[~same_blk].groupby(level=[0, 1]).sum()           # per condo
    ublk = ublk[same_blk]

    billing["share_blk"] = _share(billing, bk, area)
    billing["share_condo"] = _share(billing, ck, area)
    b = billing.merge(ublk, left_on=bk, right_index=True, how="left", suffixes=("_bill", ""))
    b = b.merge(pooled.add_suffix("_pool"), left_on=ck, right_index=True, how="left")
    for v in vals + ["n_units"]:
        own = b[f"{v}_bill"].fillna(0) if f"{v}_bill" in b else 0.0
        b[v] = own + b[v].fillna(0) * b["share_blk"] + b[f"{v}_pool"].fillna(0) * b["share_condo"]
    b = b.merge(cls.rename("unit_taxclass"), left_on=ck, right_index=True, how="left")
    b["curtaxclass"] = b["unit_taxclass"].fillna(b["curtaxclass"])
    b["is_condo"] = True

    all_units = units.groupby(ck)[vals].sum()
    orphan = all_units.index.difference(billing.set_index(ck).index.unique())
    nb = billing.groupby(ck).size()
    stats = {
        "plain_lots": len(plain), "billing_lots": len(billing), "unit_lots": len(units),
        "condos": int(all_units.shape[0]), "condos_multi_billing": int((nb > 1).sum()),
        "unit_value_pooled_mkttot": float(pooled["curmkttot"].sum()),
        "orphan_condos": int(len(orphan)),
        "orphan_mkttot": float(all_units.loc[orphan, "curmkttot"].sum()) if len(orphan) else 0.0,
    }
    plain["n_units"] = 0
    plain["is_condo"] = False
    keep = ["bbl", "boro", "block", "lot", *vals, "curtaxclass", "bldg_class", "land_area",
            "condo_number", "n_units", "is_condo"]
    out = pd.concat([plain[keep], b[keep]], ignore_index=True)
    # Post-flight (SKILL §2a): value conserved -- nothing lost, nothing double-counted.
    for v in vals:
        expect = dof[v].sum() - (all_units.loc[orphan, v].sum() if len(orphan) else 0.0)
        assert abs(out[v].sum() - expect) <= 1.0 + 1e-9 * abs(expect), f"condo roll-up broke {v}"
    assert not out["bbl"].duplicated().any()
    return out, stats


# ── PLUTO ────────────────────────────────────────────────────────────────────────────────────

def load_pluto() -> gpd.GeoDataFrame:
    files = sorted(glob.glob(str(DATA_DIR / "mappluto_raw_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No mappluto_raw_*.parquet in {DATA_DIR} (run nyc.ipynb scrape cell)")
    log(f"  PLUTO: {Path(files[-1]).name}")
    g = gpd.read_parquet(files[-1])
    for c in ("AssessLand", "AssessTot", "ExemptTot", "LotArea", "BldgArea"):
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0)
    g["BBL"] = pd.to_numeric(g["BBL"], errors="coerce").astype("int64")
    assert not g["BBL"].duplicated().any(), "PLUTO has duplicate BBLs"
    return g


# ── categorisation ──────────────────────────────────────────────────────────────────────────
# MapPLUTO LandUse codes (PLUTO data dictionary). nyc.ipynb had these shifted/mislabelled: it called
# 01 (one & two family) "Single-Family", 02 (walk-ups) "Two-Family", 09 (OPEN SPACE: parks, golf,
# cemeteries) "Parking Lot", and 10 (the actual parking facilities) "Public Facilities".
LAND_USE_MAP = {
    "01": "One & Two Family", "02": "Multi-Family Walk-Up", "03": "Multi-Family Elevator",
    "04": "Mixed Residential & Commercial", "05": "Commercial & Office",
    "06": "Industrial & Manufacturing", "07": "Transportation & Utility",
    "08": "Public Facilities & Institutions", "09": "Open Space & Outdoor Recreation",
    "10": "Parking Lot", "11": "Vacant Land",
}
# Building classes in land use 10 that are STRUCTURES, not surface lots: G0 residential garage,
# G1 parking garage. Issue #12 house rule: a parking structure is judged on its improvement ratio,
# not force-labelled underused. (G6/G7 = licensed/unlicensed surface lots stay "Parking Lot".)
PARKING_STRUCTURE_CLASSES = ("G0", "G1")


def categorize(land_use, bldg_class) -> str:
    bc = str(bldg_class) if pd.notna(bldg_class) else ""
    if bc.startswith("V"):
        return "Vacant Land"
    lu = str(land_use).strip().zfill(2) if pd.notna(land_use) else ""
    cat = LAND_USE_MAP.get(lu, "Other")
    if cat == "Parking Lot" and bc[:2] in PARKING_STRUCTURE_CLASSES:
        return "Parking Garage"
    return cat


def refined(cat: pd.Series, land: pd.Series, impr: pd.Series) -> pd.Series:
    total = land + impr
    under = (total > 0) & (impr < 0.5 * total)
    out = pd.Series(np.where(under, "Underdeveloped", None), index=cat.index, dtype=object)
    out[cat.str.contains("Parking Lot", na=False)] = "Parking Lot"
    out[cat.str.contains("Vacant", na=False)] = "Vacant"
    return out


# ── main ─────────────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, help="DOF roll year (default: latest with market values)")
    ap.add_argument("--period", type=int, help="DOF roll period (default: latest of that year)")
    ap.add_argument("--verify-year", type=int, default=2027,
                    help="Roll year PLUTO's AssessTot was built from (condo verification)")
    ap.add_argument("--verify-period", type=int, default=1)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.year and args.period:
        year, period = args.year, args.period
    else:
        year, period = latest_roll()
    log(f"Using DOF roll {year} period {period} (curmktland / curmkttot)")

    pluto = load_pluto()
    log(f"  PLUTO lots: {len(pluto):,}")
    dof = fetch_roll(year, period)
    log(f"  DOF {year}/{period} rectype-1 lots: {len(dof):,}  "
        f"market land ${dof.curmktland.sum() / 1e9:,.1f}B  total ${dof.curmkttot.sum() / 1e9:,.1f}B")
    lot_area = pluto.set_index("BBL")["LotArea"]
    vals, st = roll_to_bbl_values(dof, lot_area)
    log(f"  condo roll-up: {st}")

    # ── verify the condo join against PLUTO's AssessTot (same roll PLUTO was built from) ──
    if not args.no_verify:
        vroll = fetch_roll(args.verify_year, args.verify_period, " AND condo_number IS NOT NULL", "_condo")
        vvals, _ = roll_to_bbl_values(vroll, lot_area)
        vb = vvals[vvals["is_condo"]].merge(pluto[["BBL", "AssessTot", "AssessLand"]],
                                           left_on="bbl", right_on="BBL", how="inner")
        vb = vb[vb["AssessTot"] > 0]
        r = vb["curacttot"] / vb["AssessTot"]
        rl = vb["curactland"] / vb["AssessLand"].replace(0, np.nan)
        log(f"  VERIFY condo billing lots ({args.verify_year}/{args.verify_period} unit sums vs PLUTO): "
            f"n={len(vb):,}  AssessTot ratio p05/p50/p95 = {r.quantile(.05):.4f}/{r.median():.4f}/"
            f"{r.quantile(.95):.4f}  exact(±0.5%)={(r.sub(1).abs() <= .005).mean():.1%}  "
            f"sum ratio={vb.curacttot.sum() / vb.AssessTot.sum():.4f}  "
            f"AssessLand sum ratio={vb.curactland.sum() / vb.AssessLand.sum():.4f} "
            f"(median {rl.median():.4f})")
        nbill = vroll[vroll["lot"].between(7501, 7599)].groupby(["boro", "condo_number"]).size()
        multi_cn = set(nbill[nbill > 1].index.get_level_values(1))
        mv = vb[vb["condo_number"].isin(multi_cn)]
        rm = mv["curacttot"] / mv["AssessTot"]
        log(f"  VERIFY multi-billing-lot condos: n={len(mv):,} billing lots, "
            f"exact(±0.5%)={(rm.sub(1).abs() <= .005).mean():.1%}, median ratio {rm.median():.4f}")

    # ── join ──
    g = pluto.merge(vals, left_on="BBL", right_on="bbl", how="left")
    matched = g["bbl"].notna()
    log(f"  PLUTO lots matched to DOF {year}/{period}: {matched.sum():,} / {len(g):,} "
        f"({matched.mean():.2%}); unmatched PLUTO AssessTot ${g.loc[~matched, 'AssessTot'].sum() / 1e9:,.2f}B "
        f"(exempt ${g.loc[~matched, 'ExemptTot'].sum() / 1e9:,.2f}B)")
    dof_unmatched = vals[~vals["bbl"].isin(pluto["BBL"])]
    log(f"  DOF lots with no PLUTO polygon: {len(dof_unmatched):,}  market land "
        f"${dof_unmatched.curmktland.sum() / 1e9:,.2f}B  "
        f"(non-exempt: ${dof_unmatched.loc[dof_unmatched.curactextot < dof_unmatched.curacttot, 'curmktland'].sum() / 1e9:,.2f}B)")
    g = g[matched].copy()
    for c in ("curmktland", "curmkttot", "curactland", "curacttot", "curactextot"):
        g[c] = g[c].astype(float)

    # ── exempt filter (nyc.ipynb rule, on DOF fields) ──
    fully_exempt = ((g.curactextot >= g.curacttot) & (g.curacttot > 0)) \
        | ((g.curacttot == 0) & (g.curactland == 0)) | (g.curmkttot <= 0)
    old_rule = ((g.ExemptTot >= g.AssessTot) & (g.AssessTot > 0)) | ((g.AssessTot == 0) & (g.AssessLand == 0))
    log(f"  fully exempt (DOF {year}/{period}): {fully_exempt.sum():,}   "
        f"(PLUTO rule on same lots: {old_rule.sum():,}; disagree: {(fully_exempt != old_rule).sum():,})")
    g = g[~fully_exempt].copy()

    # ── values ──
    g["land_value"] = g["curmktland"]
    g["full_market_value"] = g["curmkttot"]
    g["improvement_value"] = (g["full_market_value"] - g["land_value"]).clip(lower=0)
    g["assessed_land_value"] = g["curactland"]
    g["assessed_total_value"] = g["curacttot"]
    g["tax_class"] = g["curtaxclass"].astype(str)
    g["REALLANDVA"] = g["land_value"]
    g["REALIMPROV"] = g["improvement_value"]
    g["area_sqft"] = g["LotArea"].where(g["LotArea"] > 0, g["land_area"].where(g["land_area"] > 0))
    for src, dst in (("full_market_value", "full_market_value_per_sqft"),
                     ("land_value", "land_value_per_sqft"),
                     ("improvement_value", "improvement_value_per_sqft"),
                     ("REALLANDVA", "REALLANDVA_per_sqft"), ("REALIMPROV", "REALIMPROV_per_sqft"),
                     ("full_market_value", "TLLDIMPROV_per_sqft")):
        g[dst] = g[src] / g["area_sqft"]
    g["exemption_flag"] = 0
    g["property_land_use_category"] = [categorize(lu, bc) for lu, bc in zip(g["LandUse"], g["BldgClass"])]
    g["property_land_use_refined"] = refined(g["property_land_use_category"], g["land_value"],
                                             g["improvement_value"])
    g = add_improvement_ratio_fields(g, land_col="land_value", improvement_col="improvement_value")
    g["bbl"] = g["BBL"].astype(str)
    g["link"] = ("https://a836-acris.nyc.gov/bblsearch/bblsearch.asp?borough="
                 + g["Borough"].map(BORO_CODE).astype(str) + "&block=" + g["Block"].astype(str)
                 + "&lot=" + g["Lot"].astype(str))

    log("  category:\n" + g["property_land_use_category"].value_counts().to_string())
    log("  refined:\n" + g["property_land_use_refined"].value_counts(dropna=False).to_string())
    log("  land/total by tax class:\n" + (g.groupby("tax_class")[["land_value", "full_market_value"]].sum()
                                          .assign(land_share=lambda d: d.land_value / d.full_market_value)
                                          .to_string()))

    cols = [
        "geometry", "bbl", "tax_class", "exemption_flag", "property_land_use_category",
        "property_land_use_refined", "full_market_value", "full_market_value_per_sqft", "land_value",
        "land_value_per_sqft", "improvement_value", "improvement_value_per_sqft",
        "assessed_land_value", "assessed_total_value", "REALLANDVA", "REALIMPROV",
        "REALLANDVA_per_sqft", "REALIMPROV_per_sqft", "TLLDIMPROV_per_sqft", "TLLDIMPROV",
        "IMPR_LAND_RATIO", "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link",
    ]
    out = gpd.GeoDataFrame(g[cols].rename(columns={"land_value": "current_full_land_value"}),
                           geometry="geometry", crs=pluto.crs or "EPSG:4326")
    bad = ~out.geometry.is_valid
    if bad.any():
        out.loc[bad, "geometry"] = out.loc[bad, "geometry"].buffer(0)
    out = out[out.geometry.notna() & ~out.geometry.is_empty]
    if out.crs.to_epsg() != 4326:
        out = out.to_crs(4326)
    assert not out["bbl"].duplicated().any()

    log(f"  OUTPUT {len(out):,} parcels  land ${out.current_full_land_value.sum() / 1e9:,.1f}B  "
        f"total ${out.full_market_value.sum() / 1e9:,.1f}B  "
        f"assessed land ${out.assessed_land_value.sum() / 1e9:,.1f}B")
    out.to_parquet(OUT, index=False)
    out.to_parquet(DATA_DIR / f"nyc-ny-parcels_{datetime.now():%Y_%m_%d}.parquet", index=False)
    log(f"WROTE {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
