"""Shared Texas parcel pipeline: everything after a source's roll has been loaded.

Each source module (tad_city.py for the Tarrant roll, pacs_city.py for PACS certified
exports + appraisal-district GIS) prepares one GeoDataFrame of roll records in EPSG:4326 with
standard columns, then calls process(). Standard input columns:
  acct (unique key; county-prefixed where accounts repeat across counties), COUNTYNAME,
  LAND_VAL, IMPR_VAL, MARKET_VALUE, LAND_ACRE (appraiser acreage or NaN), STATE_USE_CODE,
  OWNER_NAME, PARCEL_LEGAL_DESCRIPTION, SITUS_ADDR, geometry.
The steps and the reasoning behind each are in the add-city skill (§2, §3, §5, §6, §10).
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
sys.path.append(str(Path(__file__).resolve().parent))
from parcel_calculations import (add_improvement_ratio_fields, classify_property_refined,  # noqa: E402
                                 gis_area_sqft, land_value_singularities)
from tx_common import (CONDO_EXCLUDE_RE, CONDO_LEGAL_RE, NON_LAND_CATEGORIES, SQFT_PER_ACRE,  # noqa: E402
                       categorize_sptb, clip_centroid_within, dedup_accounts, fix_geometry, log,
                       merge_condo_stubs, merge_stacked_footprints, owner_is_public,
                       road_row_polygons)


def process(raw: gpd.GeoDataFrame, off_roll: gpd.GeoDataFrame, boundary, *, city: str, state: str,
            data_dir: Path, public_names: list[str] = (),
            exempt_fn: Callable[[gpd.GeoDataFrame], pd.Series] | None = None,
            link_fn: Callable[[str, str], str] = lambda county, acct: "",
            coded: pd.Series | None = None, home_county: str | None = None) -> gpd.GeoDataFrame:
    """Run the shared pipeline and write <city>-<state>-parcels.parquet (+ singularities CSV).

    `home_county` is the county whose rows are whole parcels; rows from other counties may be
    clipped by the source layer and get prorated (Fort Worth's Denton/Parker parts).
    """
    CITY, STATE, DATA_DIR = city, state, data_dir
    # Agricultural (open-space) tracts: the layer records LAND_VAL = 0 and puts the land's
    # market value in AG_VAL / MARKET_VALUE (TAD's export confirms land = market for them).
    # The map shows market land value, not the productivity value, so land = market - impr.
    ag0 = raw["LAND_VAL"].eq(0) & raw["MARKET_VALUE"].gt(raw["IMPR_VAL"].fillna(0))
    raw.loc[ag0, "LAND_VAL"] = raw.loc[ag0, "MARKET_VALUE"] - raw.loc[ag0, "IMPR_VAL"].fillna(0)
    log(f"Land = market - improvements on {int(ag0.sum()):,} rows recorded with $0 land "
        f"(by county {raw.loc[ag0, 'COUNTYNAME'].value_counts().to_dict()})")
    # Possessory interests are leaseholds on exempt (usually public) land, drawn as marker
    # polygons; not fee land (Harrisonburg precedent: leaseholds on exempt land are left off).
    pi = raw["PARCEL_LEGAL_DESCRIPTION"].fillna("").str.upper().str.contains("POSSESSORY INTEREST")
    log(f"Dropping {int(pi.sum()):,} possessory-interest (leasehold) records")
    raw = raw[~pi].copy()

    # ── 2. City limits (centroid within), cross-checked with the taxing unit ─
    parcel = clip_centroid_within(raw, boundary)
    log(f"City-limits clip -> {len(parcel):,} rows ({parcel['acct'].nunique():,} accounts); "
        f"by county {parcel['COUNTYNAME'].value_counts().to_dict()}")
    if coded is not None:
        inside_ids, coded_ids = set(parcel["acct"]), set(raw.loc[coded.reindex(raw.index, fill_value=False), "acct"])
        log(f"  taxing-unit-coded accounts: {len(coded_ids):,}; in boundary but not coded: "
            f"{len(inside_ids - coded_ids):,} | coded but outside boundary: {len(coded_ids - inside_ids):,}")

    # ── 3. Per-record classification + exemptions ────────────────────────────
    parcel["state_class"] = parcel["STATE_USE_CODE"].fillna("").astype(str).str.strip().str.upper()
    parcel["PROPERTY_CATEGORY"] = parcel["state_class"].apply(categorize_sptb)
    log(f"state_class: {parcel['state_class'].value_counts().head(25).to_dict()}")
    # Totally exempt = SPTB class X, OR a named public owner, OR the source's own signal
    # (`exempt_fn`: e.g. TAD's $0 city taxable value, or PACS EX-* exemption flags).
    by_class = parcel["PROPERTY_CATEGORY"].eq("Exempt")
    by_owner = owner_is_public(parcel["OWNER_NAME"], list(public_names))
    by_source = exempt_fn(parcel) if exempt_fn is not None else pd.Series(False, index=parcel.index)
    exempt = by_class | by_owner | by_source
    log(f"Exempt: state class X {int(by_class.sum()):,} | public owner {int(by_owner.sum()):,} | "
        f"source signal {int(by_source.sum()):,} | total {int(exempt.sum()):,}")
    log("  public-owner matches without the source signal (audit): "
        f"{parcel.loc[by_owner & ~by_source, 'OWNER_NAME'].value_counts().head(25).to_dict()}")
    parcel["exemption_flag"] = exempt.astype(int)
    parcel = parcel[parcel["exemption_flag"].eq(0)].copy()
    dropped = parcel["PROPERTY_CATEGORY"].isin(NON_LAND_CATEGORIES)
    log(f"Dropping non-land categories: {parcel.loc[dropped, 'PROPERTY_CATEGORY'].value_counts().to_dict()}")
    parcel = parcel[~dropped].copy()
    nov = parcel["LAND_VAL"].isna()
    if nov.any():
        log(f"Dropping {int(nov.sum()):,} rows still without a land value "
            f"(by county {parcel.loc[nov, 'COUNTYNAME'].value_counts().to_dict()})")
        parcel = parcel[~nov].copy()

    # ── 4. Stacked records summed onto their footprint, then account dedup ───
    # An account drawn both on a polygon of its own AND on a polygon shared with other accounts
    # (El Paso: lots 1/5/7/9 of Cimarron Canyon blk 15 attached to one polygon, lot 7 also drawn
    # alone) would be counted twice. Drop its row on the shared polygon; keep its own.
    gk = parcel.geometry.apply(lambda g: g.wkb_hex if g is not None else "")
    shared = gk.map(parcel.groupby(gk)["acct"].nunique()) > 1
    has_own = parcel["acct"].map(parcel.loc[~shared].groupby("acct").size()).fillna(0) > 0
    drop_shared = shared & has_own
    if drop_shared.any():
        log(f"Dropping {int(drop_shared.sum()):,} rows on shared polygons for accounts that also "
            f"have their own polygon")
        parcel = parcel[~drop_shared].copy()
    parcel = merge_stacked_footprints(parcel, "acct", ["LAND_VAL", "IMPR_VAL", "MARKET_VALUE"],
                                      area_col="LAND_ACRE", dup_subset=["OWNER_NAME"])
    parcel = dedup_accounts(parcel, "acct", ["LAND_VAL", "IMPR_VAL", "MARKET_VALUE", "STATE_USE_CODE"])
    assert not parcel["acct"].duplicated().any()
    roads_cache = DATA_DIR / f"{CITY}-{STATE}-roads.parquet"
    if roads_cache.exists():
        roads = gpd.read_parquet(roads_cache)
    else:
        roads = road_row_polygons(boundary.bounds)
        roads.to_parquet(roads_cache)
    blockers = gpd.GeoDataFrame(pd.concat([raw[["acct", "geometry"]], off_roll[["geometry"]]],
                                          ignore_index=True), geometry="geometry", crs=4326)
    parcel = merge_condo_stubs(parcel, "PARCEL_LEGAL_DESCRIPTION", ["LAND_VAL", "IMPR_VAL", "MARKET_VALUE"],
                               area_col="LAND_ACRE", blockers=blockers, roads=roads)
    # Remaining association common areas (Texas Tax Code §23.18: nominal $1, their value is in
    # the members' lots) are not land-value parcels; shipped they render as $0 land and count as
    # "Vacant". Drop them like exempt land.
    legal = parcel["PARCEL_LEGAL_DESCRIPTION"].fillna("").str.upper()
    nominal_common = legal.str.contains(r"COMMON|OPEN SPACE", regex=True) \
        & parcel["LAND_VAL"].fillna(0).le(1000) & parcel["land_source"].isna()
    log(f"Dropping {int(nominal_common.sum()):,} nominal-value association common areas "
        f"({parcel.loc[nominal_common].to_crs(parcel.estimate_utm_crs()).area.sum() / 4046.86:,.0f} acres)")
    parcel = parcel[~nominal_common].copy()
    log(f"After stack merge + dedup + condo merge -> {len(parcel):,}")

    ex = parcel.copy()
    ex["property_land_use_category"] = ex["PROPERTY_CATEGORY"]
    ex["land_value"] = ex["LAND_VAL"].astype(float)
    ex["improvement_value"] = ex["IMPR_VAL"].astype(float)
    ex["property_land_use_refined"] = classify_property_refined(ex, fetch_footprints=False)

    # ── 5. Canonical fields — LAND_ACRE trusted within 25% of the polygon ─────
    log("Computing GIS areas...")
    ex["geom_area_sqft"] = ex["geometry"].apply(gis_area_sqft)
    ex.loc[ex["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
    ex["reported_sqft"] = ex["LAND_ACRE"] * SQFT_PER_ACRE
    ex.loc[~(ex["reported_sqft"] >= 1), "reported_sqft"] = np.nan
    ratio = (ex["geom_area_sqft"] / ex["reported_sqft"]).replace([np.inf, -np.inf], np.nan)
    log(f"Area cross-check GIS / LAND_ACRE: median {ratio.median():.3f}, "
        f"off by >25%: {int((ratio.sub(1).abs() > .25).sum()):,} of {int(ratio.notna().sum()):,}")
    # LAND_ACRE agrees with the polygon for the vast majority, but on a few percent it holds
    # square feet, tiny fractions or a partial area; trust it only within 25% of the polygon.
    use_reported = (ex["reported_sqft"] > 0) & (ratio.sub(1).abs() <= 0.25)
    # Polygon much SMALLER than a sane roll acreage (1/1000 < drawn/roll < 2/3) on a whole-parcel
    # source: the polygon is a fragment (El Paso prop 94560: 3.52 ac on the roll, 2,359 sqft
    # drawn -> $226/sqft), so the appraiser's land area is the denominator. Below 1/1000 is a
    # units mix-up (square feet in the acres field) and the polygon is used. Clipped county parts
    # (not home_county) are prorated below instead.
    home = ex["COUNTYNAME"].eq(home_county) if home_county else pd.Series(True, index=ex.index)
    # ... but only if that acreage is the parcel's OWN: El Paso repeats a parent tract's acreage
    # (12.129 ac) on every lot of a subdivision, which would spread each lot over 12 acres.
    # A repeated acreage is a tract figure when MOST accounts carrying it look like fragments
    # (a real repeated lot size mostly matches its polygons).
    cand = home & (ratio < 1 / 1.5) & (ratio > 1 / 1000) & (ex["reported_sqft"] > 0)
    key = ex["reported_sqft"].round(0)
    rep_n = key.map(key.value_counts())
    cand_share = cand.groupby(key).transform("mean")
    fragment = cand & ~((rep_n >= 3) & (cand_share >= 0.5))
    use_reported = use_reported | fragment
    log(f"Polygon is a fragment of the roll's land area: {int(fragment.sum()):,} parcels use the appraiser's area")
    ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["geom_area_sqft"])
    ex["area_source"] = np.where(use_reported, "reported", "gis")
    # Non-Tarrant rows (Denton/Parker/Wise/Johnson parts of Fort Worth) are drawn only where
    # the City of Fort Worth's layer covers them, so a tract can be clipped to a sliver while
    # its value is the whole tract's (Walsh Ranches: 60.52 ac on the roll, 0.16 ac drawn). Prorate
    # land and improvement value by the drawn share when the roll's acreage exceeds the polygon
    # by > 50%. (Tarrant rows are whole parcels; there a big gap means a bad LAND_ACRE instead.)
    clipped = (ex["COUNTYNAME"].ne(home_county) if home_county else pd.Series(False, index=ex.index)) \
        & (ratio < 1 / 1.5)
    share = ratio.where(clipped, 1.0)
    for c in ["land_value", "improvement_value", "MARKET_VALUE"]:
        ex[c] = ex[c] * share
    ex.loc[clipped, "area_source"] = "gis_prorated"
    log(f"Prorated {int(clipped.sum()):,} non-Tarrant tracts clipped by the source layer "
        f"(median drawn share {ratio[clipped].median():.2f})")
    ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
    ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)
    ex["full_market_value"] = ex["MARKET_VALUE"].where(ex["MARKET_VALUE"] > 0,
                                                       ex["land_value"] + ex["improvement_value"])
    den = ex["land_area_sqft"].replace(0, np.nan)
    ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
    ex["land_value_per_sqft"] = ex["land_value"] / den
    ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
    ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")
    ex["link"] = [link_fn(c, a.split(":")[-1]) for c, a in zip(ex["COUNTYNAME"], ex["acct"].astype(str))]

    # ── 6. Export ─────────────────────────────────────────────────────────────
    COLUMNS = ["geometry", "exemption_flag", "property_land_use_category", "property_land_use_refined",
               "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
               "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
               "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source",
               "likely_remnant"]
    final = ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"})
    final = gpd.GeoDataFrame(fix_geometry(gpd.GeoDataFrame(final, geometry="geometry", crs=4326)),
                             geometry="geometry", crs="EPSG:4326")
    out = DATA_DIR / f"{CITY}-{STATE}-parcels.parquet"
    final.to_parquet(out, index=False)
    final.to_parquet(DATA_DIR / f"{CITY}-{STATE}-parcels_{datetime.now():%Y_%m_%d}.parquet", index=False)
    log(f"SAVED {out} | rows {len(final):,}")
    log(f"land ${final.current_full_land_value.sum():,.0f}  impr ${final.improvement_value.sum():,.0f}")
    log(f"land by county: {ex.groupby('COUNTYNAME')['land_value'].sum().round(0).to_dict()}")
    log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
    log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")
    log(f"area_source: {final['area_source'].value_counts().to_dict()}")

    # ── §6a smoke alarms ──────────────────────────────────────────────────────
    log("--- condo/stub smoke alarms (skill §6a) ---")
    a = ex["geom_area_sqft"]
    lv = pd.to_numeric(final["land_value_per_sqft"], errors="coerce")
    log(f"  footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
    log(f"  sub-500 / sub-1000 sqft footprints: {int((a < 500).sum()):,} / {int((a < 1000).sum()):,}")
    log(f"  land $/sqft p50/p95/p99/max: ${lv.median():,.2f} / ${lv.quantile(.95):,.2f} / "
        f"${lv.quantile(.99):,.2f} / ${lv.max():,.2f}")
    rp = final.geometry.representative_point()
    vc = (rp.x.round(5).astype(str) + "," + rp.y.round(5).astype(str)).value_counts()
    log(f"  stacked clusters: {int((vc > 1).sum())}, max stack: {int(vc.max())}")
    holes = final.geometry.apply(lambda g: sum(
        len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
    log(f"  parcels with interior rings (holes): {int((holes > 0).sum()):,}")
    log(f"  zero land value (renders as gp-error): "
        f"{int((final['current_full_land_value'].fillna(0) <= 0).sum()):,}")
    log(f"  bounds: {[round(v, 4) for v in final.total_bounds]}")

    # ── Singularity sweep (skill §10 policy) ──────────────────────────────────
    sing = land_value_singularities(final)
    legal_f = ex.loc[sing.index, "PARCEL_LEGAL_DESCRIPTION"].fillna("").str.upper()
    cat_f = ex.loc[sing.index, "property_land_use_category"]

    def _cause(i):
        lg, cat = legal_f[i], cat_f[i]
        if final.loc[i, "likely_remnant"] == 1:
            return "sliver <500 sqft (hidden)"
        if pd.notna(ex.loc[i, "land_source"]):
            if ex.loc[i, "land_source"] in ("common_area_parcel", "nominal_parent"):
                return "development on its common-area land (appraiser value, published)"
            return f"condo, land from {ex.loc[i, 'land_source']} (UNRESOLVED)"
        if re.search(CONDO_LEGAL_RE, lg) and not re.search(CONDO_EXCLUDE_RE, lg):
            if ex.loc[i, "n_records"] > 1:
                return "condo complex on its plat polygon (appraiser value, published)"
            return "condo unit not merged (UNRESOLVED)"
        if re.search(r"TOWNHOME|TOWNHOUSE|TNHSE", lg) or sing.loc[i, "footprint_sqft"] < 2500:
            return "small fee-simple lot (appraiser value, published)"
        if cat in ("Commercial", "Industrial", "Multifamily", "Vacant Commercial"):
            return "commercial/multifamily land (appraiser value, published)"
        if cat in ("Single Family", "Vacant Residential", "Residential Inventory"):
            return "residential lot (appraiser value, published)"
        return "other (review)"

    sing["cause"] = [_cause(i) for i in sing.index]
    sing["account"] = ex.loc[sing.index, "acct"].values
    sing["county"] = ex.loc[sing.index, "COUNTYNAME"].values
    sing["legal"] = legal_f.values
    sing["address"] = ex.loc[sing.index, "SITUS_ADDR"].values
    sing.to_csv(DATA_DIR / f"{CITY}-{STATE}-singularities.csv")
    summary = sing.groupby("cause").agg(n=("ratio", "size"), median_ratio=("ratio", "median"),
                                        max_ratio=("ratio", "max"), land=("land_value", "sum"))
    log(f"Singularity sweep: {len(sing):,} parcels >= 4x neighbours (of {len(final):,}); by cause:\n"
        f"{summary.round(1).to_string()}")
    unresolved = sing["cause"].str.contains("UNRESOLVED|review")
    if unresolved.any():
        log(f"  TO REVIEW: {sing.loc[unresolved, ['account', 'county', 'ratio', 'legal']].head(25).to_string()}")
    log("DONE")
    return final
