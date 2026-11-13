"""Shared builder for cities on the Tarrant Appraisal District (TAD) roll.

Arlington, Fort Worth and the Tarrant part of Grand Prairie all come from the same two
automated TAD sources. This module is that pipeline; each city's run_<city>.py is a small
TadCity config plus build(cfg).

Sources (both automated, no manual download):
- Geometry + nightly roll attributes: the City of Fort Worth's public copy of the TAD parcel
  layer (Parcels_Public_Vview). It also carries the Denton / Parker / Wise / Johnson County parts
  of Fort Worth, loaded from those appraisal districts (2026 values).
    https://services5.arcgis.com/3ddLCBXe1bRt7mzj/arcgis/rest/services/Parcels_Public_Vview/FeatureServer/0
- TAD's own 2026 PropertyData export (tad.org/resources/data-downloads, pipe-delimited, refreshed
  weekly). ~12.7k Tarrant accounts in the layer - mostly commercial, much of downtown Fort Worth -
  carry only a total market value with no land/improvement split, land-use code or acreage. Those
  are filled from the export by account number (the export agrees with the layer on 99.7% of land
  values where both have one). Downloaded on first run to data/jurisidictions/data/tarrant/.

Pipeline (see the add-city skill §10 for the reasoning behind each step):
  pull & cache -> counties/tax year -> drop no-roll polygons -> fill gaps from the TAD export ->
  city-limits clip (+ taxing-unit cross-check) -> per-record class & exemptions -> stacked
  records summed -> account dedup -> condo / footprint-lot developments onto their land ->
  drop leftover §23.18 common areas -> canonical fields (LAND_ACRE trusted within 25% of the
  polygon) -> export -> smoke alarms -> singularity sweep (CSV for review).
"""
from __future__ import annotations

import io
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
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
                       categorize_sptb, clip_centroid_within, dedup_accounts, fetch_arcgis_geojson,
                       fix_geometry, log, merge_condo_stubs, merge_stacked_footprints,
                       owner_is_institutional, owner_is_public, road_row_polygons)

PARCELS_LAYER = ("https://services5.arcgis.com/3ddLCBXe1bRt7mzj/arcgis/rest/services/"
                 "Parcels_Public_Vview/FeatureServer/0")
TAD_EXPORT_URL = "https://www.tad.org/content/data-download/PropertyData%28Delimited%29.ZIP"
TAD_EXPORT_DIR = ROOT / "data" / "jurisidictions" / "data" / "tarrant"

# Public owners common to every Tarrant city. Specific names only: "TARRANT COUNTY" alone also
# matches private owners (TARRANT COUNTY PROPERTIES LLC, TARRANT COUNTY FARM BUREAU).
TARRANT_PUBLIC = ["TARRANT COUNTY OF", "TARRANT COUNTY COLLEGE", "TARRANT COUNTY JUNIOR COLLEGE",
                  "TARRANT COUNTY HOSPITAL", "MHMR OF TARRANT", "TARRANT REGIONAL WATER",
                  "TRINITY RIVER AUTHORITY", "TEXAS STATE OF", "MUNICIPAL MANAGEMENT DISTRICT",
                  "UNITED STATES POSTAL"]


def tad_link(acct: str) -> str:
    return "https://www.tad.org/property?account=" + acct


# Per-county property pages. TAD's per-account URL is verified (2026-11-13); Denton/Parker per-parcel
# URL patterns are unverified, so those
# link to the district's property search instead of guessing.
COUNTY_LINKS: dict[str, Callable[[str], str]] = {
    "Tarrant": tad_link,
    "Denton": lambda acct: "https://www.dentoncad.com/property-search",
    "Parker": lambda acct: "https://www.parkercad.org/",
    "Wise": lambda acct: "https://www.wisecad.org/",
    "Johnson": lambda acct: "https://www.johnsoncad.com/",
}


@dataclass
class TadCity:
    city: str                                   # registry key, e.g. "fortworth"
    state: str
    load_boundary: Callable[[Path], object]     # (data_dir) -> shapely geometry, EPSG:4326
    city_codes: dict[str, set[str]]             # county -> taxing-unit codes for this city
    public_names: list[str] = field(default_factory=list)
    counties: tuple[str, ...] = ("Tarrant",)
    expected_tax_year: str = "2026"


def load_tad_export(year: str = "2026") -> pd.DataFrame:
    """TAD PropertyData export (real + commercial accounts), cached as parquet."""
    TAD_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    cache = TAD_EXPORT_DIR / f"tad-propertydata-{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    zpath = TAD_EXPORT_DIR / f"PropertyData_Delimited_{year}.zip"
    if not zpath.exists():
        log(f"Downloading TAD PropertyData export -> {zpath.name}")
        urllib.request.urlretrieve(TAD_EXPORT_URL, zpath)
    keep = ["RP", "Appraisal_Year", "Account_Num", "Owner_Name", "State_Use_Code",
            "LegalDescription", "City", "Land_Value", "Improvement_Value", "Total_Value",
            "Land_Acres", "Land_SqFt", "Appraised_Value"]
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
        d = pd.read_csv(io.TextIOWrapper(z.open(name), encoding="latin-1"), sep="|", usecols=keep,
                        dtype=str, quoting=3)
    d = d[d["RP"].isin(["R", "C"])].copy()
    years = d["Appraisal_Year"].value_counts().to_dict()
    assert set(years) == {year}, f"TAD export years {years}"
    for c in ["Land_Value", "Improvement_Value", "Total_Value", "Land_Acres", "Land_SqFt", "Appraised_Value"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["acct"] = d["Account_Num"].str.strip().str.zfill(8)
    d = d.drop_duplicates("acct")
    d.drop(columns=["Owner_Name"]).to_parquet(cache, index=False)
    return d.drop(columns=["Owner_Name"])


def fill_from_tad_export(raw: gpd.GeoDataFrame, year: str) -> gpd.GeoDataFrame:
    """Fill Tarrant rows missing LAND_VAL / IMPR_VAL / STATE_USE_CODE / LAND_ACRE from the export."""
    tarrant = raw["COUNTYNAME"].astype(str).str.strip().eq("Tarrant")
    gap = tarrant & raw["LAND_VAL"].isna()
    if not gap.any():
        log("TAD export gap-fill: no gaps")
        return raw
    tad = load_tad_export(year).set_index("acct")
    hit = gap & raw["acct"].isin(tad.index)
    t = tad.loc[raw.loc[hit, "acct"]]
    raw.loc[hit, "LAND_VAL"] = t["Land_Value"].values
    raw.loc[hit, "IMPR_VAL"] = t["Improvement_Value"].values
    raw.loc[hit, "STATE_USE_CODE"] = raw.loc[hit, "STATE_USE_CODE"].fillna(
        pd.Series(t["State_Use_Code"].values, index=raw.index[hit]))
    acres = t["Land_Acres"].where(t["Land_Acres"] > 0, t["Land_SqFt"] / SQFT_PER_ACRE).values
    raw.loc[hit, "LAND_ACRE"] = raw.loc[hit, "LAND_ACRE"].fillna(pd.Series(acres, index=raw.index[hit]))
    mv = raw.loc[hit, "MARKET_VALUE"]
    tot = pd.Series(t["Total_Value"].values, index=raw.index[hit])
    agree = (mv.sub(tot).abs() <= 1).mean()
    log(f"TAD export gap-fill: {int(gap.sum()):,} Tarrant rows lacked a land/improvement split; "
        f"filled {int(hit.sum()):,} (export total == layer market on {agree:.1%}); "
        f"still missing {int((gap & ~hit).sum()):,}")
    return raw


def build(cfg: TadCity) -> None:
    CITY, STATE = cfg.city, cfg.state
    DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / CITY
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Boundary + parcels ─────────────────────────────────────────────────
    boundary = cfg.load_boundary(DATA_DIR)
    # Spatial pre-filter = the city outline buffered ~200 m and simplified (not its bounding
    # box, which for Fort Worth is most of Tarrant County). The exact clip happens in step 2.
    pre = gpd.GeoSeries([boundary], crs=4326).to_crs(3857).buffer(200).simplify(50).to_crs(4326).iloc[0]
    raw = fetch_arcgis_geojson(PARCELS_LAYER, polygon=pre,
                               cache=DATA_DIR / f"{CITY}-{STATE}-geometry.parquet")
    log(f"Raw rows in envelope: {len(raw):,}")
    raw["COUNTYNAME"] = raw["COUNTYNAME"].astype(str).str.strip()
    raw = raw[raw["COUNTYNAME"].isin(cfg.counties)].copy()
    years = raw.groupby("COUNTYNAME")["TAX_YEAR"].apply(lambda s: s.dropna().astype(str).value_counts().to_dict())
    log(f"TAX_YEAR by county: {years.to_dict()}")
    assert set(raw["TAX_YEAR"].dropna().astype(str)) == {cfg.expected_tax_year}, "unexpected tax years"
    # Polygons whose account has no roll record carry no value, class or year. They cannot be
    # mapped as land value; kept as blockers for the condo gap fill, then dropped.
    no_roll = raw["TAX_YEAR"].isna()
    off_roll = raw[no_roll].copy()
    raw = raw[~no_roll].copy()
    log(f"Dropped {int(no_roll.sum()):,} polygons with no roll record")

    raw["acct"] = raw["ACCOUNT"].astype(str).str.strip()
    raw = raw[raw["acct"].ne("") & raw["acct"].ne("None")]
    for c in ["LAND_VAL", "IMPR_VAL", "MARKET_VALUE", "LAND_ACRE"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = fix_geometry(raw)
    raw = fill_from_tad_export(raw, cfg.expected_tax_year)

    # ── 2. City limits (centroid within), cross-checked with the taxing unit ─
    parcel = clip_centroid_within(raw, boundary)
    codes = raw["CITY"].fillna("").astype(str).str.split(",").apply(lambda xs: {x.strip() for x in xs})
    coded = pd.Series([bool(c & cfg.city_codes.get(cty, set())) for c, cty in zip(codes, raw["COUNTYNAME"])],
                      index=raw.index)
    inside_ids, coded_ids = set(parcel["acct"]), set(raw.loc[coded, "acct"])
    log(f"City-limits clip -> {len(parcel):,} rows ({parcel['acct'].nunique():,} accounts); "
        f"by county {parcel['COUNTYNAME'].value_counts().to_dict()}; taxing-unit-coded accounts: {len(coded_ids):,}")
    log(f"  in boundary but not coded: {len(inside_ids - coded_ids):,} | "
        f"coded but outside boundary: {len(coded_ids - inside_ids):,}")

    # ── 3. Per-record classification + exemptions ────────────────────────────
    for c in ["APPRAISED_VALUE", "CITY_TAXABLE_VALUE"]:
        parcel[c] = pd.to_numeric(parcel[c], errors="coerce")
    parcel["state_class"] = parcel["STATE_USE_CODE"].fillna("").astype(str).str.strip().str.upper()
    parcel["PROPERTY_CATEGORY"] = parcel["state_class"].apply(categorize_sptb)
    log(f"state_class: {parcel['state_class'].value_counts().head(25).to_dict()}")
    # Totally exempt property. TAD rarely codes it as state class X; it keeps the use code and
    # zeroes the taxable value, so $0 city taxable value on an appraised parcel is the signal.
    # Homes with $0 taxable are mostly 100% disabled-veteran homesteads (personal exemptions on
    # private homes), so residential classes also need an institutional owner. Rows without a
    # real APPRAISED_VALUE (Parker County's load) fall back to class X + public owner names.
    by_class = parcel["PROPERTY_CATEGORY"].eq("Exempt")
    by_owner = owner_is_public(parcel["OWNER_NAME"], [*TARRANT_PUBLIC, *cfg.public_names])
    taxable0 = parcel["CITY_TAXABLE_VALUE"].eq(0) & parcel["APPRAISED_VALUE"].gt(0)
    residential = parcel["state_class"].str.match(r"^[AB]")
    by_value = taxable0 & (~residential | owner_is_institutional(parcel["OWNER_NAME"]))
    exempt = by_class | by_owner | by_value
    log(f"Exempt: state class X {int(by_class.sum()):,} | public owner {int(by_owner.sum()):,} | "
        f"$0 city taxable {int(by_value.sum()):,} | total {int(exempt.sum()):,}")
    log("  public-owner matches without $0 taxable (audit): "
        f"{parcel.loc[by_owner & ~taxable0, 'OWNER_NAME'].value_counts().head(25).to_dict()}")
    log(f"  $0-taxable homes kept as private (personal exemptions): "
        f"{int((taxable0 & residential & ~exempt).sum()):,}")
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
    ex["land_area_sqft"] = np.where(use_reported, ex["reported_sqft"], ex["geom_area_sqft"])
    ex["area_source"] = np.where(use_reported, "reported", "gis")
    ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
    ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)
    ex["full_market_value"] = ex["MARKET_VALUE"].where(ex["MARKET_VALUE"] > 0,
                                                       ex["land_value"] + ex["improvement_value"])
    den = ex["land_area_sqft"].replace(0, np.nan)
    ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
    ex["land_value_per_sqft"] = ex["land_value"] / den
    ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
    ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")
    ex["link"] = [COUNTY_LINKS.get(c, tad_link)(a) for c, a in zip(ex["COUNTYNAME"], ex["acct"].astype(str))]

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
        if cat in ("Single Family", "Vacant Residential"):
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
