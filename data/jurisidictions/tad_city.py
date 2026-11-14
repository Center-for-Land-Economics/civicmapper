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
from tx_pipeline import process  # noqa: E402
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
    # Account numbers are only unique within a county (Denton and Parker both have R000112537):
    # key non-Tarrant rows by county too. Tarrant keeps the bare TAD account (links, export join).
    other = raw["COUNTYNAME"].ne("Tarrant")
    raw.loc[other, "acct"] = raw.loc[other, "COUNTYNAME"] + ":" + raw.loc[other, "acct"]
    for c in ["LAND_VAL", "IMPR_VAL", "MARKET_VALUE", "LAND_ACRE"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = fix_geometry(raw)
    raw = fill_from_tad_export(raw, cfg.expected_tax_year)
    # Denton County rows carry no LAND_ACRE, but tract legals state it (", 88.2523 ACRES").
    # Strict pattern (comma, decimal number, ACRES) so "LOT 6 ACRES 2.473" is not misread.
    den = raw["COUNTYNAME"].eq("Denton") & raw["LAND_ACRE"].isna()
    legal_ac = pd.to_numeric(raw.loc[den, "PARCEL_LEGAL_DESCRIPTION"].fillna("")
                             .str.extract(r",\s*(\d*\.\d+)\s+ACRES", flags=re.I)[0], errors="coerce")
    raw.loc[den, "LAND_ACRE"] = legal_ac
    if den.any():
        log(f"Denton acreage from legal descriptions: {int(legal_ac.notna().sum()):,} of {int(den.sum()):,} rows")
    codes = raw["CITY"].fillna("").astype(str).str.split(",").apply(lambda xs: {x.strip() for x in xs})
    coded = pd.Series([bool(c & cfg.city_codes.get(cty, set())) for c, cty in zip(codes, raw["COUNTYNAME"])],
                      index=raw.index)

    def tad_exempt(parcel: gpd.GeoDataFrame) -> pd.Series:
        """TAD rarely codes totally exempt property as class X; it keeps the use code and zeroes
        the taxable value, so $0 city taxable value on an appraised parcel is the signal. Homes
        with $0 taxable are mostly 100% disabled-veteran homesteads (personal exemptions on
        private homes), so residential classes also need an institutional owner. Rows without a
        real APPRAISED_VALUE (Parker County's load) fall back to class X + public owner names."""
        taxable0 = pd.to_numeric(parcel["CITY_TAXABLE_VALUE"], errors="coerce").eq(0) \
            & pd.to_numeric(parcel["APPRAISED_VALUE"], errors="coerce").gt(0)
        state_class = parcel["STATE_USE_CODE"].fillna("").astype(str).str.strip().str.upper()
        residential = state_class.str.match(r"^[AB]")
        by_value = taxable0 & (~residential | owner_is_institutional(parcel["OWNER_NAME"]))
        log(f"  $0-taxable homes kept as private (personal exemptions): "
            f"{int((taxable0 & residential & ~by_value).sum()):,}")
        return by_value

    process(raw, off_roll, boundary, city=CITY, state=STATE, data_dir=DATA_DIR,
            public_names=[*TARRANT_PUBLIC, *cfg.public_names], exempt_fn=tad_exempt,
            link_fn=lambda county, acct: COUNTY_LINKS.get(county, tad_link)(acct),
            coded=coded, home_county="Tarrant")
