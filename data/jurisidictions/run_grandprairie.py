#!/usr/bin/env python3
"""
Build Grand Prairie, TX's canonical parcel parquet: the whole city, within city limits, across
Dallas, Tarrant and Ellis Counties (plus a 0.16 km2 sliver of Johnson County with no parcels of
its own in the sources below).

Sources, one per county, combined into one tx_pipeline run:
- Dallas County (139 km2, most of the city): Dallas Central Appraisal District's 2026 certified
  data files (DCAD2026_CURRENT: certified values + supplements, files of 2026-11-09) joined on
  GIS_PARCEL_ID to DCAD's parcel polygons (PARCEL_GEOM, 2026-11). See dcad_county.py. Manual
  downloads from dallascad.org (not reachable from the build host), placed in
    data/jurisidictions/data/grandprairie/dcad/
  City of Grand Prairie accounts: CITY_JURIS_DESC == GRAND PRAIRIE.
- Tarrant County (62 km2): the Fort Worth-hosted TAD layer, filled from TAD's 2026 export, as
  for Arlington and Fort Worth (tad_city.py). TAD city code 38.
- Ellis County (20 km2): Ellis Appraisal District's 2026 certified export (PACS 8.0.33,
  2026-07-30) joined on prop_id to the district's parcel shapefile (pacs_city.py). Entity 338.
  Manual downloads placed in data/jurisidictions/data/grandprairie/ellis/.
City limits: Census TIGER/Line 2025 place 4830464.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

import pacs_city as pc
import tad_city as tc
from dcad_county import dcad_link, dcad_rows
from tx_common import log
from tx_pipeline import process

CITY, STATE = "grandprairie", "tx"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / CITY
TIGER = ROOT / "data" / "jurisidictions" / "data" / "elpaso" / "tl_2025_48_place.zip"
YEAR = "2026"


def load_boundary(data_dir: Path):
    cache = data_dir / f"{CITY}-{STATE}-boundary.parquet"
    if not cache.exists():
        g = gpd.read_file(f"zip://{TIGER}")
        g[g["GEOID"] == "4830464"].to_crs(4326).to_parquet(cache)
    return gpd.read_parquet(cache).union_all()


ellis_link = lambda prop_id: f"https://esearch.elliscad.com/Property/View/{prop_id}?year={YEAR}"  # noqa: E731

PUBLIC = ["CITY OF GRAND PRAIRIE", "GRAND PRAIRIE CITY OF", "GRAND PRAIRIE INDEPENDENT",
          "GRAND PRAIRIE ISD", "GRAND PRAIRIE HOUSING", "ARLINGTON INDEPENDENT", "MANSFIELD INDEPENDENT",
          "MIDLOTHIAN INDEPENDENT", "CEDAR HILL INDEPENDENT", "DALLAS COUNTY OF", "COUNTY OF DALLAS",
          "DALLAS CO COMMUNITY COLLEGE", "DALLAS COLLEGE", "ELLIS COUNTY OF", "COUNTY OF ELLIS",
          "TEXAS DEPARTMENT OF TRANSPORTATION", "STATE OF TEXAS", "UNITED STATES OF AMERICA",
          "TRINITY RIVER AUTHORITY", "TARRANT REGIONAL WATER"]

TAD = tc.TadCity(city=CITY, state=STATE, load_boundary=load_boundary,
                 city_codes={"Tarrant": {"38", "038"}}, counties=("Tarrant",))

ELLIS_CITY = pc.PacsCity(city=CITY, state=STATE, county="Ellis", gis_url=None, export_url="",
                         tiger_geoid="4830464", city_entity="338", link=ellis_link)
ELLIS = pc.PacsCounty(county="Ellis", gis_layer=None, export_file="ellis/ellis-certified-export.zip",
                      city_entity="338", link=ellis_link, gis_zip="ellis/ellis_parcels.zip",
                      gis_id_field="pid")

LINKS = {"Tarrant": tc.tad_link, "Dallas": dcad_link, "Ellis": ellis_link}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary(DATA_DIR)

    log("=== Dallas County (DCAD) ===")
    d_raw, d_off, d_coded, d_ex = dcad_rows(boundary, DATA_DIR, city_juris="GRAND PRAIRIE", year=YEAR)
    log("=== Tarrant County (TAD) ===")
    t_raw, t_off, t_coded, t_ex = tc.tad_rows(TAD, boundary, DATA_DIR)
    log("=== Ellis County (Ellis CAD) ===")
    e_raw, e_off, e_coded = pc._county_rows(
        ELLIS_CITY, ELLIS, boundary, DATA_DIR,
        gcache=DATA_DIR / f"{CITY}-{STATE}-ellis-geometry.parquet",
        export_zip=DATA_DIR / ELLIS.export_file,
        roll_cache=DATA_DIR / f"{CITY}-{STATE}-ellis-roll-{YEAR}.parquet",
        export_url=None, gis_url=None, prefix="ELLIS")
    e_ex = lambda p: p["roll_exempt"].fillna(False).astype(bool)  # noqa: E731

    raws = [d_raw, t_raw, e_raw]
    raw = gpd.GeoDataFrame(pd.concat(raws, ignore_index=True), geometry="geometry", crs=4326)
    off_roll = gpd.GeoDataFrame(pd.concat([d_off, t_off, e_off], ignore_index=True), geometry="geometry", crs=4326)
    coded = pd.concat([d_coded, t_coded, e_coded], ignore_index=True)

    by_county = {"Dallas": d_ex, "Tarrant": t_ex, "Ellis": e_ex}

    def exempt_fn(p: gpd.GeoDataFrame) -> pd.Series:
        out = pd.Series(False, index=p.index)
        for county, fn in by_county.items():
            sub = p["COUNTYNAME"].eq(county)
            if sub.any():
                out[sub] = fn(p[sub]).reindex(p.index[sub]).fillna(False).astype(bool)
        return out

    process(raw, off_roll, boundary, city=CITY, state=STATE, data_dir=DATA_DIR,
            public_names=[*tc.TARRANT_PUBLIC, *PUBLIC], exempt_fn=exempt_fn,
            link_fn=lambda county, acct: LINKS[county](acct), coded=coded)


if __name__ == "__main__":
    main()
