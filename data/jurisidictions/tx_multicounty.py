"""Run tx_pipeline for a Texas city built from one or more county sources.

Each source is (county name, rows function, link function). A rows function takes
(boundary, data_dir) and returns (raw, off_roll, coded, exempt_fn) in tx_pipeline's standard
columns (see collin_county.py, denton_county.py, dcad_county.py, tad_city.tad_rows,
pacs_city._county_rows). Exemptions and links are applied per county.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import geopandas as gpd
import pandas as pd

from tx_common import log
from tx_pipeline import process

ROOT = Path(__file__).resolve().parents[2]
TIGER = ROOT / "data" / "jurisidictions" / "data" / "elpaso" / "tl_2025_48_place.zip"


def tiger_boundary(data_dir: Path, city: str, state: str, geoid: str):
    cache = data_dir / f"{city}-{state}-boundary.parquet"
    if not cache.exists():
        g = gpd.read_file(f"zip://{TIGER}")
        g[g["GEOID"] == geoid].to_crs(4326).to_parquet(cache)
    return gpd.read_parquet(cache).union_all()


def build_city(city: str, state: str, geoid: str,
               sources: list[tuple[str, Callable, Callable[[str], str]]],
               public_names: list[str]) -> gpd.GeoDataFrame:
    data_dir = ROOT / "data" / "jurisidictions" / "data" / city
    data_dir.mkdir(parents=True, exist_ok=True)
    boundary = tiger_boundary(data_dir, city, state, geoid)
    raws, offs, codeds, exempt, links = [], [], [], {}, {}
    for county, rows_fn, link_fn in sources:
        log(f"=== {county} County ===")
        raw, off, coded, ex = rows_fn(boundary, data_dir)
        raws.append(raw.reset_index(drop=True))
        offs.append(off)
        codeds.append(coded.reset_index(drop=True))
        exempt[county], links[county] = ex, link_fn
    raw = gpd.GeoDataFrame(pd.concat(raws, ignore_index=True), geometry="geometry", crs=4326)
    off_roll = gpd.GeoDataFrame(pd.concat(offs, ignore_index=True), geometry="geometry", crs=4326)
    coded = pd.concat(codeds, ignore_index=True)

    def exempt_fn(p):
        out = pd.Series(False, index=p.index)
        for county, fn in exempt.items():
            sub = p["COUNTYNAME"].eq(county)
            if sub.any():
                out[sub] = fn(p[sub]).reindex(p.index[sub]).fillna(False).astype(bool)
        return out

    return process(raw, off_roll, boundary, city=city, state=state, data_dir=data_dir,
                   public_names=public_names, exempt_fn=exempt_fn,
                   link_fn=lambda county, acct: links[county](acct), coded=coded)
