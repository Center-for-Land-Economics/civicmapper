#!/usr/bin/env python3
"""
Build Plano, TX's canonical parcel parquet: the whole city, within city limits (180 km2 in Collin
County, 6.7 km2 in Denton County).

Sources:
- Collin County: Collin CAD's public parcel layer (CCAD Parcel Feature Set), 2026 certified values
  from its prior-year fields (the layer has rolled to the 2027 roll year); see collin_county.py.
  City of Plano: entityCityCode CPL.
- Denton County (the 6.7 km2 strip on the west edge): Denton CAD's 2026 certified export
  (PACS 8.0.33, 2026-07-27; manual download, shared by every Denton County city in
  data/jurisidictions/data/dentoncad/) joined on prop_id to Denton CAD's own certified parcel
  polygons (Parcels_Certified_2026 in its nightly geodatabase, same folder). City of Plano: C29.
City limits: Census TIGER/Line 2025 place 4858016.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

import pacs_city as pc
from collin_county import ccad_link, ccad_rows
from tx_common import log
from tx_pipeline import process

CITY, STATE = "plano", "tx"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / CITY
TIGER = ROOT / "data" / "jurisidictions" / "data" / "elpaso" / "tl_2025_48_place.zip"

PUBLIC = ["CITY OF PLANO", "PLANO CITY OF", "PLANO INDEPENDENT SCHOOL", "PLANO ISD",
          "FRISCO INDEPENDENT SCHOOL", "LEWISVILLE INDEPENDENT", "COLLIN COUNTY OF", "COUNTY OF COLLIN",
          "COLLIN COLLEGE", "COLLIN COUNTY COMMUNITY COLLEGE", "DALLAS AREA RAPID TRANSIT",
          "NORTH TEXAS MUNICIPAL WATER", "NORTH TEXAS TOLLWAY AUTHORITY", "PLANO HOUSING",
          "TEXAS DEPARTMENT OF TRANSPORTATION", "STATE OF TEXAS", "UNITED STATES OF AMERICA"]


DENTON_DIR = ROOT / "data" / "jurisidictions" / "data" / "dentoncad"
DENTON_GDB = DENTON_DIR / "nightly_geodatabase.gdb.zip"
denton_link = lambda prop_id: f"https://denton.prodigycad.com/property-detail/{prop_id}/2026"  # noqa: E731


def denton_geometry(boundary, cache: Path) -> None:
    """Denton CAD's certified 2026 parcel polygons around `boundary` -> cache (EPSG:4326)."""
    if cache.exists():
        return
    src = f"/vsizip/{DENTON_GDB}/nightly_geodatabase.gdb"
    bb = gpd.GeoSeries([boundary], crs=4326).to_crs(2276).buffer(2100).iloc[0]
    g = gpd.read_file(src, layer="Parcels_Certified_2026", bbox=tuple(bb.bounds))
    g = g[g.intersects(bb)][["prop_id", "geometry"]].to_crs(4326)
    g.to_parquet(cache, index=False)


def denton_rows(boundary, city_entity: str):
    """Denton CAD certified 2026 rows near `boundary` -> (raw, off_roll, coded, exempt_fn)."""
    cfg = pc.PacsCity(city=CITY, state=STATE, county="Denton", gis_url=None, export_url="",
                      tiger_geoid="", city_entity=city_entity, link=denton_link)
    cty = pc.PacsCounty(county="Denton", gis_layer=None, export_file="", city_entity=city_entity,
                        link=denton_link)
    gcache = DATA_DIR / f"{CITY}-{STATE}-denton-geometry.parquet"
    denton_geometry(boundary, gcache)
    raw, off, coded = pc._county_rows(
        cfg, cty, boundary, DATA_DIR,
        gcache=gcache,
        export_zip=DENTON_DIR / "denton-2026-certified.zip",
        roll_cache=DENTON_DIR / "denton-roll-2026.parquet",
        export_url=None, gis_url=None, prefix="DENTON")
    return raw, off, coded, (lambda p: p["roll_exempt"].fillna(False).astype(bool))


def load_boundary():
    cache = DATA_DIR / f"{CITY}-{STATE}-boundary.parquet"
    if not cache.exists():
        g = gpd.read_file(f"zip://{TIGER}")
        g[g["GEOID"] == "4858016"].to_crs(4326).to_parquet(cache)
    return gpd.read_parquet(cache).union_all()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary()
    log("=== Collin County (CCAD) ===")
    c_raw, c_off, c_coded, c_ex = ccad_rows(boundary, DATA_DIR, city_code="CPL")
    log("=== Denton County (Denton CAD) ===")
    d_raw, d_off, d_coded, d_ex = denton_rows(boundary, "C29")
    raw = gpd.GeoDataFrame(pd.concat([c_raw, d_raw], ignore_index=True), geometry="geometry", crs=4326)
    off_roll = gpd.GeoDataFrame(pd.concat([c_off, d_off], ignore_index=True), geometry="geometry", crs=4326)
    coded = pd.concat([c_coded, d_coded], ignore_index=True)
    by_county = {"Collin": c_ex, "Denton": d_ex}

    def exempt_fn(p):
        out = pd.Series(False, index=p.index)
        for county, fn in by_county.items():
            sub = p["COUNTYNAME"].eq(county)
            if sub.any():
                out[sub] = fn(p[sub]).reindex(p.index[sub]).fillna(False).astype(bool)
        return out

    links = {"Collin": ccad_link, "Denton": denton_link}
    process(raw, off_roll, boundary, city=CITY, state=STATE, data_dir=DATA_DIR,
            public_names=PUBLIC, exempt_fn=exempt_fn,
            link_fn=lambda county, acct: links[county](acct), coded=coded)


if __name__ == "__main__":
    main()
