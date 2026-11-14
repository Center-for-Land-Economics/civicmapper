"""Denton Central Appraisal District rows for a city that reaches into Denton County.

Sources (manual downloads; dentoncad.com is not reachable from the build host), shared by every
Denton County city in data/jurisidictions/data/dentoncad/:
- denton-2026-certified.zip: Denton CAD's 2026 certified appraisal export ("2026 CERTIFIED DATA -
  ALL PROPERTIES", PACS layout 8.0.33, exported 2026-07-27). Read by pacs_city (values, state
  codes via the district's table, EX exemption flags, undivided interests).
- nightly_geodatabase.gdb.zip: Denton CAD's nightly geodatabase; layer Parcels_Certified_2026 is
  the parcel polygons as of certification, keyed by prop_id (EPSG:2276). Its
  Condos_w_LinkedPID.csv links undrawn condo units to the parcel they sit on.
Links: denton.prodigycad.com/property-detail/<prop_id>/2026 (the pattern Denton CAD's own data
uses, pinned to the 2026 roll).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

import pacs_city as pc

DENTON_DIR = Path(__file__).resolve().parent / "data" / "dentoncad"
DENTON_GDB = DENTON_DIR / "nightly_geodatabase.gdb.zip"


def denton_link(prop_id: str) -> str:
    return f"https://denton.prodigycad.com/property-detail/{prop_id}/2026"


def _geometry(boundary, cache: Path) -> None:
    if cache.exists():
        return
    src = f"/vsizip/{DENTON_GDB}/nightly_geodatabase.gdb"
    bb = gpd.GeoSeries([boundary], crs=4326).to_crs(2276).buffer(2100).iloc[0]
    g = gpd.read_file(src, layer="Parcels_Certified_2026", bbox=tuple(bb.bounds))
    g = g[g.intersects(bb)][["prop_id", "geometry"]].to_crs(4326)
    g["geometry"] = g.geometry.make_valid()
    # Condo units Denton CAD does not draw: its Condos_w_LinkedPID.csv links each unit to the
    # parcel it sits on, so the unit takes that polygon (stacked and summed downstream).
    cl = pd.read_csv(zipfile.ZipFile(DENTON_GDB).open("Condos_w_LinkedPID.csv"), dtype=str,
                     usecols=["pid", "linked_pid"])
    cl["pid"] = pd.to_numeric(cl["pid"], errors="coerce")
    cl["linked_pid"] = pd.to_numeric(cl["linked_pid"], errors="coerce")
    cl = cl.dropna()
    drawn = set(pd.to_numeric(g["prop_id"], errors="coerce").dropna().astype(int))
    cl = cl[~cl["pid"].astype(int).isin(drawn) & cl["linked_pid"].astype(int).isin(drawn)]
    if len(cl):
        par = g.assign(_p=pd.to_numeric(g["prop_id"], errors="coerce")).dissolve(by="_p")["geometry"]
        add = gpd.GeoDataFrame({"prop_id": cl["pid"].astype(float).values},
                               geometry=par.loc[cl["linked_pid"].astype(float)].values, crs=4326)
        g = pd.concat([g, add], ignore_index=True)
    g.to_parquet(cache, index=False)


def denton_rows(boundary, data_dir: Path, *, city: str, state: str, city_entity: str):
    """Standard tx_pipeline rows for Denton CAD accounts around `boundary`
    -> (raw, off_roll, coded, exempt_fn)."""
    cfg = pc.PacsCity(city=city, state=state, county="Denton", gis_url=None, export_url="",
                      tiger_geoid="", city_entity=city_entity, link=denton_link)
    cty = pc.PacsCounty(county="Denton", gis_layer=None, export_file="", city_entity=city_entity,
                        link=denton_link)
    gcache = data_dir / f"{city}-{state}-denton-geometry.parquet"
    _geometry(boundary, gcache)
    raw, off, coded = pc._county_rows(
        cfg, cty, boundary, data_dir, gcache=gcache,
        export_zip=DENTON_DIR / "denton-2026-certified.zip",
        roll_cache=DENTON_DIR / "denton-roll-2026.parquet",
        export_url=None, gis_url=None, prefix="DENTON")
    return raw, off, coded, (lambda p: p["roll_exempt"].fillna(False).astype(bool))
