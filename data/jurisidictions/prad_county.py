"""Potter-Randall Appraisal District (PRAD) rows: one district covering Potter and Randall
Counties (Amarillo).

Sources (manual downloads from PRAD's data share; prad.org is not reachable from the build host),
in data/jurisidictions/data/prad/:
- potter-2026-certified.zip / randall-2026-certified.zip: PRAD's 2026 certified appraisal exports,
  one per county (PACS layout 8.0.33, exported 2026-10-01, supplements through 6). Read by
  pacs_city (values, state codes via the district's table, EX flags, undivided interests).
- CERT_TaxAcct_Name_Mailing.shp (+ .dbf/.shx/.prj): PRAD's 2026 certified GIS parcels for both
  counties, keyed by PROP_ID (EPSG:3857).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

import pacs_city as pc

PRAD_DIR = Path(__file__).resolve().parent / "data" / "prad"
EXPORTS = {"Potter": "potter-2026-certified.zip", "Randall": "randall-2026-certified.zip"}


def prad_link(prop_id: str) -> str:
    # PRAD's property page pinned to the 2026 roll (pattern confirmed by the user, prop 119174).
    return f"https://www.prad.org/property-detail/{prop_id}/2026"


def _geometry(boundary, cache: Path) -> None:
    if cache.exists():
        return
    shp = PRAD_DIR / "CERT_TaxAcct_Name_Mailing.shp"
    bb = gpd.GeoSeries([boundary], crs=4326).to_crs(3857).buffer(2500).iloc[0]
    g = gpd.read_file(shp, bbox=tuple(bb.bounds), columns=["PROP_ID", "COUNTY"])
    g = g[g.intersects(bb)].copy()
    g["geometry"] = g.geometry.make_valid()
    g.to_crs(4326).to_parquet(cache, index=False)


def prad_rows(boundary, data_dir: Path, *, county: str, city: str, state: str, city_entity: str):
    """Standard tx_pipeline rows for one PRAD county around `boundary`
    -> (raw, off_roll, coded, exempt_fn)."""
    cfg = pc.PacsCity(city=city, state=state, county=county, gis_url=None, export_url="",
                      tiger_geoid="", city_entity=city_entity, link=prad_link)
    cty = pc.PacsCounty(county=county, gis_layer=None, export_file="", city_entity=city_entity,
                        link=prad_link)
    gcache = data_dir / f"{city}-{state}-prad-geometry.parquet"
    _geometry(boundary, gcache)
    gis = gpd.read_parquet(gcache)
    ccache = data_dir / f"{city}-{state}-prad-{county.lower()}-geometry.parquet"
    if not ccache.exists():
        gis[gis["COUNTY"].fillna("").str.upper().eq(county.upper())].to_parquet(ccache, index=False)
    raw, off, coded = pc._county_rows(
        cfg, cty, boundary, data_dir, gcache=ccache,
        export_zip=PRAD_DIR / EXPORTS[county],
        roll_cache=PRAD_DIR / f"prad-{county.lower()}-roll-2026.parquet",
        export_url=None, gis_url=None, prefix=county.upper())
    return raw, off, coded, (lambda p: p["roll_exempt"].fillna(False).astype(bool))
