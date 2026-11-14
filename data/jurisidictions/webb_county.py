"""Webb County Appraisal District rows (Laredo).

Sources:
- data/jurisidictions/data/webbcad/webb-2026-certified.zip: Webb CAD's 2026 certified export
  ("2026 Webb Certified Export Files, All Entities", PACS layout 8.0.33, exported 2026-08-06;
  manual download, webbcad.org is not reachable from the build host). Read by pacs_city.
- Geometry: TxGIO StratMap 2026 land parcels for Webb County (stratmap26-landparcels_48479_lp.zip,
  collected from Webb CAD 2026-08-01, keyed by PROP_ID; manual download into the same folder).
  Fallback when absent: the City of Laredo's public parcel layer PARCEL_LANDUSE_ST_CODES, joined
  on Webb CAD's GEONUMBER (= geo_id); it predates some recent subdivisions.
Link: webbcad.org/property-detail/<prop_id>/2026 (confirmed by the user).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

import pacs_city as pc
from tx_common import fetch_arcgis_geojson, log

WEBB_DIR = Path(__file__).resolve().parent / "data" / "webbcad"
LAREDO_PARCELS = ("https://services6.arcgis.com/8FCipc4uuUgyEpbK/arcgis/rest/services/"
                  "PARCEL_LANDUSE_ST_CODES/FeatureServer/0")


def webb_link(prop_id: str) -> str:
    return f"https://www.webbcad.org/property-detail/{prop_id}/2026"


STRATMAP_WEBB = WEBB_DIR / "stratmap26-landparcels_48479_lp.zip"


def _geometry(boundary, cache: Path, roll: pd.DataFrame) -> None:
    """Webb CAD parcel polygons around `boundary` -> cache (prop_id, geometry).

    Preferred: TxGIO StratMap 2026 land parcels for Webb County (collected from Webb CAD
    2026-08-01, keyed by PROP_ID). Fallback: the City of Laredo's parcel layer joined on geo_id
    (older; misses recent subdivisions)."""
    if cache.exists():
        return
    bb = gpd.GeoSeries([boundary], crs=4326).buffer(0.01).iloc[0]
    if STRATMAP_WEBB.exists():
        src = f"/vsizip/{STRATMAP_WEBB}/fgdb/stratmap-2026-land-parcels-webb_48479.gdb"
        g = gpd.read_file(src, bbox=tuple(bb.bounds), columns=["PROP_ID"])
        g = g[g.intersects(bb)].copy()
        g["prop_id"] = pd.to_numeric(g["PROP_ID"], errors="coerce")
        g["geometry"] = g.geometry.make_valid()
        log(f"StratMap 2026 Webb parcels near the city: {len(g):,} "
            f"({int(g['prop_id'].isin(set(roll['prop_id'].astype(float))).sum()):,} with a certified record)")
        g[["prop_id", "geometry"]].to_crs(4326).to_parquet(cache, index=False)
        return
    g = fetch_arcgis_geojson(LAREDO_PARCELS, cache=WEBB_DIR / "laredo-parcel-landuse.parquet",
                             id_field="OBJECTID_12", workers=1, page=1000)
    g = g[g.intersects(bb)].copy()
    g["geo"] = g["GEONUMBER"].fillna("").str.strip()
    key = roll[["prop_id", "geo_id"]].assign(geo=roll["geo_id"].fillna("").str.strip())
    key = key[key["geo"].ne("")].drop_duplicates(["geo", "prop_id"])
    out = g[["geo", "geometry"]].merge(key[["geo", "prop_id"]], on="geo", how="left")
    log(f"Laredo parcel layer: {len(g):,} polygons; {int(out['prop_id'].notna().sum()):,} polygon-account "
        f"pairs via geo_id")
    gpd.GeoDataFrame(out, geometry="geometry", crs=4326).to_parquet(cache, index=False)


def webb_rows(boundary, data_dir: Path, *, city: str, state: str, city_entity: str):
    """Standard tx_pipeline rows for Webb CAD accounts around `boundary`
    -> (raw, off_roll, coded, exempt_fn)."""
    roll_cache = WEBB_DIR / "webb-roll-2026.parquet"
    roll = pc.load_certified_roll(WEBB_DIR / "webb-2026-certified.zip", roll_cache, "2026")
    gcache = data_dir / f"{city}-{state}-webb-geometry.parquet"
    _geometry(boundary, gcache, roll)
    cfg = pc.PacsCity(city=city, state=state, county="Webb", gis_url=None, export_url="",
                      tiger_geoid="", city_entity=city_entity, link=webb_link)
    cty = pc.PacsCounty(county="Webb", gis_layer=None, export_file="", city_entity=city_entity,
                        link=webb_link)
    raw, off, coded = pc._county_rows(
        cfg, cty, boundary, data_dir, gcache=gcache,
        export_zip=WEBB_DIR / "webb-2026-certified.zip", roll_cache=roll_cache,
        export_url=None, gis_url=None, prefix=None)
    return raw, off, coded, (lambda p: p["roll_exempt"].fillna(False).astype(bool))
