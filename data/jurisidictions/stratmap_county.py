"""Rows for a city whose appraisal district is available only through TxGIO StratMap land parcels.

StratMap (Texas Geographic Information Office) republishes each appraisal district's parcels
once a year in one statewide schema: PROP_ID, owner, legal description, LAND_VALUE, IMP_VALUE,
MKT_VALUE, state / local land use, TAX_YEAR. Counties fill it unevenly. Lubbock's 2026 edition
(collected from Lubbock Appraisal District 2026-07-01) has land and market value but no
improvement value, no state class, no exemption flag and no city field, so:
- improvement value = market - land (never below 0);
- the property class comes from the city's ZONING district at the parcel's interior point
  (zoning_class), and a parcel with no improvement value is vacant;
- exemption comes from owner names (public bodies, schools, universities, churches) and from
  a market value of $0 (exempt or right-of-way parcels StratMap carries with no value);
- there is no taxing-unit code, so the city clip is by boundary alone.
Everything derived this way is approximate, and the city's notes say so.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import geopandas as gpd
import pandas as pd

from tx_common import fix_geometry, log


def stratmap_rows(boundary, data_dir: Path, *, zip_path: Path, layer_path: str, county: str,
                  zoning: gpd.GeoDataFrame | None, zoning_class: Callable[[str], str],
                  cache_name: str):
    """Standard tx_pipeline rows from a StratMap county geodatabase -> (raw, off_roll, coded, exempt_fn).

    zoning: polygons with a 'zone' column (EPSG:4326); zoning_class(zone) -> statewide-style class
    for an IMPROVED parcel (A1 / B1 / F1 / F2 / ""), vacant parcels get C1 or C1C from it."""
    cache = data_dir / cache_name
    if cache.exists():
        g = gpd.read_parquet(cache)
    else:
        bb = gpd.GeoSeries([boundary], crs=4326).buffer(0.01).iloc[0]
        g = gpd.read_file(f"/vsizip/{zip_path}/{layer_path}", bbox=tuple(bb.bounds))
        g = g[g.intersects(bb)].to_crs(4326)
        g.to_parquet(cache, index=False)
    years = g["TAX_YEAR"].astype(str).value_counts().to_dict()
    log(f"StratMap {county} parcels near the city: {len(g):,}; tax years {years}; "
        f"collected {g['DATE_ACQ'].astype(str).value_counts().head(2).to_dict()}")
    raw = g.copy()
    raw["acct"] = raw["PROP_ID"].astype(str).str.strip()
    raw["COUNTYNAME"] = county
    land = pd.to_numeric(raw["LAND_VALUE"], errors="coerce")
    mkt = pd.to_numeric(raw["MKT_VALUE"], errors="coerce")
    imp = pd.to_numeric(raw["IMP_VALUE"], errors="coerce").fillna(0)
    imp = imp.where(imp > 0, (mkt - land).clip(lower=0))
    raw["LAND_VAL"], raw["IMPR_VAL"], raw["MARKET_VALUE"] = land, imp, mkt
    area = pd.to_numeric(raw["LEGAL_AREA"], errors="coerce")
    raw["LAND_ACRE"] = area.where(area > 0)
    raw["OWNER_NAME"] = raw["OWNER_NAME"]
    raw["PARCEL_LEGAL_DESCRIPTION"] = raw["LEGAL_DESC"]
    raw["SITUS_ADDR"] = raw["SITUS_ADDR"].fillna("").str.split(",").str[0].str.strip()
    raw = fix_geometry(gpd.GeoDataFrame(raw, geometry="geometry", crs=4326))
    zone = pd.Series("", index=raw.index)
    if zoning is not None and len(zoning):
        pts = gpd.GeoDataFrame(geometry=raw.geometry.representative_point(), index=raw.index, crs=4326)
        j = gpd.sjoin(pts, zoning[["zone", "geometry"]], how="left", predicate="within")
        zone = j.groupby(level=0)["zone"].first().reindex(raw.index).fillna("")
    cls = zone.map(zoning_class).fillna("")
    vacant = raw["IMPR_VAL"].fillna(0).le(0)
    raw["LOCAL_STATE_CODE"] = zone
    raw["STATE_USE_CODE"] = cls.where(~vacant, cls.isin(["F1", "F2"]).map({True: "C1C", False: "C1"}))
    raw["zoning_district"] = zone
    log(f"{county}: improvement = market - land on {int((pd.to_numeric(g['IMP_VALUE'], errors='coerce').fillna(0) <= 0).sum()):,} "
        f"rows; zoning matched {int(zone.ne('').sum()):,} of {len(raw):,}; class from zoning: "
        f"{raw['STATE_USE_CODE'].value_counts().to_dict()}")
    off_roll = raw.iloc[0:0][["geometry"]]
    coded = pd.Series(True, index=raw.index)
    # A parcel StratMap carries with no market value is exempt or non-taxable right-of-way
    # (Lubbock 2026: city, ISD, church, BNSF / Lubbock & Western railway land).
    raw["src_exempt"] = pd.to_numeric(raw["MARKET_VALUE"], errors="coerce").fillna(0).le(0) \
        & raw["LAND_VAL"].fillna(0).le(0)
    log(f"{county}: {int(raw['src_exempt'].sum()):,} parcels with no recorded value treated as exempt")
    return raw.reset_index(drop=True), off_roll, coded.reset_index(drop=True), \
        (lambda p: p["src_exempt"].fillna(False).astype(bool))
