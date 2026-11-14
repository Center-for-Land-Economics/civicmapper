"""Collin Central Appraisal District (CCAD) rows for a city that reaches into Collin County.

Source: CCAD's own public parcel layer, "CCAD Parcel Feature Set" (ArcGIS Online, owner CCAD_Maps),
one row per property with geometry and roll attributes. By 2026-11 the layer is on the 2027
roll year (propYear 2027, current-value fields empty), so the 2026 certified values are its
prevVal* fields (prevValYear 2026: land, improvement, market). Properties created for 2027 have
no 2026 value and are kept only as blockers; 2026 accounts retired since are not in the layer.

propCategoryCode is already the statewide SPTB class (A, B, C1, D1, F1...). Vacant land whose
propSubType is Commercial is read as C1C (vacant commercial). exemptCodes lists each property's
exemptions; total exemptions are the EX-* codes (EX-XV public property, EX-XJ schools...) plus
CHODO (11.182 community housing). HS / DV / OV65 / DP are personal exemptions on private homes.
Undivided interests (udiPropFlag T) are separate properties on the same polygon with prorated
values; tx_pipeline sums them as a stacked footprint.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from tx_common import fetch_arcgis_geojson, fix_geometry, log

CCAD_LAYER = ("https://services2.arcgis.com/uXyoacYrZTPTKD3R/arcgis/rest/services/"
              "CCAD_Parcel_Feature_Set/FeatureServer/4")
FIELDS = ["OBJECTID", "propID", "propType", "propSubType", "propCategoryCode", "propUseCode",
          "legalDescription", "udiPropFlag", "udiInterestPct", "commonInterestPct", "entityCityCode",
          "situsConcatShort", "ownerName", "landSizeAcres", "exemptCodes", "prevValYear",
          "prevValImprv", "prevValLand", "prevValMarket"]


def ccad_link(prop_id: str) -> str:
    return f"https://esearch.collincad.org/Property/View/{prop_id}?year=2026"


def _total_exempt(codes: pd.Series) -> pd.Series:
    parts = codes.fillna("").str.upper().str.split(",")
    return parts.apply(lambda xs: any(x.strip().startswith("EX-") or x.strip() == "CHODO" for x in xs))


def ccad_rows(boundary, data_dir: Path, *, city_code: str, year: str = "2026"):
    """Standard tx_pipeline rows for CCAD properties in and around `boundary`
    -> (raw, off_roll, coded, exempt_fn)."""
    pre = gpd.GeoSeries([boundary], crs=4326).to_crs(3857).buffer(200).simplify(50).to_crs(4326).iloc[0]
    g = fetch_arcgis_geojson(CCAD_LAYER, polygon=pre, out_fields=",".join(FIELDS),
                             cache=data_dir / "ccad-geometry.parquet", id_field="OBJECTID")
    g = g[g["propID"].notna()].copy()
    has = pd.to_numeric(g["prevValYear"], errors="coerce").eq(int(year))
    off_roll = g[~has][["geometry"]].copy()
    raw = g[has & g["propType"].fillna("").eq("Real")].copy()
    log(f"CCAD properties near the city: {len(g):,}; with {year} values: {int(has.sum()):,}; "
        f"without (new since): {int((~has).sum()):,}")
    raw["acct"] = "COLLIN:" + pd.to_numeric(raw["propID"], errors="coerce").astype("Int64").astype(str)
    raw["COUNTYNAME"] = "Collin"
    raw["LAND_VAL"] = pd.to_numeric(raw["prevValLand"], errors="coerce")
    raw["IMPR_VAL"] = pd.to_numeric(raw["prevValImprv"], errors="coerce")
    raw["MARKET_VALUE"] = pd.to_numeric(raw["prevValMarket"], errors="coerce")
    raw["LAND_ACRE"] = pd.to_numeric(raw["landSizeAcres"], errors="coerce").where(lambda s: s > 0)
    cls = raw["propCategoryCode"].fillna("").str.strip().str.upper()
    vac_com = cls.str.startswith("C") & raw["propSubType"].fillna("").str.upper().eq("COMMERCIAL")
    raw["LOCAL_STATE_CODE"] = cls
    raw["STATE_USE_CODE"] = cls.where(~vac_com, "C1C")
    raw["OWNER_NAME"] = raw["ownerName"]
    raw["PARCEL_LEGAL_DESCRIPTION"] = raw["legalDescription"]
    raw["SITUS_ADDR"] = raw["situsConcatShort"]
    raw["src_exempt"] = _total_exempt(raw["exemptCodes"])
    raw["city_code"] = raw["entityCityCode"].fillna("").str.strip().str.upper()
    raw = fix_geometry(gpd.GeoDataFrame(raw, geometry="geometry", crs=4326))
    log(f"CCAD {year} real properties: {len(raw):,} (total exemptions: {int(raw['src_exempt'].sum()):,}; "
        f"undivided-interest shares: {int(raw['udiPropFlag'].eq('T').sum()):,})")
    coded = raw["city_code"].eq(city_code.upper()).reset_index(drop=True)
    raw = raw.reset_index(drop=True)
    return raw, off_roll, coded, (lambda p: p["src_exempt"].fillna(False).astype(bool))
