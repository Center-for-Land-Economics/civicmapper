"""Dallas Central Appraisal District (DCAD) rows for a city that reaches into Dallas County.

Sources (manual downloads from https://www.dallascad.org/DataProducts.aspx; dallascad.org is
not reachable from the build host), placed in the city's data dir under dcad/:
- DCAD2026_CURRENT.zip: the current-year certified appraisal data files (comma delimited;
  certified 2026 values plus supplements). Used: ACCOUNT_APPRL_YEAR (values, SPTD code, city
  jurisdiction, GIS parcel id), ACCOUNT_INFO (owner, situs, legal), LAND (land area by
  section), TOTAL_EXEMPTION (totally exempt accounts).
- PARCEL_GEOM.zip: DCAD's parcel polygons (Acct = the GIS parcel id, EPSG:2276).
Condo units and other accounts that share a lot carry the lot's GIS_PARCEL_ID, so they land on
the same polygon and tx_pipeline sums them as a stacked footprint.

DCAD SPTD codes are DCAD's own (A11, C12, F10...); they are mapped to statewide classes with
DCAD's SPTD_CD_XREF crosswalk before tx_pipeline categorizes them:
  A1x/A20 -> A1, B1x -> B1, C11/C14 -> C1, C12/C13 -> C1C (commercial / industrial vacant
  lots), D10 -> D1, D20 -> D2, E -> E, F10 -> F1, F20 -> F2, G -> G1, J -> J, O -> O,
  L/M/N/S -> personal property (L1/M1/N/S).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from tx_common import fix_geometry, log

# DCAD's account pages are split by division: AcctDetailRes.aspx (residential) and
# AcctDetailCom.aspx (commercial); residential pattern confirmed by the user (2026-11-14).
DCAD_PAGES = {"RES": "https://www.dallascad.org/AcctDetailRes.aspx?ID=",
              "COM": "https://www.dallascad.org/AcctDetailCom.aspx?ID="}
_DIVISION: dict[str, str] = {}


def dcad_link(acct: str) -> str:
    return DCAD_PAGES.get(_DIVISION.get(acct, "RES"), DCAD_PAGES["RES"]) + acct


def statewide_class(code: str) -> str:
    c = str(code or "").strip().upper()
    if not c or c == "NAN":
        return ""
    if c in ("C12", "C13"):
        return "C1C"
    if c in ("C11", "C14"):
        return "C1"
    head = c[0]
    return {"A": "A1", "B": "B1", "D": "D2" if c.startswith("D2") else "D1", "E": "E",
            "F": "F2" if c.startswith("F2") else "F1", "G": "G1", "J": "J", "O": "O",
            "L": "L1", "M": "M1", "N": "N", "S": "S", "X": "X"}.get(head, c)


def dcad_rows(boundary, data_dir: Path, *, city_juris: str, year: str = "2026"):
    """Standard tx_pipeline rows for DCAD accounts in and around `boundary`
    -> (raw, off_roll, coded, exempt_fn)."""
    d = data_dir / "dcad"
    cache = data_dir / f"dcad-rows-{year}.parquet"
    if cache.exists():
        raw = gpd.read_parquet(cache)
        off_roll = gpd.read_parquet(data_dir / f"dcad-offroll-{year}.parquet")
    else:
        b = gpd.GeoSeries([boundary], crs=4326).to_crs(2276).buffer(650).iloc[0]
        geom = gpd.read_file(f"zip://{d / 'PARCEL_GEOM.zip'}!PARCEL_GEOM/PARCEL_GEOM.shp", bbox=tuple(b.bounds))
        geom = geom[geom.intersects(b)].copy()
        geom["Acct"] = geom["Acct"].astype(str).str.strip()
        geom = geom.dissolve(by="Acct", as_index=False)[["Acct", "RecAcs", "geometry"]].to_crs(4326)
        log(f"DCAD parcel polygons near the city: {len(geom):,}")
        z = zipfile.ZipFile(d / "DCAD2026_CURRENT.zip")
        rd = lambda n, cols: pd.read_csv(z.open(n), dtype=str, usecols=cols, encoding="latin-1")  # noqa: E731
        ap = rd("ACCOUNT_APPRL_YEAR.CSV", ["ACCOUNT_NUM", "APPRAISAL_YR", "IMPR_VAL", "LAND_VAL", "TOT_VAL",
                                           "CITY_JURIS_DESC", "SPTD_CODE", "GIS_PARCEL_ID", "DIVISION_CD",
                                           "CITY_TAXABLE_VAL"])
        years = ap["APPRAISAL_YR"].value_counts().to_dict()
        assert set(years) == {year}, f"unexpected DCAD years {years}"
        ap = ap[ap["DIVISION_CD"].ne("BPP") & ap["GIS_PARCEL_ID"].isin(set(geom["Acct"]))].copy()
        info = rd("ACCOUNT_INFO.CSV", ["ACCOUNT_NUM", "OWNER_NAME1", "BIZ_NAME", "STREET_NUM",
                                       "FULL_STREET_NAME", "UNIT_ID", "LEGAL1", "LEGAL2", "LEGAL3", "LEGAL4"])
        ap = ap.merge(info, on="ACCOUNT_NUM", how="left")
        land = rd("LAND.CSV", ["ACCOUNT_NUM", "AREA_SIZE", "AREA_UOM_DESC"])
        land = land[land["ACCOUNT_NUM"].isin(set(ap["ACCOUNT_NUM"]))]
        sz = pd.to_numeric(land["AREA_SIZE"], errors="coerce")
        uom = land["AREA_UOM_DESC"].fillna("").str.upper()
        land["ac"] = sz.where(uom.str.startswith("ACRE")).fillna(sz.where(uom.str.startswith("SQUARE")) / 43560.0)
        acres = land.groupby("ACCOUNT_NUM")["ac"].sum(min_count=1)
        tex = set(rd("TOTAL_EXEMPTION.CSV", ["ACCOUNT_NUM"])["ACCOUNT_NUM"])

        raw = ap.merge(geom.rename(columns={"Acct": "GIS_PARCEL_ID"}), on="GIS_PARCEL_ID", how="left")
        raw = gpd.GeoDataFrame(raw, geometry="geometry", crs=4326)
        raw["acct"] = raw["ACCOUNT_NUM"].str.strip()
        raw["COUNTYNAME"] = "Dallas"
        for c in ("LAND_VAL", "IMPR_VAL", "TOT_VAL"):
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw["MARKET_VALUE"] = raw["TOT_VAL"]
        raw["LAND_ACRE"] = raw["acct"].map(acres)
        raw["LOCAL_STATE_CODE"] = raw["SPTD_CODE"].fillna("").str.strip()
        raw["STATE_USE_CODE"] = raw["LOCAL_STATE_CODE"].map(statewide_class)
        raw["OWNER_NAME"] = raw["OWNER_NAME1"].fillna(raw["BIZ_NAME"])
        raw["PARCEL_LEGAL_DESCRIPTION"] = raw[["LEGAL1", "LEGAL2", "LEGAL3", "LEGAL4"]].fillna("").agg(
            " ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()
        raw["SITUS_ADDR"] = (raw["STREET_NUM"].fillna("").str.strip() + " "
                             + raw["FULL_STREET_NAME"].fillna("").str.strip()).str.strip()
        raw["src_exempt"] = raw["acct"].isin(tex)
        raw["city_juris"] = raw["CITY_JURIS_DESC"].fillna("").str.strip().str.upper()
        raw = fix_geometry(raw)
        keep = ["acct", "DIVISION_CD", "COUNTYNAME", "LAND_VAL", "IMPR_VAL", "MARKET_VALUE", "LAND_ACRE", "STATE_USE_CODE",
                "LOCAL_STATE_CODE", "OWNER_NAME", "PARCEL_LEGAL_DESCRIPTION", "SITUS_ADDR", "src_exempt",
                "city_juris", "GIS_PARCEL_ID", "geometry"]
        raw = raw[keep]
        used = set(raw["GIS_PARCEL_ID"])
        off_roll = geom[~geom["Acct"].isin(used)][["geometry"]].copy()
        raw.to_parquet(cache, index=False)
        off_roll.to_parquet(data_dir / f"dcad-offroll-{year}.parquet", index=False)
    log(f"DCAD {year} certified accounts on a polygon near the city: {len(raw):,} "
        f"(totally exempt: {int(raw['src_exempt'].sum()):,}); polygons with no account: {len(off_roll):,}")
    _DIVISION.update(dict(zip(raw["acct"], raw["DIVISION_CD"].fillna("RES"))))
    coded = raw["city_juris"].eq(city_juris.upper())
    return raw, off_roll, coded, (lambda p: p["src_exempt"].fillna(False).astype(bool))
