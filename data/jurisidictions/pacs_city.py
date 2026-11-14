"""Builder for Texas cities whose appraisal district publishes a PACS (True Prodigy) certified
appraisal export and its own GIS parcels (Cameron CAD for Brownsville; the same "Legacy 8.0.33"
export layout is used by Travis CAD and many other districts).

Sources, both downloaded automatically from the district's data page and cached:
- GIS parcels (shapefile zip) keyed by PROP_ID, for geometry.
- Certified appraisal export (fixed-width APPRAISAL_INFO.TXT, layout in the zip's
  Legacy8.0.33-AppraisalExportLayout.xlsx), for values, state class, acreage and exemptions.
  Only certified data is used (sup_num 0, prop_val_yr == the expected year, real property).

Roll specifics handled here (the rest is tx_pipeline.process):
- Land market value = land homestead + land non-homestead + ag market + timber market (PACS keeps
  ag/timber land's market value out of the land fields).
- Undivided interests: one record per owner. Shares (ownership_pct < 100, values prorated:
  25% + 50% + 25% of Brownsville Original Townsite blk 99 lot 6) are SUMMED; records that all say
  100% repeat the full value and are taken once.
- Total exemptions come from the roll's own EX flags (ex_exempt, EX-XV public property, EX-XG,
  EX-XJ schools, EX-XR, EX-XA, ... EX-366), not owner heuristics.
- City limits: TIGER/Line place boundary, cross-checked with the city's taxing-entity code.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import geopandas as gpd
import pandas as pd

from tx_common import fix_geometry, log
from tx_pipeline import process

ROOT = Path(__file__).resolve().parents[2]

# Legacy 8.0.33 APPRAISAL_INFO.TXT positions (1-based inclusive), verified against the layout
# workbook shipped in Cameron CAD's 2026 certified export (and identical to run_austin.py's).
PROP_FIELDS = [
    ("prop_id", 1, 12), ("prop_type_cd", 13, 17), ("prop_val_yr", 18, 22), ("sup_num", 23, 34),
    ("geo_id", 547, 596), ("py_owner_name", 609, 678), ("partial_owner", 679, 679),
    ("situs_street", 1050, 1099), ("situs_city", 1110, 1139), ("legal_desc", 1150, 1404),
    ("legal_acreage", 1660, 1675),
    ("land_hstd_val", 1796, 1810), ("land_non_hstd_val", 1811, 1825),
    ("imprv_hstd_val", 1826, 1840), ("imprv_non_hstd_val", 1841, 1855),
    ("ag_market", 1871, 1885), ("timber_market", 1901, 1915), ("appraised_val", 1916, 1930),
    ("ex366_exempt", 2730, 2730), ("ex_exempt", 2671, 2671),
    ("imprv_state_cd", 2732, 2741), ("land_state_cd", 2742, 2751), ("land_acres", 2772, 2791),
    ("ownership_pct", 4022, 4027), ("market_value", 4214, 4227), ("situs_num", 4460, 4474),
    ("entities", 5202, 5341),
    # EX-* total exemptions (Tax Code chapter 11): XD XF XG XH XI XJ XL XM XN XO XP XQ XR XS XT XU XV XA
    *[(f"ex_{c}_exempt", p, p) for c, p in zip(
        ["xd", "xf", "xg", "xh", "xi", "xj", "xl", "xm", "xn", "xo", "xp", "xq", "xr", "xs", "xt", "xu", "xv", "xa"],
        range(7374, 8310, 55))],
]
EX_FLAGS = [n for n, _, _ in PROP_FIELDS if n.startswith("ex") and n.endswith("_exempt")]
VALUE_COLS = ["land_hstd_val", "land_non_hstd_val", "imprv_hstd_val", "imprv_non_hstd_val",
              "ag_market", "timber_market", "appraised_val", "market_value"]


@dataclass
class PacsCity:
    city: str
    state: str
    county: str                                  # COUNTYNAME label
    gis_url: str                                 # district GIS parcels zip (shapefile)
    export_url: str                              # certified appraisal export zip
    tiger_geoid: str                             # TIGER/Line 2025 place GEOID for city limits
    city_entity: str                             # taxing-entity code for the city (cross-check)
    link: Callable[[str], str]
    expected_year: str = "2026"
    # District GIS value columns, used ONLY for roll records whose certified values are
    # placeholders (market value == prop_id): {"land": [...], "impr": [...], "market": col}.
    gis_values: dict | None = None
    public_names: list[str] = field(default_factory=list)
    tiger_url: str = "https://www2.census.gov/geo/tiger/TIGER2025/PLACE/tl_2025_48_place.zip"


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        log(f"Downloading {url.rsplit('/', 1)[-1]} -> {dest.name}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_certified_roll(cfg: PacsCity, data_dir: Path) -> pd.DataFrame:
    cache = data_dir / f"{cfg.city}-{cfg.state}-roll-{cfg.expected_year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    z = _download(cfg.export_url, data_dir / f"{cfg.city}-{cfg.state}-certified-export.zip")
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.upper().endswith("APPRAISAL_INFO.TXT"))
        parts = []
        for ch in pd.read_fwf(io.TextIOWrapper(zf.open(name), encoding="latin-1", newline=""),
                              colspecs=[(s - 1, e) for _, s, e in PROP_FIELDS],
                              names=[n for n, _, _ in PROP_FIELDS], dtype=str, chunksize=200_000):
            ch = ch[ch["prop_type_cd"].fillna("").str.strip().str.upper().eq("R")]
            parts.append(ch)
    d = pd.concat(parts, ignore_index=True)
    years = d["prop_val_yr"].str.strip().str.lstrip("0").value_counts().to_dict()
    sups = pd.to_numeric(d["sup_num"], errors="coerce").value_counts().to_dict()
    log(f"Certified roll: {len(d):,} real-property records; years {years}; sup_num {sups}")
    assert set(years) == {cfg.expected_year}, f"unexpected roll years {years}"
    d = d[pd.to_numeric(d["sup_num"], errors="coerce").eq(0)]
    for c in VALUE_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d["prop_id"] = pd.to_numeric(d["prop_id"], errors="coerce").astype("Int64")
    d["pct"] = pd.to_numeric(d["ownership_pct"], errors="coerce")
    d.to_parquet(cache, index=False)
    return d


def collapse_owners(d: pd.DataFrame) -> pd.DataFrame:
    """One row per prop_id: sum undivided-interest shares, take repeated full values once."""
    d = d.copy()
    d["_n"] = d.groupby("prop_id")["prop_id"].transform("size")
    multi = d["_n"] > 1
    shares = multi & d.groupby("prop_id")["pct"].transform(lambda s: (s.fillna(100) < 99.99).any())
    log(f"Multi-owner properties: {d.loc[multi, 'prop_id'].nunique():,} "
        f"({d.loc[shares, 'prop_id'].nunique():,} undivided-interest share groups summed, "
        f"the rest repeat the full value and are taken once)")
    if shares.any():
        pct_sum = d[shares].groupby("prop_id")["pct"].sum()
        log(f"  share groups: ownership % sums median {pct_sum.median():.1f}, "
            f"outside 99-101: {int(((pct_sum < 99) | (pct_sum > 101)).sum()):,}")
    first = d[~shares].drop_duplicates("prop_id")
    agg = {c: "first" for c in d.columns if c not in VALUE_COLS + ["prop_id"]}
    agg.update({c: "sum" for c in VALUE_COLS})
    summed = d[shares].groupby("prop_id", as_index=False).agg(agg)
    return pd.concat([first, summed], ignore_index=True)


def build(cfg: PacsCity) -> None:
    data_dir = ROOT / "data" / "jurisidictions" / "data" / cfg.city
    data_dir.mkdir(parents=True, exist_ok=True)

    # City limits: TIGER/Line 2025 place.
    bcache = data_dir / f"{cfg.city}-{cfg.state}-boundary.parquet"
    if not bcache.exists():
        z = _download(cfg.tiger_url, data_dir / "tl_2025_48_place.zip")
        g = gpd.read_file(f"zip://{z}")
        g[g["GEOID"] == cfg.tiger_geoid].to_crs(4326).to_parquet(bcache)
    boundary = gpd.read_parquet(bcache).union_all()

    # Geometry: district GIS parcels.
    gcache = data_dir / f"{cfg.city}-{cfg.state}-geometry.parquet"
    if gcache.exists():
        gis = gpd.read_parquet(gcache)
    else:
        z = _download(cfg.gis_url, data_dir / f"{cfg.city}-{cfg.state}-gis.zip")
        shp = next(n for n in zipfile.ZipFile(z).namelist() if n.lower().endswith(".shp"))
        gis = gpd.read_file(f"zip://{z}!{shp}").to_crs(4326)
        gis.to_parquet(gcache, index=False)
    gid = next(c for c in gis.columns if c.upper() == "PROP_ID")
    gis["prop_id"] = pd.to_numeric(gis[gid], errors="coerce").astype("Int64")
    gis = fix_geometry(gis[gis["prop_id"].notna()].copy())
    log(f"GIS parcels: {len(gis):,}")

    roll = collapse_owners(load_certified_roll(cfg, data_dir))
    roll["LAND_VAL"] = roll["land_hstd_val"] + roll["land_non_hstd_val"] + roll["ag_market"] + roll["timber_market"]
    roll["IMPR_VAL"] = roll["imprv_hstd_val"] + roll["imprv_non_hstd_val"]
    roll["MARKET_VALUE"] = roll["market_value"].where(roll["market_value"] > 0, roll["LAND_VAL"] + roll["IMPR_VAL"])
    acres = pd.to_numeric(roll["land_acres"], errors="coerce") / 10000.0
    legal = pd.to_numeric(roll["legal_acreage"], errors="coerce") / 10000.0
    roll["LAND_ACRE"] = acres.where(acres > 0, legal.where(legal > 0))
    roll["STATE_USE_CODE"] = roll["land_state_cd"].fillna("").str.strip().where(
        roll["land_state_cd"].fillna("").str.strip().ne(""), roll["imprv_state_cd"])
    roll["roll_exempt"] = roll[EX_FLAGS].fillna("").apply(lambda s: s.str.strip().str.upper().eq("T")).any(axis=1)
    log(f"Roll EX-flag total exemptions: {int(roll['roll_exempt'].sum()):,} "
        f"({ {f: int(roll[f].fillna('').str.strip().eq('T').sum()) for f in EX_FLAGS if roll[f].fillna('').str.strip().eq('T').any()} })")

    m = gis.merge(roll, on="prop_id", how="left", suffixes=("_gis", ""))
    # Placeholder values: Cameron's certified 2026 export carries market value == prop_id on 50
    # records (49 Zagar Builders lots in a new subdivision: $441,207 on prop 441207). The district's
    # own GIS copy (Sept 2026) has real figures for them ($69,713 land per lot), so those records
    # take the GIS values; every other record keeps the certified roll.
    ph = m["market_value"].eq(pd.to_numeric(m["prop_id"], errors="coerce"))
    if ph.any() and cfg.gis_values:
        gv = cfg.gis_values
        num = lambda cols: sum(pd.to_numeric(m.loc[ph, c], errors="coerce").fillna(0) for c in cols)  # noqa: E731
        m.loc[ph, "LAND_VAL"] = num(gv["land"])
        m.loc[ph, "IMPR_VAL"] = num(gv["impr"])
        m.loc[ph, "MARKET_VALUE"] = pd.to_numeric(m.loc[ph, gv["market"]], errors="coerce")
    log(f"Placeholder certified values (market == prop_id): {int(ph.sum()):,} records"
        + (" -> replaced with the district GIS copy's values" if ph.any() and cfg.gis_values else ""))
    off_roll = m[m["LAND_VAL"].isna()][["geometry"]].copy()
    raw = gpd.GeoDataFrame(m[m["LAND_VAL"].notna()].copy(), geometry="geometry", crs=4326)
    log(f"GIS polygons with a certified {cfg.expected_year} record: {len(raw):,}; without: {len(off_roll):,}")
    raw["acct"] = raw["prop_id"].astype(str)
    raw["COUNTYNAME"] = cfg.county
    raw["OWNER_NAME"] = raw["py_owner_name"]
    raw["PARCEL_LEGAL_DESCRIPTION"] = raw["legal_desc"]
    raw["SITUS_ADDR"] = (raw["situs_num"].fillna("").str.strip() + " " + raw["situs_street"].fillna("").str.strip()).str.strip()
    coded = raw["entities"].fillna("").str.split(",").apply(lambda xs: cfg.city_entity in [x.strip() for x in xs])

    process(raw, off_roll, boundary, city=cfg.city, state=cfg.state, data_dir=data_dir,
            public_names=cfg.public_names, exempt_fn=lambda p: p["roll_exempt"].fillna(False).astype(bool),
            link_fn=lambda county, acct: cfg.link(acct), coded=coded)
