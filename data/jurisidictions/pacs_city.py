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

from tx_common import CONDO_LEGAL_RE, development_key, fetch_arcgis_geojson, fix_geometry, log
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
class PacsCounty:
    """A further county of a multi-county city: its own district export + GIS layer."""
    county: str                                  # COUNTYNAME label (also the account prefix)
    gis_layer: str | None                        # ArcGIS layer URL with a prop_id field, or None
                                                 # and set gis_zip
    export_file: str                             # export zip name in the city's data dir
    city_entity: str                             # the city's taxing-entity code in this roll
    link: Callable[[str], str]
    gis_values: dict | None = None
    gis_zip: str | None = None                   # local district parcel shapefile zip (data dir)
    gis_id_field: str = "PROP_ID"                # its prop_id column (Ellis CAD: "pid")


@dataclass
class PacsCity:
    city: str
    state: str
    county: str                                  # COUNTYNAME label
    gis_url: str | None                          # district GIS parcels zip (shapefile), or
                                                 # None and set gis_layer (ArcGIS FeatureServer)
    export_url: str                              # certified appraisal export zip
    tiger_geoid: str                             # TIGER/Line 2025 place GEOID for city limits
    city_entity: str                             # taxing-entity code for the city (cross-check)
    link: Callable[[str], str]
    expected_year: str = "2026"
    # District GIS value columns, used ONLY for roll records whose certified values are
    # placeholders (market value == prop_id): {"land": [...], "impr": [...], "market": col}.
    gis_values: dict | None = None
    gis_layer: str | None = None                 # ArcGIS layer URL with a prop_id field
    public_names: list[str] = field(default_factory=list)
    tiger_url: str = "https://www2.census.gov/geo/tiger/TIGER2025/PLACE/tl_2025_48_place.zip"
    extra_counties: list[PacsCounty] = field(default_factory=list)


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        log(f"Downloading {url.rsplit('/', 1)[-1]} -> {dest.name}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_certified_roll(export_zip: Path, cache: Path, expected_year: str,
                        export_url: str | None = None) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    z = _download(export_url, export_zip) if export_url else export_zip
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
    assert set(years) == {expected_year}, f"unexpected roll years {years}"
    # Certified exports carry sup_num 0; "As of Supplement" exports (El Paso) carry each
    # property's latest supplement instead. Keep the latest supplement per property + owner.
    d["_sup"] = pd.to_numeric(d["sup_num"], errors="coerce").fillna(0)
    d = d.sort_values("_sup").drop_duplicates(["prop_id", "py_owner_name"], keep="last").drop(columns="_sup")
    for c in VALUE_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d["prop_id"] = pd.to_numeric(d["prop_id"], errors="coerce").astype("Int64")
    d["pct"] = pd.to_numeric(d["ownership_pct"], errors="coerce")
    d.to_parquet(cache, index=False)
    return d


def load_state_codes(export_zip: Path) -> pd.DataFrame:
    """The district's APPRAISAL_STATE_CODE table: local code -> statewide (PTD) category.

    Local codes differ by district (El Paso: C2 = colonia lots, C10 = commercial vacant lot,
    X21-X23 = ranch improvements on ag land = statewide D2, not exempt), so classes are mapped
    through this table rather than read from the local code's first letter.
    """
    with zipfile.ZipFile(export_zip) as zf:
        name = next((n for n in zf.namelist() if n.upper().endswith("STATE_CODE.TXT")), None)
        if name is None:
            return pd.DataFrame(columns=["code", "desc", "ptd", "ptd_desc"])
        lines = zf.read(name).decode("latin-1").splitlines()
    rows = [(l[0:10].strip().upper(), l[10:60].strip(), l[60:70].strip().upper(), l[70:120].strip())
            for l in lines if l.strip()]
    return pd.DataFrame(rows, columns=["code", "desc", "ptd", "ptd_desc"]).drop_duplicates("code")


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

    stem = f"{cfg.city}-{cfg.state}"
    home = PacsCounty(cfg.county, cfg.gis_layer, f"{stem}-certified-export.zip", cfg.city_entity,
                      cfg.link, cfg.gis_values)
    raws, offs, codeds, links = [], [], [], {}
    for i, c in enumerate([home, *cfg.extra_counties]):
        tag = "" if i == 0 else "-" + c.county.lower().replace(" ", "")
        log(f"=== {c.county} County ===")
        raw, off, coded = _county_rows(
            cfg, c, boundary, data_dir,
            gcache=data_dir / f"{stem}{tag}-geometry.parquet",
            export_zip=data_dir / c.export_file,
            roll_cache=data_dir / f"{stem}{tag}-roll-{cfg.expected_year}.parquet",
            export_url=cfg.export_url if i == 0 and cfg.export_url.endswith(".zip") else None,
            gis_url=cfg.gis_url if i == 0 else None,
            prefix=None if i == 0 else c.county.upper().replace(" ", ""))
        raws.append(raw); offs.append(off); codeds.append(coded); links[c.county] = c.link
    raw = gpd.GeoDataFrame(pd.concat(raws, ignore_index=True), geometry="geometry", crs=4326)
    off_roll = gpd.GeoDataFrame(pd.concat(offs, ignore_index=True), geometry="geometry", crs=4326)
    coded = pd.concat(codeds, ignore_index=True)

    process(raw, off_roll, boundary, city=cfg.city, state=cfg.state, data_dir=data_dir,
            public_names=cfg.public_names, exempt_fn=lambda p: p["roll_exempt"].fillna(False).astype(bool),
            link_fn=lambda county, acct: links[county](acct), coded=coded)


def _place_undrawn(gis: gpd.GeoDataFrame, roll: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Put certified records that have no polygon of their own onto the polygon of a matching
    record, so their land is not lost. The district maps follow the CURRENT (next-year) parcel
    fabric, so a certified record can lack a polygon because its account was split or re-IDed
    since, or because the district never draws condo units / undivided-interest accounts
    separately (Corpus Christi 2026: 8,200 taxable records, $280M land, 3.2%).

    Matched in order, each against every GIS polygon's own attributes:
      1. identical legal description (undivided-interest shares, re-IDed accounts);
      2. a condo legal (CONDO_LEGAL_RE) whose development name matches a polygon that is itself
         a condo of that development (ordinary subdivisions share names across units/phases:
         "KINGS CROSSING UNIT 1" vs "UNIT 11", so non-condo legals never match by name);
      3. a situs address that exactly one polygon carries, and only onto a polygon with no
         certified record of its own (an account re-IDed since certification), never onto
         another property's lot (Corpus Christi: 25 condo units onto a 7,500 sqft lot).
    A placed record shares that polygon, so the stacked-footprint / condo merge downstream sums
    it with the polygon's own record. Unmatched records stay off the map (logged).
    Returns (rows with the roll columns + geometry, prop_ids of off-roll polygons now used)."""
    have = set(gis["prop_id"].dropna().astype(int))
    und = roll[~roll["prop_id"].astype(int).isin(have) & roll["LAND_VAL"].gt(0)].copy()
    if und.empty or "legal_desc" not in gis.columns:
        log(f"Certified records without a polygon: {len(und):,} (${und['LAND_VAL'].sum():,.0f} land); "
            "GIS has no legal descriptions to place them")
        return und.iloc[0:0].assign(geometry=None), set()
    norm = lambda s: s.fillna("").astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()  # noqa: E731
    # One target per property: an account drawn as several polygons (a 2027 re-ID of an
    # 85-acre undivided tract in three pieces) is placed onto all of them, not the first piece.
    keep = ["prop_id", "legal_desc"] + [c for c in ("situs_num", "situs_street") if c in gis.columns]
    gis = gis[gis["prop_id"].notna()][keep + ["geometry"]].dissolve(by="prop_id", aggfunc="first", as_index=False)
    g = gis[["prop_id", "geometry"]].copy()
    g["on_roll"] = g["prop_id"].isin(roll["prop_id"])
    g["leg"] = norm(gis["legal_desc"])
    g["dev"] = development_key(gis["legal_desc"].fillna(""))
    und["leg"] = norm(und["legal_desc"])
    und["dev"] = development_key(und["legal_desc"].fillna(""))
    target = pd.Series(pd.NA, index=und.index, dtype="Int64")      # index into g
    how = pd.Series("", index=und.index)

    first = lambda key: g[g[key].str.len() > 3].reset_index().drop_duplicates(key).set_index(key)["index"]  # noqa: E731
    by_leg = first("leg")
    t1 = und["leg"].map(by_leg)
    target, how = target.fillna(t1), how.where(t1.isna(), "legal")
    is_condo = lambda s: s.fillna("").str.upper().str.contains(CONDO_LEGAL_RE, regex=True)  # noqa: E731
    g_condo = is_condo(gis["legal_desc"])
    by_dev = (g[g_condo & g["dev"].str.len().gt(3)].reset_index()
              .drop_duplicates("dev").set_index("dev")["index"])
    t2 = und["dev"].map(by_dev).where(is_condo(und["legal_desc"]) & und["dev"].str.len().gt(3) & target.isna())
    target, how = target.fillna(t2), how.where(t2.isna(), "development")
    if "situs_num" in gis.columns:
        addr = lambda n, s: (n.fillna("").astype(str).str.strip() + " " + norm(s)).str.strip()  # noqa: E731
        g["addr"] = addr(gis["situs_num"], gis["situs_street"])
        und["addr"] = addr(und["situs_num"], und["situs_street"])
        ok = g["addr"].str.match(r"^\d+ \S")
        cnt = g[ok].groupby("addr").size()
        uniq = g[ok & g["addr"].map(cnt).eq(1) & ~g["on_roll"]].reset_index().set_index("addr")["index"]
        t3 = und["addr"].map(uniq).where(target.isna())
        target, how = target.fillna(t3), how.where(t3.isna(), "address")
    hit = target.notna()
    placed = und[hit].drop(columns=["leg", "dev", "addr"], errors="ignore").copy()
    placed["geometry"] = g.loc[target[hit].astype(int), "geometry"].values
    placed["placed_by"] = how[hit].values
    used = set(g.loc[target[hit].astype(int), "prop_id"].dropna().astype(int))
    log(f"Certified records without a polygon: {len(und):,} (${und['LAND_VAL'].sum():,.0f} land); placed on a "
        f"matching polygon: {int(hit.sum()):,} (${placed['LAND_VAL'].sum():,.0f}; by "
        f"{placed.groupby('placed_by')['LAND_VAL'].sum().round(0).to_dict()}); left off: "
        f"{int((~hit).sum()):,} (${und.loc[~hit, 'LAND_VAL'].sum():,.0f})")
    return placed, used


def _county_rows(cfg: PacsCity, c: PacsCounty, boundary, data_dir: Path, *, gcache: Path,
                 export_zip: Path, roll_cache: Path, export_url: str | None, gis_url: str | None,
                 prefix: str | None):
    """One district's certified roll joined to its GIS polygons -> (raw, off_roll, coded)."""
    # Geometry: district GIS parcels.
    if gcache.exists():
        gis = gpd.read_parquet(gcache)
    elif c.gis_layer:
        # The district's own web-map layer, pre-filtered to the city outline (+200 m).
        pre = gpd.GeoSeries([boundary], crs=4326).to_crs(3857).buffer(200).simplify(50).to_crs(4326).iloc[0]
        gis = fetch_arcgis_geojson(c.gis_layer, polygon=pre, cache=gcache, id_field="ObjectID_1")
    else:
        z = data_dir / c.gis_zip if c.gis_zip else _download(gis_url, data_dir / f"{cfg.city}-{cfg.state}-gis.zip")
        shp = next(n for n in zipfile.ZipFile(z).namelist() if n.lower().endswith(".shp"))
        src = gpd.read_file(f"zip://{z}!{shp}", rows=1)
        bb = gpd.GeoSeries([boundary], crs=4326).to_crs(src.crs).buffer(650).iloc[0]
        gis = gpd.read_file(f"zip://{z}!{shp}", bbox=tuple(bb.bounds)).to_crs(4326)
        gis.to_parquet(gcache, index=False)
    gid = next(col for col in gis.columns if col.upper() in (c.gis_id_field.upper(), "PROP_ID"))
    gis["prop_id"] = pd.to_numeric(gis[gid], errors="coerce").astype("Int64")
    gis = fix_geometry(gis[gis["prop_id"].notna()].copy())
    log(f"GIS parcels: {len(gis):,}")

    roll = collapse_owners(load_certified_roll(export_zip, roll_cache, cfg.expected_year, export_url))
    roll["LAND_VAL"] = roll["land_hstd_val"] + roll["land_non_hstd_val"] + roll["ag_market"] + roll["timber_market"]
    roll["IMPR_VAL"] = roll["imprv_hstd_val"] + roll["imprv_non_hstd_val"]
    roll["MARKET_VALUE"] = roll["market_value"].where(roll["market_value"] > 0, roll["LAND_VAL"] + roll["IMPR_VAL"])
    acres = pd.to_numeric(roll["land_acres"], errors="coerce") / 10000.0
    legal = pd.to_numeric(roll["legal_acreage"], errors="coerce") / 10000.0
    roll["LAND_ACRE"] = acres.where(acres > 0, legal.where(legal > 0))
    local = roll["land_state_cd"].fillna("").str.strip().where(
        roll["land_state_cd"].fillna("").str.strip().ne(""), roll["imprv_state_cd"].fillna("").str.strip()).str.upper()
    sc = load_state_codes(export_zip).set_index("code")
    ptd = local.map(sc["ptd"]).where(lambda s: s.fillna("").ne(""), local)
    # Vacant land whose local code says COMMERCIAL keeps that distinction (categorize_sptb
    # reads "C1C" as Vacant Commercial); every other code is read as its statewide category.
    commercial_vacant = ptd.str.startswith("C") & local.map(sc["desc"]).fillna("").str.upper().str.contains("COMMERCIAL")
    roll["LOCAL_STATE_CODE"] = local
    roll["STATE_USE_CODE"] = ptd.where(~commercial_vacant, "C1C")
    unmapped = local[local.ne("") & ~local.isin(sc.index)].value_counts()
    log(f"State codes: {local.nunique():,} local codes mapped to statewide categories via the district's "
        f"table ({len(sc):,} entries); unmapped: {unmapped.head(8).to_dict()}")
    roll["roll_exempt"] = roll[EX_FLAGS].fillna("").apply(lambda s: s.str.strip().str.upper().eq("T")).any(axis=1)
    log(f"Roll EX-flag total exemptions: {int(roll['roll_exempt'].sum()):,} "
        f"({ {f: int(roll[f].fillna('').str.strip().eq('T').sum()) for f in EX_FLAGS if roll[f].fillna('').str.strip().eq('T').any()} })")

    m = gis.merge(roll, on="prop_id", how="left", suffixes=("_gis", ""))
    # Placeholder values: Cameron's certified 2026 export carries market value == prop_id on 50
    # records (49 Zagar Builders lots in a new subdivision: $441,207 on prop 441207). The district's
    # own GIS copy (Sept 2026) has real figures for them ($69,713 land per lot), so those records
    # take the GIS values; every other record keeps the certified roll.
    ph = m["market_value"].eq(pd.to_numeric(m["prop_id"], errors="coerce"))
    if ph.any() and c.gis_values:
        gv = c.gis_values
        num = lambda cols: sum(pd.to_numeric(m.loc[ph, c], errors="coerce").fillna(0) for c in cols)  # noqa: E731
        m.loc[ph, "LAND_VAL"] = num(gv["land"])
        m.loc[ph, "IMPR_VAL"] = num(gv["impr"])
        m.loc[ph, "MARKET_VALUE"] = pd.to_numeric(m.loc[ph, gv["market"]], errors="coerce")
    log(f"Placeholder certified values (market == prop_id): {int(ph.sum()):,} records"
        + (" -> replaced with the district GIS copy's values" if ph.any() and c.gis_values else ""))
    placed, used = _place_undrawn(gis, roll)
    m = pd.concat([m, placed], ignore_index=True)
    off_roll = m[m["LAND_VAL"].isna() & ~m["prop_id"].isin(list(used))][["geometry"]].copy()
    raw = gpd.GeoDataFrame(m[m["LAND_VAL"].notna()].copy(), geometry="geometry", crs=4326)
    log(f"GIS polygons with a certified {cfg.expected_year} record: {len(raw) - len(placed):,}; without: {len(off_roll):,}")
    raw["acct"] = raw["prop_id"].astype(str) if prefix is None else prefix + ":" + raw["prop_id"].astype(str)
    raw["COUNTYNAME"] = c.county
    raw["OWNER_NAME"] = raw["py_owner_name"]
    raw["PARCEL_LEGAL_DESCRIPTION"] = raw["legal_desc"]
    raw["SITUS_ADDR"] = (raw["situs_num"].fillna("").str.strip() + " " + raw["situs_street"].fillna("").str.strip()).str.strip()
    coded = raw["entities"].fillna("").str.split(",").apply(lambda xs: c.city_entity in [x.strip() for x in xs])
    return raw, off_roll, coded
