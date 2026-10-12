"""
Copenhagen (Denmark) parcel ETL — København + Frederiksberg kommuner.

SOURCE (rebuilt 2026-10 after a user report, see "WHY THE REBUILD" below)
  Vurderingsstyrelsen's public valuation map, Vurderingsportalen
  (https://www.vurderingsportalen.dk/kort/), is backed by three national FlatGeobuf files
  with a spatial index. No auth; the map reads them with HTTP range requests, and so do we:
  only the features inside the bbox are fetched (GDAL /vsicurl/, ~1 min, cached).

    vurddb          one row per valued property (BFE), MULTIPOLYGON of its land.
                    id = "<kommunekode> / <ejendomsnummer>". Every published valuation is a
                    column: vur_<year><e|f>_grundvaerdi / _ejendomsvaerdi
                    (e = endelig/final, f = foreløbig/preliminary), plus the old-system
                    vur_2020e_legacy_* figures and juridiskkategori_fv<year>.
    vurddb_moderej  one polygon per condominium master property (moderejendom), keyed by BFE.
                    The master property carries NO land value of its own.
    vurddb_ejl      one point per stairwell address. `ejerlejligheder` is a JSON list of the
                    condo units at that address, each with its own BFE and valuations;
                    `bfenumre_moderejendom` names the master property (or properties).

  The geometry of unassessed land (parks, exempt public land, harbour) comes from the last
  DAWA jordstykker pull for kommune 0101, cached in cache/copenhagen-dawa-jordstykker-0101.parquet.
  DAWA itself is gone (api.dataforsyningen.dk/jordstykker returns 410 since 2025), so that cache
  is the only copy; it also supplies Copenhagen's parish (sogn) and the VUR benyttelseskode
  land-use category that the original July build pulled from Datafordeler.

WHY THE REBUILD (Victor Mayland Nielsen, Utrecht University, 2026-10)
  The July build joined each DAWA jordstykke to Datafordeler VUR by BFE and kept the newest
  valuation in VUR. Two things went wrong, and together they made single-family rowhouses look
  far more valuable than the apartment city around them:
    1. CONDO BUILDINGS. A building split into ejerlejligheder is a moderejendom BFE; its land is
       valued unit by unit on the unit BFEs. The jordstykke only knows the moderejendom, so 3,441
       of 3,568 condo buildings shipped with 0 kr land. Fix: sum the units' grundværdi onto the
       moderejendom polygon (Amerikavej 15: 0 kr -> 74.0M kr over 3 units).
    2. RENTAL/ANDEL/COMMERCIAL BUILDINGS ("erhvervsejendomme"). The new-system valuations for
       these are only published as PRELIMINARY 2023/2025 figures, which Datafordeler VUR does not
       carry, so the July build fell back to the frozen old-system (legacy, ~2012-level)
       grundværdi: e.g. Vesterfælledvej 17 shipped at 28.3M kr; the 2025 preliminary grundværdi
       is 454.1M kr. Owner-occupied homes meanwhile had current 2022 values. Fix: take the newest
       published valuation per property (final preferred over preliminary within a year).

VALUATION BASIS (per property, recorded in `valuation`)
  ejerboliger (houses, rowhouses, condo units)  -> 2024 preliminary (fallback 2022 final, ...)
  erhvervsejendomme (rental, andel, commercial)  -> 2025 preliminary (fallback 2023 preliminary)
  So owner homes and the rest are one valuation year apart. Erhverv preliminaries publish land
  value only (ejendomsværdi = 0), so improvement_value / full_market_value are NULL for them; the
  Underdeveloped test for those uses the same-vintage legacy ejendomsværdi:grundværdi ratio.

DEDUP / DOUBLE-COUNT RULES (add-city SKILL §2, §6)
  * A unit is listed once in vurddb_ejl (verified: 0 duplicate unit BFEs). A unit listed under
    k master properties contributes 1/k to each.
  * 97 master properties ALSO appear in vurddb as an ordinary valued property with the same
    address and polygon. Usually that record is the building's pre-conversion valuation (e.g.
    Amager Strandvej 28: 464M kr in 2023, then 194 units summing to 767M kr in 2025), sometimes
    an undivided remainder. Taking max(record, unit sum) never double counts; summing would.
  * No vurddb rows share a vurderingsejendomid, and only 2 pairs of vurddb polygons overlap.

Run:
  PYTHONUTF8=1 python data/jurisidictions/run_copenhagen.py
  (add --refresh to re-pull the Vurderingsportalen extracts)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

VP_BASE = "https://www.vurderingsportalen.dk/kort/data"
VP_LAYERS = {"main": "vurddb", "moderej": "vurddb_moderej", "ejl": "vurddb_ejl"}
# CloudFront in front of Vurderingsportalen rejects GDAL's default user agent (403).
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/129.0 Safari/537.36")
BBOX = (12.40, 55.60, 12.75, 55.75)  # covers both kommuner with margin

KOMMUNER = {101: "Copenhagen", 147: "Frederiksberg"}
UTM = "EPSG:25832"  # ETRS89 / UTM 32N
SQM_TO_SQFT = 10.76391041671
SQM_TO_ACRES = 1.0 / 4046.8564224

# Newest first. Within a year the final (e) valuation beats the preliminary (f).
VAL_ORDER = ["2026f", "2025e", "2025f", "2024e", "2024f", "2023e", "2023f",
             "2022e", "2022f", "2021e", "2020e"]

# Overture building subtype -> category, used only where no assessor use code exists
# (Frederiksberg rental/commercial properties). Validated against Copenhagen's benyttelseskode:
# residential->Residential 69%, commercial->Commercial 53%, civic/education/etc->Institutional ~65%.
OVERTURE_SUBTYPE_CATEGORY = {
    "residential": "Residential",
    "commercial": "Commercial", "service": "Commercial",
    "industrial": "Industrial", "agricultural": "Industrial",
    "education": "Institutional", "civic": "Institutional", "religious": "Institutional",
    "medical": "Institutional", "entertainment": "Institutional", "military": "Institutional",
}


# ── 1. Source extracts ──────────────────────────────────────────────────────────
def fetch_layer(name: str, cache_dir: str, refresh: bool) -> gpd.GeoDataFrame:
    """bbox-read one Vurderingsportalen FlatGeobuf via GDAL /vsicurl/ and cache it."""
    path = os.path.join(cache_dir, f"vurderingsportalen-{VP_LAYERS[name]}.parquet")
    if os.path.exists(path) and not refresh:
        print(f"📂 cached {path}")
        return gpd.read_parquet(path)
    import pyogrio
    os.environ.setdefault("GDAL_HTTP_USERAGENT", BROWSER_UA)
    os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
    os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "4194304")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
    url = f"/vsicurl/{VP_BASE}/{VP_LAYERS[name]}"
    print(f"⬇️  {url}  bbox={BBOX}")
    gdf = pyogrio.read_dataframe(url, bbox=BBOX)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf.to_parquet(path, index=False)
    print(f"   {len(gdf):,} features -> {path}")
    return gdf


def newest(df: pd.DataFrame, kind: str = "grundvaerdi") -> tuple[pd.Series, pd.Series]:
    """Newest published valuation per row -> (value, 'YYYYe|f' label)."""
    val = pd.Series(np.nan, index=df.index, dtype=float)
    lab = pd.Series(pd.NA, index=df.index, dtype="string")
    for o in VAL_ORDER:
        col = f"vur_{o}_{kind}"
        if col not in df:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        take = val.isna() & v.notna()
        val[take] = v[take]
        lab[take] = o
    return val, lab


def valuation_label(code) -> str | None:
    if code is None or pd.isna(code):
        return None
    return f"{code[:4]} {'final' if code.endswith('e') else 'preliminary'}"


# ── 2. Condo units -> master properties ───────────────────────────────────────
def condo_units(ejl: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ejl.iterrows():
        mods = json.loads(r["bfenumre_moderejendom"] or "[]")
        for u in json.loads(r["ejerlejligheder"] or "[]"):
            u["mods"] = mods
            rows.append(u)
    u = pd.DataFrame(rows)
    assert not u["bfenummer"].duplicated().any(), "a condo unit is listed twice"
    u["g"], u["g_val"] = newest(u, "grundvaerdi")
    ejd = pd.Series(np.nan, index=u.index)
    for o in VAL_ORDER:  # ejendomsværdi from the SAME valuation as the land value
        col = f"vur_{o}_ejendomsvaerdi"
        if col in u:
            m = (u["g_val"] == o)
            ejd[m] = pd.to_numeric(u.loc[m, col], errors="coerce")
    u["ejd"] = ejd.where(ejd > 0)  # erhverv preliminaries publish 0 = "not published"
    u["leg_g"] = pd.to_numeric(u["vur_2020e_legacy_grundvaerdi"], errors="coerce")
    u["leg_ejd"] = pd.to_numeric(u["vur_2020e_legacy_ejendomsvaerdi"], errors="coerce")
    return u


def aggregate_units(u: pd.DataFrame) -> pd.DataFrame:
    x = u[u["mods"].str.len() > 0].explode("mods").copy()
    k = x.groupby(level=0)["mods"].transform("size")
    for c in ("g", "ejd", "leg_g", "leg_ejd"):
        x[c] = x[c] / k
    x["ejd_missing"] = x["ejd"].isna() & x["g"].notna()
    agg = x.groupby("mods").agg(
        units=("bfenummer", "size"),
        land=("g", lambda s: s.sum(min_count=1)),
        ejd=("ejd", lambda s: s.sum(min_count=1)),
        ejd_missing=("ejd_missing", "any"),
        leg_g=("leg_g", "sum"), leg_ejd=("leg_ejd", "sum"),
        kom=("municipalitynumber", lambda s: s.mode().iloc[0]),
        valuation=("g_val", lambda s: s.dropna().max() if s.notna().any() else pd.NA),
    )
    agg.index = agg.index.astype("int64")
    # A partial building value would understate; publish it only when every unit has one.
    agg.loc[agg["ejd_missing"], "ejd"] = np.nan
    return agg


# ── 3. Categories ─────────────────────────────────────────────────────────────
def join_dawa(feat: gpd.GeoDataFrame, dawa: gpd.GeoDataFrame) -> pd.DataFrame:
    """Each feature's dominant DAWA jordstykke (by overlap) -> parish + benyttelse category."""
    f = feat[["fid", "geometry"]].to_crs(UTM)
    d = dawa[["district", "property_land_use_category", "geometry"]].to_crs(UTM)
    ov = gpd.overlay(f, d, how="intersection", keep_geom_type=True)
    ov["a"] = ov.area
    dom = ov.sort_values("a").drop_duplicates("fid", keep="last").set_index("fid")
    return dom[["district", "property_land_use_category"]]


def overture_category(feat: gpd.GeoDataFrame, bldg: gpd.GeoDataFrame) -> pd.Series:
    b = bldg[bldg["subtype"].isin(OVERTURE_SUBTYPE_CATEGORY)].to_crs(UTM)
    f = feat[["fid", "geometry"]].to_crs(UTM)
    ov = gpd.overlay(f, b[["subtype", "geometry"]], how="intersection", keep_geom_type=True)
    ov["a"] = ov.area
    dom = (ov.groupby(["fid", "subtype"])["a"].sum().reset_index()
             .sort_values("a").drop_duplicates("fid", keep="last").set_index("fid")["subtype"])
    return dom.map(OVERTURE_SUBTYPE_CATEGORY)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-pull the Vurderingsportalen extracts")
    ap.add_argument("--no-gap-fill", action="store_true",
                    help="don't add unassessed DAWA jordstykker (parks, exempt land) for Copenhagen")
    args = ap.parse_args()

    data_dir = os.path.join(HERE, "data", "copenhagen")
    cache = os.path.join(data_dir, "cache")
    os.makedirs(cache, exist_ok=True)
    slug = "copenhagen-hovedstaden-dk"

    main_ = fetch_layer("main", cache, args.refresh)
    moder = fetch_layer("moderej", cache, args.refresh)
    ejl = fetch_layer("ejl", cache, args.refresh)
    dawa = gpd.read_parquet(os.path.join(cache, "copenhagen-dawa-jordstykker-0101.parquet"))

    # ── condo master properties (moderejendomme) ────────────────────────────────
    units = condo_units(ejl)
    agg = aggregate_units(units)
    moder["bfenummer"] = moder["bfenummer"].astype("int64")
    moder = moder.join(agg, on="bfenummer")
    moder = moder[moder["kom"].isin(KOMMUNER)].copy()
    print(f"condo buildings: {len(moder):,} ({moder['units'].sum():,.0f} units), "
          f"unit land {moder['land'].sum()/1e9:,.1f} mia. kr")

    # ── ordinary valued properties ──────────────────────────────────────────────
    main_["kom"] = main_["id"].str.split(" / ").str[0].astype(int)
    main_ = main_[main_["kom"].isin(KOMMUNER)].copy()
    main_["land"], main_["valuation"] = newest(main_, "grundvaerdi")
    ejd = pd.Series(np.nan, index=main_.index)
    for o in VAL_ORDER:
        col = f"vur_{o}_ejendomsvaerdi"
        m = main_["valuation"] == o
        ejd[m] = pd.to_numeric(main_.loc[m, col], errors="coerce")
    main_["ejd"] = ejd.where(ejd > 0)
    main_["leg_g"] = pd.to_numeric(main_["vur_2020e_legacy_grundvaerdi"], errors="coerce")
    main_["leg_ejd"] = pd.to_numeric(main_["vur_2020e_legacy_ejendomsvaerdi"], errors="coerce")
    jk = main_["juridiskkategori_fv2025"].fillna(main_["juridiskkategori_fv2024"]) \
        .fillna(main_["juridiskkategori_fv2023"]).fillna(main_["juridiskkategori_fv2022"])
    main_["owner_home"] = jk.fillna("").str.startswith("Ejerbolig")

    # Same-address master-property duplicates (see header): fold into the condo feature.
    mu = moder.to_crs(UTM)
    mm = main_.to_crs(UTM)
    mm["_a"] = mm.area
    ov = gpd.overlay(mm[["id", "_a", "beliggenhed", "geometry"]],
                     mu[["bfenummer", "beliggenhed", "geometry"]].rename(columns={"beliggenhed": "mod_addr"}),
                     how="intersection", keep_geom_type=True)
    ov["frac"] = ov.area / ov["_a"]
    same = ov["beliggenhed"].str.split(",").str[0] == ov["mod_addr"].str.split(",").str[0]
    dup = ov[(ov["frac"] > 0.8) & same].drop_duplicates("id")
    dup_land = dup.merge(main_[["id", "land", "valuation"]], on="id")
    print(f"master properties also listed as a valued property: {len(dup):,} "
          f"(record land {dup_land['land'].sum()/1e9:,.2f} mia. kr; kept as max(record, unit sum))")
    rec = dup_land.groupby("bfenummer").agg(rec_land=("land", "max"))
    moder = moder.join(rec, on="bfenummer")
    use_rec = moder["rec_land"].fillna(-1) > moder["land"].fillna(-1)
    moder.loc[use_rec, "land"] = moder.loc[use_rec, "rec_land"]
    moder.loc[use_rec, "ejd"] = np.nan  # record and units disagree on the building; don't guess
    main_ = main_[~main_["id"].isin(dup["id"])].copy()

    # ── assemble one feature per property ───────────────────────────────────────
    a = gpd.GeoDataFrame({
        "parcel_id": main_["kom"].astype(str) + "-" + main_["id"].str.split(" / ").str[1],
        "address": main_["beliggenhed"],
        "kom": main_["kom"],
        "property_type": np.where(main_["owner_home"], "Owner-occupied home (ejerbolig)",
                                  "Rental, co-op, commercial or other (erhvervsejendom)"),
        "condo_units": 0,
        "land": main_["land"], "ejd": main_["ejd"],
        "leg_g": main_["leg_g"], "leg_ejd": main_["leg_ejd"],
        "valuation": main_["valuation"],
        "floor_area_sqm": pd.to_numeric(main_["sum_areal_o_kaelder"], errors="coerce"),
        "geometry": main_.geometry.values,
    }, geometry="geometry", crs=main_.crs)
    b = gpd.GeoDataFrame({
        "parcel_id": moder["bfenummer"].astype(str),
        "address": moder["beliggenhed"],
        "kom": moder["kom"].astype(int),
        "property_type": "Condominium building (ejerlejligheder)",
        "condo_units": moder["units"].fillna(0).astype(int),
        "land": moder["land"], "ejd": moder["ejd"],
        "leg_g": moder["leg_g"], "leg_ejd": moder["leg_ejd"],
        "valuation": moder["valuation"],
        "floor_area_sqm": np.nan,
        "geometry": moder.geometry.values,
    }, geometry="geometry", crs=moder.crs)
    feat = pd.concat([a, b], ignore_index=True)
    feat = gpd.GeoDataFrame(feat, geometry="geometry", crs=a.crs)
    feat["geometry"] = feat.geometry.apply(lambda g: g if g is None or g.is_valid else g.buffer(0))
    feat["fid"] = np.arange(len(feat))
    feat["jurisdiction"] = feat["kom"].map(KOMMUNER)

    # ── categories + parish ─────────────────────────────────────────────────────
    dom = join_dawa(feat, dawa)
    feat["district"] = feat["fid"].map(dom["district"])
    feat["dawa_cat"] = feat["fid"].map(dom["property_land_use_category"])
    bldg = gpd.read_parquet(os.path.join(cache, "overture-buildings.parquet"))
    feat["bldg_cat"] = feat["fid"].map(overture_category(feat, bldg))

    is_cph = feat["kom"] == 101
    is_condo = feat["condo_units"] > 0
    cat = pd.Series("Other", index=feat.index, dtype=object)
    cat[feat["property_type"].str.startswith("Owner")] = "Residential"
    cat[is_condo] = "Residential"
    erh = feat["property_type"].str.startswith("Rental")
    cat[erh] = feat.loc[erh, "bldg_cat"].fillna("Other")
    # Copenhagen: the assessor's own use code (benyttelseskode via the July VUR pull) wins.
    use_dawa = is_cph & feat["dawa_cat"].notna() & ~feat["dawa_cat"].isin(["Unknown", "Exempt"])
    cat[use_dawa & ~is_condo] = feat.loc[use_dawa & ~is_condo, "dawa_cat"]
    # Frederiksberg has no parish source since DAWA closed; group it as one region.
    feat.loc[~is_cph, "district"] = "Frederiksberg"
    feat["property_land_use_category"] = cat
    feat["category_source"] = np.where(use_dawa & ~is_condo, "VUR benyttelseskode",
                              np.where(erh, "Overture building type (estimate)", "valuation category"))

    # ── gap fill: unassessed Copenhagen land (parks, exempt public land) ────────
    if not args.no_gap_fill:
        cover = feat.to_crs(UTM).union_all()
        d = dawa.to_crs(UTM)
        covered = d.intersection(cover).area / d.area
        gap = dawa[(covered < 0.05).values].copy()
        gap = gpd.GeoDataFrame({
            # parcel_id = BFE; the few jordstykker without one get "<ejerlav>/<matrikelnr>".
            "parcel_id": gap["bfenummer"].astype("Int64").astype(str)
                         .where(gap["bfenummer"].notna(), gap["ejerlav"] + "/" + gap["matrikelnr"]),
            "address": pd.NA, "kom": 101, "jurisdiction": "Copenhagen",
            "property_type": "Not assessed (exempt or unvalued)",
            "condo_units": 0, "land": np.where(gap["property_land_use_category"] == "Exempt", 0.0, np.nan),
            "ejd": np.nan, "leg_g": np.nan, "leg_ejd": np.nan, "valuation": pd.NA,
            "floor_area_sqm": np.nan, "district": gap["district"],
            "property_land_use_category": np.where(gap["property_land_use_category"] == "Exempt",
                                                   "Exempt", "Unknown"),
            "category_source": "not in valuation data",
            "geometry": gap.geometry.values,
        }, geometry="geometry", crs=dawa.crs)
        # A BFE already shipped as a property keeps its own polygon; a BFE split over several
        # jordstykker is unioned into one feature (SKILL §2a — never keep one piece).
        n_pieces = len(gap)
        gap = gap[~gap["parcel_id"].isin(feat["parcel_id"])]
        gap = gap.dissolve(by="parcel_id", aggfunc="first", as_index=False)
        print(f"gap fill: {n_pieces:,} jordstykker -> {len(gap):,} unassessed Copenhagen properties "
              f"({gap.to_crs(UTM).area.sum()/1e6:,.1f} km²)")
        feat = gpd.GeoDataFrame(pd.concat([feat, gap], ignore_index=True), geometry="geometry", crs=feat.crs)

    # ── canonical value fields ──────────────────────────────────────────────────
    area = feat.to_crs(UTM).area
    land = pd.to_numeric(feat["land"], errors="coerce")
    ejd = pd.to_numeric(feat["ejd"], errors="coerce")
    feat["current_full_land_value"] = land
    feat["full_market_value"] = ejd
    feat["improvement_value"] = (ejd - land).clip(lower=0)
    # Old-system building value. Display only, plus the "is this lot built?" signal the parking
    # classifier's parcel-context check needs where the current valuation has no building value.
    feat["improvement_value_legacy"] = (pd.to_numeric(feat["leg_ejd"], errors="coerce")
                                        - pd.to_numeric(feat["leg_g"], errors="coerce")).clip(lower=0)
    feat["land_area_sqm"] = area
    feat["geom_area_sqm"] = area
    feat["land_area_sqft"] = area * SQM_TO_SQFT
    feat["land_area_acres"] = area * SQM_TO_ACRES
    feat["land_value_per_sqm"] = (land / area).where(area > 0)
    feat["land_value_per_sqft"] = feat["land_value_per_sqm"] / SQM_TO_SQFT
    feat["likely_remnant"] = (area < 5).astype(int)
    feat["valuation"] = feat["valuation"].map(valuation_label)
    feat["exemption_flag"] = 0  # per non-US product decision: everything visible + filterable.

    # Refined: Vacant / Underdeveloped. Building share of value uses the CURRENT valuation where it
    # publishes a property value (owner homes), else the same-vintage legacy figures.
    cur_share = ((ejd - land) / ejd).where(ejd > 0)
    leg_share = ((feat["leg_ejd"] - feat["leg_g"]) / feat["leg_ejd"]).where(feat["leg_ejd"] > 0)
    impr_share = cur_share.fillna(leg_share)
    feat["building_share_basis"] = np.where(cur_share.notna(), "current valuation",
                                   np.where(leg_share.notna(), "legacy (pre-2020) valuation", None))
    catv = feat["property_land_use_category"]
    # Copenhagen: benyttelseskode 09 (ubebygget areal). Frederiksberg has no use code: no building
    # value in the legacy valuation AND no building footprint on the lot.
    is_frb = feat["jurisdiction"] == "Frederiksberg"
    vacant = (catv == "Vacant") | (is_frb & (leg_share == 0) & feat["bldg_cat"].isna()
                                   & (land > 0) & (feat["condo_units"] == 0))
    feat.loc[vacant, "property_land_use_category"] = "Vacant"
    excluded = vacant | catv.isin(["Exempt", "Unknown"]) | (feat["condo_units"] > 0) | ~(land > 0)
    under = ~excluded & (impr_share < 0.5)
    feat["property_land_use_refined"] = np.where(vacant, "Vacant", np.where(under, "Underdeveloped", None))

    feat["link"] = ("https://www.vurderingsportalen.dk/kort/#pos=19/"
                    + feat.representative_point().y.round(6).astype(str) + "/"
                    + feat.representative_point().x.round(6).astype(str))

    keep = ["geometry", "parcel_id", "address", "jurisdiction", "district", "property_type",
            "condo_units", "valuation", "property_land_use_category", "property_land_use_refined",
            "category_source", "building_share_basis", "exemption_flag", "likely_remnant",
            "current_full_land_value", "full_market_value", "improvement_value", "improvement_value_legacy",
            "land_value_per_sqft", "land_value_per_sqm", "land_area_sqm", "land_area_sqft",
            "land_area_acres", "geom_area_sqm", "floor_area_sqm", "link"]
    out = gpd.GeoDataFrame(feat[keep].copy(), geometry="geometry", crs=feat.crs).to_crs(4326)
    for c in ("parcel_id", "address", "jurisdiction", "district", "property_type", "valuation",
              "property_land_use_category", "property_land_use_refined", "category_source",
              "building_share_basis", "link"):
        out[c] = out[c].astype("object").where(out[c].notna(), None)

    # ── post-flight checks ─────────────────────────────────────────────────────
    assert not out["parcel_id"].duplicated().any(), "duplicate parcel_id in output"
    exp_units = agg.loc[agg.index.isin(moder["bfenummer"]), "land"].sum()
    print(f"units' land on shipped condo buildings: {exp_units/1e9:,.2f} mia. kr")

    path = os.path.join(data_dir, f"{slug}-parcels.parquet")
    out.to_parquet(path, index=False)
    print(f"\n✅ {len(out):,} features -> {path}")
    print(out.groupby("jurisdiction").agg(n=("parcel_id", "size"),
          land_bn=("current_full_land_value", lambda s: s.sum() / 1e9),
          km2=("land_area_sqm", lambda s: s.sum() / 1e6)).to_string())
    print(out.groupby("property_type").agg(n=("parcel_id", "size"),
          land_bn=("current_full_land_value", lambda s: s.sum() / 1e9),
          kr_m2=("land_value_per_sqm", "median")).to_string())
    print(out["valuation"].value_counts(dropna=False).to_string())
    print(out["property_land_use_category"].value_counts(dropna=False).to_string())
    print(out["property_land_use_refined"].value_counts(dropna=False).to_string())
    tot = out["current_full_land_value"].sum()
    print(f"\ntotal grundværdi: {tot:,.0f} kr ({tot/1e9:.1f} mia. kr)")


if __name__ == "__main__":
    sys.exit(main())
