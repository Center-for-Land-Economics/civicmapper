#!/usr/bin/env python3
"""
Build the City of Tacoma, WA canonical parcel parquet.

Tacoma is in Pierce County. Pierce County GIS publishes ONE countywide "Tax Parcels" layer
that carries geometry + assessor Land_Value / Improvement_Value / Taxable_Value + a 4-digit
Use_Code + Exemption_Code + Land_Acres, so this is a one-stop source (no separate value
join). It is countywide (~339k polygons), so we clip to the City of Tacoma boundary
(representative point within), never on City_State (that is the taxpayer MAILING address).

Sources
-------
- Parcels: Pierce County GIS "Tax Parcels" (same schema either way):
    * local shapefile export (default; DBF field names truncated to 10 chars), e.g.
      ~/Dropbox/andy/data/base/Cadastral/WA/Pierce_Parcels.shp  (NAD83 HARN WA South ftUS)
    * live service:
      https://services2.arcgis.com/1UvBaQ5y1ubjUPmd/ArcGIS/rest/services/Tax_Parcels/FeatureServer/0
  Use code list: https://www.piercecountywa.gov/DocumentCenter/View/604/Use-Code-Listing
- City boundary: Pierce County "Cities_in_Pierce_County" feature service (same AGOL org).
  Fallback (--tac-clip): Pierce County tax-area codes that belong to the City of Tacoma.

Schema findings (profiled 2026-10-18 on the 2025-11 export, Tacoma TACs = 75,708 rows)
--------------------------------------------------------------------------------------
- Land_Acres agrees with polygon area (median ratio 0.9998) for Base Parcels.
- RetiredDate is 1899-12-30 (= null) for live parcels; anything else is retired.
- TaxParcelType: Base Parcel 68.3k, Condominium 6.8k, Airspace Condominium 0.4k, misc.
- CONDOS (Olympia-style, playbook §5 / SKILL §6b): every condo unit is its own small
  polygon (median 762 sqft, Land_Acres 0) stacked inside the development's ground parcel.
  100% of Tacoma condo units sit inside a Base Parcel; 265 of the 266 host parcels are
  Use 0000 'UNKNOWN', Exemption 'EXR', $0 -- the condo common-element land. A naive
  exempt filter deletes exactly these, leaving units floating in 3D. We capture the hosts
  FIRST and merge the units DOWN onto them, summing unit land + improvement value.
- Duplicate TaxParcelNumber rows (1,537 ids, all condo/airspace): one 'Residence'/'Office'
  row carries the value, the others are $0 'Parking Space'/'Garage'/'Storage' sub-units.
  Exactly ONE valued row per id (asserted below), so summing == taking the valued row; no
  broadcast inflation (SKILL §2/§2a). Base Parcels have no duplicate ids.
- Exemption_Code: blank = taxable. Rather than hand-pick codes, a parcel is treated as
  fully exempt when it has an exemption code AND Taxable_Value <= 10% of land+improvement.
  That drops government (EX1-EX4), religious (EX5), schools (EXS), nonprofit/college (EX6,
  EX8), state-assessed rail/utility (EXA/EXB/EXC), tribal (TRIBE, NAMR), $0 'EXR' records
  and keeps partial programs on normal market parcels (EXM multifamily tax exemption,
  EX7/EX9 partials, 2HIST historic, 1HOME, PEL/PEI, DPA). Senior relief is a separate
  field (Tax_Senior_Exemption) and never drops a parcel.

Outputs
-------
- data/jurisidictions/data/tacoma/tacoma-wa-parcels.parquet
- data/jurisidictions/data/tacoma/tacoma-wa-parcels_YYYY_MM_DD.parquet

Run from the repo root:
    python data/jurisidictions/run_tacoma.py --src ~/Dropbox/andy/data/base/Cadastral/WA/Pierce_Parcels.shp
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "data"))
from parcel_calculations import (  # noqa: E402
    add_improvement_ratio_fields,
    check_area_agreement,
    classify_property_refined,
    gis_area_sqft,
)

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "tacoma"
DEFAULT_SRC = Path.home() / "Dropbox/andy/data/base/Cadastral/WA/Pierce_Parcels.shp"
CITIES_SERVICE = ("https://services2.arcgis.com/1UvBaQ5y1ubjUPmd/ArcGIS/rest/services/"
                  "Cities_in_Pierce_County/FeatureServer")
# Pierce tax-area codes inside the City of Tacoma (005 is the city proper; the rest are small
# special-district overlays). 515/195 (Parkland/Midland), 752/751 (University Place), 760
# (Lakewood), 070/071 (Fircrest), 066/067 (Fife), 105 (Ruston) are NOT Tacoma.
TACOMA_TACS = {"005", "006", "010", "011", "015", "025", "026"}
# Generous lon/lat read window around Tacoma (incl. the Port / Tideflats and NE Tacoma).
READ_BBOX = (-122.60, 47.14, -122.30, 47.35)
SQFT_PER_ACRE = 43560.0
REMNANT_SQFT = 500
EXEMPT_TAXABLE_SHARE = 0.10

# Shapefile DBF names are truncated to 10 chars; the REST service uses full names.
RENAME = {
    "TaxParcelN": "TaxParcelNumber", "TaxParcelT": "TaxParcelType",
    "TaxParce_1": "TaxParcelUnitType", "RetiredDat": "RetiredDate",
    "Improvemen": "Improvement_Value", "Taxable_Va": "Taxable_Value",
    "Tax_Area_C": "Tax_Area_Code", "Landuse_De": "Landuse_Description",
    "Exemption_": "Exemption_Code", "Site_Addre": "Site_Address",
    "Tax_Senior": "Tax_Senior_Exemption",
}
CONDO_TYPES = {"Condominium", "Airspace Condominium"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── boundary ──────────────────────────────────────────────────────────────────
def fetch_boundary():
    """City of Tacoma polygon (EPSG:4326) from Pierce County's Cities layer.

    The layer id / name field are discovered rather than hard-coded: take the first polygon
    layer and the feature whose any text attribute equals 'TACOMA'."""
    import requests
    meta = requests.get(CITIES_SERVICE, params={"f": "json"}, timeout=60).json()
    layer_id = meta["layers"][0]["id"]
    r = requests.get(f"{CITIES_SERVICE}/{layer_id}/query", params={
        "where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson"}, timeout=120)
    r.raise_for_status()
    cities = gpd.read_file(io.BytesIO(r.content))
    txt = cities.drop(columns="geometry").astype(str).apply(lambda c: c.str.strip().str.upper())
    hit = cities[(txt == "TACOMA").any(axis=1) | txt.apply(lambda c: c.str.startswith("CITY OF TACOMA")).any(axis=1)]
    if hit.empty:
        raise RuntimeError(f"No TACOMA feature in {CITIES_SERVICE}/{layer_id}; "
                           f"columns: {list(cities.columns)}")
    poly = unary_union([g.buffer(0) for g in hit.to_crs(4326).geometry])
    log(f"Tacoma boundary: {len(hit)} feature(s) from layer {layer_id}; "
        f"bbox {[round(x, 4) for x in hit.total_bounds]}")
    return poly


# ── load ──────────────────────────────────────────────────────────────────────
def load_parcels(src: Path) -> gpd.GeoDataFrame:
    window = gpd.GeoSeries([box(*READ_BBOX)], crs=4326)
    log(f"Reading {src} (Tacoma window)...")
    g = gpd.read_file(src, bbox=window)
    g = g.rename(columns=RENAME)
    log(f"  {len(g):,} polygons in window; CRS {g.crs}")
    for c in ["Land_Acres", "Land_Value", "Improvement_Value", "Taxable_Value"]:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0)
    for c in ["TaxParcelNumber", "TaxParcelType", "TaxParcelUnitType", "Use_Code",
              "Tax_Area_Code", "Landuse_Description", "Exemption_Code"]:
        g[c] = g[c].fillna("").astype(str).str.strip()
    # Live parcels carry RetiredDate = 1899-12-30 (the Esri null date) or nothing.
    rd = pd.to_datetime(g["RetiredDate"], errors="coerce")
    live = rd.isna() | (rd.dt.year < 1900)
    log(f"  dropping {int((~live).sum()):,} retired parcels")
    g = g[live].copy()
    g["geometry"] = g.geometry.apply(lambda x: x if x is None or x.is_valid else x.buffer(0))
    return g[g.geometry.notna() & ~g.geometry.is_empty].copy()


# ── classification (Pierce 4-digit DOR-style use codes) ───────────────────────
def categorize(code: str, desc: str) -> str:
    c = (code or "").strip()
    d = (desc or "").upper()
    if not c.isdigit() or c == "0000":
        return "Other"
    n = int(c)
    p2 = n // 100
    if n in (1152, 1155) or p2 == 15 or n == 1440:
        return "Mobile Home"
    if p2 == 11:
        return "Single Family"            # 1101 SFR; 1197 comm land w/ SFR
    if p2 == 12:
        return "Small Multifamily (2-4)"
    if p2 == 13:
        return "Multifamily"
    if p2 == 14:
        return "Condominium"
    if p2 == 16:
        return "Commercial"                # hotels / motels
    if p2 in (17, 18):
        return "Other Residential"         # institutional lodging, group quarters, retirement
    if 21 <= p2 <= 39:
        return "Industrial"
    if p2 in (41, 45):
        return "Right of Way"
    if p2 == 46:
        return "Parking"
    if p2 in (42, 43, 44, 47, 48, 49):
        return "Utility / Transportation"
    if n in (5020, 5030, 5060):
        return "Commercial Condominium"
    if p2 == 63 and n != 6390:
        return "Industrial"                # warehousing / storage
    if 50 <= p2 <= 66 or p2 == 69 and n != 6910:
        return "Commercial"
    if p2 in (67, 68) or n == 6910:
        return "Institutional"             # govt, schools, religious
    if n == 7630:
        return "Common Area"
    if p2 == 76:
        return "Park / Open Space"
    if 70 <= p2 <= 79:
        return "Recreation / Cultural"
    if 81 <= p2 <= 89:
        return "Resource / Agricultural"
    if n in (9140, 9178, 9188):
        return "Land (Improvement Elsewhere)"   # 'LND WITH IMPROV LAND VAL ONLY'
    if p2 == 91 or n == 9900:
        return "Vacant Land"
    if p2 in (93,):
        return "Water"
    if p2 == 94:
        return "Current Use Open Space"
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="Pierce Tax Parcels shapefile / gpkg / geojson")
    ap.add_argument("--tac-clip", action="store_true",
                    help="clip on Tacoma tax-area codes instead of the city boundary "
                         "(offline fallback; boundary clip is preferred)")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_parcels(args.src.expanduser())
    work_crs = raw.crs                        # WA South state plane, US feet
    rep = raw.geometry.representative_point()

    # ── clip to the City of Tacoma ───────────────────────────────────────────
    if args.tac_clip:
        inside = raw["Tax_Area_Code"].isin(TACOMA_TACS)
        log(f"TAC clip: {int(inside.sum()):,} of {len(raw):,}")
    else:
        poly = fetch_boundary()
        rep4326 = rep.to_crs(4326)
        inside = rep4326.within(poly)
        log(f"Boundary clip (rep point within): {int(inside.sum()):,} of {len(raw):,}")
        tac_in = raw["Tax_Area_Code"].isin(TACOMA_TACS)
        log(f"  cross-check vs Tacoma TACs: both {int((inside & tac_in).sum()):,}, "
            f"boundary-only {int((inside & ~tac_in).sum()):,}, TAC-only {int((~inside & tac_in).sum()):,}")
        log(f"  boundary-only TACs: {raw.loc[inside & ~tac_in, 'Tax_Area_Code'].value_counts().head(8).to_dict()}")
    g = raw[inside.values].copy()

    # ── account dedup preflight (SKILL §2a) ───────────────────────────────────
    dkeys = g[g["TaxParcelNumber"].duplicated(keep=False)]
    log(f"TaxParcelNumbers with >1 polygon row: {dkeys['TaxParcelNumber'].nunique():,} "
        f"(types {dkeys['TaxParcelType'].value_counts().to_dict()})")
    valued = ((dkeys["Land_Value"] > 0) | (dkeys["Improvement_Value"] > 0)).groupby(dkeys["TaxParcelNumber"]).sum()
    if len(valued):
        assert valued.max() <= 1, ("a duplicated TaxParcelNumber has >1 valued row; decide "
                                   "sum vs first before continuing (SKILL §2a)")
        log("  every duplicated id has at most ONE valued row -> summing is exact (no broadcast)")
    assert not g.loc[g["TaxParcelType"] == "Base Parcel", "TaxParcelNumber"].duplicated().any(), \
        "Base Parcels with duplicate ids: union geometry + first() values before going on"

    # ── exemption flag (unit level) ──────────────────────────────────────────
    tot = g["Land_Value"] + g["Improvement_Value"]
    has_code = g["Exemption_Code"].ne("")
    g["exemption_flag"] = (has_code & (g["Taxable_Value"] <= EXEMPT_TAXABLE_SHARE * tot)).astype(int)

    # ── condo merge: units DOWN onto the ground parcel that contains them ─────
    is_unit = g["TaxParcelType"].isin(CONDO_TYPES)
    units = g[is_unit].copy()
    base = g[~is_unit].copy()
    # A few non-condo ids ('Building Only') also repeat: union the geometry and SUM the values
    # (exactly one valued row per id, asserted above) so the output is one row per parcel.
    bd = base["TaxParcelNumber"].duplicated(keep=False)
    if bd.any():
        d = base[bd]
        agg = d.groupby("TaxParcelNumber").agg(
            {**{c: "first" for c in d.columns if c not in
                ("geometry", "TaxParcelNumber", "Land_Value", "Improvement_Value", "Taxable_Value")},
             "Land_Value": "sum", "Improvement_Value": "sum", "Taxable_Value": "sum"})
        agg["geometry"] = d.groupby("TaxParcelNumber").geometry.apply(unary_union)
        agg = gpd.GeoDataFrame(agg.reset_index(), geometry="geometry", crs=work_crs)
        base = gpd.GeoDataFrame(pd.concat([base[~bd], agg], ignore_index=True),
                                geometry="geometry", crs=work_crs)
        log(f"  non-condo duplicate ids collapsed: {d['TaxParcelNumber'].nunique():,}")
    log(f"Condo/airspace unit rows: {len(units):,} (exempt {int(units['exemption_flag'].sum()):,})")
    units = units[units["exemption_flag"] == 0]
    base_rp = gpd.GeoDataFrame(base[["TaxParcelNumber"]].rename(columns={"TaxParcelNumber": "host"}),
                               geometry=base.geometry, crs=work_crs)
    upts = gpd.GeoDataFrame(units.drop(columns="geometry"),
                            geometry=units.geometry.representative_point(), crs=work_crs)
    j = gpd.sjoin(upts, base_rp, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    units["host"] = j["host"].reindex(units.index)
    log(f"  units inside a ground parcel: {units['host'].notna().mean():.1%}")

    hosts = units.dropna(subset=["host"]).groupby("host").agg(
        u_land=("Land_Value", "sum"), u_impr=("Improvement_Value", "sum"),
        u_taxable=("Taxable_Value", "sum"), n_units=("TaxParcelNumber", "nunique"),
        u_use=("Landuse_Description", lambda s: s.value_counts().index[0]),
        u_code=("Use_Code", lambda s: s.value_counts().index[0]))
    base = base.merge(hosts, left_on="TaxParcelNumber", right_index=True, how="left")
    is_host = base["n_units"].notna()
    log(f"  merging into {int(is_host.sum()):,} ground parcels "
        f"({base.loc[is_host, 'Exemption_Code'].value_counts().head(3).to_dict()})")
    # The host keeps its own value (almost always $0 common element) plus the units' value.
    # A $0 'UNKNOWN' common-element host takes the units' dominant use code.
    for src, dst in [("u_land", "Land_Value"), ("u_impr", "Improvement_Value"),
                     ("u_taxable", "Taxable_Value")]:
        base.loc[is_host, dst] = base.loc[is_host, dst] + base.loc[is_host, src]
    adopt = is_host & base["Use_Code"].isin(["", "0000"])
    base.loc[adopt, "Use_Code"] = base.loc[adopt, "u_code"]
    base.loc[adopt, "Landuse_Description"] = base.loc[adopt, "u_use"]
    # A host is a real market parcel once it carries its units: recompute its exempt flag.
    tot = base["Land_Value"] + base["Improvement_Value"]
    base.loc[is_host, "exemption_flag"] = (
        base.loc[is_host, "Exemption_Code"].ne("")
        & ~base.loc[is_host, "Exemption_Code"].eq("EXR")
        & (base.loc[is_host, "Taxable_Value"] <= EXEMPT_TAXABLE_SHARE * tot[is_host])).astype(int)
    base["_condo_merged"] = is_host.astype(int)

    # Orphan units (no containing ground parcel): keep, unioned per TaxParcelNumber.
    orphans = units[units["host"].isna()]
    if len(orphans):
        agg = orphans.groupby("TaxParcelNumber").agg(
            {**{c: "first" for c in orphans.columns if c not in
                ("geometry", "TaxParcelNumber", "Land_Value", "Improvement_Value", "Taxable_Value")},
             "Land_Value": "sum", "Improvement_Value": "sum", "Taxable_Value": "sum"})
        agg["geometry"] = orphans.groupby("TaxParcelNumber").geometry.apply(unary_union)
        orphans = gpd.GeoDataFrame(agg.reset_index(), geometry="geometry", crs=work_crs)
        orphans["_condo_merged"] = 0
        log(f"  orphan units kept as their own polygons: {len(orphans):,}")
        base = pd.concat([base, orphans], ignore_index=True)
    p = gpd.GeoDataFrame(base, geometry="geometry", crs=work_crs)
    p = p.drop(columns=["u_land", "u_impr", "u_taxable", "n_units", "u_use", "u_code", "host"],
               errors="ignore")
    log(f"After condo merge -> {len(p):,} parcels")

    # ── category + exempt / non-market filter ────────────────────────────────
    p["property_land_use_category"] = [categorize(c, d) for c, d in
                                       zip(p["Use_Code"], p["Landuse_Description"])]
    NON_MARKET = {"Right of Way", "Utility / Transportation", "Water"}
    nonmarket = p["property_land_use_category"].isin(NON_MARKET)
    p.loc[nonmarket, "exemption_flag"] = 1
    log(f"Exempt/non-market dropped: {int(p['exemption_flag'].sum()):,} "
        f"(codes {p.loc[p['exemption_flag'] == 1, 'Exemption_Code'].value_counts().head(10).to_dict()})")
    ex = p[p["exemption_flag"] == 0].copy()

    ex["land_value"] = ex["Land_Value"].astype(float)
    ex["improvement_value"] = ex["Improvement_Value"].astype(float)
    ex["property_land_use_refined"] = classify_property_refined(
        ex, sf_cutoff=0.67, other_cutoff=0.50,
        exclude_categories=("Other", "Common Area", "Park / Open Space", "Institutional",
                            "Current Use Open Space", "Land (Improvement Elsewhere)",
                            "Resource / Agricultural"),
        category_col="property_land_use_category",
        land_col="land_value", improvement_col="improvement_value",
        fetch_footprints=False)

    # ── areas + canonical fields ─────────────────────────────────────────────
    ex = ex.to_crs(4326)
    ex["geom_area_sqft"] = ex.geometry.apply(gis_area_sqft)
    stated = ex["Land_Acres"] * SQFT_PER_ACRE
    check_area_agreement(ex["geom_area_sqft"], stated.where(stated > 0), label="Land_Acres", log=log)
    # Merged condo developments: the host polygon IS the land. Otherwise prefer stated acres.
    use_stated = (stated > 0) & (ex["_condo_merged"] == 0)
    ex["land_area_sqft"] = np.where(use_stated, stated, ex["geom_area_sqft"])
    ex["area_source"] = np.where(use_stated, "reported", "gis")
    ex["land_area_acres"] = ex["land_area_sqft"] / SQFT_PER_ACRE
    ex["likely_remnant"] = (ex["land_area_sqft"] < REMNANT_SQFT).astype(int)

    ex["full_market_value"] = ex["land_value"] + ex["improvement_value"]
    den = ex["land_area_sqft"].replace(0, np.nan)
    ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
    ex["land_value_per_sqft"] = ex["land_value"] / den
    ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
    ex = add_improvement_ratio_fields(ex, land_col="land_value", improvement_col="improvement_value")
    ex["parcel_number"] = ex["TaxParcelNumber"]
    ex["link"] = ("https://atip.piercecountywa.gov/app/v2/propertyDetail/"
                  + ex["TaxParcelNumber"] + "/summary")

    COLUMNS = ["geometry", "parcel_number", "exemption_flag", "property_land_use_category",
               "property_land_use_refined", "full_market_value", "full_market_value_per_sqft",
               "land_value", "land_value_per_sqft", "improvement_value",
               "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO", "IMPR_LAND_PCT",
               "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source", "likely_remnant"]
    for c in COLUMNS:
        if c not in ex.columns:
            ex[c] = np.nan
    final = gpd.GeoDataFrame(ex[COLUMNS].rename(columns={"land_value": "current_full_land_value"}),
                             geometry="geometry", crs=4326)

    # ── post-flight checks (playbook §4-§8, SKILL §6a) ────────────────────────
    assert final["parcel_number"].is_unique, "output must be one row per parcel"
    lv = final["land_value_per_sqft"]
    gp = final.to_crs(32610)
    a = gp.geometry.area * 10.7639
    c = gp.geometry.representative_point()
    vc = (c.x.round(1).astype(str) + "," + c.y.round(1).astype(str)).value_counts()
    holes = final.geometry.apply(lambda x: sum(len(q.interiors) for q in
                                               (x.geoms if x.geom_type == "MultiPolygon" else [x])))
    log(f"rows {len(final):,} | bounds {[round(v, 4) for v in final.total_bounds]}")
    log(f"stacked clusters: {int((vc > 1).sum())} (max stack {int(vc.max())})")
    log(f"footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}; "
        f"sub-1000 sqft carrying value: {int(((a < 1000) & (final['full_market_value'] > 0)).sum())}")
    log(f"land $/sqft p50/p99/max: {lv.median():.0f} / {lv.quantile(.99):.0f} / {lv.max():.0f}")
    log(f"parcels with holes: {int((holes > 0).sum())}; likely_remnant: {int(final['likely_remnant'].sum())}")
    log(f"land total ${final['current_full_land_value'].sum() / 1e9:.2f}B; "
        f"improvements ${final['improvement_value'].sum() / 1e9:.2f}B")
    log(f"category: {final['property_land_use_category'].value_counts().to_dict()}")
    log(f"refined: {final['property_land_use_refined'].value_counts(dropna=False).to_dict()}")

    out = args.out_dir / "tacoma-wa-parcels.parquet"
    final.to_parquet(out, index=False)
    final.to_parquet(args.out_dir / f"tacoma-wa-parcels_{datetime.now():%Y_%m_%d}.parquet", index=False)
    log(f"SAVED {out}")


if __name__ == "__main__":
    main()
