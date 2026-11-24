#!/usr/bin/env python3
"""
Build the City of Saint Paul, MN canonical parcel parquet (replaces st_paul.ipynb).

Source — the City of Saint Paul's copy of Ramsey County's "Attributed Tax Parcels" schema,
city-only, published in the city's PAULIE service (AGOL org `9meaaHE3uiba0zr8`, public):
    https://services1.arcgis.com/9meaaHE3uiba0zr8/arcgis/rest/services/PAULIE/FeatureServer/1
83,420 polygons, every one SiteCityName = 'SAINT PAUL'. ONE-STOP: geometry + Ramsey County
assessor values (EMVLand / EMVBuilding = estimated market value, 100% of market) + land-use code
+ TaxExemptYN on the same layer. TaxYear 2027 / EMVYear 2026 (the 2026 assessment).
Ramsey County's own server (maps.co.ramsey.mn.us/.../OpenData/FeatureServer/12, which the old
notebook used) was unreachable from the build environment (skill §1).

WHY THIS REPLACES st_paul.ipynb (re-evaluated 2026-11-23):
- Stale: the shipped parquet matched Ramsey's 2024 values (EMVYear2) on 99.9% of parcels.
- Singularities: condo / condo-garage / co-op / TICO-townhome units are each mapped as an
  IDENTICAL COPY of the development's lot polygon (470 stacks, up to ~600 deep). The notebook
  divided each unit's value by its own ParcelSquareFeet (~871 sqft for a condo unit, the unit's
  floor area / undivided share, not land), so a $200k unit became a ~$230/sqft needle, hundreds
  of them stacked on one spot. Here every stack is collapsed to ONE parcel with values SUMMED
  (distinct per-unit records, skill §2a "sum"), and every per-sqft rate uses the polygon's own
  geodesic area — ParcelSquareFeet is never a denominator.
- City filter was the situs-city field (skill §4 trap); this layer is the city's own fabric.

EXEMPT: TaxExemptYN='Y' (the assessor values exempt property too: 3,410 parcels, $2.31B land),
plus any 6xx exempt land-use code. Railroad / public-utility operating property (8xx, 900) is
dropped as state-assessed (same call as run_minneapolis.py). $0-ish association outlots
(458 CI/condo/co-op non-tax outlot, 577 townhome non-tax outlot) that are not under units are
dropped rather than shipped as error parcels.

CONDO LAND IS NOMINAL — the assessor's own number, kept (Seattle Westlake / Gwinnett precedent):
Ramsey County gives each residential condo unit a token land value (median $1,000; garage units
$500, storage $100) and puts the rest in the building. A merged condo development therefore has
little land value (~$1-3/sqft against ~$6-10 for its neighbours). It is published as assessed and
flagged `condo_land_nominal = 1` so the popup can say why. Co-ops and TICO townhomes DO carry real
land shares ($20k / $50k per unit) and merge to normal rates.

CONDO LAND ESTIMATE (skill §6d, run_boston.py / run_stlouis.py method; on by default):
left at the assessor's nominal figure, merged condo developments render as near-zero holes and
make every neighbouring house look like a 50-160x spike (Summit / Portland / Ashland Ave in Ramsey
Hill: houses at $20-25/sqft next to condo buildings at $0.13-0.28). For each merged development
whose land is nominal (< $3/sqft), land is ESTIMATED as lot area x the median land $/sqft of the
15 nearest non-condo, single-record, valued parcels, capped at 70% of the development's total
value. TOTAL value is unchanged: the estimate moves value from building to land. The assessor's
figure is kept in `assessor_land_value`, the row is flagged `condo_land_imputed = 1`, and
`--no-condo-impute` ships the assessor's figure instead.

ECONOMIC UNITS: Ramsey sometimes puts a whole development's land value on ONE small parcel and
codes the rest of the site as separate same-owner "vacant land" parcels (1845 University Ave W:
a 234-unit 2021 apartment complex with $4.68M land on a 6,358 sqft parcel = $735/sqft, beside
four same-owner "apartment vacant land" parcels at ~$6/sqft). A built parcel is unioned with its
same-owner, building-less parcels reachable through touching same-owner building-less parcels,
only when its own land rate is > $50/sqft AND > 5x theirs; values summed (distinct records).
Narrow on purpose (a looser rule merged ~100 ordinary houses with their side-yard lots); logged
parcel by parcel.

RECORDED AREA: ~20 single-record parcels have an assessor area (ParcelSquareFeet) > 3x their
polygon (310 Smith Ave N parking ramp: 188,615 sqft recorded, 13,625 drawn). Their rates use the
recorded area (area_source = 'assessor'), so the value is not piled onto a sliver.

KNOWN, KEPT: Ramsey values apartment land at a flat $20,000 per unit and townhome land at flat
per-unit amounts ($30k/$50k/$100k), regardless of lot size. Old walk-ups and townhome rows on small
lots therefore show 5-15x the land $/sqft of the houses beside them (535 Asbury St: 18 units on
3,578 sqft = $100/sqft). That is the assessor's method, published as assessed.

Owner / taxpayer / homestead names are read (for the economic-unit match only) but never exported.

Outputs:
- data/jurisidictions/data/stpaul/st-paul-mn-parcels.parquet  (registry: stpaul -> st-paul-mn)
- data/jurisidictions/data/stpaul/st-paul-mn-parcels_YYYY_MM_DD.parquet
PMTiles + H3 (~72k parcels), baked with --drop-remnants.

AFTER THIS SCRIPT (it resets property_land_use_refined, which has no parking code):
    python data/scripts/parking_rejoin.py --city stpaul --keep-columns
    python data/scripts/apply_parking_category.py --city stpaul \
        --parcels data/jurisidictions/data/stpaul/st-paul-mn-parcels.parquet \
        --parking data/parking/stpaul/stpaul-mn-parking-lots.parquet
then bake PMTiles and upload (parcels + parking).
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import (add_improvement_ratio_fields, classify_property_refined,  # noqa: E402
                                 gis_area_sqft)

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "stpaul"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "stpaul-mn-ramsey-2026.parquet"
URL = "https://services1.arcgis.com/9meaaHE3uiba0zr8/arcgis/rest/services/PAULIE/FeatureServer/1"
LINK_FMT = "https://www.ramseycounty.us/residents/property-home/property-tax-and-value-lookup?pid={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36"}
SQFT_PER_ACRE = 43560.0
UTM = "EPSG:26915"
CONDO_K = 15            # nearest donors for the condo land estimate
CONDO_CAP = 0.70        # estimate never exceeds this share of the development's total value
CONDO_NOMINAL_PSF = 3.0 # merged condo land below this $/sqft counts as nominal
ECON_RATIO = 5.0        # economic-unit merge: built parcel land $/sqft > this x its partners'
ECON_MIN_PSF = 50.0     # ...and > this on its own (St. Paul single-family p99 is ~$37)
IMPUTE = "--no-condo-impute" not in sys.argv


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch() -> gpd.GeoDataFrame:
    if CACHE.exists():
        log(f"Using cached {CACHE.name}")
        return gpd.read_parquet(CACHE)
    ids = requests.get(f"{URL}/query", params={"where": "1=1", "returnIdsOnly": "true", "f": "json"},
                       headers=HEADERS, timeout=120).json()
    oids = sorted(ids["objectIds"])
    log(f"Pulling {len(oids):,} parcels")
    parts = []
    for i in range(0, len(oids), 2000):
        lo, hi = oids[i], oids[min(i + 2000, len(oids)) - 1]
        for attempt in range(5):
            try:
                r = requests.get(f"{URL}/query", params={
                    "where": f"OBJECTID>={lo} AND OBJECTID<={hi}", "outFields": "*",
                    "returnGeometry": "true", "outSR": 4326, "f": "geojson"},
                    headers=HEADERS, timeout=240)
                r.raise_for_status()
                j = r.json()
                break
            except Exception as e:  # noqa: BLE001
                log(f"  retry {attempt + 1} @ {lo}: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"pull failed at objectid {lo}")
        parts.append(gpd.GeoDataFrame.from_features(j["features"], crs="EPSG:4326"))
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    g.to_parquet(CACHE, index=False)
    return g


raw = fetch()
log(f"Rows: {len(raw):,}; SiteCityName {raw['SiteCityName'].value_counts().to_dict()}; "
    f"EMVYear {raw['EMVYear'].value_counts().to_dict()}")
assert raw["ParcelID"].notna().all() and not raw["ParcelID"].duplicated().any()
raw = raw[raw["EMVLand"].notna() & raw.geometry.notna()].copy()
raw["lu"] = pd.to_numeric(raw["LandUseCode"], errors="coerce").fillna(0).astype(int)
raw["geometry"] = raw["geometry"].apply(lambda g: g if g.is_valid else g.buffer(0))

# ── exempt / out of scope ─────────────────────────────────────────────────────
ex = raw["TaxExemptYN"].eq("Y") | raw["lu"].between(600, 699)
log(f"EXEMPT dropped: {int(ex.sum()):,} parcels, land ${raw.loc[ex, 'EMVLand'].sum():,.0f}")
rr = ~ex & (raw["lu"] >= 800)
log(f"Railroad / utility operating property dropped: {int(rr.sum())}, land ${raw.loc[rr, 'EMVLand'].sum():,.0f}")
df = raw[~ex & ~rr].copy()
TOT_LAND, TOT_BLDG = float(df["EMVLand"].sum()), float(df["EMVBuilding"].sum())

# ── stacks of identical footprints (condo / co-op / TICO units) -> one parcel, SUM ──
pu = df.to_crs(UTM)
rp = pu.geometry.representative_point()
pu["_stack"] = (rp.x.round(1).astype(str) + "," + rp.y.round(1).astype(str) + ","
                + pu.geometry.area.round(0).astype(str))
vc = pu["_stack"].value_counts()
stacked = pu["_stack"].map(vc).gt(1)
log(f"Stacked identical footprints: {int((vc > 1).sum())} stacks, {int(stacked.sum()):,} records "
    f"(max {int(vc.max())}); {pu.loc[stacked, 'LandUseCodeDescription'].value_counts().head(8).to_dict()}")
CONDO_LU = {550, 551, 552, 553, 559, 434, 450, 483}
single = pu[~stacked].copy()
single["n_units"] = 1
rows = []
for _, grp in pu[stacked].groupby("_stack", sort=False):
    lead = grp.sort_values("EMVTotal", ascending=False).iloc[0].to_dict()
    for c in ("EMVLand", "EMVBuilding", "EMVTotal"):
        lead[c] = float(grp[c].sum())
    lead["n_units"] = int(len(grp))
    lead["geometry"] = unary_union(list(grp.geometry.values)).buffer(0)
    w = grp.groupby("lu")["EMVTotal"].sum()
    lead["lu"] = int(w.idxmax())
    lead["LandUseCodeDescription"] = grp.loc[grp["lu"].eq(lead["lu"]), "LandUseCodeDescription"].iloc[0]
    lead["_condo"] = bool(grp["lu"].isin(CONDO_LU).any())
    lead["SiteAddress"] = str(lead.get("SiteAddress") or "")
    rows.append(lead)
merged = gpd.GeoDataFrame(rows, geometry="geometry", crs=UTM)
single["_condo"] = single["lu"].isin(CONDO_LU)
p = gpd.GeoDataFrame(pd.concat([single, merged], ignore_index=True), geometry="geometry", crs=UTM)
assert abs(p["EMVLand"].sum() - TOT_LAND) < 1 and abs(p["EMVBuilding"].sum() - TOT_BLDG) < 1
log(f"  -> {len(merged)} merged parcels; total {len(p):,}")

outlot = p["lu"].isin([458, 577]) & p["n_units"].eq(1)
log(f"Free-standing association outlots dropped: {int(outlot.sum())} (land ${p.loc[outlot, 'EMVLand'].sum():,.0f})")
p = p[~outlot].reset_index(drop=True)

# ── economic units: one parcel carrying a development's land beside same-owner vacant pieces ──
p["_sqft_u"] = p.geometry.area * 10.763910416709722
p["_psf_u"] = p["EMVLand"] / p["_sqft_u"]
_own = p["OwnerName"].fillna("").astype(str).str.strip().str.upper()
_hits = gpd.sjoin(gpd.GeoDataFrame({"i": p.index}, geometry=p.geometry.buffer(0.5), crs=UTM),
                  gpd.GeoDataFrame({"j": p.index}, geometry=p.geometry, crs=UTM), predicate="intersects")
_hits = _hits[(_hits["i"] != _hits["j"]) & (_own.values[_hits["i"]] == _own.values[_hits["j"]])
              & (_own.values[_hits["i"]] != "")]
_vac = (p["EMVBuilding"].eq(0) & p["EMVLand"].gt(0) & p["n_units"].eq(1)).to_numpy()
_adj = _hits.groupby("i")["j"].apply(list).to_dict()
# Only a built parcel whose land rate is implausibly high on its own is a candidate; its partners
# are the same-owner, building-less parcels reachable through chains of touching same-owner
# building-less parcels (1845 University's site is four pieces, two of which touch only each other).
_cand = p.index[(p["EMVBuilding"] > 0) & (p["_psf_u"] > ECON_MIN_PSF)]
_drop, _new = set(), []
for i in _cand:
    if i in _drop:
        continue
    partners, todo = [], [j for j in _adj.get(i, []) if _vac[j]]
    while todo:
        j = todo.pop()
        if j in partners or j in _drop:
            continue
        partners.append(j)
        todo += [k for k in _adj.get(j, []) if _vac[k] and k != i]
    if not partners:
        continue
    part_psf = p.loc[partners, "EMVLand"].sum() / p.loc[partners, "_sqft_u"].sum()
    if p.at[i, "_psf_u"] <= ECON_RATIO * part_psf:
        continue
    row = p.loc[i].to_dict()
    for c in ("EMVLand", "EMVBuilding", "EMVTotal"):
        row[c] = float(p.loc[[i] + partners, c].sum())
    row["geometry"] = unary_union([p.at[k, "geometry"] for k in [i] + partners]).buffer(0)
    row["n_units"] = int(p.loc[[i] + partners, "n_units"].sum())
    _new.append(row)
    _drop |= {i, *partners}
    log(f"  economic unit: {p.at[i, 'ParcelID']} {p.at[i, 'SiteAddress']} ${p.at[i, '_psf_u']:,.0f}/sqft + "
        f"{len(partners)} same-owner vacant piece(s) at ${part_psf:,.2f}/sqft -> "
        f"${row['EMVLand'] / (row['geometry'].area * 10.7639):,.2f}/sqft")
if _new:
    p = gpd.GeoDataFrame(pd.concat([p.drop(index=list(_drop)), gpd.GeoDataFrame(_new, geometry="geometry", crs=UTM)],
                                   ignore_index=True), geometry="geometry", crs=UTM)
log(f"Economic-unit merges: {len(_new)} ({len(_drop)} parcels)")
assert abs(p["EMVLand"].sum() - TOT_LAND) < 1
p = p.drop(columns=["_sqft_u", "_psf_u"]).to_crs("EPSG:4326")


# ── classification (Ramsey County land-use codes) ─────────────────────────────
def categorize(lu: int, condo: bool) -> str:
    if lu in (550, 551, 552, 553):
        return "Condominium"
    if lu == 495:
        return "Co-op"
    if lu in (300,):
        return "Vacant Industrial"
    if lu in (400,):
        return "Vacant Commercial"
    if lu in (405, 500, 559, 579):
        return "Vacant Residential"
    if lu in (456, 457):
        return "Parking Garage/Ramp"
    if lu == 109:
        return "Agricultural / Rural"
    if 300 < lu < 400 or 478 <= lu <= 485:
        return "Industrial"
    if lu in (401, 402, 403, 404, 406, 407, 408, 413, 517):
        return "Multifamily"
    if lu == 429:
        return "Mixed Use"
    if 400 < lu < 500:
        return "Commercial"
    if lu in (505, 509, 510, 515, 545, 570, 573, 574, 575):
        return "Single Family"
    if lu in (520, 530, 540):
        return "Residential 2-3 Units"
    if lu in (547, 578):
        return "Residential Other"
    return "Other"


p["property_land_use_category"] = [categorize(l, c) for l, c in zip(p["lu"], p["_condo"])]
p["exemption_flag"] = 0

# ── canonical fields ──────────────────────────────────────────────────────────
p["land_area_sqft"] = p["geometry"].apply(gis_area_sqft)
p.loc[p["land_area_sqft"] < 1, "land_area_sqft"] = np.nan
_r = (p["land_area_sqft"] / pd.to_numeric(p["ParcelAcresPolygon"], errors="coerce").mul(SQFT_PER_ACRE)
      .where(p["n_units"].eq(1)).replace(0, np.nan))
log(f"Area cross-check vs ParcelAcresPolygon: median {_r.median():.4f}, >5% off {int((_r.sub(1).abs() > .05).sum())}")
p["area_source"] = "gis"
p["likely_remnant"] = (p["land_area_sqft"] < 500).astype(int)
# A single record whose ASSESSOR area is > 3x its polygon is valued for land the polygon does not
# show (310 Smith Ave N parking ramp: 188,615 sqft recorded, 13,625 sqft drawn -> $374/sqft).
# Divide by the recorded area for those (~20 parcels) so the rate is the assessor's own.
_rec = pd.to_numeric(p["ParcelSquareFeet"], errors="coerce")
_use_rec = p["n_units"].eq(1) & (_rec > 3 * p["land_area_sqft"]) & (_rec > 5000)
p.loc[_use_rec, "land_area_sqft"] = _rec[_use_rec]
p.loc[_use_rec, "area_source"] = "assessor"
log(f"Recorded area > 3x polygon (rate uses recorded area): {int(_use_rec.sum())} parcels")
p["land_area_acres"] = p["land_area_sqft"] / SQFT_PER_ACRE
p["land_value"] = p["EMVLand"].astype(float)
p["improvement_value"] = p["EMVBuilding"].astype(float)
p["full_market_value"] = p["land_value"] + p["improvement_value"]
den = p["land_area_sqft"]
p["full_market_value_per_sqft"] = p["full_market_value"] / den
p["land_value_per_sqft"] = p["land_value"] / den
p["improvement_value_per_sqft"] = p["improvement_value"] / den

# ── condo land estimate (skill §6d) ───────────────────────────────────────────
from scipy.spatial import cKDTree  # noqa: E402

p["assessor_land_value"] = p["land_value"]
p["condo_land_imputed"] = 0
_is_condo = p["property_land_use_category"].eq("Condominium").to_numpy() | (p["_condo"].astype(bool).to_numpy()
                                                                            & p["n_units"].gt(1).to_numpy())
_psf = (p["land_value"] / p["land_area_sqft"]).to_numpy()
_elig = _is_condo & (_psf < CONDO_NOMINAL_PSF) & p["land_area_sqft"].notna().to_numpy()
_donor = (~_is_condo & p["n_units"].eq(1).to_numpy() & (p["land_value"] > 0).to_numpy()
          & (p["land_area_sqft"] >= 500).to_numpy() & (p["improvement_value"] > 0).to_numpy())
_pts = p.to_crs(UTM).geometry.representative_point()
_xy = np.column_stack([_pts.x.to_numpy(), _pts.y.to_numpy()])
_est_psf = np.full(len(p), np.nan)
_, _idx = cKDTree(_xy[_donor]).query(_xy[_elig], k=CONDO_K)
_est_psf[_elig] = np.median(_psf[_donor][_idx], axis=1)
_tot = p["full_market_value"].to_numpy()
_est = np.minimum(_est_psf * p["land_area_sqft"].to_numpy(), CONDO_CAP * _tot)
_apply = _elig & ~np.isnan(_est) & (_est > p["land_value"].to_numpy())
log(f"Condo developments with nominal land: {int(_elig.sum())}; estimate "
    f"{'APPLIED' if IMPUTE else 'computed, NOT applied'} to {int(_apply.sum())} "
    f"(+${(_est[_apply] - p['land_value'].to_numpy()[_apply]).sum() / 1e6:,.1f}M land, moved from building; "
    f"cap binding on {int((_apply & (_est_psf * p['land_area_sqft'].to_numpy() > CONDO_CAP * _tot)).sum())})")
if IMPUTE:
    p.loc[_apply, "land_value"] = np.round(_est[_apply], 0)
    p.loc[_apply, "improvement_value"] = p.loc[_apply, "full_market_value"] - p.loc[_apply, "land_value"]
    p.loc[_apply, "condo_land_imputed"] = 1
    p["land_value_per_sqft"] = p["land_value"] / den
    p["improvement_value_per_sqft"] = p["improvement_value"] / den
p = add_improvement_ratio_fields(p, land_col="land_value", improvement_col="improvement_value")
p["bld_ar"] = pd.to_numeric(p.get("LivingAreaSquareFeet"), errors="coerce").fillna(0) + \
    pd.to_numeric(p.get("BusinessSquareFeet"), errors="coerce").fillna(0)
_refined = classify_property_refined(
    p, sf_cutoff=0.67, other_cutoff=0.50,
    exclude_categories=("Agricultural / Rural", "Other", "Condominium"),
    category_col="property_land_use_category", land_col="land_value",
    improvement_col="improvement_value", bld_ar_col="bld_ar", fetch_footprints=False)
_refined[(p["condo_land_imputed"] == 1) & (_refined == "Underdeveloped")] = None  # an estimate, not a finding
p["property_land_use_refined"] = _refined

p["parcel_id"] = p["ParcelID"].astype(str)
p["link"] = p["parcel_id"].map(LINK_FMT.format)
p["address"] = p["SiteAddress"].fillna("").astype(str).str.strip()
p.loc[p["n_units"].gt(1), "address"] = p["address"].str.replace(r"\s+(UNIT|#|APT)\s*\S+$", "", regex=True)
p["assessor_land_use"] = p["LandUseCodeDescription"].fillna("").astype(str).str.title()
p["n_condo_units"] = p["n_units"].astype(int)
p["condo_land_nominal"] = (_elig & ~(p["condo_land_imputed"].eq(1).to_numpy())).astype(int)
p["tax_year"] = 2027

COLUMNS = ["geometry", "parcel_id", "address", "assessor_land_use", "n_condo_units", "condo_land_nominal",
           "assessor_land_value", "condo_land_imputed",
           "tax_year", "exemption_flag", "property_land_use_category", "property_land_use_refined",
           "full_market_value", "full_market_value_per_sqft", "land_value", "land_value_per_sqft",
           "improvement_value", "improvement_value_per_sqft", "TLLDIMPROV", "IMPR_LAND_RATIO",
           "IMPR_LAND_PCT", "IMPR_PCT_TOTAL", "link", "land_area_acres", "area_source", "likely_remnant"]
final = gpd.GeoDataFrame(p[COLUMNS].rename(columns={"land_value": "current_full_land_value"}),
                         geometry="geometry", crs="EPSG:4326")
assert not final["parcel_id"].duplicated().any()
out = DATA_DIR / "st-paul-mn-parcels.parquet"
final.to_parquet(out, index=False)
final.to_parquet(DATA_DIR / f"st-paul-mn-parcels_{datetime.now():%Y_%m_%d}.parquet", index=False)
log(f"SAVED {out} | rows {len(final):,} | land ${final.current_full_land_value.sum():,.0f} "
    f"impr ${final.improvement_value.sum():,.0f}")
log(f"category: {final.property_land_use_category.value_counts().to_dict()}")
log(f"refined: {final.property_land_use_refined.value_counts(dropna=False).to_dict()}")

# ── smoke alarms (skill §6a) ──────────────────────────────────────────────────
lv = final["land_value_per_sqft"]
tv = final["full_market_value_per_sqft"]
a = final["land_area_acres"] * SQFT_PER_ACRE
log(f"  footprint sqft p1/p5: {round(a.quantile(.01))} / {round(a.quantile(.05))}; sub-500 {int((a < 500).sum())}")
log(f"  land $/sqft p50/p99/max ${lv.median():.2f} / ${lv.quantile(.99):.2f} / ${lv.max():.2f}")
log(f"  total $/sqft p50/p99/max ${tv.median():.2f} / ${tv.quantile(.99):.2f} / ${tv.max():.2f}")
for c in ("Condominium", "Co-op", "Single Family", "Commercial", "Multifamily"):
    s = lv[final.property_land_use_category.eq(c)]
    log(f"    {c:22s} n={len(s):6,} land $/sqft p50 ${s.median():.2f} p99 ${s.quantile(.99):.2f}")
rp2 = final.geometry.representative_point()
vc2 = (rp2.x.round(5).astype(str) + "," + rp2.y.round(5).astype(str)).value_counts()
log(f"  stacked clusters: {int((vc2 > 1).sum())}, max {int(vc2.max())}")
log(f"  zero land: {int((final.current_full_land_value <= 0).sum())}; condo_land_nominal: "
    f"{int(final.condo_land_nominal.sum())}")
for r in final.sort_values("full_market_value_per_sqft", ascending=False).head(8).itertuples():
    log(f"    top total $/sqft {r.parcel_id} {r.address[:28]:28s} {r.property_land_use_category:20s} "
        f"{r.land_area_acres * SQFT_PER_ACRE:9,.0f} sqft ${r.full_market_value_per_sqft:,.0f} (land ${r.land_value_per_sqft:,.0f})")
log("DONE")
