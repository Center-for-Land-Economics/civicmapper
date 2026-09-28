#!/usr/bin/env python3
"""
Build the St. Louis, MO canonical parcel parquet: City of St. Louis + St. Louis County.

ONE CivicMapper city (`stlouis`, `stlouis-mo-parcels.*`) stitched from the two Missouri
assessing jurisdictions that make up the core of the metro (DMV / Greater Boston pattern).
The City of St. Louis is an INDEPENDENT city (FIPS 29510, its own assessor, in no county);
St. Louis County (FIPS 29189) surrounds it on the Missouri side. The Illinois side of the metro
is out of scope. Every parcel carries
  - `jurisdiction`  — "City of St. Louis" / "St. Louis County"   (region toggle, group 1)
  - `municipality`  — the City, each of the county's 88 municipalities, or
                      "Unincorporated St. Louis County"             (region toggle, group 2)

MISSOURI ASSESSES AT A FRACTION OF MARKET (RSMo 137.115): residential 19%, commercial 32%,
agricultural 12%. Both sources publish the assessor's APPRAISED (market) value next to the
assessed value, and this build ships the APPRAISED figures. The ETL asserts the 19% / 32%
ratios so a schema change is noticed (Georgia-40% precedent, run_duluth.py).

SOURCES (public ArcGIS REST, no token; verified 2026-09-28)
- City of St. Louis — Assessor's parcel layer ("Prcl" schema), 134,358 records, tax year 2026:
    https://maps6.stlouis-mo.gov/arcgis/rest/services/St_Louis_Parcels/MapServer/0
  Market value = AprLand (appraised land) + CostAprImprove (appraised improvements); this sum is
  exactly the "Appraised total" on the city's own Address & Property Search page. AsdLand /
  AsdImprove are the 19%/32% assessed figures (not used). Codes: PropertyClassCode 17 = Exempt;
  AsrClassCode = exempt-entity code (49 = Land Reutilization Authority, 6 = parks, 1 = Board of
  Education, 14 = State of Missouri, 13 = Housing Authority, ...); AsrLandUse1 = the city's
  4-digit land-use vocabulary (stlouis-mo.gov/data/vocabularies ids 23 / 64 / 24).
- St. Louis County — Tax Parcels (OpenData), 401,481 records, TAXYR 2026:
    https://maps.stlouisco.com/hosting/rest/services/OpenData/OpenData/FeatureServer/7
  Market value = APPLANDVAL + APPIMPVAL (= TOTAPVAL exactly); ASSTLANDVAL etc. are assessed.
  TAXCODE is the county's tax-EXEMPT code (county table item d71ee5cf9cef4137b9781042abae20a9):
  A = Taxable; B-E county, F state, G MoDOT, H school district, J fire, K sewer, L library,
  M municipality, N special district, O lighting, P/Q utility, R common ground, S religious,
  T parochial school, U private school, V cemetery, X charity, Y US government (plus W =
  fraternal/veterans halls and POW = former-POW homestead exemption, both seen in the roll).
  The county appraises exempt property too, so TAXCODE is the ONLY exemption signal — the
  PROPCLASS / LUC fields put Washington University and the county government center in
  "Commercial".
Owner / mailing / legal-description / tax-balance / owner-occupancy fields are never requested
from either source (issue #12).

TRAPS HANDLED
1. CITY CONDOS: every unit is its own record, STACKED on the lot polygon and sharing the lot's
   HANDLE, and the city assesses condo units with ~$0 land (Massachusetts-style, skill §6d):
   ~7.7k units on ~520 lots, $7.7M land vs $1.72B improvements. Units are summed onto their lot. The lot's land
   is then ESTIMATED like run_boston.py (lot area x median land $/sqft of the 15 nearest
   assessor-valued non-condo City parcels, capped at 70% of the lot's value), flagged
   `condo_land_imputed = 1`, assessor figure kept in `assessor_land_value`;
   `--no-condo-impute` ships the assessor's $0 instead.
2. COUNTY CONDOS: units are stacked on the development's PLAT polygon (PARENT_LOC = 'PL...',
   no record of its own) but each unit carries its OWN appraised land (the county splits land
   among units). Units are summed onto one development footprint per PARENT_LOC; no estimate.
3. Split parcels: a City HANDLE / County LOCATOR can repeat across polygons. Each account's value
   is counted ONCE (skill §2 — first, never sum) and the polygons are unioned.
4. City LRA (Land Reutilization Authority, AsrClassCode 49): ~9.6k city-owned (mostly vacant)
   lots. Exempt, appraised at token values, dropped with the rest of the exempt roll; the
   summary reports how much there is.
5. City tax-abated parcels (Chapter 99/353 redevelopment + IsAbatedProperty): the assessor does
   NOT estimate market value for these — "appraised" is just assessed / ratio, which can sit far
   below market. Kept (they are taxable) but flagged `tax_abated = 1` for the popup.
6. Utility / railroad / right-of-way classes are excluded (skill §5).

Outputs:
- data/jurisidictions/data/stlouis/stlouis-mo-parcels.parquet  (+ dated snapshot)
- viz/public/stlouis-{jurisdiction,municipality}-overlay.geojson

Usage:
    python data/jurisidictions/run_stlouis.py                 # build (uses raw/ caches)
    python data/jurisidictions/run_stlouis.py --no-condo-impute
    python data/jurisidictions/run_stlouis.py --fetch-only
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data"))
from parcel_calculations import (  # noqa: E402
    add_improvement_ratio_fields, classify_property_refined, gis_area_sqft)

DATA_DIR = ROOT / "data" / "jurisidictions" / "data" / "stlouis"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DIR = ROOT / "viz" / "public"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh) CivicMapper-ETL", "Accept": "application/json"}
SQFT_PER_ACRE = 43560.0
MO_EPSG = 26915  # NAD83 / UTM zone 15N (metres)

CITY = "City of St. Louis"
COUNTY = "St. Louis County"

CITY_URL = "https://maps6.stlouis-mo.gov/arcgis/rest/services/St_Louis_Parcels/MapServer/0/query"
CITY_FIELDS = ",".join([
    "OBJECTID", "HANDLE", "ParcelId", "LowerAsrParcelId", "ColParcelId", "CityBlock", "Parcel",
    "GeoCityBlockPart", "SITEADDR", "LowAddrNum", "HighAddrNum", "StPreDir", "StName", "StType",
    "StdUnitNum", "ZIP", "AsrClassCode", "PropertyClassCode", "AsrLandUse1", "AsrLanduse2",
    "CDALandUse1", "VacantLot", "Condominium", "NbrOfUnits", "NbrOfApts", "NbrOfCondos",
    "NbrOfSubAccts", "SpecParcelType", "SubParcelType", "AcctPrimary", "LRMSUnitNum", "LRMSParcel",
    "FRONTAGE", "LANDAREA", "SQFT", "AsdLand", "AsdImprove", "AsdTotal", "BillLand", "BillImprove",
    "BillTotal", "AprLand", "CostAprImprove", "IsAbatedProperty", "AbatementStartYear",
    "AbatementEndYear", "RedevPhase", "TIFDist", "Zoning", "NbrOfBldgsRes", "NbrOfBldgsCom",
    "FirstYearBuilt", "LastYearBuilt", "VacBldgYear", "NBRHD", "WARD20", "OwnerCode", "LastDate",
])
COUNTY_URL = "https://maps.stlouisco.com/hosting/rest/services/OpenData/OpenData/FeatureServer/7/query"
COUNTY_FIELDS = ",".join([
    "OBJECTID", "PARENT_LOC", "LOCATOR", "TAXYR", "PROP_ADRNUM", "PROP_ADD", "PROP_ZIP", "MUNYCODE",
    "SUBDIVISION", "TAXCODE", "ASSTLANDVAL", "ASSTIMPVAL", "TOTASSMT", "APPLANDVAL", "APPIMPVAL",
    "TOTAPVAL", "PROPCLASS", "LUC", "LANDUSE2", "LUCODE", "LIVUNIT", "YEARBLT", "RESQFT", "COMSTRUC",
    "ACRES", "NBHD", "LANDUSE3", "BLDGNAME", "MUNICIPALITY", "ZONING",
])
COUNTY_MUNI_URL = "https://maps.stlouisco.com/hosting/rest/services/OpenData/OpenData/FeatureServer/6/query"
TIGER_COUNTY_URL = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/13/query"

CITY_LINK = "https://www.stlouis-mo.gov/data/address-search/index.cfm?parcelid={}&firstview=true"
COUNTY_LINK = "https://revenue.stlouisco.com/RealEstate/MapsPropertyInfo.aspx?LocatorNum={}"

# City exempt-entity codes (AsrClassCode < 100) that are NOT exemptions: 0 = unset, 50 = planned
# industrial, 55 = 5-yr obsolete-district abatement, 99 = Chapter redevelopment, 81 = State Tax
# Commission (state-assessed railroad/utility — excluded as Utility instead).
CITY_NON_EXEMPT_ENTITY = {0, 50, 55, 81, 99}
CITY_ABATEMENT_CODES = {55, 99, 147, 148, 150, 153, 155, 199, 235, 247, 248, 250, 253, 255, 299}
COUNTY_TAXABLE = "A"

CONDO_K = 15
CONDO_CAP = 0.70
UNSPLIT_MIN_SHARE = 0.5


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fetch (cached to raw/)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_geojson(query_url: str, where: str, out_fields: str, page: int = 2000,
                  order_by: str = "OBJECTID") -> gpd.GeoDataFrame:
    total = requests.get(query_url, params={"where": where, "returnCountOnly": "true", "f": "json"},
                         headers=HEADERS, timeout=120).json().get("count")
    log(f"  pulling {total:,} features from {query_url.split('/services/')[1]}")
    pages, off = [], 0
    while True:
        gdf = None
        for attempt in range(8):
            try:
                r = requests.get(query_url, params={
                    "where": where, "outFields": out_fields, "returnGeometry": "true",
                    "resultOffset": off, "resultRecordCount": page, "outSR": 4326,
                    "orderByFields": order_by, "f": "geojson"}, headers=HEADERS, timeout=300)
                r.raise_for_status()
                # ArcGIS answers overload with HTTP 200 + an {"error": ...} body (Provo trap).
                if r.content[:1] != b"{" or b'"error"' in r.content[:200]:
                    raise RuntimeError(f"non-GeoJSON body: {r.content[:160]!r}")
                gdf = gpd.read_file(io.BytesIO(r.content))
                break
            except Exception as e:  # noqa: BLE001
                log(f"    retry {attempt + 1} @off {off}: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(4 * (attempt + 1))
        if gdf is None:
            raise RuntimeError(f"page failed at offset {off}: {query_url}")
        if not len(gdf):
            break
        pages.append(gdf)
        off += len(gdf)
        if off % 20000 < page:
            log(f"    {off:,} / {total:,}")
        if total is not None and off >= total:
            break
    out = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    if total is not None and len(out) != total:
        raise RuntimeError(f"pulled {len(out):,} of {total:,} — truncated pull")
    return out


def _cached(name: str, url: str, where: str, fields: str, order_by: str = "OBJECTID") -> gpd.GeoDataFrame:
    cache = RAW_DIR / name
    if cache.exists():
        return gpd.read_parquet(cache)
    log(f"Pulling {name} ...")
    g = fetch_geojson(url, where, fields, order_by=order_by)
    g.to_parquet(cache, index=False)
    return g


def load_city() -> gpd.GeoDataFrame:
    return _cached("city-parcels.parquet", CITY_URL, "1=1", CITY_FIELDS)


def load_county() -> gpd.GeoDataFrame:
    return _cached("county-parcels.parquet", COUNTY_URL, "1=1", COUNTY_FIELDS)


def load_county_munis() -> gpd.GeoDataFrame:
    return _cached("county-municipalities.parquet", COUNTY_MUNI_URL, "1=1",
                   "OBJECTID,MUNICIPALITY,MUNI,MUNICODE")


def load_tiger_counties() -> gpd.GeoDataFrame:
    return _cached("tiger-counties.parquet", TIGER_COUNTY_URL,
                   "STATE='29' AND COUNTY IN ('510','189')", "OBJECTID,NAME,GEOID")


# ─────────────────────────────────────────────────────────────────────────────
# Classifiers
# ─────────────────────────────────────────────────────────────────────────────
def city_category(lu: int, impr: float, total: float) -> str:
    """City of St. Louis AsrLandUse1 (4-digit, stlouis-mo.gov vocabulary id 24)."""
    if lu in (1010, 1015):
        return "Vacant Land"
    if lu in (1017, 1019) or 8000 <= lu < 8300:
        return "Agricultural"
    if lu in (1110, 1111, 1310):
        return "Single Family"
    if lu in (1114, 1115, 1314, 1315):
        return "Condominium"
    if lu == 1116:
        return "Parking Garage"
    if lu in (1120, 1130, 1140, 1320, 1322, 1330, 1340):
        return "Two to Four Family"
    if lu == 1118 or 1121 <= lu <= 1139 or lu in (1150, 1155, 1160, 1170, 1180, 1185, 1190, 1350, 1360, 1385):
        return "Apartments (5+ units)"   # incl. 1121-1139 multi-building residential, 1118 Sec. 42
    if lu in (1145, 1307):
        return "Mixed Use"
    if 1200 <= lu < 1300:
        return "Group Quarters"
    if lu == 1300 or lu in (6260, 6261, 6262):
        return "Hotel"
    if 1000 <= lu < 1500:
        return "Other Residential"
    if 2000 <= lu < 4000:
        return "Industrial"
    if lu in (4216, 4223, 4224, 4630, 4640):
        return "Parking Lot"
    if lu == 4620:
        return "Parking Garage"
    if lu in (4600, 4610):  # generic / paid parking: a lot unless the building dominates
        return "Parking Lot" if (total <= 0 or impr / total < 0.25) else "Parking Garage"
    if lu == 4500:
        return "Right of Way"
    if 4100 <= lu < 4120:
        return "Railroad"
    if 4700 <= lu < 4900 or 4910 <= lu <= 4912:
        return "Utility"
    if 4000 <= lu < 5000:
        return "Industrial"
    if lu == 5115:
        return "Commercial Condominium"
    if 5100 <= lu < 5200:
        return "Industrial"
    if lu == 5000:
        return "Commercial"            # generic 'TRADE' — the city's catch-all for storefronts/offices
    if 5000 <= lu < 6000:
        return "Retail"
    if 6100 <= lu < 6200 or lu == 6399 or 6500 <= lu < 6510 or 6520 <= lu < 6600:
        return "Office"
    if 6370 <= lu <= 6377:
        return "Industrial"
    if 6510 <= lu <= 6519:
        return "Medical"
    if 6700 <= lu < 7000:
        return "Institutional"
    if lu in (7600, 7610, 7620):
        return "Open Space"
    if 7000 <= lu < 8000:
        return "Recreation & Entertainment"
    if 6000 <= lu < 6700:
        return "Commercial"
    if 9100 <= lu < 9200 or lu in (9000, 9900):
        # 9111/9141/9151/9171 "vacant and open", 9112/... "vacant and boarded" are BUILDINGS
        # standing empty, not empty land — the improvement-ratio test judges them.
        if lu in (9100, 9000, 9900, 9110, 9120, 9130, 9140, 9150, 9160, 9170):
            return "Vacant Land" if impr <= 0 else "Unoccupied Building"
        return "Unoccupied Building"
    if lu in (9510, 9520):
        return "Under Construction"
    return "Other"


def county_category(luc: str, impr: float, total: float, units: float = 0) -> str:
    """St. Louis County LUC (3-digit; descriptions from the county's STC parcel layer)."""
    try:
        c = int(str(luc).strip())
    except ValueError:
        return "Other"
    if c == 110:
        return "Single Family"
    if c == 115:
        # 'Multi-Family' — duplexes/townhomes up to Clayton apartment towers; split on units.
        return "Apartments (5+ units)" if units >= 5 else "Two to Four Family"
    if c in (116, 692):
        return "Apartments (5+ units)"  # senior apartments / LIHTC
    if 120 <= c <= 125:
        return "Group Quarters"
    if c == 130 or 150 <= c <= 159:
        return "Hotel"
    if c == 140:
        return "Other Residential"
    if c == 190:
        return "Other Residential"      # 'Other Than Dwelling' (garages, sheds on a res lot)
    if 200 <= c < 400:
        return "Industrial"
    if c == 411:
        return "Railroad"
    if c == 431:
        return "Airport"
    if c in (454, 457):
        return "Right of Way"
    if c == 460:
        return "Parking Lot" if (total <= 0 or impr / total < 0.25) else "Parking Garage"
    if 470 <= c <= 491:
        return "Utility"
    if 400 <= c < 500:
        return "Industrial"
    if c in (505, 603, 606):
        return "Commercial Condominium"
    if 500 <= c < 600:
        return "Retail"
    if c in (601, 602, 604) or 611 <= c <= 615 or c in (652, 659):
        return "Office"
    if c in (605, 637):
        return "Industrial"
    if c == 651:
        return "Medical"
    if 670 <= c < 700:
        return "Institutional"
    if c == 761:
        return "Open Space"
    if 700 <= c < 800:
        return "Recreation & Entertainment"
    if 800 <= c < 900:
        return "Agricultural"
    if c == 910:
        return "Vacant Land"
    if c == 921:
        return "Open Space"
    if 600 <= c < 700:
        return "Commercial"
    return "Other"


DROP_CATEGORIES = ("Utility", "Railroad", "Right of Way")


# ─────────────────────────────────────────────────────────────────────────────
# Adapters -> records (one row per assessor account) + footprints (one row per fkey)
#   records: jurisdiction, rec_id, fkey, land, bldg, total, category, use_code, use_desc,
#            exempt, lra, abated, is_unit, condo_marker, stated_sqft, address, link_id, fy
# ─────────────────────────────────────────────────────────────────────────────
def city_adapter(stats: dict) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    g = load_city()
    s = stats.setdefault(CITY, {})
    s["raw_records"] = int(len(g))
    for c in ["AprLand", "CostAprImprove", "AsdLand", "AsdImprove", "AsdTotal", "LANDAREA"]:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
    for c in ["AsrClassCode", "PropertyClassCode", "AsrLandUse1", "IsAbatedProperty", "Condominium",
              "VacantLot"]:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0).astype(int)

    # Missouri ratio tripwire: residential 19%, commercial 32% (assessed / appraised).
    live = g[(g["AsdTotal"] > 0) & (g["IsAbatedProperty"] == 0)]
    for pc, want in ((15, 0.19), (12, 0.32)):
        d = live[live["PropertyClassCode"] == pc]
        ratio = (d["AsdLand"] + d["AsdImprove"]).sum() / (d["AprLand"] + d["CostAprImprove"]).sum()
        log(f"City class {pc}: assessed/appraised = {ratio:.4f} (expect {want})")
        if abs(ratio - want) > 0.01:
            raise SystemExit(f"City class {pc} ratio {ratio:.4f} != {want}: AprLand/CostAprImprove "
                             "may no longer be market value")

    # A HANDLE repeats for (a) condo units stacked on their lot polygon, each its own ParcelId
    # account, and (b) a parcel split across several polygons (same ParcelId). Value per account
    # is counted once: dedup on (HANDLE, ParcelId).
    g["HANDLE"] = g["HANDLE"].astype(str)
    g["rec_id"] = g["ParcelId"].fillna("OID" + g["OBJECTID"].astype(str)).astype(str)
    recs = g.sort_values("OBJECTID").drop_duplicates(["HANDLE", "rec_id"]).copy()
    s["accounts"] = int(len(recs))
    land = recs["AprLand"]
    bldg = recs["CostAprImprove"]
    total = land + bldg
    asr = recs["AsrClassCode"]
    exempt = (recs["PropertyClassCode"] == 17) | ((asr >= 1) & (asr < 100) & ~asr.isin(CITY_NON_EXEMPT_ENTITY))
    lra = asr == 49
    lu = recs["AsrLandUse1"]
    cat = [city_category(int(a), float(b), float(t)) for a, b, t in zip(lu, bldg, total)]
    cat = pd.Series(cat, index=recs.index)
    cat[asr == 81] = "Utility"
    addr = recs["SITEADDR"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    is_condo = recs["Condominium"] == -1
    rec = pd.DataFrame({
        "jurisdiction": CITY,
        "rec_id": recs["rec_id"],
        "fkey": "C" + recs["HANDLE"],
        "land": land, "bldg": bldg, "total": total,
        "category": cat,
        "use_code": lu.astype(str),
        "use_desc": "",
        "exempt": exempt,
        "lra": lra,
        "abated": (recs["IsAbatedProperty"] == -1) | asr.isin(CITY_ABATEMENT_CODES),
        "is_unit": is_condo & (land <= 0),
        "condo_marker": is_condo | recs["NbrOfCondos"].fillna(0).gt(0),
        "stated_sqft": recs["LANDAREA"].where(recs["LANDAREA"] > 0),
        "address": addr,
        "link_id": recs["ParcelId"].astype(str),
        "fy": 2026,
        "neighborhood_code": recs["NBRHD"],
    })
    polys = g.drop_duplicates(["HANDLE", "geometry"])[["HANDLE", "geometry"]].copy()
    polys["fkey"] = "C" + polys["HANDLE"]
    geo = polys.dissolve(by="fkey", as_index=False)[["fkey", "geometry"]]
    log(f"City: {len(g):,} records -> {len(recs):,} accounts on {len(geo):,} lots "
        f"(exempt accounts {int(exempt.sum()):,}, LRA {int(lra.sum()):,}, condo units {int(is_condo.sum()):,})")
    return rec, gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326")


def county_adapter(stats: dict) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    g = load_county()
    s = stats.setdefault(COUNTY, {})
    s["raw_records"] = int(len(g))
    g = g[g["LOCATOR"].notna() & g.geometry.notna()].copy()
    for c in ["APPLANDVAL", "APPIMPVAL", "TOTAPVAL", "ASSTLANDVAL", "ASSTIMPVAL", "TOTASSMT", "ACRES"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    fy = g["TAXYR"].dropna().astype(int).value_counts()
    log(f"County: {len(g):,} records; TAXYR {fy.to_dict()}")

    tx = g[g["TAXCODE"].eq(COUNTY_TAXABLE) & (g["TOTASSMT"] > 0)]
    for pc, want in (("R", 0.19), ("C", 0.32)):
        d = tx[tx["PROPCLASS"] == pc]
        ratio = d["TOTASSMT"].sum() / d["TOTAPVAL"].sum()
        log(f"County class {pc}: assessed/appraised = {ratio:.4f} (expect ~{want})")
        # Commercial lands at ~0.30: the 32% class ratio blends with the 12% ag and 19% res
        # portions of mixed parcels. Anything far off means the fields changed meaning.
        if abs(ratio - want) > 0.03:
            raise SystemExit(f"County class {pc} ratio {ratio:.4f} far from {want}")
    bad = (g["APPLANDVAL"].fillna(0) + g["APPIMPVAL"].fillna(0) - g["TOTAPVAL"].fillna(0)).abs() > 1
    if bad.sum():
        log(f"  WARNING: {int(bad.sum())} county records with land + improvement != total")

    # Condo units: PARENT_LOC is the development's plat ('PL...'), which has no record of its
    # own; the units are stacked on the plat polygon. Everything else stands on its LOCATOR.
    child = g["PARENT_LOC"].notna() & (g["PARENT_LOC"] != g["LOCATOR"])
    g["root"] = np.where(child, g["PARENT_LOC"], g["LOCATOR"])
    recs = g.sort_values("OBJECTID").drop_duplicates(["root", "LOCATOR"]).copy()
    s["accounts"] = int(len(recs))
    land = recs["APPLANDVAL"].fillna(0.0)
    bldg = recs["APPIMPVAL"].fillna(0.0)
    total = land + bldg
    luc = recs["LUC"].fillna("").astype(str).str.strip()
    units = pd.to_numeric(recs["LIVUNIT"], errors="coerce").fillna(0)
    cat = pd.Series([county_category(a, float(b), float(t), float(u))
                     for a, b, t, u in zip(luc, bldg, total, units)], index=recs.index)
    cat[child.loc[recs.index] & cat.eq("Single Family")] = "Condominium"
    taxcode = recs["TAXCODE"].fillna("").str.strip()
    stated = (recs["ACRES"] * SQFT_PER_ACRE).where(~child.loc[recs.index] & (recs["ACRES"] > 0))
    addr = recs["PROP_ADD"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    rec = pd.DataFrame({
        "jurisdiction": COUNTY,
        "rec_id": recs["LOCATOR"].astype(str),
        "fkey": "K" + recs["root"].astype(str),
        "land": land, "bldg": bldg, "total": total,
        "category": cat,
        "use_code": luc,
        "use_desc": recs["LUCODE"].fillna(""),
        "exempt": ~taxcode.eq(COUNTY_TAXABLE),
        "taxcode": taxcode,
        "lra": False,
        "abated": False,
        "is_unit": child.loc[recs.index],
        "condo_marker": child.loc[recs.index],
        "stated_sqft": stated,
        "address": addr,
        "link_id": recs["LOCATOR"].astype(str),
        "fy": recs["TAXYR"],
        "county_municipality": recs["MUNICIPALITY"],
    })
    polys = g.drop_duplicates(["root", "geometry"])[["root", "geometry"]].copy()
    polys["fkey"] = "K" + polys["root"].astype(str)
    geo = polys.dissolve(by="fkey", as_index=False)[["fkey", "geometry"]]
    log(f"County: {len(recs):,} accounts on {len(geo):,} footprints "
        f"(exempt {int(rec['exempt'].sum()):,}; condo units {int(child.sum()):,} on "
        f"{g.loc[child, 'PARENT_LOC'].nunique():,} developments)")
    return rec, gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326")


# ─────────────────────────────────────────────────────────────────────────────
# Roll accounts up to footprints
# ─────────────────────────────────────────────────────────────────────────────
def polygonal(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    parts = [p for p in getattr(geom, "geoms", []) if isinstance(p, (Polygon, MultiPolygon))]
    if not parts:
        return None
    u = unary_union(parts)
    return u if isinstance(u, (Polygon, MultiPolygon)) else None


def acres(geoms) -> float:
    # A plain float sum: GeoSeries.apply on an EMPTY series returns geometry dtype (DMV trap).
    return float(sum(gis_area_sqft(g) for g in geoms if g is not None)) / SQFT_PER_ACRE


def roll_up(rec: pd.DataFrame, geo: gpd.GeoDataFrame, stats: dict) -> gpd.GeoDataFrame:
    jur = rec["jurisdiction"].iloc[0]
    s = stats[jur]
    ex = rec[rec["exempt"]]
    s["exempt_records"] = int(len(ex))
    s["exempt_land"] = float(ex["land"].sum())
    s["exempt_total"] = float(ex["total"].sum())
    s["lra_records"] = int(rec["lra"].sum())
    s["lra_land"] = float(rec.loc[rec["lra"], "land"].sum())
    if "taxcode" in rec:
        s["exempt_by_code"] = ex.groupby("taxcode")["land"].agg(["size", "sum"]).to_dict("index")
    exempt_land_fp = ex.groupby("fkey")["land"].sum()

    tx = rec[~rec["exempt"]]
    drop_cat = tx["category"].isin(DROP_CATEGORIES)
    s["utility_records"] = int(drop_cat.sum())
    s["utility_total"] = float(tx.loc[drop_cat, "total"].sum())
    s["zero_value_records"] = int(((~drop_cat) & (tx["total"] <= 0)).sum())
    tx = tx[~drop_cat & (tx["total"] > 0)].copy()
    s["taxable_records"] = int(len(tx))
    tx["unsplit_val"] = np.where((tx["land"] <= 0) & (tx["total"] > 0), tx["total"], 0.0)
    tx["unit_n"] = tx["is_unit"].astype(int)
    tx["abated_n"] = tx["abated"].astype(int)

    gb = tx.groupby("fkey")
    fp = pd.DataFrame({
        "land": gb["land"].sum(), "bldg": gb["bldg"].sum(), "total": gb["total"].sum(),
        "unsplit_val": gb["unsplit_val"].sum(), "n_accounts": gb.size(),
        "n_condo_units": gb["unit_n"].sum(), "abated_n": gb["abated_n"].sum(), "fy": gb["fy"].max(),
    })
    # Dominant category = the one carrying the most value on the footprint.
    dom = (tx.groupby(["fkey", "category"])["total"].sum().reset_index()
           .sort_values("total", ascending=False).drop_duplicates("fkey").set_index("fkey")["category"])
    fp["category"] = dom
    lead = tx.sort_values("total", ascending=False).drop_duplicates("fkey").set_index("fkey")
    for c in ["use_code", "use_desc", "address", "link_id"]:
        fp[c] = lead[c]
    for c in ["county_municipality", "neighborhood_code"]:
        if c in lead:
            fp[c] = lead[c]
    fp["stated_sqft"] = lead["stated_sqft"].where(fp["n_accounts"] == 1)
    fp["exempt_land_on_lot"] = exempt_land_fp.reindex(fp.index).fillna(0.0)
    fp["condo_regime"] = rec.groupby("fkey")["condo_marker"].any().reindex(fp.index).fillna(False)
    fp["jurisdiction"] = jur
    fp = fp.reset_index()

    exempt_only = geo[geo["fkey"].isin(set(ex["fkey"]) - set(fp["fkey"]))]
    s["exempt_only_lots"] = int(len(exempt_only))
    s["exempt_only_acres"] = acres(exempt_only.geometry)
    lra_fk = set(rec.loc[rec["lra"], "fkey"]) - set(fp["fkey"])
    s["lra_only_acres"] = acres(geo[geo["fkey"].isin(lra_fk)].geometry)
    s["lra_only_lots"] = int(len(lra_fk))
    out = geo.merge(fp, on="fkey", how="inner")
    missing = fp[~fp["fkey"].isin(geo["fkey"])]
    s["unmapped_lots"] = int(len(missing))
    out["geometry"] = out["geometry"].apply(polygonal)
    bad = out["geometry"].isna()
    s["bad_geometry"] = int(bad.sum())
    out = out[~bad].copy()
    log(f"{jur}: {len(out):,} taxable footprints; exempt accounts {len(ex):,} "
        f"(${ex['land'].sum() / 1e9:,.2f}B land); utility/rail/ROW dropped {s['utility_records']}; "
        f"$0-value taxable dropped {s['zero_value_records']}")
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


# ─────────────────────────────────────────────────────────────────────────────
# City condo land estimate (run_boston.py method)
# ─────────────────────────────────────────────────────────────────────────────
def impute_condo_land(ex: gpd.GeoDataFrame, stats: dict, enabled: bool) -> gpd.GeoDataFrame:
    from scipy.spatial import cKDTree

    ex["assessor_land_value"] = ex["land"]
    ex["condo_land_imputed"] = 0
    share = np.divide(ex["unsplit_val"], ex["total"], out=np.zeros(len(ex)), where=ex["total"] > 0)
    is_city = (ex["jurisdiction"] == CITY).to_numpy()
    condo_lot = ex["condo_regime"].astype(bool).to_numpy() & (share >= UNSPLIT_MIN_SHARE) & is_city
    leasehold = condo_lot & (ex["exempt_land_on_lot"] > 0).to_numpy()
    eligible = condo_lot & ~leasehold
    pts = ex.to_crs(MO_EPSG).geometry.representative_point()
    xy = np.column_stack([pts.x.to_numpy(), pts.y.to_numpy()])
    psf = (ex["land"] / ex["land_area_sqft"]).to_numpy()
    donor = (is_city & (ex["land"] > 0).to_numpy() & (share < 0.1)
             & (ex["land_area_sqft"] >= 500).to_numpy() & ~ex["condo_regime"].astype(bool).to_numpy())
    s = stats[CITY]
    s["condo_lots"] = int(condo_lot.sum())
    s["condo_lots_leasehold_kept_zero"] = int(leasehold.sum())
    est_psf = np.full(len(ex), np.nan)
    if eligible.any():
        tree = cKDTree(xy[donor])
        _, idx = tree.query(xy[eligible], k=CONDO_K)
        est_psf[eligible] = np.median(psf[donor][idx], axis=1)
    est = est_psf * ex["land_area_sqft"].to_numpy()
    tot = ex["total"].to_numpy()
    new_land = np.minimum(est, CONDO_CAP * tot)
    apply = eligible & ~np.isnan(new_land) & (new_land > ex["land"].to_numpy())
    ex["condo_land_estimate_psf"] = np.where(eligible, est_psf, np.nan)
    s["condo_lots_imputed"] = int(apply.sum())
    s["condo_land_imputed_added"] = float((new_land[apply] - ex["land"].to_numpy()[apply]).sum())
    s["condo_cap_binding"] = int((apply & (est > CONDO_CAP * tot)).sum())
    new_land = np.round(new_land, 0)
    if enabled:
        ex.loc[apply, "land"] = new_land[apply]
        ex.loc[apply, "bldg"] = ex.loc[apply, "total"] - ex.loc[apply, "land"]
        ex.loc[apply, "condo_land_imputed"] = 1
    log(f"City condo lots: {int(condo_lot.sum()):,}; estimate {'APPLIED' if enabled else 'computed, NOT applied'} "
        f"to {int(apply.sum()):,} (+${s['condo_land_imputed_added'] / 1e6:,.1f}M); kept $0 (exempt land "
        f"under a taxable building): {int(leasehold.sum()):,}")
    return ex


# ─────────────────────────────────────────────────────────────────────────────
# Municipality tagging + overlays
# ─────────────────────────────────────────────────────────────────────────────
def muni_name(raw: str | None) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or not str(raw).strip():
        return None
    r = str(raw).strip().upper()
    if r in ("UNINCORPORATED", "UNI"):
        return "Unincorporated St. Louis County"
    small = {"AND", "OF"}
    words = []
    for w in r.replace("&", " & ").split():
        if w == "ST":
            words.append("St.")
        elif w in small:
            words.append(w.lower())
        else:
            words.append("-".join(p.capitalize() for p in w.split("-")))
    return " ".join(words).replace(" & ", " & ")


def tag_municipality(ex: gpd.GeoDataFrame, munis: gpd.GeoDataFrame, stats: dict) -> gpd.GeoDataFrame:
    ex["municipality"] = None
    ex.loc[ex["jurisdiction"] == CITY, "municipality"] = CITY
    k = ex["jurisdiction"] == COUNTY
    m = munis[["MUNICIPALITY", "geometry"]].copy()
    m["geometry"] = m.geometry.apply(polygonal)
    m = m[m.geometry.notna()].to_crs(MO_EPSG)
    pts = gpd.GeoDataFrame({"i": ex.index[k]}, geometry=ex.loc[k].to_crs(MO_EPSG).geometry.representative_point().values,
                           crs=MO_EPSG)
    j = gpd.sjoin(pts, m, how="left", predicate="within").drop_duplicates("i").set_index("i")
    spatial = j["MUNICIPALITY"].map(muni_name)
    attr = ex.loc[k, "county_municipality"].map(muni_name)
    agree = (spatial.reindex(attr.index) == attr)
    s = stats[COUNTY]
    s["muni_attr_spatial_agree"] = float(agree.mean())
    # The parcel's own MUNICIPALITY is the TAXING municipality from the assessor; the boundary
    # layer is the county GIS's drawing of the same line. They agree on ~all parcels; use the
    # boundary layer where it has an answer (skill: jurisdiction is spatial), else the attribute.
    ex.loc[k, "municipality"] = spatial.reindex(attr.index).fillna(attr)
    log(f"County municipality: attribute vs boundary agree on {agree.mean():.2%}; "
        f"{ex.loc[k, 'municipality'].nunique()} municipalities incl. unincorporated")
    return ex


def write_overlays(munis: gpd.GeoDataFrame, tiger: gpd.GeoDataFrame) -> None:
    t = tiger.copy()
    t["name"] = np.where(t["GEOID"] == "29510", CITY, COUNTY)
    t = t[["name", "geometry"]]
    t["geometry"] = t.to_crs(MO_EPSG).geometry.simplify(8).to_crs(4326)
    t.to_file(PUBLIC_DIR / "stlouis-jurisdiction-overlay.geojson", driver="GeoJSON",
              COORDINATE_PRECISION=5)
    m = munis[["MUNICIPALITY", "geometry"]].copy()
    m["geometry"] = m.geometry.apply(polygonal)
    m = m[m.geometry.notna()]
    m["name"] = m["MUNICIPALITY"].map(muni_name)
    m = m.dissolve(by="name", as_index=False)[["name", "geometry"]]
    # Unincorporated county is the residual — its outline is every muni's edge, so skip it.
    m = m[m["name"] != "Unincorporated St. Louis County"]
    city = t[t["name"] == CITY]
    m = pd.concat([city, m], ignore_index=True)
    m = gpd.GeoDataFrame(m, geometry="geometry", crs="EPSG:4326")
    m["geometry"] = m.to_crs(MO_EPSG).geometry.simplify(8).to_crs(4326)
    m.to_file(PUBLIC_DIR / "stlouis-municipality-overlay.geojson", driver="GeoJSON",
              COORDINATE_PRECISION=5)
    log(f"Overlays written: jurisdiction ({len(t)}), municipality ({len(m)})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--no-condo-impute", action="store_true",
                    help="Ship the City assessor's $0 condo land instead of the neighbor estimate.")
    args = ap.parse_args()

    stats: dict = {}
    c_rec, c_geo = city_adapter(stats)
    k_rec, k_geo = county_adapter(stats)
    munis = load_county_munis()
    tiger = load_tiger_counties()
    if args.fetch_only:
        return
    parts = [roll_up(c_rec, c_geo, stats), roll_up(k_rec, k_geo, stats)]
    ex = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    log(f"Combined: {len(ex):,} taxable footprints")

    # ── area: stated lot area when it agrees with the polygon (0.5-2.0x), else geodesic ──
    ex["geom_area_sqft"] = [gis_area_sqft(g) for g in ex["geometry"]]
    ex.loc[ex["geom_area_sqft"] < 1, "geom_area_sqft"] = np.nan
    rep = pd.to_numeric(ex["stated_sqft"], errors="coerce")
    ratio = rep / ex["geom_area_sqft"]
    use_rep = (rep > 0) & ratio.between(0.5, 2.0)
    ex["land_area_sqft"] = np.where(use_rep, rep, ex["geom_area_sqft"])
    ex["area_source"] = np.where(use_rep, "assessor", "gis")
    log(f"Area: assessor={int(use_rep.sum()):,} gis={int((~use_rep).sum()):,}")
    ex = ex[ex["land_area_sqft"] > 0].copy()
    ex["likely_remnant"] = (ex["land_area_sqft"] < 500).astype(int)

    ex = impute_condo_land(ex, stats, enabled=not args.no_condo_impute)
    ex = tag_municipality(ex, munis, stats)
    write_overlays(munis, tiger)

    # ── canonical fields ──
    ex["property_land_use_category"] = ex["category"]
    ex["current_full_land_value"] = ex["land"]
    ex["improvement_value"] = ex["bldg"]
    ex["full_market_value"] = ex["total"]
    ex["exemption_flag"] = 0
    ex["tax_abated"] = (ex["abated_n"] > 0).astype(int)
    refined = classify_property_refined(
        ex, sf_cutoff=0.67, other_cutoff=0.50,
        exclude_categories=("Other", "Open Space", "Airport", "Institutional"),
        category_col="property_land_use_category",
        land_col="current_full_land_value", improvement_col="improvement_value",
        fetch_footprints=False)
    refined[(ex["condo_land_imputed"] == 1) & (refined == "Underdeveloped")] = None
    ex["property_land_use_refined"] = refined
    den = ex["land_area_sqft"]
    ex["land_area_acres"] = den / SQFT_PER_ACRE
    ex["land_value_per_sqft"] = ex["current_full_land_value"] / den
    ex["improvement_value_per_sqft"] = ex["improvement_value"] / den
    ex["full_market_value_per_sqft"] = ex["full_market_value"] / den
    ex = add_improvement_ratio_fields(ex, land_col="current_full_land_value", improvement_col="improvement_value")
    is_c = ex["jurisdiction"].eq(CITY)
    ex["link"] = np.where(is_c, [CITY_LINK.format(i) for i in ex["link_id"]],
                          [COUNTY_LINK.format(i) for i in ex["link_id"]])
    ex["parcel_id"] = ex["link_id"]
    ex["tax_year"] = ex["fy"].astype("Int64").astype(str)

    # ── smoke alarms (skill §6a) ──
    gp = ex.to_crs(MO_EPSG)
    a = gp.geometry.area * 10.7639
    log(f"footprint sqft p1/p5/p10: {[round(a.quantile(q)) for q in (.01, .05, .10)]}")
    log(f"sub-500 / sub-1000 sqft valued: {int(((a < 500) & (ex['full_market_value'] > 0)).sum())} / "
        f"{int(((a < 1000) & (ex['full_market_value'] > 0)).sum())}")
    rp = gp.geometry.representative_point()
    vc = (rp.x.round(1).astype(str) + "," + rp.y.round(1).astype(str)).value_counts()
    log(f"stacked clusters: {int((vc > 1).sum())}; max stack {int(vc.max())}")
    holes = ex.geometry.apply(lambda g: sum(len(p.interiors) for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])))
    log(f"parcels with holes: {int((holes > 0).sum())}")
    for jur, d in ex.groupby("jurisdiction"):
        l = d["land_value_per_sqft"]
        sf = d[d["property_land_use_category"] == "Single Family"]["land_value_per_sqft"]
        cd = d[d["n_condo_units"] > 0]["land_value_per_sqft"]
        log(f"  {jur:18s} land $/sqft p50/p99/max {l.median():.2f}/{l.quantile(.99):.0f}/{l.max():.0f} | "
            f"SF p50 {sf.median():.2f} | condo lots p50 {cd.median():.2f} (n={len(cd):,}) | "
            f"$0-land lots {int((d['current_full_land_value'] <= 0).sum()):,}")

    # ── export ──
    COLUMNS = ["geometry", "jurisdiction", "municipality", "parcel_id", "address", "tax_year",
               "use_code", "use_desc", "exemption_flag", "property_land_use_category",
               "property_land_use_refined", "full_market_value", "full_market_value_per_sqft",
               "current_full_land_value", "land_value_per_sqft",
               "improvement_value", "improvement_value_per_sqft",
               "assessor_land_value", "condo_land_imputed", "n_accounts", "n_condo_units", "tax_abated",
               "TLLDIMPROV", "IMPR_LAND_RATIO", "IMPR_LAND_PCT", "IMPR_PCT_TOTAL",
               "link", "land_area_acres", "area_source", "likely_remnant"]
    final = gpd.GeoDataFrame(ex[COLUMNS].reset_index(drop=True), geometry="geometry", crs="EPSG:4326")
    for c in ["n_accounts", "n_condo_units", "condo_land_imputed", "likely_remnant", "exemption_flag",
              "tax_abated"]:
        final[c] = final[c].astype("int32")
    final["use_desc"] = final["use_desc"].astype(str)
    out = DATA_DIR / "stlouis-mo-parcels.parquet"
    final.to_parquet(out, index=False)
    final.to_parquet(DATA_DIR / f"stlouis-mo-parcels_{datetime.now():%Y_%m_%d}.parquet", index=False)
    log(f"SAVED {out} | rows {len(final):,}")

    log("category: " + str(final["property_land_use_category"].value_counts().to_dict()))
    log("refined: " + str(final["property_land_use_refined"].value_counts(dropna=False).to_dict()))
    print("\n=== PER-JURISDICTION SUMMARY ===")
    for jur, d in final.groupby("jurisdiction"):
        s = stats[jur]
        land = d["current_full_land_value"].sum()
        print(f"{jur}: TY{d['tax_year'].max()}  parcels {len(d):,}  land ${land / 1e9:,.2f}B "
              f"(assessor ${d['assessor_land_value'].sum() / 1e9:,.2f}B)  total ${d['full_market_value'].sum() / 1e9:,.2f}B  "
              f"acres {d['land_area_acres'].sum():,.0f}  condo units {int(d['n_condo_units'].sum()):,} on "
              f"{int((d['n_condo_units'] > 0).sum()):,} lots  abated {int(d['tax_abated'].sum()):,}")
        print(f"    raw records {s['raw_records']:,}, accounts {s['accounts']:,}; exempt {s['exempt_records']:,} "
              f"(land ${s['exempt_land'] / 1e9:,.2f}B = {s['exempt_land'] / (s['exempt_land'] + d['assessor_land_value'].sum()):.1%} "
              f"of appraised land); fully-exempt lots {s['exempt_only_lots']:,} = {s['exempt_only_acres']:,.0f} acres "
              f"({s['exempt_only_acres'] / (s['exempt_only_acres'] + d['land_area_acres'].sum()):.1%} of parcel acreage); "
              f"utility/rail/ROW dropped {s['utility_records']}; $0-value dropped {s['zero_value_records']}")
        if jur == CITY:
            print(f"    LRA: {s['lra_records']:,} accounts, {s['lra_only_lots']:,} lots, {s['lra_only_acres']:,.0f} acres "
                  f"({s['lra_only_acres'] / (s['exempt_only_acres'] + d['land_area_acres'].sum()):.1%} of city parcel acreage), "
                  f"appraised land ${s['lra_land'] / 1e6:,.1f}M; condo lots {s.get('condo_lots', 0)}, imputed "
                  f"{s.get('condo_lots_imputed', 0)} (+${s.get('condo_land_imputed_added', 0) / 1e6:,.1f}M, cap binding "
                  f"{s.get('condo_cap_binding', 0)})")
        else:
            print(f"    exempt by TAXCODE (n, land): " + ", ".join(
                f"{k}:{v['size']}/${v['sum'] / 1e6:,.0f}M" for k, v in sorted(s["exempt_by_code"].items())))
            print(f"    municipality attribute vs boundary agreement {s['muni_attr_spatial_agree']:.2%}")
    print(f"TOTAL: {len(final):,} parcels, land ${final['current_full_land_value'].sum() / 1e9:,.2f}B, "
          f"market ${final['full_market_value'].sum() / 1e9:,.2f}B")
    (DATA_DIR / "build-stats.json").write_text(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
