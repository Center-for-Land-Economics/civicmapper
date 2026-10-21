#!/usr/bin/env python3
"""prospect_cities.py — rank candidate jurisdictions by how cheap they are to add.

Scores candidates against the six properties that actually predict ETL effort, derived
from docs/add-city-playbook.md and the add-city skill's field notes:

  1. city == county        -> playbook §4 (the city clip) disappears entirely
  2. values ON the geometry layer -> no join, so no N-x value-inflation risk
  3. land-use class with readable labels -> §7 refined categories without heuristics
  4. explicit exemption field -> §6 without owner-name guessing
  5. under ~100k parcels   -> §10 PMTiles/H3/tippecanoe bake skipped
  6. endpoint reachable    -> appraisal-district GIS is frequently firewalled

Everything is probed LIVE. Published field lists and portal descriptions are routinely
wrong or incomplete, which is exactly how a candidate like Staunton looks ideal on paper
(37 fields, use codes, acreage) and turns out to carry no dollar values at all.

Usage:
    python prospect_cities.py                      # built-in candidate list
    python prospect_cities.py --only va            # one group
    python prospect_cities.py --json out.json      # machine-readable
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0 (civicmapper prospector)"}
TIMEOUT = 35
PMTILES_THRESHOLD = 100_000

# Field-name signatures. Deliberately broad: assessors name things wildly differently
# (CNTLNDVAL, Current_Land, LANDVAL1, LAND_VALUE, land_val, ASSESSEDLAND...).
SIG = {
    "land_value": re.compile(r"(land.*(val|assmt|assess))|((val|assess).*land)|lndval|land_?val", re.I),
    "impr_value": re.compile(r"impr|bldg|build|dwell|structure.*val|imp_?val", re.I),
    "total_value": re.compile(r"(tot.*(val|appr))|(market.*val)|mkt.*val|assessed.*total", re.I),
    "land_use": re.compile(r"class|landuse|land_use|usecd|use_?code|propclas|proptype|sptb|zoning_?desc|pcdesc|usedscrp", re.I),
    "exempt": re.compile(r"exempt|exmp|tax.?status|taxable", re.I),
    "area": re.compile(r"acre|sqft|sq_?ft|calcarea|shape.?area|landsqft|statedarea", re.I),
    "owner": re.compile(r"owner", re.I),
    "parcel_id": re.compile(r"parcel|pin|apn|gpin|acct|taxid|propert.*id", re.I),
}
PARCEL_LAYER = re.compile(r"parcel|taxlot|tax_?lot|propert|cadastr", re.I)
# Layer names that advertise a SUBSET of the roll. New Orleans resolved to
# "VacantParcels_View" (28k rows — plausible-looking, but vacant-only), which a
# row-count gate alone cannot catch.
SUBSET_LAYER = re.compile(r"vacant|sample|selected|subset|historic|archive|blight|"
                          r"opportunity|zone|district|owned|surplus|proposed", re.I)


def get(url: str, insecure: bool = False):
    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def get_tolerant(url: str):
    """Return (payload, ssl_broken). Retries once with verification off so a bad cert
    is reported as a wart rather than silently disqualifying an otherwise good city."""
    try:
        return get(url), False
    except Exception as e:
        if "CERTIFICATE" in str(e).upper() or "SSL" in str(e).upper():
            try:
                return get(url, insecure=True), True
            except Exception:
                raise
        raise


def agol_search(query: str, num: int = 12) -> list[str]:
    """Candidate service URLs from the public ArcGIS Online item search."""
    url = ("https://www.arcgis.com/sharing/rest/search?f=json&num=%d"
           "&sortField=numviews&sortOrder=desc&q=%s" % (num, urllib.parse.quote(query)))
    try:
        d = get(url)
    except Exception:
        return []
    out = []
    for it in d.get("results", []):
        if it.get("type") not in ("Feature Service", "Map Service"):
            continue
        u = it.get("url") or ""
        if u and PARCEL_LAYER.search(it.get("title", "") + " " + u):
            out.append(u)
    return out


def resolve_parcel_layer(service_url: str) -> tuple[str | None, bool]:
    """Given a service root or layer URL, return the best parcel layer URL."""
    ssl_broken = False
    if re.search(r"/\d+$", service_url):
        return service_url, ssl_broken
    try:
        svc, ssl_broken = get_tolerant(service_url + "?f=json")
    except Exception:
        return None, ssl_broken
    layers = [l for l in svc.get("layers", [])
              if l.get("geometryType") in (None, "esriGeometryPolygon")]
    named = [l for l in layers if PARCEL_LAYER.search(l.get("name", ""))]
    pick = (named or layers or [None])[0]
    if not pick:
        return None, ssl_broken
    return f"{service_url}/{pick['id']}", ssl_broken


def probe(layer_url: str) -> dict:
    """Pull field list, record count, and one sample row from a live layer."""
    res: dict = {"layer_url": layer_url, "ssl_broken": False, "error": None}
    try:
        meta, broken = get_tolerant(layer_url + "?f=json")
        res["ssl_broken"] = broken
        res["layer_name"] = meta.get("name")
        res["geometry_type"] = meta.get("geometryType")
        fields = [f["name"] for f in meta.get("fields", [])]
        res["fields"] = fields

        cnt, _ = get_tolerant(layer_url + "/query?where=1%3D1&returnCountOnly=true&f=json")
        res["count"] = cnt.get("count")

        sample, _ = get_tolerant(
            layer_url + "/query?where=1%3D1&outFields=*&resultRecordCount=1"
                        "&returnGeometry=false&f=json")
        feats = sample.get("features", [])
        res["sample"] = feats[0]["attributes"] if feats else {}
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:110]}"
    return res


def _fill_rate(layer_url: str, field: str, total: int | None) -> float:
    """Share of rows where `field` is actually populated, measured server-side.

    A single sample row is NOT good enough: it produced a false negative on Roanoke,
    where the first row happened to be null while 44,227 of 44,490 rows carry real land
    values. One cheap returnCountOnly per field buys a correct answer.
    """
    if not total:
        return 0.0
    where = urllib.parse.quote(f"{field} > 0")
    try:
        d, _ = get_tolerant(f"{layer_url}/query?where={where}&returnCountOnly=true&f=json")
        n = d.get("count")
        return (n / total) if isinstance(n, int) else 0.0
    except Exception:
        return 0.0  # non-numeric field, or the server rejected the predicate


def _populated(layer_url: str, names: list[str], total: int | None,
               threshold: float = 0.10) -> tuple[bool, str | None, float]:
    """True if any candidate field is populated on >threshold of rows."""
    best_f, best_r = None, 0.0
    for n in names[:3]:                       # bound cost: 3 probes per category
        r = _fill_rate(layer_url, n, total)
        if r > best_r:
            best_f, best_r = n, r
    return (best_r >= threshold), best_f, best_r


def score(cand: dict, p: dict) -> dict:
    """Weighted score + the specific reasons, so a rejection is explainable."""
    out = {**cand, **{k: p.get(k) for k in
                      ("count", "layer_url", "layer_name", "fields", "ssl_broken", "error")}}
    reasons: list[str] = []
    pts = 0

    if p.get("error"):
        out.update(score=-1, verdict="UNREACHABLE", reasons=[p["error"]])
        return out

    fields, sample = p.get("fields", []) or [], p.get("sample", {}) or {}
    hits = {k: [f for f in fields if rx.search(f)] for k, rx in SIG.items()}
    out["matched"] = {k: v[:4] for k, v in hits.items() if v}

    # 1. jurisdiction shape — metadata, not probed
    if cand.get("city_county"):
        pts += 2
        reasons.append("+2 city==county (no clip)")
    else:
        reasons.append("+0 needs a city-limits clip")

    # 2. values present AND populated (measured server-side, not from one sample row)
    lyr, total = p["layer_url"], p.get("count")
    has_land, lf, lr = _populated(lyr, hits["land_value"], total)
    has_impr, imf, imr = _populated(lyr, hits["impr_value"], total)
    out["value_fields"] = {"land": (lf, round(lr, 3)), "improvement": (imf, round(imr, 3))}
    if has_land and has_impr:
        pts += 3
        reasons.append(f"+3 land AND improvement values on the layer "
                       f"({lf} {lr:.0%} filled, {imf} {imr:.0%})")
    elif has_land:
        pts += 2
        reasons.append(f"+2 land value only, no improvement split ({lf} {lr:.0%} filled)")
    elif hits["land_value"]:
        reasons.append(f"+0 value-named columns exist but are EMPTY "
                       f"(best {hits['land_value'][0]} at {lr:.0%})")
    else:
        reasons.append("+0 NO value columns -> needs a separate appraisal join")

    # 3. land-use classification
    if hits["land_use"]:
        readable = any(isinstance(sample.get(f), str) and len(str(sample.get(f) or "")) > 3
                       for f in hits["land_use"])
        pts += 2 if readable else 1
        reasons.append(("+2 land-use class with readable labels" if readable
                        else "+1 land-use code (numeric only, needs a lookup)"))
    else:
        reasons.append("+0 no land-use/class field -> hideUnderutilized like Hartford")

    # 4. exemption
    if hits["exempt"]:
        pts += 1
        reasons.append("+1 explicit exemption/tax-status field")
    else:
        reasons.append("+0 no exemption flag -> owner-keyword heuristic needed")

    # 5. size — and a plausibility gate on the layer itself.
    # A municipal parcel layer outside ~1k-1.5M rows is almost certainly the WRONG
    # layer: too few means a subset/sample (Hampton resolved to a 53-row layer), too
    # many means a statewide or national mosaic (Carson City resolved to 1.4M).
    n = p.get("count")
    if isinstance(n, int):
        if n < 1_000 or n > 1_500_000:
            pts -= 3
            reasons.append(f"-3 {n:,} rows is implausible for one city — LIKELY WRONG LAYER")
            out["suspect_layer"] = True
        elif n < PMTILES_THRESHOLD:
            pts += 1
            reasons.append(f"+1 {n:,} parcels — under the PMTiles threshold")
        else:
            reasons.append(f"+0 {n:,} parcels — needs a PMTiles/H3 bake")

    # A plausible row count is not enough — the layer name can still advertise a subset.
    if SUBSET_LAYER.search((p.get("layer_name") or "") + " " + (p.get("layer_url") or "")):
        pts -= 3
        reasons.append(f"-3 layer name looks like a SUBSET, not the full roll "
                       f"({p.get('layer_name')}) — re-point before trusting")
        out["suspect_layer"] = True

    # 6. reachability warts
    if p.get("ssl_broken"):
        pts -= 1
        reasons.append("-1 broken TLS cert (needs verify=False)")
    if hits["area"]:
        reasons.append("   (has an area field for the $/sqft denominator)")

    out["score"] = pts
    out["verdict"] = ("EXCELLENT" if pts >= 8 else "GOOD" if pts >= 6
                      else "WORKABLE" if pts >= 4 else "POOR")
    out["reasons"] = reasons
    return out


def evaluate(cand: dict) -> dict:
    urls = [cand["url"]] if cand.get("url") else agol_search(cand["query"])
    if not urls:
        return {**cand, "score": -1, "verdict": "NO LAYER FOUND",
                "reasons": ["AGOL search returned no parcel-like service"], "error": None}
    best = None
    for u in urls[:3]:
        layer, _ = resolve_parcel_layer(u)
        if not layer:
            continue
        s = score(cand, probe(layer))
        if best is None or s["score"] > best["score"]:
            best = s
        if best["score"] >= 8:
            break
    return best or {**cand, "score": -1, "verdict": "NO LAYER FOUND",
                    "reasons": ["could not resolve a parcel layer"], "error": None}


# Virginia independent cities are legally outside any county, so their parcel layer is
# inherently city-only. Consolidated city-counties have the same property.
CANDIDATES = [
    # --- VA independent cities (city_county=True by law) ---
    {"name": "Hampton, VA",        "group": "va", "city_county": True, "query": "Hampton Virginia parcels"},
    {"name": "Portsmouth, VA",     "group": "va", "city_county": True, "query": "Portsmouth Virginia parcels"},
    {"name": "Suffolk, VA",        "group": "va", "city_county": True, "query": "Suffolk Virginia parcels"},
    {"name": "Danville, VA",       "group": "va", "city_county": True, "query": "Danville Virginia parcels"},
    {"name": "Petersburg, VA",     "group": "va", "city_county": True, "query": "Petersburg Virginia parcels"},
    {"name": "Harrisonburg, VA",   "group": "va", "city_county": True, "query": "Harrisonburg Virginia parcels"},
    {"name": "Winchester, VA",     "group": "va", "city_county": True, "query": "Winchester Virginia parcels"},
    {"name": "Fredericksburg, VA", "group": "va", "city_county": True, "query": "Fredericksburg Virginia parcels"},
    {"name": "Salem, VA",          "group": "va", "city_county": True, "query": "Salem Virginia parcels"},
    {"name": "Hopewell, VA",       "group": "va", "city_county": True, "query": "Hopewell Virginia parcels"},
    {"name": "Manassas, VA",       "group": "va", "city_county": True,
     "url": "https://services1.arcgis.com/3wpOgOChiWXPeFWB/arcgis/rest/services/Manassas_Parcels/FeatureServer"},
    {"name": "Roanoke, VA",        "group": "va", "city_county": True,
     "url": "https://gis03.roanokeva.gov/arcgis/rest/services/Public/Parcels/FeatureServer/0"},
    # --- consolidated city-counties elsewhere ---
    {"name": "Nashville/Davidson, TN", "group": "cc", "city_county": True, "query": "Nashville Davidson parcels"},
    {"name": "Jacksonville/Duval, FL", "group": "cc", "city_county": True, "query": "Duval County parcels"},
    {"name": "Indianapolis/Marion, IN","group": "cc", "city_county": True, "query": "Marion County Indiana parcels"},
    {"name": "New Orleans, LA",        "group": "cc", "city_county": True, "query": "Orleans Parish parcels"},
    {"name": "Anchorage, AK",          "group": "cc", "city_county": True, "query": "Anchorage parcels"},
    {"name": "Broomfield, CO",         "group": "cc", "city_county": True, "query": "Broomfield parcels"},
    {"name": "Carson City, NV",        "group": "cc", "city_county": True, "query": "Carson City parcels"},
    {"name": "Augusta/Richmond, GA",   "group": "cc", "city_county": True, "query": "Richmond County Georgia parcels"},
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="filter to one group (va, cc)")
    ap.add_argument("--json", help="write full results to this path")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cands = [c for c in CANDIDATES if not args.only or c.get("group") == args.only]
    print(f"Probing {len(cands)} candidates live (this hits real endpoints)...\n", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(evaluate, cands))
    results.sort(key=lambda r: (-r.get("score", -1), r["name"]))

    print(f"{'SCORE':>5}  {'VERDICT':<14} {'PARCELS':>9}  CANDIDATE")
    print("-" * 78)
    for r in results:
        n = r.get("count")
        print(f"{r.get('score', -1):>5}  {r.get('verdict', '?'):<14} "
              f"{(f'{n:,}' if isinstance(n, int) else '—'):>9}  {r['name']}")

    print("\n" + "=" * 78 + "\nDETAIL (top candidates first)\n" + "=" * 78)
    for r in results:
        if r.get("score", -1) < 4:
            continue
        print(f"\n### {r['name']}  —  score {r['score']} ({r['verdict']})")
        if r.get("layer_url"):
            print(f"    {r['layer_url']}")
        for reason in r.get("reasons", []):
            print(f"      {reason}")
        m = r.get("matched", {})
        for k in ("land_value", "impr_value", "land_use", "exempt"):
            if m.get(k):
                print(f"      {k:<12} {m[k]}")

    rejected = [r for r in results if r.get("score", -1) < 4]
    if rejected:
        print("\n" + "=" * 78 + "\nREJECTED\n" + "=" * 78)
        for r in rejected:
            why = next((x for x in r.get("reasons", []) if x.startswith("+0") or x.startswith("-")),
                       (r.get("reasons") or ["?"])[0])
            print(f"  {r['name']:<28} {r.get('verdict','?'):<14} {why}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
