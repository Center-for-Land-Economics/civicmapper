"""Audit the hex->parcel handoff for every PMTiles city against the current bake rule.

For each registry city that has baked PMTiles metadata in the parquets-dev blob, compare
the handoff the blob currently serves (metadata parcelMinZoom, 13 for pre-rule bakes)
against what plan_h3_ladder would decide today (count caps + median-parcel prune — hex
bands finer than the city's median parcel are skipped and parcels take over earlier).
Cities where the two differ need a rebake (parquet_to_pmtiles.py --city <x> --h3).

Reads only the land_area_acres column of each parquet via HTTP range requests (a few MB
per city, no full downloads); falls back to a geometry sample when the column is absent.

Usage:  python data/scripts/audit_hex_parcel_handoff.py [city ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.parquet_registry import CITY_PARQUETS
from data.scripts.parquet_to_pmtiles import plan_h3_ladder_core

BLOB = "https://landeconomics.blob.core.windows.net/parquets-dev"
SQM_PER_ACRE = 4046.8564224

session = requests.Session()


class HttpRangeFile:
    """Minimal seekable read-only file over HTTP range requests (for pyarrow)."""

    def __init__(self, url: str):
        self.url = url
        r = session.head(url, timeout=30)
        r.raise_for_status()
        self._size = int(r.headers["Content-Length"])
        self._pos = 0
        self.closed = False

    def size(self):
        return self._size

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        self._pos = (offset if whence == 0
                     else self._pos + offset if whence == 1
                     else self._size + offset)
        return self._pos

    def read(self, nbytes=-1):
        end = (self._size if nbytes is None or nbytes < 0
               else min(self._pos + nbytes, self._size)) - 1
        if end < self._pos:
            return b""
        r = session.get(self.url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=120)
        r.raise_for_status()
        self._pos += len(r.content)
        return r.content

    def close(self):
        self.closed = True

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def flush(self):
        pass


def median_parcel_m2(pf: pq.ParquetFile) -> tuple[float, str]:
    """Median parcel area. Prefers the cheap land_area_acres column read; falls back to
    geometry from up to 5 row groups spread across the file (parquets are often spatially
    sorted, so the sample must be spread)."""
    if "land_area_acres" in pf.schema_arrow.names:
        vals = pd.to_numeric(pf.read(columns=["land_area_acres"]).column(0).to_pandas(),
                             errors="coerce")
        vals = vals[(vals > 0) & np.isfinite(vals)]
        if len(vals) > 0.5 * pf.metadata.num_rows:
            return float(vals.median()) * SQM_PER_ACRE, "land_area_acres"

    import geopandas as gpd
    from shapely import wkb

    n = pf.metadata.num_row_groups
    areas = []
    for rg in sorted({int(i) for i in np.linspace(0, n - 1, min(5, n))}):
        tbl = pf.read_row_group(rg, columns=["geometry"])
        geoms = wkb.loads(np.asarray(tbl.column(0).to_pandas(), dtype=object))
        a = gpd.GeoSeries(geoms, crs=4326).to_crs(6933).area.to_numpy()
        areas.append(a[a > 0])
    return float(np.median(np.concatenate(areas))), "geometry-sample"


def city_uses_pmtiles(key: str) -> bool:
    """The frontend dictionary is the authority on whether a city actually serves PMTiles —
    a metadata JSON can linger in the blob from an abandoned bake (e.g. spokane)."""
    dict_path = Path(__file__).parent.parent.parent / "viz" / "src" / "dictionaries" / f"{key}.json"
    if not dict_path.exists():
        return True  # no dictionary to consult; let the blob probe decide
    import json
    cfg = json.loads(dict_path.read_text(encoding="utf-8")).get("_config") or {}
    return bool(cfg.get("usePmtiles"))


def audit_city(slug: str) -> dict | None:
    meta_resp = session.get(f"{BLOB}/{slug}-parcels-metadata.json", timeout=30)
    if meta_resp.status_code == 404:
        return None  # no PMTiles bake for this city
    meta_resp.raise_for_status()
    meta = meta_resp.json()
    b = meta.get("bounds")
    if not b:
        raise RuntimeError("metadata has no bounds")
    lat, lng = (b[1] + b[3]) / 2.0, (b[0] + b[2]) / 2.0

    pf = pq.ParquetFile(pa.PythonFile(HttpRangeFile(f"{BLOB}/{slug}-parcels.parquet"), mode="r"))
    med_m2, method = median_parcel_m2(pf)

    plan = plan_h3_ladder_core(pf.metadata.num_rows, med_m2, lat, lng)
    current = int(meta.get("parcelMinZoom", 13))
    return {
        "slug": slug,
        "rows": pf.metadata.num_rows,
        "median_m2": med_m2,
        "method": method,
        "ladder": plan["res_list"],
        "pruned": plan["median_pruned"],
        "planned_pmz": plan["parcel_minzoom"],
        "current_pmz": current,
        "needs_rebake": plan["parcel_minzoom"] != current,
    }


def main():
    keys = sys.argv[1:] or sorted(CITY_PARQUETS)
    print(f"{'city':<16} {'rows':>9} {'med m²':>8} {'ladder':<16} {'pmz now':>7} {'pmz plan':>8}  verdict")
    stale = []
    for key in keys:
        cp = CITY_PARQUETS.get(key)
        if cp is None:
            print(f"{key:<16} unknown city key")
            continue
        if not city_uses_pmtiles(key):
            continue  # GeoParquet-only city (any blob metadata JSON is a leftover)
        try:
            r = audit_city(cp.slug)
        except Exception as exc:
            print(f"{key:<16} ERROR: {type(exc).__name__}: {exc}")
            continue
        if r is None:
            continue  # GeoParquet-only city, no hex/parcel handoff to audit
        verdict = "REBAKE (prune r%s)" % ",".join(map(str, r["pruned"])) if r["needs_rebake"] else "ok"
        print(f"{key:<16} {r['rows']:>9,} {r['median_m2']:>8,.0f} "
              f"{'r' + ',r'.join(map(str, r['ladder'])):<16} {r['current_pmz']:>7} {r['planned_pmz']:>8}  {verdict}")
        if r["needs_rebake"]:
            stale.append(key)
    print()
    if stale:
        print(f"{len(stale)} city(ies) need a rebake: {' '.join(stale)}")
    else:
        print("All baked handoffs match the current rule.")


if __name__ == "__main__":
    main()
