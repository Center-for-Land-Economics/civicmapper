#!/usr/bin/env python3
"""
Local Segment Anything (SAM 2.1) helper for the label page's "Segment" tool. Dev-only.

The label page sends clicks (lng/lat + include/exclude) for a square; this server reads the square's
NAIP (same grid and cache as label_finetune.py: data/labels/<campaign>/export/naip/<square>.tif, or
fetched from Planetary Computer and cached), computes SAM's image embedding once per square (kept in
memory), and returns the outline of the segmented region as a GeoJSON polygon (EPSG:4326), which the
page turns into an editable shape of the current class.

Model: facebook/sam2.1-hiera-base-plus (Apache 2.0). GPU if available (about 0.1 s per click after
the first ~1 s embedding on an RTX 3060/4060); CPU works but the first click on a square takes a few
seconds. No torchvision needed.

Run it next to start-local (in the GPU environment):
  python data/scripts/sam_server.py              # listens on 127.0.0.1:8765
The local API (api/src/labels.js) forwards /api/labels/<campaign>/sam to it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LABELS = ROOT / "data" / "labels"
CELL, MARGIN = 0.6, 48


class Sam:
    def __init__(self, model_id: str, cpu: bool):
        import torch
        from transformers import Sam2Model
        self.torch = torch
        self.device = "cuda" if (torch.cuda.is_available() and not cpu) else "cpu"
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.cache: OrderedDict = OrderedDict()  # (campaign, square) -> (embedding, transform, crs, shape)
        self.lock = threading.Lock()
        print(f"SAM 2.1 {model_id} on {self.device.upper()}", flush=True)

    def square(self, campaign: str, square_id: str):
        key = (campaign, square_id)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        img, tr, crs = load_square_image(campaign, square_id)
        rgb = img[:3].transpose(1, 2, 0)
        t = self.torch
        x = t.as_tensor(rgb.copy(), device=self.device).permute(2, 0, 1)[None].float() / 255.0
        x = t.nn.functional.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        with t.no_grad():
            emb = self.model.get_image_embeddings(pixel_values=x)
        entry = (emb, tr, crs, rgb.shape[:2])
        self.cache[key] = entry
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        return entry

    def segment(self, campaign: str, square_id: str, points: list, box: list | None):
        from pyproj import Transformer
        with self.lock:
            emb, tr, crs, (h, w) = self.square(campaign, square_id)
            to_px = Transformer.from_crs(4326, crs, always_xy=True)
            inv = ~tr
            sx, sy = 1024 / w, 1024 / h

            def px(lng, lat):
                x, y = to_px.transform(lng, lat)
                c, r = inv * (x, y)
                return [c * sx, r * sy]
            pts = [px(p[0], p[1]) for p in points]
            if any(not (0 <= q[0] < 1024 and 0 <= q[1] < 1024) for q in pts):
                raise ValueError("click outside this square's imagery")
            labels = [1 if p[2] else 0 for p in points]
            t = self.torch
            kw = {}
            if pts:
                kw["input_points"] = t.tensor([[pts]], device=self.device, dtype=t.float32)
                kw["input_labels"] = t.tensor([[labels]], device=self.device, dtype=t.long)
            if box:
                (x0, y0), (x1, y1) = px(box[0], box[1]), px(box[2], box[3])
                kw["input_boxes"] = t.tensor([[[min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]]],
                                             device=self.device, dtype=t.float32)
            with t.no_grad():
                res = self.model(image_embeddings=emb, multimask_output=len(pts) == 1 and not box, **kw)
            scores = res.iou_scores[0, 0].float().cpu().numpy()
            # One click gives SAM's three nested proposals (part / bigger part / whole). For parking
            # the whole lot is nearly always wanted: take the largest proposal whose predicted quality
            # is within 0.15 of the best. With several clicks or a box there is a single proposal.
            areas = (res.pred_masks[0, 0] > 0).flatten(1).sum(1).cpu().numpy()
            ok = [i for i in range(len(scores)) if scores[i] >= scores.max() - 0.15]
            best = max(ok, key=lambda i: areas[i])
            lg = t.nn.functional.interpolate(res.pred_masks[0, :, best:best + 1].float(), size=(h, w),
                                             mode="bilinear", align_corners=False)
            mask = (lg[0, 0] > 0).cpu().numpy()
        return mask_to_polygon(mask, tr, crs, [p for p, l in zip(pts, labels) if l], (sx, sy)), float(scores[best])


def load_square_image(campaign: str, square_id: str):
    """(4,H,W) NAIP on the label export grid (0.6 m UTM, 48 px margin), its transform and CRS."""
    import geopandas as gpd
    import rasterio
    from rasterio.transform import Affine, from_origin
    cache = LABELS / campaign / "export" / "naip" / f"{square_id}.tif"
    if cache.exists():
        with rasterio.open(cache) as src:
            return src.read(), src.transform, src.crs
    from data.scripts.label_eval import read_naip
    sq = gpd.read_file(LABELS / campaign / "squares.geojson")
    utm = sq.estimate_utm_crs()
    row = sq[sq["id"] == square_id]
    if row.empty:
        raise KeyError(f"no square {square_id}")
    geom = row.to_crs(utm).geometry.iloc[0]
    x0, y0, x1, y1 = geom.bounds
    w, h = math.ceil((x1 - x0) / CELL - 1e-6), math.ceil((y1 - y0) / CELL - 1e-6)
    tr = from_origin(x0, y1, CELL, CELL) * Affine.translation(-MARGIN, -MARGIN)
    items = row["naip_items"].iloc[0]
    items = json.loads(items) if isinstance(items, str) else list(items)
    img = read_naip(items, utm, tr, w + 2 * MARGIN, h + 2 * MARGIN)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(cache, "w", driver="GTiff", width=img.shape[2], height=img.shape[1], count=4,
                       dtype="uint8", crs=utm, transform=tr, compress="deflate") as dst:
        dst.write(img)
    return img, tr, utm


def mask_to_polygon(mask, tr, crs, pos_pts_1024, scale):
    """Largest mask region (or the one under the first include click) as a GeoJSON polygon in 4326."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import Point, mapping, shape
    polys = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=tr) if v]
    if not polys:
        return None
    pick = max(polys, key=lambda g: g.area)
    if pos_pts_1024:
        c, r = pos_pts_1024[0][0] / scale[0], pos_pts_1024[0][1] / scale[1]
        hit = Point(*(tr * (c, r)))
        under = [g for g in polys if g.buffer(1.0).contains(hit)]
        if under:
            pick = max(under, key=lambda g: g.area)
    pick = pick.simplify(CELL * 0.75, preserve_topology=True).buffer(0)
    if pick.geom_type == "MultiPolygon":
        pick = max(pick.geoms, key=lambda g: g.area)
    pick = type(pick)(pick.exterior)  # editable shapes are single-ring; interior holes are dropped
    return mapping(gpd.GeoSeries([pick], crs=crs).to_crs(4326).iloc[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="facebook/sam2.1-hiera-base-plus")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    sam = Sam(a.model, a.cpu)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # health check
            self._send(200, {"ok": True, "device": sam.device})

        def do_POST(self):
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                return self._send(403, {"ok": False, "error": "local only"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                t0 = time.time()
                geom, score = sam.segment(str(req["campaign"]), str(req["square_id"]),
                                          req.get("points") or [], req.get("box"))
                self._send(200, {"ok": True, "polygon": geom, "score": score, "ms": round((time.time() - t0) * 1000)})
            except Exception as e:  # report, keep serving
                self._send(400, {"ok": False, "error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args):
            pass

    print(f"Segment server on http://127.0.0.1:{a.port} (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
