"""
Parking segmentation models behind one interface, for training (label_finetune.py) and the
bake-off (bakeoff.py, prn_eval.py).

Every model maps 4-band NAIP (uint8, bands R,G,B,NIR; any size) to a parking probability map, and
can be trained on 512 px RGB crops with a 2-class cross-entropy (0 not parking, 1 surface parking).

Specs (strings, used on the command line):
  segformer                      the pipeline's model (UTEL-UIUC/SegFormer-large-parking, B5)
  segformer:<model.pt>           same architecture, fine-tuned weights (label_finetune.py output)
  vit:<run dir>                  a ViT backbone + head trained by label_finetune.py --arch vit
                                 (run dir holds config.json + model.pt)
  vit-init:<hf backbone id>      untrained head on a backbone (for training only)
  sam2:<base spec>               base model's mask refined with SAM 2.1 box prompts (no training)

ViT backbones (Hugging Face): facebook/dinov3-vitl16-pretrain-sat493m (DINOv3, satellite
pretraining; gated: request access on its model page, then set HF_TOKEN), facebook/dinov2-large
(open; general-purpose, a stand-in). Head: the last 4 transformer layers' patch features,
projected and fused by two 3x3 convolutions, upsampled to the input (a light DPT-style head).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TILE = 512


class ParkingModel:
    name = "model"
    threshold = 0.5

    def predict(self, img: np.ndarray) -> np.ndarray:  # img (4,H,W) uint8 -> prob (H,W) float32
        raise NotImplementedError

    # training interface -------------------------------------------------------------------
    def logits(self, crops: list[np.ndarray]):  # crops: list of (TILE,TILE,3) uint8 -> (B,2,h,w)
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def train(self, mode: bool = True):
        raise NotImplementedError

    def save(self, out_dir: Path, extra: dict | None = None) -> None:
        raise NotImplementedError


def _device(cpu: bool):
    import torch
    if cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sliding_predict(img: np.ndarray, batch_logits, batch: int = 4) -> np.ndarray:
    """Average parking probability over 512 px windows (64 px overlap), any image size."""
    import torch
    _, h, w = img.shape
    ph, pw = max(h, TILE), max(w, TILE)
    if (ph, pw) != (h, w):
        img = np.pad(img, ((0, 0), (0, ph - h), (0, pw - w)), mode="reflect")
    ys = sorted({0, max(0, ph - TILE)} | set(range(0, max(1, ph - TILE), TILE - 64)))
    xs = sorted({0, max(0, pw - TILE)} | set(range(0, max(1, pw - TILE), TILE - 64)))
    acc = np.zeros((ph, pw), np.float32)
    wt = np.zeros((ph, pw), np.float32)
    patches = [(y, x, img[:3, y:y + TILE, x:x + TILE].transpose(1, 2, 0)) for y in ys for x in xs]
    for k in range(0, len(patches), batch):
        group = patches[k:k + batch]
        with torch.no_grad():
            lg = batch_logits([g[2] for g in group])
            up = torch.nn.functional.interpolate(lg.float(), size=(TILE, TILE), mode="bilinear", align_corners=False)
            probs = torch.softmax(up, dim=1)[:, 1].cpu().numpy()
        for (y, x, _), pr in zip(group, probs):
            acc[y:y + TILE, x:x + TILE] += pr
            wt[y:y + TILE, x:x + TILE] += 1
    return (acc / np.maximum(wt, 1))[:h, :w]


# ---------------------------------------------------------------------------------------------
class SegFormerModel(ParkingModel):
    def __init__(self, weights: str | None = None, cpu: bool = False):
        import torch
        from data.scripts.parking_lot_extraction import load_segformer_bundle
        self.processor, self.model, self.device, mid = load_segformer_bundle(use_cpu=cpu)
        self.name = "segformer"
        if weights:
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
            self.name = f"segformer:{Path(weights).parent.parent.name}/{Path(weights).stem}"
        self.model.eval()

    def logits(self, crops):
        inputs = self.processor(images=crops, return_tensors="pt", do_resize=False).to(self.device)
        return self.model(**inputs).logits

    def predict(self, img):
        return sliding_predict(img, self.logits)

    def parameters(self):
        return self.model.parameters()

    def train(self, mode=True):
        self.model.train(mode)

    def freeze_stages(self, n: int):
        enc = self.model.segformer.encoder
        for i in range(n):
            for mod in (enc.patch_embeddings[i], enc.block[i], enc.layer_norm[i]):
                for p in mod.parameters():
                    p.requires_grad = False

    def save(self, out_dir: Path, extra=None):
        import torch
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "model.pt")
        (out_dir / "config.json").write_text(json.dumps({"arch": "segformer", "threshold": self.threshold,
                                                          **(extra or {})}, indent=2))


# ---------------------------------------------------------------------------------------------
class ViTSegModel(ParkingModel):
    """Hugging Face ViT backbone (DINOv3 / DINOv2) + light multi-layer fusion head."""

    def __init__(self, backbone: str, cpu: bool = False, layers: int = 4, width: int = 256):
        import torch
        from torch import nn
        from transformers import AutoImageProcessor, AutoModel
        self.device = _device(cpu)
        self.backbone_id = backbone
        self.backbone = AutoModel.from_pretrained(backbone).to(self.device)
        proc = AutoImageProcessor.from_pretrained(backbone)
        self.mean = torch.tensor(proc.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(proc.image_std, device=self.device).view(1, 3, 1, 1)
        cfg = self.backbone.config
        self.patch = int(getattr(cfg, "patch_size", 16))
        dim = int(cfg.hidden_size)
        self.layers = layers
        self.head = nn.Sequential(
            nn.Conv2d(dim * layers, width, 1), nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, 2, 1)).to(self.device)
        self.name = f"vit:{backbone.split('/')[-1]}"

    def _features(self, x):
        import torch
        side = x.shape[-1]
        size = (side // self.patch) * self.patch or self.patch
        if size != side:  # DINOv2's 14 px patches: 512 -> 504
            x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        g = size // self.patch
        feats = []
        for hs in out.hidden_states[-self.layers:]:
            tok = hs[:, -g * g:, :]           # patch tokens follow the CLS (+ register) tokens
            feats.append(tok.transpose(1, 2).reshape(hs.shape[0], -1, g, g))
        return torch.cat(feats, dim=1)

    def logits(self, crops):
        import torch
        x = torch.as_tensor(np.stack(crops), device=self.device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        return self.head(self._features(x))

    def predict(self, img):
        return sliding_predict(img, self.logits, batch=2)

    def parameters(self):
        return [p for m in (self.backbone, self.head) for p in m.parameters()]

    def train(self, mode=True):
        self.backbone.train(mode)
        self.head.train(mode)

    def freeze_backbone(self, unfreeze_last: int = 0):
        """Freeze the backbone except its last `unfreeze_last` transformer layers."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_last:
            layers = None
            for attr in ("layer", "encoder.layer"):
                obj = self.backbone
                for part in attr.split("."):
                    obj = getattr(obj, part, None)
                if obj is not None:
                    layers = obj
                    break
            if layers is None:
                raise RuntimeError("could not find the backbone's transformer layers to unfreeze")
            for blk in list(layers)[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad = True

    def save(self, out_dir: Path, extra=None):
        import torch
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"backbone": self.backbone.state_dict(), "head": self.head.state_dict()}, out_dir / "model.pt")
        (out_dir / "config.json").write_text(json.dumps({
            "arch": "vit", "backbone": self.backbone_id, "layers": self.layers, "threshold": self.threshold,
            **(extra or {})}, indent=2))

    @classmethod
    def load(cls, run_dir: Path, cpu: bool = False):
        import torch
        cfg = json.loads((run_dir / "config.json").read_text())
        m = cls(cfg["backbone"], cpu=cpu, layers=cfg.get("layers", 4))
        sd = torch.load(run_dir / "model.pt", map_location=m.device)
        m.backbone.load_state_dict(sd["backbone"])
        m.head.load_state_dict(sd["head"])
        m.threshold = float(cfg.get("threshold", 0.5))
        m.name = f"vit:{run_dir.name}"
        m.train(False)
        return m


# ---------------------------------------------------------------------------------------------
class Sam2Refined(ParkingModel):
    """Base model's parking regions re-segmented by SAM 2.1 (box + interior point prompts).
    SAM follows real edges (curbs, building lines) better than the blobby base mask; its masks are
    kept only near the base region so a prompt cannot leak across a whole block."""

    def __init__(self, base: ParkingModel, sam_id: str = "facebook/sam2.1-hiera-base-plus", cpu: bool = False,
                 min_px: int = 150, leash_px: int = 12):
        import torch
        from transformers import Sam2Model
        self.base = base
        self.device = _device(cpu)
        self.sam = Sam2Model.from_pretrained(sam_id).to(self.device).eval()
        # SAM 2's own preprocessing (done here so torchvision is not needed): square 1024 px resize,
        # ImageNet normalisation; prompts in 1024 px coordinates; masks come back at 256 px.
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.min_px, self.leash = min_px, leash_px
        self.threshold = base.threshold
        self.name = f"sam2({base.name})"

    def predict(self, img):
        import torch
        from scipy import ndimage
        prob = self.base.predict(img)
        mask = prob >= self.base.threshold
        lab, n = ndimage.label(mask)
        if n == 0:
            return prob
        rgb = img[:3].transpose(1, 2, 0)
        boxes, points, comps = [], [], []
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            comp = lab[sl] == i
            if comp.sum() < self.min_px:
                continue
            ys, xs = np.nonzero(comp)
            k = len(ys) // 2
            boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
            points.append([[int(xs[k] + sl[1].start), int(ys[k] + sl[0].start)]])
            comps.append(i)
        if not boxes:
            return prob
        h, w = mask.shape
        sx, sy = 1024 / w, 1024 / h
        x = torch.as_tensor(rgb.copy(), device=self.device).permute(2, 0, 1)[None].float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = np.zeros(mask.shape, bool)
        with torch.no_grad():
            emb = self.sam.get_image_embeddings(pixel_values=x)
        for k in range(0, len(boxes), 16):
            bx = torch.tensor([[[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes[k:k + 16]]],
                              device=self.device, dtype=torch.float32)
            pt = torch.tensor([[[[p[0][0] * sx, p[0][1] * sy]] for p in points[k:k + 16]]],
                              device=self.device, dtype=torch.float32)
            lb = torch.ones(pt.shape[:3], device=self.device, dtype=torch.long)
            with torch.no_grad():
                res = self.sam(image_embeddings=emb, input_boxes=bx, input_points=pt, input_labels=lb,
                               multimask_output=False)
            lg = torch.nn.functional.interpolate(res.pred_masks[0].float(), size=(h, w), mode="bilinear",
                                                 align_corners=False)
            masks = (lg > 0).cpu().numpy()
            for j, i in enumerate(comps[k:k + 16]):
                m = masks[j, 0]
                leash = ndimage.binary_dilation(lab == i, iterations=self.leash)
                out |= m & leash
        return np.where(out, np.maximum(prob, 0.99), np.minimum(prob, 0.01)).astype(np.float32)


# ---------------------------------------------------------------------------------------------
def load_spec(spec: str, cpu: bool = False) -> ParkingModel:
    """Build a model from a spec string (see module docstring)."""
    if spec == "segformer":
        return SegFormerModel(cpu=cpu)
    kind, _, arg = spec.partition(":")
    if kind == "segformer":
        m = SegFormerModel(arg, cpu=cpu)
        cfg = Path(arg).parent / "config.json"
        if cfg.exists():
            m.threshold = float(json.loads(cfg.read_text()).get("threshold", 0.5))
        return m
    if kind == "vit":
        return ViTSegModel.load(Path(arg), cpu=cpu)
    if kind == "vit-init":
        return ViTSegModel(arg, cpu=cpu)
    if kind == "sam2":
        return Sam2Refined(load_spec(arg, cpu=cpu), cpu=cpu)
    raise ValueError(f"unknown model spec {spec!r}")
