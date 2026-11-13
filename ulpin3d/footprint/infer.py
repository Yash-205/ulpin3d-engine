"""Run a footprint model over a whole orthophoto (any size, any CRS).

    python -m ulpin3d.cli infer-footprint --image ortho.tif --checkpoint MODEL --out out/scene1 [--ndsm ndsm.tif]

MODEL can be
    runs/.../best.pt           U-Net checkpoint (PyTorch; GPU if present, else CPU)
    models/footprint.onnx      the same U-Net exported to ONNX: identical masks, ~2.7x faster on CPU
    models/footprint_lite_*    CPU LightGBM model (folder with meta.json); the *_height
                               variants also need --ndsm (height above ground from LiDAR)

Writes
    <out>/probs.tif         2-band uint8 (building, boundary) probability on a UTM grid
    <out>/buildings.gpkg    polygons in UTM with confidence / status / area
    <out>/buildings.geojson same polygons in EPSG:4326 (for the web viewer)

Large mosaics are processed block by block (default 4096 px) with an overlap margin,
so memory stays flat whatever the scene size. Buildings are kept by the block that
holds their centroid, which avoids duplicates at block seams.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from ulpin3d.common.raster import MetricReader
from ulpin3d.footprint.dataset import MEAN, STD
from ulpin3d.footprint.postprocess import DEFAULTS as PP_DEFAULTS
from ulpin3d.footprint.postprocess import instances_from_probs, polygons_from_instances


def _weight(tile: int, ramp: int) -> np.ndarray:
    r = np.ones(tile, np.float32)
    if ramp > 0:
        x = np.linspace(0.05, 1, ramp, dtype=np.float32)
        r[:ramp], r[-ramp:] = x, x[::-1]
    return np.outer(r, r)


def sliding_window(forward, rgb: np.ndarray, tile=512, overlap=128, tta=True, batch=4):
    """forward: (N,3,t,t) float32 normalised -> (N,2,t,t) probabilities. Returns (2, H, W)."""
    H, W = rgb.shape[:2]
    stride = tile - overlap
    ph = max(tile, math.ceil((H - overlap) / stride) * stride + overlap)
    pw = max(tile, math.ceil((W - overlap) / stride) * stride + overlap)
    img = np.zeros((ph, pw, 3), np.float32)
    img[:H, :W] = rgb.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    acc = np.zeros((2, ph, pw), np.float32)
    wsum = np.zeros((ph, pw), np.float32)
    w = _weight(tile, overlap // 2)
    coords = [(y, x) for y in range(0, ph - tile + 1, stride) for x in range(0, pw - tile + 1, stride)]
    for i in range(0, len(coords), batch):
        cs = coords[i: i + batch]
        x = np.stack([img[y:y + tile, x_:x_ + tile].transpose(2, 0, 1) for y, x_ in cs]).astype(np.float32)
        p = forward(x)
        if tta:
            p = p + forward(x[..., ::-1].copy())[..., ::-1]
            p = p + forward(x[..., ::-1, :].copy())[..., ::-1, :]
            p = p / 3
        for (y, x_), pj in zip(cs, p):
            acc[:, y:y + tile, x_:x_ + tile] += pj * w
            wsum[y:y + tile, x_:x_ + tile] += w
    return (acc / np.maximum(wsum, 1e-6))[:, :H, :W]


def predict_array(model, rgb, device, tile=512, overlap=128, tta=True, batch=4):
    """Back-compatible helper for a PyTorch model."""
    import torch

    @torch.no_grad()
    def fwd(x):
        return torch.sigmoid(model(torch.from_numpy(x).to(device)).float()).cpu().numpy()

    return sliding_window(fwd, rgb, tile, overlap, tta, batch)


class TorchPredictor:
    needs_height = False

    def __init__(self, path, device="auto"):
        from ulpin3d.footprint.model import load_checkpoint
        from ulpin3d.footprint.train import pick_device

        self.device = pick_device(device)
        self.model, ck = load_checkpoint(path, self.device)
        self.gsd = ck["data_cfg"].get("gsd") or 0.15
        self.postprocess = ck.get("postprocess", {})
        self.name = Path(path).parent.name

    def predict(self, rgb, ndsm=None, tile=512, overlap=128, tta=True):
        return predict_array(self.model, rgb, self.device, tile, overlap, tta)


class OnnxPredictor:
    needs_height = False

    def __init__(self, path):
        import onnxruntime as ort

        meta = json.loads(Path(str(path) + ".json").read_text())
        self.gsd = meta["gsd"]
        self.postprocess = meta.get("postprocess", {})
        self.tile = meta.get("tile", 512)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.name = Path(path).stem

    def predict(self, rgb, ndsm=None, tile=None, overlap=128, tta=True):
        def fwd(x):
            return 1 / (1 + np.exp(-self.sess.run(None, {self.inp: x})[0]))

        return sliding_window(fwd, rgb, tile or self.tile, overlap, tta, batch=1)


class LitePredictor:
    def __init__(self, path):
        from ulpin3d.footprint.lite import FootprintLite

        self.m = FootprintLite(path)
        self.gsd = self.m.gsd
        self.postprocess = self.m.postprocess
        self.needs_height = self.m.uses_height
        self.name = Path(path).name

    def predict(self, rgb, ndsm=None, **_):
        return self.m.predict(rgb, ndsm)


def load_predictor(path, device="auto"):
    from ulpin3d.footprint.lite import is_lite

    p = str(path)
    if is_lite(p):
        return LitePredictor(p)
    if p.endswith(".onnx"):
        return OnnxPredictor(p)
    return TorchPredictor(p, device)


class _FloatReader:
    """Window reads of a single-band float raster (nDSM) on the MetricReader grid."""

    def __init__(self, path, crs, gsd):
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT

        self.src = rasterio.open(path)
        self.crs, self.gsd = crs, gsd
        self.Resampling = Resampling
        self.WarpedVRT = WarpedVRT

    def read(self, left, top, w, h):
        from rasterio.transform import from_origin as fo

        with self.WarpedVRT(self.src, crs=self.crs, transform=fo(left, top, self.gsd, self.gsd), width=w, height=h,
                            resampling=self.Resampling.bilinear, src_nodata=None, nodata=0) as v:
            return v.read(1).astype(np.float32)


def infer_scene(image: str, checkpoint: str, out_dir: str, gsd: float | None = None, ndsm: str | None = None,
                block_px: int = 4096, margin_px: int = 192, tile: int = 512, overlap: int = 128,
                tta: bool = True, device: str = "auto", pp: dict | None = None,
                scene_id: str | None = None) -> gpd.GeoDataFrame:
    pr = load_predictor(checkpoint, device)
    if pr.needs_height and not ndsm:
        raise SystemExit(f"{checkpoint} uses LiDAR height: pass --ndsm (made by run-area from the point cloud)")
    gsd = gsd or pr.gsd
    pp = {**PP_DEFAULTS, **pr.postprocess, **(pp or {})}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene_id = scene_id or Path(image).stem

    with MetricReader(image, gsd) as rd:
        G = rd.grid
        hr = _FloatReader(ndsm, rd.crs, gsd) if (ndsm and pr.needs_height) else None
        prof = dict(driver="GTiff", width=G.width, height=G.height, count=2, dtype="uint8",
                    crs=rd.crs, transform=G.transform, tiled=True, blockxsize=512, blockysize=512,
                    compress="deflate", nodata=0)
        feats = []
        with rasterio.open(out / "probs.tif", "w", **prof) as dst:
            for by in range(0, G.height, block_px):
                for bx in range(0, G.width, block_px):
                    bw, bh = min(block_px, G.width - bx), min(block_px, G.height - by)
                    x0 = G.bounds[0] + (bx - margin_px) * gsd
                    y1 = G.bounds[3] - (by - margin_px) * gsd
                    W, H = bw + 2 * margin_px, bh + 2 * margin_px
                    rgb, valid = rd.read(x0, y1, W, H)
                    if valid.mean() < 0.01:
                        continue
                    hgt = hr.read(x0, y1, W, H) if hr else None
                    p = pr.predict(rgb, hgt, tile=tile, overlap=overlap, tta=tta)
                    p *= valid[None]
                    core = p[:, margin_px:margin_px + bh, margin_px:margin_px + bw]
                    dst.write((np.clip(core, 0, 1) * 254 + 1).astype(np.uint8) * valid[None, margin_px:margin_px + bh, margin_px:margin_px + bw],
                              window=Window(bx, by, bw, bh))
                    inst = instances_from_probs(p[0], p[1], **pp)
                    tr = from_origin(x0, y1, gsd, gsd)
                    polys = polygons_from_instances(inst, p[0], p[1], tr, gsd, **pp)
                    cx0, cy1 = G.bounds[0] + bx * gsd, G.bounds[3] - by * gsd
                    cx1, cy0 = cx0 + bw * gsd, cy1 - bh * gsd
                    for f in polys:
                        c = f["geometry"].centroid
                        if cx0 <= c.x < cx1 and cy0 < c.y <= cy1:
                            feats.append(f)
                    print(f"[infer] block ({by},{bx}) -> {len(feats)} buildings so far", flush=True)
    g = gpd.GeoDataFrame(feats, geometry="geometry", crs=rd.crs) if feats else \
        gpd.GeoDataFrame({"geometry": [], "confidence": [], "status": [], "area_m2": []}, crs=rd.crs)
    g.insert(0, "building_id", [f"{scene_id}-B{i + 1:05d}" for i in range(len(g))])
    g["scene_id"] = scene_id
    g["model"] = pr.name
    g.to_file(out / "buildings.gpkg", driver="GPKG")
    g.to_crs(4326).to_file(out / "buildings.geojson", driver="GeoJSON")
    print(f"[infer] {len(g)} buildings -> {out}  status counts: {g.status.value_counts().to_dict() if len(g) else {}}")
    return g


def export_onnx(checkpoint, out_path, quantize=False, tile=512):
    """U-Net checkpoint -> ONNX (dynamic H/W), optionally int8-quantised, + meta json.

    Measured on the CPU pilot model: PyTorch 2.9 s, ONNX fp32 1.1 s (identical masks),
    ONNX int8 1.6 s (mask IoU 0.968 vs fp32, file 4x smaller) per 512x512 tile."""
    import torch

    from ulpin3d.footprint.model import load_checkpoint

    model, ck = load_checkpoint(checkpoint, "cpu")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp32 = out_path.with_suffix(".fp32.onnx")
    x = torch.randn(1, 3, tile, tile)
    torch.onnx.export(model, x, str(fp32), input_names=["image"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}}, dynamo=False)
    final = fp32
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(fp32), str(out_path), weight_type=QuantType.QUInt8)
        final = out_path
    else:
        fp32.rename(out_path)
        final = out_path
    meta = {"gsd": ck["data_cfg"].get("gsd") or 0.15, "tile": tile, "postprocess": ck.get("postprocess", {}),
            "source_checkpoint": str(checkpoint), "quantized": quantize}
    Path(str(final) + ".json").write_text(json.dumps(meta, indent=2))
    return final
