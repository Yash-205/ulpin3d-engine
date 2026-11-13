"""Model 1-lite: building footprints with gradient boosting instead of deep learning.

Trains on a laptop CPU in minutes, runs anywhere, no GPU at training or inference.
Per pixel (at 0.30 m) it looks at colour, texture and edges at three neighbourhood
sizes and, when LiDAR is available, height above ground (nDSM) and height edges. Two
LightGBM models predict building probability and boundary probability (outlines and
party walls); the same post-processing as the U-Net turns them into one polygon per
building with a confidence.

    python -m ulpin3d.cli train-footprint-lite --chips data/footprint/chips_pilot data/footprint/chips_sp \
        --out models/footprint_lite
    python -m ulpin3d.cli eval-footprint --checkpoint models/footprint_lite --chips ... --split test

Height matters most: with an nDSM a roof is simply 'flat, high, sharp-edged'; RGB alone
has to guess from colour and texture, which is where a CNN is stronger.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

WORK_GSD = 0.30
SIZES = (3, 9, 25)  # neighbourhoods in pixels at 0.30 m: ~1 m, 2.7 m, 7.5 m


def _box_stats(a, k):
    m = ndi.uniform_filter(a, k)
    v = ndi.uniform_filter(a * a, k) - m * m
    return m, np.sqrt(np.clip(v, 0, None))


def pixel_features(rgb: np.ndarray, ndsm: np.ndarray | None = None):
    """rgb uint8 (H, W, 3) at WORK_GSD; ndsm float (H, W) metres or None. -> (H, W, F), names"""
    f, names = [], []
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mx, mn = x.max(-1), x.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    exg = 2 * g - r - b
    for n, a in (("r", r), ("g", g), ("b", b), ("gray", gray), ("sat", sat), ("exg", exg), ("r_b", r - b)):
        f.append(a)
        names.append(n)
    gx, gy = ndi.sobel(gray, 1), ndi.sobel(gray, 0)
    grad = np.hypot(gx, gy)
    lap = np.abs(ndi.laplace(gray))
    for k in SIZES:
        m, s = _box_stats(gray, k)
        f += [m, s, gray - m, ndi.uniform_filter(exg, k), ndi.uniform_filter(sat, k), ndi.uniform_filter(grad, k),
              ndi.uniform_filter(lap, k)]
        names += [f"gray_m{k}", f"gray_s{k}", f"gray_rel{k}", f"exg_m{k}", f"sat_m{k}", f"grad_m{k}", f"lap_m{k}"]
    f += [grad, ndi.maximum_filter(gray, 9) - ndi.minimum_filter(gray, 9)]
    names += ["grad", "gray_range9"]
    if ndsm is not None:
        h = ndsm.astype(np.float32)
        hg = np.hypot(ndi.sobel(h, 1), ndi.sobel(h, 0)) / 8.0
        f += [h, ndi.maximum_filter(h, 9), ndi.minimum_filter(h, 9), _box_stats(h, 9)[1], ndi.uniform_filter(h, 25),
              h - ndi.minimum_filter(h, 25), hg, ndi.maximum_filter(hg, 3), np.abs(ndi.laplace(h)),
              ndi.uniform_filter((h > 2.5).astype(np.float32), 9)]
        names += ["h", "h_max9", "h_min9", "h_std9", "h_m25", "h_rel25", "h_grad", "h_grad_max3", "h_lap", "h_frac9"]
    return np.stack(f, -1).astype(np.float32), names


def _resample(a, factor, order):
    if abs(factor - 1) < 1e-3:
        return a
    z = (factor, factor) + ((1,) if a.ndim == 3 else ())
    return ndi.zoom(a, z, order=order)


def load_chip(root: Path, r, with_inst=False):
    rgb = np.array(Image.open(root / r["image"]).convert("RGB"))
    m = np.array(Image.open(root / r["mask"]))
    nd = None
    if isinstance(r.get("ndsm"), str) and r["ndsm"]:
        nd = np.array(Image.open(root / r["ndsm"])).astype(np.float32) / 100.0
    inst = np.array(Image.open(root / r["inst"])).astype(np.int32) if with_inst else None
    return rgb, m[..., 0] > 127, m[..., 1] > 127, m[..., 2] > 127, nd, inst


def chip_at_work_gsd(rgb, bld, edge, valid, nd, gsd):
    k = gsd / WORK_GSD
    rgb = _resample(rgb, k, 1).astype(np.uint8)
    bld = _resample(bld.astype(np.uint8), k, 0) > 0
    edge = ndi.binary_dilation(_resample(edge.astype(np.uint8), k, 0) > 0) if k < 1 else _resample(edge.astype(np.uint8), k, 0) > 0
    valid = _resample(valid.astype(np.uint8), k, 0) > 0
    nd = _resample(nd, k, 1) if nd is not None else None
    return rgb, bld, edge, valid, nd


def build_training_table(roots, filters=None, per_chip=1200, use_height="auto", seed=0):
    rng = np.random.default_rng(seed)
    Xs, yb, ye, names = [], [], [], None
    for root in roots:
        root = Path(root).resolve()
        idx = pd.read_csv(root / "index.csv")
        for k, v in (filters or {}).items():
            idx = idx[idx[k].isin(v if isinstance(v, list) else [v])]
        has_h = "ndsm" in idx.columns and idx["ndsm"].notna().any()
        if use_height is True and not has_h:
            continue
        for _, r in idx.iterrows():
            rgb, bld, edge, valid, nd, _ = load_chip(root, r)
            if use_height is False:
                nd = None
            elif use_height is True and nd is None:
                continue
            rgb, bld, edge, valid, nd = chip_at_work_gsd(rgb, bld, edge, valid, nd, float(r.gsd))
            F, nm = pixel_features(rgb, nd)
            if names is None:
                names = nm
            elif nm != names:
                continue  # mixed height / no-height chips: train separate models
            sel = []
            for mask, n in ((valid & bld & ~edge, per_chip // 3), (valid & edge, per_chip // 3), (valid & ~bld, per_chip // 3)):
                ii = np.flatnonzero(mask)
                if len(ii):
                    sel.append(rng.choice(ii, min(n, len(ii)), replace=False))
            if not sel:
                continue
            s = np.concatenate(sel)
            Xs.append(F.reshape(-1, F.shape[-1])[s])
            yb.append(bld.ravel()[s])
            ye.append(edge.ravel()[s])
    return np.concatenate(Xs), np.concatenate(yb).astype(int), np.concatenate(ye).astype(int), names


def train_lite(roots, out, filters=None, use_height="auto", rounds=400, per_chip=1200, val_frac=0.1):
    import lightgbm as lgb

    t0 = time.time()
    X, yb, ye, names = build_training_table(roots, filters, per_chip, use_height)
    rng = np.random.default_rng(1)
    va = rng.random(len(yb)) < val_frac
    params = {"objective": "binary", "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 80,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 0}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, y in (("building", yb), ("boundary", ye)):
        d = lgb.Dataset(X[~va], y[~va], feature_name=names)
        dv = lgb.Dataset(X[va], y[va], reference=d)
        m = lgb.train(params, d, rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)])
        m.save_model(str(out / f"{name}.txt"))
        p = m.predict(X[va]) > 0.5
        res[name] = {"val_accuracy": float((p == y[va]).mean()), "rounds": m.best_iteration}
    meta = {"type": "footprint_lite", "work_gsd": WORK_GSD, "features": names, "uses_height": "h" in names,
            "train_pixels": int(len(yb)), "train_seconds": round(time.time() - t0, 1), "val": res,
            "postprocess": {"mask_thr": 0.5, "seed_thr": 0.45, "min_seed_px": 4, "min_area_m2": 6.0}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[lite] trained on {len(yb):,} pixels ({len(names)} features, height={meta['uses_height']}) in {meta['train_seconds']} s -> {out}  {res}")
    return out


class FootprintLite:
    """Same predict interface as the U-Net wrapper: rgb (+ndsm) at self.gsd -> (2, H, W)."""

    def __init__(self, path):
        import lightgbm as lgb

        p = Path(path)
        self.meta = json.loads((p / "meta.json").read_text())
        self.gsd = self.meta["work_gsd"]
        self.uses_height = self.meta["uses_height"]
        self.mb = lgb.Booster(model_file=str(p / "building.txt"))
        self.me = lgb.Booster(model_file=str(p / "boundary.txt"))
        self.postprocess = self.meta.get("postprocess", {})

    def predict(self, rgb, ndsm=None):
        if self.uses_height and ndsm is None:
            raise ValueError("this lite model needs an nDSM (LiDAR); use the RGB-only model instead")
        F, _ = pixel_features(rgb, ndsm if self.uses_height else None)
        X = F.reshape(-1, F.shape[-1])
        H, W = rgb.shape[:2]
        pb = self.mb.predict(X).reshape(H, W)
        pe = self.me.predict(X).reshape(H, W)
        # per-pixel trees are speckly: smooth like a CNN's receptive field would
        pb = ndi.gaussian_filter(pb, self.meta.get("smooth_px", 1.5))
        pe = ndi.gaussian_filter(pe, 0.7)
        return np.stack([pb, pe]).astype(np.float32)


def is_lite(path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "meta.json").exists() and json.loads((p / "meta.json").read_text()).get("type") == "footprint_lite"


def evaluate_lite(model_path, chips_root, split="test", out_dir=None, filters=None):
    """Same metrics as evaluate.py, for a lite model."""
    from ulpin3d.footprint.metrics import BoundaryMeter, ObjectMeter, PixelMeter
    from ulpin3d.footprint.postprocess import DEFAULTS, instances_from_probs

    m = FootprintLite(model_path)
    pp = {**DEFAULTS, **m.postprocess}
    root = Path(chips_root).resolve()
    idx = pd.read_csv(root / "index.csv")
    idx = idx[idx.split == split]
    for k, v in (filters or {}).items():
        idx = idx[idx[k].isin(v if isinstance(v, list) else [v])]
    groups = {}
    t0 = time.time()
    for _, r in idx.iterrows():
        rgb, bld, edge, valid, nd, inst = load_chip(root, r, with_inst=True)
        k = float(r.gsd) / m.gsd
        rgb_w = _resample(rgb, k, 1).astype(np.uint8)
        nd_w = _resample(nd, k, 1) if nd is not None else None
        if m.uses_height and nd_w is None:
            continue
        pr = m.predict(rgb_w, nd_w)
        pr = np.stack([ndi.zoom(p, (bld.shape[0] / p.shape[0], bld.shape[1] / p.shape[1]), order=1) for p in pr])
        pb, pe = pr[0], pr[1]
        pred = pb > pp["mask_thr"]
        inst_p = instances_from_probs(pb, pe, **{**pp, "min_seed_px": int(pp.get("min_seed_px", 4) / k ** 2)}, gsd=float(r.gsd))
        for key in ("ALL", f"region={r.region}", f"scene={r.scene_id}"):
            g = groups.setdefault(key, (PixelMeter(), BoundaryMeter(2), ObjectMeter(0.5)))
            g[0].update(pred, bld, valid)
            g[1].update(pred, bld, valid)
            g[2].update(inst_p, inst, valid)
    rows = [{"group": k, **a.compute(), **b.compute(), **c.compute()} for k, (a, b, c) in groups.items()]
    df = pd.DataFrame(rows).sort_values("group")
    print(df[["group", "iou", "f1", "boundary_f1", "obj_f1", "obj_precision", "obj_recall", "n_gt"]].round(3).to_string(index=False))
    print(f"[lite] {len(idx)} chips in {time.time() - t0:.0f} s")
    if out_dir:
        o = Path(out_dir)
        o.mkdir(parents=True, exist_ok=True)
        df.to_csv(o / "per_group.csv", index=False)
        (o / "report.json").write_text(json.dumps({"model": str(model_path), "overall": df[df.group == "ALL"].iloc[0].to_dict()}, indent=2, default=float))
    return df
