"""Train, evaluate and apply the LiDAR point classifier (LightGBM).

    python -m ulpin3d.cli pc-prepare --config configs/pointcloud.yaml   # download + features
    python -m ulpin3d.cli pc-train   --config configs/pointcloud.yaml
    python -m ulpin3d.cli pc-eval    --config configs/pointcloud.yaml
    python -m ulpin3d.cli pc-infer   --laz judges.laz --model runs/pc/model.txt --out judges_classified.laz

Why gradient boosting rather than a point transformer: it trains on a laptop CPU
in minutes, handles missing sensor attributes natively, carries geometric
features across countries well, and gives per-point class probabilities we
can turn into confidence for the cadastre. A deep model can be swapped in
later behind the same interface.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ulpin3d.pointcloud.data import CLASS_NAMES, IGNORE, TO_ASPRS, map_classes
from ulpin3d.pointcloud.features import compute_features
from ulpin3d.pointcloud.ground import height_above_ground
from ulpin3d.pointcloud.io import read_points, write_classified


def featurize_cloud(path, crop=None, voxel=0.3, ks=(10, 25, 50), scheme=None, drop=None,
                    ground_kw=None):
    pts = read_points(path, crop)
    hag, *_ = height_above_ground(pts["xyz"], **(ground_kw or {}))
    d = drop or {}
    X, names, sel, inv = compute_features(pts, hag, voxel, ks, d.get("rgb", False),
                                          d.get("intensity", False), d.get("returns", False))
    y = None
    if scheme and pts["classification"] is not None:
        y = clean_labels(map_classes(pts["classification"][sel], scheme), hag[sel], scheme)
    return X, names, y, pts, sel, inv


def clean_labels(y: np.ndarray, hag: np.ndarray, scheme: str | None) -> np.ndarray:
    """Sao Paulo uses class 19 inconsistently: walls/objects in most districts, bare
    ground in others (e.g. Heliopolis). 'Other' points lying on the ground are
    ambiguous, so they are ignored for training and scoring."""
    if scheme == "saopaulo":
        y = y.copy()
        y[(y == 3) & (hag < 0.3)] = IGNORE
    return y


def balanced_rows(y: np.ndarray, per_class: int, rng) -> np.ndarray:
    keep = []
    for c in range(len(CLASS_NAMES)):
        ix = np.nonzero(y == c)[0]
        if len(ix):
            keep.append(rng.choice(ix, min(per_class, len(ix)), replace=False))
    return np.concatenate(keep) if keep else np.array([], int)


def train_lgbm(X, y, Xv=None, yv=None, params=None, rounds=600, names=None):
    import lightgbm as lgb

    p = {"objective": "multiclass", "num_class": len(CLASS_NAMES), "learning_rate": 0.08,
         "num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.7, "bagging_fraction": 0.7,
         "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 0}
    p.update(params or {})
    dtr = lgb.Dataset(X, y, feature_name=names or "auto", free_raw_data=False)
    valid = [lgb.Dataset(Xv, yv, reference=dtr)] if Xv is not None and len(Xv) else []
    cb = [lgb.log_evaluation(50)]
    if valid:
        cb.append(lgb.early_stopping(50, verbose=True))
    return lgb.train(p, dtr, rounds, valid_sets=valid, callbacks=cb)


def smooth_labels(xyz, proba, k=8):
    """Average class probabilities over the k nearest neighbours (removes salt-and-pepper)."""
    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=min(k, len(xyz)))
    return proba[nn].mean(1)


def confusion(y_true, y_pred, n=len(CLASS_NAMES)):
    m = (y_true >= 0)
    return np.bincount(n * y_true[m] + y_pred[m], minlength=n * n).reshape(n, n)


def scores(cm):
    tp = np.diag(cm).astype(float)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return {"overall_accuracy": float(tp.sum() / max(cm.sum(), 1)), "mean_iou": float(iou.mean()),
            **{f"iou_{c}": float(v) for c, v in zip(CLASS_NAMES, iou)},
            **{f"precision_{c}": float(v) for c, v in zip(CLASS_NAMES, prec)},
            **{f"recall_{c}": float(v) for c, v in zip(CLASS_NAMES, rec)}}


def classify_cloud(model, path, out_path=None, crop=None, voxel=0.3, ks=(10, 25, 50), smooth_k=8,
                   ground_kw=None):
    """Classify every point of a LAS/LAZ file. Returns (unified labels per point, max prob per point)."""
    X, names, _, pts, sel, inv = featurize_cloud(path, crop, voxel, ks, ground_kw=ground_kw)
    order = model.feature_name()
    if order and order[0] != "Column_0":
        X = X[:, [names.index(n) for n in order]]
    proba = model.predict(X)
    if smooth_k:
        proba = smooth_labels(pts["xyz"][sel], proba, smooth_k)
    lab_v = proba.argmax(1)
    conf_v = proba.max(1)
    lab, conf = lab_v[inv], conf_v[inv]
    if out_path:
        write_classified(path, out_path, TO_ASPRS[lab], {"cls_conf": (conf * 100).astype(np.uint8)}, crop)
    return lab, conf, pts


def evaluate_cloud(model, path, scheme, crop=None, voxel=0.3, ks=(10, 25, 50), smooth_k=8, ground_kw=None,
                   drop=None):
    X, names, _, pts, sel, inv = featurize_cloud(path, crop, voxel, ks, drop=drop, ground_kw=ground_kw)
    hag_v = X[:, names.index("hag")].copy()
    order = model.feature_name()
    if order and order[0] != "Column_0":
        X = X[:, [names.index(n) for n in order]]
    proba = model.predict(X)
    if smooth_k:
        proba = smooth_labels(pts["xyz"][sel], proba, smooth_k)
    pred = proba.argmax(1)[inv]
    y = clean_labels(map_classes(pts["classification"], scheme), hag_v[inv], scheme)
    return confusion(y.astype(int), pred.astype(int))


def run_prepare(cfg: dict):
    """Download tiles, compute features, save per-tile .npz shards."""
    from ulpin3d.pointcloud.data import SP_BUCKET, center_crop, download, saopaulo_tile_index, select_saopaulo_tiles

    work = Path(cfg["work_dir"])
    sp = cfg["saopaulo"]
    idx = saopaulo_tile_index(work / "cache")
    tiles = select_saopaulo_tiles(idx, sp["test_tiles"], sp["n_train"], sp["n_val"], seed=cfg.get("seed", 0))
    tiles.to_csv(work / "tiles.csv", index=False)
    rng = np.random.default_rng(cfg.get("seed", 0))
    fc = cfg["features"]
    for _, t in tiles.iterrows():
        shard = work / "shards" / f"{t.key}.npz"
        if shard.exists():
            continue
        laz = download(f"{SP_BUCKET}/{t.key}", work / "laz" / t.key)
        crop = center_crop(t, sp.get("crop_m", 250)) if sp.get("crop_m") else None
        drop = {}
        if t.split == "train":
            pd_ = cfg.get("augment", {})
            drop = {"rgb": rng.random() < pd_.get("drop_rgb", 0.3),
                    "intensity": rng.random() < pd_.get("drop_intensity", 0.3),
                    "returns": rng.random() < pd_.get("drop_returns", 0.2)}
        X, names, y, *_ = featurize_cloud(laz, crop, fc["voxel"], tuple(fc["ks"]), "saopaulo", drop,
                                          cfg.get("ground", {}))
        keep = balanced_rows(y, fc.get("per_class_per_tile", 25_000), rng) if t.split == "train" else \
            rng.choice(len(y), min(len(y), fc.get("eval_points_per_tile", 150_000)), replace=False)
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard, X=X[keep], y=y[keep], names=np.array(names), split=t.split, drop=json.dumps(drop))
        print(f"[pc-prepare] {t.key} {t.split:5s} rows={len(keep):7d} drop={drop} "
              f"classes={np.bincount(y[keep][y[keep] >= 0], minlength=4).tolist()}", flush=True)
        if not cfg.get("keep_laz", False) and t.split == "train":
            laz.unlink(missing_ok=True)


def _load_shards(work: Path, split: str):
    Xs, ys, names = [], [], None
    for p in sorted((work / "shards").glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        if str(z["split"]) != split:
            continue
        names = list(z["names"])
        y = clean_labels(z["y"], z["X"][:, names.index("hag")], "saopaulo")
        m = y >= 0
        Xs.append(z["X"][m])
        ys.append(y[m])
    if not Xs:
        return None, None, names
    return np.concatenate(Xs), np.concatenate(ys), names


def run_train(cfg: dict):
    import lightgbm as lgb  # noqa: F401

    work = Path(cfg["work_dir"])
    X, y, names = _load_shards(work, "train")
    Xv, yv, _ = _load_shards(work, "val")
    print(f"[pc-train] train rows {len(y)} val rows {0 if yv is None else len(yv)} features {len(names)}")
    model = train_lgbm(X, y, Xv, yv, cfg.get("lgbm", {}), cfg.get("rounds", 600), names)
    out = Path(cfg.get("out_dir", work / "model"))
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "model.txt"))
    (out / "features.json").write_text(json.dumps(names))
    imp = pd.Series(model.feature_importance("gain"), index=names).sort_values(ascending=False)
    imp.to_csv(out / "feature_importance.csv")
    print(imp.head(15).round(0).to_string())
    if Xv is not None:
        pv = model.predict(Xv).argmax(1)
        s = scores(confusion(yv.astype(int), pv))
        (out / "val_metrics.json").write_text(json.dumps(s, indent=2))
        print("[pc-train] val", json.dumps({k: round(v, 3) for k, v in s.items() if "iou" in k or "acc" in k}))
    return out / "model.txt"


def load_model(path):
    import lightgbm as lgb

    m = lgb.Booster(model_file=str(path))
    fj = Path(path).with_name("features.json")
    if fj.exists() and m.feature_name()[0] == "Column_0":
        names = json.loads(fj.read_text())
        m.feature_name = lambda: names  # type: ignore[method-assign]
    return m


def run_eval(cfg: dict, model_path=None):
    """Full-resolution evaluation on held-out test tiles, plus sensor-ablation runs."""
    from ulpin3d.pointcloud.data import SP_BUCKET, center_crop, download

    work = Path(cfg["work_dir"])
    out = Path(cfg.get("out_dir", work / "model"))
    model = load_model(model_path or out / "model.txt")
    tiles = pd.read_csv(work / "tiles.csv")
    fc = cfg["features"]
    crop_m = cfg["saopaulo"].get("crop_m", 250)
    variants = {"full": {}, "no_rgb": {"rgb": True}, "no_rgb_no_intensity": {"rgb": True, "intensity": True}}
    rows, total = [], {k: 0 for k in variants}
    for _, t in tiles[tiles.split == "test"].iterrows():
        laz = download(f"{SP_BUCKET}/{t.key}", work / "laz" / t.key)
        crop = center_crop(t, crop_m) if crop_m else None
        for vname, drop in variants.items():
            cm = evaluate_cloud(model, laz, "saopaulo", crop, fc["voxel"], tuple(fc["ks"]),
                                cfg.get("smooth_k", 8), cfg.get("ground", {}), drop)
            total[vname] = total[vname] + cm
            s = scores(cm)
            rows.append({"tile": t.key, "variant": vname, **s})
            print(f"[pc-eval] {t.key} {vname:20s} OA={s['overall_accuracy']:.3f} mIoU={s['mean_iou']:.3f} "
                  f"building IoU={s['iou_building']:.3f}", flush=True)
    for vname, cm in total.items():
        rows.append({"tile": "ALL", "variant": vname, **scores(cm)})
        np.savetxt(out / f"confusion_{vname}.csv", cm, fmt="%d", delimiter=",",
                   header=",".join(CLASS_NAMES))
    df = pd.DataFrame(rows)
    df.to_csv(out / "test_metrics.csv", index=False)
    print(df[df.tile == "ALL"][["variant", "overall_accuracy", "mean_iou"] + [f"iou_{c}" for c in CLASS_NAMES]]
          .round(3).to_string(index=False))
    return df
