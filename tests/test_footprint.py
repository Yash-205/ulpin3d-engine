"""Model 1 tests: raster IO, chips, post-processing, metrics, registration, training smoke."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

from ulpin3d.common.raster import MetricReader
from ulpin3d.data.chips import boundary_from_instances, make_scene_chips
from ulpin3d.data.register import best_shift
from ulpin3d.data.synthetic import make_synthetic_scene
from ulpin3d.footprint.metrics import ObjectMeter, PixelMeter, match_instances
from ulpin3d.footprint.postprocess import instances_from_probs, polygons_from_instances


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    tif, lab, n = make_synthetic_scene(d / "s.tif", d / "s.gpkg", size_m=60, gsd=0.1, seed=1)
    return d, tif, lab, n


def test_reader_resamples_and_pads(scene):
    _, tif, _, _ = scene
    with MetricReader(str(tif), gsd=0.2) as rd:
        assert rd.crs.to_epsg() == 32643
        assert abs(rd.native_gsd - 0.1) < 1e-6
        rgb, valid = rd.read(rd.bounds[0] - 5, rd.bounds[3] + 5, 100)  # half outside the image
        assert rgb.shape == (100, 100, 3)
        assert not valid[:10, :10].any() and valid[60:, 60:].all()


def test_chips_have_shared_walls(scene, tmp_path):
    d, tif, lab, n = scene
    row = pd.Series(dict(scene_id="syn", image_url=str(tif), split="train", source="local",
                         region="india", country="India"))
    rows = make_scene_chips(row, lab, tmp_path, gsd=0.15, chip_px=128, max_chips=10, neg_frac=0)
    assert rows and all(r["n_buildings"] > 0 for r in rows)
    from PIL import Image

    m = np.array(Image.open(tmp_path / rows[0]["mask"]))
    assert m[..., 1].sum() > 0 and (m[..., 1] > 0).sum() < (m[..., 0] > 0).sum()


def test_boundary_marks_party_wall():
    inst = np.zeros((20, 30), np.uint16)
    inst[5:15, 5:15], inst[5:15, 15:25] = 1, 2  # two attached buildings
    b = boundary_from_instances(inst, 1)
    assert b[10, 14] and b[10, 15]  # shared wall on both sides
    assert not b[10, 10]


def _two_attached(gsd=0.1):
    pb = np.zeros((80, 120), np.float32)
    pe = np.zeros_like(pb)
    pb[20:60, 20:60] = pb[20:60, 60:100] = 0.95
    pe[20:60, 58:62] = 0.9  # shared wall
    pe[20:22, 20:100] = pe[58:60, 20:100] = pe[20:60, 20:22] = pe[20:60, 98:100] = 0.9
    gt = np.zeros(pb.shape, np.int32)
    gt[20:60, 20:60], gt[20:60, 60:100] = 1, 2
    return pb, pe, gt


def test_postprocess_splits_attached_buildings():
    pb, pe, gt = _two_attached()
    inst = instances_from_probs(pb, pe)
    assert len(np.unique(inst[inst > 0])) == 2
    tp, fp, fn, ious = match_instances(inst, gt, np.ones_like(gt, bool))
    assert (tp, fp, fn) == (2, 0, 0) and min(ious) > 0.85
    polys = polygons_from_instances(inst, pb, pe, from_origin(0, 8, 0.1, 0.1), 0.1)
    assert len(polys) == 2
    assert all(abs(p["area_m2"] - 16.0) < 1.5 for p in polys)
    assert all(p["status"] in ("validated", "provisional") for p in polys)


def test_metrics_perfect():
    _, _, gt = _two_attached()
    pm, om = PixelMeter(), ObjectMeter()
    v = np.ones_like(gt, bool)
    pm.update(gt > 0, gt > 0, v)
    om.update(gt, gt, v)
    assert pm.compute()["iou"] == 1.0 and om.compute()["obj_f1"] == 1.0


@pytest.mark.parametrize("dy,dx", [(0, 0), (7, -4), (-12, 9)])
def test_register_recovers_shift(dy, dx):
    rng = np.random.default_rng(0)
    truth = np.zeros((256, 256), bool)
    for _ in range(25):
        y, x = rng.integers(10, 220, 2)
        truth[y:y + rng.integers(8, 30), x:x + rng.integers(8, 30)] = True
    label = np.roll(np.roll(truth, -dy, 0), -dx, 1)  # labels offset from reality
    gy, gx, peak = best_shift(truth.astype(np.float32), label, 20)
    assert (gy, gx) == (dy, dx) and peak > 0.5


@pytest.mark.slow
def test_train_eval_infer_smoke(scene, tmp_path):
    from ulpin3d.footprint.infer import infer_scene
    from ulpin3d.footprint.train import train

    d, tif, lab, _ = scene
    row = pd.Series(dict(scene_id="syn", image_url=str(tif), split="train", source="local",
                         region="india", country="India"))
    rows = make_scene_chips(row, lab, tmp_path / "chips", gsd=0.15, chip_px=128, max_chips=10, neg_frac=0)
    df = pd.DataFrame(rows)
    df.loc[df.index[-2:], "split"] = "val"
    df.to_csv(tmp_path / "chips" / "index.csv", index=False)
    cfg = {"data": {"chips_root": str(tmp_path / "chips"), "gsd": 0.15, "chip_px": 128, "crop_px": 128,
                    "train_filter": {"split": ["train"]}, "val_filter": {"split": ["val"]}},
           "model": {"arch": "Unet", "encoder": "resnet18", "encoder_weights": None},
           "train": {"out_dir": str(tmp_path / "run"), "epochs": 1, "steps_per_epoch": 2, "batch_size": 2,
                     "lr": 1e-3, "num_workers": 0}}
    ck = train(cfg)
    assert Path(ck).exists()
    g = infer_scene(str(tif), str(ck), str(tmp_path / "out"), tta=False, block_px=256, margin_px=32)
    assert (tmp_path / "out" / "buildings.geojson").exists() and (tmp_path / "out" / "probs.tif").exists()
    assert {"confidence", "status", "building_id"} <= set(g.columns)
