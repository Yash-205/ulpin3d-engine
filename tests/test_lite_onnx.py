"""CPU LightGBM footprint model, ONNX export, and LiDAR rasterisation."""
import numpy as np
import pandas as pd
import pytest

from ulpin3d.data.chips import make_scene_chips
from ulpin3d.data.synthetic import make_synthetic_scene


@pytest.fixture(scope="module")
def chips(tmp_path_factory):
    d = tmp_path_factory.mktemp("lite")
    rows = []
    for i in range(3):
        tif, lab, _ = make_synthetic_scene(d / f"s{i}.tif", d / f"s{i}.gpkg", size_m=60, gsd=0.1, seed=10 + i)
        row = pd.Series(dict(scene_id=f"s{i}", image_url=str(tif), split="test" if i == 2 else "train",
                             source="local", region="india", country="India"))
        rows += make_scene_chips(row, lab, d / "chips", gsd=0.15, chip_px=256, max_chips=6, neg_frac=0)
    pd.DataFrame(rows).to_csv(d / "chips" / "index.csv", index=False)
    return d / "chips"


def test_lite_trains_and_finds_buildings(chips, tmp_path):
    from ulpin3d.footprint.lite import evaluate_lite, is_lite, train_lite

    out = train_lite([chips], tmp_path / "lite", filters={"split": ["train"]}, use_height=False, rounds=60, per_chip=600)
    assert is_lite(out)
    df = evaluate_lite(out, chips, "test")
    assert df[df.group == "ALL"].iloc[0]["iou"] > 0.6


def test_lite_runs_through_infer_scene(chips, tmp_path):
    from ulpin3d.footprint.infer import infer_scene
    from ulpin3d.footprint.lite import train_lite

    out = train_lite([chips], tmp_path / "lite", filters={"split": ["train"]}, use_height=False, rounds=40, per_chip=400)
    tif = chips.parent / "s2.tif"
    g = infer_scene(str(tif), str(out), str(tmp_path / "o"), tta=False)
    assert len(g) > 3 and {"confidence", "status"} <= set(g.columns)


def test_onnx_export_matches_torch(tmp_path):
    pytest.importorskip("onnx")
    import torch

    from ulpin3d.footprint.infer import OnnxPredictor, TorchPredictor, export_onnx
    from ulpin3d.footprint.model import build_model, save_checkpoint

    torch.manual_seed(0)  # random init is unseeded otherwise: some draws push float32 drift past the tolerance
    m = build_model("Unet", "resnet18", None)
    ck = tmp_path / "m.pt"
    save_checkpoint(ck, m, {"model": {"arch": "Unet", "encoder": "resnet18", "encoder_weights": None},
                            "data": {"gsd": 0.15, "chip_px": 256}})
    onnx_path = export_onnx(ck, tmp_path / "m.onnx", tile=256)
    rgb = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
    a = TorchPredictor(ck, "cpu").predict(rgb, tile=256, tta=False)
    b = OnnxPredictor(onnx_path).predict(rgb, tile=256, tta=False)
    assert np.abs(a - b).max() < 1e-3


def test_rasterize_cloud_heights(tmp_path):
    from ulpin3d.pointcloud.rasterize import rasterize_cloud
    from ulpin3d.pointcloud.synthetic import make_synthetic_lidar

    p, meta = make_synthetic_lidar(tmp_path / "c.laz", size_m=50, density=10, seed=4)
    r = rasterize_cloud(p, res=0.5)
    r2 = rasterize_cloud(p, res=0.5, label_classes=(1,))  # synthetic clouds store building as class 1
    assert r["ndsm"].max() > 5 and r2["label"].mean() > 0.1
