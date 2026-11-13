"""Model 2 tests on synthetic Indian LiDAR: ground filter, features, classifier, buildings."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from ulpin3d.pointcloud.buildings import attach_heights, estimate_floors
from ulpin3d.pointcloud.classify import confusion, featurize_cloud, scores, train_lgbm
from ulpin3d.pointcloud.data import map_classes
from ulpin3d.pointcloud.ground import height_above_ground
from ulpin3d.pointcloud.io import read_points
from ulpin3d.pointcloud.synthetic import make_synthetic_lidar


@pytest.fixture(scope="module")
def clouds(tmp_path_factory):
    d = tmp_path_factory.mktemp("lidar")
    return [make_synthetic_lidar(d / f"s{i}.laz", size_m=70, density=12, seed=i) for i in range(3)]


def test_ground_filter(clouds):
    pts = read_points(clouds[0][0])
    y = map_classes(pts["classification"], "synthetic")
    hag, *_ = height_above_ground(pts["xyz"])
    g = hag < 0.5
    assert (g & (y == 0)).sum() / (y == 0).sum() > 0.97
    assert (g & (y == 1)).sum() / (y == 1).sum() < 0.05


def test_classifier_generalises_to_unseen_tile(clouds):
    Xs, ys = [], []
    for p, _ in clouds[:2]:
        X, names, y, *_ = featurize_cloud(p, voxel=0.3, ks=(10, 25), scheme="synthetic")
        Xs.append(X)
        ys.append(y)
    m = train_lgbm(np.concatenate(Xs), np.concatenate(ys), rounds=80, names=names,
                   params={"num_leaves": 31, "verbose": -1})
    X, _, y, *_ = featurize_cloud(clouds[2][0], voxel=0.3, ks=(10, 25), scheme="synthetic")
    s = scores(confusion(y.astype(int), m.predict(X).argmax(1)))
    assert s["iou_ground"] > 0.9 and s["iou_building"] > 0.9 and s["overall_accuracy"] > 0.93


def test_floor_estimates_from_lidar(clouds):
    p, meta = clouds[1]
    pts = read_points(p)
    y = map_classes(pts["classification"], "synthetic")
    hag, *_ = height_above_ground(pts["xyz"])
    m = pd.DataFrame(meta)
    fp = gpd.GeoDataFrame(m, geometry=[box(r.x, r.y, r.x + r.w, r.y + r.d).buffer(-0.15) for r in m.itertuples()])
    g = attach_heights(fp, pts["xyz"], hag, np.where(y == 1, 1, 0))
    ok = g.floors_est.notna()
    assert ok.mean() > 0.9
    assert (g.floors_est[ok] == g.floors[ok]).mean() > 0.9
    assert (g.roof_h_main[ok] - g.height[ok]).abs().median() < 0.3


def test_estimate_floors_band():
    f, lo, hi = estimate_floors(3.1 * 4 + 0.4, parapet=0)
    assert f == 4 and lo <= 4 <= hi


def test_fuse_photo_footprints_with_lidar(clouds, tmp_path):
    from ulpin3d.fuse import fuse
    from ulpin3d.pointcloud.data import TO_ASPRS
    from ulpin3d.pointcloud.io import write_classified

    p, meta = clouds[2]
    pts = read_points(p)
    y = map_classes(pts["classification"], "synthetic")
    write_classified(p, tmp_path / "cls.laz", TO_ASPRS[y])  # as pc-infer would write it
    m = pd.DataFrame(meta)
    keep = m.iloc[::2]  # photo model "missed" every other building
    fp = gpd.GeoDataFrame({"building_id": [f"B{i}" for i in range(len(keep))], "confidence": 0.9,
                           "floors_true": keep.floors.values},
                          geometry=[box(r.x, r.y, r.x + r.w, r.y + r.d).buffer(-0.15) for r in keep.itertuples()],
                          crs="EPSG:32643")
    fp.to_file(tmp_path / "b.gpkg", driver="GPKG")
    out = fuse(tmp_path / "b.gpkg", tmp_path / "cls.laz", tmp_path / "b3d.geojson")
    photo = out[out.source == "photo+lidar"]
    assert len(photo) == len(keep)
    assert (photo.floors_est.astype(int) == photo.floors_true.astype(int)).mean() > 0.85
    assert (photo.z_max.astype(float) > photo.z_min.astype(float)).all()
    assert (out.source == "lidar_only").sum() >= 1  # missed buildings recovered from LiDAR
