"""Outline fixes that do not depend on the photo model: split by plot, LiDAR roof, plan pins."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from ulpin3d.area import _assign_parcels, _plan_pins, _refine_with_lidar, _split_by_parcels

CRS = "EPSG:32643"


def _parcels():
    return gpd.GeoDataFrame({"ulpin": ["P1", "P2"]}, geometry=[box(0, 0, 10, 20), box(10, 0, 20, 20)], crs=CRS)


def test_blob_over_two_plots_is_split_but_a_sliver_is_an_encroachment():
    b = gpd.GeoDataFrame({"building_id": ["B1", "B2"], "confidence_3d": [0.9, 0.9]},
                         geometry=[box(2, 2, 18, 10),          # two attached houses merged into one blob
                                   box(1, 12, 10.8, 18)],      # one house over the line by 0.8 m
                         crs=CRS)
    out, n = _split_by_parcels(b, _parcels())
    assert n == 1 and len(out) == 3
    out = _assign_parcels(out, _parcels())
    assert sorted(out.parcel_ulpin) == ["P1", "P1", "P2"]
    assert (out.encroach_m2 > 0).sum() == 1          # the sliver is reported, not split off


def test_lidar_roof_replaces_a_clipped_photo_outline_but_not_the_neighbours_house():
    par = _parcels()
    b = gpd.GeoDataFrame({"building_id": ["B1"], "parcel_ulpin": ["P1"], "building_issues": [[]]},
                         geometry=[box(2, 2, 5, 10)], crs=CRS)         # model saw only the sunny 40%
    roof = gpd.GeoDataFrame(geometry=[box(2, 2, 18, 10)], crs=CRS)    # LiDAR: attached roofs of P1 + P2
    out, n = _refine_with_lidar(b, roof, par)
    g = out.geometry[0]
    assert n == 1 and out.outline_source[0] == "lidar_roof"
    assert abs(g.area - 8 * 8) < 3 and g.bounds[2] <= 10.4          # whole own roof, cut at the plot line


def test_plans_index_pins_plans_to_parcels(tmp_path):
    (tmp_path / "plans").mkdir()
    pd.DataFrame({"File": ["a.dxf"], "Parcel_ULPIN": ["p1"]}).to_csv(tmp_path / "plans" / "plans_index.csv", index=False)
    assert _plan_pins(tmp_path, _parcels()) == {"a.dxf": "P1"}


def test_plan_placed_on_weak_image_evidence_without_lidar(tmp_path):
    """No LiDAR; the photo model is sure about only 40% of a roof (the rest looks like soil). The
    pinned plan is slid over the plot and lands on the true roof, anchored by the confident part
    and the model's roof-edge map."""
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    from ulpin3d.area import _place_plan_on_image

    res, tr = 0.15, from_origin(0, 30, 0.15, 0.15)
    true = box(8, 8, 28, 18)                       # 20 x 10 m building
    sure = box(20, 8, 28, 18)                      # the part the model saw clearly
    shp = (200, 267)
    p = rasterize([(true, 60), (sure, 240)], out_shape=shp, transform=tr).astype(np.uint8)
    edge = rasterize([(true.exterior.buffer(0.3), 200)], out_shape=shp, transform=tr).astype(np.uint8)
    f = tmp_path / "probs.tif"
    with rasterio.open(f, "w", driver="GTiff", width=shp[1], height=shp[0], count=2, dtype="uint8",
                       crs="EPSG:32643", transform=tr) as d:
        d.write(p, 1)
        d.write(edge, 2)
    parcel = box(1, 1, 39, 29)
    plan = box(0, 0, 20, 10)                       # the sanctioned outline, in plan coordinates
    placed, score = _place_plan_on_image(plan, parcel, f, hint=sure)
    assert placed is not None and placed.intersection(true).area / placed.union(true).area > 0.9


def test_terraced_row_is_split_into_every_house_and_nothing_is_dropped():
    """LiDAR sees a row of 6 terraced houses as one roof; each house has its own narrow plot."""
    plots = gpd.GeoDataFrame({"ulpin": [f"P{i}" for i in range(6)]},
                             geometry=[box(6 * i, 0, 6 * i + 6, 20) for i in range(6)], crs=CRS)
    roof = gpd.GeoDataFrame({"building_id": ["B1"], "building_issues": [[]], "confidence_3d": [0.9]},
                            geometry=[box(0, 2, 36, 12)], crs=CRS)
    out, n = _split_by_parcels(roof, plots)
    assert n == 1 and len(out) == 6
    assert abs(out.geometry.area.sum() - 36 * 10) < 1          # every house kept, no area lost
