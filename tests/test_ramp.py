"""ramp tiles (EPSG:4326 or UTM, ~0.3 m) -> 0.15 m UTM chips with the label still on the roof."""
import io

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ulpin3d.data.ramp import split_for, tile_to_chip


def _tile(lon0=76.0, lat0=12.5, res=2.7e-6):
    img = np.zeros((3, 256, 256), np.uint8)
    img[:, 100:140, 100:140] = 255                       # a white 40 px (~12 m) roof
    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(lon0, lat0, res, res)) as d:
        d.write(img)
    x0, y0 = lon0 + 100 * res, lat0 - 100 * res
    x1, y1 = lon0 + 140 * res, lat0 - 140 * res
    lab = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}]}
    return buf.getvalue(), lab


def test_label_lands_on_the_roof():
    tif, lab = _tile()
    rgb, inst, valid, crs, *_ = tile_to_chip(tif, lab, gsd=0.15, chip_px=512)
    assert crs.to_epsg() == 32643 and inst.max() == 1 and valid.mean() > 0.9
    roof = rgb[..., 0] > 127
    iou = (roof & (inst > 0)).sum() / (roof | (inst > 0)).sum()
    assert iou > 0.85, iou
    assert 0.8 < (inst > 0).sum() * 0.15 ** 2 / 140 < 1.2      # ~12 m x 12 m, area preserved


def test_split_keeps_cells_together_and_test_only_from_india():
    a = split_for("karnataka_india", 76.0012, 12.5013, ["karnataka_india"])
    b = split_for("karnataka_india", 76.0049, 12.5041, ["karnataka_india"])  # same 0.01 deg cell
    assert a == b
    splits = {split_for("dhaka_bangladesh", 90 + i * 0.01, 23.8, ["karnataka_india"])[0] for i in range(300)}
    assert "test" not in splits and {"train", "val"} <= splits
