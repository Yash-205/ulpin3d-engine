"""Point cloud -> rasters on a metric grid: true-ortho RGB, nDSM (height above ground),
and, when the cloud carries classes, a building label mask.

Used for
  * nDSM input to the CPU footprint model when the judges supply LiDAR
  * training/test rasters from colourised, classified LiDAR (Sao Paulo), where the
    'orthophoto' is the colour of the top point in each cell, so image, height and
    labels line up perfectly
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
from scipy import ndimage as ndi

from ulpin3d.pointcloud.ground import height_above_ground
from ulpin3d.pointcloud.io import read_points


def _fill(a, valid):
    if valid.all():
        return a
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def rasterize_cloud(laz, res=0.25, crop=None, label_classes=(6,), bounds=None):
    pts = read_points(laz, crop)
    xyz = pts["xyz"]
    hag, *_ = height_above_ground(xyz)
    if bounds is None:
        L, B = np.floor(xyz[:, :2].min(0) / res) * res
        R, T = np.ceil(xyz[:, :2].max(0) / res) * res
    else:
        L, B, R, T = bounds
    W, H = int(round((R - L) / res)), int(round((T - B) / res))
    ix = np.clip(((xyz[:, 0] - L) / res).astype(int), 0, W - 1)
    iy = np.clip(((T - xyz[:, 1]) / res).astype(int), 0, H - 1)
    cell = iy * W + ix
    order = np.lexsort((xyz[:, 2], cell))  # by cell, then z ascending -> last = top point
    c_sorted = cell[order]
    last = np.r_[c_sorted[1:] != c_sorted[:-1], True]
    top = order[last]
    tc = cell[top]
    valid = np.zeros(H * W, bool)
    valid[tc] = True
    ndsm = np.zeros(H * W, np.float32)
    ndsm[tc] = np.clip(hag[top], 0, None)
    out = {"transform": from_origin(L, T, res, res), "crs": pts["crs"], "shape": (H, W), "res": res}
    valid2 = valid.reshape(H, W)
    out["valid"] = ndi.binary_closing(valid2, iterations=2)
    out["ndsm"] = _fill(ndsm.reshape(H, W), valid2)
    if pts["rgb"] is not None:
        rgb = np.zeros((H * W, 3), np.float32)
        rgb[tc] = pts["rgb"][top]
        rgb = rgb.reshape(H, W, 3)
        idx = ndi.distance_transform_edt(~valid2, return_distances=False, return_indices=True)
        out["rgb"] = (np.clip(rgb[tuple(idx)], 0, 1) * 255).astype(np.uint8)
    if pts["classification"] is not None:
        lab = np.zeros(H * W, np.uint8)
        lab[tc] = np.isin(pts["classification"][top], label_classes)
        lab = _fill(lab.reshape(H, W), valid2)
        out["label"] = ndi.binary_opening(ndi.binary_closing(lab > 0, iterations=1), iterations=1)
    return out


def write_raster(path, arr, transform, crs, nodata=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    a = arr if arr.ndim == 3 else arr[..., None]
    prof = dict(driver="GTiff", width=a.shape[1], height=a.shape[0], count=a.shape[2], dtype=str(a.dtype), crs=crs,
                transform=transform, tiled=True, blockxsize=256, blockysize=256, compress="deflate")
    if a.dtype == np.uint8 and a.shape[2] == 3:
        prof.update(compress="jpeg", photometric="YCBCR")
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as d:
        d.write(np.moveaxis(a, -1, 0))


def make_lidar_chips(tiles: "pd.DataFrame", laz_dir, out_root, crop_m=384.0, res=0.25, chip_px=512, region="brazil",
                     country="Brazil", label_classes=(6,), download_url=None):
    """Cut colourised + classified LiDAR tiles into training chips (image, masks, instances, nDSM).

    Same chip format as data/chips.py plus an `ndsm` column (uint16 PNG, centimetres)."""
    import pandas as pd
    from skimage.segmentation import watershed

    from ulpin3d.data.chips import boundary_from_instances
    from ulpin3d.pointcloud.data import center_crop, download

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, t in tiles.iterrows():
        laz = Path(laz_dir) / t.key
        if not laz.exists() and download_url:
            download(f"{download_url}/{t.key}", laz)
        crop = center_crop(t, crop_m)
        o = rasterize_cloud(laz, res, crop, label_classes)
        H, W = o["shape"]
        lab, nd, rgb, valid = o["label"], o["ndsm"], o["rgb"], o["valid"]
        # instances: split the building mask where the roof height jumps (party walls between houses of different height)
        hg = np.hypot(ndi.sobel(nd, 1), ndi.sobel(nd, 0)) / 8.0
        seeds, _ = ndi.label(ndi.binary_erosion(lab & (hg < 0.6), iterations=2))
        inst = watershed(hg, seeds, mask=lab).astype(np.uint16)
        sdir = out_root / t.split / t.key.replace(".laz", "")
        sdir.mkdir(parents=True, exist_ok=True)
        for r0 in range(0, H - chip_px + 1, chip_px):
            for c0 in range(0, W - chip_px + 1, chip_px):
                sl = (slice(r0, r0 + chip_px), slice(c0, c0 + chip_px))
                if valid[sl].mean() < 0.8:
                    continue
                stem = f"{r0:04d}_{c0:04d}"
                ins = inst[sl]
                edge = boundary_from_instances(ins, 2)
                Image.fromarray(rgb[sl]).save(sdir / f"{stem}.jpg", quality=92)
                m = np.stack([lab[sl] * 255, edge * 255, valid[sl] * 255], -1).astype(np.uint8)
                Image.fromarray(m).save(sdir / f"{stem}_m.png")
                Image.fromarray(ins).save(sdir / f"{stem}_i.png")
                Image.fromarray(np.clip(nd[sl] * 100, 0, 65535).astype(np.uint16)).save(sdir / f"{stem}_h.png")
                rel = lambda n: str((sdir / n).relative_to(out_root))
                rows.append(dict(image=rel(f"{stem}.jpg"), mask=rel(f"{stem}_m.png"), inst=rel(f"{stem}_i.png"),
                                 ndsm=rel(f"{stem}_h.png"), scene_id=t.key, source="lidar", region=region, country=country,
                                 split=t.split, gsd=res, n_buildings=int(len(np.unique(ins)) - 1),
                                 building_frac=float(lab[sl].mean())))
        print(f"[lidar-chips] {t.key} {t.split}: {sum(1 for r in rows if r['scene_id'] == t.key)} chips", flush=True)
    idx = pd.DataFrame(rows)
    idx.to_csv(out_root / "index.csv", index=False)
    return idx
