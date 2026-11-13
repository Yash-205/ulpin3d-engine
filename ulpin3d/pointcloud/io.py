"""LAS/LAZ reading and writing with a uniform field set.

The judges' point cloud may be LAS 1.2-1.4, with or without RGB, intensity or
return numbers, and with or without an existing classification. Everything
downstream works with the dict returned by `read_points`; missing attributes
are None and become NaN features (LightGBM handles missing values natively).
"""
from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np


def read_points(path: str | Path, crop=None) -> dict:
    las = laspy.read(str(path))
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    m = np.ones(len(x), bool)
    if crop is not None:
        m = (x >= crop[0]) & (y >= crop[1]) & (x <= crop[2]) & (y <= crop[3])
    dims = set(las.point_format.dimension_names)

    def get(name, dtype=np.float32):
        if name not in dims:
            return None
        a = np.asarray(las[name])[m].astype(dtype)
        return a if a.size and np.any(a != a.flat[0]) else None  # constant -> missing

    rgb = None
    if {"red", "green", "blue"} <= dims:
        r = np.stack([np.asarray(las[c])[m] for c in ("red", "green", "blue")], 1).astype(np.float32)
        if r.max() > 0:
            rgb = r / (65535.0 if r.max() > 255 else 255.0)
    try:
        crs = las.header.parse_crs()
    except Exception:
        crs = None
    return {
        "xyz": np.stack([x[m], y[m], z[m]], 1).astype(np.float64),
        "intensity": get("intensity"),
        "return_number": get("return_number"),
        "number_of_returns": get("number_of_returns"),
        "classification": get("classification", np.int32) if "classification" in dims else None,
        "rgb": rgb,
        "crs": crs,
        "header": las.header,
    }


def write_classified(src_path: str | Path, out_path: str | Path, cls: np.ndarray,
                     extra: dict | None = None, crop=None):
    """Copy the source cloud and overwrite classification (ASPRS codes)."""
    las = laspy.read(str(src_path))
    if crop is not None:
        x, y = np.asarray(las.x), np.asarray(las.y)
        m = (x >= crop[0]) & (y >= crop[1]) & (x <= crop[2]) & (y <= crop[3])
        las.points = las.points[m]
    las.classification = cls.astype(np.uint8)
    for name, arr in (extra or {}).items():
        if name not in las.point_format.dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=arr.dtype))
        las[name] = arr
    las.write(str(out_path))
