"""Ground filtering and height-above-ground (HAG).

A progressive morphological filter (Zhang et al. 2003, the idea behind PDAL's
SMRF) on a minimum-elevation grid. Openings grow from 1 m up to the largest
building we expect (default 40 m, enough for Indian industrial sheds and
apartment blocks). Cells that rise above the opened surface by more than a
slope-dependent threshold are non-ground. The DTM is interpolated from ground
cells, and every point gets HAG = z - DTM(x, y).

Pure numpy/scipy, so it runs anywhere. If PDAL is installed, `use_pdal=True`
swaps in filters.smrf.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import griddata


def min_grid(xyz: np.ndarray, cell: float):
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    ix = ((xyz[:, 0] - x0) / cell).astype(np.int64)
    iy = ((xyz[:, 1] - y0) / cell).astype(np.int64)
    W, H = ix.max() + 1, iy.max() + 1
    g = np.full(H * W, np.inf)
    np.minimum.at(g, iy * W + ix, xyz[:, 2])
    return g.reshape(H, W), (x0, y0), ix, iy


def _fill_nearest(a: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    if not invalid.any():
        return a
    idx = ndi.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def progressive_morphological_dtm(xyz: np.ndarray, cell: float = 1.0, max_window_m: float = 40.0,
                                  slope: float = 0.25, dh0: float = 0.3, dh_max: float = 3.0):
    g, origin, ix, iy = min_grid(xyz, cell)
    empty = ~np.isfinite(g)
    # remove low outliers (multipath / noise below ground)
    g = _fill_nearest(g, empty)
    med = ndi.median_filter(g, size=5)
    low = g < med - 2.0
    g = np.where(low, med, g)
    surf = g.copy()
    ground = np.ones_like(g, bool)
    w, prev = 1, 0
    while w * cell <= max_window_m:
        size = 2 * w + 1
        opened = ndi.grey_opening(surf, size=(size, size))
        dh = min(dh0 + slope * (size - (2 * prev + 1)) * cell, dh_max) if prev else dh0
        ground &= (g - opened) <= dh
        surf = opened
        prev, w = w, w * 2
    ground &= ~empty
    yy, xx = np.nonzero(ground)
    if len(yy) < 10:
        dtm = ndi.grey_opening(g, size=(15, 15))
    else:
        H, W = g.shape
        gy, gx = np.mgrid[0:H, 0:W]
        step = max(1, len(yy) // 400_000)
        dtm = griddata((yy[::step], xx[::step]), g[yy, xx][::step], (gy, gx), method="linear")
        nan = ~np.isfinite(dtm)
        dtm = _fill_nearest(dtm, nan)
    dtm = ndi.gaussian_filter(dtm, 1.0)
    return dtm, origin, cell, ix, iy


def height_above_ground(xyz: np.ndarray, cell: float = 1.0, use_pdal: bool = False, **kw):
    """Return (hag per point, dtm grid, origin, cell)."""
    if use_pdal:
        try:
            return _hag_pdal(xyz)
        except Exception:
            pass
    dtm, origin, cell, ix, iy = progressive_morphological_dtm(xyz, cell, **kw)
    # bilinear sample of the DTM at each point
    fx = (xyz[:, 0] - origin[0]) / cell - 0.5
    fy = (xyz[:, 1] - origin[1]) / cell - 0.5
    z = ndi.map_coordinates(dtm, [fy, fx], order=1, mode="nearest")
    return (xyz[:, 2] - z).astype(np.float32), dtm, origin, cell


def _hag_pdal(xyz):  # pragma: no cover - optional dependency
    import json

    import pdal

    arr = np.zeros(len(xyz), dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")])
    arr["X"], arr["Y"], arr["Z"] = xyz.T
    p = pdal.Pipeline(json.dumps({"pipeline": [{"type": "filters.smrf"}, {"type": "filters.hag_nn"}]}), arrays=[arr])
    p.execute()
    out = p.arrays[0]
    return out["HeightAboveGround"].astype(np.float32), None, None, None
