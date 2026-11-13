"""Per-point features for LiDAR classification.

Geometry carries over between countries much better than colour does: a flat
RCC roof in Delhi and a flat roof in Sao Paulo have the same eigenvalues. So
the core features are geometric, computed at several neighbourhood scales on a
voxel-thinned cloud:

  eigen     linearity, planarity, scattering, omnivariance, anisotropy,
            eigentropy, change of curvature, verticality, |normal_z|
  local z   std, range, z - local min, local max - z
  column    2D grid (1 m): height range, point count, multi-return share
  height    height above ground (HAG) from our own ground filter
  sensor    intensity (normalised per tile), return number / count,
            RGB + excess-green (NaN when the sensor did not record them)

The point attributes are optional on purpose. During training we randomly
blank them for some tiles, so the model still works on clouds without RGB or
intensity.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

FEATURE_NAMES: list[str] = []


def voxel_downsample(xyz: np.ndarray, voxel: float):
    """Return (indices of one representative point per voxel, inverse map point->voxel row)."""
    if voxel <= 0:
        idx = np.arange(len(xyz))
        return idx, idx
    k = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
    key = (k[:, 0] * 73856093) ^ (k[:, 1] * 19349663) ^ (k[:, 2] * 83492791)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return first, inv


def _eigen_feats(nbr_xyz: np.ndarray):
    """nbr_xyz: (N, k, 3) -> dict of (N,) arrays."""
    c = nbr_xyz - nbr_xyz.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / nbr_xyz.shape[1]
    w, v = np.linalg.eigh(cov)  # ascending
    w = np.clip(w, 1e-12, None)
    l3, l2, l1 = w[:, 0], w[:, 1], w[:, 2]
    s = l1 + l2 + l3
    e1, e2, e3 = l1 / s, l2 / s, l3 / s
    normal = v[:, :, 0]
    nz = np.abs(normal[:, 2])
    return {
        "linearity": (l1 - l2) / l1,
        "planarity": (l2 - l3) / l1,
        "scattering": l3 / l1,
        "omnivariance": np.cbrt(e1 * e2 * e3),
        "anisotropy": (l1 - l3) / l1,
        "eigentropy": -(e1 * np.log(e1) + e2 * np.log(e2) + e3 * np.log(e3)),
        "curvature": e3,
        "verticality": 1 - nz,
        "normal_z": nz,
    }


def neighbourhood_features(xyz: np.ndarray, ks=(10, 25, 50), chunk: int = 150_000):
    tree = cKDTree(xyz)
    out = {}
    kmax = max(ks)
    for s in range(0, len(xyz), chunk):
        e = min(len(xyz), s + chunk)
        _, nn = tree.query(xyz[s:e], k=min(kmax, len(xyz)))
        for k in ks:
            nb = xyz[nn[:, :k]]
            f = _eigen_feats(nb)
            z = nb[:, :, 2]
            f["z_std"] = z.std(1)
            f["z_range"] = z.max(1) - z.min(1)
            f["z_above_min"] = xyz[s:e, 2] - z.min(1)
            f["z_below_max"] = z.max(1) - xyz[s:e, 2]
            f["radius"] = np.linalg.norm(nb[:, -1] - xyz[s:e], axis=1)  # local density proxy
            for name, val in f.items():
                out.setdefault(f"{name}_k{k}", []).append(val.astype(np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


def column_features(xyz: np.ndarray, multi: np.ndarray | None, cell: float = 1.0):
    ix = np.floor((xyz[:, 0] - xyz[:, 0].min()) / cell).astype(np.int64)
    iy = np.floor((xyz[:, 1] - xyz[:, 1].min()) / cell).astype(np.int64)
    key = ix * (iy.max() + 1) + iy
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    zmax = np.full(len(u), -np.inf)
    zmin = np.full(len(u), np.inf)
    np.maximum.at(zmax, inv, xyz[:, 2])
    np.minimum.at(zmin, inv, xyz[:, 2])
    f = {
        "col_range": (zmax - zmin)[inv].astype(np.float32),
        "col_count": cnt[inv].astype(np.float32),
        "col_top_gap": (zmax[inv] - xyz[:, 2]).astype(np.float32),
    }
    if multi is not None:
        mr = np.bincount(inv, weights=multi.astype(np.float64), minlength=len(u)) / cnt
        f["col_multi_return"] = mr[inv].astype(np.float32)
    else:
        f["col_multi_return"] = np.full(len(xyz), np.nan, np.float32)
    return f


def point_features(pts: dict, hag: np.ndarray, sel: np.ndarray, drop_rgb=False, drop_intensity=False,
                   drop_returns=False):
    n = len(sel)
    nan = np.full(n, np.nan, np.float32)
    f = {"hag": hag[sel].astype(np.float32)}
    inten = pts.get("intensity")
    if inten is not None and not drop_intensity:
        lo, hi = np.percentile(inten, [1, 99])
        f["intensity"] = np.clip((inten[sel] - lo) / max(hi - lo, 1e-6), 0, 1.5).astype(np.float32)
    else:
        f["intensity"] = nan
    rn, nr = pts.get("return_number"), pts.get("number_of_returns")
    if rn is not None and nr is not None and not drop_returns:
        f["return_number"] = rn[sel]
        f["number_of_returns"] = nr[sel]
        f["is_last_return"] = (rn[sel] == nr[sel]).astype(np.float32)
    else:
        f["return_number"] = f["number_of_returns"] = f["is_last_return"] = nan
    rgb = pts.get("rgb")
    if rgb is not None and not drop_rgb:
        c = rgb[sel]
        s = c.sum(1) + 1e-6
        f["r"], f["g"], f["b"] = c[:, 0], c[:, 1], c[:, 2]
        f["exg"] = ((2 * c[:, 1] - c[:, 0] - c[:, 2]) / s).astype(np.float32)
        f["brightness"] = (s / 3).astype(np.float32)
    else:
        f["r"] = f["g"] = f["b"] = f["exg"] = f["brightness"] = nan
    return f


def compute_features(pts: dict, hag: np.ndarray, voxel: float = 0.3, ks=(10, 25, 50),
                     drop_rgb=False, drop_intensity=False, drop_returns=False):
    """Features for one representative point per voxel.

    Returns (X float32 (M, F), names, sel indices into pts (M,), inverse (N,) point->row).
    """
    xyz = pts["xyz"]
    sel, inv = voxel_downsample(xyz, voxel)
    sub = xyz[sel]
    f = {}
    f.update(point_features(pts, hag, sel, drop_rgb, drop_intensity, drop_returns))
    multi = None
    if pts.get("number_of_returns") is not None and not drop_returns:
        multi = (pts["number_of_returns"][sel] > 1)
    f.update(column_features(sub, multi))
    f.update(neighbourhood_features(sub, ks))
    names = list(f.keys())
    X = np.stack([f[k] for k in names], 1).astype(np.float32)
    return X, names, sel, inv
