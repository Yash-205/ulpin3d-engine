"""Synthetic aerial LiDAR of an Indian settlement, with exact labels.

Recreates what makes Indian LiDAR different from European training data:
attached RCC row-houses of mixed heights, roof parapets, black water tanks and
stair cabins ("mumty") on roofs, unfinished upper floors, boundary walls,
narrow lanes, parked vehicles and scattered trees (multi-return canopy).

Labels: 0 ground, 1 building, 2 vegetation, 3 other. Classification is written
to the LAS file using these codes, so evaluate with scheme="synthetic".
"""
from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np


def _rect_surface(rng, x0, y0, w, d, z, dens):
    n = max(1, int(w * d * dens))
    return np.c_[rng.uniform(x0, x0 + w, n), rng.uniform(y0, y0 + d, n), np.full(n, z)]


def _walls(rng, x0, y0, w, d, z0, z1, dens):
    pts = []
    per = 2 * (w + d)
    n = max(1, int(per * (z1 - z0) * dens * 0.25))  # walls are sparsely hit by nadir LiDAR
    t = rng.uniform(0, per, n)
    zz = rng.uniform(z0, z1, n)
    x = np.where(t < w, x0 + t, np.where(t < w + d, x0 + w, np.where(t < 2 * w + d, x0 + (2 * w + d - t), x0)))
    y = np.where(t < w, y0, np.where(t < w + d, y0 + (t - w), np.where(t < 2 * w + d, y0 + d, y0 + (per - t))))
    pts.append(np.c_[x, y, zz])
    return np.concatenate(pts)


def make_synthetic_lidar(out_laz: str | Path, size_m=120.0, density=20.0, seed=0, slope=(0.02, 0.01),
                         epsg=32643, origin=(500_000.0, 2_000_000.0)):
    rng = np.random.default_rng(seed)
    X0, Y0 = origin
    P, L, NR, RN = [], [], [], []

    def add(p, lab, nret=1, rnum=1):
        P.append(p)
        L.append(np.full(len(p), lab, np.uint8))
        NR.append(np.full(len(p), nret, np.uint8))
        RN.append(np.full(len(p), rnum, np.uint8))

    def ground_z(x, y):
        return 210 + slope[0] * (x - X0) + slope[1] * (y - Y0) + 0.3 * np.sin((x - X0) / 15)

    buildings = []
    y = 2.0
    while y < size_m - 10:
        depth = rng.uniform(8, 14)
        x = 2.0
        while x < size_m - 6:
            if rng.random() < 0.1:
                x += rng.uniform(4, 9)
                continue
            w = rng.uniform(4.5, 10)
            floors = rng.choice([1, 2, 3, 4, 5, 6], p=[0.12, 0.28, 0.28, 0.17, 0.1, 0.05])
            h = 3.1 * floors + 0.4 + rng.normal(0, 0.15)
            buildings.append((X0 + x, Y0 + y, w, depth * rng.uniform(0.8, 1.0), h, floors))
            x += w + (0.0 if rng.random() < 0.75 else rng.uniform(0.6, 1.5))
        y += depth + rng.uniform(2.5, 4.5)

    occupied = []
    for bx, by, w, d, h, fl in buildings:
        gz = float(ground_z(bx + w / 2, by + d / 2))
        roof = _rect_surface(rng, bx, by, w, d, 0, density)
        roof[:, 2] = gz + h + rng.normal(0, 0.03, len(roof))
        add(roof, 1)
        # parapet ring 1 m high, 0.2 m thick
        par = _walls(rng, bx, by, w, d, gz + h, gz + h + 1.0, density * 4)
        add(par, 1)
        add(_walls(rng, bx, by, w, d, gz, gz + h, density), 1)
        # water tank (cylinder) + stair cabin
        if rng.random() < 0.65:
            cx, cy = bx + rng.uniform(1, w - 1), by + rng.uniform(1, d - 1)
            n = int(3.14 * 0.6 ** 2 * density) + 5
            a, r = rng.uniform(0, 2 * np.pi, n), 0.6 * np.sqrt(rng.uniform(0, 1, n))
            add(np.c_[cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, gz + h + 1.3)], 1)
        if rng.random() < 0.5 and w > 5:
            sx, sy = bx + rng.uniform(0.5, w - 3), by + rng.uniform(0.5, d - 3)
            add(_rect_surface(rng, sx, sy, 2.5, 2.8, gz + h + 2.6, density), 1)
        occupied.append((bx, by, bx + w, by + d))

    occ = np.array(occupied)

    def free(x, y):
        inside = np.zeros(len(x), bool)
        for b in occ:
            inside |= (x >= b[0]) & (x <= b[2]) & (y >= b[1]) & (y <= b[3])
        return ~inside

    # ground (lanes, open plots)
    n = int(size_m * size_m * density)
    gx, gy = X0 + rng.uniform(0, size_m, n), Y0 + rng.uniform(0, size_m, n)
    m = free(gx, gy)
    add(np.c_[gx[m], gy[m], ground_z(gx[m], gy[m]) + rng.normal(0, 0.03, m.sum())], 0)
    # trees: canopy first returns + interior / ground last returns
    for _ in range(int(size_m * size_m / 500)):
        cx, cy = X0 + rng.uniform(0, size_m), Y0 + rng.uniform(0, size_m)
        if not free(np.array([cx]), np.array([cy]))[0]:
            continue
        r, top = rng.uniform(2, 4.5), rng.uniform(5, 12)
        k = int(3.14 * r * r * density * 1.4)
        a, rr = rng.uniform(0, 2 * np.pi, k), r * np.sqrt(rng.uniform(0, 1, k))
        x, y = cx + rr * np.cos(a), cy + rr * np.sin(a)
        gz = ground_z(x, y)
        canopy = gz + top - (rr / r) ** 2 * top * 0.4 + rng.normal(0, 0.4, k)
        add(np.c_[x, y, canopy], 2, 2, 1)
        inner = rng.random(k) < 0.35
        add(np.c_[x[inner], y[inner], gz[inner] + rng.uniform(1.5, top * 0.7, inner.sum())], 2, 3, 2)
    # vehicles and boundary walls (other)
    for _ in range(int(size_m / 4)):
        vx, vy = X0 + rng.uniform(0, size_m), Y0 + rng.uniform(0, size_m)
        if not free(np.array([vx]), np.array([vy]))[0]:
            continue
        p = _rect_surface(rng, vx, vy, 4.0, 1.8, 0, density)
        p[:, 2] = ground_z(p[:, 0], p[:, 1]) + 1.5 + rng.normal(0, 0.05, len(p))
        add(p, 3)
    for _ in range(int(size_m / 6)):
        wx, wy, L_ = X0 + rng.uniform(0, size_m), Y0 + rng.uniform(0, size_m), rng.uniform(5, 20)
        k = int(L_ * density * 0.8)
        x = wx + rng.uniform(0, L_, k)
        yv = np.full(k, wy) + rng.normal(0, 0.05, k)
        f = free(x, yv)
        add(np.c_[x[f], yv[f], ground_z(x[f], yv[f]) + rng.uniform(0.3, 1.8, f.sum())], 3)

    xyz = np.concatenate(P)
    lab = np.concatenate(L)
    nr, rn = np.concatenate(NR), np.concatenate(RN)
    xyz[:, :2] += rng.normal(0, 0.05, (len(xyz), 2))
    hdr = laspy.LasHeader(point_format=1, version="1.2")
    hdr.scales = [0.001, 0.001, 0.001]
    hdr.offsets = [X0, Y0, 0]
    try:
        from pyproj import CRS

        hdr.add_crs(CRS.from_epsg(epsg))
    except Exception:
        pass
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = lab
    las.number_of_returns = nr
    las.return_number = rn
    las.intensity = np.clip(rng.normal(np.choose(lab, [900, 1600, 600, 1200]), 150), 0, 65535).astype(np.uint16)
    Path(out_laz).parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_laz))
    meta = [dict(x=b[0], y=b[1], w=b[2], d=b[3], height=b[4], floors=int(b[5])) for b in buildings]
    return out_laz, meta
