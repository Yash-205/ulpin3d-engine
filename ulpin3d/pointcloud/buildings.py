"""Buildings from classified LiDAR: footprints, heights, roof levels, floor estimate.

Two ways to get building outlines:
  * from LiDAR alone  - rasterise building points (0.5 m), split attached blocks
                        where the roof height jumps (watershed on the nDSM)
  * fused (preferred) - take Model 1 polygons from the orthophoto and attach LiDAR
                        heights to each one. The photo gives sharp edges and
                        party walls; LiDAR gives true heights.

Per building we report ground elevation, roof height percentiles, distinct roof
levels (a stair cabin or a penthouse floor shows up as a second level), and an
estimated floor count with an uncertainty band. That estimate is a prior: the
sanctioned plan's floor count is checked against it in the validation stage.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from shapely.geometry import shape
from skimage.segmentation import watershed

BUILDING = 1


def ndsm_raster(xyz, hag, mask, res=0.5, bounds=None):
    x, y = xyz[mask, 0], xyz[mask, 1]
    h = hag[mask]
    if bounds is None:
        bounds = (xyz[:, 0].min(), xyz[:, 1].min(), xyz[:, 0].max(), xyz[:, 1].max())
    W = int(np.ceil((bounds[2] - bounds[0]) / res)) + 1
    H = int(np.ceil((bounds[3] - bounds[1]) / res)) + 1
    ix = ((x - bounds[0]) / res).astype(int)
    iy = ((bounds[3] - y) / res).astype(int)
    g = np.full(H * W, -np.inf, np.float32)
    np.maximum.at(g, iy * W + ix, h)
    g = g.reshape(H, W)
    occ = np.isfinite(g)
    g[~occ] = 0
    return g, occ, from_origin(bounds[0], bounds[3], res, res), bounds


def roof_levels(heights: np.ndarray, min_share=0.12, bin_m=0.5):
    """Distinct roof levels via a height histogram (e.g. main roof + stair cabin)."""
    if len(heights) < 20:
        return [float(np.median(heights))] if len(heights) else []
    b = np.arange(heights.min(), heights.max() + bin_m, bin_m)
    if len(b) < 3:
        return [float(np.median(heights))]
    hist, edges = np.histogram(heights, b)
    hist = ndi.uniform_filter1d(hist.astype(float), 3)
    peaks = [i for i in range(len(hist)) if hist[i] >= min_share * hist.sum()
             and hist[i] == hist[max(0, i - 2): i + 3].max()]
    return [round(float((edges[i] + edges[i + 1]) / 2), 2) for i in peaks] or [float(np.median(heights))]


def main_roof_height(h: np.ndarray, bin_m: float = 0.25) -> float:
    """Height of the dominant roof plane: the densest height bin among the upper points.

    Wall hits (low), parapets, tanks and stair cabins (above) are minorities, so
    neither the median nor a high percentile is reliable; the mode of the upper
    part of the distribution is."""
    top = np.percentile(h, 95)
    hh = h[h > 0.55 * top]
    if len(hh) < 5:
        return float(np.median(h))
    b = np.arange(hh.min(), hh.max() + bin_m, bin_m)
    if len(b) < 3:
        return float(np.median(hh))
    hist, edges = np.histogram(hh, b)
    i = int(np.argmax(ndi.uniform_filter1d(hist.astype(float), 3)))
    return float((edges[i] + edges[i + 1]) / 2)


def estimate_floors(roof_h: float, floor_h: float = 3.1, parapet: float = 1.0, ground_floor_extra: float = 0.4):
    """Indian RCC buildings: ~3.0-3.3 m floor-to-floor, ~1 m parapet, taller ground floor.

    Returns (floors, low, high); low/high use 2.8 m and 3.6 m floor heights."""
    eff = max(roof_h - parapet * (roof_h > 4) - ground_floor_extra, 0)
    f = max(1, int(round(eff / floor_h)))
    lo = max(1, int(np.floor(eff / 3.6 + 0.5)))
    hi = max(1, int(np.floor(eff / 2.8 + 0.5)))
    return f, lo, hi


def _merge_rooftop_segments(inst, nd, min_area_px):
    """Merge rooftop structures (stair cabin / mumty, lift machine room, tank stand) into
    the roof they stand on. A segment is merged only if it is small, *higher* than its
    largest neighbour and shares a real border with it, so a small lower house built
    against a taller one stays a separate building."""
    changed = True
    while changed:
        changed = False
        ids, counts = np.unique(inst[inst > 0], return_counts=True)
        for k, c in sorted(zip(ids, counts), key=lambda t: t[1]):
            if c >= min_area_px:
                break
            m = inst == k
            ring = ndi.binary_dilation(m, iterations=1) & ~m
            nb = inst[ring]
            nb = nb[(nb > 0) & (nb != k)]
            if nb.size == 0:
                continue
            vals, cnt = np.unique(nb, return_counts=True)
            j = vals[cnt.argmax()]
            if cnt.max() < 0.2 * ring.sum():
                continue
            if np.median(nd[m]) <= np.median(nd[inst == j]) + 1.0:
                continue  # not standing on top of the neighbour
            inst[m] = j
            changed = True
    return inst


def buildings_from_lidar(xyz, hag, labels, res=0.5, min_area_m2=8.0, split_jump_m=2.5, crs=None, merge_below_m2=60.0):
    m = labels == BUILDING
    if m.sum() < 10:
        return gpd.GeoDataFrame({"geometry": []}, crs=crs)
    nd, occ, tr, bounds = ndsm_raster(xyz, hag, m, res)
    occ = ndi.binary_closing(occ, iterations=2)
    occ = ndi.binary_fill_holes(occ)
    nd = ndi.grey_closing(nd, size=3)
    # split attached blocks where roof height jumps by more than split_jump_m
    grad = ndi.morphological_gradient(nd, size=3)
    seeds_mask = occ & (grad < split_jump_m)
    seeds_mask = ndi.binary_erosion(seeds_mask, iterations=2)
    markers, _ = ndi.label(seeds_mask)
    inst = watershed(grad, markers, mask=occ)
    lone = occ & (inst == 0)
    if lone.any():
        lab, _ = ndi.label(lone)
        inst = np.where(lab > 0, lab + inst.max(), inst)
    inst = _merge_rooftop_segments(inst, nd, min_area_px=int(merge_below_m2 / (res * res)))
    polys = []
    for geom, v in shapes(inst.astype(np.int32), mask=inst > 0, transform=tr):
        p = shape(geom).buffer(0)
        if p.area >= min_area_m2:
            # cell-binned outlines run half a cell wide on every side: take that back
            polys.append(p.buffer(-res / 2, join_style=2).simplify(res * 0.8))
    g = gpd.GeoDataFrame({"geometry": polys}, crs=crs)
    return attach_heights(g, xyz, hag, labels)


def attach_heights(footprints: gpd.GeoDataFrame, xyz, hag, labels, dtm_z=None, floor_h=3.1):
    """Attach LiDAR heights to polygons (e.g. Model 1 output). Adds columns in place."""
    from shapely import STRtree, points

    if len(footprints) == 0:
        return footprints
    bmask = labels == BUILDING
    bx = xyz[bmask]
    bh = hag[bmask]
    ground_z = (xyz[:, 2] - hag)[bmask]
    tree = STRtree(footprints.geometry.values)
    pts = points(bx[:, 0], bx[:, 1])
    pi, gi = tree.query(pts, predicate="within")
    out = {k: [] for k in ("n_points", "ground_z", "roof_h_main", "roof_h_p95", "roof_levels",
                           "floors_est", "floors_low", "floors_high", "height_conf", "lidar_cover")}
    order = np.argsort(gi)
    gi_s, pi_s = gi[order], pi[order]
    starts = np.searchsorted(gi_s, np.arange(len(footprints)))
    ends = np.searchsorted(gi_s, np.arange(len(footprints)), side="right")
    for k in range(len(footprints)):
        ix = pi_s[starts[k]:ends[k]]
        area = footprints.geometry.iloc[k].area
        if len(ix) < 5:
            for c in out:
                out[c].append(None)
            continue
        h = bh[ix]
        p50, p95 = float(np.percentile(h, 50)), float(np.percentile(h, 95))
        f, lo, hi = estimate_floors(main_roof_height(h), floor_h, parapet=0.0)
        density = len(ix) / max(area, 1)
        cover = min(1.0, density / 4.0)  # ~4 pts/m2 on roofs is plenty for a height
        main = main_roof_height(h)
        upper = h[h > main - 2.0]  # roof, parapet and rooftop structures; wall hits excluded
        on_plane = float(np.mean(np.abs(upper - main) < 0.6)) if len(upper) else 0.0
        conf = cover * (1.0 if on_plane >= 0.5 else 0.8)
        for c, v in zip(out, (len(ix), float(np.median(ground_z[ix])), round(main_roof_height(h), 2), round(p95, 2),
                               roof_levels(h), f, lo, hi, round(conf, 3), round(cover, 3))):
            out[c].append(v)
    for c, v in out.items():
        footprints[c] = v
    footprints["roof_levels"] = footprints["roof_levels"].map(lambda v: ",".join(map(str, v)) if isinstance(v, list) else v)
    return footprints
