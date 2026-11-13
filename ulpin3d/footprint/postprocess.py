"""Probability maps -> individual building polygons with a confidence score.

1. seeds     = building_prob - boundary_prob > seed_thr   (cores of buildings)
2. instances = watershed on -building_prob from the seeds, inside building_prob > mask_thr
3. polygons  = vectorise each instance, simplify, square-off near-rectangular shapes
4. confidence per building = mean building prob inside it, discounted by how much
   of it is boundary-like (ambiguous splits) and how small it is.

The confidence is what the validation engine thresholds later
(validated / provisional / flagged).
"""
from __future__ import annotations

import numpy as np
from rasterio.features import shapes
from scipy import ndimage as ndi
from shapely.geometry import shape
from skimage.segmentation import watershed

DEFAULTS = dict(mask_thr=0.5, seed_thr=0.55, min_seed_px=12, min_area_m2=6.0,
                simplify_px=1.2, rect_snap=0.85, conf_validated=0.85, conf_provisional=0.60)


def instances_from_probs(pb: np.ndarray, pe: np.ndarray, mask_thr=0.5, seed_thr=0.55,
                         min_seed_px=12, min_area_m2=6.0, gsd=None, **_) -> np.ndarray:
    """Label map of buildings. With `gsd` (metres per pixel), blobs under `min_area_m2` are removed,
    the same rule polygons_from_instances applies, so scoring counts what the pipeline would emit."""
    mask = pb > mask_thr
    seeds = (pb - pe) > seed_thr
    seeds = ndi.binary_opening(seeds, iterations=1)
    markers, n = ndi.label(seeds)
    if n:
        sizes = ndi.sum(np.ones_like(markers), markers, index=np.arange(1, n + 1))
        small = np.where(sizes < min_seed_px)[0] + 1
        markers[np.isin(markers, small)] = 0
    inst = watershed(-pb, markers, mask=mask)
    # building pixels that got no seed (very small buildings): own components
    orphan = mask & (inst == 0)
    if orphan.any():
        lab, k = ndi.label(orphan)
        inst = np.where(lab > 0, lab + inst.max(), inst)
    if gsd and min_area_m2 > 0:
        npx = np.bincount(inst.ravel())
        small = np.flatnonzero(npx < min_area_m2 / (gsd * gsd))
        small = small[small > 0]
        if len(small):
            inst = np.where(np.isin(inst, small), 0, inst)
    return inst.astype(np.int32)


def _square_off(poly, rect_snap):
    mrr = poly.minimum_rotated_rectangle
    if mrr.area > 0 and poly.area / mrr.area >= rect_snap:
        return mrr
    return poly


def polygons_from_instances(inst, pb, pe, transform, gsd, min_area_m2=6.0, simplify_px=1.2,
                            rect_snap=0.85, conf_validated=0.85, conf_provisional=0.60, **_):
    """Return list of dicts {geometry, area_m2, confidence, status} in raster CRS."""
    out = []
    if inst.max() == 0:
        return out
    ids = np.unique(inst)
    ids = ids[ids > 0]
    mean_pb = ndi.mean(pb, inst, ids)
    mean_pe = ndi.mean(pe, inst, ids)
    npx = ndi.sum(np.ones_like(pb), inst, ids)
    stats = {int(i): (float(a), float(b), float(c)) for i, a, b, c in zip(ids, mean_pb, mean_pe, npx)}
    min_px = min_area_m2 / (gsd * gsd)
    for geom, val in shapes(inst.astype(np.int32), mask=inst > 0, transform=transform):
        i = int(val)
        mb, me, n = stats[i]
        if n < min_px:
            continue
        poly = shape(geom).buffer(0)
        if poly.is_empty:
            continue
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        poly = poly.simplify(simplify_px * gsd, preserve_topology=True)
        poly = _square_off(poly, rect_snap)
        size_factor = min(1.0, (n * gsd * gsd) / 25.0) ** 0.25  # < 25 m2 slightly penalised
        conf = float(np.clip(mb * (1 - 0.5 * me) * size_factor, 0, 1))
        status = ("validated" if conf >= conf_validated else
                  "provisional" if conf >= conf_provisional else "flagged")
        out.append(dict(geometry=poly, area_m2=round(poly.area, 2), confidence=round(conf, 4),
                        status=status, mean_prob=round(mb, 4)))
    return out
