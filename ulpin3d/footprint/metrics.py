"""Evaluation metrics for building extraction.

pixel    IoU / F1 / precision / recall on the building mask (valid pixels only)
boundary boundary-F1 with a pixel tolerance (how well outlines line up)
object   instance matching at IoU >= 0.5: precision / recall / F1 per building
         - this is the number that matters for a cadastre (one ULPIN per building)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries


class PixelMeter:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def update(self, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
        p, g = pred & valid, gt & valid
        self.tp += int((p & g).sum())
        self.fp += int((p & ~g).sum())
        self.fn += int((~p & g).sum())

    def compute(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return {"iou": tp / max(tp + fp + fn, 1), "f1": 2 * prec * rec / max(prec + rec, 1e-9),
                "precision": prec, "recall": rec}


class BoundaryMeter:
    def __init__(self, tol_px: int = 2):
        self.tol = tol_px
        self.tp_p = self.n_p = self.tp_g = self.n_g = 0

    def update(self, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
        bp = find_boundaries(pred, mode="inner") & valid
        bg = find_boundaries(gt, mode="inner") & valid
        if not bp.any() and not bg.any():
            return
        dg = ndi.distance_transform_edt(~bg) if bg.any() else np.full(bg.shape, 1e9)
        dp = ndi.distance_transform_edt(~bp) if bp.any() else np.full(bp.shape, 1e9)
        self.tp_p += int((dg[bp] <= self.tol).sum())
        self.n_p += int(bp.sum())
        self.tp_g += int((dp[bg] <= self.tol).sum())
        self.n_g += int(bg.sum())

    def compute(self):
        p = self.tp_p / max(self.n_p, 1)
        r = self.tp_g / max(self.n_g, 1)
        return {"boundary_f1": 2 * p * r / max(p + r, 1e-9), "boundary_precision": p, "boundary_recall": r}


def match_instances(pred_inst: np.ndarray, gt_inst: np.ndarray, valid: np.ndarray, iou_thr=0.5):
    """Greedy one-to-one matching. Returns (tp, fp, fn, list of matched IoUs)."""
    pred_inst = np.where(valid, pred_inst, 0)
    gt_inst = np.where(valid, gt_inst, 0)
    pids = np.unique(pred_inst)
    pids = pids[pids > 0]
    gids = np.unique(gt_inst)
    gids = gids[gids > 0]
    if len(pids) == 0 or len(gids) == 0:
        return 0, len(pids), len(gids), []
    both = (pred_inst > 0) & (gt_inst > 0)
    pairs = defaultdict(int)
    pk = pred_inst[both].astype(np.int64)
    gk = gt_inst[both].astype(np.int64)
    keys, cnt = np.unique(pk * (2 ** 31) + gk, return_counts=True)
    parea = dict(zip(*np.unique(pred_inst[pred_inst > 0], return_counts=True)))
    garea = dict(zip(*np.unique(gt_inst[gt_inst > 0], return_counts=True)))
    cands = []
    for k, c in zip(keys, cnt):
        p, g = int(k // (2 ** 31)), int(k % (2 ** 31))
        iou = c / (parea[p] + garea[g] - c)
        if iou >= iou_thr:
            cands.append((iou, p, g))
    cands.sort(reverse=True)
    used_p, used_g, ious = set(), set(), []
    for iou, p, g in cands:
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        ious.append(iou)
    tp = len(ious)
    return tp, len(pids) - tp, len(gids) - tp, ious


class ObjectMeter:
    def __init__(self, iou_thr=0.5):
        self.thr = iou_thr
        self.tp = self.fp = self.fn = 0
        self.ious = []

    def update(self, pred_inst, gt_inst, valid):
        tp, fp, fn, ious = match_instances(pred_inst, gt_inst, valid, self.thr)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.ious += ious

    def compute(self):
        p = self.tp / max(self.tp + self.fp, 1)
        r = self.tp / max(self.tp + self.fn, 1)
        return {"obj_precision": p, "obj_recall": r, "obj_f1": 2 * p * r / max(p + r, 1e-9),
                "obj_mean_iou": float(np.mean(self.ious)) if self.ious else 0.0,
                "n_pred": self.tp + self.fp, "n_gt": self.tp + self.fn}
