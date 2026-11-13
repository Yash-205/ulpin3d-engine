"""Score the plan parser against synthetic ground truth (any input format)."""
from __future__ import annotations

import json

import numpy as np
from shapely.geometry import shape

from ulpin3d.floorplan.georef import apply_fit, fit_plan_to_footprint
from ulpin3d.floorplan.parse import parse_plan


def score_plan(path, gt_path, allow_scale=False):
    gt = json.loads(open(gt_path).read())
    r = parse_plan(path)
    plate = shape(gt["plate"])
    fit = fit_plan_to_footprint(r["outline"], plate, allow_scale=allow_scale)
    gtu = {u["label"]: shape(u["polygon"]) for u in gt["units"]}
    rows, matched = [], 0
    for u in r["units"]:
        g = apply_fit(u["polygon"], fit)
        t = gtu.get(u["label"])
        iou = g.intersection(t).area / g.union(t).area if t is not None else 0.0
        rows.append({"label": u["label"], "iou": iou, "area_err_pct": 100 * (u["area_m2"] - t.area) / t.area if t is not None else None})
        matched += iou >= 0.8
    n_gt, n_pred = len(gtu), len(r["units"])
    p, rc = matched / max(n_pred, 1), matched / max(n_gt, 1)
    errs = [abs(x["area_err_pct"]) for x in rows if x["area_err_pct"] is not None]
    return {"file": str(path), "n_gt": n_gt, "n_pred": n_pred, "unit_precision": p, "unit_recall": rc,
            "unit_f1": 2 * p * rc / max(p + rc, 1e-9), "mean_iou": float(np.mean([x["iou"] for x in rows])) if rows else 0,
            "area_err_pct_mean": float(np.mean(errs)) if errs else None,
            "floors_ok": r["floors"] == list(range(gt["floors"][0], gt["floors"][1] + 1)),
            "georef_rms_m": fit["rms_m"], "georef_iou": fit["iou"], "scale": fit["scale"], "units": rows}
