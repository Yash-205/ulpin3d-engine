"""Score a checkpoint on held-out chips and write a report.

    python -m ulpin3d.cli eval-footprint --checkpoint runs/stage2_india/best.pt \
        --chips data/chips_india --split test --out reports/india_test

Produces report.json + per_scene.csv with pixel, boundary and object metrics,
broken down by region / country / scene. Also a confidence calibration table:
for each status bucket (validated / provisional / flagged), what fraction of
predicted buildings actually match a real one. This checks that "validated"
means what it says.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ulpin3d.footprint.dataset import ChipDataset, eval_augs
from ulpin3d.footprint.metrics import BoundaryMeter, ObjectMeter, PixelMeter, match_instances
from ulpin3d.footprint.model import load_checkpoint
from ulpin3d.footprint.postprocess import DEFAULTS as PP_DEFAULTS
from ulpin3d.footprint.postprocess import instances_from_probs
from ulpin3d.footprint.train import pick_device


def _status_of_instances(inst, pb, pe, gsd, pp):
    from scipy import ndimage as ndi

    ids = np.unique(inst)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return {}
    mb = ndi.mean(pb, inst, ids)
    me = ndi.mean(pe, inst, ids)
    n = ndi.sum(np.ones_like(pb), inst, ids)
    out = {}
    for i, a, b, c in zip(ids, mb, me, n):
        conf = a * (1 - 0.5 * b) * min(1.0, c * gsd * gsd / 25.0) ** 0.25
        out[int(i)] = ("validated" if conf >= pp["conf_validated"] else
                       "provisional" if conf >= pp["conf_provisional"] else "flagged")
    return out


@torch.no_grad()
def evaluate_checkpoint(checkpoint, chips_root, split="test", out_dir=None, device="auto",
                        batch_size=8, tta=True, filters: dict | None = None):
    dev = pick_device(device)
    model, ck = load_checkpoint(checkpoint, dev)
    pp = {**PP_DEFAULTS, **ck.get("postprocess", {})}
    root = Path(chips_root).resolve()
    idx = pd.read_csv(root / "index.csv")
    idx = idx[idx.split == split]
    for k, v in (filters or {}).items():
        idx = idx[idx[k].isin(v if isinstance(v, list) else [v])]
    if not len(idx):
        raise SystemExit(f"no chips with split={split}")
    ds = ChipDataset(root, idx, eval_augs(int(ck["data_cfg"].get("chip_px") or 512)), with_instances=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    groups = {}
    calib = {"validated": [0, 0], "provisional": [0, 0], "flagged": [0, 0]}

    def meters(key):
        if key not in groups:
            groups[key] = (PixelMeter(), BoundaryMeter(2), ObjectMeter(0.5))
        return groups[key]

    for b in dl:
        x = b["image"].to(dev)
        p = torch.sigmoid(model(x).float())
        if tta:
            p = (p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1)
                 + torch.sigmoid(model(x.flip(-2)).float()).flip(-2)) / 3
        p = p.cpu().numpy()
        tgt, val, ig = b["target"].numpy() > 0.5, b["valid"].numpy() > 0.5, b["inst"].numpy()
        for j, i in enumerate(b["idx"].tolist()):
            r = ds.df.iloc[i]
            pb, pe = p[j, 0], p[j, 1]
            pred = pb > pp["mask_thr"]
            inst = instances_from_probs(pb, pe, **pp, gsd=float(r.gsd))
            for key in ("ALL", f"region={r.region}", f"country={r.country}", f"scene={r.scene_id}"):
                pm, bm, om = meters(key)
                pm.update(pred, tgt[j, 0], val[j])
                bm.update(pred, tgt[j, 0], val[j])
                om.update(inst, ig[j], val[j])
            # calibration: which predicted instances matched a GT building?
            st = _status_of_instances(inst, pb, pe, float(r.gsd), pp)
            gt_ids = np.where(val[j], ig[j], 0)
            for pid, s in st.items():
                m = (inst == pid) & val[j]
                if not m.any():
                    continue
                g, cnt = np.unique(gt_ids[m], return_counts=True)
                best_iou = 0.0
                for gid, c in zip(g, cnt):
                    if gid == 0:
                        continue
                    best_iou = max(best_iou, c / (m.sum() + (gt_ids == gid).sum() - c))
                calib[s][0] += int(best_iou >= 0.5)
                calib[s][1] += 1

    rows = []
    for k, (pm, bm, om) in groups.items():
        rows.append({"group": k, **pm.compute(), **bm.compute(), **om.compute()})
    df = pd.DataFrame(rows).sort_values("group")
    cal = {s: {"n": n, "precision": round(tp / n, 4) if n else None} for s, (tp, n) in calib.items()}
    report = {"checkpoint": str(checkpoint), "split": split, "n_chips": int(len(idx)),
              "overall": df[df.group == "ALL"].iloc[0].to_dict(), "calibration": cal}
    if out_dir:
        o = Path(out_dir)
        o.mkdir(parents=True, exist_ok=True)
        df.to_csv(o / "per_group.csv", index=False)
        (o / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(df[["group", "iou", "f1", "boundary_f1", "obj_f1", "obj_precision", "obj_recall", "n_gt"]]
          .round(3).to_string(index=False))
    print("calibration:", json.dumps(cal))
    return report, df
