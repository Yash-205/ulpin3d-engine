"""Train / fine-tune the building footprint model from a YAML config.

    python -m ulpin3d.cli train-footprint --config configs/footprint_stage1_pretrain.yaml
    python -m ulpin3d.cli train-footprint --config configs/footprint_stage2_india.yaml

Stage 2 sets `train.init_checkpoint` to the stage-1 best checkpoint, uses a lower LR
and (optionally) freezes the encoder for the first epochs, so the network adapts to
Indian roofs, lanes and construction without forgetting what it learned.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from ulpin3d.footprint.dataset import ChipDataset, balanced_sampler, eval_augs, train_augs
from ulpin3d.footprint.metrics import BoundaryMeter, ObjectMeter, PixelMeter
from ulpin3d.footprint.model import FootprintLoss, build_model, load_checkpoint, save_checkpoint
from ulpin3d.footprint.postprocess import DEFAULTS as PP_DEFAULTS
from ulpin3d.footprint.postprocess import instances_from_probs


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def filter_index(df: pd.DataFrame, flt: dict | None) -> pd.DataFrame:
    for k, v in (flt or {}).items():
        v = v if isinstance(v, (list, tuple)) else [v]
        df = df[df[k].isin(v)]
    return df


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def evaluate(model, loader, device, pp: dict, amp=False, max_batches=None, gsd=None):
    model.eval()
    pix, bnd, obj = PixelMeter(), BoundaryMeter(2), ObjectMeter(0.5)
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = b["image"].to(device)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        tgt = b["target"].numpy() > 0.5
        val = b["valid"].numpy() > 0.5
        inst_gt = b["inst"].numpy() if "inst" in b else None
        for j in range(prob.shape[0]):
            pb, pe = prob[j, 0], prob[j, 1]
            pred = pb > pp["mask_thr"]
            pix.update(pred, tgt[j, 0], val[j])
            bnd.update(pred, tgt[j, 0], val[j])
            if inst_gt is not None:
                obj.update(instances_from_probs(pb, pe, **pp, gsd=gsd), inst_gt[j], val[j])
    out = {**pix.compute(), **bnd.compute()}
    if obj.tp + obj.fp + obj.fn:
        out.update(obj.compute())
    return out


def train(cfg: dict) -> Path:
    tc, dc = cfg["train"], cfg["data"]
    seed_all(tc.get("seed", 0))
    device = pick_device(tc.get("device", "auto"))
    out_dir = Path(tc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    pp = {**PP_DEFAULTS, **cfg.get("postprocess", {})}

    roots = dc["chips_root"] if isinstance(dc["chips_root"], list) else [dc["chips_root"]]
    frames = []
    for r in roots:
        r = Path(r).resolve()
        d = pd.read_csv(r / "index.csv")
        d["image"] = [str(Path(r) / p) for p in d.image]
        d["mask"] = [str(Path(r) / p) for p in d["mask"]]
        d["inst"] = [str(Path(r) / p) for p in d.inst]
        frames.append(d)
    idx = pd.concat(frames, ignore_index=True)
    tr = filter_index(idx, dc.get("train_filter"))
    va = filter_index(idx, dc.get("val_filter"))
    if dc.get("max_val_chips") and len(va) > dc["max_val_chips"]:
        va = va.sample(dc["max_val_chips"], random_state=0)
    print(f"[train] device={device} train={len(tr)} val={len(va)}")
    print(tr.groupby(["region", "country"]).size().to_string())
    if len(tr) == 0:
        raise SystemExit("no training chips after filtering")

    crop = dc.get("crop_px", dc.get("chip_px", 512))
    ds_tr = ChipDataset("/", tr, train_augs(crop, dc.get("aug_strength", "strong")))
    ds_va = ChipDataset("/", va, eval_augs(dc.get("chip_px", 512)), with_instances=True)
    steps = tc.get("steps_per_epoch") or math.ceil(len(tr) / tc["batch_size"])
    sampler = balanced_sampler(tr, dc.get("sample_by", "region"), dc.get("sample_weights"),
                               num_samples=steps * tc["batch_size"])
    nw = tc.get("num_workers", 4)
    dl_tr = DataLoader(ds_tr, batch_size=tc["batch_size"], sampler=sampler, num_workers=nw,
                       pin_memory=device.type == "cuda", drop_last=True, persistent_workers=nw > 0)
    dl_va = DataLoader(ds_va, batch_size=tc.get("val_batch_size", tc["batch_size"]), shuffle=False,
                       num_workers=nw)

    if tc.get("init_checkpoint"):
        model, ck = load_checkpoint(tc["init_checkpoint"], device)
        cfg["model"] = ck["model_cfg"]
        print(f"[train] fine-tuning from {tc['init_checkpoint']} (epoch {ck.get('epoch')}, {ck.get('metrics')})")
    else:
        model = build_model(**cfg["model"]).to(device)
    model.train()

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    other = [p for p in model.parameters() if id(p) not in enc_ids]
    lr = float(tc["lr"])
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": lr * tc.get("encoder_lr_mult", 1.0)},
        {"params": other, "lr": lr},
    ], weight_decay=float(tc.get("weight_decay", 1e-4)))
    total = tc["epochs"] * steps
    warm = int(tc.get("warmup_frac", 0.03) * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))) * 0.98 + 0.02)
    use_amp = tc.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = FootprintLoss(**cfg.get("loss", {}))

    best, best_key, patience = -1.0, tc.get("select_metric", "obj_f1"), 0
    hist = []
    freeze_n = tc.get("freeze_encoder_epochs", 0)
    for ep in range(1, tc["epochs"] + 1):
        for p in enc_params:
            p.requires_grad_(ep > freeze_n)
        model.train()
        t0, run = time.time(), []
        for b in dl_tr:
            x, y, v = b["image"].to(device), b["target"].to(device), b["valid"].to(device)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(x)
            loss, parts = crit(logits.float(), y, v)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(loss.item())
        m = evaluate(model, dl_va, device, pp, use_amp, gsd=float(dc.get("gsd") or 0.15)) if len(va) else {}
        rec = {"epoch": ep, "loss": float(np.mean(run)), "time_s": round(time.time() - t0, 1),
               "lr": opt.param_groups[1]["lr"], **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}
        hist.append(rec)
        pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
        print("[train] " + json.dumps(rec), flush=True)
        save_checkpoint(out_dir / "last.pt", model, cfg, m, ep)
        score = m.get(best_key, m.get("iou", -rec["loss"]))
        if score > best:
            best, patience = score, 0
            save_checkpoint(out_dir / "best.pt", model, cfg, m, ep)
        else:
            patience += 1
            if patience >= tc.get("early_stop_patience", 1_000):
                print(f"[train] early stop at epoch {ep}")
                break
    (out_dir / "best_metrics.json").write_text(json.dumps({"best_" + best_key: best, "history": hist}, indent=2))
    return out_dir / "best.pt"
