"""Building footprint network: U-Net with two output heads.

channel 0  building probability
channel 1  boundary probability (outlines + shared walls between buildings)

Instances = watershed of (building - boundary) seeds, see postprocess.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover
    smp = None


def build_model(arch: str = "Unet", encoder: str = "efficientnet-b4",
                encoder_weights: str | None = "imagenet", in_channels: int = 3) -> nn.Module:
    if smp is None:
        raise ImportError("pip install segmentation-models-pytorch")
    cls = getattr(smp, arch)
    kw = dict(encoder_name=encoder, encoder_weights=encoder_weights,
              in_channels=in_channels, classes=2)
    if arch == "Unet":
        kw["decoder_attention_type"] = "scse"
    return cls(**kw)


def dice_loss(logits, target, valid, eps=1.0):
    p = torch.sigmoid(logits) * valid
    t = target * valid
    num = 2 * (p * t).sum((1, 2)) + eps
    den = p.sum((1, 2)) + t.sum((1, 2)) + eps
    return 1 - (num / den).mean()


def focal_bce(logits, target, valid, gamma=2.0, pos_weight=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                             pos_weight=torch.as_tensor(pos_weight, device=logits.device))
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    loss = bce * (1 - pt) ** gamma
    return (loss * valid).sum() / valid.sum().clamp_min(1)


class FootprintLoss(nn.Module):
    """BCE+Dice on buildings, focal BCE on boundaries, masked by valid pixels."""

    def __init__(self, w_building=1.0, w_dice=1.0, w_boundary=1.0, boundary_pos_weight=3.0):
        super().__init__()
        self.w = (w_building, w_dice, w_boundary)
        self.bpw = boundary_pos_weight

    def forward(self, logits, target, valid):
        lb, le = logits[:, 0], logits[:, 1]
        tb, te = target[:, 0], target[:, 1]
        bce = F.binary_cross_entropy_with_logits(lb, tb, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1)
        d = dice_loss(lb, tb, valid)
        e = focal_bce(le, te, valid, pos_weight=self.bpw)
        total = self.w[0] * bce + self.w[1] * d + self.w[2] * e
        return total, {"bce": bce.item(), "dice": d.item(), "boundary": e.item()}


def save_checkpoint(path, model, cfg: dict, metrics: dict | None = None, epoch: int = 0):
    torch.save({"state_dict": model.state_dict(), "model_cfg": cfg.get("model", {}),
                "data_cfg": {k: cfg.get("data", {}).get(k) for k in ("gsd", "chip_px")},
                "metrics": metrics or {}, "epoch": epoch, "postprocess": cfg.get("postprocess", {})},
               path)


def load_checkpoint(path, device="cpu", strict=True):
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = dict(ck["model_cfg"])
    mc["encoder_weights"] = None
    model = build_model(**mc)
    model.load_state_dict(ck["state_dict"], strict=strict)
    model.to(device).eval()
    return model, ck
