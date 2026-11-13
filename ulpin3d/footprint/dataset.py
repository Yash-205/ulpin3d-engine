"""PyTorch dataset over the chip index, with augmentations aimed at domain shift.

Drone mosaics differ wildly between surveys (camera, haze, season, sun angle,
JPEG artefacts, GSD). The judges' data will be from a survey we have never seen,
so augmentation is deliberately heavy on photometric and scale variation.
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def train_augs(size: int = 512, strength: str = "strong") -> A.Compose:
    s = 1.0 if strength == "strong" else 0.5
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.35, 1.0), ratio=(0.8, 1.25), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(rotate=(-30, 30), scale=(0.75, 1.3), p=0.4 * s),
        A.OneOf([
            A.RandomBrightnessContrast(0.3, 0.3),
            A.HueSaturationValue(12, 30, 25),
            A.RandomGamma((70, 150)),
            A.CLAHE(),
        ], p=0.9 * s),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.Downscale(scale_range=(0.4, 0.8)),
            A.ImageCompression(quality_range=(35, 85)),
        ], p=0.5 * s),
        A.OneOf([
            A.RandomFog(fog_coef_range=(0.05, 0.25)),
            A.RandomShadow(),
            A.GaussNoise(std_range=(0.02, 0.08)),
        ], p=0.3 * s),
        A.ToGray(p=0.05),
    ])


def eval_augs(size: int = 512) -> A.Compose:
    return A.Compose([A.CenterCrop(size, size, pad_if_needed=True)])


def normalize(img_u8: np.ndarray) -> torch.Tensor:
    x = (img_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class ChipDataset(Dataset):
    def __init__(self, root: str | Path, index: pd.DataFrame, augs=None, with_instances=False):
        self.root = Path(root)
        self.df = index.reset_index(drop=True)
        self.augs = augs
        self.with_instances = with_instances

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = np.array(Image.open(self.root / r["image"]).convert("RGB"))
        m = np.array(Image.open(self.root / r["mask"]))
        masks = [m[..., 0] > 127, m[..., 1] > 127, m[..., 2] > 127]
        if self.with_instances:
            masks.append(np.array(Image.open(self.root / r["inst"])).astype(np.int32))
        masks = [x.astype(np.int32) if x.dtype == bool else x for x in masks]
        if self.augs is not None:
            out = self.augs(image=img, masks=masks)
            img, masks = out["image"], out["masks"]
        y = torch.from_numpy(np.stack([masks[0], masks[1]]).astype(np.float32))
        valid = torch.from_numpy(masks[2].astype(np.float32))
        item = {"image": normalize(img), "target": y, "valid": valid, "idx": i}
        if self.with_instances:
            item["inst"] = torch.from_numpy(masks[3].astype(np.int64))
        return item


def balanced_sampler(df: pd.DataFrame, by: str = "region", weights: dict | None = None,
                     num_samples: int | None = None) -> WeightedRandomSampler:
    """Sample so each group (region / country / scene) contributes evenly, optionally
    re-weighted, e.g. {'india': 2.0, 'south_asia': 1.5, 'africa': 1.0}."""
    # `by` may be a list of columns, e.g. [source, region] -> groups "ramp/india", "oam/south_asia"
    key = df[by].astype(str).agg("/".join, axis=1) if isinstance(by, (list, tuple)) else df[by]
    counts = key.value_counts()
    w = key.map(lambda g: (weights or {}).get(g, 1.0) / counts[g]).values
    return WeightedRandomSampler(torch.as_tensor(np.array(w, dtype=np.float64)),
                                 num_samples=num_samples or len(df), replacement=True)
