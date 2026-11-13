"""LiDAR training data: class mapping, Sao Paulo tile catalog, sample extraction.

Unified classes (written back as ASPRS codes):
    0 ground      -> 2
    1 building    -> 6
    2 vegetation  -> 5
    3 other       -> 1   (walls, fences, vehicles, street furniture, unclassified)

Source mappings
    Sao Paulo PMSP 2017 (inferred from geometry, see README):
        2 bare ground, 20 paved ground/roads -> ground
        6 building -> building, 3/4/5 vegetation -> vegetation, 19/1 -> other
    ASPRS standard (e.g. most Indian survey deliveries, DALES-like)
        2 -> ground, 6 -> building, 3/4/5 -> vegetation, 9 water -> ground, rest -> other
"""
from __future__ import annotations

import concurrent.futures as cf
import struct
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CLASS_NAMES = ["ground", "building", "vegetation", "other"]
TO_ASPRS = np.array([2, 6, 5, 1], np.uint8)
IGNORE = -1

MAPPINGS = {
    "saopaulo": {2: 0, 20: 0, 6: 1, 3: 2, 4: 2, 5: 2, 19: 3, 1: 3},
    "asprs": {2: 0, 8: 0, 9: 0, 11: 0, 6: 1, 3: 2, 4: 2, 5: 2, 1: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 19: 3},
    "ahn": {2: 0, 9: 0, 6: 1, 26: 3, 1: IGNORE},  # AHN lumps vegetation into 1, so ignore it
    "synthetic": {0: 0, 1: 1, 2: 2, 3: 3},
}

SP_BUCKET = "https://laz-m3dc-pmsp.s3.sa-east-1.amazonaws.com"


def map_classes(cls: np.ndarray, scheme: str) -> np.ndarray:
    lut = np.full(256, IGNORE, np.int16)
    for k, v in MAPPINGS[scheme].items():
        lut[k] = v
    return lut[np.clip(cls, 0, 255)]


def saopaulo_tile_index(cache: Path, workers: int = 8) -> pd.DataFrame:
    """Bounds of every Sao Paulo LAZ tile, from 375-byte header range reads (~3 min once)."""
    p = cache / "saopaulo_tiles.csv"
    if p.exists():
        return pd.read_csv(p)
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys, tok = [], None
    while True:
        prm = {"list-type": "2"}
        if tok:
            prm["continuation-token"] = tok
        t = ET.fromstring(requests.get(SP_BUCKET, params=prm, timeout=60).content)
        keys += [(c.find("s:Key", ns).text, int(c.find("s:Size", ns).text)) for c in t.findall("s:Contents", ns)]
        if t.find("s:IsTruncated", ns).text == "true":
            tok = t.find("s:NextContinuationToken", ns).text
        else:
            break
    s = requests.Session()

    def hdr(k):
        for a in range(5):
            try:
                r = s.get(f"{SP_BUCKET}/{k}", headers={"Range": "bytes=0-374"}, timeout=60).content
                maxx, minx, maxy, miny, maxz, minz = struct.unpack("<6d", r[179:227])
                n = struct.unpack("<I", r[107:111])[0]
                return k, minx, miny, maxx, maxy, n
            except Exception:
                time.sleep(1 + a)
        return k, None, None, None, None, None

    with cf.ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(hdr, [k for k, sz in keys if sz > 1e6]))
    df = pd.DataFrame(rows, columns=["key", "minx", "miny", "maxx", "maxy", "n"]).dropna()
    cache.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


def download(url: str, out: Path) -> Path:
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(out)
    return out


def select_saopaulo_tiles(idx: pd.DataFrame, test_keys: list[str], n_train: int, n_val: int,
                          urban_bbox=(318_000, 7_380_000, 345_000, 7_405_000), buffer_m=1500, seed=0):
    """Pick spatially separated train/val tiles inside the dense urban core."""
    rng = np.random.default_rng(seed)
    d = idx.copy()
    d["cx"], d["cy"] = (d.minx + d.maxx) / 2, (d.miny + d.maxy) / 2
    d = d[(d.cx > urban_bbox[0]) & (d.cx < urban_bbox[2]) & (d.cy > urban_bbox[1]) & (d.cy < urban_bbox[3])]
    test = idx[idx.key.isin(test_keys)].copy()
    test["cx"], test["cy"] = (test.minx + test.maxx) / 2, (test.miny + test.maxy) / 2
    far = np.ones(len(d), bool)
    for _, t in test.iterrows():
        far &= np.hypot(d.cx - t.cx, d.cy - t.cy) > buffer_m
    pool = d[far & ~d.key.isin(test_keys)]
    pick = pool.iloc[rng.permutation(len(pool))[: n_train + n_val]]
    out = pd.concat([pick.assign(split=["train"] * min(n_train, len(pick)) + ["val"] * max(0, len(pick) - n_train)),
                     test.assign(split="test")], ignore_index=True)
    return out[["key", "minx", "miny", "maxx", "maxy", "n", "split"]]


def center_crop(row, size_m: float):
    cx, cy = (row.minx + row.maxx) / 2, (row.miny + row.maxy) / 2
    h = size_m / 2
    return (cx - h, cy - h, cx + h, cy + h)
