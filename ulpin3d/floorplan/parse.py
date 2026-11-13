"""Sanctioned floor plan (DXF / vector PDF / scanned image) -> units with polygons.

All three formats are reduced to the same thing: a wall mask in metres plus text
labels with positions. Then:

1. wall thickness   thick walls (party / external / corridor walls, 230 mm in Indian
                    brick construction) are separated from 115 mm room partitions
                    with a morphological opening at ~160 mm.
2. close openings   door gaps in the thick walls (up to ~1.2 m) are closed.
3. regions          enclosed free space between thick walls = one flat, or a
                    common area (corridor, lobby, stair, lift, shaft).
4. labels           "FLAT 401", "UNIT B-12", "SHOP 3" ... name the region; room
                    names and "2BHK" are kept as attributes. "CORRIDOR", "LIFT",
                    "STAIRCASE" ... mark common areas (3D ULPIN unit 00).
5. floors           "TYPICAL FLOOR PLAN (1ST TO 6TH FLOOR)" -> floors 1..6.
6. scale            DXF $INSUNITS; PDF/scan "SCALE 1:100" plus page units or DPI.

Every unit carries a confidence and a list of issues. A scan without a readable
scale is still parsed; its scale is then fixed by fitting the plan to the
building footprint (georef.py).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from scipy import ndimage as ndi
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import polygonize, unary_union

RES = 0.02  # metres per pixel for the internal wall raster
# OCR drops spaces ("FLAT101"), so no word boundaries before the number
UNIT_RE = re.compile(r"(FLAT|UNIT|APARTMENT|APT|SHOP|OFFICE|SHOWROOM|GODOWN)\.?\s*(NO\.?)?\s*[-:]?\s*([A-Z]?-?\d{1,4}[A-Z]?)(?![0-9])")
COMMON_RE = re.compile(r"(CORRIDOR|LOBBY|PASSAGE|STAIR\w*|LIFT|ELEVATOR|DUCT|SHAFT|REFUGE|LANDING|FOYER|ATRIUM)")
BHK_RE = re.compile(r"(\d)\s*BHK")
SCALE_RE = re.compile(r"1\s*:\s*(\d{2,4})(?![0-9])")
# frequent OCR confusions on drawings
OCR_FIX = [(re.compile(r"\b(FAT|FIAT|ELAT|F1AT|FLAI|PLAT|FLA[7I1L]|FLT|FL4T)(?=\s*\d)"), "FLAT"), (re.compile(r"\b(UFT|L1FT|LIFI|LIET)\b"), "LIFT"),
           (re.compile(r"\bC0RRIDOR\b"), "CORRIDOR"), (re.compile(r"\bSTAIRCASF\b"), "STAIRCASE"),
           (re.compile(r"\bSCALF"), "SCALE")]


def norm_text(t: str) -> str:
    t = t.upper().strip()
    for rx, rep in OCR_FIX:
        t = rx.sub(rep, t)
    return t
ORD = r"(\d{1,2})\s*(?:ST|ND|RD|TH)?"
FLOOR_RANGE_RE = re.compile(ORD + r"\s*(?:TO|-|&|AND)\s*" + ORD + r"\s*FLOOR")
FLOOR_ONE_RE = re.compile(ORD + r"\s*FLOOR")


@dataclass
class PlanDrawing:
    """Format-neutral plan: wall geometry or wall mask + texts, in metres (y up)."""
    texts: list  # [(text, x_m, y_m)]
    wall_polys: list | None = None  # shapely polygons (vector sources)
    wall_mask: np.ndarray | None = None  # bool raster (scans), with transform
    transform: Affine | None = None
    source: str = ""
    scale: float | None = None
    issues: list = field(default_factory=list)


# ----------------------------------------------------------------------------- loaders
def load_dxf(path) -> PlanDrawing:
    import ezdxf

    doc = ezdxf.readfile(str(path))
    units = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0}.get(doc.header.get("$INSUNITS", 4), 0.001)
    msp = doc.modelspace()
    lines, texts = [], []
    skip = re.compile(r"TEXT|DIM|ANNO|HATCH|FURN|GRID|TITLE", re.I)
    for e in msp:
        t = e.dxftype()
        lay = e.dxf.get("layer", "")
        if t in ("TEXT", "MTEXT"):
            s = e.plain_text() if t == "MTEXT" else e.dxf.text
            p = e.dxf.insert
            texts.append((s.upper(), p[0] * units, p[1] * units))
            continue
        if skip.search(lay):
            continue
        if t == "LWPOLYLINE":
            pts = [(x * units, y * units) for x, y, *_ in e.get_points()]
            if e.closed and len(pts) > 2:
                pts.append(pts[0])
            lines += [LineString(pts[i:i + 2]) for i in range(len(pts) - 1)]
        elif t == "LINE":
            lines.append(LineString([(e.dxf.start[0] * units, e.dxf.start[1] * units),
                                     (e.dxf.end[0] * units, e.dxf.end[1] * units)]))
        elif t == "POLYLINE":
            pts = [(v.dxf.location[0] * units, v.dxf.location[1] * units) for v in e.vertices]
            lines += [LineString(pts[i:i + 2]) for i in range(len(pts) - 1)]
    return PlanDrawing(texts, wall_polys=_faces_from_lines(lines), source="dxf", scale=None)


def load_pdf(path, page=0) -> PlanDrawing:
    import fitz

    doc = fitz.open(str(path))
    pg = doc[page]
    words = pg.get_text("words")
    lines_txt = {}
    for x0, y0, x1, y1, w, b, l, _ in words:
        lines_txt.setdefault((b, l), []).append((x0, y0, x1, y1, w))
    raw_texts = []
    for items in lines_txt.values():
        s = " ".join(w for *_, w in items).upper()
        xs = [i[0] for i in items] + [i[2] for i in items]
        ys = [i[1] for i in items] + [i[3] for i in items]
        raw_texts.append((s, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2))
    scale = _find_scale([t for t, *_ in raw_texts])
    issues = []
    if scale is None:
        scale = 100
        issues.append("no 'SCALE 1:N' found on sheet; assumed 1:100 (verify against footprint)")
    m_per_pt = 0.0254 / 72 * scale
    H = pg.rect.height
    tx = lambda x, y: (x * m_per_pt, (H - y) * m_per_pt)
    filled, segs = [], []
    for d in pg.get_drawings():
        pts = []
        for it in d["items"]:
            if it[0] == "l":
                pts += [tx(*it[1]), tx(*it[2])]
                segs.append(LineString([tx(*it[1]), tx(*it[2])]))
            elif it[0] == "re":
                r = it[1]
                ring = [tx(r.x0, r.y0), tx(r.x1, r.y0), tx(r.x1, r.y1), tx(r.x0, r.y1), tx(r.x0, r.y0)]
                segs += [LineString(ring[i:i + 2]) for i in range(4)]
                pts += ring
            elif it[0] == "qu":
                q = it[1]
                ring = [tx(*q.ul), tx(*q.ur), tx(*q.lr), tx(*q.ll), tx(*q.ul)]
                segs += [LineString(ring[i:i + 2]) for i in range(4)]
                pts += ring
        fill = d.get("fill")
        if fill is not None and len(pts) >= 3:
            filled.append((sum(fill) / 3, Polygon(pts).buffer(0)))
    walls = None
    if filled:
        # painter's model: dark fills add wall, light fills drawn later cut holes
        g = Polygon()
        for lum, p in filled:
            g = g.union(p) if lum < 0.5 else g.difference(p)
        walls = [g] if not g.is_empty else None
    if walls is None:
        walls = _faces_from_lines(segs)
    texts = [(s, *tx(x, y)) for s, x, y in raw_texts]
    return PlanDrawing(texts, wall_polys=walls, source="pdf", scale=scale, issues=issues)


def load_image(path, dpi=None, ocr=True) -> PlanDrawing:
    from PIL import Image

    im = Image.open(path)
    dpi = dpi or (im.info.get("dpi", (None,))[0])
    g = np.asarray(im.convert("L")).astype(np.float32)
    g, angle = _deskew(g)
    texts_px = _ocr(g) if ocr else []
    scale = _find_scale([t for t, *_ in texts_px])
    issues = []
    if not dpi:
        dpi = 150
        issues.append("image has no DPI; assumed 150 (verify against footprint)")
    if scale is None:
        scale = 100
        issues.append("no 'SCALE 1:N' read; assumed 1:100 (verify against footprint)")
    mpp = 0.0254 / float(dpi) * scale
    from skimage.filters import threshold_otsu

    dark = g < threshold_otsu(g)
    # remove text, dimension lines and door arcs: thinner than a 115 mm partition
    k = max(1, int(round(0.04 / mpp)))
    walls = ndi.binary_opening(dark, structure=np.ones((2 * k + 1, 2 * k + 1)))
    H = g.shape[0]
    tr = Affine(mpp, 0, 0, 0, -mpp, H * mpp)
    texts = [(t, x * mpp, (H - y) * mpp) for t, x, y in texts_px]
    d = PlanDrawing(texts, wall_mask=walls, transform=tr, source="image", scale=scale, issues=issues)
    d.issues.append(f"deskewed by {angle:+.2f} deg")
    return d


def load_plan(path, **kw) -> PlanDrawing:
    ext = Path(path).suffix.lower()
    if ext == ".dxf":
        return load_dxf(path)
    if ext == ".pdf":
        return load_pdf(path, **kw)
    return load_image(path, **kw)


def _faces_from_lines(lines):
    """Polygonise linework; faces thinner than 0.4 m are wall fragments."""
    if not lines:
        return []
    faces = list(polygonize(unary_union(lines)))
    walls = []
    for f in faces:
        thick = 2 * f.area / max(f.length, 1e-9)
        if thick < 0.2:
            walls.append(f)
    return walls


def _find_scale(texts):
    for t in texts:
        m = SCALE_RE.search(t)
        if m and ("SCALE" in t or "SCAL" in t or len(t) < 12):
            return float(m.group(1))
    return None


def _deskew(g):
    """Estimate skew from the dominant orientation of long dark edges (+/- 5 deg)."""
    from skimage.transform import rotate

    dark = g < np.percentile(g, 20)
    best, best_a = -1, 0.0
    small = dark[::2, ::2]
    for a in np.arange(-5, 5.01, 0.25):
        r = ndi.rotate(small.astype(np.float32), a, reshape=False, order=0)
        s = (r.sum(1) ** 2).sum() + (r.sum(0) ** 2).sum()  # projection profiles peak when aligned
        if s > best:
            best, best_a = s, a
    if abs(best_a) < 0.05:
        return g, 0.0
    return rotate(g, best_a, resize=True, mode="constant", cval=255, preserve_range=True), best_a


_OCR = None


def _ocr(g):
    global _OCR
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return []
    if _OCR is None:
        _OCR = RapidOCR()
    res, _ = _OCR(np.clip(g, 0, 255).astype(np.uint8))
    out = []
    for box_, txt, conf in res or []:
        b = np.asarray(box_)
        out.append((norm_text(txt), float(b[:, 0].mean()), float(b[:, 1].mean())))
    return out


# ----------------------------------------------------------------------------- core
def _wall_raster(d: PlanDrawing):
    if d.wall_mask is not None:
        return d.wall_mask, d.transform
    polys = [p for p in d.wall_polys if not p.is_empty]
    if not polys:
        raise ValueError("no walls found in drawing")
    u = unary_union(polys)
    b = u.bounds
    pad = 1.0
    W = int(math.ceil((b[2] - b[0] + 2 * pad) / RES))
    H = int(math.ceil((b[3] - b[1] + 2 * pad) / RES))
    tr = Affine(RES, 0, b[0] - pad, 0, -RES, b[3] + pad)
    m = rasterize([(u, 1)], out_shape=(H, W), transform=tr, fill=0, dtype="uint8").astype(bool)
    return m, tr


def parse_floors(texts):
    for t, *_ in texts:
        s = norm_text(t)
        s = re.sub(r"(\d)(ST|ND|RD|TH)?(TO)(\d)", r"\1\2 TO \4", s)  # OCR: 1STTO6THFLOOR
        s = s.replace("FLOORPLAN", "FLOOR PLAN").replace("THFLOOR", "TH FLOOR").replace("STFLOOR", "ST FLOOR")
        m = FLOOR_RANGE_RE.search(s)
        if m:
            return list(range(int(m.group(1)), int(m.group(2)) + 1))
        if "GROUND FLOOR" in s or "STILT" in s:
            return [0]
        m = re.search(r"BASEMENT\s*-?\s*(\d)?", s)
        if m and "FLOOR" in s or (m and "PLAN" in s):
            return [-(int(m.group(1) or 1))]
        m = FLOOR_ONE_RE.search(s)
        if m and "PLAN" in s:
            return [int(m.group(1))]
    return None


def extract_units(d: PlanDrawing, thick_m=0.16, door_m=1.25, min_unit_m2=8.0, min_common_m2=1.0):
    mask, tr = _wall_raster(d)
    res = abs(tr.a)
    # 1. thick walls survive an opening slightly wider than a 115 mm partition
    r = max(1, int(round(thick_m / 2 / res)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = xx ** 2 + yy ** 2 <= r * r
    thick = ndi.binary_opening(mask, structure=disk)
    # 2. close door openings in thick walls
    rc = max(1, int(round(door_m / 2 / res)))
    thick_c = ndi.binary_closing(np.pad(thick, rc), structure=np.ones((2 * rc + 1, 2 * rc + 1)))[rc:-rc, rc:-rc]
    thick_c |= thick
    # floor plate = everything enclosed by walls
    plate = ndi.binary_fill_holes(ndi.binary_closing(mask, structure=np.ones((2 * rc + 1,) * 2)))
    free = plate & ~thick_c
    lab, n = ndi.label(free)
    # 3. texts -> pixel coords
    inv = ~tr
    tpix = []
    for t, x, y in d.texts:
        c, rr = inv * (x, y)
        tpix.append((t, int(rr), int(c)))
    floors = parse_floors(d.texts)
    units, common, unlabeled = [], [], []
    areas = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1)) * res * res
    for k in range(1, n + 1):
        a = areas[k - 1]
        if a < min_common_m2:
            continue
        reg = lab == k
        txts = [t for t, rr, c in tpix if 0 <= rr < reg.shape[0] and 0 <= c < reg.shape[1] and reg[rr, c]]
        # texts sitting on thin partitions or just outside: search a small neighbourhood
        if not txts:
            near = ndi.binary_dilation(reg, iterations=int(0.3 / res))
            txts = [t for t, rr, c in tpix if 0 <= rr < near.shape[0] and 0 <= c < near.shape[1] and near[rr, c]]
        geom = _vectorise(reg, tr, res)
        joined = " ".join(norm_text(t) for t in txts)
        ulab = UNIT_RE.search(joined)
        clab = COMMON_RE.search(joined)
        rec = {"polygon": geom, "area_m2": round(geom.area, 2), "texts": txts}
        if ulab and a >= min_unit_m2:
            bhk = BHK_RE.search(joined)
            rooms = sorted({t for t in txts if re.search(r"BED|LIVING|KITCHEN|TOILET|DINING|HALL|BALCONY|PUJA|STORE", t)})
            rec.update(label=f"{ulab.group(1)} {ulab.group(3)}", bhk=int(bhk.group(1)) if bhk else None, rooms=rooms,
                       kind="unit", carpet_area_m2=round(geom.area, 2))
            rec["confidence"], rec["issues"] = _unit_confidence(rec, joined)
            units.append(rec)
        elif clab:
            rec.update(label=clab.group(1), kind="common", confidence=0.9, issues=[])
            common.append(rec)
        elif a >= min_unit_m2 and re.search(r"\bF\w{0,3}\s*\d|BHK", joined):
            # a flat whose label OCR could not read; numbered by the series repair below
            bhk = BHK_RE.search(joined)
            digits = re.findall(r"\d{2,4}", joined)
            rec.update(label=f"FLAT {digits[0] if digits else '?'}", bhk=int(bhk.group(1)) if bhk else None,
                       rooms=[], kind="unit", carpet_area_m2=round(geom.area, 2), confidence=0.6,
                       issues=["unit label unreadable; number inferred"])
            units.append(rec)
        elif a >= min_unit_m2:
            rec.update(label=None, kind="unlabeled", confidence=0.3, issues=["enclosed space with no unit/common label"])
            unlabeled.append(rec)
    _repair_sequence(units, floors)
    outline = _vectorise(plate, tr, res)
    labels = [u["label"] for u in units]
    dup = {l for l in labels if labels.count(l) > 1}
    for u in units:
        if u["label"] in dup:
            u["issues"].append("duplicate unit label on this plan")
            u["confidence"] = min(u["confidence"], 0.5)
    return {"units": units, "common": common, "unlabeled": unlabeled, "outline": outline,
            "floors": floors, "scale": d.scale, "source": d.source, "issues": list(d.issues)}


def _lev(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def _repair_sequence(units, floors=None, n_regions=None):
    """Repair flat numbers misread by OCR, using what the plan itself implies.

    Indian plans number flats <floor><nn> (101..108 on the 1st floor). With the
    floor (from the title) and the number of flats (from the geometry) we know
    the expected label set; each label that is outside it, or duplicated, is
    matched to the closest unused expected label by edit distance. Repairs are
    flagged and capped at confidence 0.7."""
    import collections

    if not units:
        return
    nums = [re.sub(r"\D", "", u["label"]) for u in units]
    long_ = [n for n in nums if len(n) >= 3]
    if len(long_) < max(2, len(units) // 2):
        return  # not a floor-coded numbering scheme
    if floors:
        prefix = str(floors[0])
    else:
        prefix = collections.Counter(n[:-2] for n in long_).most_common(1)[0][0]
    N = max(n_regions or 0, len(units), max((int(n[-2:]) for n in long_ if n[:-2] == prefix), default=0))
    expected = [f"{prefix}{i:02d}" for i in range(1, N + 1)]
    counts = collections.Counter(nums)
    keep = {n for n in nums if n in expected and counts[n] == 1}
    free = [e for e in expected if e not in keep]
    word = units[0]["label"].split()[0]
    for u, n in sorted(zip(units, nums), key=lambda t: t[1] in keep):
        if n in keep:
            continue
        if not free:
            u["issues"].append(f"label '{u['label']}' not in expected series {expected[0]}..{expected[-1]}")
            u["confidence"] = min(u["confidence"], 0.5)
            continue
        dists = sorted((_lev(n, e), abs(int(e) - int(n or 0)), e) for e in free)
        best = dists[0][2]
        free.remove(best)
        same_read = counts[n] > 1
        tie = len(dists) > 1 and dists[1][0] == dists[0][0]
        u["issues"].append(f"label read as '{u['label']}', repaired to {word} {best} (numbering series)")
        u["label"] = f"{word} {best}"
        if same_read or tie:
            # several flats read the same way / equally close candidates: the plan does not
            # say which is which, so do not let an ID be issued automatically
            u["issues"].append("ambiguous label repair: confirm flat number on the sanctioned plan")
            u["confidence"] = min(u["confidence"], 0.4)
        else:
            u["confidence"] = min(u["confidence"], 0.7)


def _unit_confidence(rec, joined):
    issues, c = [], 1.0
    if len(set(UNIT_RE.findall(joined))) > 1:
        issues.append("two unit labels in one enclosed space (missing party wall?)")
        c -= 0.4
    a = rec["area_m2"]
    if a < 20 or a > 400:
        issues.append(f"unusual flat size {a:.0f} m2")
        c -= 0.2
    p = rec["polygon"]
    if p.area / max(p.minimum_rotated_rectangle.area, 1e-9) < 0.6:
        issues.append("irregular outline")
        c -= 0.1
    if rec.get("bhk") and not rec.get("rooms"):
        c -= 0.05
    return round(max(c, 0.05), 3), issues


def _vectorise(reg, tr, res):
    polys = [shape(g) for g, v in shapes(reg.astype(np.uint8), mask=reg, transform=tr) if v == 1]
    if not polys:
        return Polygon()
    p = max(polys, key=lambda q: q.area)
    p = Polygon(p.exterior).simplify(res * 2, preserve_topology=True)
    mrr = p.minimum_rotated_rectangle
    return mrr if p.area / max(mrr.area, 1e-9) > 0.97 else p


def parse_plan(path, **kw):
    return extract_units(load_plan(path), **kw)
