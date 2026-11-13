"""Synthetic Indian sanctioned floor plans with exact unit ground truth.

A typical residential floor of the kind submitted for building permission in
Indian municipalities:
  * rectangular floor plate, double-loaded corridor, stair + lift core
  * 1/2/3 BHK flats on both sides, numbered per floor ("FLAT 401")
  * 230 mm (9") external and party walls, 115 mm (4.5") internal partitions
    (standard Indian brick sizes; the parser uses this to tell unit boundaries
    from room partitions)
  * door openings, room names, title "TYPICAL FLOOR PLAN (1ST TO 6TH FLOOR)", "SCALE 1:100"

Writes the same plan as DXF (mm), vector PDF (1:100 on paper) and PNG (scan-like,
with noise, blur and a small skew), plus <name>_gt.json with unit polygons in metres.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

T_EXT, T_INT = 0.23, 0.115
ORD = {1: "1ST", 2: "2ND", 3: "3RD"}


def _ord(n):
    return ORD.get(n if n < 20 else n % 10, f"{n}TH") if n not in (11, 12, 13) else f"{n}TH"


def _unit_rooms(rng, x0, y0, w, d, bhk, facing_north):
    """Split a unit rectangle into rooms. Returns list of (name, box) and internal wall boxes."""
    rooms, walls = [], []
    # depth split: front zone (corridor side) = kitchen/toilet, back = living/bedrooms
    front = d * rng.uniform(0.35, 0.42)
    if facing_north:  # corridor is at the top (north) edge of this unit
        yb0, yb1, yf0, yf1 = y0, y0 + d - front, y0 + d - front, y0 + d
    else:
        yf0, yf1, yb0, yb1 = y0, y0 + front, y0 + front, y0 + d
    # back zone columns: living + bedrooms
    nback = 1 + bhk
    cuts = np.sort(rng.uniform(0.3, 0.7, 0)) if nback == 1 else None
    widths = rng.dirichlet(np.ones(nback) * 6) * w
    xs = np.concatenate([[x0], x0 + np.cumsum(widths)])
    names = ["LIVING"] + [f"BED {i + 1}" for i in range(bhk)]
    for i in range(nback):
        rooms.append((names[i], box(xs[i], yb0, xs[i + 1], yb1)))
        if i > 0:
            walls.append(box(xs[i] - T_INT / 2, yb0, xs[i] + T_INT / 2, yb1))
    # front zone: kitchen | toilet(s) | entrance/foyer
    nt = 1 if bhk == 1 else 2
    fw = rng.dirichlet([4, 2] + [2] * (nt - 1)) * w
    fx = np.concatenate([[x0], x0 + np.cumsum(fw)])
    fnames = ["KITCHEN", "TOILET"] + (["TOILET 2"] if nt == 2 else [])
    for i in range(len(fw)):
        rooms.append((fnames[i], box(fx[i], yf0, fx[i + 1], yf1)))
        if i > 0:
            walls.append(box(fx[i] - T_INT / 2, yf0, fx[i] + T_INT / 2, yf1))
    ysplit = yb1 if not facing_north else yb1
    walls.append(box(x0, ysplit - T_INT / 2, x0 + w, ysplit + T_INT / 2) if facing_north
                 else box(x0, yf1 - T_INT / 2, x0 + w, yf1 + T_INT / 2))
    return rooms, walls, cuts


def make_plan(seed=0, n_per_side=None, floor_from=1, floor_to=6):
    rng = np.random.default_rng(seed)
    n = n_per_side or int(rng.integers(2, 5))
    bhks = [int(rng.choice([1, 2, 2, 3])) for _ in range(2 * n)]
    widths = [{1: rng.uniform(5.0, 6.2), 2: rng.uniform(7.0, 8.4), 3: rng.uniform(9.0, 10.5)}[b] for b in bhks]
    depth = rng.uniform(8.5, 10.5)
    corr = rng.uniform(1.5, 2.1)
    core_w = rng.uniform(5.5, 6.5)
    # plate: south row | corridor | north row ; core in the middle of the length
    south_w = sum(widths[:n]) + core_w
    north_w = sum(widths[n:]) + core_w
    L = max(south_w, north_w)
    W = 2 * depth + corr
    walls, units, rooms_all, common = [], [], [], []
    plate = box(0, 0, L, W)
    # external walls
    walls.append(plate.difference(plate.buffer(-T_EXT, join_style=2)))
    # corridor walls
    yc0, yc1 = depth, depth + corr
    walls.append(box(0, yc0 - T_EXT / 2, L, yc0 + T_EXT / 2))
    walls.append(box(0, yc1 - T_EXT / 2, L, yc1 + T_EXT / 2))
    doors = []
    for side in (0, 1):
        ws = widths[:n] if side == 0 else widths[n:]
        bh = bhks[:n] if side == 0 else bhks[n:]
        # put core after the first half of units
        split = len(ws) // 2
        x = 0.0
        y0, y1 = (0, depth) if side == 0 else (depth + corr, W)
        seq = list(range(split)) + ["core"] + list(range(split, len(ws)))
        for k in seq:
            if k == "core":
                cw = (L - sum(ws))
                if side == 0:
                    stair = box(x, y0, x + cw * 0.6, y1)
                    lift = box(x + cw * 0.6, y0 + (y1 - y0) * 0.45, x + cw, y1)
                    common += [("STAIRCASE", stair), ("LIFT", lift), ("LOBBY", box(x + cw * 0.6, y0, x + cw, y0 + (y1 - y0) * 0.45))]
                    walls.append(box(x + cw * 0.6 - T_EXT / 2, y0, x + cw * 0.6 + T_EXT / 2, y1))
                    walls.append(box(x + cw * 0.6, y0 + (y1 - y0) * 0.45 - T_EXT / 2, x + cw, y0 + (y1 - y0) * 0.45 + T_EXT / 2))
                    doors.append(box(x + cw * 0.15, yc0 - 0.2, x + cw * 0.45, yc0 + 0.2))  # stair opening
                else:
                    common.append(("DUCT / SHAFT", box(x, y0, x + cw, y1)))
                walls.append(box(x - T_EXT / 2, y0, x + T_EXT / 2, y1))
                x += cw
                walls.append(box(x - T_EXT / 2, y0, x + T_EXT / 2, y1))
                continue
            w = ws[k]
            if x > 0:
                walls.append(box(x - T_EXT / 2, y0, x + T_EXT / 2, y1))  # party wall
            inner = box(x + (T_EXT / 2 if x > 0 else T_EXT), y0 + (T_EXT if side == 0 else T_EXT / 2),
                        x + w - (T_EXT / 2 if x + w < L - 1e-6 else T_EXT), y1 - (T_EXT / 2 if side == 0 else T_EXT))
            rooms, iw, _ = _unit_rooms(rng, inner.bounds[0], inner.bounds[1], inner.bounds[2] - inner.bounds[0],
                                       inner.bounds[3] - inner.bounds[1], bh[k], facing_north=(side == 0))
            walls += iw
            idx = len(units) + 1
            units.append({"index": idx, "bhk": bh[k], "polygon": inner, "rooms": rooms})
            # main door into corridor, internal doors between rooms
            dx = x + w * rng.uniform(0.55, 0.8)
            yd = yc0 if side == 0 else yc1
            doors.append(box(dx, yd - 0.2, dx + 1.0, yd + 0.2))
            for nm, r in rooms:
                b = r.bounds
                if nm.startswith("BED") or nm.startswith("TOILET") or nm == "KITCHEN":
                    # door on the wall between front and back zones
                    cx = b[0] + (b[2] - b[0]) * rng.uniform(0.25, 0.6)
                    yy = b[3] if (side == 0) == nm.startswith("BED") else b[1]
                    doors.append(box(cx, yy - 0.12, cx + 0.8, yy + 0.12))
            x += w
    wall_union = unary_union(walls).difference(unary_union(doors)).intersection(plate)
    # labels: flats numbered per floor, e.g. 401..408 for the 4th floor drawn as "typical"
    fl = floor_from
    for i, u in enumerate(units):
        u["label"] = f"FLAT {fl}{i + 1:02d}"
    gt = {
        "plate": mapping(plate), "floors": [floor_from, floor_to], "scale": 100,
        "units": [{"label": u["label"], "bhk": u["bhk"], "polygon": mapping(u["polygon"]),
                   "carpet_area_m2": round(u["polygon"].area, 2)} for u in units],
        "common": [{"label": nm, "polygon": mapping(p)} for nm, p in common],
    }
    return dict(walls=wall_union, units=units, common=common, plate=plate, doors=doors, gt=gt,
                title=(f"TYPICAL FLOOR PLAN ({_ord(floor_from)} TO {_ord(floor_to)} FLOOR)" if floor_to > floor_from
                       else f"{_ord(floor_from)} FLOOR PLAN" if floor_from > 0 else "GROUND FLOOR PLAN"))


def _iter_polys(g):
    if g.is_empty:
        return []
    return list(g.geoms) if hasattr(g, "geoms") else [g]


def write_dxf(plan, path):
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4  # millimetres
    msp = doc.modelspace()
    for lay in ("WALL", "TEXT", "DIM", "DOOR"):
        doc.layers.add(lay)
    mm = 1000.0
    for p in _iter_polys(plan["walls"]):
        for ring in [p.exterior, *p.interiors]:
            msp.add_lwpolyline([(x * mm, y * mm) for x, y in ring.coords], close=True, dxfattribs={"layer": "WALL"})
    for u in plan["units"]:
        c = u["polygon"].centroid
        msp.add_text(u["label"], height=250, dxfattribs={"layer": "TEXT"}).set_placement((c.x * mm, c.y * mm + 300))
        msp.add_text(f"({u['bhk']}BHK)", height=180, dxfattribs={"layer": "TEXT"}).set_placement((c.x * mm, c.y * mm - 100))
        for nm, r in u["rooms"]:
            rc = r.representative_point()
            b = r.bounds
            msp.add_text(nm, height=120, dxfattribs={"layer": "TEXT"}).set_placement((rc.x * mm, rc.y * mm - 600))
            msp.add_text(f"{b[2] - b[0]:.2f}x{b[3] - b[1]:.2f}", height=90,
                         dxfattribs={"layer": "DIM"}).set_placement((rc.x * mm, rc.y * mm - 800))
    for nm, p in plan["common"]:
        c = p.representative_point()
        msp.add_text(nm, height=180, dxfattribs={"layer": "TEXT"}).set_placement((c.x * mm, c.y * mm))
    b = plan["plate"].bounds
    msp.add_text("CORRIDOR", height=180, dxfattribs={"layer": "TEXT"}).set_placement(((b[2] * 0.25) * mm, (plan["gt"]["units"] and _corr_y(plan)) * mm))
    msp.add_text(plan["title"], height=400, dxfattribs={"layer": "TEXT"}).set_placement((0, -1500))
    msp.add_text("SCALE 1:100", height=250, dxfattribs={"layer": "TEXT"}).set_placement((0, -2100))
    doc.saveas(path)


def _corr_y(plan):
    ys = sorted({round(u["polygon"].bounds[3], 2) for u in plan["units"]})
    b = plan["plate"].bounds
    return (b[1] + b[3]) / 2


def write_pdf(plan, path, scale=100):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly

    b = plan["plate"].bounds
    margin = 3.0
    Wm, Hm = b[2] - b[0] + 2 * margin, b[3] - b[1] + 2 * margin + 3
    fig = plt.figure(figsize=(Wm * 1000 / scale / 25.4, Hm * 1000 / scale / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(b[0] - margin, b[0] - margin + Wm)
    ax.set_ylim(b[1] - margin - 3, b[1] - margin - 3 + Hm)
    ax.axis("off")
    for p in _iter_polys(plan["walls"]):
        ax.add_patch(MPoly(np.asarray(p.exterior.coords), closed=True, fc="#222", ec="#000", lw=0.3))
        for r in p.interiors:
            ax.add_patch(MPoly(np.asarray(r.coords), closed=True, fc="white", ec="#000", lw=0.3))
    pt = lambda m: m * 1000 / scale / 0.3528  # metres on ground -> points on paper
    for u in plan["units"]:
        c = u["polygon"].centroid
        ax.text(c.x, c.y + 0.4, u["label"], ha="center", fontsize=pt(0.30), weight="bold")
        ax.text(c.x, c.y - 0.1, f"({u['bhk']}BHK)", ha="center", fontsize=pt(0.2))
        for nm, r in u["rooms"]:
            rc = r.representative_point()
            ax.text(rc.x, rc.y - 0.9, nm, ha="center", fontsize=pt(0.14))
    for nm, p in plan["common"]:
        c = p.representative_point()
        ax.text(c.x, c.y, nm, ha="center", fontsize=pt(0.18))
    ax.text(b[2] * 0.25, _corr_y(plan), "CORRIDOR", ha="center", va="center", fontsize=pt(0.18))
    ax.text(b[0], b[1] - 1.6, plan["title"], fontsize=pt(0.45), weight="bold")
    ax.text(b[0], b[1] - 2.5, "SCALE 1:100", fontsize=pt(0.3))
    fig.savefig(path)
    plt.close(fig)


def write_scan(pdf_path, png_path, dpi=150, seed=0, skew_deg=None):
    import fitz
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(seed)
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(dpi=dpi)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    skew = rng.uniform(-1.5, 1.5) if skew_deg is None else skew_deg
    im = im.rotate(skew, expand=True, fillcolor=255, resample=Image.BICUBIC)
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    a = np.asarray(im).astype(np.float32)
    a = a * rng.uniform(0.85, 1.0) + rng.normal(0, 8, a.shape) + rng.uniform(0, 25)
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(png_path, dpi=(dpi, dpi))
    return skew


def make_plan_files(out_dir, name="plan", seed=0, **kw):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = make_plan(seed, **kw)
    write_dxf(plan, out / f"{name}.dxf")
    write_pdf(plan, out / f"{name}.pdf")
    skew = write_scan(out / f"{name}.pdf", out / f"{name}.png", seed=seed)
    gt = dict(plan["gt"], title=plan["title"], scan_dpi=150, scan_skew_deg=skew)
    (out / f"{name}_gt.json").write_text(json.dumps(gt))
    return plan, gt
