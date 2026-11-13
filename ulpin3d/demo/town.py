"""A synthetic Indian ward: one consistent survey package for end-to-end testing.

Generates the folder the judges would hand over, plus the ground truth:

    <out>/area.yaml                      area settings (parcel id length, height limit)
    <out>/ortho/ortho.tif                drone orthophoto, 0.10 m, UTM 43N
    <out>/lidar/cloud.laz                aerial LiDAR, ~10 pts/m2, RGB-coloured, UNCLASSIFIED
    <out>/parcels/parcels.geojson        2D parcels with 12-char ULPINs
    <out>/plans/*.pdf|dxf|png            sanctioned typical-floor plans (neutral file names)
    <out>/ownership/ownership.csv        record-of-rights rows per flat
    <out>/truth.json                     what the pipeline should find

The ward has:
  * apartment blocks (stilt + typical floors) with sanctioned plans in different
    formats (vector PDF, DXF, scanned image), placed and rotated inside their parcels
  * one block with an extra floor beyond its sanction (-> 'unsanctioned')
  * independent houses (G+1..G+3) without plans
  * one house built 1.5 m over its parcel boundary (-> 'encroachment')
  * an open plot with trees; roads; boundary walls; water tanks; stair cabins

All names and IDs are invented and the data is marked SYNTHETIC.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import yaml
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from shapely import affinity
from shapely.geometry import Point, box, mapping

from ulpin3d.floorplan.synthetic import make_plan, write_dxf, write_pdf, write_scan

EPSG = 32643
ORIGIN = (373_000.0, 2_048_000.0)  # Pune, UTM 43N
FIRST = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Karan", "Meera", "Arjun", "Kavya",
         "Rahul", "Pooja", "Siddharth", "Nisha", "Aditya", "Divya", "Manish", "Shreya", "Varun", "Ishita"]
LAST = ["Sharma", "Patil", "Kulkarni", "Deshpande", "Iyer", "Reddy", "Joshi", "Gupta", "Nair", "Shinde",
        "Mehta", "Rao", "Kapoor", "Jadhav", "Bose", "Menon"]


def _ulpin(k):
    return f"27PUNEW1{k:04d}"  # 12 chars: state 27 + PUNE + ward W1 + plot no. (synthetic)


def _sample_in(poly, density, rng):
    x0, y0, x1, y1 = poly.bounds
    n = int((x1 - x0) * (y1 - y0) * density * 1.05) + 8
    x, y = rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)
    m = shapely.contains_xy(poly, x, y)
    return x[m], y[m]


def _ring_points(poly, z0, z1, density, rng):
    ring = poly.exterior
    n = max(8, int(ring.length * (z1 - z0) * density))
    t = rng.uniform(0, 1, n)
    pts = np.array([ring.interpolate(v, normalized=True).coords[0] for v in t])
    return pts[:, 0], pts[:, 1], rng.uniform(z0, z1, n)


def make_town(out, seed=0, gsd=0.10, density=10.0):
    rng = np.random.default_rng(seed)
    out = Path(out)
    for d in ("ortho", "lidar", "parcels", "plans", "ownership"):
        (out / d).mkdir(parents=True, exist_ok=True)
    X0, Y0 = ORIGIN
    road = 9.0
    pw, ph = 56.0, 40.0  # parcel size
    ncol, nrow = 3, 3
    W = ncol * pw + (ncol + 1) * road
    H = nrow * ph + (nrow + 1) * road

    # ---------------------------------------------------------------- parcels
    parcels = []
    k = 101
    for r in range(nrow):
        for c in range(ncol):
            x = X0 + road + c * (pw + road)
            y = Y0 + road + r * (ph + road)
            parcels.append(dict(ulpin=_ulpin(k), plot_no=str(k), geometry=box(x, y, x + pw, y + ph)))
            k += 1
    roles = ["apt", "house", "apt", "house", "open", "apt", "house", "apt", "encroach"]

    buildings, plans_meta, owners = [], [], []
    plan_formats = ["pdf", "dxf", "png", "pdf"]
    api = 0
    for p, role in zip(parcels, roles):
        cx, cy = p["geometry"].centroid.x, p["geometry"].centroid.y
        if role == "apt":
            pseed = 100 + api
            plan = make_plan(pseed, n_per_side=2, floor_from=1, floor_to=4 + api)
            n_typ = 4 + api
            extra = 1 if api == 2 else 0  # one block has an unsanctioned extra floor
            plate = plan["plate"]
            ang = float(rng.uniform(-10, 10))
            local = affinity.rotate(plate, ang, origin=(0, 0))
            dx, dy = cx - local.centroid.x, cy - local.centroid.y
            fp = affinity.translate(local, dx, dy)
            stair = next(pp for nm, pp in plan["common"] if nm == "STAIRCASE")
            cabin = affinity.translate(affinity.rotate(stair, ang, origin=(0, 0)), dx, dy).buffer(-0.3)
            levels = 1 + n_typ + extra  # stilt + typical (+ extra)
            fmt = plan_formats[api]
            name = f"BP-2025-{3100 + 17 * api}_typical_floor"
            if fmt == "pdf":
                write_pdf(plan, out / "plans" / f"{name}.pdf")
            elif fmt == "dxf":
                write_dxf(plan, out / "plans" / f"{name}.dxf")
            else:
                write_pdf(plan, out / "plans" / f"_tmp_{name}.pdf")
                write_scan(out / "plans" / f"_tmp_{name}.pdf", out / "plans" / f"{name}.png", seed=pseed)
                (out / "plans" / f"_tmp_{name}.pdf").unlink()
            plans_meta.append(dict(file=f"{name}.{fmt}", parcel=p["ulpin"], floors=[1, n_typ],
                                   flats_per_floor=len(plan["units"])))
            for lv in range(1, n_typ + 1):
                for i, u in enumerate(plan["units"]):
                    owners.append(dict(parcel_ulpin=p["ulpin"], unit_label=f"FLAT {lv}{i + 1:02d}",
                                       owner_name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                                       share_pct=100, right_type="Ownership",
                                       document=f"Sale deed {int(rng.integers(2008, 2025))}/{int(rng.integers(1000, 9999))}"))
            buildings.append(dict(parcel=p["ulpin"], kind="apartment", geometry=fp, levels=levels,
                                  sanctioned_top=n_typ, unsanctioned_levels=[levels - 1] if extra else [],
                                  plan=f"{name}.{fmt}", roof_style="flat", cabin=cabin))
            api += 1
        elif role in ("house", "encroach"):
            w, d = rng.uniform(9, 14), rng.uniform(10, 15)
            ang = float(rng.uniform(-6, 6))
            hb = affinity.rotate(box(-w / 2, -d / 2, w / 2, d / 2), ang)
            if role == "encroach":
                # pushed against the eastern boundary, 1.5 m over it
                g = p["geometry"]
                hb = affinity.translate(hb, g.bounds[2] - hb.bounds[2] + 1.5, cy)
            else:
                hb = affinity.translate(hb, cx + rng.uniform(-8, 8), cy + rng.uniform(-4, 4))
            lv = int(rng.integers(2, 4))
            buildings.append(dict(parcel=p["ulpin"], kind="house", geometry=hb, levels=lv, sanctioned_top=None,
                                  unsanctioned_levels=[], plan=None, encroach=role == "encroach", roof_style="flat"))
    for b in buildings:
        b["height"] = 3.5 + 3.1 * (b["levels"] - 1) + 0.4  # stilt/ground 3.5 m, typical 3.1 m, slab
        b["ground_z"] = 560.0 + 0.004 * (b["geometry"].centroid.x - X0) + 0.003 * (b["geometry"].centroid.y - Y0)

    trees = []
    for p, role in zip(parcels, roles):
        n = 9 if role == "open" else 2
        for _ in range(n):
            for _try in range(20):
                pt = Point(rng.uniform(*p["geometry"].bounds[0::2]), rng.uniform(*p["geometry"].bounds[1::2]))
                if not any(b["geometry"].buffer(2.5).contains(pt) for b in buildings):
                    trees.append((pt.x, pt.y, rng.uniform(2.0, 4.0), rng.uniform(6, 12)))
                    break

    def gz(x, y):
        return 560.0 + 0.004 * (x - X0) + 0.003 * (y - Y0)

    # ---------------------------------------------------------------- orthophoto
    ncols, nrows = int(W / gsd), int(H / gsd)
    tr = from_origin(X0, Y0 + H, gsd, gsd)
    img = np.empty((nrows, ncols, 3), np.float32)
    noise = ndi.gaussian_filter(rng.normal(0, 1, (nrows // 8 + 1, ncols // 8 + 1)), 2)
    noise = np.kron(noise, np.ones((8, 8)))[:nrows, :ncols]
    img[:] = [158, 140, 112]
    img += noise[..., None] * 14
    roads = box(X0, Y0, X0 + W, Y0 + H).difference(shapely.union_all([p["geometry"] for p in parcels]))
    rm = rasterize([(roads, 1)], (nrows, ncols), transform=tr).astype(bool)
    img[rm] = [92, 92, 96] + rng.normal(0, 5, (rm.sum(), 3))
    walls = shapely.union_all([p["geometry"].exterior.buffer(0.12) for p in parcels])
    wm = rasterize([(walls, 1)], (nrows, ncols), transform=tr).astype(bool)
    img[wm] = [190, 185, 175]
    roof_cols = np.array([[150, 150, 148], [175, 168, 158], [196, 160, 122], [205, 178, 152], [130, 128, 126]], float)
    for b in buildings:
        sh = affinity.translate(b["geometry"], b["height"] * 0.18, -b["height"] * 0.12)
        m = rasterize([(sh, 1)], (nrows, ncols), transform=tr).astype(bool)
        img[m] *= 0.55
    for b in buildings:
        g = b["geometry"]
        m = rasterize([(g, 1)], (nrows, ncols), transform=tr).astype(bool)
        img[m] = roof_cols[rng.integers(len(roof_cols))] + rng.normal(0, 5, (m.sum(), 3))
        edge = m & ~ndi.binary_erosion(m, iterations=3)
        img[edge] *= 0.78
        c = g.centroid
        for _ in range(1 + (b["kind"] == "apartment") * 3):
            t = Point(c.x + rng.uniform(-3, 3), c.y + rng.uniform(-3, 3)).buffer(0.6)
            tm = rasterize([(t, 1)], (nrows, ncols), transform=tr).astype(bool) & m
            img[tm] = [28, 28, 30]
        # stair cabin ("mumty") stands over the staircase; houses: over the middle
        cab = b.get("cabin") or affinity.rotate(box(c.x - 1.5, c.y - 1.5, c.x + 1.5, c.y + 1.5), 10)
        cm = rasterize([(cab, 1)], (nrows, ncols), transform=tr).astype(bool) & m
        img[cm] = img[cm] * 0.88 + 18
        b["tanks_cabin"] = cab
    for x, y, r, h in trees:
        t = Point(x, y).buffer(r)
        tm = rasterize([(t, 1)], (nrows, ncols), transform=tr).astype(bool)
        img[tm] = np.array([58, 92, 48]) * rng.uniform(0.85, 1.15) + rng.normal(0, 10, (tm.sum(), 3))
    img = np.clip(ndi.gaussian_filter(img, (0.6, 0.6, 0)), 2, 253).astype(np.uint8)
    prof = dict(driver="GTiff", width=ncols, height=nrows, count=3, dtype="uint8", crs=f"EPSG:{EPSG}",
                transform=tr, tiled=True, blockxsize=512, blockysize=512, compress="jpeg", photometric="YCBCR")
    with rasterio.open(out / "ortho" / "ortho.tif", "w", **prof) as dst:
        dst.write(img.transpose(2, 0, 1))
        dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)

    # ---------------------------------------------------------------- LiDAR (unclassified, RGB)
    P, C = [], []  # points, true class (0 ground, 1 building, 2 veg, 3 other)

    def add(x, y, z, cls):
        P.append(np.c_[x, y, z])
        C.append(np.full(len(x), cls, np.uint8))

    full = box(X0, Y0, X0 + W, Y0 + H)
    occupied = shapely.union_all([b["geometry"] for b in buildings])
    gx, gy = _sample_in(full.difference(occupied), density, rng)
    add(gx, gy, gz(gx, gy) + rng.normal(0, 0.03, len(gx)), 0)
    for b in buildings:
        g = b["geometry"]
        z_roof = b["ground_z"] + b["height"]
        x, y = _sample_in(g, density, rng)
        add(x, y, z_roof + rng.normal(0, 0.03, len(x)), 1)
        add(*_ring_points(g, z_roof, z_roof + 1.0, density * 3, rng), 1)  # parapet
        add(*_ring_points(g, b["ground_z"], z_roof, density * 0.15, rng), 1)  # walls (sparse)
        x, y = _sample_in(b["tanks_cabin"], density, rng)
        add(x, y, np.full(len(x), z_roof + 2.6), 1)
    for x0, y0, r, h in trees:
        x, y = _sample_in(Point(x0, y0).buffer(r), density * 1.3, rng)
        d = np.hypot(x - x0, y - y0) / r
        add(x, y, gz(x, y) + h - d ** 2 * h * 0.4 + rng.normal(0, 0.4, len(x)), 2)
        k = rng.random(len(x)) < 0.35
        add(x[k], y[k], gz(x[k], y[k]) + rng.uniform(1.5, h * 0.7, k.sum()), 2)
    for p in parcels:  # 1.8 m boundary walls
        x, y, zz = _ring_points(p["geometry"], 0, 1.8, density * 0.8, rng)
        add(x, y, gz(x, y) + zz, 3)
    xyz = np.concatenate(P)
    cls = np.concatenate(C)
    xyz[:, :2] += rng.normal(0, 0.04, (len(xyz), 2))
    # colour from the orthophoto
    col = np.clip(((xyz[:, 0] - X0) / gsd).astype(int), 0, ncols - 1)
    row = np.clip(((Y0 + H - xyz[:, 1]) / gsd).astype(int), 0, nrows - 1)
    rgb = img[row, col].astype(np.uint16) * 256
    import laspy

    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.scales = [0.001] * 3
    hdr.offsets = [X0, Y0, 0]
    from pyproj import CRS

    hdr.add_crs(CRS.from_epsg(EPSG))
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz.T
    las.classification = np.ones(len(xyz), np.uint8)  # delivered unclassified
    las.red, las.green, las.blue = rgb.T
    las.intensity = np.clip(rng.normal(np.choose(cls, [900, 1500, 600, 1100]), 120), 0, 65535).astype(np.uint16)
    nret = np.where(cls == 2, 2, 1).astype(np.uint8)
    las.number_of_returns = nret
    las.return_number = np.ones(len(xyz), np.uint8)
    las.write(str(out / "lidar" / "cloud.laz"))

    # ---------------------------------------------------------------- vectors, records, truth
    gpd.GeoDataFrame(parcels, crs=EPSG).to_crs(4326).to_file(out / "parcels" / "parcels.geojson", driver="GeoJSON")
    pd.DataFrame(owners).to_csv(out / "ownership" / "ownership.csv", index=False)
    (out / "area.yaml").write_text(yaml.safe_dump({
        "name": "Synthetic Ward W1, Pune (demo)", "synthetic": True, "parcel_chars": 12,
        "permissible_height_m": 30.0, "parcel_id_field": "ulpin"}, sort_keys=False))
    truth = {"buildings": [{"parcel": b["parcel"], "kind": b["kind"], "levels": b["levels"],
                            "height": round(b["height"], 2), "sanctioned_top": b["sanctioned_top"],
                            "unsanctioned_levels": b["unsanctioned_levels"], "plan": b["plan"],
                            "encroach": bool(b.get("encroach")), "polygon_utm": mapping(b["geometry"])}
                           for b in buildings],
             "plans": plans_meta, "n_flats": len(owners), "epsg": EPSG,
             "lidar_points": int(len(xyz)), "lidar_true_class_counts": np.bincount(cls, minlength=4).tolist()}
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth
