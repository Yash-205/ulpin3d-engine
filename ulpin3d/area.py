"""One command for a whole area: survey folder in, 3D cadastre out.

    python -m ulpin3d.cli run-area --area data/judges_area --out out/area \
        [--footprint-model models/footprint_india.pt] [--pc-model models/pointcloud/model.txt]

Expected folder (anything missing is reported, and the pipeline uses what is there):

    area.yaml          optional: name, parcel_chars (12), permissible_height_m, parcel_id_field
    ortho/*.tif        drone orthophoto(s)
    lidar/*.la[sz]     point cloud(s), classified or not
    parcels/*.geojson|gpkg|shp   2D cadastral parcels with their ULPIN
    plans/**           sanctioned plans (DXF / PDF / scans). Pin a plan to its plot either with a
                       sub-folder named after the parcel ULPIN, or with plans/plans_index.csv
                       (columns: file, parcel_ulpin: the building-permission register). Pinned plans
                       are followed even when the AI outline is poor: they are placed on the plot's
                       LiDAR roof and their outline becomes the official footprint. Unpinned plans are
                       matched to buildings by shape.
    ownership/*.csv    record of rights: parcel_ulpin, unit_label, owner_name, share_pct, right_type, document
    footprints/*.geojson  optional building outlines; trusted as surveyed unless area.yaml says
                       footprints_source: ai (then they are checked against LiDAR like model outlines)

Steps: ingest + report -> building outlines (surveyed file / orthophoto model /
LiDAR) -> LiDAR classification + heights -> parcels (outlines spanning several plots are
split along plot lines) + encroachment -> plan matching (plan outline becomes the official
footprint; built area beyond it is flagged) -> units per building -> 3D ULPINs per parcel -> ownership -> checks
(overlaps, storey heights, anomaly model) -> GeoJSON, CityJSON, report, viewer.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import mapping

from ulpin3d.cadastre import assign_ids, build_cadastre, to_cityjson


def _log(msg):
    print(f"[run-area] {msg}", flush=True)


# --------------------------------------------------------------------------- ingest
def ingest(area: Path) -> dict:
    cfg = {}
    if (area / "area.yaml").exists():
        cfg = yaml.safe_load((area / "area.yaml").read_text()) or {}
    files = lambda sub, exts: sorted(p for e in exts for p in (area / sub).rglob(f"*{e}")) if (area / sub).exists() else []
    inv = {
        "cfg": cfg,
        "ortho": files("ortho", [".tif", ".tiff", ".TIF"]),
        "lidar": files("lidar", [".las", ".laz", ".LAS", ".LAZ"]),
        "parcels": files("parcels", [".geojson", ".gpkg", ".shp", ".json"]),
        "plans": [p for p in files("plans", [".dxf", ".DXF", ".pdf", ".PDF", ".png", ".jpg", ".jpeg", ".tif", ".tiff"])
                  if not p.name.startswith("_")],
        "ownership": files("ownership", [".csv"]),
        "footprints": files("footprints", [".geojson", ".gpkg", ".shp"]),
    }
    rep = {"area": str(area), "name": cfg.get("name", area.name)}
    if inv["ortho"]:
        import rasterio

        with rasterio.open(inv["ortho"][0]) as s:
            rep["ortho"] = {"file": inv["ortho"][0].name, "crs": str(s.crs), "gsd_m": round(abs(s.res[0]), 4),
                            "size_px": [s.width, s.height]}
    if inv["lidar"]:
        import laspy

        with laspy.open(str(inv["lidar"][0])) as f:
            h = f.header
            area_m2 = max((h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1]), 1)
            dims = set(h.point_format.dimension_names)
            rep["lidar"] = {"file": inv["lidar"][0].name, "points": int(h.point_count),
                            "density_pts_m2": round(h.point_count / area_m2, 1), "rgb": "red" in dims}
    rep["parcels"] = [p.name for p in inv["parcels"]]
    rep["plans"] = [str(p.relative_to(area / "plans")) for p in inv["plans"]]
    rep["ownership"] = [p.name for p in inv["ownership"]]
    has_o, has_l, has_p = bool(inv["ortho"]), bool(inv["lidar"]), bool(inv["plans"])
    rep["data_grade"] = ("A" if has_o and has_l and has_p else "B" if (has_o or has_l) and (has_l or has_p) else "C")
    rep["missing"] = [k for k in ("ortho", "lidar", "parcels", "plans", "ownership") if not inv[k]]
    return inv, rep


# --------------------------------------------------------------------------- steps
def _classified_cloud(inv, out, pc_model, report):
    from ulpin3d.pointcloud.io import read_points

    src = inv["lidar"][0]
    pts = read_points(src)
    cls = pts["classification"]
    already = cls is not None and np.isin(cls, [2, 6]).sum() > 0.5 * len(cls)
    if already and not pc_model:
        report["lidar_classification"] = "used classes delivered with the cloud"
        return src
    if not pc_model:
        report["lidar_classification"] = "cloud unclassified and no model given; heights unavailable"
        return None
    from ulpin3d.pointcloud.classify import classify_cloud, load_model

    dst = out / "work" / "classified.laz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    lab, conf, _ = classify_cloud(load_model(pc_model), src, dst)
    report["lidar_classification"] = {"model": str(pc_model), "seconds": round(time.time() - t, 1),
                                      "counts": dict(zip(["ground", "building", "vegetation", "other"],
                                                         np.bincount(lab, minlength=4).tolist())),
                                      "mean_confidence": round(float(conf.mean()), 3)}
    return dst


def _buildings(inv, out, cloud, footprint_model, report, crs_hint=None):
    from ulpin3d.fuse import ASPRS_TO_UNIFIED
    from ulpin3d.pointcloud.buildings import attach_heights, buildings_from_lidar
    from ulpin3d.pointcloud.ground import height_above_ground
    from ulpin3d.pointcloud.io import read_points

    src = None
    if inv["footprints"]:
        g = gpd.read_file(inv["footprints"][0])
        g["confidence"] = g.get("confidence", 0.95)
        src = f"outlines file footprints/{inv['footprints'][0].name}"
        g, declared = _declared_heights(g)
        if declared:
            src += f" ({declared} with declared floor counts)"
    elif inv["ortho"] and footprint_model:
        from ulpin3d.footprint.infer import infer_scene, load_predictor

        nd = None
        if load_predictor(footprint_model).needs_height:
            if cloud is None:
                raise SystemExit(f"{footprint_model} needs LiDAR height but the area has no usable point cloud; "
                                 "use an RGB-only model (e.g. models/footprint_lite_rgb)")
            from ulpin3d.pointcloud.rasterize import rasterize_cloud, write_raster

            r = rasterize_cloud(cloud, res=0.3)
            nd = out / "work" / "ndsm.tif"
            write_raster(nd, r["ndsm"].astype(np.float32), r["transform"], r["crs"])
        g = infer_scene(str(inv["ortho"][0]), footprint_model, str(out / "work" / "footprints"), ndsm=str(nd) if nd else None,
                        scene_id="B", tta=False)
        src = f"orthophoto model {Path(footprint_model).name}" + (" + LiDAR height" if nd else "")
    else:
        g = None
    pts = hag = labels = None
    if cloud is not None:
        pts = read_points(cloud)
        lut = np.full(256, 3, np.int16)
        for k, v in ASPRS_TO_UNIFIED.items():
            lut[k] = v
        labels = lut[pts["classification"]]
        hag, *_ = height_above_ground(pts["xyz"])
    if g is None:
        if pts is None:
            raise SystemExit("no building source: give an orthophoto + --footprint-model, a LiDAR cloud, or footprints/")
        g = buildings_from_lidar(pts["xyz"], hag, labels, crs=pts["crs"])
        g["confidence"] = (0.6 + 0.35 * g["lidar_cover"].fillna(0)).clip(0, 0.95)
        src = "LiDAR building points (no orthophoto model given)"
    else:
        utm = pts["crs"] if pts is not None and pts["crs"] is not None else g.estimate_utm_crs()
        g = g.to_crs(utm)
        if pts is not None:
            g = attach_heights(g, pts["xyz"], hag, labels)
    if g.crs is None:
        g = g.set_crs(crs_hint)
    g = g.reset_index(drop=True)
    g["roof_features"] = _roof_features(g, pts, hag, labels) if pts is not None else [[] for _ in range(len(g))]
    g["building_id"] = [f"B{i + 1:04d}" for i in range(len(g))]
    hc = g["height_conf"].fillna(0.5).clip(0.5, 1.0) if "height_conf" in g else 0.75
    g["confidence_3d"] = (g["confidence"].astype(float) * hc).round(4)
    report["buildings"] = {"source": src, "count": int(len(g)), "with_lidar_height": int(g.get("roof_h_main", pd.Series()).notna().sum())}
    _log(f"{len(g)} buildings from {src}")
    lidar_fp = None
    if pts is not None:
        lidar_fp = buildings_from_lidar(pts["xyz"], hag, labels, crs=pts["crs"])
        lidar_fp = lidar_fp.set_crs(g.crs, allow_override=True) if lidar_fp.crs is None else lidar_fp.to_crs(g.crs)
    return g, lidar_fp


def _declared_heights(g, storey_m=3.2):
    """Outline layers often carry floor counts or heights (municipal GIS, OpenStreetMap building:levels).
    Without LiDAR they are the only height evidence: used, but marked as declared, not measured
    (height confidence 0.7, so such buildings stay provisional at best). LiDAR, when present, overrides."""
    lv_col = next((c for c in ("building:levels", "levels", "floors", "n_floors", "storeys", "num_floors") if c in g), None)
    h_col = next((c for c in ("height", "building:height", "height_m") if c in g), None)
    if not lv_col and not h_col:
        return g, 0
    num = lambda c: pd.to_numeric(g[c].astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0], errors="coerce") if c else None
    lv, h = num(lv_col), num(h_col)
    if lv is None:
        lv = (h / storey_m).round()
    if h is None:
        h = lv * storey_m
    h = h.fillna(lv * storey_m)
    ok = lv.notna() & (lv >= 1)
    g["floors_est"] = lv.where(ok)
    g["roof_h_main"] = h.where(ok)
    g["height_conf"] = np.where(ok, 0.7, np.nan)
    return g, int(ok.sum())


def _parcels(inv, cfg, utm):
    if not inv["parcels"]:
        return None
    p = gpd.read_file(inv["parcels"][0]).to_crs(utm)
    fld = cfg.get("parcel_id_field") or next((c for c in p.columns if c.lower() in ("ulpin", "ulpin_id", "bhu_aadhaar", "parcel_id")), None)
    if fld is None:
        raise SystemExit(f"parcels file has no ULPIN column (looked for ulpin/parcel_id); columns: {list(p.columns)}")
    p = p.rename(columns={fld: "ulpin"})
    p["ulpin"] = p["ulpin"].astype(str).str.upper()
    return p


def _split_by_parcels(b, parcels, min_piece_m2=15.0, min_frac=0.25, min_plot_frac=0.3):
    """An image model often merges attached houses on neighbouring plots into one blob, and LiDAR sees a
    row of terraced houses as one roof. When an outline has >= 2 substantial pieces in different parcels,
    cut it along the plot lines (each piece keeps the heights measured for the blob). A piece is
    substantial when it is a fair share of the outline OR covers a fair share of its own plot (one house
    of a 10-house row is only 10 % of the roof but fills half its narrow plot). Slivers over a boundary
    are not separate buildings: they stay attached to the piece they touch (and _assign_parcels reports
    them as encroachment). Nothing is dropped."""
    if parcels is None or not len(b):
        return b, 0
    from shapely.ops import unary_union

    rows, n_split = [], 0
    for _, r in b.iterrows():
        g = r.geometry
        big, small = [], []
        for pg in parcels.geometry[parcels.intersects(g)]:
            x = g.intersection(pg)
            for q in (x.geoms if hasattr(x, "geoms") else [x]):
                if q.geom_type != "Polygon" or q.area <= 0.01:
                    continue
                sub = q.area >= min_piece_m2 and (q.area >= min_frac * g.area or q.area >= min_plot_frac * pg.area)
                (big if sub else small).append(q)
        if len(big) < 2:
            rows.append(r.to_dict())
            continue
        for q in small:  # attach each sliver to the substantial piece it shares the longest edge with
            k = max(range(len(big)), key=lambda i: big[i].buffer(0.05).intersection(q).area)
            big[k] = unary_union([big[k], q]).buffer(0.01).buffer(-0.01)
        n_split += 1
        for x in big:
            d = r.to_dict()
            d["geometry"] = x
            d["confidence_3d"] = round(float(d.get("confidence_3d", 0.8)) * 0.9, 4)
            d["split_note"] = f"detected outline spanned {len(big)} plots: split along the plot boundary"
            rows.append(d)
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=b.crs).reset_index(drop=True)
    out["building_id"] = [f"B{i + 1:04d}" for i in range(len(out))]
    return out, n_split


def _refine_with_lidar(b, lidar_fp, parcels, min_iou=0.85, min_share=0.2):
    """No plan needed: LiDAR sees the true roof edge even where the photo model clipped a roof in
    shadow or ran it on into the courtyard. For each outline, take the LiDAR roofs it mostly covers
    (>= min_share of each roof), clip them to the building's plot, and use that when it disagrees
    with the photo outline. Outlines LiDAR cannot see (no roof points) are kept as detected."""
    if "outline_source" not in b:
        b["outline_source"] = "detected"
    if lidar_fp is None or not len(lidar_fp):
        return b, 0
    n = 0
    for i, g in enumerate(b.geometry):
        cand = lidar_fp.geometry[lidar_fp.intersects(g)]
        own = None
        if parcels is not None and b.parcel_ulpin[i]:
            own = parcels.geometry[parcels.ulpin == b.parcel_ulpin[i]].iloc[0]
        # share of each roof covered by the outline, counting only the part of the roof on this plot
        # (attached houses on neighbouring plots come out of LiDAR as one roof)
        roofs = [r for r in cand
                 if r.intersection(g).area >= min_share * (r.intersection(own.buffer(0.3)).area if own is not None else r.area)]
        if not roofs:
            continue
        roof = gpd.GeoSeries(roofs, crs=b.crs).union_all()
        if own is not None:
            # attached roofs merge across plot lines in LiDAR: drop big pieces that lie in another plot
            # (the neighbour's house) but keep small overhangs, which are real encroachments
            keep = [roof.intersection(own.buffer(0.3))]
            outside = roof.difference(own.buffer(0.3))
            for q in (outside.geoms if hasattr(outside, "geoms") else [outside]):
                if not q.is_empty and q.area < max(15.0, 0.25 * roof.area):
                    keep.append(q)
            roof = gpd.GeoSeries(keep, crs=b.crs).union_all().buffer(0.05).buffer(-0.05)
        if roof.geom_type == "MultiPolygon":
            roof = max(roof.geoms, key=lambda q: q.area)
        if roof.is_empty or roof.geom_type != "Polygon" or roof.area < 8:
            continue
        iou = g.intersection(roof).area / max(g.union(roof).area, 1e-9)
        if iou >= min_iou:
            continue
        b.at[i, "geometry"] = roof
        b.at[i, "outline_source"] = "lidar_roof"
        # the outline is now LiDAR's roof edge: trust it like a LiDAR-only outline, not like the weak photo guess
        if "confidence" in b:
            b.at[i, "confidence"] = max(float(b.at[i, "confidence"]), 0.8)
            hc = float(np.clip(b.at[i, "height_conf"], 0.5, 1.0)) if "height_conf" in b and pd.notna(b.at[i, "height_conf"]) else 0.75
            b.at[i, "confidence_3d"] = round(float(b.at[i, "confidence"]) * hc, 4)
        b.at[i, "building_issues"] = list(b.at[i, "building_issues"]) + [
            f"photo outline disagreed with the LiDAR roof (IoU {iou:.2f}): LiDAR roof outline used"]
        n += 1
    return b, n


def _encroachment(g, par, tol_m=0.3, min_m2=1.0):
    outside = g.difference(par.geometry.buffer(tol_m)).area
    return round(outside, 1) if outside > min_m2 else 0.0


def _assign_parcels(b, parcels, tol_m=0.3, min_m2=1.0):
    b["parcel_ulpin"] = None
    b["encroach_m2"] = 0.0
    b["building_issues"] = [[n] if isinstance(n, str) else [] for n in b.get("split_note", [None] * len(b))]
    if parcels is None:
        for i in range(len(b)):
            b.at[i, "building_issues"].append("no parcel layer: provisional parcel id used")
        return b
    for i, g in enumerate(b.geometry):
        ov = parcels.geometry.intersection(g).area
        j = int(ov.values.argmax()) if len(ov) else -1
        if j < 0 or ov.values[j] <= 0:
            b.at[i, "building_issues"].append("building lies outside every parcel")
            continue
        par = parcels.iloc[j]
        b.at[i, "parcel_ulpin"] = par.ulpin
        outside = _encroachment(g, par, tol_m, min_m2)
        if outside:
            b.at[i, "encroach_m2"] = outside
            b.at[i, "building_issues"].append(f"encroachment: {outside:.1f} m2 beyond parcel {par.ulpin} boundary")
    return b


def _roof_features(b, pts, hag, labels, min_pts=5, cell=0.5):
    """Raised structures on each roof (stair cabin, lift machine room) from LiDAR: building
    points more than 1.8 m above the main roof, grouped on a 0.5 m grid."""
    from scipy import ndimage as ndi
    from shapely import contains_xy

    feats = []
    bm = labels == 1
    X, Y, H = pts["xyz"][bm, 0], pts["xyz"][bm, 1], hag[bm]
    for g, roof in zip(b.geometry, b.get("roof_h_main", [None] * len(b))):
        if roof is None or pd.isna(roof):
            feats.append([])
            continue
        x0, y0, x1, y1 = g.bounds
        m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        m[m] = contains_xy(g, X[m], Y[m])
        m &= H > float(roof) + 1.8
        if m.sum() < min_pts:
            feats.append([])
            continue
        ix = ((X[m] - x0) / cell).astype(int)
        iy = ((Y[m] - y0) / cell).astype(int)
        grid = np.zeros((iy.max() + 1, ix.max() + 1), bool)
        grid[iy, ix] = True
        lab, n = ndi.label(ndi.binary_closing(grid, iterations=1))
        out = []
        for k in range(1, n + 1):
            sel = lab[iy, ix] == k
            if sel.sum() >= min_pts:
                out.append((float(X[m][sel].mean()), float(Y[m][sel].mean())))
        feats.append(out)
    return feats


def _plan_pins(area, parcels) -> dict:
    """plan file (relative to plans/) -> parcel ULPIN, from plans/plans_index.csv if present."""
    f = area / "plans" / "plans_index.csv"
    if not f.exists() or parcels is None:
        return {}
    t = pd.read_csv(f)
    t.columns = [c.strip().lower() for c in t.columns]
    fc = next((c for c in t.columns if c in ("file", "plan", "plan_file")), None)
    pc = next((c for c in t.columns if c in ("parcel_ulpin", "parcel", "ulpin", "plot_ulpin")), None)
    if not fc or not pc:
        return {}
    return {str(a).strip(): str(v).strip().upper() for a, v in zip(t[fc], t[pc])}


def _place_plan_on_image(outline, parcel_geom, probs_path, hint=None, min_contrast=0.15, min_inside=0.25):
    """No LiDAR, and the photo model only caught part of the roof: slide the plan's exact (true-to-
    scale) outline over the plot and keep the pose where the model's building probability is high
    inside the plan and low just outside it. Template matching, so weak evidence spread over the
    whole roof still counts. Returns (placed polygon, contrast) or (None, best contrast)."""
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import from_bounds
    from scipy import ndimage as ndi
    from scipy.signal import fftconvolve
    from shapely import affinity

    from ulpin3d.floorplan.georef import _mrr_angle

    L, B, R, T = parcel_geom.buffer(3.0).bounds
    with rasterio.open(probs_path) as src:
        win = from_bounds(L, B, R, T, src.transform).round_offsets().round_lengths()
        p = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
        edge = src.read(2, window=win, boundless=True, fill_value=0).astype(np.float32) if src.count > 1 else None
        tr = src.window_transform(win)
    scale = 255.0 if p.max() > 1.5 else 1.0
    p /= scale
    if edge is not None:
        edge /= scale
    res = abs(tr.a)
    if p.size == 0:
        return None, 0.0
    inside_par = rasterize([parcel_geom.buffer(0.3)], out_shape=p.shape, transform=tr, fill=0, default_value=1).astype(np.float32)
    # the part of the roof the model WAS sure about must end up inside the plan (it anchors the pose)
    sure = None
    if hint is not None and not hint.is_empty and hint.area > 5:
        sure = rasterize([hint.buffer(-0.5) if hint.buffer(-0.5).area > 2 else hint], out_shape=p.shape, transform=tr,
                         fill=0, default_value=1).astype(np.float32)
        sure = sure if sure.sum() > 10 else None
    angles = {round(_mrr_angle(parcel_geom), 2)} | ({round(_mrr_angle(hint), 2)} if hint is not None and hint.area > 5 else set())
    pa, cx, cy = _mrr_angle(outline), outline.centroid.x, outline.centroid.y
    ring_px = max(2, int(round(1.5 / res)))
    best = (-1.0, None)
    for a in angles:
        for k in range(4):
            rot = affinity.rotate(outline, a - pa + 90 * k, origin=(cx, cy))
            x0, y0, x1, y1 = rot.bounds
            hw = int(math.ceil(max(cx - x0, x1 - cx) / res)) + ring_px + 1
            hh = int(math.ceil(max(cy - y0, y1 - cy) / res)) + ring_px + 1
            ktr = rasterio.transform.from_origin(cx - (hw + 0.5) * res, cy + (hh + 0.5) * res, res, res)
            K = rasterize([rot], out_shape=(2 * hh + 1, 2 * hw + 1), transform=ktr, fill=0, default_value=1).astype(np.float32)
            if K.sum() < 10 or K.shape[0] > p.shape[0] or K.shape[1] > p.shape[1]:
                continue
            Rg = ndi.binary_dilation(K > 0, iterations=ring_px).astype(np.float32) - K
            # 0.3 m band on both sides of the plan's outline: where the model's edge map should light up
            Eb = (ndi.binary_dilation(K > 0, iterations=2) & ~ndi.binary_erosion(K > 0, iterations=2)).astype(np.float32)
            n_in, n_ring = K.sum(), max(Rg.sum(), 1)
            corr = lambda img, ker: fftconvolve(img, ker[::-1, ::-1], mode="same")
            m_in = corr(p, K) / n_in
            m_ring = corr(p * inside_par, Rg) / np.maximum(corr(inside_par, Rg), 1)
            fits = corr(inside_par, K) / n_in > 0.97  # the whole plan must sit inside the plot
            ok = fits & (m_in >= min_inside)
            score = m_in - m_ring
            if edge is not None:  # a roof edge is a sharp line even where the roof itself looks like soil
                score = score + 0.5 * corr(edge, Eb) / max(Eb.sum(), 1)
            if sure is not None:
                cover = corr(sure, K) / sure.sum()
                ok &= cover >= 0.9
                score = score + 0.3 * cover
            score = np.where(ok, score, -1.0)
            j = int(score.argmax())
            if score.flat[j] > best[0]:
                r, c = np.unravel_index(j, score.shape)
                px, py = tr * (c + 0.5, r + 0.5)
                best = (float(score.flat[j]), affinity.translate(rot, px - cx, py - cy))
    if best[1] is None or best[0] < min_contrast:
        return None, max(best[0], 0.0)
    return best[1], best[0]


def _match_plans(inv, area, b, parcels, report, lidar_fp=None, probs=None):
    """Parse every plan, score it against every candidate building, then assign greedily:
    best fits first, one plan per building unless their floor ranges differ. Scans may
    rescale but are penalised for drifting from their stated 1:N scale; the LiDAR floor
    count breaks ties between look-alike blocks. A 180-degree ambiguity is settled with
    the roof cabin seen in LiDAR when possible."""
    from ulpin3d.floorplan.georef import apply_fit, fit_plan_to_footprint, resolve_orientation
    from ulpin3d.floorplan.parse import parse_plan

    plans, rows, cands = [], [], []
    pins = _plan_pins(area, parcels)
    for path in inv["plans"]:
        rel = str(path.relative_to(area / "plans"))
        try:
            p = parse_plan(path)
        except Exception as e:
            rows.append({"plan": rel, "status": "unreadable", "detail": repr(e)[:200]})
            continue
        p["name"] = rel
        k = len(plans)
        plans.append(p)
        scan = p["source"] == "image"
        pinned = pins.get(rel) or pins.get(path.name) or path.parent.name.upper()
        cand = b.index.tolist()
        if parcels is not None and pinned in set(parcels.ulpin):
            cand = b.index[b.parcel_ulpin == pinned].tolist()
            p["pinned_parcel"] = pinned
        if not scan and "pinned_parcel" not in p:  # vector plans are true to scale: pre-filter by area
            a = p["outline"].area
            cand = [i for i in cand if 0.7 < b.geometry[i].area / max(a, 1e-6) < 1.4] or cand
        top = max(p["floors"]) if p["floors"] else None
        for i in cand:
            f = fit_plan_to_footprint(p["outline"], b.geometry[i], allow_scale=scan)
            if f["iou"] < 0.8:
                continue
            score = f["iou"]
            if scan:
                score -= 3 * max(0.0, abs(f["scale"] - 1) - 0.01)
            fl = b.floors_est[i] if "floors_est" in b and pd.notna(b.floors_est[i]) else None
            if top is not None and fl is not None:
                fl = int(fl)
                # LiDAR storeys must cover the plan's floors plus a ground/stilt floor;
                # a building lower than its plan is very unlikely, an extra floor is possible
                score -= 0.15 * max(0, top + 1 - fl) + 0.04 * max(0, fl - top - 1)
            cands.append((score, k, i, f))
    # pinned plans the AI outline could not carry: place them on the plot's LiDAR roof (or, without
    # LiDAR, on whatever was detected inside the plot) and let the plan define the footprint
    have = {k for _, k, _, _ in cands}
    for k, p in enumerate(plans):
        if k in have or "pinned_parcel" not in p:
            continue
        par = parcels[parcels.ulpin == p["pinned_parcel"]].geometry.iloc[0]
        target, how = None, None
        if lidar_fp is not None and len(lidar_fp):
            t = lidar_fp.geometry.intersection(par)
            t = t[t.area > 10].union_all() if (t.area > 10).any() else None
            if t is not None and not t.is_empty:
                target, how = t, "LiDAR roof"
        if target is None and probs is not None:
            det = b.geometry[b.intersects(par)].intersection(par)
            hint = det.union_all() if len(det) else None
            placed, contrast = _place_plan_on_image(p["outline"], par, probs, hint)
            if placed is not None:
                target, how = placed, f"image evidence across the plot (contrast {contrast:.2f})"
        if target is None:
            t = b.geometry[b.intersects(par)].intersection(par)
            if len(t) and t.area.sum() > 10:
                target, how = t.union_all(), "detected outline inside the plot"
        if target is None:
            continue
        if target.geom_type == "MultiPolygon":
            target = max(target.geoms, key=lambda q: q.area)
        f = fit_plan_to_footprint(p["outline"], target, allow_scale=p["source"] == "image")
        placed = apply_fit(p["outline"], f)
        inside = placed.intersection(par.buffer(0.5)).area / max(placed.area, 1e-9)
        if f["iou"] < 0.7 or inside < 0.9:
            continue
        ov = b.geometry.intersection(placed).area
        if not len(ov) or ov.max() <= 0:
            continue
        i = int(ov.values.argmax())
        p["placed_on"] = how
        cands.append((f["iou"] - 0.05, k, b.index[i], f))
    cands.sort(key=lambda t: -t[0])
    taken = defaultdict(list)  # building -> floor sets already covered
    assigned = {}
    for score, k, i, f in cands:
        if k in assigned:
            continue
        fl = set(plans[k]["floors"] or [])
        if any(fl & s or not fl or not s for s in taken[i]):
            continue
        assigned[k] = (i, f, score)
        taken[i].append(fl)
    matches = defaultdict(list)
    for k, p in enumerate(plans):
        if k not in assigned:
            best = max((c for c in cands if c[1] == k), default=None, key=lambda t: t[0])
            rows.append({"plan": p["name"], "status": "no matching building",
                         "best_iou": round(best[3]["iou"], 3) if best else None, "floors": p["floors"]})
            continue
        i, f, score = assigned[k]
        cores = [c["polygon"] for c in p["common"] if re.search(r"STAIR|LIFT|ELEVATOR", c["label"] or "")]
        f = resolve_orientation(f, cores, b.roof_features[i] if "roof_features" in b else [])
        p["fit"] = f
        matches[i].append(p)
        rivals = sorted((c[0] for c in cands if c[1] == k and c[2] != i), reverse=True)
        amb = bool(rivals) and rivals[0] > score - 0.01
        rows.append({"plan": p["name"], "status": "matched" + (" (close second candidate)" if amb else "")
                     + (f" (pinned plot; placed on {p['placed_on']})" if p.get("placed_on") else ""),
                     "building_id": b.building_id[i], "parcel": b.parcel_ulpin[i], "iou": f["iou"], "rms_m": f["rms_m"],
                     "scale": f["scale"], "units_per_floor": len(p["units"]), "floors": p["floors"],
                     "orientation": f.get("orientation_note", "ambiguous: assumed north-up" if f.get("ambiguous") else "unique fit"),
                     "plan_issues": p["issues"] + f["issues"]})
        _log(f"plan {p['name']} -> {b.building_id[i]} on {b.parcel_ulpin[i]} (IoU {f['iou']:.3f}, RMS {f['rms_m']:.2f} m, "
             f"{len(p['units'])} flats/floor, floors {p['floors'][0]}-{p['floors'][-1]}; "
             f"{f.get('orientation_note', 'orientation ambiguous' if f.get('ambiguous') else 'orientation unique')})")
    report["plans"] = rows
    return matches


def _follow_plans(b, matches, parcels, lidar_fp, report):
    """Where a sanctioned plan is matched, its (true-to-scale) outline becomes the building's official
    footprint. What was actually built beyond it, measured on LiDAR roofs when available, is flagged
    as a possible unauthorised extension instead of silently becoming the footprint."""
    from shapely.ops import unary_union

    from ulpin3d.floorplan.georef import apply_fit

    if "outline_source" not in b:
        b["outline_source"] = "detected"
    n_follow, n_ext = 0, 0
    for i, ps in matches.items():
        good = [p for p in ps if p.get("fit") and p["fit"]["status"] != "halt"]
        if not good:
            continue
        plan_fp = unary_union([apply_fit(p["outline"], p["fit"]) for p in good]).buffer(0)
        if plan_fp.geom_type == "MultiPolygon":
            plan_fp = max(plan_fp.geoms, key=lambda q: q.area)
        det = b.geometry[i]
        iou = det.intersection(plan_fp).area / max(det.union(plan_fp).area, 1e-9)
        par = parcels[parcels.ulpin == b.parcel_ulpin[i]].iloc[0] if parcels is not None and b.parcel_ulpin[i] else None
        issues = [x for x in b.at[i, "building_issues"] if not x.startswith("encroachment")]
        ref, src = det, "image outline"
        if lidar_fp is not None and len(lidar_fp):
            near = lidar_fp.geometry[lidar_fp.intersects(plan_fp)]
            if len(near):
                ref, src = near.union_all(), "LiDAR roof"
        if par is not None:
            ref = ref.intersection(par.geometry)
        extra = ref.difference(plan_fp.buffer(0.5)).area
        if extra > max(5.0, 0.05 * plan_fp.area):
            issues.append(f"built area exceeds the sanctioned outline by {extra:.0f} m2 ({src}): possible unauthorised extension")
            n_ext += 1
        if iou < 0.85:
            issues.append(f"footprint taken from the sanctioned plan (AI outline agreed only IoU {iou:.2f})")
        b.at[i, "geometry"] = plan_fp
        b.at[i, "outline_source"] = "sanctioned_plan"
        # the outline is now the sanctioned one, so outline confidence is the plan fit's, not the photo
        # model's; an unverified height (no LiDAR) still keeps the building at most provisional
        fit_q = min(p["fit"]["iou"] for p in good)
        hc = float(np.clip(b.at[i, "height_conf"], 0.5, 1.0)) if "height_conf" in b and pd.notna(b.at[i, "height_conf"]) else 0.75
        b.at[i, "confidence"] = max(float(b.at[i, "confidence"]), round(0.95 * fit_q, 4))
        b.at[i, "confidence_3d"] = round(float(b.at[i, "confidence"]) * hc, 4)
        b.at[i, "encroach_m2"] = _encroachment(plan_fp, par) if par is not None else 0.0
        if b.at[i, "encroach_m2"]:
            issues.append(f"encroachment: {b.at[i, 'encroach_m2']:.1f} m2 beyond parcel {par.ulpin} boundary")
        b.at[i, "building_issues"] = issues
        n_follow += 1
    report["plan_outlines"] = {"buildings_following_plan": n_follow, "possible_extensions": n_ext}
    if n_follow:
        _log(f"{n_follow} buildings take their footprint from the sanctioned plan; {n_ext} built beyond it")
    return b


def _ownership(inv, units, report):
    if not inv["ownership"]:
        report["ownership"] = "no ownership records supplied"
        return
    o = pd.concat([pd.read_csv(p) for p in inv["ownership"]], ignore_index=True)
    o.columns = [c.strip().lower() for c in o.columns]
    o["key"] = o.parcel_ulpin.astype(str).str.upper() + "|" + o.unit_label.astype(str).str.upper().str.strip()
    rec = {k: r for k, r in zip(o.key, o.to_dict("records"))}
    used = set()
    for u in units:
        k = f"{u.get('parcel_ulpin')}|{str(u.get('label', '')).upper()}"
        r = rec.get(k)
        if r:
            used.add(k)
            u.update(owner_name=r.get("owner_name"), share_pct=r.get("share_pct"), right_type=r.get("right_type"),
                     document=r.get("document"))
        elif u["kind"] == "unit":
            u["owner_name"] = None
            u["issues"].append("no ownership record for this flat")
    unmatched = sorted(set(rec) - used)
    report["ownership"] = {"records": len(rec), "linked": len(used), "records_without_mapped_unit": len(unmatched),
                           "examples_unmatched": unmatched[:10]}


def _checks(units, report):
    """Topology + anomaly checks across the area."""
    from sklearn.ensemble import IsolationForest

    by = defaultdict(list)
    for u in units:
        by[(u["building_id"], u["level"], u["zclass"])].append(u)
    n_ov = 0
    for key, us in by.items():
        for a in range(len(us)):
            for c in range(a + 1, len(us)):
                ov = us[a]["geometry"].intersection(us[c]["geometry"]).area
                if ov > 0.5 and us[a]["kind"] != "whole_level" and us[c]["kind"] != "whole_level":
                    n_ov += 1
                    for u in (us[a], us[c]):
                        u["issues"].append(f"overlaps another unit on the same level by {ov:.1f} m2")
                        u["confidence"] = min(u["confidence"], 0.5)
    n_h = 0
    for u in units:
        h = u["z_max"] - u["z_min"]
        if u["kind"] != "air_rights" and not 2.4 <= h <= 6.0:
            n_h += 1
            u["issues"].append(f"storey height {h:.2f} m outside 2.4-6.0 m")
            u["confidence"] = min(u["confidence"], 0.55)
    flats = [u for u in units if u["kind"] == "unit"]
    n_iso = 0
    if len(flats) >= 20:
        X = np.array([[u["area_m2"], u["z_max"] - u["z_min"], u["geometry"].length / math.sqrt(max(u["area_m2"], 1))]
                      for u in flats])
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(X)
        s = iso.score_samples(X)
        thr = np.percentile(s, 2) if len(s) > 50 else s.min() - 1
        for u, v in zip(flats, s):
            u["anomaly_score"] = round(float(-v), 3)
            if v < thr and v < -0.6:
                n_iso += 1
                u["issues"].append("statistical outlier among flats (size / height / shape)")
    report["checks"] = {"overlapping_unit_pairs": n_ov, "bad_storey_heights": n_h, "anomaly_outliers": n_iso}


# --------------------------------------------------------------------------- main
def run_area(area, out, footprint_model=None, pc_model=None, parcel_chars=None):
    area, out = Path(area), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    inv, report = ingest(area)
    cfg = inv["cfg"]
    pchars = parcel_chars or int(cfg.get("parcel_chars", 12))
    _log(f"area '{report['name']}' grade {report['data_grade']}; missing: {report['missing'] or 'nothing'}")
    cloud = _classified_cloud(inv, out, pc_model, report) if inv["lidar"] else None
    b, lidar_fp = _buildings(inv, out, cloud, footprint_model, report)
    utm = b.crs
    parcels = _parcels(inv, cfg, utm)
    b, n_split = _split_by_parcels(b, parcels)
    if n_split:
        _log(f"{n_split} outlines spanned several plots and were split along plot lines -> {len(b)} buildings")
        report["buildings"]["split_along_plots"] = n_split
    b = _assign_parcels(b, parcels)
    surveyed = bool(inv["footprints"]) and str(cfg.get("footprints_source", "survey")).lower() == "survey"
    if not surveyed:  # surveyed outlines are trusted as they are; AI outlines are checked against LiDAR
        b, n_ref = _refine_with_lidar(b, lidar_fp, parcels)
        if n_ref:
            _log(f"{n_ref} photo outlines corrected with the LiDAR roof")
            report["buildings"]["corrected_with_lidar"] = n_ref
        for i, par in enumerate(b.parcel_ulpin):  # outline changed: re-measure encroachment
            if par and parcels is not None and b.outline_source[i] == "lidar_roof":
                e = _encroachment(b.geometry[i], parcels[parcels.ulpin == par].iloc[0])
                b.at[i, "encroach_m2"] = e
                b.at[i, "building_issues"] = [x for x in b.at[i, "building_issues"] if not x.startswith("encroachment")] + (
                    [f"encroachment: {e:.1f} m2 beyond parcel {par} boundary"] if e else [])
    probs = out / "work" / "footprints" / "probs.tif"
    from_model = str(report["buildings"].get("source", "")).startswith("orthophoto model")
    matches = _match_plans(inv, area, b, parcels, report, lidar_fp, probs if from_model and probs.exists() else None)
    b = _follow_plans(b, matches, parcels, lidar_fp, report)

    units, per_parcel = [], defaultdict(list)
    gz_all = pd.to_numeric(b.get("ground_z"), errors="coerce") if "ground_z" in b else pd.Series(dtype=float)
    area_ground = float(gz_all.median()) if gz_all.notna().any() else 0.0
    for i, row in b.iterrows():
        par = row.parcel_ulpin or f"PRV{i:09d}"[:pchars].ljust(pchars, "0")
        bld = dict(geometry=row.geometry, ground_z=row.get("ground_z"), roof_h_main=row.get("roof_h_main"),
                   floors_est=row.get("floors_est") if pd.notna(row.get("floors_est")) else None,
                   confidence_3d=row.confidence_3d)
        if bld["ground_z"] is None or pd.isna(bld["ground_z"]):
            bld["ground_z"] = area_ground  # no LiDAR returns on this outline: use the area's ground level
            b.at[i, "building_issues"] = list(row.building_issues) + ["no LiDAR points inside outline: height unknown"]
            b.at[i, "ground_z"] = area_ground
        if bld["roof_h_main"] is None or pd.isna(bld["roof_h_main"]):
            bld["roof_h_main"] = 0.0
        us, issues = build_cadastre(bld, matches.get(i, []), par, cfg.get("permissible_height_m"), parcel_chars=pchars,
                                    basements=_basements(cfg, row.building_id, par),
                                    basement_height_m=float(cfg.get("basement_height_m", 3.0)))
        b.at[i, "building_issues"] = list(b.at[i, "building_issues"]) + issues
        for u in us:
            u["building_id"] = row.building_id
            u["parcel_ulpin"] = par
            if row.encroach_m2 > 0:
                u["confidence"] = min(u["confidence"], 0.8)
                u["issues"].append("building encroaches beyond its parcel")
        per_parcel[par] += us
        units += us
    _ownership(inv, units, report)
    _checks(units, report)
    for par, us in per_parcel.items():  # IDs once per parcel, over all its buildings
        assign_ids(us, par, pchars)

    # ------------------------------------------------------------ outputs
    status = Counter(u["status"] for u in units)
    kinds = Counter(u["kind"] for u in units)
    ids = [u["ulpin3d"] for u in units if u["ulpin3d"]]
    assert len(ids) == len(set(ids)), "duplicate 3D ULPIN"
    b["n_units"] = b.building_id.map(Counter(u["building_id"] for u in units))
    b["n_unsanctioned_levels"] = b.building_id.map(Counter(u["building_id"] for u in units if u["status"] == "unsanctioned")).fillna(0)
    b["issues"] = b.building_issues.map(lambda v: "; ".join(v))
    bo = b.drop(columns=["building_issues", "roof_features"], errors="ignore").copy()
    bo.to_crs(4326).to_file(out / "buildings.geojson", driver="GeoJSON")
    urows = []
    for u in units:
        r = {k: v for k, v in u.items() if k not in ("geometry", "x", "y")}
        r["issues"] = "; ".join(dict.fromkeys(r.get("issues", [])))
        urows.append(r)
    ug = gpd.GeoDataFrame(urows, geometry=[u["geometry"] for u in units], crs=utm)
    ug.to_crs(4326).to_file(out / "units.geojson", driver="GeoJSON")
    if parcels is not None:
        pc = parcels.copy()
        pc["n_buildings"] = pc.ulpin.map(Counter(b.parcel_ulpin)).fillna(0).astype(int)
        pc["n_ids"] = pc.ulpin.map(Counter(u["parcel_ulpin"] for u in units if u["ulpin3d"])).fillna(0).astype(int)
        pc.to_crs(4326).to_file(out / "parcels.geojson", driver="GeoJSON")
    to_cityjson(units, utm.to_epsg(), out / "cadastre.city.json", "AREA", None)
    report.update(
        seconds=round(time.time() - t0, 1), crs=str(utm),
        totals={"buildings": int(len(b)), "units": len(units), "ids_issued": len(ids), "by_status": dict(status),
                "by_kind": dict(kinds),
                "unsanctioned_floors": [f"{u['building_id']} level {u['level']}" for u in units if u["status"] == "unsanctioned"],
                "encroachments": [f"{r.building_id} on {r.parcel_ulpin}: {r.encroach_m2} m2" for r in b.itertuples() if r.encroach_m2 > 0]})
    (out / "area_report.json").write_text(json.dumps(report, indent=2, default=str))
    _write_markdown(report, out / "area_report.md")
    from ulpin3d.viewer_export import export_viewer

    export_viewer(out, b, units, parcels, inv["ortho"][0] if inv["ortho"] else None, utm, report, cfg)
    from ulpin3d.card import write_cards

    n_cards = len(write_cards(out / "units.geojson", out / "cards", report.get("name") or "", pchars))
    _log(f"{n_cards} property cards -> {out / 'cards' / 'index.html'}")
    from ulpin3d.register_db import build_db

    _log(f"indexed register (SQLite, 3D R-tree) -> {build_db(out, pchars, force=True)}")
    _log(f"done in {report['seconds']} s: {len(b)} buildings, {len(units)} units, {len(ids)} IDs issued {dict(status)}")
    _log(f"open {out / 'viewer' / 'index.html'} in a browser")
    return report


def _basements(cfg, building_id, parcel):
    """`basements:` in area.yaml: a number for every building, or {building id or parcel ULPIN: levels}."""
    b = cfg.get("basements") or 0
    if isinstance(b, dict):
        return int(b.get(building_id, b.get(parcel, b.get("default", 0))) or 0)
    return int(b)


def _write_markdown(r, path):
    t = r["totals"]
    lines = [f"# Area report: {r['name']}", "",
             f"Data grade **{r['data_grade']}**; missing: {', '.join(r['missing']) or 'nothing'}; run time {r['seconds']} s.", "",
             f"- Buildings: {t['buildings']} ({r['buildings']['source']})",
             f"- Units: {t['units']}; 3D ULPINs issued: {t['ids_issued']}",
             f"- Status: " + ", ".join(f"{k} {v}" for k, v in t["by_status"].items()),
             f"- Unsanctioned floors: {', '.join(t['unsanctioned_floors']) or 'none'}",
             f"- Encroachments: {', '.join(t['encroachments']) or 'none'}",
             f"- Checks: {r.get('checks')}", f"- Ownership: {r.get('ownership')}", "", "## Plans", ""]
    for p in r.get("plans", []):
        lines.append(f"- `{p['plan']}`: {p['status']}" + (f" → {p.get('building_id')} on {p.get('parcel')} "
                                                          f"(IoU {p.get('iou')}, RMS {p.get('rms_m')} m, floors {p.get('floors')})"
                                                          if p.get("building_id") else ""))
    Path(path).write_text("\n".join(lines) + "\n")
