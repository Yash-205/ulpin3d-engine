"""Write the 3D cadastre viewer next to a run-area result.

    <out>/viewer/index.html        the viewer (Three.js, opens from disk, no server)
    <out>/viewer/cadastre_data.js  window.CADASTRE = {...}  (loaded by index.html)
    <out>/viewer/cadastre_view.json  same data, for "Load area" in the viewer

Coordinates are local metres around the area centre (x east, y north), heights in
metres above the lowest ground point; the true CRS origin and base height are in
`meta`, so nothing is lost. The orthophoto is embedded as a JPEG data URI so the
page also works from file:// (browsers refuse local image files as WebGL textures).
"""
from __future__ import annotations

import base64
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE = Path(__file__).parent / "viewer" / "index.html"


def _ring(poly, ox, oy):
    if poly is None or poly.is_empty:
        return []
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return [[round(x - ox, 2), round(y - oy, 2)] for x, y in list(poly.exterior.coords)[:-1]]


def _ortho_png(path, utm, bounds, max_px=2048):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.transform import from_bounds
    from PIL import Image

    L, B, R, T = bounds
    scale = max((R - L), (T - B)) / max_px
    w, h = int((R - L) / scale), int((T - B) / scale)
    with rasterio.open(path) as src, WarpedVRT(src, crs=utm, transform=from_bounds(L, B, R, T, w, h),
                                                width=w, height=h, resampling=Resampling.average) as v:
        a = v.read([1, 2, 3]).transpose(1, 2, 0)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_view(b, units, parcels, ortho, utm, report, cfg):
    geoms = list(b.geometry) + ([g for g in parcels.geometry] if parcels is not None else [])
    xs = np.concatenate([np.array(g.bounds)[[0, 2]] for g in geoms])
    ys = np.concatenate([np.array(g.bounds)[[1, 3]] for g in geoms])
    pad = 6.0
    L, B, R, T = xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad
    ox, oy = (L + R) / 2, (B + T) / 2
    # stand every building on the (flat) orthophoto: heights are drawn relative to each
    # building's own ground; true elevations stay in `zabs` for the panel
    gmap = {r.building_id: (float(r.ground_z) if "ground_z" in b and pd.notna(r.ground_z) else None) for r in b.itertuples()}
    zs = [u["z_min"] for u in units if u.get("z_min") is not None]
    fallback = float(np.median(zs)) if zs else 0.0
    gnd = lambda u: gmap.get(u.get("building_id")) if gmap.get(u.get("building_id")) is not None else fallback
    base = float(min(gmap.values(), default=fallback) or fallback) if any(v is not None for v in gmap.values()) else fallback
    f = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else v
    view = {
        "meta": {"name": report.get("name"), "synthetic": bool(cfg.get("synthetic")), "crs": str(utm),
                 "origin": [round(ox, 3), round(oy, 3)], "base_z": round(base, 3),
                 "bounds": [round(L - ox, 2), round(B - oy, 2), round(R - ox, 2), round(T - oy, 2)],
                 "grade": report.get("data_grade"), "totals": report.get("totals"),
                 "building_source": report.get("buildings", {}).get("source"),
                 "permissible_height_m": cfg.get("permissible_height_m"), "parcel_chars": int(cfg.get("parcel_chars", 12))},
        "ortho": _ortho_png(ortho, utm, (L, B, R, T)) if ortho else None,
        "parcels": [{"ulpin": r.ulpin, "poly": _ring(r.geometry, ox, oy),
                     "area": round(r.geometry.area, 1)} for r in parcels.itertuples()] if parcels is not None else [],
        "buildings": [{"id": r.building_id, "parcel": r.parcel_ulpin, "poly": _ring(r.geometry, ox, oy),
                       "floors": f(getattr(r, "floors_est", None)), "conf": round(float(r.confidence_3d), 3),
                       "issues": r.issues, "encroach": float(r.encroach_m2)} for r in b.itertuples()],
        "units": [],
    }
    for u in units:
        view["units"].append({
            "id": u.get("ulpin3d"), "cand": u.get("ulpin3d_candidate"), "label": u.get("label"), "kind": u["kind"],
            "level": int(u["level"]), "zc": u["zclass"], "z0": round(u["z_min"] - gnd(u), 2), "z1": round(u["z_max"] - gnd(u), 2),
            "zabs": [round(u["z_min"], 2), round(u["z_max"], 2)], "poly": _ring(u["geometry"], ox, oy),
            "status": u["status"], "conf": u["confidence"], "carpet": f(u.get("carpet_area_m2")), "area": u.get("area_m2"),
            "vol": u.get("volume_m3"), "bhk": f(u.get("bhk")), "owner": f(u.get("owner_name")), "share": f(u.get("share_pct")),
            "right": f(u.get("right_type")), "doc": f(u.get("document")), "issues": list(dict.fromkeys(u.get("issues", []))),
            "bid": u.get("building_id"), "parcel": u.get("parcel_ulpin"), "source": u.get("source")})
    return view


def export_viewer(out, b, units, parcels, ortho, utm, report, cfg):
    vd = Path(out) / "viewer"
    vd.mkdir(parents=True, exist_ok=True)
    view = build_view(b, units, parcels, ortho, utm, report, cfg)
    js = json.dumps(view, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (vd / "cadastre_data.js").write_text("window.CADASTRE=" + js + ";\n")
    (vd / "cadastre_view.json").write_text(js)
    shutil.copy(TEMPLATE, vd / "index.html")
    vend = TEMPLATE.parent / "vendor"
    if vend.exists():  # offline copy of the 3D library
        shutil.copytree(vend, vd / "vendor", dirs_exist_ok=True)
    return vd
