"""Build a run-area test folder from a real Indian drone scene (no synthetic data).

    python -m ulpin3d.cli fetch-area --title "Sangam Vihar" --size 250 --out data/real/sangam_vihar

Downloads a window of an OpenAerialMap drone orthophoto (CC-BY 4.0) and the Google /
Microsoft open building outlines for the same window, and writes:

    <out>/area.yaml
    <out>/ortho/ortho.tif             the drone window, UTM, ~5 cm
    <out>/footprints/open_buildings.geojson   building outlines to use until Model 1 is trained

There is no LiDAR, no parcel map, no plans and no ownership for these places, so the
run is data grade C: single-level buildings with provisional parcel IDs. That is the
point of the test: the pipeline must cope with whatever subset of data it is given.
The open outlines are often shifted 2-5 m against drone imagery; you will see it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
import yaml
from rasterio.transform import from_origin
from shapely.geometry import box

from ulpin3d.common.raster import MetricReader
from ulpin3d.data.sources import vida_labels

OAM_API = "https://api.openaerialmap.org/meta"
INDIA_BBOX = "68.1,6.5,97.4,35.7"


def find_scene(title: str):
    page = 1
    while page < 15:
        r = requests.get(OAM_API, params={"bbox": INDIA_BBOX, "limit": 100, "page": page}, timeout=120).json()
        if not r["results"]:
            break
        for x in r["results"]:
            if title.lower() in (x.get("title") or "").lower() and str(x.get("uuid", "")).lower().endswith((".tif", ".tiff")):
                return x
        page += 1
    raise SystemExit(f"no OpenAerialMap scene with '{title}' in its title over India")


def fetch_area(title: str, out, size_m: float = 250.0, gsd: float = 0.08, min_confidence: float = 0.7):
    out = Path(out)
    x = find_scene(title)
    print(f"[fetch-area] scene '{x['title']}' by {x.get('provider')} ({x.get('acquisition_start', '')[:10]}), "
          f"gsd {x.get('gsd', 0):.3f} m")
    with MetricReader(x["uuid"], gsd) as rd:
        L, B, R, T = rd.bounds
        cx, cy = (L + R) / 2, (B + T) / 2
        half = min(size_m / 2, (R - L) / 2, (T - B) / 2)
        x0, y1 = cx - half, cy + half
        n = int(2 * half / gsd)
        rgb, valid = rd.read(x0, y1, n)
        crs = rd.crs
    (out / "ortho").mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs=crs, transform=from_origin(x0, y1, gsd, gsd),
                tiled=True, blockxsize=512, blockysize=512, compress="jpeg", photometric="YCBCR")
    with rasterio.open(out / "ortho" / "ortho.tif", "w", **prof) as dst:
        dst.write(rgb.transpose(2, 0, 1))
        dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
    win = gpd.GeoSeries([box(x0, y1 - 2 * half, x0 + 2 * half, y1)], crs=crs)
    lonlat = win.to_crs(4326).total_bounds
    fp = vida_labels(tuple(lonlat), "IND", min_confidence)
    fp = fp.to_crs(crs)
    fp = fp[fp.geometry.within(win.iloc[0])]
    fp = fp[fp.area > 12]
    (out / "footprints").mkdir(parents=True, exist_ok=True)
    fp["confidence"] = fp["confidence"].fillna(0.75) if "confidence" in fp else 0.75
    fp[["geometry", "confidence"]].to_crs(4326).to_file(out / "footprints" / "open_buildings.geojson", driver="GeoJSON")
    (out / "area.yaml").write_text(yaml.safe_dump({
        "name": f"{x['title']} (real drone imagery, OpenAerialMap)", "synthetic": False, "parcel_chars": 12,
        "source": {"imagery": f"OpenAerialMap, {x.get('provider')}, CC-BY 4.0", "outlines": "Google/Microsoft Open Buildings (VIDA)"},
    }, sort_keys=False))
    print(f"[fetch-area] {2 * half:.0f} m window, {len(fp)} building outlines -> {out}")
    return out


def lidar_area(key: str, out, size_m: float = 250.0, gsd: float = 0.15, laz_dir="data/pointcloud/laz"):
    """A real LiDAR + image package from a Sao Paulo tile (default: a held-out test tile).

    ortho/ortho.tif     true orthophoto made from the colour of the top LiDAR point (0.15 m)
    lidar/cloud.laz     the cloud with its classes REMOVED (as an unclassified delivery would be)
    truth/buildings_from_city_classes.tif   the city's own building class, for checking
    """
    import laspy

    from ulpin3d.pointcloud.data import SP_BUCKET, center_crop, download, saopaulo_tile_index
    from ulpin3d.pointcloud.rasterize import rasterize_cloud, write_raster

    out = Path(out)
    laz = Path(laz_dir) / key
    if not laz.exists():
        download(f"{SP_BUCKET}/{key}", laz)
    idx = saopaulo_tile_index(Path(laz_dir).parent / "cache")
    row = idx[idx.key == key].iloc[0]
    crop = center_crop(row, size_m)
    from pyproj import CRS

    r = rasterize_cloud(laz, gsd, crop)
    crs = r["crs"] or CRS.from_epsg(31983)  # PMSP publishes in SIRGAS 2000 / UTM 23S; some tiles omit the tag
    write_raster(out / "ortho" / "ortho.tif", r["rgb"], r["transform"], crs)
    write_raster(out / "truth" / "buildings_from_city_classes.tif", r["label"].astype(np.uint8), r["transform"], crs)
    src = laspy.read(str(laz))
    x, y = np.asarray(src.x), np.asarray(src.y)
    m = (x >= crop[0]) & (y >= crop[1]) & (x <= crop[2]) & (y <= crop[3])
    las = laspy.create(point_format=src.header.point_format.id, file_version="1.2")  # fresh header
    las.header.offsets = np.asarray(src.header.offsets)
    las.header.scales = np.asarray(src.header.scales)
    las.header.add_crs(CRS.from_user_input(crs))
    las.x, las.y, las.z = np.asarray(src.x)[m], np.asarray(src.y)[m], np.asarray(src.z)[m]
    for d in ("intensity", "return_number", "number_of_returns", "red", "green", "blue", "gps_time"):
        if d in src.point_format.dimension_names and d in las.point_format.dimension_names:
            setattr(las, d, np.asarray(getattr(src, d))[m])
    las.classification = np.ones(int(m.sum()), np.uint8)
    (out / "lidar").mkdir(parents=True, exist_ok=True)
    las.write(str(out / "lidar" / "cloud.laz"))
    (out / "area.yaml").write_text(yaml.safe_dump({
        "name": f"Sao Paulo {key.replace('MDS_color_', '').replace('.laz', '')} (real LiDAR, held-out tile)",
        "synthetic": False, "parcel_chars": 12,
        "source": {"lidar": "Prefeitura de Sao Paulo / GeoSampa 2017 LiDAR (classes removed)",
                   "ortho": "true orthophoto rendered from the LiDAR point colours"}}, sort_keys=False))
    print(f"[fetch-area-lidar] {int(m.sum()):,} points, {size_m:.0f} m window -> {out}")
    return out


def outline_iou(buildings_path, truth_tif) -> dict:
    """Pixel IoU / precision / recall of building outlines against a truth raster (1 = building)."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(truth_tif) as t:
        truth, tr, crs = t.read(1) > 0, t.transform, t.crs
    b = gpd.read_file(buildings_path).to_crs(crs)
    pred = rasterize([(g, 1) for g in b.geometry], out_shape=truth.shape, transform=tr).astype(bool)
    i, u = (pred & truth).sum(), (pred | truth).sum()
    return {"iou": round(i / max(u, 1), 3), "precision": round(i / max(pred.sum(), 1), 3),
            "recall": round(i / max(truth.sum(), 1), 3), "buildings": int(len(b))}
