"""Fuse Model 1 (orthophoto footprints) with Model 2 (classified LiDAR) into 3D building
records: the hand-off to floor/unit generation, validation and 3D ULPIN assignment.

    python -m ulpin3d.cli fuse --buildings out/site/buildings.gpkg \
        --laz out/site/classified.laz --out out/site/buildings_3d.geojson

Per building:
    footprint (from the photo: sharp edges, party walls split)
    ground_z, roof_h_main, roof_levels, floors_est [floors_low..floors_high]   (from LiDAR)
    lod1 extrusion: z_min = ground_z, z_max = ground_z + roof_h_main
    confidence = footprint_conf * height_factor, where height_factor drops when
                 LiDAR coverage is thin or the roof is very uneven
    status validated / provisional / flagged (same thresholds as Model 1)
LiDAR-only buildings with no photo footprint (e.g. hidden under trees) are added
with status 'flagged' and source 'lidar_only'.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np

from ulpin3d.pointcloud.buildings import attach_heights, buildings_from_lidar
from ulpin3d.pointcloud.data import TO_ASPRS
from ulpin3d.pointcloud.ground import height_above_ground
from ulpin3d.pointcloud.io import read_points

ASPRS_TO_UNIFIED = {int(v): i for i, v in enumerate(TO_ASPRS)}


def fuse(buildings_path, laz_path, out_path, conf_validated=0.85, conf_provisional=0.60,
         min_lidar_only_area=20.0):
    fp = gpd.read_file(buildings_path)
    pts = read_points(laz_path)
    if pts["classification"] is None:
        raise ValueError("point cloud has no classification - run pc-infer first")
    lut = np.full(256, 3, np.int16)
    for k, v in ASPRS_TO_UNIFIED.items():
        lut[k] = v
    labels = lut[pts["classification"]]
    hag, *_ = height_above_ground(pts["xyz"])
    if fp.crs is not None and pts["crs"] is not None and fp.crs != pts["crs"]:
        fp = fp.to_crs(pts["crs"])
    fp = attach_heights(fp, pts["xyz"], hag, labels)
    hf = fp["height_conf"].fillna(0.5).clip(0.5, 1.0)
    fp["source"] = np.where(fp["n_points"].notna(), "photo+lidar", "photo")
    fp["confidence_3d"] = (fp["confidence"] * hf).round(4)
    fp["z_min"] = fp["ground_z"]
    fp["z_max"] = fp["ground_z"] + fp["roof_h_main"]
    # buildings LiDAR sees but the photo model missed
    lo = buildings_from_lidar(pts["xyz"], hag, labels, crs=fp.crs)
    if len(lo):
        lo = lo[lo.area >= min_lidar_only_area]
        hit = gpd.sjoin(lo, fp[["geometry"]], how="left", predicate="intersects")
        lo = lo[~lo.index.isin(hit[hit.index_right.notna()].index)].copy()
        if len(lo):
            lo["source"] = "lidar_only"
            lo["confidence"] = np.nan
            lo["confidence_3d"] = (0.5 * lo["height_conf"].fillna(0.5)).round(4)
            lo["z_min"], lo["z_max"] = lo["ground_z"], lo["ground_z"] + lo["roof_h_main"]
            lo["building_id"] = [f"LIDAR-B{i + 1:05d}" for i in range(len(lo))]
            fp = gpd.GeoDataFrame(np.concatenate([fp.values, lo.reindex(columns=fp.columns).values]),
                                  columns=fp.columns, geometry="geometry", crs=fp.crs)
    c = fp["confidence_3d"].astype(float)
    fp["status"] = np.where(c >= conf_validated, "validated", np.where(c >= conf_provisional, "provisional", "flagged"))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fp.to_crs(4326).to_file(out, driver="GeoJSON")
    print(f"[fuse] {len(fp)} buildings -> {out}  {fp.status.value_counts().to_dict()}  sources {fp.source.value_counts().to_dict()}")
    return fp
