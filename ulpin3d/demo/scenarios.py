"""One test area per data situation the judges (or a real district) might hand over.

    python -m ulpin3d.cli test-scenarios                # cuts these from tests/data/ward_w1 on the fly, runs, grades
    python -m ulpin3d.cli make-examples --out out/scen  # only to look at the generated input folders

All synthetic scenarios are cut from the same ward (`ward_w1`, known right answer in truth.json),
so the only thing that changes between them is which inputs exist. Each scenario's area.yaml
carries an `expect:` block: what the system *should* be able to find with those inputs (an
extra floor cannot be seen without heights; flats cannot be cut without a plan). `check-town`
grades the run against that block, so every scenario has an honest PASS / FAIL.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import shape

# name: (inputs kept, title, expectations)
#   inputs: ortho, lidar, plans, footprints (surveyed outlines from municipal GIS), parcels + ownership always
FULL = dict(buildings=True, footprint_iou_min=0.9, floors=True, plans=True, unsanctioned=True, encroachments=True,
            flats=True, owners=True)
SCENARIOS = {
    "1_drone_lidar_plans": (
        {"ortho", "lidar", "plans"},
        "Drone + LiDAR + sanctioned plans (complete survey package)",
        FULL),
    "2_drone_lidar_no_plans": (
        {"ortho", "lidar"},
        "Drone + LiDAR, no sanctioned plans (unauthorised / old construction)",
        dict(buildings=True, footprint_iou_min=0.9, floors=True, plans=None, unsanctioned=None, encroachments=True,
             flats=False, owners=None)),
    "3_drone_plans_no_lidar": (
        {"ortho", "plans"},
        "Drone + sanctioned plans, no LiDAR (most common in practice: drone-only survey + permit register)",
        dict(buildings=True, footprint_iou_min=0.8, floors=None, plans=True, unsanctioned=None, encroachments=True,
             flats=True, owners=True)),
    "4_drone_only": (
        {"ortho"},
        "Drone imagery only (SVAMITVA-style orthophoto, nothing else)",
        dict(buildings=True, accurate=None, footprint_iou_min=None, floors=None, plans=None, unsanctioned=None,
             encroachments=True, flats=False, owners=None)),
    "5_lidar_plans_no_drone": (
        {"lidar", "plans"},
        "LiDAR + sanctioned plans, no drone photo",
        FULL),
    "6_lidar_only": (
        {"lidar"},
        "LiDAR only, no drone photo, no plans",
        dict(buildings=True, footprint_iou_min=0.9, floors=True, plans=None, unsanctioned=None, encroachments=True,
             flats=False, owners=None)),
    "7_survey_outlines_plans": (
        {"footprints", "plans"},
        "No drone, no LiDAR: municipal building outlines + sanctioned plans (desk-based update)",
        dict(buildings=True, footprint_iou_min=0.9, floors=None, plans=True, unsanctioned=None, encroachments=True,
             flats=True, owners=True)),
    "8_plans_matched_by_shape": (
        {"ortho", "lidar", "plans", "no_permit_register"},
        "Drone + LiDAR + plans, but no permit register: each plan is matched to its building by shape alone",
        FULL),
}


def make_scenarios(base, out, names=None) -> list[Path]:
    base, out = Path(base), Path(out)
    truth = json.loads((base / "truth.json").read_text())
    cfg0 = yaml.safe_load((base / "area.yaml").read_text()) or {}
    made = []
    for name, (keep, title, expect) in SCENARIOS.items():
        if names and name not in names:
            continue
        d = out / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for sub in ("parcels", "ownership", *(k for k in ("ortho", "lidar", "plans") if k in keep)):
            shutil.copytree(base / sub, d / sub)
        if "plans" in keep and "no_permit_register" not in keep:  # the building-permission register: which plan was sanctioned for which plot
            pd.DataFrame([{"file": p["file"], "parcel_ulpin": p["parcel"]} for p in truth["plans"]]).to_csv(
                d / "plans" / "plans_index.csv", index=False)
        if "footprints" in keep:  # municipal GIS layer: the as-built outlines, drawn by a surveyor
            (d / "footprints").mkdir()
            g = gpd.GeoDataFrame({"parcel": [b["parcel"] for b in truth["buildings"]]},
                                 geometry=[shape(b["polygon_utm"]) for b in truth["buildings"]], crs=truth["epsg"])
            g.to_file(d / "footprints" / "municipal_buildings.geojson", driver="GeoJSON")
        shutil.copy(base / "truth.json", d / "truth.json")
        cfg = {**cfg0, "name": f"Ward W1 · {title}", "scenario": name,
               "inputs": sorted((keep - {"no_permit_register"}) | {"parcels", "ownership"}), "expect": expect}
        (d / "area.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        made.append(d)
    return made
