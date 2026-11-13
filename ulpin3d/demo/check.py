"""Compare a run-area result with the synthetic town's truth.json.

`check_town` measures; `grade` turns the measurements into PASS / FAIL lines against the area's
`expect:` block (area.yaml). Without an `expect:` block the full survey package is assumed.
An expectation of None means "not possible with these inputs" and is reported as n/a.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import shape

FULL = dict(buildings=True, footprint_iou_min=0.9, floors=True, plans=True, unsanctioned=True, encroachments=True,
            flats=True, owners=True)


def check_town(area_dir, out_dir) -> dict:
    t = json.loads((Path(area_dir) / "truth.json").read_text())
    rep = json.loads((Path(out_dir) / "area_report.json").read_text())
    b = gpd.read_file(Path(out_dir) / "buildings.geojson").to_crs(t["epsg"])
    u = gpd.read_file(Path(out_dir) / "units.geojson")
    res = {"buildings_true": len(t["buildings"]), "buildings_found": len(b)}
    ious, floors_ok, matched, detected, accurate, weak_with_id = [], 0, 0, 0, 0, 0
    fl = pd.to_numeric(b["floors_est"], errors="coerce") if "floors_est" in b else pd.Series([None] * len(b), dtype=float)
    with_id = set(u[u.ulpin3d.notna() & u.kind.isin(["unit", "whole_level"])].building_id)
    for tb in t["buildings"]:
        g = shape(tb["polygon_utm"])
        iou = (b.geometry.intersection(g).area / b.geometry.union(g).area)
        j = int(iou.values.argmax()) if len(b) else 0
        best = float(iou.values[j]) if len(b) else 0.0
        detected += best > 0.1
        accurate += best >= 0.8
        if best > 0.1 and best < 0.8 and b.building_id.iloc[j] in with_id:
            weak_with_id += 1  # a wrong outline was given a 3D ULPIN: the one mistake a register must not make
        if best > 0.5:
            matched += 1
            ious.append(best)
            floors_ok += bool(pd.notna(fl.iloc[j]) and int(fl.iloc[j]) == tb["levels"])
    n = len(t["buildings"])
    res.update(buildings_detected=detected, buildings_matched=matched, outlines_accurate=f"{accurate}/{n}",
               weak_outlines_given_id=weak_with_id,
               footprint_iou_mean=round(sum(ious) / max(len(ious), 1), 3), floors_exact=f"{floors_ok}/{n}")
    plans_ok = sum(1 for p in rep.get("plans", []) for tp in t["plans"] if p["plan"] == tp["file"] and p.get("parcel") == tp["parcel"])
    res["plans_on_right_parcel"] = f"{plans_ok}/{len(t['plans'])}"
    exp_uns = {(tb["parcel"], lv) for tb in t["buildings"] for lv in tb["unsanctioned_levels"]}
    got_uns = {(r.parcel_ulpin, int(r.level)) for r in u[u.status == "unsanctioned"].itertuples()}
    res["unsanctioned_found"] = f"{len(exp_uns & got_uns)}/{len(exp_uns)} (false alarms {len(got_uns - exp_uns)})"
    exp_enc = {tb["parcel"] for tb in t["buildings"] if tb["encroach"]}
    got_enc = set(b[b.encroach_m2 > 0].parcel_ulpin)
    res["encroachments_found"] = f"{len(exp_enc & got_enc)}/{len(exp_enc)} (false alarms {len(got_enc - exp_enc)})"
    flats = u[u.kind == "unit"]
    res["flats_true"] = t["n_flats"]
    res["flats_found"] = int(len(flats))
    own = rep.get("ownership", {})
    res["owners_linked"] = f"{own.get('linked', 0)}/{own.get('records', 0)}"
    ids = u.ulpin3d.dropna()
    res["ids_issued"] = int(len(ids))
    res["ids_unique"] = bool(ids.is_unique)
    return res


def _full(s):  # "3/4 (false alarms 0)" -> all found and no false alarms
    a, rest = s.split("/", 1)
    n = rest.split(" ")[0]
    return a == n and ("false alarms" not in s or "false alarms 0" in s)


def expectations(area_dir) -> dict:
    f = Path(area_dir) / "area.yaml"
    cfg = yaml.safe_load(f.read_text()) if f.exists() else {}
    return {**FULL, **((cfg or {}).get("expect") or {})}


def grade(r: dict, expect: dict) -> list[tuple[str, str, str]]:
    """[(check, result, PASS|FAIL|n/a)] for the expectations that apply."""
    rows = []

    def add(name, value, want, ok):
        rows.append((name, str(value), "n/a" if want is None else ("PASS" if ok else "FAIL")))

    e = expect
    add("buildings detected", f"{r['buildings_detected']}/{r['buildings_true']} ({r['buildings_found']} outlines)",
        e.get("buildings"), r["buildings_detected"] == r["buildings_true"] == r["buildings_found"])
    add("outlines accurate (IoU >= 0.8)", r["outlines_accurate"], e.get("accurate", True), _full(r["outlines_accurate"]))
    m = e.get("footprint_iou_min")
    add(f"mean footprint IoU{f' >= {m}' if m else ''}", r["footprint_iou_mean"], m, m is not None and r["footprint_iou_mean"] >= m)
    add("no 3D ID on an inaccurate outline", f"{r['weak_outlines_given_id']} given an ID", True, r["weak_outlines_given_id"] == 0)
    add("floor count exact", r["floors_exact"], e.get("floors"), _full(r["floors_exact"]))
    add("plans on the right plot", r["plans_on_right_parcel"], e.get("plans"), _full(r["plans_on_right_parcel"]))
    add("unsanctioned floor found", r["unsanctioned_found"], e.get("unsanctioned"), _full(r["unsanctioned_found"]))
    add("encroachment found", r["encroachments_found"], e.get("encroachments"), _full(r["encroachments_found"]))
    want_flats = e.get("flats")
    add("flats cut from plans", f"{r['flats_found']}/{r['flats_true']}", want_flats,
        r["flats_found"] == (r["flats_true"] if want_flats else 0))
    add("owners linked to flats", r["owners_linked"], e.get("owners"), _full(r["owners_linked"]))
    add("3D ULPINs unique", f"{r['ids_issued']} issued, unique={r['ids_unique']}", True, r["ids_unique"] and r["ids_issued"] > 0)
    return rows
