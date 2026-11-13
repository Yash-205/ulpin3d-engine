"""Run every example area and write one results table.

    python -m ulpin3d.cli run-examples                       # -> examples/results/<region>[/<situation>]/viewer/index.html
    python -m ulpin3d.cli run-examples --only 3_ sangam      # substring filter

Areas: every folder in `examples/` with an area.yaml (real data only; folders starting with "_" are
skipped); an area with a `scenarios:` block is run once per scenario. `test-scenarios` runs the 8
synthetic data situations instead, cut on the fly from tests/data/ward_w1 (the test suite). Each run is a separate process (clean memory, its own run.log).
Synthetic situations are graded with `check-town` against their own `expect:` block; real areas report
totals and, when they ship a truth raster (`truth/*.tif`), outline IoU against it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

DEFAULT_MODEL = "models/footprint_india.onnx"


def _real_areas(root: Path, out: Path, tmp: Path):
    """Real areas. One with a `scenarios:` block in its area.yaml is cut into one package per scenario
    (only from the data it really has), like the synthetic ward: results/<area>/<scenario>/."""
    for f in sorted(root.glob("*/area.yaml")):
        d = f.parent
        if d.name.startswith("_") or d.resolve() == out.resolve():
            continue
        cfg = yaml.safe_load(f.read_text()) or {}
        if not cfg.get("scenarios"):
            yield d.name, d
            continue
        base = {k: v for k, v in cfg.items() if k not in ("scenarios", "inputs", "footprint_model", "name")}
        for sname, sc in cfg["scenarios"].items():
            s_dir = tmp / d.name / sname
            s_dir.mkdir(parents=True)
            for sub in sc["inputs"]:
                src = d / (sc.get("footprints_from", "footprints") if sub == "footprints" else sub)
                shutil.copytree(src, s_dir / sub)
            if (d / "truth").exists():
                shutil.copytree(d / "truth", s_dir / "truth")
            child = {**base, "name": f"{cfg.get('name', d.name).split(' · ')[0]} · {sc['title']}", "inputs": sc["inputs"]}
            if sc.get("footprint_model"):
                child["footprint_model"] = sc["footprint_model"]
            (s_dir / "area.yaml").write_text(yaml.safe_dump(child, sort_keys=False, allow_unicode=True))
            yield f"{d.name}/{sname}", s_dir


def run_examples(root, out, only=None, footprint_model=DEFAULT_MODEL, python=sys.executable, base=None) -> list[dict]:
    """Real areas under `root` (grouped by area, then scenario). With `base` (the synthetic ward in
    tests/data/ward_w1) the 8 graded data situations are run instead: that is the test suite."""
    from ulpin3d.demo.check import check_town, expectations, grade
    from ulpin3d.demo.scenarios import make_scenarios

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ulpin3d_scenarios_"))
    if base:
        areas = [(d.name, d) for d in make_scenarios(Path(base), tmp)]
    else:
        areas = list(_real_areas(Path(root), out, tmp))
    results = []
    try:
        for name, d in areas:
            if only and not any(o in name for o in only):
                continue
            results.append(_run_one(name, d, out / name, footprint_model, python, check_town, expectations, grade))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    write_markdown(results, out / "RESULTS.md")
    return results


def _run_one(name, d, o, footprint_model, python, check_town, expectations, grade):
    cfg = yaml.safe_load((d / "area.yaml").read_text()) or {}
    if o.exists():
        shutil.rmtree(o)
    o.mkdir(parents=True)
    cmd = [python, "-m", "ulpin3d.cli", "run-area", "--area", str(d), "--out", str(o)]
    has_ortho = (d / "ortho").exists() and any((d / "ortho").iterdir())
    model = cfg.get("footprint_model") or (footprint_model if has_ortho and not (d / "footprints").exists()
                                           and cfg.get("footprint_model", "") is not None else None)
    if model:
        cmd += ["--footprint-model", model]
    print(f"[examples] {name}", flush=True)
    t0 = time.time()
    with open(o / "run.log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    row = {"area": name, "title": cfg.get("name"), "inputs": cfg.get("inputs"), "model": model,
           "seconds": round(time.time() - t0, 1), "ok_run": rc == 0}
    if rc == 0:
        rep = json.loads((o / "area_report.json").read_text())
        row["totals"] = {k: rep["totals"][k] for k in ("buildings", "units", "ids_issued", "by_status")}
        row["grade"] = rep.get("data_grade")
        if (d / "truth.json").exists():
            checks = grade(check_town(d, o), expectations(d))
            row["checks"] = checks
            row["verdict"] = "FAIL" if any(c[2] == "FAIL" for c in checks) else "PASS"
        else:
            tr = sorted((d / "truth").glob("*.tif")) if (d / "truth").exists() else []
            if tr:
                from ulpin3d.demo.real_area import outline_iou

                row["outline_vs_truth"] = outline_iou(o / "buildings.geojson", tr[0])
            row["verdict"] = "RAN"
        shutil.rmtree(o / "work", ignore_errors=True)  # intermediate rasters / point clouds: not needed to view a result
    else:
        row["verdict"] = "ERROR"
    print(f"[examples]   -> {row['verdict']} in {row['seconds']} s  ({o / 'viewer' / 'index.html'})", flush=True)
    return row


INPUT_NAMES = {"ortho": "drone/aerial photo", "lidar": "LiDAR", "plans": "plans", "footprints": "official building outlines",
               "parcels": "parcels", "ownership": "ownership"}


def _short(v, verdict):
    """'1/1 (false alarms 0)' -> '1/1'; impossible checks -> '—'."""
    return "—" if verdict == "n/a" else v.split(" (")[0]


def scenario_table(results) -> list[str]:
    """One row per graded area: the headline checks, readable at a glance."""
    rows = [r for r in results if r.get("checks")]
    if not rows:
        return []
    L = ["| Inputs | Outlines correct | Floors | Plans on right plot | Extra floor found | Encroachment found | "
         "Flats · owners | 3D IDs | Verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = {name: (val, v) for name, val, v in r["checks"]}
        ins = [INPUT_NAMES[k] for k in ("footprints", "ortho", "lidar", "plans") if k in (r.get("inputs") or [])]
        missing = [INPUT_NAMES[k] for k in ("ortho", "lidar", "plans") if k not in (r.get("inputs") or [])]
        label = " + ".join(ins) + (f", no {', no '.join(missing)}" if missing and len(ins) > 1 else (" only" if len(ins) == 1 else ""))
        if r["area"].split("/")[-1].startswith("8_"):
            label += " (no permit register: plans matched by shape)"
        acc_val, acc_v = c["outlines accurate (IoU >= 0.8)"]
        n_ok, n = acc_val.split("/")
        acc = acc_val if n_ok == n else f"{acc_val}, the other {int(n) - int(n_ok)} get no ID"
        fl_val, fl_v = c["flats cut from plans"]
        own_val, _ = c["owners linked to flats"]
        flats = f"{fl_val.split('/')[0]} · {own_val.split('/')[0]}" if fl_v == "PASS" and int(fl_val.split("/")[0]) else "whole floors"
        L.append(f"| {label} | {acc} | {_short(*c['floor count exact'])} | {_short(*c['plans on the right plot'])} | "
                 f"{_short(*c['unsanctioned floor found'])} | {_short(*c['encroachment found'])} | {flats} | "
                 f"{r['totals']['ids_issued']} | **{r['verdict']}** |")
    return L + ["", "\"—\" = impossible with those inputs, not a failure (e.g. an extra floor cannot be seen without "
                "height data). Found counts have no false alarms unless the details below say so. Every area also checks "
                "that no 3D ULPIN is issued on an inaccurate outline.", ""]


def real_table(results) -> list[str]:
    """Real areas, grouped by area: what each data combination produced and, where an official
    outline layer exists, how well our outlines match it."""
    rows = [r for r in results if not r.get("checks")]
    if not rows:
        return []
    L = ["| Area | Scenario | Inputs | Buildings | 3D units | IDs issued | Outlines vs official map (IoU · precision · recall) |",
         "|---|---|---|---|---|---|---|"]
    last = None
    for r in rows:
        area, _, scen = r["area"].partition("/")
        t = r.get("totals", {})
        o = r.get("outline_vs_truth")
        ov = f"{o['iou']:.2f} · {o['precision']:.2f} · {o['recall']:.2f}" if o else "no official map open"
        L.append(f"| {'**' + area + '**' if area != last else ''} | [{scen or '—'}]({r['area']}/viewer/index.html) | "
                 f"{', '.join(INPUT_NAMES.get(i, i) for i in (r.get('inputs') or []))} | {t.get('buildings', 'ERROR')} | "
                 f"{t.get('units', '—')} | {t.get('ids_issued', '—')} | {ov} |")
        last = area
    return L + ["", "IoU 1.0 = our building outlines cover exactly the official outlines. IDs are withheld (flagged) "
                "wherever the evidence is too weak, so fewer IDs on poorer data is the system working as intended.", ""]


def write_markdown(results, path):
    graded = any(r.get("checks") for r in results)
    L = ["# Results", "", "Generated by `python -m ulpin3d.cli " + ("test-scenarios`." if graded else "run-examples`."), ""]
    if graded:
        L += ["Each situation is graded against what its inputs make possible (`expect:` in its area.yaml).", "",
              "## Data situations at a glance", ""] + scenario_table(results)
    L += real_table(results) + ["## Details", ""]
    for r in results:
        L += ["", f"### `{r['area']}`: {r.get('title') or ''}", ""]
        t = r.get("totals", {})
        if t:
            L.append(f"{t.get('buildings')} buildings, {t.get('units')} 3D units, {t.get('ids_issued')} IDs issued, "
                     f"status {t.get('by_status')}. Run time {r['seconds']} s. [Open in 3D]({r['area']}/viewer/index.html)")
            L.append("")
        if r.get("model"):
            L += [f"Footprint model: `{r['model']}`", ""]
        if r.get("checks"):
            L += ["| Check | Result | |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in r["checks"]]
        if r.get("outline_vs_truth"):
            L.append(f"Outlines vs the official map: {r['outline_vs_truth']}")
        if not r["ok_run"]:
            L.append("Run failed: see run.log")
    Path(path).write_text("\n".join(L) + "\n")
