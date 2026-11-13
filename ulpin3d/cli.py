"""Command line for the 3D ULPIN engine.

    python -m ulpin3d.cli catalog        --config configs/data_footprint.yaml
    python -m ulpin3d.cli labels         --config configs/data_footprint.yaml
    python -m ulpin3d.cli register       --config configs/data_footprint.yaml --checkpoint runs/stage1/best.pt
    python -m ulpin3d.cli chips          --config configs/data_footprint.yaml
    python -m ulpin3d.cli train-footprint --config configs/footprint_stage1_pretrain.yaml
    python -m ulpin3d.cli eval-footprint  --checkpoint runs/stage2_india/best.pt --chips data/chips --split test
    python -m ulpin3d.cli infer-footprint --image ortho.tif --checkpoint runs/stage2_india/best.pt --out out/
    python -m ulpin3d.cli synthetic       --out data/synthetic --n 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def _cfg(p):
    return yaml.safe_load(Path(p).read_text())


def cmd_catalog(a):
    from ulpin3d.data.sources import assign_splits, oam_catalog, open_cities_catalog

    c = _cfg(a.config)
    dc = c["catalog"]
    work = Path(c["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    if dc.get("open_cities", {}).get("enabled", True):
        oc = open_cities_catalog(tuple(dc["open_cities"].get("tiers", ["train_tier_1"])))
        cities = dc["open_cities"].get("cities")
        if cities:
            oc = oc[oc.city.isin(cities)]
        parts.append(oc)
        print(f"[catalog] open cities: {len(oc)} scenes")
    if dc.get("oam", {}).get("enabled", True):
        o = dc["oam"]
        oam = oam_catalog(work / "cache", max_gsd=o.get("max_gsd", 0.3),
                          countries=tuple(o.get("countries", ["India", "Bangladesh", "Nepal", "Sri Lanka"])))
        if o.get("exclude_title_regex"):
            oam = oam[~oam.title.str.contains(o["exclude_title_regex"], case=False, regex=True, na=False)]
        if o.get("max_size_mb"):
            oam = oam[(oam.size_mb <= o["max_size_mb"]) | (oam.region == "india")]
        parts.append(oam)
        print(f"[catalog] OpenAerialMap: {len(oam)} scenes\n{oam.country.value_counts().to_string()}")
    cat = pd.concat(parts, ignore_index=True)
    cat = assign_splits(cat, dc.get("test_title_regex"), dc.get("val_frac", 0.1),
                        dc.get("test_exclusion_buffer_m", 1000))
    cat.to_csv(work / "catalog.csv", index=False)
    print(cat.groupby(["region", "split"]).size().unstack(fill_value=0).to_string())
    print(f"[catalog] -> {work / 'catalog.csv'}")


def cmd_labels(a):
    from concurrent.futures import ThreadPoolExecutor

    from ulpin3d.data.sources import fetch_labels

    c = _cfg(a.config)
    work = Path(c["work_dir"])
    cat = pd.read_csv(work / "catalog.csv")
    cat = cat[cat.split.isin(["train", "val", "test"])]
    out = work / "labels"

    def one(r):
        try:
            p = fetch_labels(r, out, c.get("labels", {}).get("min_confidence", 0.70))
            import geopandas as gpd

            n = len(gpd.read_file(p))
            print(f"[labels] {r.scene_id:40s} {n:7d} buildings", flush=True)
        except Exception as e:
            print(f"[labels] {r.scene_id} FAILED {e!r}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, [r for _, r in cat.iterrows()]))


def cmd_register(a):
    from ulpin3d.data.register import estimate_scene_shift
    from ulpin3d.footprint.model import load_checkpoint
    from ulpin3d.footprint.train import pick_device

    c = _cfg(a.config)
    work = Path(c["work_dir"])
    cat = pd.read_csv(work / "catalog.csv")
    dev = pick_device(a.device)
    model, ck = load_checkpoint(a.checkpoint, dev)
    reg_gsd = ck["data_cfg"].get("gsd") or 0.15
    for col in ("shift_x", "shift_y", "shift_peak", "shift_n"):
        if col not in cat:
            cat[col] = 0.0
    for i, r in cat.iterrows():
        if r.source != "oam" or r.split not in ("train", "val", "test"):
            continue
        lp = work / "labels" / f"{r.scene_id}.gpkg"
        if not lp.exists():
            continue
        try:
            sx, sy, pk, n = estimate_scene_shift(r.image_url, lp, model, dev, gsd=reg_gsd,
                                                 max_shift_m=c.get("register", {}).get("max_shift_m", 5.0))
            cat.loc[i, ["shift_x", "shift_y", "shift_peak", "shift_n"]] = [sx, sy, pk, n]
            print(f"[register] {r.scene_id:40s} shift=({sx:+.2f},{sy:+.2f}) m peak={pk:.2f} n={n}", flush=True)
        except Exception as e:
            print(f"[register] {r.scene_id} FAILED {e!r}")
        cat.to_csv(work / "catalog.csv", index=False)


def cmd_chips(a):
    from ulpin3d.data.chips import make_chips

    c = _cfg(a.config)
    work = Path(c["work_dir"])
    cat = pd.read_csv(work / "catalog.csv")
    if a.regions:
        cat = cat[cat.region.isin(a.regions.split(","))]
    ch = c.get("chips", {})
    if a.sources:
        cat = cat[cat.source.isin(a.sources.split(","))]
    out = Path(a.out or ch.get("out", work / "chips"))
    mc = ch.get("max_chips_per_scene", 400)
    budgets = mc if isinstance(mc, dict) else {reg: mc for reg in cat.region.unique()}
    for reg, n in budgets.items():
        sub = cat[cat.region == reg]
        if len(sub):
            make_chips(sub, work / "labels", out, workers=a.workers, gsd=ch.get("gsd", 0.15),
                       chip_px=ch.get("chip_px", 512), max_chips=n, neg_frac=ch.get("neg_frac", 0.1),
                       min_valid=ch.get("min_valid", 0.8))
    idx = pd.read_csv(out / "index.csv")
    print(idx.groupby(["region", "split"]).size().unstack(fill_value=0).to_string())


def cmd_export_review(a):
    """Export test-scene labels (+ model predictions if a checkpoint is given) for
    hand-correction in QGIS. Save corrected files to <work>/labels_reviewed/<scene>.gpkg."""
    import geopandas as gpd

    c = _cfg(a.config)
    work = Path(c["work_dir"])
    cat = pd.read_csv(work / "catalog.csv")
    out = work / "review"
    out.mkdir(parents=True, exist_ok=True)
    rows = cat[cat.split == a.split]
    lines = []
    for _, r in rows.iterrows():
        lp = work / "labels" / f"{r.scene_id}.gpkg"
        if not lp.exists():
            continue
        g = gpd.read_file(lp)
        g.to_file(out / f"{r.scene_id}_labels.gpkg", driver="GPKG")
        if a.checkpoint:
            from ulpin3d.footprint.infer import infer_scene

            infer_scene(r.image_url, a.checkpoint, str(out / r.scene_id), scene_id=r.scene_id)
        lines.append(f"{r.scene_id}\t{r.title}\t/vsicurl/{r.image_url}")
    (out / "README.txt").write_text(
        "Open each image URL in QGIS (Layer > Add Raster > Protocol HTTP), load <scene>_labels.gpkg,\n"
        "fix shifted / merged / missing buildings (model predictions in <scene>/buildings.gpkg help),\n"
        "and save as ../labels_reviewed/<scene>.gpkg. Re-run `chips` to rebuild test chips.\n\n"
        + "\n".join(lines))
    print((out / "README.txt").read_text())


def cmd_train_footprint(a):
    from ulpin3d.footprint.train import train

    c = _cfg(a.config)
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        d = c
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = yaml.safe_load(v)
    print("[train] best:", train(c))


def cmd_eval_footprint(a):
    from ulpin3d.footprint.evaluate import evaluate_checkpoint

    evaluate_checkpoint(a.checkpoint, a.chips, a.split, a.out, a.device, tta=not a.no_tta)


def cmd_infer_footprint(a):
    from ulpin3d.footprint.infer import infer_scene

    infer_scene(a.image, a.checkpoint, a.out, gsd=a.gsd, ndsm=a.ndsm, device=a.device, tta=not a.no_tta)


def cmd_synthetic(a):
    from ulpin3d.data.synthetic import make_synthetic_scene

    out = Path(a.out)
    rows = []
    for i in range(a.n):
        tif, lab, nb = make_synthetic_scene(out / f"syn_{i:02d}.tif", out / f"syn_{i:02d}_labels.gpkg",
                                            size_m=a.size_m, gsd=a.gsd, seed=i,
                                            block_angle_deg=float((i * 17) % 40 - 20))
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(tif) as s:
            b = transform_bounds(s.crs, 4326, *s.bounds)
        rows.append(dict(scene_id=f"syn_{i:02d}", source="local", tier="", region="india" if i % 3 == 0 else "south_asia",
                         country="India" if i % 3 == 0 else "Synthetic", iso3="IND", city="", title=f"synthetic {i}",
                         image_url=str(tif.resolve()), label_url=str(Path(lab).resolve()), gsd=a.gsd,
                         size_mb=0, bbox=json.dumps(list(b)), split="test" if i == a.n - 1 else ("val" if i == a.n - 2 else "train")))
        print(f"[synthetic] {tif} {nb} buildings")
    pd.DataFrame(rows).to_csv(out / "catalog.csv", index=False)


def cmd_pc(a):
    from ulpin3d.pointcloud import classify as C

    c = _cfg(a.config)
    {"pc-prepare": C.run_prepare, "pc-train": C.run_train, "pc-eval": C.run_eval}[a.cmd](c)


def cmd_pc_infer(a):
    from ulpin3d.pointcloud.classify import classify_cloud, load_model
    from ulpin3d.pointcloud.data import CLASS_NAMES

    lab, conf, _ = classify_cloud(load_model(a.model), a.laz, a.out, voxel=a.voxel)
    import numpy as np

    counts = np.bincount(lab, minlength=len(CLASS_NAMES))
    print({n: int(c) for n, c in zip(CLASS_NAMES, counts)}, "mean confidence", float(conf.mean()))


def cmd_fuse(a):
    from ulpin3d.fuse import fuse

    fuse(a.buildings, a.laz, a.out)


def cmd_parse_plan(a):
    import geopandas as gpd

    from ulpin3d.floorplan.parse import parse_plan

    r = parse_plan(a.plan)
    print(f"floors {r['floors']}  scale 1:{r['scale']}  source {r['source']}  issues {r['issues']}")
    for u in sorted(r["units"], key=lambda u: u["label"]):
        print(f"  {u['label']:12s} {str(u.get('bhk') or '-')}BHK  {u['carpet_area_m2']:7.2f} m2  conf {u['confidence']:.2f}  {'; '.join(u['issues'])}")
    print("  common:", [c["label"] for c in r["common"]], " unlabeled:", len(r["unlabeled"]))
    if a.out:
        rows = [dict(label=u["label"], kind=u["kind"], area_m2=u["area_m2"], confidence=u.get("confidence"),
                     issues="; ".join(u.get("issues", []))) for u in r["units"] + r["common"] + r["unlabeled"]]
        gpd.GeoDataFrame(rows, geometry=[u["polygon"] for u in r["units"] + r["common"] + r["unlabeled"]]).to_file(a.out)


def cmd_cadastre(a):
    import geopandas as gpd

    from ulpin3d.cadastre import build_cadastre, to_cityjson, to_geojson
    from ulpin3d.floorplan.parse import parse_plan

    b = gpd.read_file(a.buildings)
    crs = b.estimate_utm_crs()
    b = b.to_crs(crs)
    row = b[b.building_id == a.building_id].iloc[0]
    bld = dict(row.drop("geometry")) | {"geometry": row.geometry}
    plans = [dict(parse_plan(p), name=Path(p).name) for p in a.plans]
    units, issues = build_cadastre(bld, plans, a.parcel, a.permissible_height)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    to_geojson(units, crs, out / "units.geojson", a.building_id)
    to_cityjson(units, crs.to_epsg(), out / "units.city.json", a.building_id, bld)
    import collections

    print("issues:", *issues, sep="\n  ")
    print(collections.Counter(u["status"] for u in units), "->", out)


def cmd_run_area(a):
    from ulpin3d.area import run_area

    run_area(a.area, a.out, footprint_model=a.footprint_model, pc_model=a.pc_model)


def cmd_demo_town(a):
    from ulpin3d.demo.town import make_town

    t = make_town(a.out, seed=a.seed)
    print(f"[demo-town] {len(t['buildings'])} buildings, {len(t['plans'])} plans, {t['n_flats']} flats, "
          f"{t['lidar_points']:,} LiDAR points -> {a.out}")


def cmd_check_town(a):
    from ulpin3d.demo.check import check_town, expectations, grade

    rows = grade(check_town(a.area, a.out), expectations(a.area))
    for name, val, verdict in rows:
        print(f"  {name:28s} {val:34s} {verdict}")
    ok = not any(v == "FAIL" for _, _, v in rows)
    print("PASS: result matches what these inputs make possible" if ok else "FAIL: see the lines above")
    raise SystemExit(0 if ok else 1)


def cmd_fetch_area_nl(a):
    from ulpin3d.demo.nl_area import fetch_area_nl

    fetch_area_nl(a.bbox, a.laz, a.out, a.name)


def cmd_fetch_area_osm(a):
    from ulpin3d.demo.osm_area import fetch_area_osm

    fetch_area_osm(a.bbox, a.out, a.name)


def cmd_build_db(a):
    from ulpin3d.register_db import build_db

    print(f"[build-db] {build_db(a.out, a.parcel_chars, force=True)}")


def cmd_make_examples(a):
    from ulpin3d.demo.scenarios import make_scenarios

    for d in make_scenarios(a.base, a.out, a.only):
        print(f"[make-examples] {d}")


def cmd_run_examples(a):
    from ulpin3d.demo.examples import run_examples

    r = run_examples(a.root, a.out, a.only, a.footprint_model, base=getattr(a, "base", None))
    bad = [x["area"] for x in r if x["verdict"] in ("FAIL", "ERROR")]
    print(f"[run-examples] {len(r)} areas, {len(bad)} failed {bad or ''} -> {a.out}/RESULTS.md")
    raise SystemExit(1 if bad else 0)


def cmd_fetch_area(a):
    from ulpin3d.demo.real_area import fetch_area

    fetch_area(a.title, a.out, a.size)


def cmd_verify_id(a):
    from ulpin3d.ulpin import parse, verify

    ok, why = verify(a.id, a.parcel_chars)
    if not ok:
        print(f"INVALID: {why}")
        raise SystemExit(1)
    u = parse(a.id, a.parcel_chars)
    lv = "ground" if u.level == 0 else (f"basement {-u.level}" if u.level < 0 else f"floor {u.level}")
    print(f"VALID  {u.pretty()}  parcel {u.parcel} · {lv} · class {u.zclass} · unit {u.unit}")


def cmd_property_card(a):
    from ulpin3d.card import write_cards

    units = Path(a.units)
    out = Path(a.out) if a.out else units.parent / "cards"
    seals = write_cards(units, out, a.name, a.parcel_chars, a.id or None)
    print(f"{len(seals)} property card(s) -> {out}")
    for uid in list(seals)[:5]:
        print(f"  {out / (uid + '.html')}")


def cmd_verify_card(a):
    from ulpin3d.card import verify_card

    ok, why = verify_card(a.units, a.id, a.seal)
    print(("VALID  " if ok else "INVALID: ") + why)
    if not ok:
        raise SystemExit(1)


def cmd_serve(a):
    try:
        import uvicorn
        from ulpin3d.api import create_app
    except ImportError:
        raise SystemExit("the API needs: pip install fastapi uvicorn   (or pip install -e '.[api]')")
    app = create_app(a.out, a.parcel_chars)
    print(f"serving {a.out} at http://{a.host}:{a.port}  (docs: /docs)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def cmd_export_onnx(a):
    from ulpin3d.footprint.infer import export_onnx

    p = export_onnx(a.checkpoint, a.out, quantize=a.int8)
    print(f"[export-onnx] -> {p} (+ {p}.json)")


def cmd_train_lite(a):
    from ulpin3d.footprint.lite import train_lite

    use_h = {"auto": "auto", "yes": True, "no": False}[a.height]
    train_lite(a.chips, a.out, filters={"split": a.splits.split(",")}, use_height=use_h, per_chip=a.per_chip)


def cmd_ramp_chips(a):
    from ulpin3d.data.ramp import make_ramp_chips

    make_ramp_chips(a.out, a.datasets, a.test_datasets, max_tiles=a.max_tiles, workers=a.workers)


def cmd_eval_lite(a):
    from ulpin3d.footprint.lite import evaluate_lite

    evaluate_lite(a.model, a.chips, a.split, a.out)


def cmd_fetch_area_lidar(a):
    from ulpin3d.demo.real_area import lidar_area

    lidar_area(a.key, a.out, a.size)


def cmd_check_outlines(a):
    from ulpin3d.demo.real_area import outline_iou

    r = outline_iou(a.buildings, a.truth)
    print(f"  building outline IoU {r['iou']:.3f}  precision {r['precision']:.3f}  recall {r['recall']:.3f}  ({r['buildings']} buildings)")
    ok = r["iou"] >= a.min_iou
    print(f"{'PASS' if ok else 'FAIL'}: IoU {'>=' if ok else '<'} {a.min_iou}")
    raise SystemExit(0 if ok else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ulpin3d")
    sp = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("catalog", cmd_catalog), ("labels", cmd_labels), ("chips", cmd_chips)]:
        s = sp.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--workers", type=int, default=6)
        s.add_argument("--regions", default=None, help="e.g. india,south_asia")
        s.add_argument("--sources", default=None, help="e.g. oam or open_cities")
        s.add_argument("--out", default=None, help="chips output dir (overrides config)")
        s.set_defaults(fn=fn)
    s = sp.add_parser("export-review")
    s.add_argument("--config", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--checkpoint", default=None)
    s.set_defaults(fn=cmd_export_review)
    s = sp.add_parser("register")
    s.add_argument("--config", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_register)
    s = sp.add_parser("train-footprint")
    s.add_argument("--config", required=True)
    s.add_argument("--set", nargs="*", help="override config, e.g. train.epochs=2")
    s.set_defaults(fn=cmd_train_footprint)
    s = sp.add_parser("eval-footprint")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--chips", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--out", default=None)
    s.add_argument("--device", default="auto")
    s.add_argument("--no-tta", action="store_true")
    s.set_defaults(fn=cmd_eval_footprint)
    s = sp.add_parser("infer-footprint")
    s.add_argument("--image", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--gsd", type=float, default=None)
    s.add_argument("--ndsm", default=None, help="height-above-ground GeoTIFF (needed by *_height lite models)")
    s.add_argument("--device", default="auto")
    s.add_argument("--no-tta", action="store_true")
    s.set_defaults(fn=cmd_infer_footprint)
    s = sp.add_parser("synthetic")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=6)
    s.add_argument("--size-m", type=float, default=120.0)
    s.add_argument("--gsd", type=float, default=0.05)
    s.set_defaults(fn=cmd_synthetic)
    for name in ("pc-prepare", "pc-train", "pc-eval"):
        s = sp.add_parser(name)
        s.add_argument("--config", required=True)
        s.set_defaults(fn=cmd_pc)
    s = sp.add_parser("pc-infer")
    s.add_argument("--laz", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--voxel", type=float, default=0.3)
    s.set_defaults(fn=cmd_pc_infer)
    s = sp.add_parser("fuse", help="footprints + classified LiDAR -> 3D building records")
    s.add_argument("--buildings", required=True)
    s.add_argument("--laz", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_fuse)
    s = sp.add_parser("parse-plan", help="sanctioned plan (DXF/PDF/scan) -> units")
    s.add_argument("--plan", required=True)
    s.add_argument("--out", default=None, help="write units as GeoJSON (plan coordinates, metres)")
    s.set_defaults(fn=cmd_parse_plan)
    s = sp.add_parser("cadastre", help="fused building + plans -> 3D units with 3D ULPINs")
    s.add_argument("--buildings", required=True, help="buildings_3d.geojson from `fuse`")
    s.add_argument("--building-id", required=True)
    s.add_argument("--plans", nargs="+", required=True)
    s.add_argument("--parcel", required=True, help="2D ULPIN of the land parcel (12 chars)")
    s.add_argument("--permissible-height", type=float, default=None)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_cadastre)
    s = sp.add_parser("run-area", help="survey folder -> 3D cadastre (+ viewer) in one go")
    s.add_argument("--area", required=True, help="folder with ortho/ lidar/ parcels/ plans/ ownership/")
    s.add_argument("--out", required=True)
    s.add_argument("--footprint-model", default=None, help="trained Model 1 checkpoint (.pt)")
    s.add_argument("--pc-model", default="models/pointcloud/model.txt", help="Model 2 LightGBM file")
    s.set_defaults(fn=cmd_run_area)
    s = sp.add_parser("demo-town", help="generate a synthetic ward survey package for testing")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_demo_town)
    s = sp.add_parser("check-town", help="compare a run-area result with the synthetic ward's truth.json")
    s.add_argument("--area", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_check_town)
    s = sp.add_parser("fetch-area-nl", help="real Dutch package: 8 cm aerial + AHN LiDAR + Kadaster parcels + BAG register")
    s.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("XMIN", "YMIN", "XMAX", "YMAX"), help="EPSG:28992")
    s.add_argument("--laz", required=True, help="AHN4 tile covering the box (geotiles.citg.tudelft.nl)")
    s.add_argument("--out", required=True)
    s.add_argument("--name", default="Delft (real Dutch open data)")
    s.set_defaults(fn=cmd_fetch_area_nl)
    s = sp.add_parser("fetch-area-osm", help="real building outlines + floor counts from OpenStreetMap (e.g. BKC Mumbai)")
    s.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    s.add_argument("--out", required=True)
    s.add_argument("--name", default="OpenStreetMap area")
    s.set_defaults(fn=cmd_fetch_area_osm)
    s = sp.add_parser("build-db", help="units.geojson -> indexed SQLite register with a 3D R-tree (register.sqlite)")
    s.add_argument("--out", required=True, help="run-area output folder")
    s.add_argument("--parcel-chars", type=int, default=12)
    s.set_defaults(fn=cmd_build_db)
    s = sp.add_parser("make-examples", help="cut one test area per data situation out of the synthetic ward")
    s.add_argument("--base", default="tests/data/ward_w1")
    s.add_argument("--out", default="out/scenarios")
    s.add_argument("--only", nargs="*", default=None)
    s.set_defaults(fn=cmd_make_examples)
    s = sp.add_parser("run-examples", help="run + grade every example area, write RESULTS.md")
    s.add_argument("--root", default="examples")
    s.add_argument("--out", default="examples/results")
    s.add_argument("--only", nargs="*", default=None, help="substring filter on area names")
    s.add_argument("--footprint-model", default="models/footprint_india.onnx")
    s.set_defaults(fn=cmd_run_examples)
    s = sp.add_parser("test-scenarios", help="run + grade the 8 synthetic data situations (cut from tests/data/ward_w1)")
    s.add_argument("--base", default="tests/data/ward_w1")
    s.add_argument("--root", default=None)
    s.add_argument("--out", default="test_runs/scenarios")
    s.add_argument("--only", nargs="*", default=None)
    s.add_argument("--footprint-model", default="models/footprint_india.onnx")
    s.set_defaults(fn=cmd_run_examples)
    s = sp.add_parser("fetch-area", help="make a test folder from a real Indian OpenAerialMap drone scene")
    s.add_argument("--title", required=True, help='e.g. "Sangam Vihar", "IIT BHU", "singhnagar"')
    s.add_argument("--size", type=float, default=250, help="window size in metres")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_fetch_area)
    s = sp.add_parser("verify-id", help="check a 3D ULPIN offline")
    s.add_argument("id")
    s.add_argument("--parcel-chars", type=int, default=12)
    s.set_defaults(fn=cmd_verify_id)
    s = sp.add_parser("property-card", help="units.geojson -> sealed 3D Property Cards (HTML + QR)")
    s.add_argument("--units", required=True, help="units.geojson from run-area")
    s.add_argument("--id", nargs="*", help="only these 3D ULPINs (default: every issued ID)")
    s.add_argument("--out", help="folder for the cards (default: cards/ next to units.geojson)")
    s.add_argument("--name", default="", help="area name printed on the card")
    s.add_argument("--parcel-chars", type=int, default=12)
    s.set_defaults(fn=cmd_property_card)
    s = sp.add_parser("verify-card", help="check a card's seal against the register (units.geojson)")
    s.add_argument("--units", required=True)
    s.add_argument("--id", required=True)
    s.add_argument("--seal", required=True, help="seal printed on the card, or the 32 characters in its QR")
    s.set_defaults(fn=cmd_verify_card)
    s = sp.add_parser("serve", help="read-only REST API over a run-area result (FastAPI)")
    s.add_argument("--out", required=True, help="run-area output folder")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--parcel-chars", type=int, default=12)
    s.set_defaults(fn=cmd_serve)
    s = sp.add_parser("export-onnx", help="U-Net checkpoint -> ONNX for fast CPU inference (~2.7x faster than PyTorch)")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out", required=True, help="e.g. models/footprint_india.onnx")
    s.add_argument("--int8", action="store_true", help="also quantise weights to int8: 4x smaller file, ~3%% mask change, not faster on most CPUs")
    s.set_defaults(fn=cmd_export_onnx)
    s = sp.add_parser("train-footprint-lite", help="CPU LightGBM footprint model (no GPU)")
    s.add_argument("--chips", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--splits", default="train,val")
    s.add_argument("--height", choices=["auto", "yes", "no"], default="auto", help="use LiDAR height chips")
    s.add_argument("--per-chip", type=int, default=1500)
    s.set_defaults(fn=cmd_train_lite)
    s = sp.add_parser("ramp-chips", help="human-labelled ramp tiles (India + Bangladesh + Myanmar) -> chips")
    s.add_argument("--out", default="data/footprint/chips_ramp")
    s.add_argument("--datasets", nargs="*", default=None, help="e.g. karnataka_india dhaka_bangladesh (default: all South Asian)")
    s.add_argument("--test-datasets", nargs="*", default=["karnataka_india"])
    s.add_argument("--max-tiles", type=int, default=None, help="per dataset, for a quick try")
    s.add_argument("--workers", type=int, default=16)
    s.set_defaults(fn=cmd_ramp_chips)
    s = sp.add_parser("eval-footprint-lite")
    s.add_argument("--model", required=True)
    s.add_argument("--chips", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_eval_lite)
    s = sp.add_parser("fetch-area-lidar", help="make a real LiDAR + image test folder from a Sao Paulo tile")
    s.add_argument("--key", default="MDS_color_3315-333.laz", help="tile name (default: held-out Paraisopolis tile)")
    s.add_argument("--size", type=float, default=250)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_fetch_area_lidar)
    s = sp.add_parser("check-outlines", help="score building outlines against a truth raster (1 = building)")
    s.add_argument("--buildings", required=True, help="buildings.geojson from run-area")
    s.add_argument("--truth", required=True)
    s.add_argument("--min-iou", type=float, default=0.75)
    s.set_defaults(fn=cmd_check_outlines)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
