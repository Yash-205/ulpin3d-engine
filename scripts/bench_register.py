"""Benchmark: the old in-memory GeoJSON register vs the indexed SQLite register, at district scale.

    python scripts/bench_register.py --units test_runs/scenarios/1_drone_lidar_plans/units.geojson --copies 500

Tiles a real area's units into a copies x 1 km grid (fresh IDs, same shapes) so the register has
~100k units, then times start-up and each API query both ways. Writes reports/register_benchmark.md.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from ulpin3d.card import verify_card, verify_feature
from ulpin3d.register_db import Register, build_db


def tile(src: Path, out: Path, copies: int):
    fc = json.loads(src.read_text())
    side = math.ceil(math.sqrt(copies))
    feats = []
    for k in range(copies):
        dx, dy = (k % side) * 0.01, (k // side) * 0.01

        def mv(c):
            return [c[0] + dx, c[1] + dy, *c[2:]] if isinstance(c[0], (int, float)) else [mv(q) for q in c]

        for f in fc["features"]:
            p = dict(f["properties"])
            if p.get("ulpin3d"):
                p["ulpin3d"] = f"{p['ulpin3d']}{k:03d}"
            p["parcel_ulpin"] = f"{p.get('parcel_ulpin')}-{k:03d}"
            feats.append({"type": "Feature", "properties": p,
                          "geometry": {"type": f["geometry"]["type"], "coordinates": mv(f["geometry"]["coordinates"])}})
    out.mkdir(parents=True, exist_ok=True)
    (out / "units.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return feats


def timeit(fn, n=20):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--copies", type=int, default=100)
    ap.add_argument("--md", default="reports/register_benchmark.md")
    a = ap.parse_args()
    d = Path(tempfile.mkdtemp())
    feats = tile(Path(a.units), d, a.copies)
    n = len(feats)
    some = next(f for f in reversed(feats) if f["properties"].get("ulpin3d"))
    uid, par = some["properties"]["ulpin3d"], some["properties"]["parcel_ulpin"]
    owner = next((str(f["properties"]["owner_name"]).split()[-1] for f in reversed(feats) if f["properties"].get("owner_name")), "zzz")
    assert owner != "zzz", "pick a units.geojson with owners (e.g. the synthetic ward) so the owner search is real"
    c = shape(some["geometry"]).representative_point()
    zmid = (some["properties"]["z_min"] + some["properties"]["z_max"]) / 2
    print(f"{n:,} units, file {(d / 'units.geojson').stat().st_size / 1e6:.0f} MB")

    # ---- old: what api.py did before (parse everything, keep it in RAM, scan per query)
    t = time.perf_counter()
    fc = json.loads((d / "units.geojson").read_text())["features"]
    geoms = [shape(f["geometry"]) for f in fc]
    tree = STRtree(geoms)
    by_id = {f["properties"]["ulpin3d"]: i for i, f in enumerate(fc) if f["properties"].get("ulpin3d")}
    old_start = (time.perf_counter() - t) * 1000

    def old_point():
        pt = Point(c.x, c.y)
        return [i for i in tree.query(pt, predicate="intersects")
                if fc[int(i)]["properties"]["z_min"] <= zmid <= fc[int(i)]["properties"]["z_max"]]

    old = {
        "3D ULPIN lookup": lambda: fc[by_id[uid]],
        "units on a parcel": lambda: [f for f in fc if f["properties"].get("parcel_ulpin") == par],
        "owner name search": lambda: [f for f in fc if owner.lower() in str(f["properties"].get("owner_name") or "").lower()][:100],
        "status + level filter": lambda: [f for f in fc if f["properties"].get("status") == "validated" and f["properties"].get("level") == 3][:100],
        "3D point (lon, lat, z)": old_point,
        "verify a card seal": lambda: verify_card(d / "units.geojson", uid, "0" * 32),
    }
    # ---- new: indexed SQLite
    t = time.perf_counter()
    build_db(d, force=True)
    build_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    reg = Register(d)
    new_start = (time.perf_counter() - t) * 1000
    new = {
        "3D ULPIN lookup": lambda: reg.by_ulpin(uid),
        "units on a parcel": lambda: reg.parcel(par),
        "owner name search": lambda: reg.units(owner=owner, limit=100),
        "status + level filter": lambda: reg.units(status="validated", level=3, limit=100),
        "3D point (lon, lat, z)": lambda: reg.at_point(c.x, c.y, zmid),
        "verify a card seal": lambda: verify_feature(reg.by_ulpin(uid), uid, "0" * 32),
    }
    assert len(old_point()) == len(reg.at_point(c.x, c.y, zmid)) >= 1
    assert len(old["units on a parcel"]()) == len(reg.parcel(par))
    assert len(old["owner name search"]()) == len(reg.units(owner=owner, limit=100)) > 0
    rows = [("start-up (load / open register)", old_start, new_start)]
    for k in old:
        rows.append((k, timeit(old[k], 5 if "seal" in k else 20), timeit(new[k])))
    L = [f"# Register benchmark: {n:,} 3D units", "",
         f"Real units from `{a.units}` tiled {a.copies}x (fresh IDs). Old = previous API (whole GeoJSON parsed into "
         f"RAM, scanned per query). New = `register.sqlite` (B-tree indexes + 3D R-tree). Median of repeated runs, "
         f"one laptop CPU. Building the database once took {build_ms / 1000:.1f} s; it is "
         f"{(d / 'register.sqlite').stat().st_size / 1e6:.0f} MB on disk vs {(d / 'units.geojson').stat().st_size / 1e6:.0f} MB GeoJSON.", "",
         "| Query | Old (ms) | New (ms) | Speed-up |", "|---|---:|---:|---:|"]
    for k, o, nw in rows:
        sp = "both instant (< 1 ms)" if o < 1 and nw < 1 else f"{o / max(nw, 1e-6):,.0f}x"
        L.append(f"| {k} | {o:,.2f} | {nw:,.2f} | {sp} |")
    L += ["", "Old start-up also keeps the whole register in RAM; the database is read from disk per query, so memory "
          "stays flat as the district grows."]
    Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
