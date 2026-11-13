"""A real area from OpenStreetMap building outlines and floor counts (e.g. Bandra Kurla Complex, Mumbai).

    python -m ulpin3d.cli fetch-area-osm --bbox 72.858 19.056 72.872 19.072 --out examples/bkc_mumbai \
        --name "Bandra Kurla Complex, Mumbai"

Writes footprints/osm_buildings.geojson (outlines with `building:levels` / `height` where mappers
recorded them, plus the building name) and area.yaml. For BKC there is no open drone imagery, LiDAR,
parcel map or sanctioned plan, so this is a grade-C package: real outlines, declared floor counts where
OSM has them, single levels elsewhere. OSM data (c) OpenStreetMap contributors, ODbL.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import requests
import yaml
from shapely.geometry import Polygon

OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter")
KEEP = ("name", "building", "building:levels", "height", "building:min_level", "addr:street")


def fetch_area_osm(bbox, out, name="OpenStreetMap area"):
    lon0, lat0, lon1, lat1 = bbox
    q = f'[out:json][timeout:120];way["building"]({lat0},{lon0},{lat1},{lon1});out tags geom;'
    import time

    r = None
    for attempt in range(6):  # public Overpass servers are often busy: retry, rotate mirrors
        try:
            r = requests.post(OVERPASS[attempt % len(OVERPASS)], data={"data": q},
                              headers={"User-Agent": "ulpin3d-engine/0.1 (research prototype)"}, timeout=180)
            if r.ok:
                break
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    if r is None or not r.ok:
        raise SystemExit("OpenStreetMap (Overpass) is not answering; try again in a few minutes")
    rows, geoms = [], []
    for e in r.json()["elements"]:
        pts = [(p["lon"], p["lat"]) for p in e.get("geometry", [])]
        if len(pts) < 4:
            continue
        g = Polygon(pts).buffer(0)
        if g.is_empty:
            continue
        t = e.get("tags", {})
        rows.append({"osm_id": e["id"], **{k: t.get(k) for k in KEEP}})
        geoms.append(g)
    g = gpd.GeoDataFrame(rows, geometry=geoms, crs=4326)
    out = Path(out)
    (out / "footprints").mkdir(parents=True, exist_ok=True)
    g.to_file(out / "footprints" / "osm_buildings.geojson", driver="GeoJSON")
    n_lv = int(g["building:levels"].notna().sum()) if "building:levels" in g else 0
    (out / "area.yaml").write_text(yaml.safe_dump({
        "name": name, "synthetic": False, "parcel_chars": 12,
        "source": {"outlines": "OpenStreetMap contributors (ODbL), building ways with building:levels / height"}},
        sort_keys=False, allow_unicode=True))
    print(f"[fetch-area-osm] {len(g)} buildings, {n_lv} with a floor count -> {out}")
    return out
