"""Building + sanctioned plans -> volumetric units with 3D ULPINs (the cadastre record).

Inputs per building
    footprint polygon (UTM), ground_z, roof_h_main, floors_est   <- fuse.py (Model 1 + 2)
    parsed plans (parse.py), each covering a floor range           <- Model 3
    parent parcel id (2D ULPIN, 12 chars by default)

Steps
    1. georeference each plan onto the footprint (georef.py); halt/flag per quality gates
    2. floor heights: ground floor + typical floor height derived from the LiDAR roof
       height and the observed floor count (falls back to 3.1 m)
    3. stack units: z_min/z_max per level; carpet area from the plan
    4. reality vs sanction: levels LiDAR sees but no plan covers -> one whole-level unit
       per extra floor, status 'unsanctioned' (flag, don't fail)
    5. air rights: from the roof up to the permissible height (if given) -> class A
    6. 3D ULPIN per unit (ulpin.py, deterministic), with status / confidence / issues
    7. export GeoJSON (2D footprint + z attributes) and CityJSON 2.0 (LOD1 solids)

    6b. underground: `basements` declared levels below ground, stacked down from ground_z (class S,
       negative level, kind 'basement'); capped at provisional because LiDAR cannot observe them

Status of a unit = min over: plan unit confidence, georeference quality, building
3D confidence; then the same thresholds as everywhere else:
    >= 0.85 validated (ID issued) · 0.60-0.85 provisional (ID issued, marked) · < 0.60 flagged (no ID)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from shapely.geometry import mapping

from ulpin3d.floorplan.georef import apply_fit, fit_plan_to_footprint
from ulpin3d.ulpin import assign_units, b32, make

VALID, PROV = 0.85, 0.60


def _status(c):
    return "validated" if c >= VALID else ("provisional" if c >= PROV else "flagged")


def floor_levels(ground_z, roof_h, n_floors, gf_h=None, typ_h=None):
    """Return list of (level, z_min, z_max) for levels 0..n_floors-1."""
    if typ_h is None:
        typ_h = (roof_h - 0.4) / max(n_floors, 1) if roof_h else 3.1
        typ_h = float(np.clip(typ_h, 2.7, 4.5))
    gf_h = gf_h or typ_h + 0.4
    out, z = [], ground_z
    for lv in range(n_floors):
        h = gf_h if lv == 0 else typ_h
        out.append((lv, round(z, 3), round(z + h, 3)))
        z += h
    return out


def build_cadastre(building: dict, plans: list, parcel: str, permissible_height_m: float | None = None,
                   plan_is_scan: bool | None = None, parcel_chars: int = 12, basements: int = 0,
                   basement_height_m: float = 3.0):
    fp = building["geometry"]
    gz = float(building.get("ground_z") or 0.0)
    roof = float(building.get("roof_h_main") or 0.0)
    n_obs = int(building.get("floors_est") or 0)
    bconf = float(building.get("confidence_3d") or building.get("confidence") or 0.9)
    sanctioned = sorted({f for p in plans for f in (p.get("floors") or [])})
    top_planned = max(sanctioned) if sanctioned else 0
    n_levels = max(n_obs, top_planned + 1)
    levels = {lv: (z0, z1) for lv, z0, z1 in floor_levels(gz, roof, n_levels)}
    units, issues = [], []
    covered = set()
    for p in plans:
        scan = plan_is_scan if plan_is_scan is not None else p.get("source") == "image"
        fit = p.get("fit") or fit_plan_to_footprint(p["outline"], fp, allow_scale=scan)
        g_conf = 1.0 if fit["status"] == "ok" else (0.7 if fit["status"] == "flag" else 0.3)
        if fit["ambiguous"]:
            g_conf = min(g_conf, 0.8)
        issues += [f"plan {p.get('name', '')}: {i}" for i in fit["issues"]]
        floors = p.get("floors") or []
        if not floors:
            issues.append(f"plan {p.get('name', '')}: floor range not found on sheet; not stacked")
            continue
        for lv in floors:
            if lv not in levels:
                continue
            covered.add(lv)
            z0, z1 = levels[lv]
            for kind, items in (("unit", p["units"]), ("common", p["common"])):
                for u in items:
                    geom = apply_fit(u["polygon"], fit)
                    label = u["label"]
                    if kind == "unit" and str(label).split()[-1].isdigit() and len(str(label).split()[-1]) >= 3:
                        # typical plan drawn for one floor: renumber per level (401 on the 4th floor)
                        num = str(label).split()[-1]
                        label = f"{str(label).split()[0]} {lv}{num[-2:]}"
                    conf = min(float(u.get("confidence", 0.9)), g_conf, bconf)
                    units.append(dict(level=lv, zclass="S", common=kind == "common", label=label, kind=kind,
                                      geometry=geom, z_min=z0, z_max=z1, carpet_area_m2=u.get("carpet_area_m2"),
                                      area_m2=round(geom.area, 2), volume_m3=round(geom.area * (z1 - z0), 1),
                                      bhk=u.get("bhk"), confidence=round(conf, 3),
                                      issues=list(u.get("issues", [])) + fit["issues"],
                                      x=geom.centroid.x, y=geom.centroid.y, source="sanctioned_plan"))
    # ground floor / levels LiDAR sees that no plan covers
    for lv in range(n_levels):
        if lv in covered:
            continue
        z0, z1 = levels[lv]
        unsanctioned = sanctioned and lv > top_planned
        st_issue = (f"level {lv} observed by LiDAR but not in sanctioned plans (unsanctioned floor)"
                    if unsanctioned else f"no plan for level {lv}; whole level registered as one unit")
        units.append(dict(level=lv, zclass="S", common=False, label=f"LEVEL {lv}", kind="whole_level",
                          geometry=fp, z_min=z0, z_max=z1, area_m2=round(fp.area, 2), volume_m3=round(fp.area * (z1 - z0), 1),
                          confidence=round(min(bconf, 0.7 if not unsanctioned else 0.65), 3), issues=[st_issue],
                          x=fp.centroid.x, y=fp.centroid.y, source="lidar", unsanctioned=bool(unsanctioned)))
        if unsanctioned:
            issues.append(st_issue)
    # air rights above the roof
    top = max(z1 for _, z1 in levels.values()) if levels else gz + roof
    if permissible_height_m and gz + permissible_height_m > top + 0.5:
        units.append(dict(level=n_levels, zclass="A", common=False, label="AIR RIGHTS", kind="air_rights",
                          geometry=fp, z_min=round(top, 3), z_max=round(gz + permissible_height_m, 3),
                          area_m2=round(fp.area, 2), volume_m3=round(fp.area * (gz + permissible_height_m - top), 1),
                          confidence=round(bconf, 3), issues=[], x=fp.centroid.x, y=fp.centroid.y, source="zoning"))
    # underground levels (basement parking, services): LiDAR cannot see below ground, so these come from
    # a declared depth (permit / surveyed depth marker) and never exceed 'provisional'
    for k in range(1, int(basements or 0) + 1):
        z1, z0 = round(gz - (k - 1) * basement_height_m, 3), round(gz - k * basement_height_m, 3)
        units.append(dict(level=-k, zclass="S", common=False, label=f"BASEMENT {k}", kind="basement",
                          geometry=fp, z_min=z0, z_max=z1, area_m2=round(fp.area, 2),
                          volume_m3=round(fp.area * basement_height_m, 1), confidence=round(min(bconf, 0.75), 3),
                          issues=[f"basement {k} from declared depth ({basement_height_m:g} m per level), not observed by LiDAR"],
                          x=fp.centroid.x, y=fp.centroid.y, source="declared_depth"))
    assign_ids(units, parcel, parcel_chars)
    return units, issues


def _pref(u):
    num = str(u.get("label", "")).split()[-1]
    return int(num[-2:]) if u["kind"] == "unit" and num.isdigit() and len(num) >= 3 else None


def assign_ids(units, parcel, parcel_chars=12):
    """Give every unit of a parcel its 3D ULPIN (call once per parcel, over all its
    buildings, so two buildings on one plot can never collide). Flagged units keep
    a candidate ID but are not issued one."""
    ids = assign_units(parcel, [dict(level=u["level"], zclass=u["zclass"], x=u["x"], y=u["y"], preferred_unit=_pref(u),
                                     common=u["common"] or u["kind"] in ("whole_level", "air_rights"))
                                for u in units], parcel_chars)
    for u, i in zip(units, ids):
        u["status"] = "unsanctioned" if u.get("unsanctioned") else _status(u["confidence"])
        u["ulpin3d"] = i if u["status"] != "flagged" else None
        u["ulpin3d_candidate"] = i
    return units


def to_geojson(units, crs, path, building_id=""):
    import geopandas as gpd

    rows = []
    for u in units:
        r = {k: v for k, v in u.items() if k not in ("geometry", "x", "y")}
        r["issues"] = "; ".join(r.get("issues", []))
        r["building_id"] = building_id
        rows.append(r)
    g = gpd.GeoDataFrame(rows, geometry=[u["geometry"] for u in units], crs=crs).to_crs(4326)
    g.to_file(path, driver="GeoJSON")
    return g


def to_cityjson(units, crs_epsg: int, path, building_id="B1", building: dict | None = None):
    """CityJSON 2.0: one Building with BuildingUnit children (LOD1 extruded solids)."""
    verts, objects = [], {}
    scale = 0.001

    def solid(poly, z0, z1):
        if poly.geom_type == "MultiPolygon":  # a unit in several pieces (e.g. a roof split by a plot line)
            parts = [solid(q, z0, z1)["boundaries"][0] for q in poly.geoms if q.area > 0]
            return {"type": "MultiSolid", "lod": "1", "boundaries": parts}
        ring = list(poly.exterior.coords)[:-1]
        if poly.exterior.is_ccw is False:
            ring = ring[::-1]
        base = len(verts)
        n = len(ring)
        for x, y in ring:
            verts.append([x, y, z0])
        for x, y in ring:
            verts.append([x, y, z1])
        bottom = [[list(range(base + n - 1, base - 1, -1))]]
        top = [[list(range(base + n, base + 2 * n))]]
        sides = [[[base + i, base + (i + 1) % n, base + n + (i + 1) % n, base + n + i]] for i in range(n)]
        return {"type": "Solid", "lod": "1", "boundaries": [bottom + top + sides]}

    children = []
    for u in units:
        oid = u.get("ulpin3d") or f"{building_id}-pending-{len(children)}"
        objects[oid] = {"type": "BuildingUnit", "parents": [building_id],
                        "attributes": {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                                       for k, v in u.items() if k in ("label", "kind", "level", "zclass", "status",
                                                                       "confidence", "carpet_area_m2", "volume_m3",
                                                                       "z_min", "z_max", "ulpin3d_candidate")},
                        "geometry": [solid(u["geometry"], u["z_min"], u["z_max"])]}
        children.append(oid)
    if building:
        objects[building_id] = {"type": "Building", "children": children,
                                "attributes": {"floors_est": building.get("floors_est"),
                                               "roof_h_main": building.get("roof_h_main")},
                                "geometry": [solid(building["geometry"], building.get("ground_z", 0),
                                                   building.get("ground_z", 0) + building.get("roof_h_main", 0))]}
    V = np.array(verts)
    origin = V.min(0)
    q = np.round((V - origin) / scale).astype(int).tolist()
    cj = {"type": "CityJSON", "version": "2.0",
          "transform": {"scale": [scale] * 3, "translate": origin.tolist()},
          "metadata": {"referenceSystem": f"https://www.opengis.net/def/crs/EPSG/0/{crs_epsg}"},
          "CityObjects": objects, "vertices": q}
    Path(path).write_text(json.dumps(cj))
    return cj
