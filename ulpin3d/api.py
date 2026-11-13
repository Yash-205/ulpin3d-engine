"""Read-only REST API over a run-area result (the PRD's "FastAPI backend").

    python -m ulpin3d.cli serve --out out/<area>          # then open http://127.0.0.1:8000/docs

GET /health                    what is loaded
GET /summary                   totals, status counts, levels
GET /ulpin/{id}                verify a 3D ULPIN offline (check character) and return its record
GET /units                     filter: building, parcel, status, level, kind, owner (substring), limit
GET /point?lon=&lat=&z=        which volume contains this 3D point (the vertical-cadastre query)
GET /parcel/{ulpin}/units      every unit on a parcel, grouped by level
GET /card/{id}                 the 3D Property Card (HTML)
GET /verify-card               ?id=&seal= : does the printed seal match the register?

Every query is answered from register.sqlite (built on first start, rebuilt when units.geojson
changes): indexed lookups instead of scanning the GeoJSON, and a 3D R-tree for /point. `z` is the
elevation in the same datum as the unit's `z_min`/`z_max` (metres above the survey's vertical datum).
Nothing is written to the register; the API only reads.
"""
from __future__ import annotations

import math
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from ulpin3d.card import verify_feature
from ulpin3d.register_db import Register
from ulpin3d.ulpin import parse, verify


def _clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _public(props: dict) -> dict:
    return {k: _clean(v) for k, v in props.items()}


def create_app(out_dir, parcel_chars: int = 12) -> FastAPI:
    out = Path(out_dir)
    if not (out / "units.geojson").exists():
        raise FileNotFoundError(f"{out / 'units.geojson'} not found: run `run-area` first")
    reg = Register(out, parcel_chars)
    report = reg.report
    app = FastAPI(title="Bhu-Aadhaar 3D: 3D ULPIN API", version="0.2.0",
                  description="Read-only access to a 3D cadastre produced by ulpin3d run-area (indexed SQLite register).")

    @app.get("/health")
    def health():
        return {"ok": True, "area": report.get("name"), "units": reg.n_units, "ids": reg.n_ids(),
                "cards": (out / "cards").exists(), "store": reg.path.name}

    @app.get("/summary")
    def summary():
        return {"area": report.get("name"), "data_grade": report.get("data_grade"), "totals": report.get("totals"),
                **reg.summary()}

    @app.get("/ulpin/{uid}")
    def get_ulpin(uid: str):
        ok, why = verify(uid, parcel_chars)
        if not ok:
            raise HTTPException(422, f"not a valid 3D ULPIN: {why}")
        u = parse(uid, parcel_chars)
        f = reg.by_ulpin(u.pretty().replace("-", ""))
        parts = {"parcel": u.parcel, "level": u.level, "class": u.zclass, "unit": u.unit}
        if f is None:
            return {"valid": True, "in_register": False, "parts": parts}
        return {"valid": True, "in_register": True, "parts": parts,
                "unit": {**_public(f["properties"]), "geometry": f["geometry"]}}

    @app.get("/units")
    def units(building: str | None = None, parcel: str | None = None, status: str | None = None,
              level: int | None = None, kind: str | None = None, owner: str | None = None,
              limit: int = Query(100, ge=1, le=5000)):
        rows = [_public(f["properties"]) for f in reg.units(building, parcel, status, level, kind, owner, limit)]
        return {"count": len(rows), "units": rows}

    @app.get("/point")
    def point(lon: float, lat: float, z: float | None = None):
        """Units whose outline contains (lon, lat); with `z`, only those whose z-range also contains it."""
        hits = [_public(f["properties"]) for f in reg.at_point(lon, lat, z)]
        hits.sort(key=lambda p: (p.get("z_min") is None, p.get("z_min")))
        return {"lon": lon, "lat": lat, "z": z, "count": len(hits), "units": hits}

    @app.get("/parcel/{ulpin}/units")
    def parcel_units(ulpin: str):
        rows = [_public(f["properties"]) for f in reg.parcel(ulpin)]
        if not rows:
            raise HTTPException(404, f"no units on parcel {ulpin}")
        levels: dict[int, list] = {}
        for r in rows:
            levels.setdefault(int(r["level"]), []).append(r)
        return {"parcel": ulpin.upper(), "count": len(rows), "levels": {k: levels[k] for k in sorted(levels, reverse=True)}}

    @app.get("/card/{uid}")
    def card(uid: str):
        uid = "".join(c for c in uid if c.isalnum()).upper()
        path = out / "cards" / f"{uid}.html"
        if not path.exists():
            raise HTTPException(404, f"no property card for {uid}")
        return FileResponse(path, media_type="text/html")

    @app.get("/verify-card")
    def verify_card_ep(id: str, seal: str):
        uid = "".join(c for c in id if c.isalnum()).upper()
        ok, why = verify_feature(reg.by_ulpin(uid), uid, seal)
        return {"valid": ok, "detail": why}

    return app
