import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from ulpin3d.api import create_app
from ulpin3d.card import write_cards
from ulpin3d.ulpin import make

SQ = [[[73.85, 18.52], [73.8501, 18.52], [73.8501, 18.5201], [73.85, 18.5201], [73.85, 18.52]]]


def _area(tmp_path):
    feats = []
    for lv, z0, z1, kind, owner in ((-1, 97.0, 100.0, "basement", None), (0, 100.0, 103.5, "unit", "A. Sharma"), (1, 103.5, 106.6, "unit", "B. Verma")):
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": SQ},
                      "properties": {"ulpin3d": str(make("27PUNEW10101", lv, "S", 1)), "parcel_ulpin": "27PUNEW10101", "building_id": "B0001",
                                     "level": lv, "zclass": "S", "kind": kind, "label": f"L{lv}", "z_min": z0, "z_max": z1, "area_m2": 70.0,
                                     "volume_m3": 210.0, "status": "provisional" if kind == "basement" else "validated", "owner_name": owner,
                                     "carpet_area_m2": None, "bhk": None, "share_pct": None, "right_type": None, "document": None}})
    (tmp_path / "units.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    write_cards(tmp_path / "units.geojson", tmp_path / "cards")
    return TestClient(create_app(tmp_path)), [f["properties"]["ulpin3d"] for f in feats]


def test_vertical_point_query_separates_stacked_units(tmp_path):
    c, ids = _area(tmp_path)
    lon, lat = 73.85005, 18.52005
    assert [u["level"] for u in c.get("/point", params=dict(lon=lon, lat=lat)).json()["units"]] == [-1, 0, 1]
    for z, lv in ((98.5, -1), (101.0, 0), (105.0, 1)):
        r = c.get("/point", params=dict(lon=lon, lat=lat, z=z)).json()
        assert [u["level"] for u in r["units"]] == [lv]
    assert c.get("/point", params=dict(lon=lon, lat=lat, z=200)).json()["count"] == 0
    assert c.get("/point", params=dict(lon=74.0, lat=18.0)).json()["count"] == 0


def test_ulpin_lookup_filters_and_cards(tmp_path):
    c, ids = _area(tmp_path)
    r = c.get(f"/ulpin/{ids[1]}").json()
    assert r["valid"] and r["in_register"] and r["unit"]["owner_name"] == "A. Sharma" and r["parts"]["level"] == 0
    assert c.get(f"/ulpin/{ids[0]}").json()["parts"]["level"] == -1
    assert c.get("/ulpin/27PUNEW10101H0S014").status_code == 422  # wrong check character
    assert c.get("/units", params=dict(owner="verma")).json()["count"] == 1
    assert c.get("/units", params=dict(status="provisional")).json()["units"][0]["kind"] == "basement"
    assert c.get("/units", params=dict(level=0)).json()["count"] == 1
    assert list(c.get("/parcel/27PUNEW10101/units").json()["levels"]) == ["1", "0", "-1"]
    assert c.get("/parcel/NOPE/units").status_code == 404
    assert c.get("/summary").json()["levels"] == [-1, 0, 1]
    assert "A. Sharma" in c.get(f"/card/{ids[1]}").text and c.get("/card/27PUNEW10101H9S01X").status_code == 404
    seal = json.loads((tmp_path / "cards" / "seals.json").read_text())[ids[1]]
    assert c.get("/verify-card", params=dict(id=ids[1], seal=seal)).json()["valid"]
    assert not c.get("/verify-card", params=dict(id=ids[1], seal="0" * 32)).json()["valid"]
