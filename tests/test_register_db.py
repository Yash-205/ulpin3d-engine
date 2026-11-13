import json
import os
import time

from ulpin3d.register_db import DB_NAME, Register, build_db, is_fresh

SQ = [[[73.85, 18.52], [73.8501, 18.52], [73.8501, 18.5201], [73.85, 18.5201], [73.85, 18.52]]]
SQ2 = [[[73.86, 18.53], [73.8601, 18.53], [73.8601, 18.5301], [73.86, 18.5301], [73.86, 18.53]]]


def _write(d, owners):
    feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": SQ if i % 2 == 0 else SQ2},
              "properties": {"ulpin3d": f"ID{i:03d}", "parcel_ulpin": "p1" if i < 3 else "P2", "building_id": "B1",
                             "level": i, "kind": "unit", "status": "validated", "owner_name": o,
                             "z_min": 100.0 + 3 * i, "z_max": 103.0 + 3 * i}}
             for i, o in enumerate(owners)]
    (d / "units.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def test_indexes_answer_like_a_scan(tmp_path):
    _write(tmp_path, ["Priya Sharma", "Rohan Patil", None, "A. Sharma-Iyer", "Om_Prakash 100%"])
    r = Register(tmp_path)
    assert [f["properties"]["ulpin3d"] for f in r.units(owner="SHARMA")] == ["ID000", "ID003"]   # trigram index
    assert [f["properties"]["ulpin3d"] for f in r.units(owner="m_")] == ["ID004"]                 # short / wildcard chars: scan path, escaped
    assert [f["properties"]["ulpin3d"] for f in r.units(owner="100%")] == ["ID004"]
    assert len(r.parcel("P1")) == 3 and r.by_ulpin("ID001")["properties"]["owner_name"] == "Rohan Patil"
    assert [f["properties"]["level"] for f in r.at_point(73.85005, 18.52005)] == [0, 2, 4]           # 3D R-tree, all stacked units
    assert [f["properties"]["level"] for f in r.at_point(73.85005, 18.52005, z=107.5)] == [2]       # ... and only the one at that height
    assert r.at_point(73.95, 18.52) == []
    assert "INDEX ix_parcel" in " ".join(r.explain("SELECT id FROM units WHERE parcel_ulpin = ?", ("P1",)))


def test_rebuilt_when_the_register_changes(tmp_path):
    _write(tmp_path, ["A", "B"])
    build_db(tmp_path)
    assert is_fresh(tmp_path)
    time.sleep(0.01)
    _write(tmp_path, ["A", "B", "C"])
    os.utime(tmp_path / "units.geojson")
    assert not is_fresh(tmp_path)
    assert Register(tmp_path).n_units == 3 and (tmp_path / DB_NAME).exists()
