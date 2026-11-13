import json

import pytest

from ulpin3d.card import qr_payload, record, seal, verify_card, write_cards
from ulpin3d.ulpin import make


def _units(tmp_path, owner="A. Sharma"):
    uid = str(make("27PUNEW10101", 4, "S", 2))
    feats = [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[73.85, 18.52], [73.8501, 18.52], [73.8501, 18.5201], [73.85, 18.5201], [73.85, 18.52]]]},
         "properties": {"ulpin3d": uid, "parcel_ulpin": "27PUNEW10101", "building_id": "B0001", "level": 4, "zclass": "S",
                        "kind": "unit", "label": "FLAT 402", "z_min": 572.9, "z_max": 576.0, "area_m2": 71.2, "carpet_area_m2": 64.5,
                        "volume_m3": 220.7, "bhk": 2, "status": "validated", "confidence": 0.93, "owner_name": owner,
                        "share_pct": 100, "right_type": "ownership", "document": "DEED/2025/0042", "issues": ""}},
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[73.85, 18.52], [73.8501, 18.52], [73.8501, 18.5201], [73.85, 18.52]]]},
         "properties": {"ulpin3d": None, "kind": "whole_level", "status": "flagged", "level": 0, "zclass": "S"}},
    ]
    p = tmp_path / "units.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return p, uid


def test_cards_written_only_for_issued_ids(tmp_path):
    p, uid = _units(tmp_path)
    seals = write_cards(p, tmp_path / "cards", "Test ward")
    assert list(seals) == [uid]
    html = (tmp_path / "cards" / f"{uid}.html").read_text()
    assert "FLAT 402" in html and seals[uid] in html and "<svg" in html and "A. Sharma" in html
    assert json.loads((tmp_path / "cards" / "seals.json").read_text()) == seals
    assert uid in (tmp_path / "cards" / "index.html").read_text()
    assert qr_payload(uid, seals[uid]) == f"ULPIN3D:{uid};SEAL:{seals[uid][:32].upper()}"


def test_seal_detects_any_change_to_the_record(tmp_path):
    p, uid = _units(tmp_path)
    s = write_cards(p, tmp_path / "cards")[uid]
    assert verify_card(p, uid, s)[0] and verify_card(p, uid, s[:32].upper())[0]
    assert not verify_card(p, uid, s[:8])[0]  # too short to trust
    assert not verify_card(p, "27PUNEW10101H4S034", s)[0]  # not in the register
    for field, value in (("owner_name", "B. Verma"), ("carpet_area_m2", 84.5), ("z_max", 579.0), ("status", "provisional")):
        fc = json.loads(p.read_text())
        fc["features"][0]["properties"][field] = value
        q = tmp_path / f"tampered_{field}.geojson"
        q.write_text(json.dumps(fc))
        assert not verify_card(q, uid, s)[0], field
    fc = json.loads(p.read_text())
    fc["features"][0]["geometry"]["coordinates"][0][1][0] += 0.0001  # outline moved
    q = tmp_path / "tampered_geom.geojson"
    q.write_text(json.dumps(fc))
    assert not verify_card(q, uid, s)[0]


def test_keyed_seal_needs_the_key(tmp_path):
    p, uid = _units(tmp_path)
    rec = record(json.loads(p.read_text())["features"][0])
    assert seal(rec, "authority-secret") != seal(rec, "") and seal(rec, "authority-secret") != seal(rec, "other")
    s = write_cards(p, tmp_path / "cards", key="authority-secret")[uid]
    assert verify_card(p, uid, s, key="authority-secret")[0] and not verify_card(p, uid, s, key="")[0]


def test_unknown_id_is_an_error(tmp_path):
    p, _ = _units(tmp_path)
    with pytest.raises(KeyError):
        write_cards(p, tmp_path / "cards", ids=["27PUNEW10101H4S034"])
