"""End-to-end: synthetic ward survey package -> run-area -> compare with ground truth."""
from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parents[1] / "models" / "pointcloud" / "model.txt"


@pytest.mark.slow
@pytest.mark.skipif(not MODEL.exists(), reason="trained LiDAR model not present")
def test_run_area_on_synthetic_ward(tmp_path):
    from ulpin3d.area import run_area
    from ulpin3d.demo.check import check_town
    from ulpin3d.demo.town import make_town

    make_town(tmp_path / "ward", seed=0)
    run_area(tmp_path / "ward", tmp_path / "out", pc_model=str(MODEL))
    r = check_town(tmp_path / "ward", tmp_path / "out")
    assert r["buildings_found"] == r["buildings_true"] == r["buildings_matched"]
    assert r["floors_exact"] == "8/8" and r["plans_on_right_parcel"] == "4/4"
    assert r["unsanctioned_found"].startswith("1/1") and "false alarms 0" in r["unsanctioned_found"]
    assert r["encroachments_found"].startswith("1/1") and "false alarms 0" in r["encroachments_found"]
    assert r["flats_found"] == r["flats_true"] and r["ids_unique"]
    assert (tmp_path / "out" / "viewer" / "index.html").exists()
    assert (tmp_path / "out" / "viewer" / "cadastre_data.js").stat().st_size > 10_000
