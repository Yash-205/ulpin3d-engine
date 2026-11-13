"""Model 3 (plan parsing), georeferencing and cadastre assembly on synthetic Indian plans."""
import pytest
from shapely import affinity
from shapely.geometry import shape

from ulpin3d.cadastre import build_cadastre, to_cityjson
from ulpin3d.floorplan.evaluate import score_plan
from ulpin3d.floorplan.georef import apply_fit, fit_plan_to_footprint
from ulpin3d.floorplan.parse import parse_floors, parse_plan
from ulpin3d.floorplan.synthetic import make_plan_files
from ulpin3d.ulpin import parse, verify


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    d = tmp_path_factory.mktemp("plan")
    p, gt = make_plan_files(d, "p", seed=5, floor_from=1, floor_to=6)
    return d, gt


@pytest.mark.parametrize("ext", ["dxf", "pdf"])
def test_vector_plans_parse_exactly(plan, ext):
    d, gt = plan
    s = score_plan(d / f"p.{ext}", d / "p_gt.json")
    assert s["unit_f1"] == 1.0 and s["mean_iou"] > 0.95 and s["area_err_pct_mean"] < 1.0 and s["floors_ok"]


def test_scanned_plan(plan):
    d, gt = plan
    s = score_plan(d / "p.png", d / "p_gt.json", allow_scale=True)
    assert s["unit_recall"] >= 0.75 and s["floors_ok"] and abs(s["scale"] - 1) < 0.03


def test_floor_title_parsing():
    assert parse_floors([("TYPICAL FLOOR PLAN (1ST TO 6TH FLOOR)", 0, 0)]) == [1, 2, 3, 4, 5, 6]
    assert parse_floors([("TYPICALFLOORPLAN(2NDTO4THFLOOR)", 0, 0)]) == [2, 3, 4]
    assert parse_floors([("GROUND FLOOR PLAN", 0, 0)]) == [0]
    assert parse_floors([("STILT PARKING PLAN", 0, 0)]) == [0]


def test_georef_recovers_pose_and_flags_bad_scale(plan):
    d, gt = plan
    plate = shape(gt["plate"])
    fp = affinity.translate(affinity.rotate(plate, 31, origin=(0, 0)), 400000, 2000000)
    f = fit_plan_to_footprint(plate, fp)
    assert f["iou"] > 0.99 and f["rms_m"] < 0.1
    big = affinity.scale(plate, 1.1, 1.1)
    f2 = fit_plan_to_footprint(big, fp, allow_scale=True)
    assert any("scale mismatch" in i for i in f2["issues"])


def test_cadastre_ids_and_unsanctioned_floor(plan, tmp_path):
    d, gt = plan
    plate = shape(gt["plate"])
    fp = affinity.translate(affinity.rotate(plate, 12, origin=(0, 0)), 373000, 2048000)
    bld = dict(geometry=fp, ground_z=560.0, roof_h_main=3.5 + 7 * 3.1, floors_est=8, confidence_3d=0.95)
    units, issues = build_cadastre(bld, [parse_plan(d / "p.pdf")], "27MH0PUN0A01", permissible_height_m=36)
    ids = [u["ulpin3d"] for u in units if u["ulpin3d"]]
    assert len(ids) == len(set(ids)) and all(verify(i)[0] for i in ids)
    lv7 = [u for u in units if u["level"] == 7]
    assert lv7 and lv7[0]["status"] == "unsanctioned"
    f402 = [u for u in units if u["label"] == "FLAT 402"]
    assert f402 and parse(f402[0]["ulpin3d"]).unit == 2 and parse(f402[0]["ulpin3d"]).level == 4
    assert any(u["zclass"] == "A" for u in units)
    cj = to_cityjson(units, 32643, tmp_path / "c.city.json", "B1", bld)
    assert cj["version"] == "2.0" and len(cj["CityObjects"]) == len(units) + 1


def test_declared_basements_become_underground_units():
    from shapely.geometry import box
    from ulpin3d.cadastre import build_cadastre
    from ulpin3d.ulpin import parse

    bld = dict(geometry=box(0, 0, 20, 12), ground_z=100.0, roof_h_main=9.5, floors_est=3, confidence_3d=0.95)
    us, _ = build_cadastre(bld, [], "27PUNEW10101", basements=2, basement_height_m=3.0)
    base = sorted([u for u in us if u["kind"] == "basement"], key=lambda u: u["level"], reverse=True)
    assert [u["level"] for u in base] == [-1, -2]
    assert (base[0]["z_min"], base[0]["z_max"]) == (97.0, 100.0) and (base[1]["z_min"], base[1]["z_max"]) == (94.0, 97.0)
    assert all(u["status"] == "provisional" and u["ulpin3d"] and "not observed by LiDAR" in u["issues"][0] for u in base)
    assert [parse(u["ulpin3d"]).level for u in base] == [-1, -2]
    assert len({u["ulpin3d"] for u in us}) == len(us)
    assert not [u for u in build_cadastre(bld, [], "27PUNEW10101")[0] if u["kind"] == "basement"]
