import itertools

import pytest

from ulpin3d.ulpin import SYM36, assign_units, level_code, level_from_code, luhn36_check, make, parse, verify


def test_matches_web_demo_vectors():
    # computed with the demo's JavaScript luhn36()
    assert luhn36_check("27MH0BKC0G01H0S01") == "5"
    assert luhn36_check("853ZQTS2B2B2H0S01") == "G"
    assert luhn36_check("27MH0BKC0G01H4AK0") == "K"


def test_level_codes():
    assert level_code(0) == "H0" and level_code(1) == "H1" and level_code(-1) == "GV" and level_code(-2) == "GU"
    for lv in range(-40, 200):
        assert level_from_code(level_code(lv)) == lv


def test_roundtrip_and_pretty():
    u = make("853ZQTS2B2B2", 4, "S", 2)
    s = str(u)
    assert len(s) == 18 and parse(s) == u
    assert u.pretty() == "853ZQTS2B2B2H4-S-02-" + s[-1]
    assert parse(u.pretty()) == u


def test_every_single_character_typo_is_caught():
    s = str(make("27MH0BKC0G01", 3, "S", 7))
    for i, c in itertools.product(range(len(s)), SYM36):
        if c != s[i]:
            assert not verify(s[:i] + c + s[i + 1:])[0]


def test_adjacent_swaps_mostly_caught():
    s = str(make("27MH0BKC0G01", 3, "S", 7))
    swaps = [s[:i] + s[i + 1] + s[i] + s[i + 2:] for i in range(len(s) - 1) if s[i] != s[i + 1]]
    caught = sum(not verify(t)[0] for t in swaps)
    assert caught / len(swaps) > 0.9


def test_assign_units_deterministic_and_common_area():
    units = [dict(level=1, x=0, y=10), dict(level=1, x=10, y=10), dict(level=1, x=0, y=0),
             dict(level=1, x=5, y=5, common=True), dict(level=0, x=0, y=0)]
    a = assign_units("27MH0BKC0G01", units)
    assert a == assign_units("27MH0BKC0G01", list(units))
    assert len(set(a)) == len(a)
    assert parse(a[0]).unit == 1 and parse(a[1]).unit == 2 and parse(a[2]).unit == 3
    assert parse(a[3]).unit == 0 and parse(a[4]).level == 0


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        make("SHORT", 0)
    with pytest.raises(ValueError):
        make("27MH0BKC0G01", 0, "X")
    assert make("27MH0BKC0G01AB", 0, "S", 1, parcel_chars=14).payload.startswith("27MH0BKC0G01AB")


def test_two_common_areas_get_distinct_ids():
    a = assign_units("27MH0BKC0G01", [dict(level=2, x=0, y=0, common=True), dict(level=2, x=9, y=0, common=True),
                                      dict(level=2, x=4, y=4)])
    assert len(set(a)) == 3


def test_plan_numbers_kept_as_unit_codes():
    a = assign_units("27MH0BKC0G01", [dict(level=4, x=0, y=0, preferred_unit=2), dict(level=4, x=5, y=0, preferred_unit=1),
                                      dict(level=4, x=9, y=0, preferred_unit=1), dict(level=4, x=3, y=3, common=True)])
    assert parse(a[0]).unit == 2 and parse(a[3]).unit == 0
    assert len(set(a)) == 4 and {parse(a[1]).unit, parse(a[2]).unit} == {3, 4}  # duplicate 1 -> renumbered
