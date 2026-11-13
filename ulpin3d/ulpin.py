"""3D ULPIN: deterministic IDs for volumetric property units (slide 2 format).

    853ZQTS2B2B2 H0 - S - 01 - X
    └─parcel──┘ │    │   │    └ check character: Luhn mod-36 over everything before it
                │    │   └ unit code, base-32, 2 chars (00 = whole level / common area)
                │    └ class: S surface (on/above/below ground), A air-rights, I infrastructure
                └ level, base-32, 2 chars: 544 ("H0") = ground; 545 "H1" = 1st floor; 543 "GV" = B1

18 characters printed without separators: 853ZQTS2B2B2H0S01X.

IDs are never produced by a model. The AI decides *which* spaces exist; this
module names them, so the same space always gets the same ID and a single
mistyped character is caught offline (the Luhn mod-36 check). The arithmetic
matches the web demo (3d_ULPIN/index.html: luhn36, b32p2, floorP).

`parcel_chars` = 12 follows the slide (the ULPIN's last 2 characters carry the
level). Set it to 14 to keep the full national ULPIN plus a separate level
(20 characters); see README "ID format decision".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SYM36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
B32 = "0123456789ABCDEFGHIJKLMNOPQRSTUV"
GROUND_LEVEL = 544  # "H0"
CLASSES = {"S": "surface", "A": "air-rights", "I": "infrastructure"}


def luhn36_check(payload: str) -> str:
    """Luhn mod-N (N=36) check character, identical to the demo's luhn36()."""
    s, f = 0, 2
    for ch in reversed(payload.upper()):
        v = SYM36.find(ch)
        if v < 0:
            raise ValueError(f"illegal character {ch!r}")
        x = v * f
        if x >= 36:
            x = x // 36 + x % 36
        s += x
        f = 1 if f == 2 else 2
    return SYM36[(36 - s % 36) % 36]


def b32(n: int, width: int = 2) -> str:
    if n < 0 or n >= 32 ** width:
        raise ValueError(f"{n} does not fit in {width} base-32 chars")
    out = ""
    while n:
        out = B32[n % 32] + out
        n //= 32
    return out.rjust(width, "0")


def unb32(s: str) -> int:
    n = 0
    for ch in s.upper():
        v = B32.find(ch)
        if v < 0:
            raise ValueError(f"illegal base-32 char {ch!r}")
        n = n * 32 + v
    return n


def level_code(level: int) -> str:
    """0 = ground, 1 = first floor, -1 = basement 1 ... (range -544..+479)."""
    return b32(GROUND_LEVEL + level)


def level_from_code(code: str) -> int:
    return unb32(code) - GROUND_LEVEL


@dataclass(frozen=True)
class ULPIN3D:
    parcel: str  # base 2D parcel identifier (12 or 14 chars)
    level: int
    zclass: str
    unit: int

    @property
    def payload(self) -> str:
        return f"{self.parcel}{level_code(self.level)}{self.zclass}{b32(self.unit)}"

    @property
    def check(self) -> str:
        return luhn36_check(self.payload)

    def __str__(self) -> str:
        return self.payload + self.check

    def pretty(self) -> str:
        return f"{self.parcel}{level_code(self.level)}-{self.zclass}-{b32(self.unit)}-{self.check}"


def make(parcel: str, level: int, zclass: str = "S", unit: int = 1, parcel_chars: int = 12) -> ULPIN3D:
    parcel = parcel.upper()
    if len(parcel) != parcel_chars or not re.fullmatch(r"[0-9A-Z]+", parcel):
        raise ValueError(f"parcel must be {parcel_chars} alphanumeric chars, got {parcel!r}")
    if zclass not in CLASSES:
        raise ValueError(f"class must be one of {list(CLASSES)}")
    if not 0 <= unit < 1024:
        raise ValueError("unit must be 0..1023")
    level_code(level)  # range check
    return ULPIN3D(parcel, level, zclass, unit)


def parse(s: str, parcel_chars: int = 12) -> ULPIN3D:
    t = re.sub(r"[^0-9A-Za-z]", "", s).upper()
    n = parcel_chars + 6
    if len(t) != n:
        raise ValueError(f"a 3D ULPIN has {n} characters, got {len(t)}")
    ok, why = verify(t, parcel_chars)
    if not ok:
        raise ValueError(why)
    p = parcel_chars
    return ULPIN3D(t[:p], level_from_code(t[p:p + 2]), t[p + 2], unb32(t[p + 3:p + 5]))


def verify(s: str, parcel_chars: int = 12) -> tuple[bool, str]:
    t = re.sub(r"[^0-9A-Za-z]", "", s).upper()
    if len(t) != parcel_chars + 6:
        return False, f"length {len(t)}, expected {parcel_chars + 6}"
    if any(ch not in SYM36 for ch in t):
        return False, "illegal character"
    if t[parcel_chars + 2] not in CLASSES:
        return False, f"class {t[parcel_chars + 2]!r} not in S/A/I"
    exp = luhn36_check(t[:-1])
    if exp != t[-1]:
        return False, f"check character should be {exp}, not {t[-1]}"
    return True, "ok"


def assign_units(parcel: str, units, parcel_chars: int = 12):
    """Deterministic numbering inside a parcel.

    `units`: iterable of dicts with level, zclass, and a sort key (x, y), e.g. the
    unit centroid. Per (level, class) group, units are numbered 1.. in reading
    order (north to south, then west to east), so re-running the pipeline on the
    same building reproduces the same IDs. unit=0 is reserved for 'whole level /
    common area'. A plan label such as "402" is kept as an attribute, not used as
    the code, because plan labels repeat and skip numbers.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for i, u in enumerate(units):
        groups[(u["level"], u.get("zclass", "S"))].append((i, u))
    ids = [None] * sum(len(v) for v in groups.values())
    for (lvl, zc), items in groups.items():
        items.sort(key=lambda t: (-round(t[1]["y"], 1), round(t[1]["x"], 1)))
        regular = [t for t in items if not t[1].get("common")]
        common = [t for t in items if t[1].get("common")]
        # flats keep their plan number where it is clean and unique (FLAT 402 -> unit 02 on level 4)
        prefs = [t[1].get("preferred_unit") for t in regular]
        ok_pref = {p for p in prefs if p is not None and 0 < p < 1024 and prefs.count(p) == 1}
        # a number claimed by two flats is given to neither (avoids implying the wrong flat)
        used = set(ok_pref) | {0} | {p for p in prefs if p is not None}
        nxt = 1
        for i, u in regular:
            p = u.get("preferred_unit")
            if p in ok_pref:
                ids[i] = str(make(parcel, lvl, zc, p, parcel_chars))
                continue
            while nxt in used:
                nxt += 1
            used.add(nxt)
            ids[i] = str(make(parcel, lvl, zc, nxt, parcel_chars))
        # first common space on a level = unit 00 (whole level / common area);
        # further common spaces (2nd lobby, refuge area) get the next free numbers
        for j, (i, u) in enumerate(common):
            if j == 0:
                ids[i] = str(make(parcel, lvl, zc, 0, parcel_chars))
                continue
            while nxt in used:
                nxt += 1
            used.add(nxt)
            ids[i] = str(make(parcel, lvl, zc, nxt, parcel_chars))
    return ids
