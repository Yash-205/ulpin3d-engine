"""The 3D register as an indexed SQLite database (stdlib only, one file, opens instantly).

    python -m ulpin3d.cli build-db --out out/<area>        # -> out/<area>/register.sqlite

Why: units.geojson is the exchange format, but answering a query from it means parsing the whole
file and scanning every unit. A district has millions of units. The database keeps:

  units          one row per 3D unit; the original GeoJSON feature is stored verbatim, zlib-compressed
                 (seals are computed from it, so a card verifies the same either way)
  B-tree indexes ulpin3d (unique), parcel, building, (status, level), kind
  owner_fts      FTS5 trigram index on owner names, so "contains 'sharma'" is an index lookup, not a scan
  units_rtree    SQLite's built-in R*Tree in THREE dimensions: lon, lat and elevation. "Which unit
                 contains this 3D point?" (the vertical-cadastre question) becomes a box lookup that
                 touches only the few stacked units at that spot, then an exact polygon test.
  meta           area report, parcel id length, source file fingerprint (rebuilt when stale)

Built automatically by run-area and by the API when missing or older than units.geojson.
"""
from __future__ import annotations

import json
import math
import sqlite3
import zlib
from pathlib import Path

DB_NAME = "register.sqlite"
SCHEMA_VERSION = 2


def _clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _bounds(geom: dict):
    xs, ys = [], []

    def walk(c):
        if c and isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
        else:
            for q in c:
                walk(q)

    walk(geom["coordinates"])
    return min(xs), max(xs), min(ys), max(ys)


def _fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime_ns)}"


def build_db(out_dir, parcel_chars: int = 12, force: bool = False) -> Path:
    out = Path(out_dir)
    src = out / "units.geojson"
    db = out / DB_NAME
    if not force and is_fresh(out):
        return db
    tmp = db.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE units(
            id INTEGER PRIMARY KEY, ulpin3d TEXT, building_id TEXT, parcel_ulpin TEXT, level INTEGER,
            kind TEXT, status TEXT, owner_lc TEXT, z_min REAL, z_max REAL, feature BLOB NOT NULL);
        CREATE VIRTUAL TABLE units_rtree USING rtree(id, minx, maxx, miny, maxy, minz, maxz);
        CREATE VIRTUAL TABLE owner_fts USING fts5(owner, tokenize='trigram');
    """)
    fc = json.loads(src.read_text())
    rows, boxes = [], []
    for i, f in enumerate(fc["features"]):
        p = f["properties"]
        z0, z1 = _clean(p.get("z_min")), _clean(p.get("z_max"))
        lv = _clean(p.get("level"))
        rows.append((i, _clean(p.get("ulpin3d")), p.get("building_id"), (p.get("parcel_ulpin") or "").upper(),
                     int(lv) if lv is not None else None, p.get("kind"), p.get("status"),
                     str(_clean(p.get("owner_name")) or "").lower() or None, z0, z1,
                     zlib.compress(json.dumps(f, separators=(",", ":")).encode(), 6)))
        if f.get("geometry"):
            x0, x1, y0, y1 = _bounds(f["geometry"])
            boxes.append((i, x0, x1, y0, y1, z0 if z0 is not None else -1e9, z1 if z1 is not None else 1e9))
    con.executemany("INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.executemany("INSERT INTO units_rtree VALUES (?,?,?,?,?,?,?)", boxes)
    con.executemany("INSERT INTO owner_fts(rowid, owner) VALUES (?, ?)", [(r[0], r[7]) for r in rows if r[7]])
    con.executescript("""
        CREATE UNIQUE INDEX ix_ulpin ON units(ulpin3d) WHERE ulpin3d IS NOT NULL;
        CREATE INDEX ix_parcel ON units(parcel_ulpin, level);
        CREATE INDEX ix_building ON units(building_id);
        CREATE INDEX ix_status_level ON units(status, level);
        CREATE INDEX ix_kind ON units(kind);
        ANALYZE;
    """)
    report = json.loads((out / "area_report.json").read_text()) if (out / "area_report.json").exists() else {}
    meta = {"schema": str(SCHEMA_VERSION), "source": _fingerprint(src), "parcel_chars": str(parcel_chars),
            "report": json.dumps(report, default=str), "n_units": str(len(rows))}
    con.executemany("INSERT INTO meta VALUES (?,?)", meta.items())
    con.commit()
    con.execute("VACUUM")
    con.close()
    tmp.replace(db)
    return db


def is_fresh(out_dir) -> bool:
    out = Path(out_dir)
    db, src = out / DB_NAME, out / "units.geojson"
    if not db.exists() or not src.exists():
        return False
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        m = dict(con.execute("SELECT key, value FROM meta").fetchall())
        con.close()
    except sqlite3.Error:
        return False
    return m.get("schema") == str(SCHEMA_VERSION) and m.get("source") == _fingerprint(src)


class Register:
    """Read-only queries over register.sqlite. Thread-safe enough for the API (one connection per call)."""

    def __init__(self, out_dir, parcel_chars: int = 12):
        self.out = Path(out_dir)
        self.path = build_db(self.out, parcel_chars)
        self.uri = f"file:{self.path}?mode=ro&immutable=1"
        m = dict(self._q("SELECT key, value FROM meta"))
        self.report = json.loads(m.get("report") or "{}")
        self.n_units = int(m.get("n_units", 0))

    def _q(self, sql, args=()):
        con = sqlite3.connect(self.uri, uri=True, check_same_thread=False)
        try:
            return con.execute(sql, args).fetchall()
        finally:
            con.close()

    @staticmethod
    def _feat(row):
        return json.loads(zlib.decompress(row[0]))

    def n_ids(self) -> int:
        return self._q("SELECT count(*) FROM units WHERE ulpin3d IS NOT NULL")[0][0]

    def by_ulpin(self, uid: str):
        r = self._q("SELECT feature FROM units WHERE ulpin3d = ?", (uid,))
        return self._feat(r[0]) if r else None

    def units(self, building=None, parcel=None, status=None, level=None, kind=None, owner=None, limit=100):
        where, args = [], []
        for col, v in (("building_id", building), ("parcel_ulpin", parcel.upper() if parcel else None),
                       ("status", status), ("level", level), ("kind", kind)):
            if v is not None and v != "":
                where.append(f"{col} = ?")
                args.append(v)
        if owner:
            o = owner.lower()
            if len(o) >= 3 and not any(ch in o for ch in "%_\\"):  # trigram index: substring match, no row scan
                where.append("id IN (SELECT rowid FROM owner_fts WHERE owner LIKE ?)")
            else:
                o = o.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                where.append("owner_lc LIKE ? ESCAPE '\\'")
            args.append(f"%{o}%")
        sql = "SELECT feature FROM units" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id LIMIT ?"
        return [self._feat(r) for r in self._q(sql, (*args, int(limit)))]

    def parcel(self, parcel: str):
        return [self._feat(r) for r in self._q("SELECT feature FROM units WHERE parcel_ulpin = ? ORDER BY level, id",
                                                (parcel.upper(),))]

    def at_point(self, lon: float, lat: float, z: float | None = None):
        """Units whose outline contains (lon, lat) and, with z, whose height range contains z."""
        from shapely.geometry import Point, shape

        sql = ("SELECT u.feature FROM units_rtree r JOIN units u ON u.id = r.id "
               "WHERE r.minx <= ? AND r.maxx >= ? AND r.miny <= ? AND r.maxy >= ?")
        args = [lon, lon, lat, lat]
        if z is not None:
            sql += " AND r.minz <= ? AND r.maxz >= ?"
            args += [z, z]
        pt = Point(lon, lat)
        hits = [f for f in map(self._feat, self._q(sql, args)) if shape(f["geometry"]).intersects(pt)]
        if z is not None:  # units without a height range are never "at" a given height
            hits = [f for f in hits if _clean(f["properties"].get("z_min")) is not None
                    and f["properties"]["z_min"] <= z <= f["properties"]["z_max"]]
        return hits

    def summary(self):
        q = lambda col: dict(self._q(f"SELECT {col}, count(*) FROM units GROUP BY {col}"))
        levels = [r[0] for r in self._q("SELECT DISTINCT level FROM units WHERE level IS NOT NULL ORDER BY level")]
        return {"by_status": q("status"), "by_kind": q("kind"), "levels": levels}

    def explain(self, sql: str, args=()):
        return [r[-1] for r in self._q("EXPLAIN QUERY PLAN " + sql, args)]

