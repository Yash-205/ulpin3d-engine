"""3D Property Card (Bhu-Aadhaar 3D): a printable, sealed record per 3D ULPIN.

    <out>/cards/<3D ULPIN>.html   one card per issued ID (opens from disk, prints to A4/PDF)
    <out>/cards/index.html        list of all cards
    <out>/cards/seals.json        {3D ULPIN: seal}: the register of seals

The seal is SHA-256 over the canonical record: the ID, parcel, building, level,
class, label, elevation range, areas, volume, status, ownership fields and a
hash of the unit's outline as stored in `units.geojson`. Change any of them and
the seal changes, so a card that does not match the register is detected
(`verify-card`). The QR code carries `ULPIN3D:<id>;SEAL:<first 32 hex>`.

A bare hash proves a card matches the register; it does not prove who issued it.
Set the environment variable ULPIN3D_SEAL_KEY (the issuing authority's secret)
and the seal becomes HMAC-SHA256, which cannot be recomputed without the key.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import math
import os
from pathlib import Path

from ulpin3d.ulpin import parse

SEALED = ("ulpin3d", "parcel_ulpin", "building_id", "level", "zclass", "kind", "label", "z_min", "z_max",
          "area_m2", "carpet_area_m2", "volume_m3", "bhk", "status", "owner_name", "share_pct", "right_type", "document")
QR_SEAL_CHARS = 32


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def record(feature: dict) -> dict:
    """Canonical sealed record from one feature of units.geojson."""
    p = feature["properties"]
    r = {k: _clean(p.get(k)) for k in SEALED}
    geom = json.dumps(feature["geometry"]["coordinates"], separators=(",", ":"))
    r["outline_sha256"] = hashlib.sha256(geom.encode()).hexdigest()
    return r


def seal(rec: dict, key: str | None = None) -> str:
    key = key if key is not None else os.environ.get("ULPIN3D_SEAL_KEY")
    msg = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if key:
        return hmac.new(key.encode(), msg, hashlib.sha256).hexdigest()
    return hashlib.sha256(msg).hexdigest()


def qr_payload(uid: str, s: str) -> str:
    return f"ULPIN3D:{uid};SEAL:{s[:QR_SEAL_CHARS].upper()}"


def _qr_svg(text: str) -> str:
    try:
        import segno
    except ImportError:  # the card still prints, with the seal as text
        return '<div class="noqr">QR unavailable<br>(pip install segno)</div>'
    import io

    buf = io.BytesIO()
    segno.make(text, error="m").save(buf, kind="svg", scale=4, border=2, xmldecl=False, svgns=True, nl=False)
    return buf.getvalue().decode()


def _centroid(geometry):
    from shapely.geometry import shape

    c = shape(geometry).centroid
    return c.x, c.y


def _level_name(lv):
    return "Ground floor" if lv == 0 else (f"Basement {-lv}" if lv < 0 else f"Floor {lv}")


def _g(v):
    """94.0 -> '94', 62.5 -> '62.5' (values that came through GeoJSON as floats)."""
    return f"{float(v):g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def _n(v, d=1, unit=""):
    return "—" if v is None else f"{float(v):,.{d}f}{unit}"


CSS = """
:root{--ink:#15252a;--muted:#56676c;--line:#c7d2cf;--bg:#e9efed;--card:#fff;--accent:#0b6a73;
--ok:#2e8a4c;--warn:#b97c0c;--bad:#bf3d2a;--unsanc:#7347c2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;padding:24px 16px}
.card{max-width:760px;margin:0 auto;background:var(--card);border:1px solid var(--line);border-top:6px solid var(--accent);padding:28px 32px}
.head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;border-bottom:1px solid var(--line);padding-bottom:16px}
.eyebrow{font-size:11px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
h1{font-size:22px;margin:4px 0 2px}.sub{color:var(--muted)}
.pill{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:4px 10px;border:1.5px solid;border-radius:999px;white-space:nowrap}
.validated{color:var(--ok)}.provisional{color:var(--warn)}.flagged{color:var(--bad)}.unsanctioned{color:var(--unsanc)}
.idrow{display:flex;gap:24px;align-items:center;margin:20px 0;flex-wrap:wrap}
.id{font:600 26px/1.2 "IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;letter-spacing:.04em;overflow-wrap:anywhere}
.parts{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:var(--muted)}
.parts b{display:block;font:600 14px "IBM Plex Mono",ui-monospace,Menlo,monospace;color:var(--ink)}
.qr{flex:none;width:148px;text-align:center;font-size:11px;color:var(--muted)}.qr svg{width:148px;height:148px;display:block}
.noqr{width:148px;height:148px;border:1px dashed var(--line);display:grid;place-items:center}
h2{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:22px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
dl{display:grid;grid-template-columns:170px 1fr;gap:6px 16px;margin:0}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}
.seal{font:12px/1.5 "IBM Plex Mono",ui-monospace,Menlo,monospace;overflow-wrap:anywhere;background:var(--bg);padding:10px 12px}
.note{font-size:12px;color:var(--muted);margin-top:14px}
.warnbox{border-left:4px solid var(--warn);background:#fbf3e0;padding:8px 12px;margin-top:14px;font-size:13px}
table{border-collapse:collapse;width:100%;background:var(--card)}td,th{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left}
td.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (max-width:560px){.card{padding:20px 16px}dl{grid-template-columns:1fr}dt{margin-top:6px}.id{font-size:20px}}
@media print{body{background:#fff;padding:0}.card{border:none;border-top:6px solid var(--accent);max-width:none}.noprint{display:none}}
"""


def card_html(feature: dict, area_name: str = "", parcel_chars: int = 12, key: str | None = None) -> tuple[str, str]:
    """Return (html, seal) for one unit feature that has a 3D ULPIN."""
    rec = record(feature)
    uid = rec["ulpin3d"]
    s = seal(rec, key)
    u = parse(uid, parcel_chars)
    lon, lat = _centroid(feature["geometry"])
    p = feature["properties"]
    e = lambda v: html.escape("—" if v is None else str(v))
    keyed = bool(key if key is not None else os.environ.get("ULPIN3D_SEAL_KEY"))
    kind = {"unit": "Apartment / unit", "common": "Common area", "whole_level": "Whole level",
            "air_rights": "Air rights volume"}.get(rec["kind"], rec["kind"])
    title = rec["label"] or kind
    h = (rec["z_max"] - rec["z_min"]) if rec["z_min"] is not None and rec["z_max"] is not None else None
    carpet = rec["carpet_area_m2"]
    rows = [("Kind", kind), ("Level", _level_name(u.level)),
            ("Elevation (above datum)", f"{_n(rec['z_min'], 2)} – {_n(rec['z_max'], 2)} m"),
            ("Storey height", _n(h, 2, " m")), ("Floor area", _n(rec["area_m2"], 1, " m²"))]
    if carpet is not None:
        rows.append(("Carpet area", f"{_n(carpet, 1)} m² · {float(carpet) * 10.7639:,.0f} sq ft"))
    rows += [("Volume", _n(rec["volume_m3"], 0, " m³"))]
    if rec["bhk"] is not None:
        rows.append(("Type", f"{_g(rec['bhk'])} BHK"))
    rows += [("Building", rec["building_id"]), ("Parent parcel (2D ULPIN)", rec["parcel_ulpin"]),
             ("Location (WGS84)", f"lat {lat:.6f}°, lon {lon:.6f}°")]
    own = [("Owner", rec["owner_name"]), ("Share", None if rec["share_pct"] is None else f"{_g(rec['share_pct'])} %"),
           ("Right", rec["right_type"]), ("Document", rec["document"])]
    dl = lambda rr: "<dl>" + "".join(f"<dt>{e(a)}</dt><dd>{e(b)}</dd>" for a, b in rr) + "</dl>"
    issues = _clean(p.get("issues"))
    warn = ""
    if rec["status"] != "validated":
        warn = (f'<div class="warnbox"><b>Status: {e(rec["status"])}.</b> This record is not yet validated and is '
                f'not proof of title.{" Findings: " + e(issues) + "." if issues else ""}</div>')
    conf = _clean(p.get("confidence"))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D Property Card {e(uid)}</title><style>{CSS}</style></head><body>
<div class="card">
<div class="head"><div><div class="eyebrow">Bhu-Aadhaar 3D · 3D Property Card</div>
<h1>{e(title)}</h1><div class="sub">{e(area_name)}</div></div>
<span class="pill {e(rec['status'])}">{e(rec['status'])}</span></div>
<div class="idrow"><div style="flex:1;min-width:240px"><div class="eyebrow">3D ULPIN</div>
<div class="id">{e(u.pretty())}</div>
<div class="parts"><span>parcel<b>{e(u.parcel)}</b></span><span>level<b>{e(uid[parcel_chars:parcel_chars + 2])}</b></span>
<span>class<b>{e(u.zclass)}</b></span><span>unit<b>{e(uid[parcel_chars + 3:parcel_chars + 5])}</b></span>
<span>check<b>{e(uid[-1])}</b></span></div></div>
<div class="qr">{_qr_svg(qr_payload(uid, s))}scan to verify</div></div>
<h2>Volumetric unit</h2>{dl(rows)}
<h2>Rights on record</h2>{dl(own)}
{warn}
<h2>Seal</h2><div class="seal">{'HMAC-SHA256' if keyed else 'SHA-256'}: {s}</div>
<div class="note">The seal covers the ID, parcel, building, level, elevations, areas, volume, status, ownership
fields and the unit outline. Check it against the register:
<code>python -m ulpin3d.cli verify-card --units units.geojson --id {e(uid)} --seal {s[:QR_SEAL_CHARS]}</code>.
{'' if keyed else 'This seal shows the card matches the register; an issuing-authority key (ULPIN3D_SEAL_KEY) makes it an authenticated seal.'}
Confidence of the automated survey: {e(conf)}.</div>
<p class="noprint note"><a href="index.html">All cards</a> · <a href="javascript:print()">Print / save as PDF</a></p>
</div></body></html>"""
    return doc, s


def write_cards(units_geojson, out_dir, area_name: str = "", parcel_chars: int = 12, ids=None, key: str | None = None) -> dict:
    """Write one card per issued 3D ULPIN (or only `ids`). Returns {id: seal}."""
    fc = json.loads(Path(units_geojson).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    want = {i.upper() for i in ids} if ids else None
    seals, rows = {}, []
    for f in fc["features"]:
        uid = _clean(f["properties"].get("ulpin3d"))
        if not uid or (want and uid not in want):
            continue
        doc, s = card_html(f, area_name, parcel_chars, key)
        (out / f"{uid}.html").write_text(doc, encoding="utf-8")
        seals[uid] = s
        p = f["properties"]
        rows.append((uid, p.get("label"), p.get("building_id"), p.get("status"), _clean(p.get("owner_name"))))
    if want and want - set(seals):
        raise KeyError(f"no issued 3D ULPIN in {units_geojson}: {sorted(want - set(seals))}")
    if not want:
        (out / "seals.json").write_text(json.dumps(seals, indent=1))
        e = lambda v: html.escape("—" if v is None else str(v))
        trs = "".join(f'<tr><td class="mono"><a href="{e(i)}.html">{e(i)}</a></td><td>{e(l)}</td><td>{e(b)}</td>'
                      f'<td class="{e(st)}">{e(st)}</td><td>{e(o)}</td></tr>' for i, l, b, st, o in sorted(rows))
        (out / "index.html").write_text(
            f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>3D Property Cards</title><style>{CSS}</style></head><body><div style='max-width:960px;margin:0 auto'>"
            f'<div class="eyebrow">Bhu-Aadhaar 3D</div><h1>3D Property Cards · {e(area_name)}</h1>'
            f'<p class="sub">{len(rows)} cards</p><div style="overflow-x:auto"><table><tr><th>3D ULPIN</th><th>Unit</th><th>Building</th>'
            f"<th>Status</th><th>Owner</th></tr>{trs}</table></div></div></body></html>", encoding="utf-8")
    return seals


def verify_feature(feature: dict | None, uid: str, claimed_seal: str, key: str | None = None) -> tuple[bool, str]:
    """Check a claimed seal against one register record (the feature whose 3D ULPIN is `uid`)."""
    claimed = claimed_seal.strip().lower()
    if len(claimed) < QR_SEAL_CHARS:
        return False, f"seal too short: need at least {QR_SEAL_CHARS} hex characters"
    if feature is None:
        return False, f"{uid} is not in the register"
    actual = seal(record(feature), key)
    if hmac.compare_digest(actual[:len(claimed)], claimed):
        return True, "seal matches the register"
    return False, "seal does NOT match the register: the card or the record was altered"


def verify_card(units_geojson, uid: str, claimed_seal: str, key: str | None = None) -> tuple[bool, str]:
    """Does the seal printed on a card (full, or the 32 hex chars in its QR) match the register?"""
    uid = "".join(c for c in uid if c.isalnum()).upper()
    feat = next((f for f in json.loads(Path(units_geojson).read_text())["features"]
                 if _clean(f["properties"].get("ulpin3d")) == uid), None)
    return verify_feature(feat, uid, claimed_seal, key)
