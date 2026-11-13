"""Place a floor plan on the ground: fit the plan outline to the building footprint.

Plans are drawn in local sheet coordinates. The building footprint (Model 1 /
LiDAR fusion) is in UTM. We estimate a similarity transform (rotation, uniform
scale, translation; optionally a mirror for plans drawn flipped):

  1. coarse  align minimum rotated rectangles: 4 x 90 deg rotations (+ mirror)
  2. scale   sqrt(area_footprint / area_plan), or fixed at 1 when the plan scale
             is trusted
  3. refine  ICP on boundary samples (Umeyama closed form)
  4. score   IoU and RMS boundary distance; pick the best candidate

Quality gates (from the solution design):
  scale deviation > 3 %             -> "scale mismatch"   (plan scale is wrong, or not this building)
  RMS boundary distance > 0.5 m     -> "halt: surveyor"   (do not issue IDs automatically)
  IoU < 0.85                        -> "shape mismatch"   (extension / demolition / wrong plan)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import Polygon


def _boundary_pts(p: Polygon, step: float = 0.25):
    ring = p.exterior
    n = max(16, int(ring.length / step))
    return np.array([ring.interpolate(i / n, normalized=True).coords[0] for i in range(n)])


def _umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    c = (np.trace(np.diag(D) @ S) / s.var(0).sum()) if with_scale else 1.0
    t = mu_d - c * R @ mu_s
    return c, R, t


def _apply(geom, c, R, t):
    a, b, d, e = c * R[0, 0], c * R[0, 1], c * R[1, 0], c * R[1, 1]
    return affinity.affine_transform(geom, [a, b, d, e, t[0], t[1]])


def _mrr_angle(p):
    xs, ys = p.minimum_rotated_rectangle.exterior.coords.xy
    e = [(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(2)]
    v = max(e, key=lambda q: q[0] ** 2 + q[1] ** 2)
    return math.degrees(math.atan2(v[1], v[0]))


def fit_plan_to_footprint(plan_outline: Polygon, footprint: Polygon, allow_scale=True, allow_mirror=False,
                          icp_iters=30):
    """Returns dict(transform=(c, R, t), iou, rms_m, scale, mirrored, status, issues)."""
    best, cands = None, []
    fa = _mrr_angle(footprint)
    pa = _mrr_angle(plan_outline)
    s0 = math.sqrt(footprint.area / max(plan_outline.area, 1e-9)) if allow_scale else 1.0
    dst = _boundary_pts(footprint)
    tree = cKDTree(dst)
    for mirror in ([False, True] if allow_mirror else [False]):
        base = affinity.scale(plan_outline, -1, 1, origin=(0, 0)) if mirror else plan_outline
        pa_m = _mrr_angle(base)
        for k in range(4):
            ang = math.radians(fa - pa_m + 90 * k)
            R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            c = s0
            cs = np.array(base.centroid.coords[0])
            t = np.array(footprint.centroid.coords[0]) - c * R @ cs
            src = _boundary_pts(base)
            for _ in range(icp_iters):
                cur = (c * (R @ src.T)).T + t
                dist, idx = tree.query(cur)
                keep = dist < np.percentile(dist, 90) + 1e-6  # trim outliers (balconies, notches)
                c2, R2, t2 = _umeyama(src[keep], dst[idx[keep]], with_scale=allow_scale)
                if np.allclose([c2, *R2.ravel(), *t2], [c, *R.ravel(), *t], atol=1e-6):
                    break
                c, R, t = c2, R2, t2
            g = _apply(base, c, R, t)
            inter = g.intersection(footprint).area
            iou = inter / max(g.union(footprint).area, 1e-9)
            d1, _ = tree.query((c * (R @ src.T)).T + t)
            rms = float(np.sqrt(np.mean(d1 ** 2)))
            rot = abs((math.degrees(math.atan2(R[1, 0], R[0, 0])) + 180) % 360 - 180)
            cands.append(dict(c=c, R=R, t=t, iou=iou, rms_m=rms, mirrored=mirror, rot=rot))
    top = max(x["iou"] for x in cands)
    tied = [x for x in cands if x["iou"] >= top - 0.01]
    # symmetric outlines fit equally well rotated 180 deg: prefer the plan as drawn
    # (sanctioned plans are north-up), and say so
    best = min(tied, key=lambda x: (x["rot"], x["rms_m"]))
    issues = []
    if len({round(x["rot"]) for x in tied}) > 1:
        issues.append("orientation ambiguous (symmetric outline): assumed plan drawn north-up; "
                      "confirm with the north arrow or stair-cabin position in LiDAR")
    if allow_scale and abs(best["c"] - 1) > 0.03:
        issues.append(f"scale mismatch: plan is {best['c'] * 100:.1f}% of the footprint size")
    if best["rms_m"] > 0.5:
        issues.append(f"boundary RMS {best['rms_m']:.2f} m > 0.5 m: halt, surveyor review")
    if best["iou"] < 0.85:
        issues.append(f"shape mismatch: IoU {best['iou']:.2f} (extension, demolition or wrong plan?)")
    hard = [i for i in issues if not i.startswith("orientation")]
    status = "halt" if best["rms_m"] > 0.5 else ("flag" if hard else "ok")
    return {"transform": (best["c"], best["R"], best["t"]), "iou": round(best["iou"], 4),
            "rms_m": round(best["rms_m"], 3), "scale": round(float(best["c"]), 4), "mirrored": best["mirrored"],
            "rotation_deg": round(float(best["rot"]), 2), "ambiguous": any(i.startswith("orientation") for i in issues),
            "status": status, "issues": issues,
            "alternatives": [{"transform": (x["c"], x["R"], x["t"]), "mirrored": x["mirrored"], "iou": round(x["iou"], 4),
                              "rms_m": round(x["rms_m"], 3), "rotation_deg": round(float(x["rot"]), 2)} for x in tied]}


def resolve_orientation(fit, core_polys, roof_features, max_dist_m=4.0):
    """Settle a 180-degree ambiguity with LiDAR: the stair cabin / lift machine room on
    the roof sits above the staircase and lift drawn on the plan.

    core_polys: plan-coordinate polygons of STAIRCASE / LIFT; roof_features: list of
    (x, y) of raised roof structures in footprint coordinates. Returns the fit with the
    winning alternative, or the input fit unchanged if the evidence is unclear."""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    if not fit.get("ambiguous") or not core_polys or not roof_features:
        return fit
    core = unary_union(core_polys)
    scored = []
    for alt in fit["alternatives"]:
        g = apply_fit(core, alt)
        d = float(np.mean([g.distance(Point(x, y)) for x, y in roof_features]))
        scored.append((d, alt))
    scored.sort(key=lambda t: t[0])
    if scored[0][0] > max_dist_m or (len(scored) > 1 and scored[1][0] - scored[0][0] < 1.5):
        return fit
    d, alt = scored[0]
    out = dict(fit, **{k: alt[k] for k in ("transform", "mirrored", "iou", "rms_m", "rotation_deg")})
    out["ambiguous"] = False
    out["issues"] = [i for i in fit["issues"] if not i.startswith("orientation")]
    out["orientation_note"] = f"orientation confirmed: stair/lift core lies {d:.1f} m from the roof cabin seen in LiDAR"
    return out


def apply_fit(geom, fit):
    c, R, t = fit["transform"]
    if fit.get("mirrored"):
        geom = affinity.scale(geom, -1, 1, origin=(0, 0))
    return _apply(geom, c, R, t)
