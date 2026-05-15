import os
import json
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import triangle as tr

from shapely.geometry import Point, box, Polygon, MultiPolygon
from shapely.affinity import rotate, scale
from shapely.ops import unary_union

def sample_two_part_nonsymmetric_layout():
    """
    Two-part asymmetric layout:
    - left and right parts
    - different shapes, sizes, and vertical offsets
    - NO symmetry
    """
    parts = []
    infos = []

    kinds = ["square", "rounded_square", "rectangle", "rounded_rectangle", "circle", "ellipse"]

    left_kind = np.random.choice(kinds)
    # force difference
    if np.random.rand() < 0.8:
        right_kind = np.random.choice([k for k in kinds if k != left_kind])
    else:
        right_kind = np.random.choice(kinds)

    # independent sizes
    left_sx = np.random.uniform(0.35, 1.0)
    left_sy = np.random.uniform(0.25, 0.85)

    right_sx = np.random.uniform(0.30, 1.1)
    right_sy = np.random.uniform(0.25, 0.95)

    # independent positions (key for non-symmetry)
    gap = np.random.uniform(0.3, 0.9)

    left_x = -gap * np.random.uniform(0.8, 1.3)
    right_x = gap * np.random.uniform(0.6, 1.5)

    left_y = np.random.uniform(-0.4, 0.4)
    right_y = np.random.uniform(-0.4, 0.4)

    def build(kind, center, sx, sy, role):
        cx, cy = center

        if kind == "square":
            s = min(sx, sy)
            geom = make_square(center=(cx, cy), half_size=s)
            info = {"role": role, "kind": "square", "center": (cx, cy), "half_size": s}

        elif kind == "circle":
            r = 0.5 * (sx + sy)
            geom = make_circle(center=(cx, cy), radius=r)
            info = {"role": role, "kind": "circle", "center": (cx, cy), "radius": r}

        elif kind == "rectangle":
            geom = make_rectangle(center=(cx, cy), hx=sx, hy=sy)
            info = {"role": role, "kind": "rectangle", "center": (cx, cy), "hx": sx, "hy": sy}

        elif kind == "ellipse":
            geom = make_ellipse(center=(cx, cy), rx=sx, ry=sy)
            info = {"role": role, "kind": "ellipse", "center": (cx, cy), "rx": sx, "ry": sy}

        elif kind == "rounded_square":
            s = min(sx, sy)
            r = np.random.uniform(0.2, 0.4) * s
            geom = make_rounded_square(center=(cx, cy), half_size=s, radius=r)
            info = {"role": role, "kind": "rounded_square", "center": (cx, cy), "half_size": s, "corner_radius": r}

        elif kind == "rounded_rectangle":
            r = np.random.uniform(0.2, 0.4) * min(sx, sy)
            geom = make_rounded_rectangle(center=(cx, cy), hx=sx, hy=sy, radius=r)
            info = {"role": role, "kind": "rounded_rectangle", "center": (cx, cy), "hx": sx, "hy": sy, "corner_radius": r}

        return geom, info

    left_geom, left_info = build(left_kind, (left_x, left_y), left_sx, left_sy, "left")
    right_geom, right_info = build(right_kind, (right_x, right_y), right_sx, right_sy, "right")

    parts.extend([left_geom, right_geom])
    infos.extend([left_info, right_info])

    geom = fix_geom(unary_union(parts))
    return geom, {"layout": "two_part_nonsymmetric", "parts": infos}

def sample_three_part_nonsymmetric_layout():
    """
    Always generate a non-symmetric 3-part composite shape.

    Typical examples:
      - left square + middle rectangle + right circle
      - left rounded rectangle + middle rectangle + right ellipse
      - left circle + middle rounded rectangle + right square

    Key property:
      no left-right mirror symmetry is enforced.
    """
    parts = []
    infos = []

    # -----------------------------
    # Middle body: usually a bridge
    # -----------------------------
    mid_kind = np.random.choice(
        ["rectangle", "rounded_rectangle"],
        p=[0.45, 0.55]
    )
    mid_hx = np.random.uniform(0.35, 0.85)
    mid_hy = np.random.uniform(0.16, 0.42)

    # Slight vertical offset so even the center is not always perfectly centered
    mid_y = np.random.uniform(-0.12, 0.12)

    if mid_kind == "rectangle":
        middle = make_rectangle(center=(0.0, mid_y), hx=mid_hx, hy=mid_hy)
        infos.append({
            "role": "middle",
            "kind": "rectangle",
            "center": (0.0, float(mid_y)),
            "hx": float(mid_hx),
            "hy": float(mid_hy),
        })
    else:
        mid_r = np.random.uniform(0.18, 0.40) * min(mid_hx, mid_hy)
        middle = make_rounded_rectangle(center=(0.0, mid_y), hx=mid_hx, hy=mid_hy, radius=mid_r)
        infos.append({
            "role": "middle",
            "kind": "rounded_rectangle",
            "center": (0.0, float(mid_y)),
            "hx": float(mid_hx),
            "hy": float(mid_hy),
            "corner_radius": float(mid_r),
        })

    parts.append(middle)

    # ---------------------------------
    # Left / right: intentionally different
    # ---------------------------------
    left_kind = np.random.choice(
        ["square", "rounded_square", "rectangle", "rounded_rectangle", "circle", "ellipse"],
        p=[0.18, 0.20, 0.12, 0.15, 0.18, 0.17]
    )

    # force right kind to often differ from left
    if np.random.rand() < 0.80:
        right_candidates = ["square", "rounded_square", "rectangle", "rounded_rectangle", "circle", "ellipse"]
        right_candidates.remove(left_kind)
        right_kind = np.random.choice(right_candidates)
    else:
        right_kind = np.random.choice(
            ["square", "rounded_square", "rectangle", "rounded_rectangle", "circle", "ellipse"]
        )

    # left part
    left_x = -(mid_hx + np.random.uniform(0.18, 0.60))
    left_y = mid_y + np.random.uniform(-0.32, 0.32)
    left_sx = np.random.uniform(0.32, 0.95)
    left_sy = np.random.uniform(0.24, 0.82)

    # right part -- independent, so not symmetric
    right_x = +(mid_hx + np.random.uniform(0.10, 0.72))
    right_y = mid_y + np.random.uniform(-0.38, 0.38)
    right_sx = np.random.uniform(0.28, 1.05)
    right_sy = np.random.uniform(0.22, 0.90)

    def build_part(kind, center, sx, sy, role):
        cx, cy = center

        if kind == "square":
            s = min(sx, sy)
            geom = make_square(center=(cx, cy), half_size=s)
            info = {
                "role": role,
                "kind": "square",
                "center": (float(cx), float(cy)),
                "half_size": float(s),
            }

        elif kind == "rounded_square":
            s = min(sx, sy)
            r = np.random.uniform(0.18, 0.38) * s
            geom = make_rounded_square(center=(cx, cy), half_size=s, radius=r)
            info = {
                "role": role,
                "kind": "rounded_square",
                "center": (float(cx), float(cy)),
                "half_size": float(s),
                "corner_radius": float(r),
            }

        elif kind == "rectangle":
            geom = make_rectangle(center=(cx, cy), hx=sx, hy=sy)
            info = {
                "role": role,
                "kind": "rectangle",
                "center": (float(cx), float(cy)),
                "hx": float(sx),
                "hy": float(sy),
            }

        elif kind == "rounded_rectangle":
            r = np.random.uniform(0.18, 0.38) * min(sx, sy)
            geom = make_rounded_rectangle(center=(cx, cy), hx=sx, hy=sy, radius=r)
            info = {
                "role": role,
                "kind": "rounded_rectangle",
                "center": (float(cx), float(cy)),
                "hx": float(sx),
                "hy": float(sy),
                "corner_radius": float(r),
            }

        elif kind == "circle":
            rad = 0.5 * (sx + sy)
            geom = make_circle(center=(cx, cy), radius=rad)
            info = {
                "role": role,
                "kind": "circle",
                "center": (float(cx), float(cy)),
                "radius": float(rad),
            }

        elif kind == "ellipse":
            geom = make_ellipse(center=(cx, cy), rx=sx, ry=sy)
            info = {
                "role": role,
                "kind": "ellipse",
                "center": (float(cx), float(cy)),
                "rx": float(sx),
                "ry": float(sy),
            }

        else:
            raise ValueError(kind)

        return geom, info

    left_geom, left_info = build_part(left_kind, (left_x, left_y), left_sx, left_sy, "left")
    right_geom, right_info = build_part(right_kind, (right_x, right_y), right_sx, right_sy, "right")

    parts.append(left_geom)
    parts.append(right_geom)
    infos.append(left_info)
    infos.append(right_info)

    # ---------------------------------------------------------
    # Optional extra bump / notch bias to create 凹 / 凸 feeling
    # ---------------------------------------------------------
    if np.random.rand() < 0.75:
        extra_kind = np.random.choice(["rectangle", "rounded_rectangle", "circle"], p=[0.35, 0.45, 0.20])

        # choose top or bottom, but not centered too perfectly
        side = np.random.choice(["top", "bottom"])
        ex = np.random.uniform(-0.45, 0.45) * max(mid_hx, 0.4)
        ey = np.random.uniform(0.45, 1.00) * max(mid_hy, 0.25)
        if side == "bottom":
            ey = -ey

        if extra_kind == "circle":
            rr = np.random.uniform(0.12, 0.35)
            extra = make_circle(center=(ex, ey), radius=rr)
            extra_info = {
                "role": side,
                "kind": "circle",
                "center": (float(ex), float(ey)),
                "radius": float(rr),
            }
        elif extra_kind == "rectangle":
            ehx = np.random.uniform(0.14, 0.40)
            ehy = np.random.uniform(0.10, 0.30)
            extra = make_rectangle(center=(ex, ey), hx=ehx, hy=ehy)
            extra_info = {
                "role": side,
                "kind": "rectangle",
                "center": (float(ex), float(ey)),
                "hx": float(ehx),
                "hy": float(ehy),
            }
        else:
            ehx = np.random.uniform(0.14, 0.40)
            ehy = np.random.uniform(0.10, 0.30)
            er = np.random.uniform(0.18, 0.38) * min(ehx, ehy)
            extra = make_rounded_rectangle(center=(ex, ey), hx=ehx, hy=ehy, radius=er)
            extra_info = {
                "role": side,
                "kind": "rounded_rectangle",
                "center": (float(ex), float(ey)),
                "hx": float(ehx),
                "hy": float(ehy),
                "corner_radius": float(er),
            }

        parts.append(extra)
        infos.append(extra_info)

    geom = fix_geom(unary_union(parts))
    return geom, {"layout": "three_part_nonsymmetric", "parts": infos}
# ============================================================
# Basic geometry helpers
# ============================================================
def fix_geom(geom):
    """Fix minor invalidities from shapely boolean / buffer operations."""
    if geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def get_main_polygon(geom):
    geom = fix_geom(geom)
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    raise ValueError(f"Unsupported geometry type: {type(geom)}")


def largest_polygon(geom):
    return get_main_polygon(geom)


def polygon_to_xy(geom):
    geom = get_main_polygon(geom)
    x, y = geom.exterior.xy
    return np.asarray(x), np.asarray(y)


def polygon_exterior_coords(poly):
    poly = get_main_polygon(poly)
    coords = np.asarray(poly.exterior.coords[:-1], dtype=np.float64)
    return coords


def ring_coords(poly):
    """
    Return exterior ring coords without repeated last point.
    """
    poly = get_main_polygon(poly)
    return np.asarray(poly.exterior.coords[:-1], dtype=np.float64)


def make_circle(center=(0.0, 0.0), radius=1.0):
    return Point(center[0], center[1]).buffer(radius, resolution=16)


def make_square(center=(0.0, 0.0), half_size=1.0):
    cx, cy = center
    return box(cx - half_size, cy - half_size, cx + half_size, cy + half_size)


def make_rectangle(center=(0.0, 0.0), hx=1.0, hy=0.5):
    cx, cy = center
    return box(cx - hx, cy - hy, cx + hx, cy + hy)


def make_ellipse(center=(0.0, 0.0), rx=1.0, ry=0.7):
    cx, cy = center
    unit = Point(cx, cy).buffer(1.0, resolution=16)
    return scale(unit, xfact=rx, yfact=ry, origin=(cx, cy))


def make_rounded_square(center=(0.0, 0.0), half_size=1.0, radius=0.25):
    cx, cy = center
    radius = min(radius, half_size) - 1e-6
    core = box(
        cx - (half_size - radius),
        cy - (half_size - radius),
        cx + (half_size - radius),
        cy + (half_size - radius),
    )
    return fix_geom(core.buffer(radius, resolution=16))


def make_rounded_rectangle(center=(0.0, 0.0), hx=1.0, hy=0.6, radius=0.2):
    cx, cy = center
    radius = min(radius, hx, hy) - 1e-6
    core = box(
        cx - (hx - radius),
        cy - (hy - radius),
        cx + (hx - radius),
        cy + (hy - radius),
    )
    return fix_geom(core.buffer(radius, resolution=16))


def clean_socket_hole_profile(profile, smooth_radius=0.00030, simplify_tol=0.00008):
    """
    Clean the plug profile before generating the socket cavity.
    Plug geometry stays unchanged; only the socket cavity is smoothed slightly.
    """
    g = get_main_polygon(profile)

    if smooth_radius > 0:
        g = fix_geom(g.buffer(smooth_radius, resolution=16))
        g = fix_geom(g.buffer(-smooth_radius, resolution=16))
        g = get_main_polygon(g)

    if simplify_tol > 0:
        g = fix_geom(g.simplify(simplify_tol, preserve_topology=True))
        g = get_main_polygon(g)

    g = fix_geom(g.buffer(0))
    g = get_main_polygon(g)
    return g


def make_socket_outer_from_inner(inner_profile, wall_thickness=0.004, outer_margin=0.0015):
    """
    Build the socket outer wall from the inner cavity profile.
    """
    outer_profile = fix_geom(inner_profile.buffer(wall_thickness + outer_margin, resolution=16))
    outer_profile = get_main_polygon(outer_profile)
    return outer_profile


# ============================================================
# Primitive sampling
# ============================================================
def sample_primitive(kind=None, center=(0.0, 0.0), size_scale=1.0):
    if kind is None:
        kind = np.random.choice(
            ["circle", "ellipse", "square", "rounded_square", "rectangle", "rounded_rectangle"],
            p=[0.1, 0.15, 0.1, 0.15, 0.25, 0.25]
        )

    cx, cy = center

    if kind == "circle":
        r = np.random.uniform(0.45, 0.95) * size_scale
        geom = make_circle(center=(cx, cy), radius=r)
        params = {"kind": kind, "radius": float(r)}

    elif kind == "ellipse":
        rx = np.random.uniform(0.50, 1.00) * size_scale
        ry = np.random.uniform(0.40, 0.90) * size_scale
        geom = make_ellipse(center=(cx, cy), rx=rx, ry=ry)
        params = {"kind": kind, "rx": float(rx), "ry": float(ry)}

    elif kind == "square":
        s = np.random.uniform(0.45, 0.90) * size_scale
        geom = make_square(center=(cx, cy), half_size=s)
        params = {"kind": kind, "half_size": float(s)}

    elif kind == "rounded_square":
        s = np.random.uniform(0.45, 0.90) * size_scale
        radius = np.random.uniform(0.18, 0.40) * s
        geom = make_rounded_square(center=(cx, cy), half_size=s, radius=radius)
        params = {"kind": kind, "half_size": float(s), "corner_radius": float(radius)}

    elif kind == "rectangle":
        hx = np.random.uniform(0.55, 1.15) * size_scale
        hy = np.random.uniform(0.35, 0.90) * size_scale
        geom = make_rectangle(center=(cx, cy), hx=hx, hy=hy)
        params = {"kind": kind, "hx": float(hx), "hy": float(hy)}

    elif kind == "rounded_rectangle":
        hx = np.random.uniform(0.55, 1.15) * size_scale
        hy = np.random.uniform(0.35, 0.90) * size_scale
        radius = np.random.uniform(0.20, 0.40) * min(hx, hy)
        geom = make_rounded_rectangle(center=(cx, cy), hx=hx, hy=hy, radius=radius)
        params = {"kind": kind, "hx": float(hx), "hy": float(hy), "corner_radius": float(radius)}

    else:
        raise ValueError(f"Unknown primitive kind: {kind}")

    return fix_geom(geom), params


# ============================================================
# Layout samplers
# ============================================================
def sample_single_layout():
    geom, info = sample_primitive(center=(0.0, 0.0), size_scale=1.0)
    return geom, {"layout": "single", "parts": [info]}


def sample_chain_layout():
    body_kind = np.random.choice(["rectangle", "rounded_rectangle"], p=[0.45, 0.55])

    body_hx = np.random.uniform(0.55, 1.10)
    body_hy = np.random.uniform(0.30, 0.75)

    if body_kind == "rectangle":
        body = make_rectangle(center=(0.0, 0.0), hx=body_hx, hy=body_hy)
        body_info = {"kind": "rectangle", "hx": float(body_hx), "hy": float(body_hy)}
    else:
        radius = np.random.uniform(0.20, 0.38) * min(body_hx, body_hy)
        body = make_rounded_rectangle(center=(0.0, 0.0), hx=body_hx, hy=body_hy, radius=radius)
        body_info = {
            "kind": "rounded_rectangle",
            "hx": float(body_hx),
            "hy": float(body_hy),
            "corner_radius": float(radius),
        }

    parts = [body]
    infos = [body_info]

    cap_types = ["none", "circle", "ellipse", "square", "rounded_square"]
    cap_probs = [0.18, 0.30, 0.12, 0.18, 0.22]

    left_kind = np.random.choice(cap_types, p=cap_probs)
    right_kind = np.random.choice(cap_types, p=cap_probs)

    x_left = -body_hx
    x_right = body_hx

    for side, kind, x_anchor in [
        ("left", left_kind, x_left),
        ("right", right_kind, x_right),
    ]:
        if kind == "none":
            infos.append({"side": side, "kind": "none"})
            continue

        size_scale = np.random.uniform(0.75, 1.35) * body_hy

        if kind == "circle":
            cap = make_circle(center=(x_anchor, 0.0), radius=size_scale)
            info = {"side": side, "kind": "circle", "radius": float(size_scale)}

        elif kind == "ellipse":
            rx = np.random.uniform(0.90, 1.25) * size_scale
            ry = np.random.uniform(0.70, 1.10) * size_scale
            cap = make_ellipse(center=(x_anchor, 0.0), rx=rx, ry=ry)
            info = {"side": side, "kind": "ellipse", "rx": float(rx), "ry": float(ry)}

        elif kind == "square":
            cap = make_square(center=(x_anchor, 0.0), half_size=size_scale)
            info = {"side": side, "kind": "square", "half_size": float(size_scale)}

        elif kind == "rounded_square":
            radius = np.random.uniform(0.18, 0.38) * size_scale
            cap = make_rounded_square(center=(x_anchor, 0.0), half_size=size_scale, radius=radius)
            info = {
                "side": side,
                "kind": "rounded_square",
                "half_size": float(size_scale),
                "corner_radius": float(radius),
            }
        else:
            raise ValueError(kind)

        parts.append(cap)
        infos.append(info)

    geom = fix_geom(unary_union(parts))
    return geom, {"layout": "chain", "parts": infos}


def sample_double_lobe_layout():
    kind = np.random.choice(["circle", "ellipse", "rounded_square"], p=[0.60, 0.20, 0.20])

    s1 = np.random.uniform(0.50, 0.95)
    s2 = np.random.uniform(0.50, 0.95)
    d = np.random.uniform(0.45, 1.15) * min(s1, s2)

    c1 = (-d / 2.0, 0.0)
    c2 = (d / 2.0, 0.0)

    if kind == "circle":
        p1 = make_circle(center=c1, radius=s1)
        p2 = make_circle(center=c2, radius=s2)
        info1 = {"kind": "circle", "center": c1, "radius": float(s1)}
        info2 = {"kind": "circle", "center": c2, "radius": float(s2)}

    elif kind == "ellipse":
        rx1 = np.random.uniform(0.90, 1.20) * s1
        ry1 = np.random.uniform(0.70, 1.10) * s1
        rx2 = np.random.uniform(0.90, 1.20) * s2
        ry2 = np.random.uniform(0.70, 1.10) * s2
        p1 = make_ellipse(center=c1, rx=rx1, ry=ry1)
        p2 = make_ellipse(center=c2, rx=rx2, ry=ry2)
        info1 = {"kind": "ellipse", "center": c1, "rx": float(rx1), "ry": float(ry1)}
        info2 = {"kind": "ellipse", "center": c2, "rx": float(rx2), "ry": float(ry2)}

    else:
        r1 = np.random.uniform(0.18, 0.35) * s1
        r2 = np.random.uniform(0.18, 0.35) * s2
        p1 = make_rounded_square(center=c1, half_size=s1, radius=r1)
        p2 = make_rounded_square(center=c2, half_size=s2, radius=r2)
        info1 = {"kind": "rounded_square", "center": c1, "half_size": float(s1), "corner_radius": float(r1)}
        info2 = {"kind": "rounded_square", "center": c2, "half_size": float(s2), "corner_radius": float(r2)}

    geom = fix_geom(unary_union([p1, p2]))
    return geom, {"layout": "double_lobe", "parts": [info1, info2]}


def sample_clover4_layout():
    kind = np.random.choice(["circle", "ellipse", "rounded_square"], p=[0.72, 0.10, 0.18])

    r = np.random.uniform(0.22, 0.55)
    offset = np.random.uniform(0.30, 0.78)

    centers = [
        (+offset, 0.0),
        (-offset, 0.0),
        (0.0, +offset),
        (0.0, -offset),
    ]

    parts = []
    infos = []

    for c in centers:
        if kind == "circle":
            g = make_circle(center=c, radius=r)
            info = {"kind": "circle", "center": c, "radius": float(r)}

        elif kind == "ellipse":
            rx = np.random.uniform(0.90, 1.15) * r
            ry = np.random.uniform(0.75, 1.05) * r
            g = make_ellipse(center=c, rx=rx, ry=ry)
            info = {"kind": "ellipse", "center": c, "rx": float(rx), "ry": float(ry)}

        else:
            cr = np.random.uniform(0.18, 0.36) * r
            g = make_rounded_square(center=c, half_size=r, radius=cr)
            info = {"kind": "rounded_square", "center": c, "half_size": float(r), "corner_radius": float(cr)}

        parts.append(g)
        infos.append(info)

    if np.random.rand() < 0.35:
        center_r = np.random.uniform(0.06, 0.22)
        parts.append(make_circle(center=(0.0, 0.0), radius=center_r))
        infos.append({"kind": "circle", "center": (0.0, 0.0), "radius": float(center_r)})

    geom = fix_geom(unary_union(parts))
    return geom, {"layout": "clover4", "parts": infos}


# ============================================================
# Filtering + normalization
# ============================================================
def shape_metrics(geom):
    area = geom.area
    hull_area = geom.convex_hull.area
    convexity = area / hull_area if hull_area > 1e-8 else 0.0

    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    aspect = max(w, h) / max(min(w, h), 1e-8)

    return {
        "area": area,
        "convexity": convexity,
        "aspect": aspect,
        "width": w,
        "height": h,
    }


def is_good_shape(geom, min_convexity=0.55, max_aspect=3.6, min_area=0.20):
    if geom.is_empty:
        return False
    m = shape_metrics(geom)
    if m["convexity"] < min_convexity:
        return False
    if m["aspect"] > max_aspect:
        return False
    if m["area"] < min_area:
        return False
    return True

def postprocess_shape(geom):
    if np.random.rand() < 0.40:
        eps = np.random.uniform(0.015, 0.045)
        geom = geom.buffer(eps, resolution=16).buffer(-eps, resolution=16)

    if np.random.rand() < 0.35:
        sx = np.random.uniform(0.90, 1.15)
        sy = np.random.uniform(0.90, 1.15)
        geom = scale(geom, xfact=sx, yfact=sy, origin=(0.0, 0.0))

    rot_deg = np.random.uniform(0.0, 180.0)
    geom = rotate(geom, rot_deg, origin=(0.0, 0.0), use_radians=False)

    return fix_geom(geom), {"rotation_deg": float(rot_deg)}


def normalize_shape_to_bbox(geom, target_extent=0.010):
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    scale_factor = target_extent / max(w, h)
    geom = scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(0.0, 0.0))
    return fix_geom(geom), float(scale_factor)


# ============================================================
# One-shape sampler
# ============================================================
def sample_shape_raw():
    layout = np.random.choice(
        ["three_part_nonsymmetric", "two_part_nonsymmetric", "single", "clover4"],
        p=[0.25,0.4,0.2,0.15]
    )

    if layout == "three_part_nonsymmetric":
        geom, meta = sample_three_part_nonsymmetric_layout()
    elif layout == "two_part_nonsymmetric":
        geom, meta = sample_two_part_nonsymmetric_layout()
    elif layout == "single":
        geom, meta = sample_single_layout()
    elif layout == "clover4":
        geom, meta = sample_clover4_layout()
    else:
        raise ValueError(layout)

    geom, post = postprocess_shape(geom)
    meta.update(post)
    return geom, meta

def sample_valid_shape(max_tries=300, target_extent=0.010):
    for _ in range(max_tries):
        geom, meta = sample_shape_raw()
        if not is_good_shape(geom):
            continue

        geom, norm_scale = normalize_shape_to_bbox(geom, target_extent=target_extent)
        meta.update(shape_metrics(geom))
        meta["norm_scale"] = float(norm_scale)
        return geom, meta

    raise RuntimeError("Failed to sample a valid shape.")


# ============================================================
# Triangle-based socket meshing
# ============================================================
def ensure_ccw(coords):
    """
    Ensure polygon coords are counterclockwise.
    coords shape: [N, 2], no repeated last point
    """
    area2 = 0.0
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    if area2 < 0:
        return coords[::-1].copy()
    return coords.copy()


def ensure_cw(coords):
    ccw = ensure_ccw(coords)
    return ccw[::-1].copy()


def segments_from_ring(start_idx, count):
    segs = []
    for i in range(count):
        segs.append([start_idx + i, start_idx + ((i + 1) % count)])
    return np.asarray(segs, dtype=np.int32)


def build_annulus_2d_mesh_triangle(
    outer_profile,
    inner_profile,
    max_area=None,
):
    """
    Constrained triangulation of:
        outer_profile - inner_profile
    """
    outer_xy = ring_coords(outer_profile)
    inner_xy = ring_coords(inner_profile)

    # Triangle likes explicit boundaries.
    outer_xy = ensure_ccw(outer_xy)
    inner_xy = ensure_cw(inner_xy)

    vertices = np.vstack([outer_xy, inner_xy])

    outer_segments = segments_from_ring(0, len(outer_xy))
    inner_offset = len(outer_xy)
    inner_segments = segments_from_ring(inner_offset, len(inner_xy))
    segments = np.vstack([outer_segments, inner_segments])

    hole_pt = np.array([[inner_profile.representative_point().x,
                         inner_profile.representative_point().y]], dtype=np.float64)

    A = {
        "vertices": vertices,
        "segments": segments,
        "holes": hole_pt,
    }

    if max_area is None:
        mesh2d = tr.triangulate(A, "pq")
    else:
        mesh2d = tr.triangulate(A, f"pqa{max_area}")

    if "triangles" not in mesh2d or len(mesh2d["triangles"]) == 0:
        raise RuntimeError("triangle failed to generate 2D annulus mesh.")

    return mesh2d, outer_segments, inner_segments, outer_xy, inner_xy


# def extrude_annulus_socket(
#     mesh2d,
#     outer_segments,
#     inner_segments,
#     cavity_depth,
#     socket_height,
# ):
#     """
#     Build a true socket:
#     - top surface ring at z = socket_height
#     - cavity walls from z = socket_height down to z = socket_height - cavity_depth
#     - closed bottom floor under the cavity
#     - outer walls around the whole socket height
#     - bottom surface

#     Coordinate convention:
#     z in [0, socket_height]
#     top opening at z = socket_height
#     """
#     if cavity_depth <= 0 or cavity_depth > socket_height:
#         raise ValueError("cavity_depth must be in (0, socket_height].")

#     v2 = mesh2d["vertices"]
#     t2 = mesh2d["triangles"]
#     n = len(v2)

#     z_bottom = 0.0
#     z_cavity_floor = socket_height - cavity_depth
#     z_top = socket_height

#     vb = np.column_stack([v2, np.full(n, z_bottom)])
#     vf = np.column_stack([v2, np.full(n, z_cavity_floor)])
#     vt = np.column_stack([v2, np.full(n, z_top)])

#     vertices3d = np.vstack([vb, vf, vt])

#     idx_b = 0
#     idx_f = n
#     idx_t = 2 * n

#     faces = []

#     # Bottom surface: full annulus at z_bottom
#     for tri in t2:
#         faces.append([tri[0] + idx_b, tri[2] + idx_b, tri[1] + idx_b])

#     # Cavity floor: full annulus at z_cavity_floor
#     # This creates the "bottom" of the socket material under the hole ring.
#     for tri in t2:
#         faces.append([tri[0] + idx_f, tri[1] + idx_f, tri[2] + idx_f])

#     # Top surface: only outer wall ring should exist, not across the hole.
#     # Reuse annulus triangles at z_top.
#     for tri in t2:
#         faces.append([tri[0] + idx_t, tri[1] + idx_t, tri[2] + idx_t])

#     # Outer walls: full height z_bottom -> z_top
#     for a, b in outer_segments:
#         faces.append([a + idx_b, b + idx_b, b + idx_t])
#         faces.append([a + idx_b, b + idx_t, a + idx_t])

#     # Inner cavity walls: only from cavity floor -> top
#     for a, b in inner_segments:
#         faces.append([a + idx_f, b + idx_t, b + idx_f])
#         faces.append([a + idx_f, a + idx_t, b + idx_t])

#     faces = np.asarray(faces, dtype=np.int64)

#     mesh3d = trimesh.Trimesh(vertices=vertices3d, faces=faces, process=True)
#     mesh3d.remove_duplicate_faces()
#     mesh3d.remove_degenerate_faces()
#     mesh3d.remove_unreferenced_vertices()
#     mesh3d.fix_normals()
#     return mesh3d

def build_filled_2d_mesh_triangle(profile, max_area=None):
    """
    Triangulate a filled polygon, no hole.
    """
    xy = ring_coords(profile)
    xy = ensure_ccw(xy)

    segments = segments_from_ring(0, len(xy))

    A = {
        "vertices": xy,
        "segments": segments,
    }

    if max_area is None:
        mesh2d = tr.triangulate(A, "pq")
    else:
        mesh2d = tr.triangulate(A, f"pqa{max_area}")

    if "triangles" not in mesh2d or len(mesh2d["triangles"]) == 0:
        raise RuntimeError("triangle failed to generate filled 2D mesh.")

    return mesh2d


def extrude_annulus_socket(
    mesh2d,
    outer_segments,
    inner_segments,
    cavity_depth,
    socket_height,
    outer_profile=None,
    inner_profile=None,
    triangle_max_area=None,
):
    """
    Build a true printable socket:

    - top surface: annulus, outer minus inner hole
    - outer wall: full height
    - inner wall: cavity floor to top
    - cavity floor: filled inner profile
    - bottom: filled outer profile

    Coordinate convention:
        z in [0, socket_height]
        top opening at z = socket_height
    """
    if cavity_depth <= 0 or cavity_depth > socket_height:
        raise ValueError("cavity_depth must be in (0, socket_height].")

    if outer_profile is None or inner_profile is None:
        raise ValueError("Need outer_profile and inner_profile to close bottom/floor.")

    v_ann = mesh2d["vertices"]
    t_ann = mesh2d["triangles"]
    n_ann = len(v_ann)

    z_bottom = 0.0
    z_cavity_floor = socket_height - cavity_depth
    z_top = socket_height

    vertices_all = []
    faces = []

    # ============================================================
    # 1. Annulus vertices at top
    # ============================================================
    idx_top = len(vertices_all)
    vt = np.column_stack([v_ann, np.full(n_ann, z_top)])
    vertices_all.append(vt)

    # Top annulus surface
    for tri in t_ann:
        faces.append([
            idx_top + tri[0],
            idx_top + tri[1],
            idx_top + tri[2],
        ])

    # ============================================================
    # 2. Annulus vertices at cavity floor
    #    Used only for inner wall connection.
    # ============================================================
    idx_floor_ann = idx_top + n_ann
    vf_ann = np.column_stack([v_ann, np.full(n_ann, z_cavity_floor)])
    vertices_all.append(vf_ann)

    # Inner cavity wall: from cavity floor to top
    for a, b in inner_segments:
        faces.append([idx_floor_ann + a, idx_floor_ann + b, idx_top + b])
        faces.append([idx_floor_ann + a, idx_top + b, idx_top + a])

    # ============================================================
    # 3. Outer wall needs bottom and top outer ring
    # ============================================================
    idx_bottom_ann = idx_floor_ann + n_ann
    vb_ann = np.column_stack([v_ann, np.full(n_ann, z_bottom)])
    vertices_all.append(vb_ann)

    # Outer walls: bottom to top
    for a, b in outer_segments:
        faces.append([idx_bottom_ann + a, idx_bottom_ann + b, idx_top + b])
        faces.append([idx_bottom_ann + a, idx_top + b, idx_top + a])

    # ============================================================
    # 4. Bottom surface: filled outer profile
    # ============================================================
    outer_mesh2d = build_filled_2d_mesh_triangle(
        outer_profile,
        max_area=triangle_max_area,
    )

    v_outer = outer_mesh2d["vertices"]
    t_outer = outer_mesh2d["triangles"]

    idx_outer_bottom = idx_bottom_ann + n_ann
    vb_outer = np.column_stack([v_outer, np.full(len(v_outer), z_bottom)])
    vertices_all.append(vb_outer)

    # Bottom face should point downward, so reverse winding
    for tri in t_outer:
        faces.append([
            idx_outer_bottom + tri[0],
            idx_outer_bottom + tri[2],
            idx_outer_bottom + tri[1],
        ])

    # ============================================================
    # 5. Cavity floor: filled inner profile
    # ============================================================
    inner_mesh2d = build_filled_2d_mesh_triangle(
        inner_profile,
        max_area=triangle_max_area,
    )

    v_inner = inner_mesh2d["vertices"]
    t_inner = inner_mesh2d["triangles"]

    idx_inner_floor = idx_outer_bottom + len(v_outer)
    vf_inner = np.column_stack([v_inner, np.full(len(v_inner), z_cavity_floor)])
    vertices_all.append(vf_inner)

    # Cavity floor should point upward
    for tri in t_inner:
        faces.append([
            idx_inner_floor + tri[0],
            idx_inner_floor + tri[1],
            idx_inner_floor + tri[2],
        ])

    # ============================================================
    # Final mesh cleanup
    # ============================================================
    vertices3d = np.vstack(vertices_all)
    faces = np.asarray(faces, dtype=np.int64)

    mesh3d = trimesh.Trimesh(vertices=vertices3d, faces=faces, process=True)

    mesh3d.update_faces(mesh3d.unique_faces())
    mesh3d.remove_degenerate_faces()
    mesh3d.remove_unreferenced_vertices()
    mesh3d.merge_vertices()
    mesh3d.fix_normals()

    return mesh3d

# ============================================================
# Plug / socket generation
# ============================================================
def create_plug_with_grasp_post(
    profile,
    insertion_height=0.030,
    grasp_post_height=0.1,
    grasp_post_radius=0.004,
    post_blend_height=0.004,
):
    """
    Plug is not constant cross-section all the way.
    Bottom: diverse insertion shape.
    Top: centered simple cylindrical post for easy grasping.
    """
    profile = fix_geom(profile)

    insertion_mesh = trimesh.creation.extrude_polygon(profile, height=insertion_height)

    blend_radius = max(grasp_post_radius * 1.15, grasp_post_radius + 0.0008)
    blend_mesh = trimesh.creation.cylinder(radius=blend_radius, height=post_blend_height, sections=64)
    blend_mesh.apply_translation([0.0, 0.0, insertion_height + post_blend_height / 2.0])

    post_mesh = trimesh.creation.cylinder(radius=grasp_post_radius, height=grasp_post_height, sections=64)
    post_mesh.apply_translation([0.0, 0.0, insertion_height + post_blend_height + grasp_post_height / 2.0])

    plug_mesh = trimesh.util.concatenate([insertion_mesh, blend_mesh, post_mesh])
    plug_mesh.update_faces(plug_mesh.unique_faces())
    plug_mesh.remove_duplicate_faces()
    plug_mesh.remove_degenerate_faces()
    plug_mesh.remove_unreferenced_vertices()
    plug_mesh.fix_normals()

    total_height = insertion_height + post_blend_height + grasp_post_height
    plug_mesh.apply_translation([0.0, 0.0, total_height / 2.0])

    meta = {
        "insertion_height": float(insertion_height),
        "grasp_post_height": float(grasp_post_height),
        "post_blend_height": float(post_blend_height),
        "grasp_post_radius": float(grasp_post_radius),
        "plug_total_height": float(total_height),
    }
    return plug_mesh, meta


def create_socket_from_profile(
    profile,
    socket_height=0.05,
    clearance=0.0006,
    wall_thickness=0.004,
    entrance_relief=0.0005,   # kept for metadata compatibility; not used in this simpler version
    entrance_depth=0.003,     # used as cavity depth
    outer_margin=0.0015,
    socket_smooth_radius=0.00030,
    socket_simplify_tol=0.00008,
    trim_extra_height=0.003,  # kept for metadata compatibility; not used
    triangle_max_area=None,
):
    """
    New socket construction:
    1. clean plug profile
    2. offset for clearance -> hole_profile
    3. offset again outward -> outer_profile
    4. triangulate annulus directly using triangle
    5. extrude into a true 3D socket mesh

    No raw extrude_polygon on annulus.
    """
    profile = get_main_polygon(profile)

    # clean_profile = clean_socket_hole_profile(
    #     profile,
    #     smooth_radius=socket_smooth_radius,
    #     simplify_tol=socket_simplify_tol,
    # )
    clean_profile = profile

    hole_profile = get_main_polygon(
        fix_geom(clean_profile.buffer(clearance, resolution=16))
    )

    outer_profile = make_socket_outer_from_inner(
        hole_profile,
        wall_thickness=wall_thickness,
        outer_margin=outer_margin,
    )

    # choose a conservative max area if not provided
    if triangle_max_area is None:
        bbox = outer_profile.bounds
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        scale_ref = max(w, h)
        triangle_max_area = (0.06 * scale_ref) ** 2

    mesh2d, outer_segments, inner_segments, outer_xy, inner_xy = build_annulus_2d_mesh_triangle(
        outer_profile=outer_profile,
        inner_profile=hole_profile,
        max_area=triangle_max_area,
    )

    cavity_depth = min(max(entrance_depth, 1e-5), socket_height)

    # socket_mesh = extrude_annulus_socket(
    #     mesh2d=mesh2d,
    #     outer_segments=outer_segments,
    #     inner_segments=inner_segments,
    #     cavity_depth=cavity_depth,
    #     socket_height=socket_height,
    # )
    socket_mesh = extrude_annulus_socket(
        mesh2d=mesh2d,
        outer_segments=outer_segments,
        inner_segments=inner_segments,
        cavity_depth=cavity_depth,
        socket_height=socket_height,
        outer_profile=outer_profile,
        inner_profile=hole_profile,
        triangle_max_area=triangle_max_area,
    )

    meta = {
        "socket_height": float(socket_height),
        "clearance": float(clearance),
        "wall_thickness": float(wall_thickness),
        "entrance_relief": float(entrance_relief),
        "entrance_depth": float(cavity_depth),
        "outer_margin": float(outer_margin),
        "socket_smooth_radius": float(socket_smooth_radius),
        "socket_simplify_tol": float(socket_simplify_tol),
        "trim_extra_height": float(trim_extra_height),
        "triangle_max_area": float(triangle_max_area),
        "socket_construction": "triangle_constrained_annulus_extrusion",
    }

    socket_profile = get_main_polygon(fix_geom(outer_profile.difference(hole_profile)))
    return socket_mesh, hole_profile, socket_profile, meta


# ============================================================
# Export + asset-format helpers
# ============================================================
def write_mat_socket(path):
    with open(path, "w") as f:
        f.write("newmtl mat0\n")
        f.write("Ka 0.5000 0.5000 0.5000\n")
        f.write("Kd 0.0823529 0.47451 0.329412\n")
        f.write("illum 1\n")


# def write_mat_plug(path):
#     with open(path, "w") as f:
#         f.write("newmtl mat0\n")
#         f.write("Ka 0.5000 0.5000 0.5000\n")
#         f.write("Kd 0.964706 0.933333 0.898039\n")
#         f.write("illum 1\n")

def write_mat_plug(path):
    with open(path, "w") as f:
        f.write("newmtl mat0\n")
        f.write("Ka 0.2 0.0 0.0\n")        # ambient (dark red)
        f.write("Kd 1.0 0.0 0.0\n")        # diffuse (red)
        f.write("illum 1\n")

# def write_mat_socket(path):
#     with open(path, "w") as f:
#         f.write("newmtl mat0\n")
#         f.write("Ka 0.0 0.0 0.0\n")   # ambient
#         f.write("Kd 0.0 0.0 0.0\n")   # diffuse
#         f.write("illum 1\n")


# def write_mat_plug(path):
#     with open(path, "w") as f:
#         f.write("newmtl mat0\n")
#         f.write("Ka 0.0 0.0 0.0\n")   # ambient
#         f.write("Kd 0.0 0.0 0.0\n")   # diffuse
#         f.write("illum 1\n")

def attach_material(obj_path, mtl):
    with open(obj_path, "r") as f:
        content = f.read()

    header = f"mtllib {mtl}\nusemtl mat0\n"

    with open(obj_path, "w") as f:
        f.write(header + content)

def generate_grasp(
    insertion_height,
    post_blend_height,
    grasp_post_height,
    asset_id="",
    xy_noise=0.0015,
    z_clearance_min=0.04,
    z_clearance_max=0.07,
    yaw_noise=0.08,
):
    """
    Generate a pre-grasp pose clearly above the actual top of the plug/grasp post.

    Important:
    In create_plug_with_grasp_post(...), the whole plug mesh is translated by
        total_height / 2
    after construction.

    Before translation:
        plug spans z in [0, total_height]
    After translation:
        plug spans z in [total_height/2, 3*total_height/2]

    So the actual top of the mesh is:
        top_z = 1.5 * total_height
    """
    total_height = insertion_height + post_blend_height + grasp_post_height
    top_z = 1.5 * total_height

    x = np.random.uniform(-xy_noise, xy_noise)
    y = np.random.uniform(-xy_noise, xy_noise)
    z = top_z + np.random.uniform(z_clearance_min, z_clearance_max) + np.random.uniform(0.025,0.045)

    qx = 0.0
    qy = 1.0
    qz = np.random.uniform(-yaw_noise, yaw_noise)
    qw = 0.0

    return {
        f"asset_{asset_id}": [x, y, z, qx, qy, qz, qw]
    }

def save_asset_bundle(
    save_dir,
    asset_id="10001",
    target_extent=0.010,
    insertion_height=0.030,
    grasp_post_height=0.035,
    grasp_post_radius=0.004,
    post_blend_height=0.004,
    socket_height=0.035,
    clearance=0.0006,
    wall_thickness=0.004,
    entrance_relief=0.0005,
    entrance_depth=0.003,
    outer_margin=0.0015,
    socket_smooth_radius=0.00030,
    socket_simplify_tol=0.00008,
    trim_extra_height=0.003,
    triangle_max_area=None,
    seed=None,
    add_base_under_socket=False,
):
    """
    Same export format as before:
      - asset_plug.obj / asset_socket.obj
      - asset_plug.mat / asset_socket.mat
      - plug_grasp.json
      - assembly_height.json
      - shape_meta.json
      - geometry_meta.json
    """
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    profile, shape_meta = sample_valid_shape(target_extent=target_extent)

    plug_mesh, plug_meta = create_plug_with_grasp_post(
        profile,
        insertion_height=insertion_height,
        grasp_post_height=grasp_post_height,
        grasp_post_radius=grasp_post_radius,
        post_blend_height=post_blend_height,
    )

    socket_mesh, hole_profile, socket_profile, socket_meta = create_socket_from_profile(
        profile,
        socket_height=socket_height,
        clearance=clearance,
        wall_thickness=wall_thickness,
        entrance_relief=entrance_relief,
        entrance_depth=entrance_depth,
        outer_margin=outer_margin,
        socket_smooth_radius=socket_smooth_radius,
        socket_simplify_tol=socket_simplify_tol,
        trim_extra_height=trim_extra_height,
        triangle_max_area=triangle_max_area,
    )
    # ============================================================
    # Add mounting base under socket
    # ============================================================

    base_size_x = 0.06
    base_size_y = 0.06
    base_height = 0.01

    base_mesh = trimesh.creation.box(
        extents=[base_size_x, base_size_y, base_height]
    )

    # Slight overlap for robust boolean
    base_mesh.apply_translation([
        0.0,
        0.0,
        -base_height / 2.0 + 0.0005
    ])

    # TRUE manifold union
    socket_mesh = trimesh.boolean.union(
        [
            socket_mesh,
            base_mesh
        ],
        engine="blender"
    )

    # Shift upward so z starts at 0
    socket_mesh.apply_translation([
        0.0,
        0.0,
        base_height
    ])

    socket_mesh.process(validate=True)
    socket_mesh.fix_normals()

    plug_path = os.path.join(save_dir, "asset_plug.obj")
    socket_path = os.path.join(save_dir, "asset_socket.obj")
    plug_mesh.export(plug_path)
    socket_mesh.export(socket_path)

    plug_mat = os.path.join(save_dir, "asset_plug.mat")
    socket_mat = os.path.join(save_dir, "asset_socket.mat")
    write_mat_plug(plug_mat)
    write_mat_socket(socket_mat)
    attach_material(plug_path, "asset_plug.mat")
    attach_material(socket_path, "asset_socket.mat")

    grasp = generate_grasp(
        insertion_height=insertion_height,
        post_blend_height=post_blend_height,
        grasp_post_height=grasp_post_height,
        asset_id=asset_id,
    )
    with open(os.path.join(save_dir, "plug_grasp.json"), "w") as f:
        json.dump(grasp, f, indent=2)

    with open(os.path.join(save_dir, "assembly_height.json"), "w") as f:
        json.dump({f"asset_{asset_id}": socket_height + 0.015}, f, indent=2)

    with open(os.path.join(save_dir, "shape_meta.json"), "w") as f:
        json.dump(shape_meta, f, indent=2)

    with open(os.path.join(save_dir, "geometry_meta.json"), "w") as f:
        merged_meta = {}
        merged_meta.update(plug_meta)
        merged_meta.update(socket_meta)
        json.dump(merged_meta, f, indent=2)

    return {
        "profile": profile,
        "hole_profile": hole_profile,
        "socket_profile": socket_profile,
        "plug_mesh": plug_mesh,
        "socket_mesh": socket_mesh,
        "shape_meta": shape_meta,
        "plug_meta": plug_meta,
        "socket_meta": socket_meta,
        "save_dir": save_dir,
    }


def visualize_profiles(profile, hole_profile=None, socket_profile=None, save_path=None):
    fig, axes = plt.subplots(1, 3 if socket_profile is not None else 2, figsize=(12, 4))
    axes = np.array(axes).reshape(-1)

    profiles = [("plug profile", profile)]
    if hole_profile is not None:
        profiles.append(("socket hole", hole_profile))
    if socket_profile is not None:
        profiles.append(("full socket profile", socket_profile))

    all_pts = []
    for _, g in profiles:
        x, y = polygon_to_xy(g)
        all_pts.append(np.stack([x, y], axis=1))
    all_pts = np.concatenate(all_pts, axis=0)
    lim = 1.15 * np.max(np.abs(all_pts))

    for ax, (title, g) in zip(axes, profiles):
        x, y = polygon_to_xy(g)
        ax.fill(x, y, alpha=0.72)
        ax.plot(x, y, linewidth=1.0)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    asset_ids = [f"{asset_id}" for asset_id in range(100000, 100300)]

failist = []

for asset_id in asset_ids:

    print(f"\nGenerating asset {asset_id}")

    base_dir = (
        f"/home/ubuntu/automate/"
        f"IsaacGymEnvs_Assembly/assets/automate/mesh/{asset_id}"
    )

    success = False

    base_seed = int(asset_id) * 2

    for retry_idx in range(100):

        seed = base_seed + retry_idx * 10000

        try:

            print(f"Trying seed {seed}")

            result = save_asset_bundle(
                save_dir=base_dir,
                asset_id=asset_id,
                target_extent=0.010,
                insertion_height=0.030,
                grasp_post_height=0.035,
                grasp_post_radius=0.004,
                post_blend_height=0.004,
                socket_height=0.035,
                clearance=0.0008,
                wall_thickness=0.004,
                entrance_relief=0.0005,
                entrance_depth=0.030,
                outer_margin=0.0015,
                socket_smooth_radius=0.00030,
                socket_simplify_tol=0.00008,
                trim_extra_height=0.003,
                triangle_max_area=None,
                seed=seed,
                add_base_under_socket=True,
            )
            visualize_profiles(
                result["profile"],
                hole_profile=result["hole_profile"],
                socket_profile=result["socket_profile"],
                save_path=os.path.join(base_dir, "profiles.png"),
            )
            print(
                f"Successfully generated asset "
                f"{asset_id} with seed {seed}"
            )
            success = True
            break
        except Exception as e:
            print(
                f"Seed {seed} failed "
                f"for asset {asset_id}"
            )
            print(e)
    if not success:
        print(f"FAILED asset {asset_id}")
        failist.append(asset_id)
