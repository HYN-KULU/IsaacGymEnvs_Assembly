import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point, box
from shapely.affinity import rotate, scale
from shapely.ops import unary_union


def fix_geom(geom):
    """Fix minor invalidities from boolean operations."""
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


# def polygon_to_xy(geom):
#     x, y = geom.exterior.xy
#     return np.asarray(x), np.asarray(y)

def polygon_to_xy(geom):
    """Handles both single Polygons and disconnected MultiPolygons."""
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        return np.asarray(x), np.asarray(y)
    
    elif geom.geom_type == 'MultiPolygon':
        # Combine all parts into one array, separated by NaNs 
        # so matplotlib knows where to 'lift the pen'
        all_x, all_y = [], []
        for part in geom.geoms:
            x, y = part.exterior.xy
            all_x.extend(x.tolist() + [np.nan])
            all_y.extend(y.tolist() + [np.nan])
        return np.asarray(all_x), np.asarray(all_y)
    
    return np.array([]), np.array([])


def make_circle(center=(0.0, 0.0), radius=1.0):
    return Point(center[0], center[1]).buffer(radius, resolution=128)


def make_square(center=(0.0, 0.0), half_size=1.0):
    cx, cy = center
    return box(cx - half_size, cy - half_size, cx + half_size, cy + half_size)


def make_rectangle(center=(0.0, 0.0), hx=1.0, hy=0.5):
    cx, cy = center
    return box(cx - hx, cy - hy, cx + hx, cy + hy)

def make_ellipse(center=(0.0, 0.0), rx=1.0, ry=0.7):
    cx, cy = center
    circle = Point(cx, cy).buffer(1.0, resolution=128)
    return scale(circle, xfact=rx, yfact=ry, origin=(cx, cy))

def make_rounded_square(center=(0.0, 0.0), half_size=1.0, radius=0.25):
    cx, cy = center
    radius = min(radius, half_size) - 1e-6
    core = box(
        cx - (half_size - radius),
        cy - (half_size - radius),
        cx + (half_size - radius),
        cy + (half_size - radius),
    )
    return fix_geom(core.buffer(radius, resolution=64))


def make_rounded_rectangle(center=(0.0, 0.0), hx=1.0, hy=0.6, radius=0.2):
    radius = min(radius, hx, hy) - 1e-6
    cx, cy = center
    core = box(
        cx - (hx - radius),
        cy - (hy - radius),
        cx + (hx - radius),
        cy + (hy - radius),
    )
    return fix_geom(core.buffer(radius, resolution=64))


def sample_primitive(kind=None, center=(0.0, 0.0), size_scale=1.0):
    """
    Sample one primitive shape.

    kind:
      - circle
      - square
      - rounded_square
      - rectangle
      - rounded_rectangle
    """
    if kind is None:
        kind = np.random.choice(
            ["circle", "ellipse", "square", "rounded_square", "rectangle", "rounded_rectangle"],
            p=[0.25, 0.20, 0.12, 0.15, 0.14, 0.14]
        )

    cx, cy = center

    if kind == "circle":
        r = np.random.uniform(0.45, 0.95) * size_scale
        geom = make_circle(center=(cx, cy), radius=r)
        params = {"kind": kind, "radius": float(r)}

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
        params = {
            "kind": kind,
            "hx": float(hx),
            "hy": float(hy),
            "corner_radius": float(radius),
        }
    elif kind == "ellipse":
        rx = np.random.uniform(0.5, 1.0) * size_scale
        ry = np.random.uniform(0.4, 0.9) * size_scale
        geom = make_ellipse(center=(cx, cy), rx=rx, ry=ry)
        params = {"kind": kind, "rx": float(rx), "ry": float(ry)}

    else:
        raise ValueError(f"Unknown primitive kind: {kind}")

    return fix_geom(geom), params


def sample_single_layout():
    geom, info = sample_primitive(center=(0.0, 0.0), size_scale=1.0)
    return geom, {"layout": "single", "parts": [info]}


def sample_chain_layout():
    """
    Shape = body + optional left cap + optional right cap.
    Covers many useful insertion-friendly families procedurally.
    """
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

    cap_types = ["none", "circle", "square", "rounded_square"]
    cap_probs = [0.18, 0.36, 0.20, 0.26]

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
    """
    Two overlapping primitives with offset centers.
    Includes the 2-lobe / peanut / bone-like family.
    """
    kind = np.random.choice(["circle", "rounded_square"], p=[0.78, 0.22])

    s1 = np.random.uniform(0.50, 0.95)
    s2 = np.random.uniform(0.50, 0.95)
    d = np.random.uniform(0.45, 1.15) * min(s1, s2)

    c1 = (-d / 2.0, 0.0)
    c2 = ( d / 2.0, 0.0)

    if kind == "circle":
        p1 = make_circle(center=c1, radius=s1)
        p2 = make_circle(center=c2, radius=s2)
        info1 = {"kind": "circle", "center": c1, "radius": float(s1)}
        info2 = {"kind": "circle", "center": c2, "radius": float(s2)}
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
    """
    Four symmetric lobes around the center.
    Includes the rounded cross / clover family.
    """
    kind = np.random.choice(["circle", "rounded_square"], p=[0.85, 0.15])

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


def is_good_shape(geom, min_convexity=0.78, max_aspect=2.8, min_area=0.30):
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
    # Mild cleanup smoothing to remove tiny boolean seams
    if np.random.rand() < 0.40:
        eps = np.random.uniform(0.015, 0.045)
        geom = geom.buffer(eps, resolution=64).buffer(-eps, resolution=64)

    # Mild anisotropic scaling for diversity
    if np.random.rand() < 0.35:
        sx = np.random.uniform(0.90, 1.15)
        sy = np.random.uniform(0.90, 1.15)
        geom = scale(geom, xfact=sx, yfact=sy, origin=(0.0, 0.0))

    # Global random rotation
    rot_deg = np.random.uniform(0.0, 180.0)
    geom = rotate(geom, rot_deg, origin=(0.0, 0.0), use_radians=False)

    return fix_geom(geom), {"rotation_deg": float(rot_deg)}


def sample_shape_raw():
    """
    Unified procedural generator.
    Instead of fixed whole-shape categories, we sample a composition layout.
    """
    layout = np.random.choice(
        ["single", "chain", "double_lobe", "clover4"],
        p=[0.12, 0.48, 0.20, 0.20]
    )

    if layout == "single":
        geom, meta = sample_single_layout()
    elif layout == "chain":
        geom, meta = sample_chain_layout()
    elif layout == "double_lobe":
        geom, meta = sample_double_lobe_layout()
    elif layout == "clover4":
        geom, meta = sample_clover4_layout()
    else:
        raise ValueError(layout)

    geom, post = postprocess_shape(geom)
    meta.update(post)
    return geom, meta


def sample_valid_shape(max_tries=300):
    for _ in range(max_tries):
        geom, meta = sample_shape_raw()
        if is_good_shape(geom):
            meta.update(shape_metrics(geom))
            return geom, meta
    raise RuntimeError("Failed to sample a valid shape.")


def normalize_shape_to_bbox(geom, target_extent=2.0):
    """
    Scale the shape so that max(width, height) == target_extent.
    """
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    scale_factor = target_extent / max(w, h)
    geom = scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(0.0, 0.0))
    return fix_geom(geom), float(scale_factor)


def visualize_random_shapes(n=25, seed=42, normalize=True, save_path="procedural_composed_shapes.png"):
    np.random.seed(seed)

    shapes = []
    titles = []

    for _ in range(n):
        geom, meta = sample_valid_shape()

        if normalize:
            geom, sf = normalize_shape_to_bbox(geom, target_extent=2.2)
            meta["norm_scale"] = sf

        shapes.append(geom)
        titles.append(
            f"{meta['layout']}\\n"
            f"conv={meta['convexity']:.2f}, asp={meta['aspect']:.2f}"
        )

    cols = 13
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    all_pts = []
    for g in shapes:
        x, y = polygon_to_xy(g)
        all_pts.append(np.stack([x, y], axis=1))
    all_pts = np.concatenate(all_pts, axis=0)
    lim = 1.12 * np.max(np.abs(all_pts))

    for i, ax in enumerate(axes):
        if i >= len(shapes):
            ax.axis("off")
            continue

        x, y = polygon_to_xy(shapes[i])
        ax.fill(x, y, alpha=0.72)
        ax.plot(x, y, linewidth=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.25)
        ax.set_title(titles[i], fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def visualize_rejected_shapes(n=5, seed=42):
    np.random.seed(seed)
    plt.figure(figsize=(15, 3))
    
    count = 0
    while count < n:
        geom, meta = sample_shape_raw()
        # Only grab it if it's actually "bad"
        if not is_good_shape(geom):
            m = shape_metrics(geom)
            x, y = polygon_to_xy(geom)
            
            plt.subplot(1, n, count + 1)
            plt.fill(x, y, color='salmon', alpha=0.5)
            plt.plot(x, y, color='red', lw=1)
            plt.title(f"Conv: {m['convexity']:.2f}\nAsp: {m['aspect']:.2f}")
            plt.axis('equal')
            plt.axis('off')
            count += 1
    # plt.show()
    plt.savefig("rejected_shapes.png", dpi=200, bbox_inches="tight")
# Run this to see what your professor wants

if __name__ == "__main__":
    # visualize_random_shapes(n=169, seed=42, normalize=True, save_path="procedural_composed_shapes.png")
    visualize_rejected_shapes(n=49)
