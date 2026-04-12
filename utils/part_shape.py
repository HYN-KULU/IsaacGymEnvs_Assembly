import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point, box
from shapely.affinity import rotate, scale
from shapely.ops import unary_union


def fix_geom(geom):
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def polygon_to_xy(geom):
    x, y = geom.exterior.xy
    return np.asarray(x), np.asarray(y)


def make_circle(center=(0, 0), radius=1.0):
    return Point(center[0], center[1]).buffer(radius, resolution=128)


def make_square(center=(0, 0), half_size=1.0):
    cx, cy = center
    return box(cx - half_size, cy - half_size, cx + half_size, cy + half_size)


def make_rounded_square(center=(0, 0), half_size=1.0, radius=0.25):
    cx, cy = center
    radius = min(radius, half_size) - 1e-6
    core = box(
        cx - (half_size - radius),
        cy - (half_size - radius),
        cx + (half_size - radius),
        cy + (half_size - radius),
    )
    return fix_geom(core.buffer(radius, resolution=64))


def make_rectangle(center=(0, 0), hx=1.0, hy=0.5):
    cx, cy = center
    return box(cx - hx, cy - hy, cx + hx, cy + hy)


def make_endcap(kind, anchor_x, size, body_half_width):
    """
    Endcap is attached around x = anchor_x.
    The vertical size is loosely tied to body_half_width.
    """
    if kind == "none":
        return None

    if kind == "circle":
        return make_circle(center=(anchor_x, 0), radius=size)

    if kind == "square":
        return make_square(center=(anchor_x, 0), half_size=size)

    if kind == "rounded_square":
        radius = np.random.uniform(0.18, 0.38) * size
        return make_rounded_square(center=(anchor_x, 0), half_size=size, radius=radius)

    raise ValueError(f"Unknown endcap kind: {kind}")


def sample_part_based_shape():
    """
    Procedural shape generator:
      shape = union( center rectangle, left endcap, right endcap )

    No fixed semantic categories are needed.
    """
    # Main body
    body_length = np.random.uniform(0.8, 2.0)
    body_half_width = np.random.uniform(0.35, 0.85)

    body = make_rectangle(center=(0, 0), hx=body_length / 2, hy=body_half_width)

    # Independent endcap choices
    cap_types = ["none", "circle", "square", "rounded_square"]
    probs = [0.20, 0.35, 0.20, 0.25]

    left_type = np.random.choice(cap_types, p=probs)
    right_type = np.random.choice(cap_types, p=probs)

    # Endcap sizes
    # Tie size to body width so unions stay reasonable
    left_size = np.random.uniform(0.75, 1.35) * body_half_width
    right_size = np.random.uniform(0.75, 1.35) * body_half_width

    left_x = -body_length / 2
    right_x = body_length / 2

    parts = [body]

    left_cap = make_endcap(left_type, left_x, left_size, body_half_width)
    right_cap = make_endcap(right_type, right_x, right_size, body_half_width)

    if left_cap is not None:
        parts.append(left_cap)
    if right_cap is not None:
        parts.append(right_cap)

    geom = fix_geom(unary_union(parts))

    # Mild extra scaling for diversity
    if np.random.rand() < 0.35:
        sx = np.random.uniform(0.90, 1.15)
        sy = np.random.uniform(0.90, 1.15)
        geom = scale(geom, xfact=sx, yfact=sy, origin=(0, 0))

    # Rotation
    rot_deg = np.random.uniform(0, 180)
    geom = rotate(geom, rot_deg, origin=(0, 0), use_radians=False)
    geom = fix_geom(geom)

    meta = {
        "left_type": left_type,
        "right_type": right_type,
        "body_length": body_length,
        "body_half_width": body_half_width,
        "left_size": left_size,
        "right_size": right_size,
        "rotation_deg": rot_deg,
    }
    return geom, meta


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
    }


def is_good_shape(geom, min_convexity=0.80, max_aspect=2.8, min_area=0.4):
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


def sample_valid_shape(max_tries=200):
    for _ in range(max_tries):
        geom, meta = sample_part_based_shape()
        if is_good_shape(geom):
            meta.update(shape_metrics(geom))
            return geom, meta
    raise RuntimeError("Failed to sample a valid shape.")


def visualize_random_shapes(n=20, seed=0):
    np.random.seed(seed)

    shapes = []
    titles = []

    for _ in range(n):
        geom, meta = sample_valid_shape()
        shapes.append(geom)
        titles.append(
            f"L={meta['left_type']}, R={meta['right_type']}\n"
            f"conv={meta['convexity']:.2f}, asp={meta['aspect']:.2f}"
        )

    cols = 5
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
    plt.savefig("part_shape.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    visualize_random_shapes(n=20, seed=42)