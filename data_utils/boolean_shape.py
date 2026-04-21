import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point, box
from shapely.affinity import scale, rotate
from shapely.ops import unary_union


# -----------------------------
# Geometry helpers
# -----------------------------
def fix_geom(geom):
    """Fix small numerical invalidities from boolean operations."""
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def polygon_to_xy(geom):
    x, y = geom.exterior.xy
    return np.asarray(x), np.asarray(y)


# -----------------------------
# Primitive shape families
# -----------------------------
def make_circle(radius=1.0):
    return Point(0, 0).buffer(radius, resolution=128)


def make_ellipse(rx=1.2, ry=0.9):
    g = Point(0, 0).buffer(1.0, resolution=128)
    return scale(g, xfact=rx, yfact=ry, origin=(0, 0))


def make_capsule(length=2.0, radius=0.6):
    """
    Stadium / rounded-rectangle shape:
    union of 2 circles and a rectangle.
    Symmetric and very insertion-friendly.
    """
    left = Point(-length / 2, 0).buffer(radius, resolution=128)
    right = Point(length / 2, 0).buffer(radius, resolution=128)
    middle = box(-length / 2, -radius, length / 2, radius)
    return fix_geom(unary_union([left, right, middle]))

def make_rectangle(hx=1.0, hy=0.8):
    """
    Axis-aligned rectangle centered at origin.
    hx, hy = half widths
    """
    return box(-hx, -hy, hx, hy)


from shapely.geometry import box

def make_rounded_rectangle(hx=1.0, hy=0.8, radius=0.25):
    """
    Rounded rectangle centered at origin.
    hx, hy: outer half-width and half-height
    radius: corner radius
    """
    radius = min(radius, hx, hy) - 1e-6

    core = box(-(hx - radius), -(hy - radius), hx - radius, hy - radius)
    rounded = core.buffer(radius, resolution=64)
    return rounded

def make_bone(length=2.0, lobe_radius=0.8, neck_half_width=0.35):
    """
    Bone / dumbbell:
    union of two larger circles and a narrower connector.
    """
    left = Point(-length / 2, 0).buffer(lobe_radius, resolution=128)
    right = Point(length / 2, 0).buffer(lobe_radius, resolution=128)
    middle = box(-length / 2, -neck_half_width, length / 2, neck_half_width)
    return fix_geom(unary_union([left, right, middle]))


def make_rounded_cross(center_r=0.55, arm_r=0.45, arm_offset=0.70):
    """
    A mild symmetric 'cross/clover-4' shape.
    Still smooth and not too concave if parameters are controlled.
    """
    parts = [Point(0, 0).buffer(center_r, resolution=128)]
    parts.append(Point(+arm_offset, 0).buffer(arm_r, resolution=128))
    parts.append(Point(-arm_offset, 0).buffer(arm_r, resolution=128))
    parts.append(Point(0, +arm_offset).buffer(arm_r, resolution=128))
    parts.append(Point(0, -arm_offset).buffer(arm_r, resolution=128))
    return fix_geom(unary_union(parts))


# -----------------------------
# Shape-quality filters
# -----------------------------
def bbox_sizes(geom):
    minx, miny, maxx, maxy = geom.bounds
    return maxx - minx, maxy - miny


def shape_metrics(geom):
    """
    Compute a few simple metrics for filtering.
    """
    area = geom.area
    hull_area = geom.convex_hull.area
    convexity = area / hull_area if hull_area > 1e-8 else 0.0

    width, height = bbox_sizes(geom)
    aspect = max(width, height) / max(min(width, height), 1e-8)

    # Equivalent radius from area
    eq_radius = np.sqrt(area / np.pi)

    return {
        "area": area,
        "convexity": convexity,
        "aspect": aspect,
        "eq_radius": eq_radius,
        "width": width,
        "height": height,
    }


def is_insertion_friendly(
    geom,
    min_convexity=0.82,
    max_aspect=2.4,
    min_area=0.5,
):
    """
    Reject shapes that are too concave, too elongated, or too tiny.
    """
    geom = fix_geom(geom)
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


# -----------------------------
# Random generator
# -----------------------------
def random_symmetric_boolean_shape(max_tries=200):
    """
    Randomly generate one symmetric, insertion-friendly shape.
    Returns:
        geom, meta
    """

    for _ in range(max_tries):
        family = np.random.choice(
            ["circle", "ellipse", "capsule", "bone", "rounded_cross", "rectangle", "rounded_rectangle"],
            # p=[0,0,0,0,0,0,1]
            p=[0.12, 0.12, 0.20, 0.22, 0.12, 0.12, 0.10]
        )

        if family == "circle":
            r = np.random.uniform(0.85, 1.15)
            geom = make_circle(r)
            meta = {"family": family, "r": r}

        elif family == "ellipse":
            rx = np.random.uniform(0.95, 1.35)
            ry = np.random.uniform(0.75, 1.15)
            geom = make_ellipse(rx=rx, ry=ry)
            meta = {"family": family, "rx": rx, "ry": ry}

        elif family == "capsule":
            length = np.random.uniform(0.8, 1.8)
            radius = np.random.uniform(0.45, 0.85)
            geom = make_capsule(length=length, radius=radius)
            meta = {"family": family, "length": length, "radius": radius}

        elif family == "bone":
            length = np.random.uniform(0.8, 1.8)
            lobe_radius = np.random.uniform(0.55, 0.95)
            neck_half_width = np.random.uniform(0.30, 0.75 * lobe_radius)
            geom = make_bone(
                length=length,
                lobe_radius=lobe_radius,
                neck_half_width=neck_half_width,
            )
            meta = {
                "family": family,
                "length": length,
                "lobe_radius": lobe_radius,
                "neck_half_width": neck_half_width,
            }

        elif family == "rounded_cross":
            center_r = np.random.uniform(0.40, 0.70)
            arm_r = np.random.uniform(0.28, 0.52)
            arm_offset = np.random.uniform(0.45, 0.85)
            geom = make_rounded_cross(
                center_r=center_r,
                arm_r=arm_r,
                arm_offset=arm_offset,
            )
            meta = {
                "family": family,
                "center_r": center_r,
                "arm_r": arm_r,
                "arm_offset": arm_offset,
            }
        elif family == "rectangle":
            hx = np.random.uniform(0.6, 1.2)
            hy = np.random.uniform(0.6, 1.2)
            geom = make_rectangle(hx, hy)
            meta = {"family": family, "hx": hx, "hy": hy}

        elif family == "rounded_rectangle":
            hx = np.random.uniform(0.6, 1.2)
            hy = np.random.uniform(0.6, 1.2)
            radius = np.random.uniform(0.05, 0.25 * min(hx, hy))
            geom = make_rounded_rectangle(hx, hy, radius)
            meta = {
                "family": family,
                "hx": hx,
                "hy": hy,
                "corner_radius": radius
            }
        else:
            continue

        # Mild anisotropic scaling only
        if np.random.rand() < 0.35:
            sx = np.random.uniform(0.90, 1.15)
            sy = np.random.uniform(0.90, 1.15)
            geom = scale(geom, xfact=sx, yfact=sy, origin=(0, 0))
            meta["scale_x"] = sx
            meta["scale_y"] = sy

        # Random in-plane rotation
        rot_deg = np.random.uniform(0, 180)
        geom = rotate(geom, rot_deg, origin=(0, 0), use_radians=False)
        meta["rotation_deg"] = rot_deg

        geom = fix_geom(geom)

        if is_insertion_friendly(geom):
            meta.update(shape_metrics(geom))
            return geom, meta

    raise RuntimeError("Failed to sample a valid insertion-friendly shape.")


# -----------------------------
# Visualization
# -----------------------------
def visualize_random_shapes(n=25, seed=0, show_titles=True):
    np.random.seed(seed)

    shapes = []
    titles = []

    for _ in range(n):
        geom, meta = random_symmetric_boolean_shape()
        shapes.append(geom)

        if show_titles:
            titles.append(
                f"{meta['family']}\n"
                f"conv={meta['convexity']:.2f}, asp={meta['aspect']:.2f}"
            )
        else:
            titles.append("")

    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    all_pts = []
    for g in shapes:
        x, y = polygon_to_xy(g)
        all_pts.append(np.stack([x, y], axis=1))
    all_pts = np.concatenate(all_pts, axis=0)
    lim = 1.15 * np.max(np.abs(all_pts))

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
        ax.set_title(titles[i], fontsize=10)

    plt.tight_layout()
    plt.savefig("boolean_shape.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    visualize_random_shapes(n=25, seed=42)
