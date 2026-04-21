
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate

def generate_symmetric_profile(
    base_radius=0.004,
    n_pts=256,
    c2=None,
    c4=None,
    c6=None,
    rotation_deg=None,
    min_radius_ratio=0.45,
):
    """
    Generate a simple symmetric 2D shape using a low-frequency radial function:

        r(theta) = r0 * (1 + c2*cos(2θ) + c4*cos(4θ) + c6*cos(6θ))

    This family includes:
      - circle-like shapes
      - ellipse / dumbbell / bone-like shapes
      - rounded-rectangle-ish shapes
      - mild hexagon-ish shapes

    Parameters
    ----------
    base_radius : float
        Overall size of the shape.
    n_pts : int
        Number of boundary points.
    c2, c4, c6 : float or None
        Fourier coefficients. If None, they are sampled randomly.
    rotation_deg : float or None
        In-plane rotation in degrees. If None, sampled randomly.
    min_radius_ratio : float
        Lower bound on radius relative to base_radius to avoid overly pinched shapes.

    Returns
    -------
    poly : shapely.geometry.Polygon
        Generated 2D polygon.
    params : dict
        Dictionary of sampled coefficients.
    """
    if c2 is None:
        c2 = np.random.uniform(-0.15, 0.30)
    if c4 is None:
        c4 = np.random.uniform(-0.10, 0.12)
    if c6 is None:
        c6 = np.random.uniform(-0.05, 0.05)

    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    r = base_radius * (
        1.0
        + c2 * np.cos(2 * theta)
        + c4 * np.cos(4 * theta)
        + c6 * np.cos(6 * theta)
    )

    # Keep the radius positive and avoid overly pinched / unstable boundaries
    min_allowed = min_radius_ratio * base_radius
    if np.min(r) < min_allowed:
        scale = (base_radius - min_allowed) / (base_radius - np.min(r) + 1e-8)
        r = base_radius + (r - base_radius) * scale

    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    poly = Polygon(pts)

    # Fix occasional numerical invalidity
    if not poly.is_valid:
        poly = poly.buffer(0)

    if rotation_deg is None:
        rotation_deg = np.random.uniform(0, 180)

    poly = rotate(poly, rotation_deg, origin=(0, 0), use_radians=False)
    return poly, {"c2": float(c2), "c4": float(c4), "c6": float(c6), "rotation_deg": float(rotation_deg)}


def polygon_to_xy(poly):
    """Return x, y arrays for plotting a shapely polygon exterior."""
    x, y = poly.exterior.xy
    return np.asarray(x), np.asarray(y)


def sample_shape(kind="random", base_radius=0.004, n_pts=256, rotation_deg=0.0):
    """
    Convenience presets for debugging / visualization.

    kind options:
      - "circle"
      - "bone"
      - "boxy"
      - "hexish"
      - "random"
    """
    presets = {
        "circle": dict(c2=0.0, c4=0.0, c6=0.0),
        "bone": dict(c2=0.28, c4=-0.04, c6=0.0),
        "boxy": dict(c2=0.0, c4=0.12, c6=0.0),
        "hexish": dict(c2=0.0, c4=0.0, c6=0.05),
    }

    if kind == "random":
        return generate_symmetric_profile(
            base_radius=base_radius,
            n_pts=n_pts,
            rotation_deg=rotation_deg,
        )

    if kind not in presets:
        raise ValueError(f"Unknown kind: {kind}")

    return generate_symmetric_profile(
        base_radius=base_radius,
        n_pts=n_pts,
        rotation_deg=rotation_deg,
        **presets[kind],
    )


def visualize_shapes(shapes, titles=None, figsize=(12, 8), save_path=None):
    """
    Plot a list of shapely polygons.

    Parameters
    ----------
    shapes : list of Polygon
    titles : list of str or None
    figsize : tuple
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    n = len(shapes)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    # Shared axis range for consistent scale
    all_xy = []
    for poly in shapes:
        x, y = polygon_to_xy(poly)
        all_xy.append(np.stack([x, y], axis=1))
    all_xy = np.concatenate(all_xy, axis=0)
    lim = 1.1 * np.max(np.abs(all_xy))

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        poly = shapes[i]
        x, y = polygon_to_xy(poly)
        ax.fill(x, y, alpha=0.7)
        ax.plot(x, y, linewidth=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.25)
        if titles is not None:
            ax.set_title(titles[i], fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, axes


if __name__ == "__main__":
    np.random.seed(0)

    shapes = []
    titles = []

    for kind in ["circle", "bone", "boxy", "hexish"]:
        poly, params = sample_shape(kind=kind, rotation_deg=0.0)
        shapes.append(poly)
        titles.append(f"{kind}\\n{params}")

    for i in range(4):
        poly, params = sample_shape(kind="random", rotation_deg=0.0)
        shapes.append(poly)
        titles.append(
            "random\\n"
            f"c2={params['c2']:.2f}, "
            f"c4={params['c4']:.2f}, "
            f"c6={params['c6']:.2f}"
        )

    visualize_shapes(shapes, titles=titles, save_path="procedural_shapes_examples.png")
    print("Saved procedural_shapes_examples.png")
