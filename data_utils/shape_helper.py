from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import trimesh
def fix_geom(geom):
    """Fix minor invalidities from boolean operations."""
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


def polygon_exterior_coords(poly):
    poly = get_main_polygon(poly)
    coords = np.asarray(poly.exterior.coords[:-1], dtype=np.float64)
    return coords

from shapely.geometry import Polygon, MultiPolygon

def clean_socket_hole_profile(profile, smooth_radius=0.00035, simplify_tol=0.00008):
    """
    Clean the plug profile before generating socket cavity.

    This keeps the plug unchanged, but makes the socket nicer by:
    - removing tiny dents / spikes
    - smoothing boundary
    - simplifying overly dense polygons
    """
    g = fix_geom(profile)

    # Keep only largest component
    if isinstance(g, MultiPolygon):
        g = max(g.geoms, key=lambda p: p.area)

    # Morphological smoothing (very important)
    if smooth_radius > 0:
        g = fix_geom(g.buffer(smooth_radius, resolution=64))
        g = fix_geom(g.buffer(-smooth_radius, resolution=64))

    # Simplify small jagged edges
    if simplify_tol > 0:
        g = fix_geom(g.simplify(simplify_tol, preserve_topology=True))

    # Final cleanup
    if isinstance(g, MultiPolygon):
        g = max(g.geoms, key=lambda p: p.area)

    g = fix_geom(g.buffer(0))
    return g

def expand_contour_from_centroid(poly, clearance=0.0006, smooth_rounds=1):
    """
    Build the socket inner wall by directly expanding the plug contour points
    away from the centroid.

    This is NOT a perfect normal-offset curve, but it is simple, controllable,
    and avoids boolean subtraction for the cavity construction.
    """
    poly = get_main_polygon(poly)
    coords = polygon_exterior_coords(poly)

    centroid = np.asarray([poly.centroid.x, poly.centroid.y], dtype=np.float64)
    vec = coords - centroid[None, :]
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)

    expanded = coords + clearance * (vec / norm)

    # light Laplacian-style smoothing to reduce tiny spikes
    for _ in range(max(0, smooth_rounds)):
        prev_pts = np.roll(expanded, 1, axis=0)
        next_pts = np.roll(expanded, -1, axis=0)
        expanded = 0.25 * prev_pts + 0.50 * expanded + 0.25 * next_pts

    expanded_poly = Polygon(expanded)
    expanded_poly = fix_geom(expanded_poly)

    if isinstance(expanded_poly, MultiPolygon):
        expanded_poly = max(expanded_poly.geoms, key=lambda g: g.area)

    if expanded_poly.is_empty:
        raise RuntimeError("Expanded inner contour became empty/invalid.")

    return expanded_poly


def make_socket_outer_from_inner(inner_profile, wall_thickness=0.004, outer_margin=0.0015):
    """
    Create the socket outer wall by expanding outward from the inner wall.
    """
    outer_profile = fix_geom(inner_profile.buffer(wall_thickness + outer_margin, resolution=64))
    outer_profile = get_main_polygon(outer_profile)
    return outer_profile