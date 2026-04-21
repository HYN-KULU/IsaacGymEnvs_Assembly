import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.affinity import rotate, scale

# --- Helper Functions from your code ---
def fix_geom(geom):
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom

def polygon_to_xy(geom):
    """Handles both single Polygons and MultiPolygons safely."""
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        return np.asarray(x), np.asarray(y)
    elif geom.geom_type == 'MultiPolygon':
        all_x, all_y = [], []
        for part in geom.geoms:
            x, y = part.exterior.xy
            all_x.extend(x.tolist() + [np.nan])
            all_y.extend(y.tolist() + [np.nan])
        return np.asarray(all_x), np.asarray(all_y)
    return np.array([]), np.array([])

def shape_metrics(geom):
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    aspect = max(w, h) / max(min(w, h), 1e-8)
    return {"aspect": aspect, "width": w, "height": h}

# --- Needle Generator ---
def sample_needle_shape():
    """Forces the generation of an extreme Aspect Ratio shape."""
    
    # 1. Start with a very thin, very long horizontal rectangle (the basic needle)
    body_hx = np.random.uniform(2.5, 4.0)   # Very Long
    body_hy = np.random.uniform(0.05, 0.15) # Extremely Thin
    
    geom = box(-body_hx, -body_hy, body_hx, body_hy)
    
    # 2. Optionally make it a 'Rounded Needle' (a pin)
    if np.random.rand() < 0.5:
        radius = np.random.uniform(0.8, 1.0) * body_hy
        geom = geom.buffer(radius, resolution=32)
    
    # 3. Rotate it to make it look less like a 'tape strip'
    rot_deg = np.random.uniform(0.0, 180.0)
    geom = rotate(geom, rot_deg, origin=(0.0, 0.0), use_radians=False)
    
    return fix_geom(geom)

# --- Visualization Function ---
def visualize_needles(n=9, rows=3, cols=3, seed=123):
    """Generates and visualizes guaranteed extreme aspect ratio shapes."""
    np.random.seed(seed)
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()
    
    count = 0
    while count < n:
        geom = sample_needle_shape()
        m = shape_metrics(geom)
        
        # Only include if it meets the definition of an extreme needle
        if m["aspect"] > 8.0:
            x, y = polygon_to_xy(geom)
            
            ax = axes[count]
            # Use a different color to highlight these are 'extreme'
            ax.fill(x, y, color='skyblue', alpha=0.5)
            ax.plot(x, y, color='blue', lw=1)
            
            # Label with the key metric that usually fails
            ax.set_title(f"Aspect: {m['aspect']:.1f}", fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
            count += 1
            
    plt.tight_layout()
    # Save the output to a new file
    save_path = "extreme_needle_shapes.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Visualization saved to {save_path}")
    plt.show()

# --- Run the specialized visualization ---
if __name__ == "__main__":
    visualize_needles(n=9)