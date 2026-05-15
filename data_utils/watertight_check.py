import trimesh
import sys
import numpy as np


def check_watertight(obj_path: str):
    # force='mesh' tries to load as a single mesh
    mesh = trimesh.load(obj_path, force='mesh')

    print("========== Mesh Info ==========")
    print(f"File: {obj_path}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")

    print("\n========== Watertight Check ==========")
    print(f"Is watertight: {mesh.is_watertight}")
    print(f"Is winding consistent: {mesh.is_winding_consistent}")
    print(f"Euler number: {mesh.euler_number}")

    # Boundary edges: edges used by only one face
    edges_sorted = mesh.edges_sorted
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)

    boundary_edges = unique_edges[counts == 1]
    nonmanifold_edges = unique_edges[counts > 2]

    print("\n========== Edge Diagnostics ==========")
    print(f"Boundary edges / hole edges: {len(boundary_edges)}")
    print(f"Non-manifold edges: {len(nonmanifold_edges)}")

    if len(boundary_edges) > 0:
        print("\nMesh has holes or open boundaries.")
    elif len(nonmanifold_edges) > 0:
        print("\nMesh has non-manifold edges.")
    elif mesh.is_watertight:
        print("\nMesh is watertight.")
    else:
        print("\nMesh is not watertight, but no simple boundary edges were found.")

    return {
        "is_watertight": mesh.is_watertight,
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.faces),
        "num_boundary_edges": len(boundary_edges),
        "num_nonmanifold_edges": len(nonmanifold_edges),
        "is_winding_consistent": mesh.is_winding_consistent,
        "euler_number": mesh.euler_number,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python check_watertight.py path/to/model.obj")
        sys.exit(1)

    obj_path = sys.argv[1]
    check_watertight(obj_path)