import sys
import numpy as np
import trimesh
import plotly.graph_objects as go


def get_boundary_edges(mesh):
    """
    Boundary edges are edges used by exactly one face.
    These indicate holes / open boundaries.
    """
    edges_sorted = mesh.edges_sorted
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    return boundary_edges


def make_edge_trace(vertices, edges, name="Boundary edges", color="red", width=5):
    """
    Convert mesh edges into a Plotly 3D line trace.
    """
    x, y, z = [], [], []

    for edge in edges:
        v0, v1 = vertices[edge[0]], vertices[edge[1]]

        x += [v0[0], v1[0], None]
        y += [v0[1], v1[1], None]
        z += [v0[2], v1[2], None]

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
    )


def visualize_obj(obj_path, output_html=None):
    mesh = trimesh.load(obj_path, force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded object is not a single Trimesh.")

    vertices = mesh.vertices
    faces = mesh.faces

    print("========== Mesh Info ==========")
    print(f"File: {obj_path}")
    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    print(f"Watertight: {mesh.is_watertight}")
    print(f"Winding consistent: {mesh.is_winding_consistent}")
    print(f"Euler number: {mesh.euler_number}")

    boundary_edges = get_boundary_edges(mesh)

    print("\n========== Boundary Info ==========")
    print(f"Boundary edges: {len(boundary_edges)}")

    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.55,
        name="OBJ mesh",
        color="lightblue",
        flatshading=True,
    )

    traces = [mesh_trace]

    if len(boundary_edges) > 0:
        boundary_trace = make_edge_trace(
            vertices,
            boundary_edges,
            name="Boundary / hole edges",
            color="red",
            width=6,
        )
        traces.append(boundary_trace)

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=f"OBJ visualization: {obj_path}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        showlegend=True,
    )

    if output_html is None:
        output_html = obj_path.replace(".obj", "_plotly.html")

    fig.write_html(output_html)
    print(f"\nSaved Plotly visualization to:")
    print(output_html)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_obj_plotly.py path/to/model.obj")
        print("")
        print("Example:")
        print("  python visualize_obj_plotly.py /home/ubuntu/automate/IsaacGymEnvs_Assembly/assets/automate/mesh/40124/asset_socket.obj")
        sys.exit(1)

    obj_path = sys.argv[1]
    visualize_obj(obj_path)