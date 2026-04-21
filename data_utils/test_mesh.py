import numpy as np
import triangle as tr
import trimesh
import plotly.graph_objects as go


def build_square_socket_2d(
    outer_half=1.0,
    inner_half=0.3,
    max_area=0.01,
):
    """
    Build a 2D triangulation of:
        outer square - inner square hole
    """
    # Outer square: counterclockwise
    outer_vertices = np.array([
        [-outer_half, -outer_half],
        [ outer_half, -outer_half],
        [ outer_half,  outer_half],
        [-outer_half,  outer_half],
    ], dtype=np.float64)

    # Inner square hole
    inner_vertices = np.array([
        [-inner_half, -inner_half],
        [-inner_half,  inner_half],
        [ inner_half,  inner_half],
        [ inner_half, -inner_half],
    ], dtype=np.float64)

    vertices = np.vstack([outer_vertices, inner_vertices])

    # Boundary edges
    outer_segments = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ], dtype=np.int32)

    inner_offset = len(outer_vertices)
    inner_segments = np.array([
        [inner_offset + 0, inner_offset + 1],
        [inner_offset + 1, inner_offset + 2],
        [inner_offset + 2, inner_offset + 3],
        [inner_offset + 3, inner_offset + 0],
    ], dtype=np.int32)

    segments = np.vstack([outer_segments, inner_segments])

    holes = np.array([[0.0, 0.0]], dtype=np.float64)

    A = {
        "vertices": vertices,
        "segments": segments,
        "holes": holes,
    }

    mesh2d = tr.triangulate(A, f"pqa{max_area}")

    return mesh2d, outer_segments, inner_segments


def extrude_socket_to_3d(mesh2d, outer_segments, inner_segments, height=0.5):
    """
    Extrude the 2D square-with-hole mesh into a 3D socket mesh.
    Produces:
      - bottom surface
      - top surface
      - outer side walls
      - inner side walls
    """
    v2 = mesh2d["vertices"]
    t2 = mesh2d["triangles"]

    n = len(v2)

    # bottom z=0, top z=height
    bottom = np.column_stack([v2, np.zeros(n)])
    top = np.column_stack([v2, np.full(n, height)])
    vertices3d = np.vstack([bottom, top])

    faces = []

    # Bottom cap
    # Reverse winding so normals point outward/downward
    for tri in t2:
        faces.append([tri[0], tri[2], tri[1]])

    # Top cap
    for tri in t2:
        faces.append([tri[0] + n, tri[1] + n, tri[2] + n])

    # Outer wall
    for a, b in outer_segments:
        faces.append([a, b, b + n])
        faces.append([a, b + n, a + n])

    # Inner wall
    # Flip orientation relative to outer wall
    for a, b in inner_segments:
        faces.append([a, b + n, b])
        faces.append([a, a + n, b + n])

    faces = np.asarray(faces, dtype=np.int64)

    mesh3d = trimesh.Trimesh(vertices=vertices3d, faces=faces, process=True)
    mesh3d.remove_duplicate_faces()
    mesh3d.remove_degenerate_faces()
    mesh3d.remove_unreferenced_vertices()
    mesh3d.fix_normals()

    return mesh3d


def show_mesh_plotly(mesh):
    v = mesh.vertices
    f = mesh.faces

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                opacity=1.0,
                flatshading=False,
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2, roughness=0.6),
                showscale=False,
            )
        ]
    )

    fig.update_layout(
        title="3D Square Socket with Square Hole",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()


if __name__ == "__main__":
    mesh2d, outer_segments, inner_segments = build_square_socket_2d(
        outer_half=1.0,
        inner_half=0.3,
        max_area=0.01,
    )

    socket_mesh = extrude_socket_to_3d(
        mesh2d,
        outer_segments,
        inner_segments,
        height=0.5,
    )

    socket_mesh.export("square_socket.obj")
    print("Saved square_socket.obj")

    show_mesh_plotly(socket_mesh)