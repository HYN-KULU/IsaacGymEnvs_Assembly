import trimesh
import plotly.graph_objects as go

mesh_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/assets/automate/mesh/10001/asset_socket.obj"

mesh = trimesh.load(mesh_path, force="mesh")

vertices = mesh.vertices
faces = mesh.faces

fig = go.Figure(
    data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=1.0,
        )
    ]
)

fig.update_layout(
    scene=dict(aspectmode="data"),
    margin=dict(l=0, r=0, b=0, t=30),
    title="Socket Mesh",
)

fig.show()