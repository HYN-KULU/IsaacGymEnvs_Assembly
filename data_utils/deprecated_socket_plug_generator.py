import os
import json
import numpy as np
from shortuuid import random
import trimesh
from shapely.geometry import Polygon



DATASET_DIR = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/assets/automate/mesh"


def create_plug(PLUG_RADIUS, PLUG_LENGTH, TOLERANCE):

    plug = trimesh.creation.cylinder(
        radius=PLUG_RADIUS,
        height=PLUG_LENGTH,
        sections=64
    )

    plug.apply_translation([0, 0, PLUG_LENGTH / 2])
    return plug


def random_polygon(hole_radius):

    n = np.random.randint(5, 8)

    angles = np.sort(np.random.rand(n) * 2*np.pi)

    outer_min = hole_radius + 0.025
    outer_max = hole_radius + 0.05

    radii = np.random.uniform(outer_min, outer_max, n)

    pts = np.stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ], axis=1)

    return Polygon(pts)

def create_socket(PLUG_RADIUS, TOLERANCE, SOCKET_HEIGHT):
    hole_radius = PLUG_RADIUS + TOLERANCE
    socket_parts = []
    
    num_layers = np.random.randint(1, 4)
    
    # 随机高度分配
    raw_weights = np.random.uniform(0.8, 1.5, num_layers)
    layer_heights = (raw_weights / raw_weights.sum()) * SOCKET_HEIGHT
    
    current_z = 0

    # --- 关键修改：固定整体偏移方向 ---
    # 我们预设一个“重心偏移”，让 socket 的外壳中心远离 (0,0)
    # 这样 (0,0) 处的孔就自然靠边了
    shift_distance = np.random.uniform(0, 0.03) 
    angle = np.random.uniform(0, 2 * np.pi)
    
    offset_x = shift_distance * np.cos(angle)
    offset_y = shift_distance * np.sin(angle)
    
    # 为了防止孔掉出外壳，基准外径必须大于 (偏移量 + 孔径)
    min_base_radius = shift_distance + hole_radius + 0.01

    for i in range(num_layers):
        # 即使有 scale 变化，也要确保半径能盖住原点
        # 底部大，顶部稍微小一点，但顶部也要盖住孔
        layer_scale = 1.0 - (i * 0.2 / num_layers) 
        
        n = 64 
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # 计算当前层外壳的半径
        # 必须确保在 offset 的对侧，半径也能覆盖到原点 (0,0)
        r_base = min_base_radius * layer_scale
        
        # 添加一些随机扰动，让每一层形状不同
        noise = np.random.uniform(-0.005, 0.005, n)
        radii = r_base + noise
        radii = np.convolve(radii, [0.2, 0.6, 0.2], mode='same')

        # 外壳顶点：整体向 offset 方向移动
        outer_pts = np.stack([
            radii * np.cos(angles) + offset_x, 
            radii * np.sin(angles) + offset_y
        ], axis=1)
        
        # 内部圆孔：永远固定在原点 (0,0)，不随 offset 移动
        # 这保证了 Plug 垂直插下去时不会撞到壁
        hole_angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
        inner_pts = np.stack([
            hole_radius * np.cos(hole_angles),
            hole_radius * np.sin(hole_angles)
        ], axis=1)

        # 创建多边形
        poly = Polygon(outer_pts, [inner_pts])
        
        # 确保多边形有效
        if not poly.is_valid:
            poly = poly.buffer(0)

        layer_mesh = trimesh.creation.extrude_polygon(poly, height=layer_heights[i])
        
        z_rot = np.random.uniform(0, 2 * np.pi)
        layer_mesh.apply_transform(trimesh.transformations.rotation_matrix(z_rot, [0, 0, 1]))
        
        layer_mesh.apply_translation([0, 0, current_z])
        socket_parts.append(layer_mesh)
        current_z += layer_heights[i]

    socket = trimesh.util.concatenate(socket_parts)
    socket.remove_duplicate_faces()
    socket.fix_normals()
    
    return socket

def generate_grasp(PLUG_LENGTH=0.09, asset_id = ""):

    x = np.random.uniform(-0.003, 0.003)
    y = np.random.uniform(-0.003, 0.003)

    z = PLUG_LENGTH + np.random.uniform(0.07, 0.12)

    qx = 0.0
    qy = 1.0
    qz = np.random.uniform(-0.1, 0.1)
    qw = 0.0

    return {
        f"asset_{asset_id}": [x, y, z, qx, qy, qz, qw]
    }


def write_mat_socket(path):

    with open(path, "w") as f:
        f.write("newmtl mat0\n")
        f.write("Ka 0.5000 0.5000 0.5000\n")
        f.write("Kd 0.0823529 0.47451 0.329412\n")
        f.write("illum 1\n")
def write_mat_plug(path):
    with open(path, "w") as f:
        f.write("newmtl mat0\n")
        f.write("Ka 0.5000 0.5000 0.5000\n")
        f.write("Kd 0.964706 0.933333 0.898039\n")
        f.write("illum 1\n")


def attach_material(obj_path, mtl):

    with open(obj_path, "r") as f:
        content = f.read()

    header = f"mtllib {mtl}\nusemtl mat0\n"

    with open(obj_path, "w") as f:
        f.write(header + content)


def generate_asset(asset_id="10001"):
        PLUG_RADIUS = 0.004
        # PLUG_RADIUS = np.random.uniform(0.003, 0.005)
        # PLUG_LENGTH = np.random.uniform(0.04, 0.07)
        PLUG_LENGTH = 0.005
        TOLERANCE = 0.0008

        SOCKET_HEIGHT = PLUG_LENGTH-np.random.uniform(0.03, 0.035)
        save_dir = os.path.join(DATASET_DIR, asset_id)
        os.makedirs(save_dir, exist_ok=True)

        plug = create_plug(PLUG_RADIUS, PLUG_LENGTH, TOLERANCE)
        socket = create_socket(PLUG_RADIUS, TOLERANCE, SOCKET_HEIGHT)

        plug_path = os.path.join(save_dir, "asset_plug.obj")
        socket_path = os.path.join(save_dir, "asset_socket.obj")

        plug.export(plug_path)
        socket.export(socket_path)

        plug_mat = os.path.join(save_dir, "asset_plug.mat")
        socket_mat = os.path.join(save_dir, "asset_socket.mat")

        write_mat_plug(plug_mat)
        write_mat_socket(socket_mat)

        attach_material(plug_path, "asset_plug.mat")
        attach_material(socket_path, "asset_socket.mat")

        grasp = generate_grasp(PLUG_LENGTH, asset_id)

        with open(os.path.join(save_dir, "plug_grasp.json"), "w") as f:
            json.dump(grasp, f, indent=2)
        with open(os.path.join(save_dir, "assembly_height.json"), "w") as f:
            json.dump({f"asset_{asset_id}":SOCKET_HEIGHT + 0.015}, f, indent=2)

        # print("Generated asset 10001")


if __name__ == "__main__":
    # for asset_id in range(10101,10110):
    for asset_id in [100000]:
    # for asset_id in [10010,10018,10026,10029,10032,10037]:
        generate_asset(str(asset_id))
