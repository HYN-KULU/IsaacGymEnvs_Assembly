import os
import json
import numpy as np
import trimesh
from shapely.geometry import Polygon


PLUG_RADIUS = 0.005
PLUG_LENGTH = 0.06
TOLERANCE = 0.0005

SOCKET_HEIGHT = 0.04
DATASET_DIR = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/assets/automate/mesh"


def create_plug():

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

def create_socket():
    hole_radius = PLUG_RADIUS + TOLERANCE
    socket_parts = []
    
    num_layers = np.random.randint(4, 7)
    
    # 厚度随机分配逻辑
    raw_weights = np.random.uniform(0.5, 2.0, num_layers) 
    trend = np.linspace(2.5, 0.5, num_layers)
    final_weights = raw_weights * trend
    layer_heights = (final_weights / final_weights.sum()) * SOCKET_HEIGHT
    
    base_outer_max = hole_radius + 0.05
    current_z = 0

    # --- 新增：整体延伸方向的随机偏移 ---
    # 设定一个总体的偏移趋势，让 socket 朝某个随机方向“歪”过去
    total_drift_x = np.random.uniform(-0.03, 0.03)
    total_drift_y = np.random.uniform(-0.03, 0.03)
    
    for i in range(num_layers):
        # 顶层依然保持较小
        if i == num_layers - 1:
            scale = 0.35 
        else:
            scale = 1.0 - (i * (0.5 / num_layers))
            
        n = 32
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        r_min = hole_radius + 0.005
        r_max = max(r_min + 0.005, base_outer_max * scale)
        
        radii = np.random.uniform(r_min, r_max, n)
        radii = np.convolve(radii, [0.2, 0.6, 0.2], mode='same')

        # --- 核心修改：计算每一层的中心偏移 ---
        # 随着层数 i 增加，中心点向 total_drift 方向移动
        # 这会让孔看起来在底层中央，但在顶层边缘；或者反过来
        layer_offset_x = (i / num_layers) * total_drift_x
        layer_offset_y = (i / num_layers) * total_drift_y

        # 在生成外壳顶点时加上偏移，但 inner_pts（孔）保持在 (0,0)
        outer_pts = np.stack([
            radii * np.cos(angles) + layer_offset_x, 
            radii * np.sin(angles) + layer_offset_y
        ], axis=1)
        
        # 统一的圆孔，始终位于原点 (0,0)
        hole_angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
        inner_pts = np.stack([
            hole_radius * np.cos(hole_angles),
            hole_radius * np.sin(hole_angles)
        ], axis=1)

        # 创建多边形
        poly = Polygon(outer_pts, [inner_pts])
        h = layer_heights[i]
        layer_mesh = trimesh.creation.extrude_polygon(poly, height=h)
        
        # 随机旋转
        z_rot = np.random.uniform(0, 2 * np.pi)
        layer_mesh.apply_transform(trimesh.transformations.rotation_matrix(z_rot, [0, 0, 1]))
        
        # 放置高度
        layer_mesh.apply_translation([0, 0, current_z])
        socket_parts.append(layer_mesh)
        
        current_z += h

    socket = trimesh.util.concatenate(socket_parts)
    socket.remove_duplicate_faces()
    socket.fix_normals()
    
    return socket

# def create_socket():
#     hole_radius = PLUG_RADIUS + TOLERANCE
#     socket_parts = []
    
#     # 1. 随机层数 (4到6层)
#     num_layers = np.random.randint(4, 7)
    
#     # 2. 更加随机的厚度分配
#     # 基础权重，底部层获得更高的厚度潜力
#     raw_weights = np.random.uniform(0.5, 2.0, num_layers) 
#     trend = np.linspace(2.5, 0.5, num_layers) # 强化底部厚、顶部薄的趋势
#     final_weights = raw_weights * trend
#     layer_heights = (final_weights / final_weights.sum()) * SOCKET_HEIGHT
    
#     # 3. 基础半径设置
#     base_outer_max = hole_radius + 0.05
#     current_z = 0
    
#     for i in range(num_layers):
#         # --- 核心修改：控制顶层半径变小 ---
#         if i == num_layers - 1:
#             # 顶层 scale 设小（例如 0.35），它会比底层小很多
#             scale = 0.35 
#         else:
#             # 中间层按线性递减
#             scale = 1.0 - (i * (0.6 / num_layers))
            
#         # 4. 顶点生成
#         n = 32 # 使用较多顶点让边缘平滑一些
#         angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
#         # 设定当前层的半径范围
#         # 确保最小值永远比孔径 (hole_radius) 大一点，避免几何错误
#         r_min = hole_radius + 0.005
#         r_max = max(r_min + 0.005, base_outer_max * scale)
        
#         # 随机生成半径并做简单的平滑处理
#         radii = np.random.uniform(r_min, r_max, n)
#         radii = np.convolve(radii, [0.2, 0.6, 0.2], mode='same')

#         # 转换为 2D 点
#         outer_pts = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        
#         # 统一的圆孔
#         hole_pts = 64
#         hole_angles = np.linspace(0, 2*np.pi, hole_pts, endpoint=False)
#         inner_pts = np.stack([
#             hole_radius * np.cos(hole_angles),
#             hole_radius * np.sin(hole_angles)
#         ], axis=1)

#         # 5. 创建多边形并拉伸
#         poly = Polygon(outer_pts, [inner_pts])
#         h = layer_heights[i]
#         layer_mesh = trimesh.creation.extrude_polygon(poly, height=h)
        
#         # 每一层加一个完全随机的旋转，增加不规则感
#         z_rot = np.random.uniform(0, 2 * np.pi)
#         layer_mesh.apply_transform(trimesh.transformations.rotation_matrix(z_rot, [0, 0, 1]))
        
#         # 放置
#         layer_mesh.apply_translation([0, 0, current_z])
#         socket_parts.append(layer_mesh)
        
#         current_z += h

#     # 6. 合并所有层
#     socket = trimesh.util.concatenate(socket_parts)
#     socket.remove_duplicate_faces()
#     socket.fix_normals()
    
#     return socket

# def create_socket():
#     hole_radius = PLUG_RADIUS + TOLERANCE
#     socket_parts = []
    
#     num_layers = 5 # 固定层数会让结构更稳定
#     weights = np.arange(num_layers, 0, -1)
#     layer_heights = (weights / weights.sum()) * SOCKET_HEIGHT
    
#     base_outer_max = hole_radius + 0.04
#     current_z = 0
    
#     for i in range(num_layers):
#         scale = 1.0 - (i * (0.5 / num_layers))
        
#         # 增加顶点数，让形状更平滑
#         n = 32 
#         angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
#         # 使用平滑一些的噪声（比如在基础半径上加正弦波动）
#         # 而不是完全随机的 radii
#         noise = np.random.uniform(-0.01, 0.01, n) * scale
#         radii = (base_outer_max * scale) + noise
        
#         outer_pts = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        
#         hole_angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
#         inner_pts = np.stack([hole_radius * np.cos(hole_angles), hole_radius * np.sin(hole_angles)], axis=1)

#         poly = Polygon(outer_pts, [inner_pts])
#         h = layer_heights[i]
#         layer_mesh = trimesh.creation.extrude_polygon(poly, height=h)
        
#         # 限制旋转角度，比如每层只比上一层多转一点点
#         z_rot = (np.pi / 12) * i + np.random.uniform(-0.1, 0.1)
#         layer_mesh.apply_transform(trimesh.transformations.rotation_matrix(z_rot, [0, 0, 1]))
        
#         layer_mesh.apply_translation([0, 0, current_z])
#         socket_parts.append(layer_mesh)
#         current_z += h

#     socket = trimesh.util.concatenate(socket_parts)
#     socket.fix_normals()
#     return socket

# def create_socket():
#     hole_radius = PLUG_RADIUS + TOLERANCE
    
#     # 1. Define the 2D Outer Boundary
#     n = np.random.randint(6, 10)
#     angles = np.sort(np.random.rand(n) * 2 * np.pi)
#     radii = np.random.uniform(hole_radius + 0.02, hole_radius + 0.04, n)
    
#     outer_pts = np.stack([
#         radii * np.cos(angles),
#         radii * np.sin(angles)
#     ], axis=1)
    
#     # 2. Define the 2D Inner Hole (reversed for proper path orientation)
#     hole_angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
#     inner_pts = np.stack([
#         hole_radius * np.cos(hole_angles),
#         hole_radius * np.sin(hole_angles)
#     ], axis=1)[::-1] # Reverse to tell trimesh this is a void

#     # 3. Create the Polygon and Extrude
#     # We use trimesh.creation.extrude_polygon to handle the hole natively
#     poly = Polygon(outer_pts, [inner_pts])
#     socket = trimesh.creation.extrude_polygon(poly, height=SOCKET_HEIGHT)
    
#     # 4. Clean up
#     socket.fix_normals()
#     socket.remove_degenerate_faces()
    
#     return socket
def generate_grasp():

    x = np.random.uniform(-0.003, 0.003)
    y = np.random.uniform(-0.003, 0.003)

    z = PLUG_LENGTH + np.random.uniform(0.07, 0.12)

    qx = 0.0
    qy = 1.0
    qz = np.random.uniform(-0.1, 0.1)
    qw = 0.0

    return {
        "asset_10001": [x, y, z, qx, qy, qz, qw]
    }


def write_mat(path):

    with open(path, "w") as f:
        f.write("newmtl mat0\n")
        f.write("Ka 0.5000 0.5000 0.5000\n")
        f.write("Kd 0.0823529 0.47451 0.329412\n")
        f.write("illum 1\n")


def attach_material(obj_path, mtl):

    with open(obj_path, "r") as f:
        content = f.read()

    header = f"mtllib {mtl}\nusemtl mat0\n"

    with open(obj_path, "w") as f:
        f.write(header + content)


def generate_asset():

    asset_id = "10001"

    save_dir = os.path.join(DATASET_DIR, asset_id)
    os.makedirs(save_dir, exist_ok=True)

    plug = create_plug()
    socket = create_socket()

    plug_path = os.path.join(save_dir, "asset_plug.obj")
    socket_path = os.path.join(save_dir, "asset_socket.obj")

    plug.export(plug_path)
    socket.export(socket_path)

    plug_mat = os.path.join(save_dir, "asset_plug.mat")
    socket_mat = os.path.join(save_dir, "asset_socket.mat")

    write_mat(plug_mat)
    write_mat(socket_mat)

    attach_material(plug_path, "asset_plug.mat")
    attach_material(socket_path, "asset_socket.mat")

    grasp = generate_grasp()

    with open(os.path.join(save_dir, "plug_grasp.json"), "w") as f:
        json.dump(grasp, f, indent=2)

    print("Generated asset 10001")


if __name__ == "__main__":
    generate_asset()
# import os
# import numpy as np
# import trimesh
# import json


# # ==========================
# # parameters
# # ==========================

# PLUG_RADIUS = 0.005
# PLUG_LENGTH = 0.06

# TOLERANCE = 0.001

# SOCKET_HEIGHT = 0.05
# SOCKET_OUTER_RADIUS = 0.04

# DATASET_DIR = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/assets/automate/mesh"


# # ==========================
# # plug
# # ==========================

# def create_plug():

#     plug = trimesh.creation.cylinder(
#         radius=PLUG_RADIUS,
#         height=PLUG_LENGTH,
#         sections=64
#     )

#     plug.apply_translation([0,0,PLUG_LENGTH/2])

#     return plug


# # ==========================
# # socket (annular cylinder)
# # ==========================

# def create_socket():

#     hole_radius = PLUG_RADIUS + TOLERANCE
#     outer_radius = SOCKET_OUTER_RADIUS * np.random.uniform(0.9,1.1)

#     sections = 64
#     theta = np.linspace(0,2*np.pi,sections,endpoint=False)

#     outer_top = np.stack([
#         outer_radius*np.cos(theta),
#         outer_radius*np.sin(theta),
#         np.ones_like(theta)*SOCKET_HEIGHT
#     ],axis=1)

#     outer_bottom = np.stack([
#         outer_radius*np.cos(theta),
#         outer_radius*np.sin(theta),
#         np.zeros_like(theta)
#     ],axis=1)

#     inner_top = np.stack([
#         hole_radius*np.cos(theta),
#         hole_radius*np.sin(theta),
#         np.ones_like(theta)*SOCKET_HEIGHT
#     ],axis=1)

#     inner_bottom = np.stack([
#         hole_radius*np.cos(theta),
#         hole_radius*np.sin(theta),
#         np.zeros_like(theta)
#     ],axis=1)

#     vertices = np.vstack([
#         outer_top,
#         outer_bottom,
#         inner_top,
#         inner_bottom
#     ])

#     faces = []

#     for i in range(sections):

#         j = (i+1)%sections

#         # outer wall
#         faces.append([i, j, sections+i])
#         faces.append([j, sections+j, sections+i])

#         # inner wall
#         faces.append([
#             2*sections+i,
#             3*sections+i,
#             2*sections+j
#         ])
#         faces.append([
#             2*sections+j,
#             3*sections+i,
#             3*sections+j
#         ])

#         # top ring
#         faces.append([
#             i,
#             2*sections+i,
#             j
#         ])
#         faces.append([
#             j,
#             2*sections+i,
#             2*sections+j
#         ])

#         # bottom ring
#         faces.append([
#             sections+i,
#             sections+j,
#             3*sections+i
#         ])
#         faces.append([
#             sections+j,
#             3*sections+j,
#             3*sections+i
#         ])

#     mesh = trimesh.Trimesh(vertices=vertices,faces=faces)

#     mesh.process(validate=True)

#     return mesh


# # ==========================
# # grasp pose
# # ==========================

# def generate_grasp():

#     x = np.random.uniform(-0.003,0.003)
#     y = np.random.uniform(-0.003,0.003)

#     z = PLUG_LENGTH + np.random.uniform(0.07,0.12)

#     qx = 0.0
#     qy = 1.0
#     qz = np.random.uniform(-0.1,0.1)
#     qw = 0.0

#     return {
#         "asset_10001":[x,y,z,qx,qy,qz,qw]
#     }


# # ==========================
# # material file
# # ==========================

# def write_mat(path,name):

#     with open(path,"w") as f:

#         f.write(f"newmtl {name}\n")
#         f.write("Ka 0.2 0.2 0.2\n")
#         f.write("Kd 0.8 0.8 0.8\n")
#         f.write("Ks 0.1 0.1 0.1\n")
#         f.write("d 1.0\n")
#         f.write("illum 2\n")


# # ==========================
# # generator
# # ==========================

# def generate_asset():

#     asset_id = "10001"

#     save_dir = os.path.join(DATASET_DIR,asset_id)

#     os.makedirs(save_dir,exist_ok=True)

#     plug = create_plug()
#     socket = create_socket()

#     plug.export(os.path.join(save_dir,"asset_plug.obj"))
#     socket.export(os.path.join(save_dir,"asset_socket.obj"))

#     write_mat(os.path.join(save_dir,"asset_plug.mat"),"plug")
#     write_mat(os.path.join(save_dir,"asset_socket.mat"),"socket")

#     grasp = generate_grasp()

#     with open(os.path.join(save_dir,"plug_grasp.json"),"w") as f:
#         json.dump(grasp,f,indent=2)

#     print("Generated asset_10001")


# # ==========================
# # run
# # ==========================

# if __name__ == "__main__":

#     generate_asset()