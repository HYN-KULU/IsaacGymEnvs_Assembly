import os
import trimesh
import matplotlib.pyplot as plt

# ==== 1. 路径设置 ====
base_dir = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/assets/automate/mesh"
save_dir = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/assets/automate/mesh_visualization"

# 创建保存文件夹（如果不存在）
os.makedirs(save_dir, exist_ok=True)

# ==== 2. 遍历所有编号文件夹 ====
for folder in sorted(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue  # 跳过非文件夹
    
    plug_path = os.path.join(folder_path, "asset_plug.obj")
    socket_path = os.path.join(folder_path, "asset_socket.obj")

    # 检查两个文件是否存在
    if not os.path.exists(plug_path) or not os.path.exists(socket_path):
        print(f"⚠️  跳过 {folder}，缺少 obj 文件。")
        continue

    print(f"🧩 正在处理 {folder} ...")

    # ==== 3. 可视化 plug ====
    mesh = trimesh.load(plug_path)
    vertices, faces = mesh.vertices, mesh.faces
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2],
                    triangles=faces, color=(0.3,0.6,1.0,0.6), edgecolor="gray")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title(f"Plug {folder}")
    plt.savefig(os.path.join(save_dir, f"{folder}_plug.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)  # ✅ 释放内存

    # ==== 4. 可视化 socket ====
    mesh = trimesh.load(socket_path)
    vertices, faces = mesh.vertices, mesh.faces
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2],
                    triangles=faces, color=(1.0,0.6,0.3,0.6), edgecolor="gray")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title(f"Socket {folder}")
    plt.savefig(os.path.join(save_dir, f"{folder}_socket.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

print("✅ 所有 plug/socket 图片已保存到：", save_dir)
