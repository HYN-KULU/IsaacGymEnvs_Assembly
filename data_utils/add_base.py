import argparse
from pathlib import Path


DEFAULT_ASSET_DIR = Path(
    "/home/ubuntu/automate/"
    "IsaacGymEnvs_Assembly/assets/automate/mesh/80003"
)

BASE_SIZE_X = 0.06  # 6 cm
BASE_SIZE_Y = 0.06  # 6 cm
BASE_HEIGHT = 0.01  # 1 cm


def add_base_with_trimesh(socket_path, output_path):
    import trimesh

    socket_mesh = trimesh.load(socket_path, force="mesh")
    socket_mesh.process(validate=True)
    socket_mesh.remove_unreferenced_vertices()
    socket_mesh.fix_normals()

    base_mesh = trimesh.creation.box(
        extents=[BASE_SIZE_X, BASE_SIZE_Y, BASE_HEIGHT]
    )

    # Match diverse_shape_generator.py:
    # base starts below the socket at z in [-BASE_HEIGHT, 0],
    # then the combined mesh is shifted up so the base is [0, BASE_HEIGHT].
    base_mesh.apply_translation([0.0, 0.0, -BASE_HEIGHT / 2.0])

    combined_mesh = trimesh.util.concatenate([socket_mesh, base_mesh])
    combined_mesh.apply_translation([0.0, 0.0, BASE_HEIGHT])

    combined_mesh.merge_vertices()
    combined_mesh.remove_duplicate_faces()
    combined_mesh.remove_degenerate_faces()
    combined_mesh.remove_unreferenced_vertices()
    combined_mesh.fix_normals()
    combined_mesh.export(output_path)

    return combined_mesh.bounds, combined_mesh.extents, combined_mesh.is_watertight


def _format_float(value):
    return f"{value:.8f}"


def _parse_vertex(line):
    parts = line.split()
    return float(parts[1]), float(parts[2]), float(parts[3])


def _shift_vertex_line(line, dz):
    x, y, z = _parse_vertex(line)
    return (
        f"v {_format_float(x)} {_format_float(y)} "
        f"{_format_float(z + dz)}\n"
    )


def add_base_without_trimesh(socket_path, output_path):
    lines = socket_path.read_text().splitlines(keepends=True)
    vertex_count = 0
    bounds_min = [float("inf"), float("inf"), float("inf")]
    bounds_max = [float("-inf"), float("-inf"), float("-inf")]
    output_lines = []

    for line in lines:
        if line.startswith("v "):
            x, y, z = _parse_vertex(line)
            z += BASE_HEIGHT
            vertex_count += 1
            bounds_min[0] = min(bounds_min[0], x)
            bounds_min[1] = min(bounds_min[1], y)
            bounds_min[2] = min(bounds_min[2], z)
            bounds_max[0] = max(bounds_max[0], x)
            bounds_max[1] = max(bounds_max[1], y)
            bounds_max[2] = max(bounds_max[2], z)
            output_lines.append(_shift_vertex_line(line, BASE_HEIGHT))
        else:
            output_lines.append(line)

    hx = BASE_SIZE_X / 2.0
    hy = BASE_SIZE_Y / 2.0
    z0 = 0.0
    z1 = BASE_HEIGHT
    base_vertices = [
        (-hx, -hy, z0),
        (hx, -hy, z0),
        (hx, hy, z0),
        (-hx, hy, z0),
        (-hx, -hy, z1),
        (hx, -hy, z1),
        (hx, hy, z1),
        (-hx, hy, z1),
    ]
    base_faces = [
        (1, 4, 3),
        (1, 3, 2),
        (5, 6, 7),
        (5, 7, 8),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 4, 8),
        (3, 8, 7),
        (4, 1, 5),
        (4, 5, 8),
    ]

    output_lines.append("\n# 6 cm x 6 cm x 1 cm mounting base\n")
    for x, y, z in base_vertices:
        output_lines.append(
            f"v {_format_float(x)} {_format_float(y)} {_format_float(z)}\n"
        )
        bounds_min[0] = min(bounds_min[0], x)
        bounds_min[1] = min(bounds_min[1], y)
        bounds_min[2] = min(bounds_min[2], z)
        bounds_max[0] = max(bounds_max[0], x)
        bounds_max[1] = max(bounds_max[1], y)
        bounds_max[2] = max(bounds_max[2], z)

    for face in base_faces:
        shifted = [str(vertex_count + index) for index in face]
        output_lines.append(f"f {' '.join(shifted)}\n")

    output_path.write_text("".join(output_lines))

    extents = [bounds_max[i] - bounds_min[i] for i in range(3)]
    return bounds_min, bounds_max, extents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add a 6 cm x 6 cm x 1 cm base under an asset socket OBJ."
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help="Directory containing asset_socket.obj.",
    )
    parser.add_argument(
        "--socket-name",
        default="asset_socket.obj",
        help="Input socket OBJ filename inside asset-dir.",
    )
    parser.add_argument(
        "--output-name",
        default="asset_socket_with_base.obj",
        help="Output socket OBJ filename inside asset-dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    socket_path = args.asset_dir / args.socket_name
    output_path = args.asset_dir / args.output_name

    if not socket_path.exists():
        raise FileNotFoundError(socket_path)

    print(f"Loading socket: {socket_path}")
    print(f"Saving modified socket: {output_path}")

    try:
        bounds, extents, watertight = add_base_with_trimesh(socket_path, output_path)
        print("Used trimesh mesh pipeline.")
        print(f"Final bounds:\n{bounds}")
        print(f"Final extents: {extents}")
        print(f"Final watertight: {watertight}")
    except ModuleNotFoundError as exc:
        if exc.name != "trimesh":
            raise
        bounds_min, bounds_max, extents = add_base_without_trimesh(
            socket_path, output_path
        )
        print("trimesh is not installed; used OBJ fallback writer.")
        print(f"Final bounds:\n[{bounds_min},\n {bounds_max}]")
        print(f"Final extents: {extents}")


if __name__ == "__main__":
    main()
