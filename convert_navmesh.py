import habitat_sim
import numpy as np
import os

scene_root = "/home/nvme01/uni-lavira/data/scene_datasets/mp3d"
out_root   = "/home/clk/genark_ws/genark/scene_datasets"

for scan in os.listdir(scene_root):
    navmesh_file = f"{scene_root}/{scan}/{scan}.navmesh"
    out_dir  = f"{out_root}/{scan}"
    out_file = f"{out_dir}/navmesh.npz"

    if not os.path.exists(navmesh_file):
        print(f"✗ {scan}: .navmesh not found")
        continue
    if os.path.exists(out_file):
        print(f"- {scan}: already converted")
        continue

    os.makedirs(out_dir, exist_ok=True)
    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(navmesh_file)
    verts = np.array(pf.build_navmesh_vertices(), dtype=np.float32)
    faces = np.array(pf.build_navmesh_vertex_indices(), dtype=np.int32).reshape(-1, 3)
    np.savez(out_file, verts=verts, faces=faces)
    print(f"✓ {scan}: {len(verts)} verts, {len(faces)} faces")
