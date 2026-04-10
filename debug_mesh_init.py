#!/usr/bin/env python3
"""
诊断脚本：检查初始位置是否在NavMesh内
"""
import sys
import json
import numpy as np
sys.path.insert(0, "/data1/home/clk/project/Genesis")

from genark_config import load_config

CONFIG = load_config("configs/uninavid_r2r.yaml")
SCENE_DATASETS = CONFIG.get("SCENE_DATASETS", "/data1/home/clk/project/genark/scene_datasets")
CAMERA_HEIGHT = CONFIG["SIMULATOR"]["CAMERA_HEIGHT"]
EPISODES_FILE = CONFIG.get("EPISODES_FILE", "/data1/home/clk/navid_ws/data/datasets/OpenNav_R2R-CE_100_bertidx.json")

def _hab_to_genesis(pos_list):
    x, y, z = pos_list[0], pos_list[1], pos_list[2]
    return x, -z, y

SCAN_NAME = "2azQ1b91cZZ"

# 加载episodes
with open(EPISODES_FILE) as f:
    data = json.load(f)
    all_episodes = data['episodes'] if isinstance(data, dict) else data

scene_episodes = [ep for ep in all_episodes if ep['scene_id'] == f"mp3d/{SCAN_NAME}/{SCAN_NAME}.glb"]
print(f"Found {len(scene_episodes)} episodes for {SCAN_NAME}\n")

# 加载NavMesh
navmesh_path = f"{SCENE_DATASETS}/{SCAN_NAME}/navmesh.npz"
nm = np.load(navmesh_path)
V_hab = nm["verts"]  # (M, 3) Habitat coordinates
F_idx = nm["faces"]  # (N, 3) triangle indices

# 转换到Genesis坐标
V_gen = np.zeros_like(V_hab)
V_gen[:, 0] =  V_hab[:, 0]   # X
V_gen[:, 1] = -V_hab[:, 2]   # Y (Hab -Z)
V_gen[:, 2] =  V_hab[:, 1]   # Z (Hab Y)

print(f"NavMesh vertices (Genesis coords):")
print(f"  X: [{V_gen[:, 0].min():.2f}, {V_gen[:, 0].max():.2f}]")
print(f"  Y: [{V_gen[:, 1].min():.2f}, {V_gen[:, 1].max():.2f}]")
print(f"  Z: [{V_gen[:, 2].min():.2f}, {V_gen[:, 2].max():.2f}]")
print()

# 检查前5个episode的初始位置
for i, ep in enumerate(scene_episodes[:5]):
    sp = ep["start_position"]
    gx, gy, gz = _hab_to_genesis(sp)

    # Camera位置
    cam_z = gz + CAMERA_HEIGHT

    print(f"Episode {i}:")
    print(f"  Habitat start: x={sp[0]:.2f}, y={sp[1]:.2f}, z={sp[2]:.2f}")
    print(f"  Genesis start: x={gx:.2f}, y={gy:.2f}, z={gz:.2f}")
    print(f"  Camera (z+{CAMERA_HEIGHT}): x={gx:.2f}, y={gy:.2f}, z={cam_z:.2f}")

    # 简单检查：是否在XY bounds内
    in_x = V_gen[:, 0].min() <= gx <= V_gen[:, 0].max()
    in_y = V_gen[:, 1].min() <= gy <= V_gen[:, 1].max()
    in_z = V_gen[:, 2].min() <= gz <= V_gen[:, 2].max()

    print(f"  In bounds: X={in_x}, Y={in_y}, Z={in_z}")

    if not (in_x and in_y):
        print(f"  ⚠️  位置在mesh XY范围外!")
    print()

print("\n" + "="*60)
print("问题分析:")
print("="*60)
print(f"""
1. 检查NavMesh bounds是否与初始位置匹配
2. 检查坐标转换 _hab_to_genesis() 是否正确
3. 检查CAMERA_HEIGHT={CAMERA_HEIGHT}是否合理

比较:
- env_worker.py 中 CAMERA_HEIGHT = {CAMERA_HEIGHT}
- 检查是否应该是 CAMERA_HEIGHT - AGENT_RADIUS？
""")
