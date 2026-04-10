#!/usr/bin/env python3
"""
Render all episodes for 2azQ1b91cZZ using GLB, save one JPG per episode.
Usage:
    cd /data1/home/clk/project/genark
    CUDA_VISIBLE_DEVICES=3 python test_render_scene.py
"""
import sys, os, json
sys.path.insert(0, "/data1/home/clk/project/Genesis")

import genesis as gs
import torch
import numpy as np
import cv2

CAMERA_HEIGHT = 1.25
LIGHT_SCALE   = 4.0
FOV           = 105
CAM_RES       = (640, 480)
BATCH_SIZE    = 50          # render this many episodes at a time (VRAM limit)

SCAN          = "2azQ1b91cZZ"
EPISODES_FILE = "/data1/home/clk/project/genark/episodes/mp3d_2azQ1b91cZZ.json"
MESH_PATH     = f"/data1/home/clk/{SCAN}.glb"
OUT_DIR       = f"/data1/home/clk/project/genark/test_render_output/{SCAN}"


def hab_to_genesis(pos):
    return pos[0], -pos[2], pos[1]


def calculate_yaw(rot):
    x, y, z, w = rot
    return float(np.arctan2(-(y * y - w * w), -2 * w * y))


def render_batch(episodes, scene_built=False):
    """Render one batch of episodes. Returns when done."""
    N = len(episodes)

    gs.init(backend=gs.cuda)
    renderer = gs.renderers.BatchRenderer(use_rasterizer=True)
    scene = gs.Scene(
        renderer=renderer,
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            ambient_light=(1.0, 1.0, 1.0),
            plane_reflection=False,
        ),
    )

    scene.add_entity(morph=gs.morphs.Mesh(file=MESH_PATH, fixed=True, collision=False))
    cam = scene.add_camera(res=CAM_RES, fov=FOV, GUI=False)
    scene.build(n_envs=N, env_spacing=(0.0, 0.0))

    # Camera poses
    pos_list, yaw_list = [], []
    for ep in episodes:
        gx, gy, gz = hab_to_genesis(ep["start_position"])
        pos_list.append([gx, gy, gz + CAMERA_HEIGHT])
        yaw_list.append(calculate_yaw(ep["start_rotation"]))

    cam_pos = torch.tensor(pos_list, dtype=torch.float32, device=gs.device)
    cam_yaw = torch.tensor(yaw_list, dtype=torch.float32, device=gs.device)

    lookat = cam_pos.clone()
    lookat[:, 0] += torch.cos(cam_yaw)
    lookat[:, 1] += torch.sin(cam_yaw)
    lookat[:, 2] = cam_pos[:, 2]
    cam.set_pose(pos=cam_pos, lookat=lookat)

    # Render
    rgb_raw, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, force_render=True)

    # Apply light scale
    if rgb_raw.dtype == torch.uint8:
        rgb_float = rgb_raw.float() / 255.0
    else:
        rgb_float = rgb_raw.float()
    rgb_scaled = torch.clamp(rgb_float * LIGHT_SCALE * 255.0, 0, 255).byte()
    rgb_np = rgb_scaled.cpu().numpy()

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    for i in range(N):
        ep_id = episodes[i]["episode_id"]
        bgr = cv2.cvtColor(rgb_np[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUT_DIR, f"episode_{ep_id}.jpg"), bgr)

    raw_mean = rgb_raw.float().mean().item()
    raw_max = rgb_raw.max().item()
    print(f"  Batch {N} eps: raw mean={raw_mean:.2f}, max={raw_max}, "
          f"scaled mean={rgb_scaled.float().mean().item():.2f}")


if __name__ == "__main__":
    import subprocess

    with open(EPISODES_FILE) as f:
        all_episodes = json.load(f)["episodes"]

    print(f"Scene: {SCAN}")
    print(f"Mesh:  {MESH_PATH}")
    print(f"Total episodes: {len(all_episodes)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Output: {OUT_DIR}/")

    # Split into batches, each in its own subprocess (Genesis can only init once)
    for batch_start in range(0, len(all_episodes), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_episodes))
        batch = all_episodes[batch_start:batch_end]
        ep_ids = [ep["episode_id"] for ep in batch]

        print(f"\n--- Batch {batch_start//BATCH_SIZE + 1}: "
              f"episodes {batch_start}-{batch_end-1} "
              f"(ids: {ep_ids[0]}..{ep_ids[-1]}) ---")

        # Write batch to temp file
        tmp_file = f"/tmp/_batch_{SCAN}.json"
        with open(tmp_file, "w") as f:
            json.dump({"episodes": batch}, f)

        # Run in subprocess
        ret = subprocess.run(
            [sys.executable, "-c",
             f"""
import sys, json
sys.path.insert(0, "/data1/home/clk/project/Genesis")
sys.path.insert(0, "/data1/home/clk/project/genark")
from test_render_scene import render_batch
with open("{tmp_file}") as f:
    eps = json.load(f)["episodes"]
render_batch(eps)
"""],
            env=os.environ.copy(),
        )
        if ret.returncode != 0:
            print(f"  FAILED with exit code {ret.returncode}")

    # Count results
    jpgs = [f for f in os.listdir(OUT_DIR) if f.endswith('.jpg')]
    print(f"\nDone! {len(jpgs)}/{len(all_episodes)} images saved to {OUT_DIR}/")
