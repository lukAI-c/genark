"""
Run UniNaVid inference on a single episode from OpenNav_R2R-CE_100_bertidx.json
and save the rendered observation stream as an mp4 video.

Usage (run from this directory, with the `genesis` conda env active):
    python eval_single_episode.py --episode-id 42
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

import genesis as gs

# Reuse navmesh / sliding helpers from the sibling multi-eval script.
_PARENT_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_SCRIPTS not in sys.path:
    sys.path.append(_PARENT_SCRIPTS)  # append, not insert, so local agent_uninavid_genesis wins
from genesis_gt_path_multi_eval import (  # noqa: E402
    batch_find_floor,
    batch_get_sliding_position,
    calculate_initial_yaw_tensor,
)

from agent_uninavid_genesis import UniNaVid_Agent  # noqa: E402

# ---------------- Defaults ----------------
EPISODES_FILE = "/ssd/hongyuding/code/Genesis/scripts/OpenNav_R2R-CE_100_bertidx.json"
SCENE_DATASETS = "/ssd/hongyuding/data/genesis"
MODEL_PATH = "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"

CAMERA_HEIGHT = 1.25
CAM_RES = (640, 480)
FOV = 105
LIGHT_SCALE = 4.0

ACTION_MAP = {0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right"}

STEP_MOVE = 0.25
STEP_TURN = np.deg2rad(30.0)
ALLOW_SLIDING = True
MAX_STEP_HEIGHT = 0.5


def load_episode(path: str, episode_id):
    with open(path, "r") as f:
        data = json.load(f)
    eps = data["episodes"] if isinstance(data, dict) and "episodes" in data else data
    matches = [e for e in eps if str(e["episode_id"]) == str(episode_id)]
    if not matches:
        raise ValueError(f"Episode id {episode_id} not found in {path}")
    return matches[0]


def run(episode_id, max_steps, output_video, fps):
    episode = load_episode(EPISODES_FILE, episode_id)
    instruction_text = episode["instruction"]["instruction_text"]
    scene_id = episode["scene_id"]
    scan_name = os.path.basename(os.path.dirname(scene_id))

    print(f"Episode {episode_id} | scene {scan_name}")
    print(f"Instruction: {instruction_text}")

    gs.init(backend=gs.cuda)
    renderer = gs.renderers.BatchRenderer(use_rasterizer=True)
    scene = gs.Scene(
        renderer=renderer,
        show_viewer=False,
        vis_options=gs.options.VisOptions(ambient_light=(1.0, 1.0, 1.0), plane_reflection=False),
    )

    obj_dir = f"{SCENE_DATASETS}/{scan_name}/matterport_mesh"
    obj_files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]
    if not obj_files:
        raise FileNotFoundError(f"No .obj under {obj_dir}")
    obj_path = os.path.join(obj_dir, obj_files[0])
    scene.add_entity(morph=gs.morphs.Mesh(file=obj_path, fixed=True, collision=False))

    nm = np.load(f"{SCENE_DATASETS}/{scan_name}/navmesh.npz")
    V_habitat = torch.tensor(nm["verts"], device=gs.device, dtype=torch.float32)
    F = torch.tensor(nm["faces"], device=gs.device, dtype=torch.long)
    V_genesis = torch.zeros_like(V_habitat)
    V_genesis[:, 0] = V_habitat[:, 0]
    V_genesis[:, 1] = -V_habitat[:, 2]
    V_genesis[:, 2] = V_habitat[:, 1]
    tri_v0_3d = V_genesis[F[:, 0]]
    tri_v1_3d = V_genesis[F[:, 1]]
    tri_v2_3d = V_genesis[F[:, 2]]
    tri_v0_2d = tri_v0_3d[:, :2]
    tri_v1_2d = tri_v1_3d[:, :2]
    tri_v2_2d = tri_v2_3d[:, :2]

    print(f"Loading UniNaVid from {MODEL_PATH}")
    agent = UniNaVid_Agent(
        model_path=MODEL_PATH,
        result_path="output_uninavid_single",
        exp_save="",
        save_debug_images=False,
    )
    agent.reset()

    cam = scene.add_camera(res=CAM_RES, fov=FOV, GUI=False)
    scene.build(n_envs=1, env_spacing=(0.0, 0.0))

    sx = episode["start_position"][0]
    sy = -episode["start_position"][2]
    sz = episode["start_position"][1] + CAMERA_HEIGHT
    cam_pos = torch.tensor([[sx, sy, sz]], device=gs.device, dtype=torch.float32)
    cam_rot_q = torch.tensor([episode["start_rotation"]], device=gs.device, dtype=torch.float32)
    cam_yaw = calculate_initial_yaw_tensor(cam_rot_q)

    _, _, current_tri_idx = batch_find_floor(
        cam_pos[:, :2], cam_pos[:, 2] - CAMERA_HEIGHT,
        tri_v0_2d, tri_v1_2d, tri_v2_2d, tri_v0_3d, tri_v1_3d, tri_v2_3d,
    )

    def update_cam_pose():
        dir_x = torch.cos(cam_yaw)
        dir_y = torch.sin(cam_yaw)
        target = cam_pos.clone()
        target[:, 0] += dir_x
        target[:, 1] += dir_y
        target[:, 2] = cam_pos[:, 2]
        cam.set_pose(pos=cam_pos, lookat=target)

    update_cam_pose()

    goal_pos = episode["goals"][0]["position"]
    goal_pos_t = torch.tensor([goal_pos], device=gs.device, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_video)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (CAM_RES[0], CAM_RES[1])
    )

    finished = False
    step = 0
    t0 = time.time()
    while not finished and step < max_steps:
        rgb_batch, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, force_render=True)
        if rgb_batch.dtype == torch.uint8:
            rgb_batch = rgb_batch.float() / 255.0
        rgb_batch = torch.clamp(rgb_batch * LIGHT_SCALE * 255.0, 0, 255).byte()
        rgb_cpu = rgb_batch.cpu().numpy()[0]  # HWC RGB

        # Distance to goal (habitat coords)
        hx = cam_pos[0, 0].item()
        hy = cam_pos[0, 2].item() - CAMERA_HEIGHT
        hz = -cam_pos[0, 1].item()
        dist = float(np.linalg.norm(np.array([hx, hy, hz]) - np.array(goal_pos)))

        obs = {"rgb": rgb_cpu, "instruction": {"text": instruction_text}}
        try:
            action_dict = agent.act(obs, {}, episode["episode_id"], step_idx=step)
            action = int(action_dict["action"]) if action_dict and "action" in action_dict else 0
        except Exception as e:
            print(f"Agent failure: {e}")
            action = 0

        # Annotate frame and write
        frame_bgr = rgb_cpu[..., ::-1].copy()
        cv2.putText(frame_bgr, f"step {step} act {ACTION_MAP[action]} dist {dist:.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(frame_bgr)

        print(f"[{step}] action={ACTION_MAP[action]} dist={dist:.2f}")

        if action == 0:
            finished = True
            break

        if action == 2:
            cam_yaw = cam_yaw + STEP_TURN
        elif action == 3:
            cam_yaw = cam_yaw - STEP_TURN
        elif action == 1:
            dir_x = torch.cos(cam_yaw)
            dir_y = torch.sin(cam_yaw)
            desired_pos = torch.stack(
                [cam_pos[:, 0] + dir_x * STEP_MOVE, cam_pos[:, 1] + dir_y * STEP_MOVE], dim=1
            )
            current_floor_z = cam_pos[:, 2] - CAMERA_HEIGHT
            valid_mask, new_floor_z, new_tri_idx = batch_find_floor(
                desired_pos, current_floor_z,
                tri_v0_2d, tri_v1_2d, tri_v2_2d, tri_v0_3d, tri_v1_3d, tri_v2_3d,
            )
            if ALLOW_SLIDING and not valid_mask.item():
                slide_pos = batch_get_sliding_position(
                    cam_pos[:, :2], desired_pos, current_tri_idx,
                    tri_v0_2d, tri_v1_2d, tri_v2_2d,
                )
                slide_valid, slide_z, slide_tri_idx = batch_find_floor(
                    slide_pos, current_floor_z,
                    tri_v0_2d, tri_v1_2d, tri_v2_2d, tri_v0_3d, tri_v1_3d, tri_v2_3d,
                )
                if slide_valid.item():
                    desired_pos = slide_pos
                    new_floor_z = slide_z
                    new_tri_idx = slide_tri_idx
                    valid_mask = slide_valid
            if valid_mask.item():
                cam_pos[:, 0] = desired_pos[:, 0]
                cam_pos[:, 1] = desired_pos[:, 1]
                cam_pos[:, 2] = new_floor_z + CAMERA_HEIGHT
                current_tri_idx = new_tri_idx

        update_cam_pose()
        step += 1

    writer.release()
    print(f"\nDone. {step} steps in {time.time()-t0:.1f}s. Video -> {output_video}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episode-id", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--output", type=str, default="output_uninavid_single/episode_42.mp4")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()
    run(args.episode_id, args.max_steps, args.output, args.fps)
