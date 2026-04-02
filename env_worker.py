#!/usr/bin/env python3
"""
env_worker.py — Genesis environment worker for one Matterport scene.

Runs the Genesis renderer + NavMesh physics on its assigned GPU, then calls
the model server (HTTP) for batch inference.  Writes per-episode metrics to
a JSON file when finished.

Usage (normally launched by orchestrator.py):
    CUDA_VISIBLE_DEVICES=2 python env_worker.py \
        --scene_id mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb \
        --episodes_file /tmp/worker_0_episodes.json \
        --server_url http://127.0.0.1:8100 \
        --output_file /tmp/worker_0_metrics.json \
        [--max_steps 500] [--cam_res 640 480] [--fov 64] [--light_scale 4.0]
"""

import sys
import os
import json
import argparse
import time
import traceback
from datetime import datetime

sys.path.append("/data1/home/clk/project/Genesis")

import numpy as np
import torch
import requests
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def _write_progress(progress_file: str | None, payload: dict):
    """Atomically write worker heartbeat JSON for orchestrator polling."""
    if not progress_file:
        return
    tmp = f"{progress_file}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, progress_file)
    except Exception:
        pass

import genesis as gs

# ---------------------------------------------------------------------------
# Constants — loaded from YAML config, with hardcoded fallbacks
# ---------------------------------------------------------------------------
from genark_config import load_config, sim_constants
_DEFAULT_CFG = os.path.join(os.path.dirname(__file__), "configs", "uninavid_r2r.yaml")
if os.path.exists(_DEFAULT_CFG):
    _sc = sim_constants(load_config(_DEFAULT_CFG))
else:
    _sc = {}

SCENE_DATASETS  = "/data1/home/clk/project/genark/scene_datasets"
CAMERA_HEIGHT   = _sc.get("CAMERA_HEIGHT",   1.25)
STEP_MOVE       = _sc.get("STEP_MOVE",       0.25)
STEP_TURN       = _sc.get("STEP_TURN",       np.deg2rad(30.0))
ALLOW_SLIDING   = _sc.get("ALLOW_SLIDING",   True)
MAX_STEP_HEIGHT = _sc.get("MAX_STEP_HEIGHT", 0.5)
ACTION_MAP      = {0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right"}


# ---------------------------------------------------------------------------
# GPU NavMesh helpers (vectorised tensor ops — no Python loop over agents)
# ---------------------------------------------------------------------------

def batch_cross_2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def batch_is_point_in_triangle(pt, v0, v1, v2):
    def sign(p1, p2, p3):
        return (p1[..., 0] - p3[..., 0]) * (p2[..., 1] - p3[..., 1]) - \
               (p2[..., 0] - p3[..., 0]) * (p1[..., 1] - p3[..., 1])
    d1, d2, d3 = sign(pt, v0, v1), sign(pt, v1, v2), sign(pt, v2, v0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def batch_get_z_on_triangle(v0, v1, v2, x, y):
    edge1  = v1 - v0
    edge2  = v2 - v0
    normal = torch.cross(edge1, edge2, dim=-1)
    mask   = torch.abs(normal[..., 2]) < 1e-6
    d_x    = x - v0[..., 0]
    d_y    = y - v0[..., 1]
    num    = normal[..., 0] * d_x + normal[..., 1] * d_y
    den    = normal[..., 2]
    den    = torch.where(torch.abs(den) < 1e-6, torch.ones_like(den), den)
    z      = v0[..., 2] - num / den
    return torch.where(mask.expand_as(z), torch.full_like(z, float("inf")), z)


def batch_find_floor(pos_2d, current_floor_z,
                     tri_v0_2d, tri_v1_2d, tri_v2_2d,
                     tri_v0_3d, tri_v1_3d, tri_v2_3d):
    pt    = pos_2d.unsqueeze(1)
    v0_2  = tri_v0_2d.unsqueeze(0);  v1_2 = tri_v1_2d.unsqueeze(0);  v2_2 = tri_v2_2d.unsqueeze(0)
    inside = batch_is_point_in_triangle(pt, v0_2, v1_2, v2_2)

    v0_3  = tri_v0_3d.unsqueeze(0);  v1_3 = tri_v1_3d.unsqueeze(0);  v2_3 = tri_v2_3d.unsqueeze(0)
    z_vals = batch_get_z_on_triangle(v0_3, v1_3, v2_3, pt[..., 0], pt[..., 1])

    z_diff     = torch.abs(z_vals - current_floor_z.unsqueeze(1))
    valid_cand = inside & (z_diff < MAX_STEP_HEIGHT)
    z_diff_m   = torch.where(valid_cand, z_diff, torch.full_like(z_diff, float("inf")))

    min_dist, best_idx = torch.min(z_diff_m, dim=1)
    valid_mask = min_dist != float("inf")
    best_z     = torch.gather(z_vals, 1, best_idx.unsqueeze(1)).squeeze(1)
    return valid_mask, best_z, best_idx


def batch_get_sliding_position(current_pos, desired_pos, current_tri_idx,
                               tri_v0_2d, tri_v1_2d, tri_v2_2d):
    p0 = tri_v0_2d[current_tri_idx]
    p1 = tri_v1_2d[current_tri_idx]
    p2 = tri_v2_2d[current_tri_idx]

    edge_starts = torch.stack([p0, p1, p2], dim=1)
    edge_ends   = torch.stack([p1, p2, p0], dim=1)
    edge_vecs   = edge_ends - edge_starts

    p_des = desired_pos.unsqueeze(1)
    p_cur = current_pos.unsqueeze(1)
    cp1   = batch_cross_2d(edge_vecs, p_des - edge_starts)
    cp0   = batch_cross_2d(edge_vecs, p_cur - edge_starts)
    crossing = (cp1 * cp0) < 0

    v_ap = p_des - edge_starts
    dot  = (v_ap * edge_vecs).sum(dim=-1)
    sq   = (edge_vecs * edge_vecs).sum(dim=-1)
    sq   = torch.where(sq < 1e-6, torch.ones_like(sq), sq)
    t    = torch.clamp(dot / sq, 0.0, 1.0)
    candidates = edge_starts + t.unsqueeze(-1) * edge_vecs

    has_crossing, first_idx = torch.max(crossing.long(), dim=1)
    selected = torch.gather(candidates, 1,
                            first_idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    return torch.where(has_crossing.view(-1, 1).bool(), selected, current_pos)


def calculate_initial_yaw_tensor(rotations):
    x, y, z, w = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    dir_x_h = -2 * w * y
    dir_z_h = y * y - w * w
    return torch.atan2(-dir_z_h, dir_x_h)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _hab_to_genesis(pos_list):
    x, y, z = pos_list[0], pos_list[1], pos_list[2]
    return x, -z, y


def _genesis_to_hab(cam_pos_gen: torch.Tensor) -> torch.Tensor:
    x = cam_pos_gen[:, 0]
    y = cam_pos_gen[:, 2] - CAMERA_HEIGHT
    z = -cam_pos_gen[:, 1]
    return torch.stack([x, y, z], dim=1)


def _update_camera(cam, cam_pos: torch.Tensor, cam_yaw: torch.Tensor):
    dir_x  = torch.cos(cam_yaw)
    dir_y  = torch.sin(cam_yaw)
    lookat = cam_pos.clone()
    lookat[:, 0] += dir_x
    lookat[:, 1] += dir_y
    lookat[:, 2]  = cam_pos[:, 2]
    cam.set_pose(pos=cam_pos, lookat=lookat)


# ---------------------------------------------------------------------------
# Model server HTTP client
# ---------------------------------------------------------------------------

class ModelServerClient:
    """Thin wrapper around the model-server REST API."""

    def __init__(self, server_url: str, timeout: int = 300):
        self.base    = server_url.rstrip("/")
        self.timeout = timeout
        self.sid     = None

    def wait_healthy(self, retries: int = 60, interval: float = 5.0):
        for _ in range(retries):
            try:
                r = requests.get(f"{self.base}/health", timeout=10)
                if r.status_code == 200:
                    print(f"[Worker] Model server healthy: {r.json()}", flush=True)
                    return
            except Exception:
                pass
            time.sleep(interval)
        raise RuntimeError(f"Model server at {self.base} did not become healthy")

    def create_session(self, n_agents: int, instructions: list[str]) -> str:
        r = requests.post(
            f"{self.base}/create_session",
            json={"n_agents": n_agents, "instructions": instructions},
            timeout=self.timeout,
        )
        r.raise_for_status()
        self.sid = r.json()["session_id"]
        print(f"[Worker] Session created: {self.sid}", flush=True)
        return self.sid

    def step(self, rgb_np: np.ndarray, active: np.ndarray, step_idx: int) -> list[int]:
        """Send (N, H, W, 3) uint8 rgb + active mask, receive action list."""
        shape_str  = " ".join(str(s) for s in rgb_np.shape)
        active_str = " ".join(str(int(a)) for a in active)
        r = requests.post(
            f"{self.base}/step",
            data={
                "session_id": self.sid,
                "step_idx":   str(step_idx),
                "shape_str":  shape_str,
                "active_str": active_str,
            },
            files={"rgb": ("rgb.bin", rgb_np.tobytes(), "application/octet-stream")},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["actions"]

    def delete_session(self):
        if self.sid:
            try:
                requests.delete(f"{self.base}/session/{self.sid}", timeout=30)
            except Exception:
                pass
            self.sid = None


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_scene(scene_id: str, scene_episodes: list, server_url: str,
              output_file: str, max_steps: int,
              cam_res: tuple[int, int], fov: int, light_scale: float,
              progress_file: str | None = None,
              save_parallel: bool = False, run_args: dict | None = None,
              success_distance: float = 3.0):
    """Run all episodes for one scene, write metrics to output_file."""

    SCAN_NAME = os.path.basename(os.path.dirname(scene_id))
    N         = len(scene_episodes)
    print(f"\n{'#'*60}\nScene {SCAN_NAME}  ({N} episodes)\n{'#'*60}", flush=True)
    _write_progress(progress_file, {
        "scan": SCAN_NAME, "total": N, "done": 0, "active": N,
        "step": 0, "max_step": max_steps,
        "fwd": 0, "left": 0, "right": 0, "stop": 0,
        "status": "starting",
    })

    client = ModelServerClient(server_url)
    # Workers may start before server is ready; wait a bit
    client.wait_healthy(retries=60, interval=5.0)

    # ----------------------------------------------------------------
    # 1. Genesis init
    # ----------------------------------------------------------------
    gs.init(backend=gs.cuda)
    renderer = gs.renderers.BatchRenderer(use_rasterizer=True)
    scene    = gs.Scene(
        renderer=renderer,
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            ambient_light=(1.0, 1.0, 1.0),
            plane_reflection=False,
        ),
    )

    # ----------------------------------------------------------------
    # 2. Load scene mesh
    # ----------------------------------------------------------------
    obj_dir = f"{SCENE_DATASETS}/{SCAN_NAME}/matterport_mesh"
    obj_files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]
    if not obj_files:
        print(f"[ERROR] No .obj found for {SCAN_NAME}", flush=True)
        _write_empty(output_file)
        return
    scene.add_entity(morph=gs.morphs.Mesh(
        file=os.path.join(obj_dir, obj_files[0]), fixed=True, collision=False
    ))

    # ----------------------------------------------------------------
    # 3. Load NavMesh → GPU tensors
    # ----------------------------------------------------------------
    navmesh_path = f"{SCENE_DATASETS}/{SCAN_NAME}/navmesh.npz"
    if not os.path.exists(navmesh_path):
        print(f"[ERROR] navmesh.npz not found: {navmesh_path}", flush=True)
        _write_empty(output_file)
        return
    nm    = np.load(navmesh_path)
    V_hab = torch.tensor(nm["verts"], device=gs.device, dtype=torch.float32)
    F_idx = torch.tensor(nm["faces"], device=gs.device, dtype=torch.long)

    V_gen = torch.zeros_like(V_hab)
    V_gen[:, 0] =  V_hab[:, 0]
    V_gen[:, 1] = -V_hab[:, 2]
    V_gen[:, 2] =  V_hab[:, 1]

    tri_v0_3d = V_gen[F_idx[:, 0]]; tri_v1_3d = V_gen[F_idx[:, 1]]; tri_v2_3d = V_gen[F_idx[:, 2]]
    tri_v0_2d = tri_v0_3d[:, :2];   tri_v1_2d = tri_v1_3d[:, :2];   tri_v2_2d = tri_v2_3d[:, :2]

    # ----------------------------------------------------------------
    # 4. Build scene + camera
    # ----------------------------------------------------------------
    cam = scene.add_camera(res=cam_res, fov=fov, GUI=False)
    scene.build(n_envs=N, env_spacing=(0.0, 0.0))

    # ----------------------------------------------------------------
    # 5. Initialise agent poses
    # ----------------------------------------------------------------
    cam_pos = torch.zeros(N, 3, device=gs.device, dtype=torch.float32)
    cam_yaw = torch.zeros(N,    device=gs.device, dtype=torch.float32)
    instructions   = []
    goal_positions = []
    gt_paths       = []

    for i, ep in enumerate(scene_episodes):
        sp = ep["start_position"]
        gx, gy, gz = _hab_to_genesis(sp)
        cam_pos[i]  = torch.tensor([gx, gy, gz + CAMERA_HEIGHT],
                                   dtype=torch.float32, device=gs.device)
        rot_q       = torch.tensor([ep["start_rotation"]],
                                   dtype=torch.float32, device=gs.device)
        cam_yaw[i]  = calculate_initial_yaw_tensor(rot_q).squeeze()
        instructions.append(ep["instruction"]["instruction_text"])
        goal_positions.append(ep["goals"][0]["position"])
        gt_paths.append(ep["reference_path"])

    # for i, ep in enumerate(scene_episodes):
    #     print(f"\n{'='*60}\nSlot {i+1}/{N} | ep {ep['episode_id']}\n"
    #           f"Instruction: {ep['instruction']['instruction_text']}\n{'='*60}",
    #           flush=True)

    # ----------------------------------------------------------------
    # 6. Create model-server session
    # ----------------------------------------------------------------
    client.create_session(N, instructions)

    # ----------------------------------------------------------------
    # 7. Initial floor snap + camera pose
    # ----------------------------------------------------------------
    _, _, current_tri_idx = batch_find_floor(
        cam_pos[:, :2], cam_pos[:, 2] - CAMERA_HEIGHT,
        tri_v0_2d, tri_v1_2d, tri_v2_2d,
        tri_v0_3d, tri_v1_3d, tri_v2_3d,
    )
    _update_camera(cam, cam_pos, cam_yaw)

    # ----------------------------------------------------------------
    # 8. Metrics tracking
    # ----------------------------------------------------------------
    active_mask    = torch.ones(N, dtype=torch.bool, device=gs.device)
    stop_called    = [False] * N   # whether agent explicitly called STOP
    goal_pos_t     = torch.tensor(goal_positions, dtype=torch.float32, device=gs.device)
    pred_path_lists = [[] for _ in range(N)]
    distances_lists = [[] for _ in range(N)]

    init_hab = _genesis_to_hab(cam_pos)
    for i in range(N):
        pred_path_lists[i].append(init_hab[i].cpu().numpy())
        distances_lists[i].append(torch.norm(init_hab[i] - goal_pos_t[i]).item())

    print(f"\n[{SCAN_NAME}] Starting rollout: {N} episodes, max {max_steps} steps",
          flush=True)
    _write_progress(progress_file, {
        "scan": SCAN_NAME, "total": N, "done": 0, "active": N,
        "step": 0, "max_step": max_steps,
        "fwd": 0, "left": 0, "right": 0, "stop": 0,
        "status": "running",
    })

    # ----------------------------------------------------------------
    # 9. Main step loop
    # ----------------------------------------------------------------
    try:
        for step_idx in range(max_steps):
            n_active = int(active_mask.sum().item())
            if n_active == 0:
                break

            # 9a. Parallel render
            rgb_batch, _, _, _ = cam.render(
                rgb=True, depth=False, segmentation=False, force_render=True
            )
            if rgb_batch.dtype == torch.uint8:
                rgb_batch = rgb_batch.float() / 255.0
            rgb_batch = torch.clamp(rgb_batch * light_scale * 255.0, 0, 255).byte()
            rgb_np    = rgb_batch.cpu().numpy()   # (N, H, W, 3) uint8

            # 9b. Batch inference via model server
            active_np = active_mask.cpu().numpy()
            actions   = client.step(rgb_np, active_np, step_idx)
            action_t  = torch.tensor(actions, device=gs.device, dtype=torch.long)

            action_summary = {
                "fwd":  int((action_t[active_mask] == 1).sum()),
                "left": int((action_t[active_mask] == 2).sum()),
                "right":int((action_t[active_mask] == 3).sum()),
                "stop": int((action_t[active_mask] == 0).sum()),
            }

            # 9c. Mark newly-stopped agents
            newly_stopped = (action_t == 0) & active_mask
            active_mask[newly_stopped] = False
            n_done_now = int((~active_mask).sum().item())
            for i in range(N):
                if newly_stopped[i]:
                    stop_called[i] = True
                    dist = distances_lists[i][-1] if distances_lists[i] else float("nan")
                    print(f"  [{SCAN_NAME}] ep {scene_episodes[i]['episode_id']} "
                          f"stopped @ step {step_idx:04d} | dist={dist:.2f}m",
                          flush=True)

            # Write heartbeat for orchestrator polling
            _write_progress(progress_file, {
                "scan":    SCAN_NAME,
                "total":   N,
                "done":    n_done_now,
                "active":  int(active_mask.sum().item()),
                "step":    step_idx + 1,
                "max_step": max_steps,
                "fwd":     action_summary["fwd"],
                "left":    action_summary["left"],
                "right":   action_summary["right"],
                "stop":    action_summary["stop"],
                "status":  "running",
            })

            if not active_mask.any():
                break

            # 9d. Rotate
            cam_yaw[(action_t == 2) & active_mask] += STEP_TURN
            cam_yaw[(action_t == 3) & active_mask] -= STEP_TURN

            # 9e. Forward movement with NavMesh collision
            fwd_m = (action_t == 1) & active_mask
            if fwd_m.any():
                fwd_idx = fwd_m.nonzero(as_tuple=True)[0]
                dx = torch.cos(cam_yaw[fwd_idx])
                dy = torch.sin(cam_yaw[fwd_idx])
                desired = torch.stack([
                    cam_pos[fwd_idx, 0] + dx * STEP_MOVE,
                    cam_pos[fwd_idx, 1] + dy * STEP_MOVE,
                ], dim=1)
                cur_floor_z = cam_pos[fwd_idx, 2] - CAMERA_HEIGHT

                valid, new_z, new_tri = batch_find_floor(
                    desired, cur_floor_z,
                    tri_v0_2d, tri_v1_2d, tri_v2_2d,
                    tri_v0_3d, tri_v1_3d, tri_v2_3d,
                )

                if ALLOW_SLIDING:
                    invalid = ~valid
                    if invalid.any():
                        inv_local = invalid.nonzero(as_tuple=True)[0]
                        slide_pos = batch_get_sliding_position(
                            cam_pos[fwd_idx[inv_local], :2],
                            desired[inv_local],
                            current_tri_idx[fwd_idx[inv_local]],
                            tri_v0_2d, tri_v1_2d, tri_v2_2d,
                        )
                        sv, sz, st = batch_find_floor(
                            slide_pos, cur_floor_z[inv_local],
                            tri_v0_2d, tri_v1_2d, tri_v2_2d,
                            tri_v0_3d, tri_v1_3d, tri_v2_3d,
                        )
                        if sv.any():
                            can_slide = inv_local[sv]
                            desired[can_slide] = slide_pos[sv]
                            new_z[can_slide]   = sz[sv]
                            new_tri[can_slide] = st[sv]
                            valid[can_slide]   = True

                moved = fwd_idx[valid]
                cam_pos[moved, 0] = desired[valid, 0]
                cam_pos[moved, 1] = desired[valid, 1]
                cam_pos[moved, 2] = new_z[valid] + CAMERA_HEIGHT
                current_tri_idx[moved] = new_tri[valid]

            # 9f. Update camera
            _update_camera(cam, cam_pos, cam_yaw)

            # 9g. Record metrics
            curr_hab = _genesis_to_hab(cam_pos)
            for i in range(N):
                if active_mask[i]:
                    pred_path_lists[i].append(curr_hab[i].cpu().numpy())
                    distances_lists[i].append(
                        torch.norm(curr_hab[i] - goal_pos_t[i]).item()
                    )

    finally:
        client.delete_session()

    # ----------------------------------------------------------------
    # 10. Compute per-episode metrics
    # ----------------------------------------------------------------
    metrics_all = {}
    for i, ep in enumerate(scene_episodes):
        pred_path   = np.array(pred_path_lists[i])
        gt_path_arr = np.array(gt_paths[i], dtype=float)
        dists       = distances_lists[i]

        # Habitat-aligned metrics:
        #   success requires STOP called AND distance < SUCCESS_DISTANCE
        #   oracle_success uses strict < (not <=)
        ep_success = 1.0 if (stop_called[i] and dists[-1] < success_distance) else 0.0
        metric = {
            "steps_taken":      len(pred_path) - 1,
            "distance_to_goal": dists[-1],
            "success":          ep_success,
            "oracle_success":   1.0 if min(dists) < success_distance else 0.0,
        }
        gt_length = dists[0]
        diffs     = pred_path[1:] - pred_path[:-1]
        path_len  = float(np.linalg.norm(diffs, axis=1).sum()) if len(diffs) else 0.0
        metric["path_length"] = path_len
        metric["spl"] = (
            float(ep_success * gt_length / max(gt_length, path_len))
            if max(gt_length, path_len) > 0 else 0.0
        )
        try:
            dtw_dist     = fastdtw(pred_path, gt_path_arr, dist=euclidean)[0]
            metric["ndtw"] = float(np.exp(-dtw_dist / (len(gt_path_arr) * success_distance)))
        except Exception:
            metric["ndtw"] = 0.0
        metric["sdtw"] = float(metric["ndtw"] * ep_success)

        metrics_all[ep["episode_id"]] = metric
        print(f"  ep {ep['episode_id']:>6} | {metric}", flush=True)

    with open(output_file, "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"[{SCAN_NAME}] Metrics written to {output_file}", flush=True)

    if save_parallel:
        _save_to_output_parallel(metrics_all, run_args or {})

    _write_progress(progress_file, {
        "scan": SCAN_NAME, "total": N, "done": len(metrics_all), "active": 0,
        "step": max_steps, "max_step": max_steps,
        "fwd": 0, "left": 0, "right": 0, "stop": 0,
        "status": "finished",
    })


def _save_to_output_parallel(metrics_all: dict, args_dict: dict):
    """Save structured results to output_parallel/ following orchestrator.py conventions."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_id = args_dict.get("scene_id", "unknown")
    scan_name = os.path.basename(os.path.dirname(scene_id))

    out_root = os.path.join(os.path.dirname(__file__), "output_parallel")
    run_dir = os.path.join(out_root, f"{ts}__{scan_name}__standalone")
    os.makedirs(run_dir, exist_ok=True)

    # Per-episode metrics
    metrics_path = os.path.join(run_dir, "metrics_final.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"[Worker] Saved {len(metrics_all)} episode metrics → {metrics_path}", flush=True)

    # Average metrics
    if metrics_all:
        keys = list(next(iter(metrics_all.values())).keys())
        avg = {k: float(np.mean([m[k] for m in metrics_all.values()])) for k in keys}
    else:
        avg = {}
    avg_path = os.path.join(run_dir, "avg_metrics.json")
    with open(avg_path, "w") as f:
        json.dump(avg, f, indent=2)
    print(f"[Worker] Saved avg metrics → {avg_path}", flush=True)

    if avg:
        print("\n=== Average Metrics ===", flush=True)
        for k, v in avg.items():
            print(f"  {k:<22}: {v:.4f}", flush=True)

    # Run config
    cfg_path = os.path.join(run_dir, "run_config.json")
    with open(cfg_path, "w") as f:
        json.dump(args_dict, f, indent=2)
    print(f"[Worker] Saved run config → {cfg_path}", flush=True)

    return run_dir


def _write_empty(output_file: str):
    with open(output_file, "w") as f:
        json.dump({}, f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_id",      type=str, required=True)
    p.add_argument("--episodes_file", type=str, required=True,
                   help="JSON file with list of episode dicts for this scene")
    p.add_argument("--server_url",    type=str, default="http://127.0.0.1:8100")
    p.add_argument("--output_file",   type=str, required=True)
    p.add_argument("--progress_file", type=str, default=None,
                   help="JSON file for heartbeat progress updates (optional)")
    p.add_argument("--max_steps",     type=int, default=_sc.get("MAX_STEPS", 500))
    p.add_argument("--cam_res",       type=int, nargs=2,
                   default=list(_sc.get("CAM_RES", (640, 480))))
    p.add_argument("--fov",           type=int, default=_sc.get("FOV", 120))
    p.add_argument("--light_scale",   type=float, default=_sc.get("LIGHT_SCALE", 4.0))
    p.add_argument("--success_distance", type=float, default=_sc.get("SUCCESS_DISTANCE", 3.0))
    p.add_argument("--save_parallel", action="store_true",
                   help="Also save structured results to output_parallel/ dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.episodes_file) as f:
        episodes = json.load(f)

    run_args = {
        "scene_id":      args.scene_id,
        "episodes_file": args.episodes_file,
        "server_url":    args.server_url,
        "max_steps":     args.max_steps,
        "cam_res":       list(args.cam_res),
        "fov":           args.fov,
        "light_scale":   args.light_scale,
        "n_episodes":    len(episodes),
    }

    try:
        run_scene(
            scene_id       = args.scene_id,
            scene_episodes = episodes,
            server_url     = args.server_url,
            output_file    = args.output_file,
            progress_file  = args.progress_file,
            max_steps      = args.max_steps,
            cam_res        = tuple(args.cam_res),
            fov            = args.fov,
            light_scale    = args.light_scale,
            save_parallel  = args.save_parallel,
            run_args       = run_args,
            success_distance = args.success_distance,
        )
    except Exception:
        traceback.print_exc()
        _write_empty(args.output_file)
        sys.exit(1)
