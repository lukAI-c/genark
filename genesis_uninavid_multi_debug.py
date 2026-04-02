import sys
sys.path.append("/data1/home/clk/project/Genesis")
import genesis as gs
import genesis.utils.tools as gs_tools
import torch
import numpy as np
import math
import os
import shutil
import gzip
import json
import random
import time
import cv2
import multiprocessing

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Import UniNaVid Agent
from agent_uninavid_genesis import UniNaVid_Agent
from genark_web_viewer import WebBroadcaster

# --- File Paths (CHANGE THESE TO YOUR ACTUAL FILE PATHS) ---
EPISODES_FILE = "/data1/home/clk/navid_ws/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
SCENE_DATASETS = "/data1/home/clk/project/genark/scene_datasets"
MODEL_PATH = "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2" # Updated to correct path

# --- Constants & Configuration ---
NUM_EPISODES = 1839 # Total episodes to run serially, original: 100
CAMERA_HEIGHT = 0.88 
CAM_RES = (640, 480)
FOV = 64  # 105
LIGHT_SCALE = 4.0
SAVE_IMAGES = False
SAVE_FAILED_EPISODES = False
SAVE_DEBUG_IMAGES = False
WEB_VIEWER = False
WEB_PORT = 9998

ACTION_MAP = {
    0: "stop",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right"
}

# Movement Parameters
STEP_MOVE = 0.25
STEP_TURN = np.deg2rad(30.0)
ALLOW_SLIDING = True
MAX_STEP_HEIGHT = 0.5

def load_data(num_envs):
    """Loads and samples episodes from JSON or JSON.GZ file."""
    print(f"Loading data from {EPISODES_FILE}...")
    if EPISODES_FILE.endswith('.gz'):
        with gzip.open(EPISODES_FILE, 'rt') as f:
            data = json.load(f)
    else:
        with open(EPISODES_FILE, 'r') as f:
            data = json.load(f)
    
    all_episodes = data['episodes']
    total_episodes = len(all_episodes)
    print(f"Found {total_episodes} episodes.")
    
    return all_episodes

# --- GPU NavMesh Logic ---

def batch_cross_2d(a, b):
    """
    a: (..., 2)
    b: (..., 2)
    returns: (...,) z-component of cross product
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def batch_is_point_in_triangle(pt, v0, v1, v2):
    """
    pt: (N, 1, 2)
    v0, v1, v2: (1, M, 2)
    Returns: (N, M) boolean
    """
    # vectors
    # v0_pt = pt - v0 # (N, M, 2)
    # but we can use cross product logic directly
    
    def sign(p1, p2, p3):
        return (p1[..., 0] - p3[..., 0]) * (p2[..., 1] - p3[..., 1]) - \
               (p2[..., 0] - p3[..., 0]) * (p1[..., 1] - p3[..., 1])

    d1 = sign(pt, v0, v1)
    d2 = sign(pt, v1, v2)
    d3 = sign(pt, v2, v0)

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    return ~(has_neg & has_pos)

def batch_get_z_on_triangle(v0, v1, v2, x, y):
    """
    v0, v1, v2: (1, M, 3) - 3D vertices
    x, y: (N, 1) - query coords
    Returns: (N, M) z values
    """
    # edge1 = v1 - v0
    # edge2 = v2 - v0
    # normal = cross(edge1, edge2)
    
    # Pre-calculated normals could be passed to save time, but calculating here for simplicity
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = torch.cross(edge1, edge2, dim=-1) # (1, M, 3)
    
    # Avoid div by zero
    mask = torch.abs(normal[..., 2]) < 1e-6
    # normal[..., 2][mask] = 1.0 # dummy to avoid nan, we will filter later
    # Actually, if normal.z is 0, it's a vertical wall, we can't step on it.
    
    d_x = x - v0[..., 0] # (N, M)
    d_y = y - v0[..., 1] # (N, M)
    
    # z = v1[2] - (normal[0] * d_x + normal[1] * d_y) / normal[2]
    num = normal[..., 0] * d_x + normal[..., 1] * d_y
    den = normal[..., 2]
    
    # Safe division
    den = torch.where(torch.abs(den) < 1e-6, torch.ones_like(den), den)
    
    z = v0[..., 2] - num / den
    
    # If it was vertical, set Z to infinity
    z = torch.where(mask.expand_as(z), torch.full_like(z, float('inf')), z)
    
    return z

def batch_find_floor(pos_2d, current_floor_z, tri_v0_2d, tri_v1_2d, tri_v2_2d, tri_v0_3d, tri_v1_3d, tri_v2_3d):
    """
    Finds the valid floor triangle and z-height for each agent.
    
    pos_2d: (N, 2)
    current_floor_z: (N,)
    tri_*: (M, 2) or (M, 3)
    
    Returns:
        valid_mask: (N,) bool
        new_z: (N,) float
        tri_indices: (N,) long (index of the triangle)
    """
    N = pos_2d.shape[0]
    M = tri_v0_2d.shape[0]
    
    # Reshape for broadcasting
    # Agent: (N, 1, ...)
    # Triangle: (1, M, ...)
    pt = pos_2d.unsqueeze(1) # (N, 1, 2)
    
    v0_2d = tri_v0_2d.unsqueeze(0) # (1, M, 2)
    v1_2d = tri_v1_2d.unsqueeze(0)
    v2_2d = tri_v2_2d.unsqueeze(0)
    
    # 1. Check inside
    inside_mask = batch_is_point_in_triangle(pt, v0_2d, v1_2d, v2_2d) # (N, M)
    
    # 2. Calculate Z
    v0_3d = tri_v0_3d.unsqueeze(0)
    v1_3d = tri_v1_3d.unsqueeze(0)
    v2_3d = tri_v2_3d.unsqueeze(0)
    
    # Only need to calc Z for inside triangles, but calculating all is vector-friendly
    z_values = batch_get_z_on_triangle(v0_3d, v1_3d, v2_3d, pt[..., 0], pt[..., 1]) # (N, M)
    
    # 3. Filter by height difference
    z_diff = torch.abs(z_values - current_floor_z.unsqueeze(1)) # (N, M)
    height_mask = z_diff < MAX_STEP_HEIGHT
    
    # 4. Combined validity
    valid_candidates = inside_mask & height_mask # (N, M)
    
    # 5. Find best (min distance)
    # Set invalid distances to infinity
    z_diff_masked = torch.where(valid_candidates, z_diff, torch.full_like(z_diff, float('inf')))
    
    min_dist, best_idx = torch.min(z_diff_masked, dim=1) # (N,)
    
    valid_mask = min_dist != float('inf')
    best_z = torch.gather(z_values, 1, best_idx.unsqueeze(1)).squeeze(1)
    
    return valid_mask, best_z, best_idx

def batch_get_sliding_position(current_pos, desired_pos, current_tri_idx, tri_v0_2d, tri_v1_2d, tri_v2_2d):
    """
    Computes sliding position for agents.
    
    current_pos: (N, 2)
    desired_pos: (N, 2)
    current_tri_idx: (N,) indices of the triangle the agent is currently in
    
    Returns:
        candidate_pos: (N, 2)
    """
    # Gather current triangle vertices
    # (N, 2)
    p0 = tri_v0_2d[current_tri_idx]
    p1 = tri_v1_2d[current_tri_idx]
    p2 = tri_v2_2d[current_tri_idx]
    
    # Edges: (N, 3, 2)
    # Edge 0: p0 -> p1
    # Edge 1: p1 -> p2
    # Edge 2: p2 -> p0
    edge_starts = torch.stack([p0, p1, p2], dim=1) # (N, 3, 2)
    edge_ends = torch.stack([p1, p2, p0], dim=1)   # (N, 3, 2)
    
    edge_vecs = edge_ends - edge_starts # (N, 3, 2)
    edge_lens = torch.norm(edge_vecs, dim=-1, keepdim=True) # (N, 3, 1)
    edge_dirs = edge_vecs / (edge_lens + 1e-6) # (N, 3, 2)
    
    move_vec = (desired_pos - current_pos).unsqueeze(1) # (N, 1, 2)
    
    # Check crossing
    # cp1 = cross(edge_vec, desired_pos - edge_start)
    # cp0 = cross(edge_vec, current_pos - edge_start)
    
    p_desired = desired_pos.unsqueeze(1) # (N, 1, 2)
    p_current = current_pos.unsqueeze(1) # (N, 1, 2)
    
    vec_to_desired = p_desired - edge_starts
    vec_to_current = p_current - edge_starts
    
    cp1 = batch_cross_2d(edge_vecs, vec_to_desired) # (N, 3)
    cp0 = batch_cross_2d(edge_vecs, vec_to_current) # (N, 3)
    
    crossing_mask = (cp1 * cp0) < 0 # (N, 3)
    
    # Calculate slide (Closest Point on Segment - Habitat Logic)
    # t = (AP . AB) / (AB . AB)
    
    # Vector from edge start to desired pos
    v_ap = p_desired - edge_starts # (N, 3, 2)
    
    dot_ap_ab = (v_ap * edge_vecs).sum(dim=-1) # (N, 3)
    sq_len_ab = (edge_vecs * edge_vecs).sum(dim=-1) # (N, 3)
    
    # Avoid div by zero
    sq_len_ab = torch.where(sq_len_ab < 1e-6, torch.ones_like(sq_len_ab), sq_len_ab)
    
    t = dot_ap_ab / sq_len_ab
    t = torch.clamp(t, 0.0, 1.0)
    
    candidates = edge_starts + t.unsqueeze(-1) * edge_vecs # (N, 3, 2)
    
    # We prioritize the first crossing edge.
    # If no crossing (shouldn't happen if we are sliding), return current_pos
    # If multiple, pick first.
    
    # crossing_mask: (N, 3)
    # We want indices of first True.
    # A trick: conversion to float, add noise? No.
    # value, index = max(mask) ? returns first?
    
    has_crossing, first_idx = torch.max(crossing_mask.long(), dim=1)
    
    # Gather the selected candidate
    # (N, 2)
    selected_candidate = torch.gather(candidates, 1, first_idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    
    # If no crossing, stay at current_pos (or desired_pos? Logic says current_pos if sliding fails)
    final_pos = torch.where(has_crossing.view(-1, 1).bool(), selected_candidate, current_pos)
    
    return final_pos

def calculate_initial_yaw_tensor(rotations):
    # rotations: (N, 4) [x, y, z, w]
    x, y, z, w = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    dir_x_h = -2 * w * y
    dir_z_h = y * y - w * w
    dir_x_g = dir_x_h
    dir_y_g = -dir_z_h
    return torch.atan2(dir_y_g, dir_x_g)

# --- Process Scene Function ---

def process_scene(scene_id, scene_episodes, metrics_queue, scene_loop_idx, total_scenes):
    try:
        SCAN_NAME = os.path.basename(os.path.dirname(scene_id))
        
        print(f"\n########################################################")
        print(f"Starting Scene {scene_loop_idx}/{total_scenes}: {SCAN_NAME} ({len(scene_episodes)} episodes)")
        print(f"########################################################")
        
        # --- 1. Initialization (Per Process) ---
        gs.init(backend=gs.cuda)
        
        renderer = gs.renderers.BatchRenderer(use_rasterizer=True)
        
        # --- 2. Setup Agents ---
        eva_vit_path = os.path.join("model_zoo", "eva_vit_g.pth")
        if os.path.exists(eva_vit_path):
            print(f"Found eva_vit_g.pth at {eva_vit_path}")
            
        agent = None
        if 'UniNaVid_Agent' in globals():
            print(f"Initializing UniNaVid Agent with model: {MODEL_PATH}")
            try:
                agent = UniNaVid_Agent(model_path=MODEL_PATH, result_path="output_uninavid", exp_save="", save_debug_images=SAVE_DEBUG_IMAGES)
            except Exception as e:
                print(f"Failed to initialize agent: {e}")
        
        metrics_all = {}
        
        # Create Scene
        scene = gs.Scene(
             renderer=renderer,
             show_viewer=False,
             vis_options=gs.options.VisOptions(ambient_light=(1.0, 1.0, 1.0), plane_reflection=False)
        )

        # --- Load Scene Mesh ---
        obj_dir = f"{SCENE_DATASETS}/{SCAN_NAME}/matterport_mesh"
        try:
            obj_files = [f for f in os.listdir(obj_dir) if f.endswith('.obj')]
            if not obj_files:
                raise FileNotFoundError
            obj_path = os.path.join(obj_dir, obj_files[0])
            scene.add_entity(morph=gs.morphs.Mesh(file=obj_path, fixed=True, collision=False))
        except (FileNotFoundError, IndexError):
            print(f"Error: No .obj file found for SCAN_NAME '{SCAN_NAME}' in {obj_dir}")
            metrics_queue.put({})
            return

        # --- Load NavMesh & Convert to GPU ---
        navmesh_npz_path = f"{SCENE_DATASETS}/{SCAN_NAME}/navmesh.npz"
        try:
            nm = np.load(navmesh_npz_path)
        except FileNotFoundError:
            print(f"Error: navmesh.npz not found at {navmesh_npz_path}")
            metrics_queue.put({})
            return
            
        V_habitat = torch.tensor(nm['verts'], device=gs.device, dtype=torch.float32)
        F = torch.tensor(nm['faces'], device=gs.device, dtype=torch.long)
        
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

        # Add a single camera
        cam = scene.add_camera(res=CAM_RES, fov=FOV, GUI=False)
        
        # Build Scene
        scene.build(n_envs=1, env_spacing=(0.0, 0.0))

        broadcaster = None
        if WEB_VIEWER:
            broadcaster = WebBroadcaster(
                output_dir="genark_web_output_uninavid",
                port=WEB_PORT,
                quality=85,
                throttle_fps=10,
            )

        # --- Loop Over Episodes in this Scene ---
        for ep_idx, episode in enumerate(scene_episodes):
            print(f"\n========================================================")
            print(f"Scene {SCAN_NAME} | Episode {ep_idx+1}/{len(scene_episodes)} | Global ID: {episode['episode_id']}")
            print(f"Instruction: {episode['instruction']['instruction_text']}")
            print(f"========================================================")
            
            # Reset Agent
            if agent is not None:
                agent.reset()
                
            # Prepare Episode Data
            start_pos = episode['start_position']
            sx = start_pos[0]
            sy = -start_pos[2]
            sz = start_pos[1] + CAMERA_HEIGHT
            
            start_rot = episode['start_rotation']
            
            gt_path = episode['reference_path']
            goal_pos = episode['goals'][0]['position']
            
            # Max steps
            if agent is not None:
                max_steps = 500
            else:
                max_steps = 500
                
            # Tensors (Batch size = 1)
            cam_pos = torch.tensor([[sx, sy, sz]], device=gs.device, dtype=torch.float32)
            cam_rot_q = torch.tensor([start_rot], device=gs.device, dtype=torch.float32)
            cam_yaw = calculate_initial_yaw_tensor(cam_rot_q)
            
            current_step = 0
            finished = False
            
            # Initial Floor Check
            _, _, current_tri_idx = batch_find_floor(
                cam_pos[:, :2], 
                cam_pos[:, 2] - CAMERA_HEIGHT,
                tri_v0_2d, tri_v1_2d, tri_v2_2d, 
                tri_v0_3d, tri_v1_3d, tri_v2_3d
            )
            
            # Initial Camera Update
            dir_x = torch.cos(cam_yaw)
            dir_y = torch.sin(cam_yaw)
            target_lookat = cam_pos.clone()
            target_lookat[:, 0] += dir_x
            target_lookat[:, 1] += dir_y
            target_lookat[:, 2] = cam_pos[:, 2]
            cam.set_pose(pos=cam_pos, lookat=target_lookat)
            
            # Metrics setup
            goal_pos_tensor = torch.tensor([goal_pos], device=gs.device, dtype=torch.float32)
            pred_path_list = []
            distances_list = []
            
            def get_habitat_coords(cam_pos_gen):
                x = cam_pos_gen[:, 0]
                y = cam_pos_gen[:, 2] - CAMERA_HEIGHT
                z = -cam_pos_gen[:, 1]
                return torch.stack([x, y, z], dim=1)
                
            curr_pos_hab = get_habitat_coords(cam_pos)
            pred_path_list.append(curr_pos_hab.cpu().numpy()[0])
            curr_dist = torch.norm(curr_pos_hab - goal_pos_tensor, dim=1).item()
            distances_list.append(curr_dist)
            
            # Episode Loop
            step_loop_idx = 0
            
            # Images for this episode
            episode_images = [] if (SAVE_IMAGES or SAVE_FAILED_EPISODES) else None
            
            # --- 1) 准备计时累加器和步数计数器 ---
            sum_render = 0.0
            sum_copy = 0.0
            sum_agent = 0.0
            sum_physics = 0.0
            cnt_steps = 0
            cnt_physics_steps = 0
            
            while not finished and step_loop_idx < max_steps:
                # --- 2) 记录总时间并计算耗时 ---
                cnt_steps += 1
                
                # --- T_render 计时段 ---
                torch.cuda.synchronize()
                t0_render = time.perf_counter()
                
                # 1. Render
                rgb_batch, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, force_render=True)
                
                if rgb_batch.dtype == torch.uint8:
                    rgb_batch = rgb_batch.float() / 255.0
                rgb_batch = rgb_batch * LIGHT_SCALE
                rgb_batch = torch.clamp(rgb_batch * 255.0, 0, 255).byte()
                
                # --- T_render 计时段 ---
                torch.cuda.synchronize()
                sum_render += time.perf_counter() - t0_render
                
                # --- T_cpu_copy 计时段 ---
                torch.cuda.synchronize()
                t0_copy = time.perf_counter()
                
                rgb_cpu = rgb_batch.cpu().numpy()[0] # (H, W, 3)
                
                # --- T_cpu_copy 计时段 ---
                torch.cuda.synchronize()
                sum_copy += time.perf_counter() - t0_copy
                
                if episode_images is not None:
                    episode_images.append(rgb_cpu[..., ::-1])

                # 2. Agent Action
                action = 0 # Stop
                
                if agent is not None:
                    obs = {
                         "rgb": rgb_cpu,
                         "instruction": {"text": episode['instruction']['instruction_text']}
                    }
                    info = {}
                    
                    # --- T_agent 计时段 ---
                    torch.cuda.synchronize()
                    t0_agent = time.perf_counter()
                    
                    try:
                        action_dict = agent.act(obs, info, episode['episode_id'], step_idx=step_loop_idx)
                        if action_dict and 'action' in action_dict:
                            action = int(action_dict['action'])
                        else:
                            action = 0
                    except Exception as e:
                        print(f"Agent failed: {e}")
                        action = 0
                        
                    # --- T_agent 计时段 ---
                    torch.cuda.synchronize()
                    sum_agent += time.perf_counter() - t0_agent
                
                else:
                    action = 0 

                if broadcaster is not None:
                    broadcaster.send_frame(rgb_cpu, metadata={
                        "episode_id": episode['episode_id'],
                        "episode": f"{ep_idx+1}/{len(scene_episodes)}",
                        "step": step_loop_idx,
                        "action": ACTION_MAP.get(action, "unknown"),
                        "instruction": episode['instruction']['instruction_text'][:120],
                    })

                if action == 0:
                    finished = True
                    break
                
                # --- T_physics 计时段 ---
                torch.cuda.synchronize()
                t0_physics = time.perf_counter()
                cnt_physics_steps += 1
                    
                # 3. Physics
                is_forward = (action == 1)
                is_left = (action == 2)
                is_right = (action == 3)
                
                if is_left:
                    cam_yaw += STEP_TURN
                if is_right:
                    cam_yaw -= STEP_TURN
                    
                if is_forward:
                    dir_x = torch.cos(cam_yaw)
                    dir_y = torch.sin(cam_yaw)
                    move_vec_x = dir_x * STEP_MOVE
                    move_vec_y = dir_y * STEP_MOVE
                    
                    desired_pos_x = cam_pos[:, 0] + move_vec_x
                    desired_pos_y = cam_pos[:, 1] + move_vec_y
                    desired_pos = torch.stack([desired_pos_x, desired_pos_y], dim=1)
                    
                    current_floor_z = cam_pos[:, 2] - CAMERA_HEIGHT
                    
                    valid_mask, new_floor_z, new_tri_idx = batch_find_floor(
                        desired_pos, current_floor_z,
                        tri_v0_2d, tri_v1_2d, tri_v2_2d, 
                        tri_v0_3d, tri_v1_3d, tri_v2_3d
                    )
                    
                    if ALLOW_SLIDING:
                         needs_slide = (~valid_mask)
                         if needs_slide.item():
                             slide_pos = batch_get_sliding_position(
                                cam_pos[:, :2], desired_pos, current_tri_idx,
                                tri_v0_2d, tri_v1_2d, tri_v2_2d
                             )
                             slide_valid, slide_z, slide_tri_idx = batch_find_floor(
                                slide_pos, current_floor_z,
                                tri_v0_2d, tri_v1_2d, tri_v2_2d, 
                                tri_v0_3d, tri_v1_3d, tri_v2_3d
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
                        
                # Update Camera Pose
                dir_x = torch.cos(cam_yaw)
                dir_y = torch.sin(cam_yaw)
                target_lookat = cam_pos.clone()
                target_lookat[:, 0] += dir_x
                target_lookat[:, 1] += dir_y
                target_lookat[:, 2] = cam_pos[:, 2]
                cam.set_pose(pos=cam_pos, lookat=target_lookat)
                
                # Record
                curr_pos_hab = get_habitat_coords(cam_pos)
                pred_path_list.append(curr_pos_hab.cpu().numpy()[0])
                curr_dist = torch.norm(curr_pos_hab - goal_pos_tensor, dim=1).item()
                distances_list.append(curr_dist)
                
                # --- T_physics 计时段 ---
                torch.cuda.synchronize()
                sum_physics += time.perf_counter() - t0_physics
                
                step_loop_idx += 1
                
            # --- 3) 打印单个 Episode 的耗时统计 ---
            avg_render = sum_render / max(1, cnt_steps)
            avg_copy = sum_copy / max(1, cnt_steps)
            avg_agent = sum_agent / max(1, cnt_steps)
            avg_physics = sum_physics / max(1, cnt_physics_steps)
            
            print(f"\n--- Episode {ep_idx+1} Timing Profiling (Avg per step) ---")
            print(f"  T_render  : {avg_render:.4f} s")
            print(f"  T_cpu_copy: {avg_copy:.4f} s")
            print(f"  T_agent   : {avg_agent:.4f} s")
            print(f"  T_physics : {avg_physics:.4f} s")
            print(f"----------------------------------------------------\n")    
            # End of Episode
            # Calculate Metric
            metric = {}
            pred_path = np.array(pred_path_list)
            gt_path_arr = np.array(gt_path).astype(float)
            
            metric['steps_taken'] = step_loop_idx
            metric['distance_to_goal'] = distances_list[-1]
            metric['success'] = 1.0 if distances_list[-1] <= 3.0 else 0.0
            metric['oracle_success'] = 1.0 if min(distances_list) <= 3.0 else 0.0
            
            # SPL
            gt_length = distances_list[0] 
            
            diffs = pred_path[1:] - pred_path[:-1]
            path_length = float(np.linalg.norm(diffs, axis=1).sum())
            metric['path_length'] = path_length
            metric['spl'] = float(metric['success'] * gt_length / max(gt_length, path_length)) if max(gt_length, path_length) > 0 else 0.0
            
            # NDTW (simplified)
            dtw_distance = float('inf')
            if 'fastdtw' in globals():
                 try:
                     dtw_distance = fastdtw(pred_path, gt_path_arr, dist=euclidean)[0]
                 except: pass
                 
            if dtw_distance != float('inf'):
                metric['ndtw'] = float(np.exp(-dtw_distance / (len(gt_path_arr) * 3.0)))
            else:
                metric['ndtw'] = 0.0
            metric['sdtw'] = float(metric['ndtw'] * metric['success'])
            
            metrics_all[episode['episode_id']] = metric
            print(f"Episode Metric: {metric}")
            
            # Save Images if needed
            should_save = SAVE_IMAGES or (SAVE_FAILED_EPISODES and metric['success'] == 0.0)
            if should_save and episode_images:
                save_dir = f"output/episode_{episode['episode_id']}"
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.makedirs(save_dir)
                for i, img in enumerate(episode_images):
                    cv2.imwrite(os.path.join(save_dir, f"step_{i:04d}.png"), img)
                print(f"Saved images to {save_dir}")

            del cam_pos, cam_rot_q, cam_yaw
            
        if broadcaster is not None:
            broadcaster.finish()

        metrics_queue.put(metrics_all)
        time.sleep(2)
        os._exit(0)
        
    except Exception as e:
        print(f"Error in process_scene {scene_id}: {e}")
        import traceback
        traceback.print_exc()
        try:
            if broadcaster is not None:
                broadcaster.finish()
        except Exception:
            pass
        metrics_queue.put({})
        time.sleep(2)
        os._exit(1)

def run_serial_evaluation(total_episodes: int):
    # Load ALL episodes
    episodes = load_data(total_episodes)
    if not episodes:
        print("No episodes loaded.")
        return

    # Group episodes by scene
    episodes_by_scene = {}
    for ep in episodes:
        scene_id = ep['scene_id']
        if scene_id not in episodes_by_scene:
            episodes_by_scene[scene_id] = []
        episodes_by_scene[scene_id].append(ep)

    print(f"Total Scenes to evaluate: {len(episodes_by_scene)}")
    for sid, eps in episodes_by_scene.items():
        print(f"  - {sid}: {len(eps)} episodes")

    # Set start method for multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    metrics_all = {}
    scene_loop_idx = 0
    
    for scene_id, scene_episodes in episodes_by_scene.items():
        scene_loop_idx += 1
        SCAN_NAME = os.path.basename(os.path.dirname(scene_id))
        
        # DEBUG: Skip scenes until we reach Z6MFQCViBuw
        # if SCAN_NAME != "Z6MFQCViBuw":
        #     continue

        metrics_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=process_scene, args=(scene_id, scene_episodes, metrics_queue, scene_loop_idx, len(episodes_by_scene)))
        p.start()
        
        # Wait for the process to finish
        scene_metrics = metrics_queue.get()
        p.join()
        
        if scene_metrics:
            metrics_all.update(scene_metrics)

    # Final Average
    output_metrics_file = f"metrics_multi_scene_eval.json"
    with open(output_metrics_file, 'w') as f:
        json.dump(metrics_all, f, indent=4)
        
    if len(metrics_all) > 0:
        avg_metrics = {}
        keys = next(iter(metrics_all.values())).keys()
        for k in keys:
            avg_metrics[k] = float(np.mean([m[k] for m in metrics_all.values()]))
        print("\n=== Average Metrics ===")
        import pprint
        pprint.pprint(avg_metrics)

if __name__ == "__main__":
    try:
        run_serial_evaluation(total_episodes=NUM_EPISODES)
    except Exception as e:
        import traceback
        traceback.print_exc()
