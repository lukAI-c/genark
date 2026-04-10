import genesis as gs
import genesis.utils.tools as gs_tools
import torch
import numpy as np
import math
import os
import shutil
import json
import random
import time
import cv2

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Import UniNaVid Agent
from agent_uninavid_genesis import UniNaVid_Agent

# --- File Paths (CHANGE THESE TO YOUR ACTUAL FILE PATHS) ---
EPISODES_FILE = "/home/hongyuding/code/Genesis/scripts/scenes_merged/mp3d_zsNo4HB9uLZ.json"
SCENE_DATASETS = "/home/hongyuding/data/genesis"
MODEL_PATH = "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"

# --- Constants & Configuration ---
NUM_PARALLEL_ENVS = 1 # Uni-NaVid is heavy, start with 1
CAMERA_HEIGHT = 0.88
CAM_RES = (640, 480)
FOV = 64
LIGHT_SCALE = 4.0
SAVE_IMAGES = False
SAVE_FAILED_EPISODES = False

ACTION_MAP = {
    0: "stop",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right"
}

# Movement Parameters
STEP_MOVE = 0.25
STEP_TURN = np.deg2rad(15.0)
ALLOW_SLIDING = True
MAX_STEP_HEIGHT = 0.5

def load_data(num_envs):
    """Loads and samples episodes from the JSON file."""
    print(f"Loading data from {EPISODES_FILE}...")
    with open(EPISODES_FILE, 'r') as f:
        data = json.load(f)
    
    all_episodes = data['episodes']
    total_episodes = len(all_episodes)
    print(f"Found {total_episodes} episodes.")
    
    sampled_episodes = random.choices(all_episodes, k=num_envs)
    print(f"Sampled {len(sampled_episodes)} episodes for parallel execution.")
    
    return sampled_episodes

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
    
    # Mask out non-crossing
    # We need to select one candidate per agent.
    # Let's just pick the first one that crosses.
    
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

def run_parallel_episodes(num_envs: int):
    episodes = load_data(num_envs)
    if not episodes:
        print("No episodes loaded.")
        return

    scene_id = episodes[0]['scene_id']
    SCAN_NAME = os.path.basename(os.path.dirname(scene_id))
    print(f"Running {len(episodes)} episodes in parallel for scene: {SCAN_NAME}")

    # --- 1. Initialization ---
    gs.init(backend=gs.cuda)
    
    renderer = gs.renderers.BatchRenderer(use_rasterizer=True)
    scene = gs.Scene(
        renderer=renderer,
        show_viewer=False,
        vis_options=gs.options.VisOptions(ambient_light=(1.0, 1.0, 1.0), plane_reflection=False)
    )

    # --- 2. Load Scene ---
    obj_dir = f"{SCENE_DATASETS}/{SCAN_NAME}/matterport_mesh"
    try:
        obj_files = [f for f in os.listdir(obj_dir) if f.endswith('.obj')]
        if not obj_files:
            raise FileNotFoundError
        obj_path = os.path.join(obj_dir, obj_files[0])
        scene.add_entity(morph=gs.morphs.Mesh(file=obj_path, fixed=True, collision=False))
    except (FileNotFoundError, IndexError):
        print(f"Error: No .obj file found for SCAN_NAME '{SCAN_NAME}'")
        return

    # --- 3. Load NavMesh & Convert to GPU ---
    navmesh_npz_path = f"{SCENE_DATASETS}/{SCAN_NAME}/navmesh.npz"
    try:
        nm = np.load(navmesh_npz_path)
    except FileNotFoundError:
        print(f"Error: navmesh.npz not found")
        return
        
    V_habitat = torch.tensor(nm['verts'], device=gs.device, dtype=torch.float32)
    F = torch.tensor(nm['faces'], device=gs.device, dtype=torch.long)
    
    V_genesis = torch.zeros_like(V_habitat)
    V_genesis[:, 0] = V_habitat[:, 0]
    V_genesis[:, 1] = -V_habitat[:, 2] 
    V_genesis[:, 2] = V_habitat[:, 1]
    
    # Pre-gather triangle vertices for batch processing
    # (M, 3)
    tri_v0_3d = V_genesis[F[:, 0]]
    tri_v1_3d = V_genesis[F[:, 1]]
    tri_v2_3d = V_genesis[F[:, 2]]
    
    # (M, 2)
    tri_v0_2d = tri_v0_3d[:, :2]
    tri_v1_2d = tri_v1_3d[:, :2]
    tri_v2_2d = tri_v2_3d[:, :2]

    # --- 4. Setup Agents ---
    
    # Check for eva_vit_g.pth
    eva_vit_path = os.path.join("model_zoo", "eva_vit_g.pth")
    if os.path.exists(eva_vit_path):
        print(f"Found eva_vit_g.pth at {eva_vit_path}")
    else:
        print(f"Warning: eva_vit_g.pth not found at {eva_vit_path}. UniNaVid might fail if it relies on it.")

    # Initialize UniNaVid Agent
    agent = None
    if 'UniNaVid_Agent' in globals():
        if NUM_PARALLEL_ENVS == 1:
            print(f"Initializing UniNaVid Agent with model: {MODEL_PATH}")
            try:
                # result_path and exp_save can be adjusted
                agent = UniNaVid_Agent(model_path=MODEL_PATH, result_path="output_uninavid", exp_save="")
                agent.reset()
            except Exception as e:
                print(f"Failed to initialize agent: {e}")
                print("Continuing with Ground Truth actions only if possible (or exiting)...")
                # return # Uncomment to stop if agent fails
        else:
             print("Warning: NUM_PARALLEL_ENVS > 1. UniNaVid Agent currently supports only 1 environment. Using GT actions for others?")
    
    # Prepare data for GPU tensors
    start_pos_list = []
    start_rot_list = []
    actions_list = []
    gt_paths_list = []
    goal_pos_list = []
    max_steps = 0
    
    # Add a single camera for batched rendering
    # We will update this single camera with a batch of poses (N, 3)
    cam = scene.add_camera(res=CAM_RES, fov=FOV, GUI=False)
    
    for i, episode in enumerate(episodes):
        start_pos = episode['start_position']
        # Convert to Genesis coord
        sx = start_pos[0]
        sy = -start_pos[2]
        sz = start_pos[1] + CAMERA_HEIGHT
        start_pos_list.append([sx, sy, sz])
        
        start_rot_list.append(episode['start_rotation'])
        
        acts = episode['actions']
        actions_list.append(acts)
        
        gt_paths_list.append(episode['locations'])
        goal_pos_list.append(episode['goal_position'])

        if len(acts) > max_steps:
            max_steps = len(acts)

    # Build scene with N parallel environments
    # env_spacing=(0.0, 0.0) ensures all environments are superimposed at the origin,
    # so we don't need to handle coordinate offsets for NavMesh or Camera.
    scene.build(n_envs=num_envs, env_spacing=(0.0, 0.0))

    # Create Tensors on GPU
    N = len(episodes)
    
    # State
    cam_pos = torch.tensor(start_pos_list, device=gs.device, dtype=torch.float32) # (N, 3)
    cam_rot_q = torch.tensor(start_rot_list, device=gs.device, dtype=torch.float32) # (N, 4)
    cam_yaw = calculate_initial_yaw_tensor(cam_rot_q) # (N,)
    
    current_step = torch.zeros(N, device=gs.device, dtype=torch.long)
    finished = torch.zeros(N, device=gs.device, dtype=torch.bool)
    
    # Actions: Pad with 0 (STOP)
    actions_padded = torch.zeros((N, max_steps), device=gs.device, dtype=torch.long)
    episode_lengths = torch.tensor([len(a) for a in actions_list], device=gs.device, dtype=torch.long)
    
    if agent is not None:
        # If agent is driving, allow it to run longer than GT
        episode_lengths = torch.full((N,), 500, device=gs.device, dtype=torch.long)
        print("Agent active: Overriding episode limits to 500 steps.")
    
    for i, acts in enumerate(actions_list):
        actions_padded[i, :len(acts)] = torch.tensor(acts, device=gs.device, dtype=torch.long)

    # Initial Triangle Index (for sliding)
    # Perform a check to find where they are
    _, _, current_tri_idx = batch_find_floor(
        cam_pos[:, :2], 
        cam_pos[:, 2] - CAMERA_HEIGHT,
        tri_v0_2d, tri_v1_2d, tri_v2_2d, 
        tri_v0_3d, tri_v1_3d, tri_v2_3d
    )

    # Initial Pose Update
    dir_x = torch.cos(cam_yaw)
    dir_y = torch.sin(cam_yaw)
    target_lookat = cam_pos.clone()
    target_lookat[:, 0] += dir_x
    target_lookat[:, 1] += dir_y
    target_lookat[:, 2] = cam_pos[:, 2] # Look horizontal
    
    # Update camera (Batched)
    cam.set_pose(pos=cam_pos, lookat=target_lookat)

    # --- Metric Tracking Initialization ---
    # Goal positions (N, 3) in Habitat coordinates
    goal_pos_tensor = torch.tensor(goal_pos_list, device=gs.device, dtype=torch.float32)
    
    # Store path and distances history (on GPU to avoid sync overhead)
    # We will append (N, 3) and (N,) tensors at each step
    pred_paths_gpu = [] 
    distances_gpu = []
    
    # Helper to get Habitat coords from Genesis cam_pos
    def get_habitat_coords(cam_pos_gen):
        # Genesis: x, y, z
        # Habitat: x, z - H, -y
        x = cam_pos_gen[:, 0]
        y = cam_pos_gen[:, 2] - CAMERA_HEIGHT
        z = -cam_pos_gen[:, 1]
        return torch.stack([x, y, z], dim=1)

    # Initial record
    curr_pos_hab = get_habitat_coords(cam_pos)
    pred_paths_gpu.append(curr_pos_hab.clone()) # clone to be safe
    
    # Initial distance
    # Euclidean distance to goal
    curr_dist = torch.norm(curr_pos_hab - goal_pos_tensor, dim=1)
    distances_gpu.append(curr_dist.clone())

    print("Starting simulation loop (GPU accelerated)...")
    
    start_time = time.time()
    
    # Total frames counter
    total_frames = 0

    # --- Main Loop ---
    # Total steps = max_steps + 1 (initial)
    # We will loop until all finished
    
    step_idx = 0
    # Pre-allocate indices tensor for gathering actions
    agent_indices = torch.arange(N, device=gs.device, dtype=torch.long)
    
    # Store images in CPU memory if saving is enabled
    # List of lists: all_images[env_idx] -> list of numpy arrays
    all_images = [[] for _ in range(N)] if (SAVE_IMAGES or SAVE_FAILED_EPISODES) else None
    
    # Time tracking
    t_render = 0.0
    t_img_proc = 0.0
    t_cpu_sync = 0.0 # Time spent moving data to CPU (if saving)
    t_physics = 0.0
    t_step_loop_start = time.time()
    
    while not torch.all(finished):
        iter_start = time.time()
        
        # 1. Render
        # Directly render using the batched camera
        # returns: (rgb_batch, depth_batch, seg_batch, normal_batch)
        rgb_batch, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, force_render=True)
        torch.cuda.synchronize()
        t1 = time.time()
        t_render += (t1 - iter_start)
        
        # 2. Image Processing on GPU
        # In RL training, we typically normalize to [0, 1] or keep as uint8.
        # Here we apply the same processing as before but keep it on GPU.
        if rgb_batch.dtype == torch.uint8:
            rgb_batch = rgb_batch.float() / 255.0
            
        rgb_batch = rgb_batch * LIGHT_SCALE
        rgb_batch = torch.clamp(rgb_batch * 255.0, 0, 255).byte()
        torch.cuda.synchronize()
        t2 = time.time()
        t_img_proc += (t2 - t1)
        
        # At this point, `rgb_batch` is a (N, H, W, 3) uint8 Tensor on GPU.
        
        # We need CPU images if we are saving them OR if we have an agent running
        rgb_cpu = None
        if (SAVE_IMAGES or SAVE_FAILED_EPISODES) or (agent is not None):
            rgb_cpu = rgb_batch.cpu().numpy()
            
            # --- AGENT RGB FIX ---
            # Agent expects RGB images, likely uint8.
            # But we should verify if it expects BGR (cv2 default) or RGB.
            # Usually models trained with PIL/torchvision expect RGB.
            # Genesis render returns RGB by default.
            # However, rgb_cpu[i] is taken directly from render output.
            # Let's keep it as is, assuming Agent handles normalization internally (e.g. CLIP processor).
            # But ensure shape is correct (H, W, 3).
            pass
            
        if (SAVE_IMAGES or SAVE_FAILED_EPISODES):
            # Move to CPU memory and store in list
            # We do NOT save to disk here to avoid blocking GPU
            finished_cpu = finished.cpu().numpy()
            
            for i in range(N):
                if not finished_cpu[i]:
                    # RGB -> BGR for cv2
                    img_bgr = rgb_cpu[i][..., ::-1]
                    all_images[i].append(img_bgr)
        
        t3 = time.time()
        t_cpu_sync += (t3 - t2)

        # This is ready to be fed into an RL policy (e.g. after permuting to NCHW).
        # Since we are not training here, we just hold the reference.
        
        # 4. Physics / State Update
        
        # Check if steps exceeded
        over_steps = current_step >= episode_lengths
        finished = finished | over_steps
        
        # If all finished, break
        if torch.all(finished):
            break
            
        # Get actions for active agents
        if agent is not None:
            # Agent-driven actions
            current_actions = torch.zeros(N, device=gs.device, dtype=torch.long)
            
            # Since we forced N=1, we check index 0
            if not finished[0]:
                 ep = episodes[0]
                 
                 # Instruction extraction
                 instruction_text = ep['instruction_text']
                 
                 # Print Step Information
                 print(f"\n[Step {step_idx}] Agent Input (Instruction): {instruction_text}")

                 obs = {
                     "rgb": rgb_cpu[0],
                     "instruction": {"text": instruction_text}
                 }

                 # --- DEBUG: Save Input Image ---
                 debug_img_path = f"debug_agent_input_step_{step_idx}.png"
                 # Convert RGB to BGR for OpenCV
                 cv2.imwrite(debug_img_path, rgb_cpu[0][..., ::-1])
                 print(f"[Debug] Saved agent input image to {debug_img_path}")
                 print(f"[Debug] Image Stats - Mean: {np.mean(rgb_cpu[0])}, Max: {np.max(rgb_cpu[0])}, Min: {np.min(rgb_cpu[0])}")
                 # -------------------------------

                 # Info can include 'top_down_map_vlnce' if needed, but we disabled map saving
                 info = {}
                 
                 try:
                     action_dict = agent.act(obs, info, ep['episode_id'])
                     print(f"[Step {step_idx}] Agent Output (Action Dict): {action_dict}")
                     
                     if action_dict and 'action' in action_dict:
                         current_actions[0] = int(action_dict['action'])
                         print(f"[Step {step_idx}] Executing Agent Action: {ACTION_MAP.get(int(action_dict['action']), 'UNKNOWN')}")
                     else:
                         current_actions[0] = 0
                         print(f"[Step {step_idx}] No valid action, executing STOP.")
                 except Exception as e:
                     print(f"Agent inference failed: {e}")
                     current_actions[0] = 0
        else:
            # Mask out finished agents to STOP (0)
            # Ensure we don't go out of bounds if current_step is large (though with GT we shouldn't)
            safe_step = torch.clamp(current_step, max=actions_padded.shape[1]-1)
            current_actions = actions_padded[agent_indices, safe_step]
            current_actions = torch.where(finished, torch.zeros_like(current_actions), current_actions)
        
        # Update step counters for active
        current_step = torch.where(~finished, current_step + 1, current_step)
        
        # Check explicit STOP action
        is_stop = current_actions == 0
        finished = finished | is_stop
        
        # Calculate intended movement
        # 1: forward, 2: left, 3: right
        is_forward = current_actions == 1
        is_left = current_actions == 2
        is_right = current_actions == 3
        
        # Update Yaw
        cam_yaw = torch.where(is_left, cam_yaw + STEP_TURN, cam_yaw)
        cam_yaw = torch.where(is_right, cam_yaw - STEP_TURN, cam_yaw)
        
        # Forward movement
        # desired_pos = current + dir * step
        dir_x = torch.cos(cam_yaw)
        dir_y = torch.sin(cam_yaw)
        
        move_vec_x = dir_x * STEP_MOVE
        move_vec_y = dir_y * STEP_MOVE
        
        desired_pos_x = cam_pos[:, 0] + torch.where(is_forward, move_vec_x, torch.zeros_like(move_vec_x))
        desired_pos_y = cam_pos[:, 1] + torch.where(is_forward, move_vec_y, torch.zeros_like(move_vec_y))
        
        desired_pos = torch.stack([desired_pos_x, desired_pos_y], dim=1) # (N, 2)
        
        # Check validity (Only for those moving forward)
        # We can check for all, doesn't hurt.
        current_floor_z = cam_pos[:, 2] - CAMERA_HEIGHT
        
        valid_mask, new_floor_z, new_tri_idx = batch_find_floor(
            desired_pos, current_floor_z,
            tri_v0_2d, tri_v1_2d, tri_v2_2d, 
            tri_v0_3d, tri_v1_3d, tri_v2_3d
        )
        
        # Sliding Logic
        if ALLOW_SLIDING:
            needs_slide = (~valid_mask) & is_forward
            if torch.any(needs_slide):
                slide_pos = batch_get_sliding_position(
                    cam_pos[:, :2], desired_pos, current_tri_idx,
                    tri_v0_2d, tri_v1_2d, tri_v2_2d
                )
                
                # Check slide validity
                slide_valid, slide_z, slide_tri_idx = batch_find_floor(
                    slide_pos, current_floor_z,
                    tri_v0_2d, tri_v1_2d, tri_v2_2d, 
                    tri_v0_3d, tri_v1_3d, tri_v2_3d
                )
                
                # Update desired_pos/z/tri_idx where sliding worked
                update_slide = needs_slide & slide_valid
                
                desired_pos = torch.where(update_slide.unsqueeze(1), slide_pos, desired_pos)
                new_floor_z = torch.where(update_slide, slide_z, new_floor_z)
                new_tri_idx = torch.where(update_slide, slide_tri_idx, new_tri_idx)
                valid_mask = valid_mask | update_slide

        # Apply updates
        should_move = is_forward & valid_mask
        
        cam_pos[:, 0] = torch.where(should_move, desired_pos[:, 0], cam_pos[:, 0])
        cam_pos[:, 1] = torch.where(should_move, desired_pos[:, 1], cam_pos[:, 1])
        cam_pos[:, 2] = torch.where(should_move, new_floor_z + CAMERA_HEIGHT, cam_pos[:, 2])
        
        # Update current triangle index
        current_tri_idx = torch.where(should_move, new_tri_idx, current_tri_idx)
        
        # Update Cameras in Genesis
        target_lookat = cam_pos.clone()
        target_lookat[:, 0] += dir_x
        target_lookat[:, 1] += dir_y
        target_lookat[:, 2] = cam_pos[:, 2]
        
        # Directly pass GPU tensors to set_pose (Batched)
        # No need to sync to CPU
        cam.set_pose(pos=cam_pos, lookat=target_lookat)

        torch.cuda.synchronize()
        t4 = time.time()
        t_physics += (t4 - t3)

        # --- Record Metrics ---
        curr_pos_hab = get_habitat_coords(cam_pos)
        pred_paths_gpu.append(curr_pos_hab.clone())
        curr_dist = torch.norm(curr_pos_hab - goal_pos_tensor, dim=1)
        distances_gpu.append(curr_dist.clone())

        step_idx += 1
        active_count = (~finished).sum().item() # This item() call is a small sync, needed for logic
        total_frames += active_count
        
        if step_idx % 10 == 0:
            print(f"Step {step_idx}, Active Agents: {active_count}")

    end_time = time.time()
    total_time = end_time - start_time
    total_fps = total_frames / total_time if total_time > 0 else 0
    avg_env_fps = total_fps / N if N > 0 else 0

    print(f"\n--- Simulation Statistics ---")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Frames Rendered: {total_frames}")
    print(f"Aggregate FPS (Total Throughput): {total_fps:.2f}")
    print(f"Average FPS per Environment: {avg_env_fps:.2f}")
    
    print(f"\n--- Performance Breakdown ---")
    print(f"Rendering: {t_render:.2f}s ({(t_render/total_time)*100:.1f}%)")
    print(f"Image Processing: {t_img_proc:.2f}s ({(t_img_proc/total_time)*100:.1f}%)")
    print(f"CPU Sync/Storage: {t_cpu_sync:.2f}s ({(t_cpu_sync/total_time)*100:.1f}%)")
    print(f"Physics/Nav/Logic: {t_physics:.2f}s ({(t_physics/total_time)*100:.1f}%)")
    
    print(f"-----------------------------\n")

    # --- Metrics Calculation ---
    print("Calculating metrics...")
    
    # Process collected data
    # pred_paths_gpu: list of T tensors of shape (N, 3) -> (T, N, 3) -> (N, T, 3)
    pred_paths_tensor = torch.stack(pred_paths_gpu, dim=1)
    pred_paths_cpu = pred_paths_tensor.cpu().numpy()
    
    distances_tensor = torch.stack(distances_gpu, dim=1)
    distances_cpu = distances_tensor.cpu().numpy()
    
    # We also need steps_taken for each agent
    steps_taken_cpu = current_step.cpu().numpy()
    
    metrics_all = {}
    
    for i in range(N):
        ep_id = episodes[i]['episode_id']
        gt_path = np.array(gt_paths_list[i]).astype(float)
        
        # Slicing: steps_taken[i] is the number of actions.
        # pred_paths includes initial pos (t=0) + positions after each step.
        num_steps = steps_taken_cpu[i]
        
        pred_path = pred_paths_cpu[i][:num_steps+1]
        distances = distances_cpu[i][:num_steps+1]
        
        # gt_length is start-to-goal Euclidean distance (distances[0])
        gt_length = float(distances[0])
        
        # DTW
        dtw_distance = float('inf')
        if 'fastdtw' in globals():
            try:
                dtw_distance = fastdtw(pred_path, gt_path, dist=euclidean)[0]
            except Exception as e:
                print(f"DTW calculation failed for {ep_id}: {e}")

        metric = {}
        metric['steps_taken'] = int(num_steps)
        metric['distance_to_goal'] = float(distances[-1])
        metric['success'] = 1.0 if distances[-1] <= 3.0 else 0.0 
        metric['oracle_success'] = 1.0 if (distances <= 3.0).any() else 0.0
        
        # Path length (Euclidean path length of agent)
        diffs = pred_path[1:] - pred_path[:-1]
        path_length = float(np.linalg.norm(diffs, axis=1).sum())
        metric['path_length'] = path_length
        
        metric['spl'] = float(metric['success'] * gt_length / max(gt_length, path_length)) if max(gt_length, path_length) > 0 else 0.0
        
        # NDTW
        if dtw_distance != float('inf'):
            metric['ndtw'] = float(np.exp(-dtw_distance / (len(gt_path) * 3.0)))
        else:
            metric['ndtw'] = 0.0
            
        metric['sdtw'] = float(metric['ndtw'] * metric['success'])
        
        metrics_all[ep_id] = metric
        
    # Save to JSON
    output_metrics_file = f"metrics_{SCAN_NAME}.json"
    print(f"Saving metrics to {output_metrics_file}...")
    with open(output_metrics_file, 'w') as f:
        json.dump(metrics_all, f, indent=4)
        
    # Calculate averages
    if len(metrics_all) > 0:
        avg_metrics = {}
        # Assumes all entries have the same keys
        keys = next(iter(metrics_all.values())).keys()
        for k in keys:
            avg_metrics[k] = float(np.mean([m[k] for m in metrics_all.values()]))
            
        import pprint
        pprint.pprint(avg_metrics)

    if (SAVE_IMAGES or SAVE_FAILED_EPISODES):
        print("Saving images to disk...")
        save_start_time = time.time()
        
        # Create output root
        os.makedirs("output", exist_ok=True)
        
        saved_count = 0
        for i in range(N):
            episode_id = episodes[i]['episode_id']
            
            # Check if we should save
            should_save = False
            if SAVE_IMAGES:
                should_save = True
            elif SAVE_FAILED_EPISODES:
                # Check metrics
                if episode_id in metrics_all:
                     # If success is 0.0, it failed
                     if metrics_all[episode_id]['success'] == 0.0:
                         should_save = True
            
            if should_save:
                save_dir = f"output/episode_{episode_id}"
                
                # Remove existing dir if any
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.makedirs(save_dir)
                
                # Save all images for this episode
                imgs = all_images[i]
                for step, img in enumerate(imgs):
                    path = os.path.join(save_dir, f"step_{step:04d}.png")
                    cv2.imwrite(path, img)
                saved_count += 1
                
        save_end_time = time.time()
        print(f"Images saved for {saved_count} episodes in {save_end_time - save_start_time:.2f}s")

    print("All episodes finished.")

if __name__ == "__main__":
    try:
        run_parallel_episodes(num_envs=NUM_PARALLEL_ENVS)
    except Exception as e:
        import traceback
        traceback.print_exc()
