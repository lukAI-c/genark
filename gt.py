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

# --- File Paths (CHANGE THESE TO YOUR ACTUAL FILE PATHS) ---
EPISODES_FILE = "/home/dinghongyu/code/Genesis/scripts/scenes_merged/mp3d_zsNo4HB9uLZ.json"
SCENE_DATASETS = "/home/dinghongyu/data/lavira_data/scene_datasets/obj_files"

# --- Constants & Configuration ---
NUM_PARALLEL_ENVS = 1024 
CAMERA_HEIGHT = 0.88
CAM_RES = (640, 480)
FOV = 64
LIGHT_SCALE = 4.0
SAVE_IMAGES = False

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
    
    # Calculate slide
    # proj_len = dot(move_vec, edge_dir)
    proj_len = (move_vec * edge_dirs).sum(dim=-1, keepdim=True) # (N, 3, 1)
    slide_vecs = edge_dirs * proj_len # (N, 3, 2)
    candidates = p_current + slide_vecs # (N, 3, 2)
    
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
    
    # Prepare data for GPU tensors
    start_pos_list = []
    start_rot_list = []
    actions_list = []
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
    all_images = [[] for _ in range(N)] if SAVE_IMAGES else None
    
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
        
        if SAVE_IMAGES:
            # Move to CPU memory and store in list
            # We do NOT save to disk here to avoid blocking GPU
            rgb_cpu = rgb_batch.cpu().numpy()
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
        # Mask out finished agents to STOP (0)
        current_actions = actions_padded[agent_indices, current_step]
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

    if SAVE_IMAGES:
        print("Saving images to disk...")
        save_start_time = time.time()
        
        # Create output root
        os.makedirs("output", exist_ok=True)
        
        for i in range(N):
            episode_id = episodes[i]['episode_id']
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
                
        save_end_time = time.time()
        print(f"Images saved in {save_end_time - save_start_time:.2f}s")

    print("All episodes finished.")

if __name__ == "__main__":
    try:
        run_parallel_episodes(num_envs=NUM_PARALLEL_ENVS)
    except Exception as e:
        import traceback
        traceback.print_exc()
