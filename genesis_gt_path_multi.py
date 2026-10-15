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
import argparse
import subprocess
import gzip
import cv2

from genark_web_viewer import WebBroadcaster


def _annotate_frame(rgb: np.ndarray, step: int, action_str: str,
                    instruction: str) -> np.ndarray:
    """Overlay step/action/instruction on an RGB frame; returns BGR for cv2.imwrite."""
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    line1 = f"step={step}  action={action_str}  [GT]"
    cv2.putText(frame, line1, (6, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    words = instruction.split()
    lines, cur = [], ""
    for w_word in words:
        test = (cur + " " + w_word).strip()
        if cv2.getTextSize(test, font, 0.45, 1)[0][0] > frame.shape[1] - 12:
            lines.append(cur); cur = w_word
        else:
            cur = test
    if cur:
        lines.append(cur)
    for i, ln in enumerate(lines[:2]):
        cv2.putText(frame, ln, (6, 42 + i * 16), font, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
    return frame


def _compile_mp4(frames_dir: str, output_path: str, fps: int = 5) -> None:
    ffmpeg_bin = "/usr/bin/ffmpeg"
    if not os.path.isfile(ffmpeg_bin):
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    pattern = os.path.join(frames_dir, "frame_%06d.jpg")
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-framerate", str(fps), "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_path,
    ]
    subprocess.run(cmd, check=True, timeout=300)

# --- File Paths (CHANGE THESE TO YOUR ACTUAL FILE PATHS) ---
EPISODES_FILE = "/home/clk/genark_ws/genark/data/datasets/zsNo4HB9uLZ_episodes.json"
# Default GT pools (searched in order) for files that don't have a sibling *_gt.
R2R_PREPROCESSED_DIR = "/home/clk/genark_ws/genark/data/datasets/R2R_VLNCE_v1-3_preprocessed"
DEFAULT_GT_CANDIDATES = [
    f"{R2R_PREPROCESSED_DIR}/val_unseen/val_unseen_gt.json.gz",
    f"{R2R_PREPROCESSED_DIR}/val_seen/val_seen_gt.json.gz",
    f"{R2R_PREPROCESSED_DIR}/train/train_gt.json.gz",
]
SCENE_DATASETS = "/home/clk/genark_ws/genark/scene_datasets"

# --- Constants & Configuration ---
NUM_PARALLEL_ENVS = 256 
CAMERA_HEIGHT = 0.88
CAM_RES = (640, 480)
FOV = 64
LIGHT_SCALE = 4.0
SAVE_IMAGES = False
WEB_VIEWER = True
WEB_PORT = 9998
WEB_ENV_IDX = 0

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

def _open_json(path):
    """Open .json or .json.gz and return parsed content."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    with open(path, 'r') as f:
        return json.load(f)


def load_data(num_envs, episodes_file=None, episode_ids=None, scene=None,
              gt_file=None):
    """Load episodes and merge GT action sequences.

    `episodes_file` is a guide-style JSON (plain or .gz) with an `episodes`
    list. Each episode is merged with GT actions from `gt_file` (a dict keyed
    by episode_id with `actions`/`locations`/`forward_steps`). If `gt_file` is
    None, a sibling `*_gt.json(.gz)` is searched first, then the defaults in
    `DEFAULT_GT_CANDIDATES` are tried until all episode ids resolve.

    The simulator builds one scene per run, so episodes are filtered down to a
    single scene (either the one passed via `scene`, or the scene with the
    most episodes in the file).
    """
    path = episodes_file or EPISODES_FILE
    print(f"Loading data from {path}...")
    data = _open_json(path)
    all_episodes = data['episodes']
    print(f"Found {len(all_episodes)} episodes in file.")

    # Locate GT source(s)
    gt_paths = []
    if gt_file is not None:
        gt_paths = [gt_file]
    else:
        for suf in ('.json.gz', '.json'):
            if path.endswith(suf):
                cand = path[:-len(suf)] + '_gt' + suf
                if os.path.isfile(cand):
                    gt_paths.append(cand)
                break
        gt_paths.extend(p for p in DEFAULT_GT_CANDIDATES if os.path.isfile(p))
    if not gt_paths:
        raise FileNotFoundError("No GT file found. Pass --gt_file explicitly.")

    # Merge GT, also pulling start_position/start_rotation from the matching
    # guide episode (sibling of the *_gt file). The GT action sequence was
    # computed from the guide's starting pose, so we must use that pose here.
    # Match guide episodes by (scene_id, trajectory_id) first, falling back to
    # episode_id — this handles datasets like OpenNav that keep trajectory_id
    # stable but renumber episode_ids (or perturb start poses).
    merged_by_local_key = {}  # keyed by id(ep) for stable dedup
    for gp in gt_paths:
        print(f"Loading GT actions from {gp}...")
        gt = _open_json(gp)

        # Locate sibling guide file
        guide_path = None
        for suf in ('.json.gz', '.json'):
            if gp.endswith('_gt' + suf):
                cand = gp[:-len('_gt' + suf)] + suf
                if os.path.isfile(cand):
                    guide_path = cand
                    break
        guide_by_traj = {}
        guide_by_id = {}
        if guide_path is not None:
            print(f"  (pulling guide poses from {guide_path})")
            gdata = _open_json(guide_path)
            for gep in gdata['episodes']:
                guide_by_id[str(gep['episode_id'])] = gep
                tk = (gep['scene_id'], gep.get('trajectory_id'))
                if tk[1] is not None:
                    guide_by_traj[tk] = gep

        for ep in all_episodes:
            if id(ep) in merged_by_local_key:
                continue
            # Prefer (scene, trajectory_id) match; fall back to episode_id
            tk = (ep['scene_id'], ep.get('trajectory_id'))
            gep = guide_by_traj.get(tk) if tk[1] is not None else None
            if gep is None:
                gep = guide_by_id.get(str(ep['episode_id']))
            gt_key = str(gep['episode_id']) if gep is not None else str(ep['episode_id'])
            if gt_key not in gt:
                continue
            ep_copy = dict(ep)
            ep_copy['actions'] = gt[gt_key]['actions']
            if gep is not None:
                ep_copy['start_position'] = gep['start_position']
                ep_copy['start_rotation'] = gep['start_rotation']
            merged_by_local_key[id(ep)] = ep_copy
        if len(merged_by_local_key) == len(all_episodes):
            break

    missing = len(all_episodes) - len(merged_by_local_key)
    if missing:
        print(f"Warning: {missing} episodes had no GT entry and were dropped.")
    all_episodes = [merged_by_local_key[id(ep)]
                    for ep in all_episodes if id(ep) in merged_by_local_key]

    # Filter to a single scene (simulator loads one mesh per run)
    scenes = {}
    for ep in all_episodes:
        scenes.setdefault(ep['scene_id'], []).append(ep)
    if scene is not None:
        chosen = next((s for s in scenes if scene in s), None)
        if chosen is None:
            raise ValueError(f"Scene '{scene}' not found. Available: "
                             f"{sorted(os.path.basename(os.path.dirname(s)) for s in scenes)}")
    else:
        chosen = max(scenes, key=lambda s: len(scenes[s]))
    all_episodes = scenes[chosen]
    print(f"Filtered to scene '{chosen}': {len(all_episodes)} episodes.")

    if episode_ids is not None:
        wanted = set(int(x) for x in episode_ids)
        filtered = [ep for ep in all_episodes if int(ep['episode_id']) in wanted]
        print(f"Filtered to {len(filtered)} episodes by episode_ids.")
        return filtered

    if num_envs is None or num_envs <= 0 or num_envs >= len(all_episodes):
        print(f"Returning all {len(all_episodes)} episodes.")
        return all_episodes

    sampled = random.choices(all_episodes, k=num_envs)
    print(f"Sampled {len(sampled)} episodes for parallel execution.")
    return sampled

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

def run_parallel_episodes(num_envs: int, episodes_file: str | None = None,
                          episode_ids: list | None = None,
                          video_output_dir: str | None = None,
                          video_fps: int = 5,
                          enable_web: bool = True,
                          scene: str | None = None,
                          gt_file: str | None = None):
    episodes = load_data(num_envs, episodes_file=episodes_file,
                         episode_ids=episode_ids, scene=scene,
                         gt_file=gt_file)
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
    glb_path = f"/home/clk/genark_ws/genark/glb_cache/{SCAN_NAME}.glb"
    if not os.path.isfile(glb_path):
        print(f"Error: .glb file not found at '{glb_path}'")
        return
    scene.add_entity(morph=gs.morphs.Mesh(
        file=glb_path, fixed=True, collision=False,
        euler=(-90.0, 0.0, 0.0),  # Habitat Y-up -> Genesis Z-up
    ))

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
    scene.build(n_envs=len(episodes), env_spacing=(0.0, 0.0))

    # Per-episode MP4 recorder (one MP4 per agent)
    record_video = video_output_dir is not None
    ep_frame_dirs = [None] * len(episodes)
    ep_frame_counts = [0] * len(episodes)
    ep_compiled = [False] * len(episodes)
    instructions_text = [
        ep.get("instruction_text") or ep.get("instruction", {}).get("instruction_text", "")
        for ep in episodes
    ]
    if record_video:
        os.makedirs(video_output_dir, exist_ok=True)
        for i, ep in enumerate(episodes):
            d = os.path.join(video_output_dir, f"_tmp_{ep['episode_id']}")
            os.makedirs(d, exist_ok=True)
            ep_frame_dirs[i] = d
        print(f"[gt_path] per-episode MP4 recording → {video_output_dir} "
              f"({len(episodes)} episodes, fps={video_fps})")

    broadcaster = None
    if WEB_VIEWER and enable_web:
        broadcaster = WebBroadcaster(
            output_dir="genark_web_output_gt",
            port=WEB_PORT,
            quality=85,
            throttle_fps=30,
        )

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

    # Fixed world Z-up, broadcast to per-env. Re-supplied every set_pose call
    # to overwrite any stale up-vector left in the shared camera buffer.
    world_up = torch.tensor([0.0, 0.0, 1.0], device=gs.device, dtype=torch.float32)
    cam_up = world_up.unsqueeze(0).expand(N, 3).contiguous()

    # Initial Pose Update
    dir_x = torch.cos(cam_yaw)
    dir_y = torch.sin(cam_yaw)
    target_lookat = cam_pos.clone()
    target_lookat[:, 0] += dir_x
    target_lookat[:, 1] += dir_y
    target_lookat[:, 2] = cam_pos[:, 2] # Look horizontal

    # Update camera (Batched)
    cam.set_pose(pos=cam_pos, lookat=target_lookat, up=cam_up)

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
        
        if broadcaster is not None:
            active_count_web = int((~finished).sum().item())
            broadcaster.send_frame(rgb_batch, env_idx=WEB_ENV_IDX, metadata={
                "step": step_idx,
                "active": active_count_web,
                "total_envs": N,
            })

        # Per-episode MP4 frame capture (snapshot before action consumed this step)
        if record_video:
            rgb_cpu_all = rgb_batch.cpu().numpy()
            finished_cpu = finished.cpu().numpy()
            cur_step_cpu = current_step.cpu().numpy()
            ep_lens_cpu = episode_lengths.cpu().numpy()
            for i in range(N):
                if ep_compiled[i]:
                    continue
                if finished_cpu[i]:
                    continue
                # Look up the action this agent is about to execute (for overlay)
                k = int(cur_step_cpu[i])
                if k < ep_lens_cpu[i]:
                    a_int = int(actions_list[i][k])
                else:
                    a_int = 0
                act_str = ACTION_MAP.get(a_int, "?")
                bgr = _annotate_frame(rgb_cpu_all[i], step_idx, act_str,
                                      instructions_text[i])
                fp = os.path.join(ep_frame_dirs[i],
                                  f"frame_{ep_frame_counts[i]:06d}.jpg")
                cv2.imwrite(fp, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                ep_frame_counts[i] += 1

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
        
        # Directly pass GPU tensors to set_pose (Batched). Re-supply the
        # fixed Z-up every frame so the shared camera buffer cannot carry
        # stale up-vectors across steps.
        cam.set_pose(pos=cam_pos, lookat=target_lookat, up=cam_up)

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

    if broadcaster is not None:
        broadcaster.finish()

    # Compile per-episode MP4s
    if record_video:
        for i in range(N):
            if ep_compiled[i] or ep_frame_counts[i] == 0:
                continue
            ep_id = episodes[i]['episode_id']
            mp4 = os.path.join(video_output_dir, f"episode_{ep_id}.mp4")
            try:
                _compile_mp4(ep_frame_dirs[i], mp4, video_fps)
                print(f"  saved {mp4} ({ep_frame_counts[i]} frames)")
            except Exception as e:
                print(f"  ffmpeg failed for ep {ep_id}: {e}")
            finally:
                shutil.rmtree(ep_frame_dirs[i], ignore_errors=True)
                ep_compiled[i] = True

    print("All episodes finished.")

    if broadcaster is not None and enable_web:
        broadcaster.wait_for_exit()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes_file", type=str, default=None,
                   help="Episode JSON (must have 'actions' field). Default: hardcoded EPISODES_FILE")
    p.add_argument("--episode_ids", type=int, nargs="+", default=None,
                   help="If set, only run episodes whose episode_id is in this list")
    p.add_argument("--num_envs", type=int, default=NUM_PARALLEL_ENVS,
                   help="Used only when --episode_ids is not given")
    p.add_argument("--video_output_dir", type=str, default=None,
                   help="If set, save one MP4 per episode (episode_<id>.mp4)")
    p.add_argument("--video_fps", type=int, default=5)
    p.add_argument("--no_web", action="store_true",
                   help="Disable WebBroadcaster (no HTTP viewer, no wait_for_exit)")
    p.add_argument("--gt_file", type=str, default=None,
                   help="Explicit GT json(.gz) providing actions per episode_id. "
                        "If unset, tries sibling *_gt and R2R preprocessed defaults.")
    p.add_argument("--scene", type=str, default=None,
                   help="Substring match against scene_id (e.g. 'zsNo4HB9uLZ'). "
                        "If unset, the scene with the most episodes is chosen.")
    args = p.parse_args()
    try:
        run_parallel_episodes(
            num_envs=args.num_envs,
            episodes_file=args.episodes_file,
            episode_ids=args.episode_ids,
            video_output_dir=args.video_output_dir,
            video_fps=args.video_fps,
            enable_web=not args.no_web,
            scene=args.scene,
            gt_file=args.gt_file,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
