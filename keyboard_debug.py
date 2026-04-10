#!/usr/bin/env python3
"""
keyboard_debug.py — Interactive Genesis scene walkthrough via Web WASD controls.

Allows real-time keyboard navigation of a Genesis scene to debug initial
positions and NavMesh behavior.

Usage:
    CUDA_VISIBLE_DEVICES=3 python keyboard_debug.py \
        --scene_id 2azQ1b91cZZ \
        --episode_idx 0 \
        --port 9997

Then open browser at http://<server>:9997/
"""

import sys
import os
import json
import argparse
import time
import math
import threading
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, "/data1/home/clk/project/Genesis")

import numpy as np
import torch
import cv2

# Add genark to path
sys.path.insert(0, os.path.dirname(__file__))
from genark_config import load_config, sim_constants

import genesis as gs

# Config
CONFIG = load_config("configs/uninavid_r2r.yaml")
SCENE_DATASETS = CONFIG.get("SCENE_DATASETS", "/data1/home/clk/project/genark/scene_datasets")
CAMERA_HEIGHT = CONFIG["SIMULATOR"]["CAMERA_HEIGHT"]
FORWARD_STEP_SIZE = CONFIG["SIMULATOR"]["FORWARD_STEP_SIZE"]
TURN_ANGLE = CONFIG["SIMULATOR"]["TURN_ANGLE"]
AGENT_RADIUS = CONFIG["SIMULATOR"]["AGENT_RADIUS"]
MAX_STEP_HEIGHT = CONFIG["SIMULATOR"].get("MAX_STEP_HEIGHT", 0.5)
LIGHT_SCALE = CONFIG["SIMULATOR"].get("LIGHT_SCALE", 4.0)
CAM_RES = (CONFIG["SIMULATOR"]["RGB_SENSOR"]["WIDTH"],
           CONFIG["SIMULATOR"]["RGB_SENSOR"]["HEIGHT"])
HFOV = CONFIG["SIMULATOR"]["RGB_SENSOR"].get("HFOV", 105)

# Episodes file
EPISODES_FILE = CONFIG.get("EPISODES_FILE",
    "/data1/home/clk/navid_ws/data/datasets/OpenNav_R2R-CE_100_bertidx.json")

# ============================================================================
# NumPy/Tensor helpers (from env_worker.py)
# ============================================================================

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
    """Find walkable floor triangle for each agent position."""
    pt    = pos_2d.unsqueeze(1)
    v0_2  = tri_v0_2d.unsqueeze(0); v1_2 = tri_v1_2d.unsqueeze(0); v2_2 = tri_v2_2d.unsqueeze(0)
    inside = batch_is_point_in_triangle(pt, v0_2, v1_2, v2_2)

    v0_3  = tri_v0_3d.unsqueeze(0); v1_3 = tri_v1_3d.unsqueeze(0); v2_3 = tri_v2_3d.unsqueeze(0)
    z_vals = batch_get_z_on_triangle(v0_3, v1_3, v2_3, pt[..., 0], pt[..., 1])

    z_diff     = torch.abs(z_vals - current_floor_z.unsqueeze(1))
    valid_cand = inside & (z_diff < MAX_STEP_HEIGHT)
    z_diff_m   = torch.where(valid_cand, z_diff, torch.full_like(z_diff, float("inf")))

    min_dist, best_idx = torch.min(z_diff_m, dim=1)
    valid_mask = min_dist != float("inf")
    best_z     = torch.gather(z_vals, 1, best_idx.unsqueeze(1)).squeeze(1)
    return valid_mask, best_z, best_idx


def batch_get_sliding_position(current_pos, desired_pos, current_tri_idx,
                               tri_v0_2d, tri_v1_2d, tri_v2_2d,
                               agent_radius: float = 0.0):
    """Slide position against triangle edges, inset by agent_radius."""
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
    sq_safe = torch.where(sq < 1e-6, torch.ones_like(sq), sq)
    t    = torch.clamp(dot / sq_safe, 0.0, 1.0)
    candidates = edge_starts + t.unsqueeze(-1) * edge_vecs

    if agent_radius > 0.0:
        centroid = ((p0 + p1 + p2) / 3.0).unsqueeze(1)
        perp_a = torch.stack([-edge_vecs[..., 1], edge_vecs[..., 0]], dim=-1)
        perp_b = -perp_a
        to_ctr = centroid - edge_starts
        dot_a  = (perp_a * to_ctr).sum(dim=-1)
        inward = torch.where(dot_a.unsqueeze(-1) > 0, perp_a, perp_b)
        inward_len = torch.norm(inward, dim=-1, keepdim=True).clamp_min(1e-6)
        inward_unit = inward / inward_len
        candidates = candidates + inward_unit * agent_radius

    has_crossing, first_idx = torch.max(crossing.long(), dim=1)
    selected = torch.gather(candidates, 1,
                            first_idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    return torch.where(has_crossing.view(-1, 1).bool(), selected, current_pos)


def _hab_to_genesis(pos_list):
    """Convert Habitat (X, Y, Z) to Genesis (X, Y, Z)."""
    x, y, z = pos_list[0], pos_list[1], pos_list[2]
    return x, -z, y


def _update_camera(cam, cam_pos: torch.Tensor, cam_yaw: torch.Tensor):
    """Update Genesis camera pose."""
    dir_x  = torch.cos(cam_yaw)
    dir_y  = torch.sin(cam_yaw)
    lookat = cam_pos.clone()
    lookat[:, 0] += dir_x
    lookat[:, 1] += dir_y
    lookat[:, 2]  = cam_pos[:, 2]
    N = cam_pos.shape[0]
    up = torch.zeros_like(cam_pos)
    up[:, 2] = 1.0
    cam.set_pose(pos=cam_pos, lookat=lookat, up=up)


def calculate_initial_yaw_tensor(rotations):
    """Convert Habitat start_rotation quaternion [x,y,z,w] → Genesis yaw."""
    x, y, z, w = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    dir_x_g =  -2 * w * y
    dir_y_g = -(y * y - w * w)
    return torch.atan2(dir_y_g, dir_x_g)


# ============================================================================
# HTTP Server with shared state + action queue
# ============================================================================

class DebugHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for debug web interface."""

    shared_state = {
        'frame_bytes': b'',
        'pos': [0.0, 0.0, 0.0],
        'yaw': 0.0,
        'on_mesh': True,
        'tri_idx': 0,
        'scene': 'unknown',
        'episode': 0,
        'total_episodes': 1,
    }
    action_queue = queue.Queue()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif self.path == '/frame':
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.end_headers()
            self.wfile.write(self.shared_state.get('frame_bytes', b''))

        elif self.path == '/state':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            state = {
                'pos': self.shared_state['pos'],
                'yaw': self.shared_state['yaw'],
                'on_mesh': self.shared_state['on_mesh'],
                'tri_idx': self.shared_state['tri_idx'],
                'scene': self.shared_state['scene'],
                'episode': self.shared_state['episode'],
                'total_episodes': self.shared_state['total_episodes'],
            }
            self.wfile.write(json.dumps(state).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/action':
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            try:
                data = json.loads(body.decode('utf-8'))
                action = data.get('action', 'none')
                self.action_queue.put(action)
            except:
                pass

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()


# ============================================================================
# HTML UI
# ============================================================================

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genesis Scene Debug — Keyboard Navigation</title>
    <style>
        body {
            font-family: monospace;
            margin: 0;
            padding: 10px;
            background: #1a1a1a;
            color: #00ff00;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .viewer {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        #feed {
            max-width: 100%;
            max-height: 80vh;
            border: 2px solid #00ff00;
            background: black;
        }
        .info {
            background: #222;
            padding: 15px;
            border: 1px solid #00ff00;
            border-radius: 5px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 80vh;
            overflow-y: auto;
        }
        .info h2 {
            margin-top: 0;
            text-decoration: underline;
        }
        .ok { color: #00ff00; }
        .warn { color: #ffff00; }
        .error { color: #ff0000; }
        .controls {
            margin-top: 10px;
            padding: 10px;
            background: #222;
            border: 1px solid #00ff00;
            border-radius: 5px;
        }
        .controls p {
            margin: 5px 0;
        }
        #fps {
            color: #888;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="viewer">
            <img id="feed" src="/frame" alt="Loading...">
            <div class="info" id="controls">
                <h2>⌨️ Controls</h2>
                <p><b>W</b> — Move forward</p>
                <p><b>S</b> — Move backward</p>
                <p><b>A</b> — Turn left (30°)</p>
                <p><b>D</b> — Turn right (30°)</p>
                <p><b>R</b> — Reset to start</p>
                <p id="fps"></p>
            </div>
        </div>
        <div>
            <div class="info" id="status">
                <h2>📊 Status</h2>
                <p>Connecting...</p>
            </div>
        </div>
    </div>

    <script>
        const feed = document.getElementById('feed');
        const statusDiv = document.getElementById('status');
        const fpsDiv = document.getElementById('fps');

        let lastFrameTime = Date.now();
        let frameCount = 0;

        // Live frame updates
        setInterval(() => {
            feed.src = '/frame?' + Date.now();
            frameCount++;
            const now = Date.now();
            if (now - lastFrameTime >= 1000) {
                const fps = frameCount * 1000 / (now - lastFrameTime);
                fpsDiv.textContent = `FPS: ${fps.toFixed(1)}`;
                frameCount = 0;
                lastFrameTime = now;
            }
        }, 150);

        // Status updates
        setInterval(async () => {
            try {
                const resp = await fetch('/state');
                const state = await resp.json();

                const onMesh = state.on_mesh ? '<span class="ok">✓ ON_MESH</span>' :
                               '<span class="error">✗ OFF_MESH</span>';

                statusDiv.innerHTML = `<h2>📊 Status</h2>
<p><b>Scene:</b> ${state.scene}</p>
<p><b>Episode:</b> ${state.episode}/${state.total_episodes}</p>
<hr>
<p><b>Position:</b></p>
<p>  X = ${state.pos[0].toFixed(2)} m</p>
<p>  Y = ${state.pos[1].toFixed(2)} m</p>
<p>  Z = ${state.pos[2].toFixed(2)} m</p>
<p><b>Yaw:</b> ${(state.yaw * 180 / Math.PI).toFixed(1)}°</p>
<p><b>Mesh:</b> ${onMesh}</p>
<p><b>Triangle:</b> ${state.tri_idx}</p>`;
            } catch (e) {
                statusDiv.innerHTML = '<h2>📊 Status</h2><p class="error">Error connecting...</p>';
            }
        }, 500);

        // Keyboard handler
        document.addEventListener('keydown', async (e) => {
            let action = null;

            if (e.key.toLowerCase() === 'w') action = 'forward';
            else if (e.key.toLowerCase() === 's') action = 'backward';
            else if (e.key.toLowerCase() === 'a') action = 'turn_left';
            else if (e.key.toLowerCase() === 'd') action = 'turn_right';
            else if (e.key.toLowerCase() === 'r') action = 'reset';

            if (action) {
                e.preventDefault();
                try {
                    await fetch('/action', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({action})
                    });
                } catch (err) {
                    console.error('Action error:', err);
                }
            }
        });
    </script>
</body>
</html>
"""


# ============================================================================
# Main simulation loop
# ============================================================================

def run_debug_scene(scene_id: str, episode_idx: int, port: int):
    """Run interactive debug walkthrough for one scene and episode."""

    # Parse scene ID
    SCAN_NAME = scene_id.strip()
    print(f"\n{'='*60}\nScene: {SCAN_NAME} | Episode: {episode_idx}\n{'='*60}")

    # Load episodes
    with open(EPISODES_FILE) as f:
        data = json.load(f)
        all_episodes = data['episodes'] if isinstance(data, dict) else data

    scene_episodes = [ep for ep in all_episodes if ep['scene_id'] == f"mp3d/{SCAN_NAME}/{SCAN_NAME}.glb"]
    if not scene_episodes:
        print(f"[ERROR] No episodes found for scene {SCAN_NAME}")
        return

    if episode_idx >= len(scene_episodes):
        print(f"[ERROR] Episode {episode_idx} not available (only {len(scene_episodes)} total)")
        return

    episode = scene_episodes[episode_idx]
    print(f"Instruction: {episode['instruction']['instruction_text']}")

    # ================================================================
    # Initialize Genesis
    # ================================================================
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

    # ================================================================
    # Load scene mesh
    # ================================================================
    obj_dir = f"{SCENE_DATASETS}/{SCAN_NAME}/matterport_mesh"
    if not os.path.exists(obj_dir):
        print(f"[ERROR] Scene directory not found: {obj_dir}")
        return

    obj_files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]
    if not obj_files:
        print(f"[ERROR] No .obj files found in {obj_dir}")
        return

    scene.add_entity(morph=gs.morphs.Mesh(
        file=os.path.join(obj_dir, obj_files[0]),
        fixed=True, collision=False
    ))

    # ================================================================
    # Load NavMesh
    # ================================================================
    navmesh_path = f"{SCENE_DATASETS}/{SCAN_NAME}/navmesh.npz"
    if not os.path.exists(navmesh_path):
        print(f"[ERROR] NavMesh not found: {navmesh_path}")
        return

    nm = np.load(navmesh_path)
    V_hab = torch.tensor(nm["verts"], device=gs.device, dtype=torch.float32)
    F_idx = torch.tensor(nm["faces"], device=gs.device, dtype=torch.long)

    V_gen = torch.zeros_like(V_hab)
    V_gen[:, 0] =  V_hab[:, 0]
    V_gen[:, 1] = -V_hab[:, 2]
    V_gen[:, 2] =  V_hab[:, 1]

    tri_v0_3d = V_gen[F_idx[:, 0]]; tri_v1_3d = V_gen[F_idx[:, 1]]; tri_v2_3d = V_gen[F_idx[:, 2]]
    tri_v0_2d = tri_v0_3d[:, :2];   tri_v1_2d = tri_v1_3d[:, :2];   tri_v2_2d = tri_v2_3d[:, :2]

    print(f"NavMesh: {len(F_idx)} triangles")

    # ================================================================
    # Build scene + camera
    # ================================================================
    cam = scene.add_camera(res=CAM_RES, fov=HFOV, GUI=False)
    scene.build(n_envs=1, env_spacing=(0.0, 0.0))

    # ================================================================
    # Initialize agent pose
    # ================================================================
    sp = episode["start_position"]
    gx, gy, gz = _hab_to_genesis(sp)
    cam_pos = torch.tensor([[gx, gy, gz + CAMERA_HEIGHT]], dtype=torch.float32, device=gs.device)
    cam_rot = torch.tensor([episode["start_rotation"]], dtype=torch.float32, device=gs.device)
    cam_yaw = calculate_initial_yaw_tensor(cam_rot)

    # Store initial for reset
    initial_cam_pos = cam_pos.clone()
    initial_cam_yaw = cam_yaw.clone()

    print(f"Initial pos: x={cam_pos[0,0]:.2f}, y={cam_pos[0,1]:.2f}, z={cam_pos[0,2]:.2f}")
    print(f"Initial yaw: {float(cam_yaw[0]) * 180 / math.pi:.1f}°")

    # Find initial floor and snap Z coordinate
    valid, z_floor, tri_idx = batch_find_floor(
        cam_pos[:, :2], cam_pos[:, 2] - CAMERA_HEIGHT,
        tri_v0_2d, tri_v1_2d, tri_v2_2d,
        tri_v0_3d, tri_v1_3d, tri_v2_3d,
    )
    current_tri_idx = tri_idx

    # CRITICAL: Update cam_pos Z based on floor snapping (like in env_worker.py movement)
    if valid[0]:
        cam_pos[0, 2] = z_floor[0] + CAMERA_HEIGHT
        print(f"✓ Initial position ON_MESH (tri {int(current_tri_idx[0])})")
        print(f"  Floor Z snapped from {gz:.2f} to {float(z_floor[0]):.2f}")
    else:
        print(f"✗ Initial position OFF_MESH — attempting sliding fallback...")
        # Try sliding if initial position is invalid
        slide_pos = batch_get_sliding_position(
            cam_pos[:, :2], cam_pos[:, :2], current_tri_idx,
            tri_v0_2d, tri_v1_2d, tri_v2_2d,
            agent_radius=AGENT_RADIUS,
        )
        valid_slide, z_slide, tri_slide = batch_find_floor(
            slide_pos, cam_pos[:, 2] - CAMERA_HEIGHT,
            tri_v0_2d, tri_v1_2d, tri_v2_2d,
            tri_v0_3d, tri_v1_3d, tri_v2_3d,
        )
        if valid_slide[0]:
            cam_pos[0, :2] = slide_pos[0]
            cam_pos[0, 2] = z_slide[0] + CAMERA_HEIGHT
            current_tri_idx = tri_slide
            print(f"  ✓ Sliding succeeded (tri {int(current_tri_idx[0])})")

    _update_camera(cam, cam_pos, cam_yaw)

    # ================================================================
    # Start HTTP server
    # ================================================================
    server = HTTPServer(('0.0.0.0', port), DebugHTTPHandler)
    print(f"\n[HTTP] Server running on http://0.0.0.0:{port}/")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # ================================================================
    # Main render loop
    # ================================================================
    print("[Render] Starting main loop (Ctrl+C to quit)...\n")

    try:
        while True:
            # Process actions
            try:
                action = DebugHTTPHandler.action_queue.get_nowait()
            except queue.Empty:
                action = None

            if action == 'forward':
                # Move forward
                desired_pos = cam_pos[:, :2] + torch.tensor(
                    [math.cos(float(cam_yaw[0])) * FORWARD_STEP_SIZE,
                     math.sin(float(cam_yaw[0])) * FORWARD_STEP_SIZE],
                    device=gs.device
                ).unsqueeze(0)

                valid, new_z, new_tri = batch_find_floor(
                    desired_pos, cam_pos[:, 2] - CAMERA_HEIGHT,
                    tri_v0_2d, tri_v1_2d, tri_v2_2d,
                    tri_v0_3d, tri_v1_3d, tri_v2_3d,
                )

                if valid[0]:
                    cam_pos[0, :2] = desired_pos[0]
                    cam_pos[0, 2] = new_z[0] + CAMERA_HEIGHT
                    current_tri_idx = new_tri
                else:
                    # Try sliding
                    slide_pos = batch_get_sliding_position(
                        cam_pos[:, :2], desired_pos, current_tri_idx,
                        tri_v0_2d, tri_v1_2d, tri_v2_2d,
                        agent_radius=AGENT_RADIUS,
                    )
                    valid_slide, slide_z, slide_tri = batch_find_floor(
                        slide_pos, cam_pos[:, 2] - CAMERA_HEIGHT,
                        tri_v0_2d, tri_v1_2d, tri_v2_2d,
                        tri_v0_3d, tri_v1_3d, tri_v2_3d,
                    )
                    if valid_slide[0]:
                        cam_pos[0, :2] = slide_pos[0]
                        cam_pos[0, 2] = slide_z[0] + CAMERA_HEIGHT
                        current_tri_idx = slide_tri

            elif action == 'backward':
                desired_pos = cam_pos[:, :2] - torch.tensor(
                    [math.cos(float(cam_yaw[0])) * FORWARD_STEP_SIZE,
                     math.sin(float(cam_yaw[0])) * FORWARD_STEP_SIZE],
                    device=gs.device
                ).unsqueeze(0)

                valid, new_z, new_tri = batch_find_floor(
                    desired_pos, cam_pos[:, 2] - CAMERA_HEIGHT,
                    tri_v0_2d, tri_v1_2d, tri_v2_2d,
                    tri_v0_3d, tri_v1_3d, tri_v2_3d,
                )

                if valid[0]:
                    cam_pos[0, :2] = desired_pos[0]
                    cam_pos[0, 2] = new_z[0] + CAMERA_HEIGHT
                    current_tri_idx = new_tri

            elif action == 'turn_left':
                cam_yaw[0] += math.radians(TURN_ANGLE)

            elif action == 'turn_right':
                cam_yaw[0] -= math.radians(TURN_ANGLE)

            elif action == 'reset':
                cam_pos[:] = initial_cam_pos
                cam_yaw[:] = initial_cam_yaw
                valid, _, current_tri_idx = batch_find_floor(
                    cam_pos[:, :2], cam_pos[:, 2] - CAMERA_HEIGHT,
                    tri_v0_2d, tri_v1_2d, tri_v2_2d,
                    tri_v0_3d, tri_v1_3d, tri_v2_3d,
                )

            # Render
            _update_camera(cam, cam_pos, cam_yaw)
            rgb_batch, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, force_render=True)

            if rgb_batch.dtype == torch.uint8:
                rgb_batch = rgb_batch.float() / 255.0
            rgb_batch = torch.clamp(rgb_batch * LIGHT_SCALE * 255.0, 0, 255).byte()
            rgb_np = rgb_batch[0].cpu().numpy()  # (H, W, 3)

            # Add debug overlay
            frame = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]

            # Position text
            pos_x, pos_y, pos_z = float(cam_pos[0, 0]), float(cam_pos[0, 1]), float(cam_pos[0, 2])
            yaw_deg = float(cam_yaw[0]) * 180 / math.pi
            on_mesh = "ON_MESH" if valid[0] else "OFF_MESH"
            mesh_color = (0, 255, 0) if valid[0] else (0, 0, 255)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
            cv2.putText(frame, f"Pos: x={pos_x:7.2f} y={pos_y:7.2f} z={pos_z:7.2f}",
                       (5, 25), font, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Yaw: {yaw_deg:6.1f}°  Tri: {int(current_tri_idx[0])}",
                       (5, 50), font, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, on_mesh,
                       (5, 75), font, 0.6, mesh_color, 2)

            # Encode to JPEG (quality=75 for faster encoding)
            _, frame_jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            DebugHTTPHandler.shared_state['frame_bytes'] = frame_jpg.tobytes()
            DebugHTTPHandler.shared_state['pos'] = [pos_x, pos_y, pos_z]
            DebugHTTPHandler.shared_state['yaw'] = float(cam_yaw[0])
            DebugHTTPHandler.shared_state['on_mesh'] = bool(valid[0])
            DebugHTTPHandler.shared_state['tri_idx'] = int(current_tri_idx[0])
            DebugHTTPHandler.shared_state['scene'] = SCAN_NAME
            DebugHTTPHandler.shared_state['episode'] = episode_idx
            DebugHTTPHandler.shared_state['total_episodes'] = len(scene_episodes)

            time.sleep(0.02)  # ~50 FPS

    except KeyboardInterrupt:
        print("\n[Quit] Shutting down...")
        server.shutdown()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis interactive scene debug")
    parser.add_argument("--scene_id", required=True, help="Scene ID (e.g., 2azQ1b91cZZ)")
    parser.add_argument("--episode_idx", type=int, default=0, help="Episode index in scene")
    parser.add_argument("--port", type=int, default=9997, help="HTTP server port")
    args = parser.parse_args()

    run_debug_scene(args.scene_id, args.episode_idx, args.port)
