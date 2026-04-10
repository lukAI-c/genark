"""
Visualize the reference_path trajectory of a single episode (default: episode_id=42)
from OpenNav_R2R-CE_100_bertidx.json. Renders RGB frames along the path and saves
them as PNGs (plus an optional MP4 if cv2 has a video writer available).
"""
import genesis as gs
import torch
import numpy as np
import os
import shutil
import json
import cv2

# --- Config ---
EPISODES_FILE = "/Users/dinghongyu/code/genesis-habitat/Genesis/scripts/OpenNav_R2R-CE_100_bertidx.json"
SCENE_DATASETS = "/Users/dinghongyu/code/genesis-habitat/Genesis/scripts/data/scene"
EPISODE_ID = 42
OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"output/episode_{EPISODE_ID}.mp4"

CAMERA_HEIGHT = 0.88
CAM_RES = (640, 480)
FOV = 64
LIGHT_SCALE = 1.0

# Path interpolation: distance (meters) between consecutive rendered frames
STEP_DIST = 0.1
# Number of frames used to smoothly turn between segments
TURN_FRAMES = 10
VIDEO_FPS = 15


def habitat_to_genesis(p):
    # habitat (x, y_up, z) -> genesis (x, -z, y_up)
    return np.array([p[0], -p[2], p[1]], dtype=np.float32)


def yaw_from_quat(q):
    # q = [x, y, z, w] (habitat convention used by sibling script)
    x, y, z, w = q
    dir_x_h = -2 * w * y
    dir_z_h = y * y - w * w
    return float(np.arctan2(-dir_z_h, dir_x_h))


def build_camera_path(waypoints_g, start_yaw):
    """Returns list of (pos[3], yaw) tuples densely sampled along the polyline."""
    poses = []
    cur_yaw = start_yaw

    # Initial pose at first waypoint with starting yaw
    poses.append((waypoints_g[0].copy(), cur_yaw))

    for i in range(len(waypoints_g) - 1):
        p0 = waypoints_g[i]
        p1 = waypoints_g[i + 1]
        delta = p1 - p0
        seg_len = float(np.linalg.norm(delta[:2]))
        if seg_len < 1e-6:
            continue
        target_yaw = float(np.arctan2(delta[1], delta[0]))

        # Smooth yaw turn in place before walking the segment
        d_yaw = (target_yaw - cur_yaw + np.pi) % (2 * np.pi) - np.pi
        for k in range(1, TURN_FRAMES + 1):
            y = cur_yaw + d_yaw * (k / TURN_FRAMES)
            poses.append((p0.copy(), y))
        cur_yaw = target_yaw

        # Walk the segment
        n_steps = max(1, int(np.ceil(seg_len / STEP_DIST)))
        for k in range(1, n_steps + 1):
            t = k / n_steps
            p = p0 + delta * t
            poses.append((p.copy(), cur_yaw))

    return poses


def wrap_text(text, font, font_scale, thickness, max_width):
    """Greedy word-wrap so each line's pixel width <= max_width."""
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        trial = w if not cur else cur + " " + w
        (tw, _), _ = cv2.getTextSize(trial, font, font_scale, thickness)
        if tw <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_caption(text, width):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_h = 20
    pad_x = 10
    pad_y = 8
    lines = wrap_text(text, font, font_scale, thickness, width - 2 * pad_x)
    h = pad_y * 2 + line_h * len(lines)
    canvas = np.zeros((h, width, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        y = pad_y + line_h * (i + 1) - 5
        cv2.putText(canvas, line, (pad_x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def main():
    # --- Load episode ---
    print(f"Loading {EPISODES_FILE}")
    with open(EPISODES_FILE, "r") as f:
        data = json.load(f)
    episode = next((e for e in data["episodes"] if e["episode_id"] == EPISODE_ID), None)
    if episode is None:
        raise RuntimeError(f"episode_id {EPISODE_ID} not found")

    scene_id = episode["scene_id"]
    SCAN_NAME = os.path.basename(os.path.dirname(scene_id))
    instruction_text = episode["instruction"].get("instruction_text", "")
    print(f"Scene: {SCAN_NAME}")
    print(f"Instruction: {instruction_text}")
    print(f"Reference path length: {len(episode['reference_path'])}")

    # Convert waypoints (habitat -> genesis), lift to camera height
    waypoints_g = []
    for p in episode["reference_path"]:
        pg = habitat_to_genesis(p)
        pg[2] += CAMERA_HEIGHT
        waypoints_g.append(pg)

    start_yaw = yaw_from_quat(episode["start_rotation"])
    poses = build_camera_path(waypoints_g, start_yaw)
    print(f"Generated {len(poses)} camera poses")

    # --- Init Genesis ---
    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        show_viewer=False,
        vis_options=gs.options.VisOptions(ambient_light=(1.0, 1.0, 1.0), plane_reflection=False),
    )

    obj_dir = f"{SCENE_DATASETS}/{SCAN_NAME}/matterport_mesh"
    obj_files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]
    if not obj_files:
        raise FileNotFoundError(f"No .obj in {obj_dir}")
    obj_path = os.path.join(obj_dir, obj_files[0])
    scene.add_entity(morph=gs.morphs.Mesh(file=obj_path, fixed=True, collision=False))

    cam = scene.add_camera(res=CAM_RES, fov=FOV, GUI=False)
    scene.build(n_envs=1, env_spacing=(0.0, 0.0))

    # --- Pre-render caption strip ---
    caption = render_caption(instruction_text, CAM_RES[0])
    out_w = CAM_RES[0]
    out_h = CAM_RES[1] + caption.shape[0]

    # --- Prepare output ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, VIDEO_FPS, (out_w, out_h))

    # --- Render loop ---
    for step, (pos, yaw) in enumerate(poses):
        lookat = pos.copy()
        lookat[0] += float(np.cos(yaw))
        lookat[1] += float(np.sin(yaw))

        cam.set_pose(pos=tuple(pos.tolist()), lookat=tuple(lookat.tolist()))

        rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, force_render=True)
        img = np.asarray(rgb)
        if img.ndim == 4:
            img = img[0]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        img = np.clip(img * LIGHT_SCALE * 255.0, 0, 255).astype(np.uint8)
        img_bgr = img[..., ::-1]

        frame = np.vstack([img_bgr, caption])
        video_writer.write(frame)

        if step % 20 == 0:
            print(f"  rendered {step}/{len(poses)}")

    video_writer.release()
    print(f"Saved {len(poses)} frames to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
