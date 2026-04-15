#!/usr/bin/env python3
"""
orchestrator.py — Multi-GPU evaluation orchestrator.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  model_server.py  (GPU: --model_gpu)                        │
  │  UniNaVid loaded once, serves all workers via HTTP          │
  └──────────────────────────┬──────────────────────────────────┘
                             │  HTTP  (raw-binary multipart)
          ┌──────────────────┼────────────────────┐
          │                  │                    │
  ┌───────▼──────┐   ┌───────▼──────┐   ┌────────▼─────┐
  │ env_worker   │   │ env_worker   │   │ env_worker   │
  │ (scene A)    │   │ (scene B)    │   │ (scene C)    │
  │ GPU worker_gpus[0 % n_workers]  │   │ …            │
  └──────────────┘   └──────────────┘   └──────────────┘

Usage:
    python orchestrator.py \\
        --model_gpu    5            \\   # physical GPU for UniNaVid
        --worker_gpus  0 1 2 3      \\   # physical GPUs for Genesis workers
        --port         8100         \\   # model-server port
        --max_parallel 8            \\   # max episodes per scene (Vulkan VRAM limit)
        --max_steps    500          \\
        --output       metrics_final.json

The orchestrator:
  1. Starts model_server.py as a subprocess on --model_gpu.
  2. Waits until /health returns 200.
  3. Groups episodes by scene; dispatches one env_worker.py subprocess per
     scene, round-robin over --worker_gpus.
  4. Runs at most --max_concurrent workers simultaneously.
  5. Collects per-episode metrics from each worker's output JSON.
  6. Aggregates and saves the final metrics.
"""

import sys
import os
import json
import gzip
import time
import argparse
import tempfile
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm
from datetime import datetime

from genark_config import load_config, sim_constants

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CFG     = os.path.join(os.path.dirname(__file__), "configs", "uninavid_r2r.yaml")
WORKER_SCRIPT   = os.path.join(os.path.dirname(__file__), "env_worker.py")
SERVER_SCRIPT   = os.path.join(os.path.dirname(__file__), "model_server.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_episodes(episodes_file: str) -> list[dict]:
    tqdm.write(f"[Orch] Loading episodes from {episodes_file} …")
    if episodes_file.endswith(".gz"):
        with gzip.open(episodes_file, "rt") as f:
            data = json.load(f)
    else:
        with open(episodes_file) as f:
            data = json.load(f)
    eps = data["episodes"]
    tqdm.write(f"[Orch] Loaded {len(eps)} episodes.")
    return eps


def group_by_scene(episodes: list[dict], max_per_scene: int | None) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for ep in episodes:
        grouped.setdefault(ep["scene_id"], []).append(ep)
    if max_per_scene is not None:
        grouped = {k: v[:max_per_scene] for k, v in grouped.items()}
    return grouped


def wait_server_healthy(url: str, retries: int = 120, interval: float = 5.0):
    """Block until model server /health returns 200 or raise after timeout."""
    tqdm.write(f"[Orch] Waiting for model server at {url} …")
    for attempt in range(retries):
        try:
            r = requests.get(f"{url}/health", timeout=10)
            if r.status_code == 200:
                tqdm.write(f"[Orch] Server healthy after {attempt * interval:.0f}s.")
                return
        except Exception:
            pass
        time.sleep(interval)
    raise RuntimeError(f"Model server did not become healthy after {retries * interval}s")


def start_model_server(model_gpu: int, port: int, model_path: str,
                       log_path: str, chunk_size: int = 16,
                       generate_cfg: dict | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(model_gpu)
    gen = generate_cfg or {}
    cmd = [
        sys.executable, SERVER_SCRIPT,
        "--port",       str(port),
        "--model_path", model_path,
        "--chunk_size", str(chunk_size),
        "--do_sample",      str(gen.get("DO_SAMPLE", True)),
        "--temperature",    str(gen.get("TEMPERATURE", 0.2)),
        "--max_new_tokens", str(gen.get("MAX_NEW_TOKENS", 128)),
    ]
    tqdm.write(f"[Orch] Starting model server on GPU {model_gpu} (log → {log_path})")
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
    return proc


def _safe_read_progress(path: str) -> dict | None:
    """Read a worker heartbeat JSON; return None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def run_worker(
    scene_id:      str,
    episodes:      list[dict],
    gpu_id:        int,
    server_url:    str,
    tmp_dir:       str,
    worker_idx:    int,
    max_steps:     int,
    cam_res:       list[int],
    fov:              int,
    light_scale:      float,
    success_distance: float,
    progress_file:    str,
    log_file:         str,
    web_port:         int | None = None,
    web_env_idx:      int = 0,
    web_output_dir:   str | None = None,
    video_output_dir: str | None = None,
    video_fps:        int = 5,
    use_glb:          bool = False,
) -> dict:
    """
    Launch one env_worker.py or env_worker_glb.py subprocess for a single scene.
    Returns the metrics dict (episode_id → metrics).
    """
    ep_file  = os.path.join(tmp_dir, f"worker_{worker_idx}_episodes.json")
    out_file = os.path.join(tmp_dir, f"worker_{worker_idx}_metrics.json")

    with open(ep_file, "w") as f:
        json.dump(episodes, f)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Select worker script based on rendering path
    worker_script = WORKER_SCRIPT.replace("env_worker.py",
                                          "env_worker_glb.py" if use_glb else "env_worker.py")

    cmd = [
        sys.executable, worker_script,
        "--scene_id",      scene_id,
        "--episodes_file", ep_file,
        "--server_url",    server_url,
        "--output_file",   out_file,
        "--progress_file", progress_file,
        "--max_steps",     str(max_steps),
        "--cam_res",       str(cam_res[0]), str(cam_res[1]),
        "--fov",           str(fov),
        "--light_scale",   str(light_scale),
        "--success_distance", str(success_distance),
    ]
    if web_port is not None:
        cmd += ["--web_port", str(web_port),
                "--web_env_idx", str(web_env_idx)]
        if web_output_dir is not None:
            cmd += ["--web_output_dir", web_output_dir]
    if video_output_dir is not None:
        cmd += ["--video_output_dir", video_output_dir,
                "--video_fps", str(video_fps)]

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
        proc.wait()

    if os.path.exists(out_file):
        with open(out_file) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VLN multi-GPU orchestrator")
    p.add_argument("--cfg",           type=str, default=DEFAULT_CFG,
                   help="Path to YAML config file")
    # All below can override the config file values
    p.add_argument("--episodes_file", type=str, default=None)
    p.add_argument("--model_path",    type=str, default=None)
    p.add_argument("--model_gpu",     type=int, default=None)
    p.add_argument("--worker_gpus",   type=int, nargs="+", default=None)
    p.add_argument("--port",          type=int, default=None)
    p.add_argument("--max_parallel",  type=int, default=None)
    p.add_argument("--max_concurrent",type=int, default=None)
    p.add_argument("--max_steps",     type=int, default=None)
    p.add_argument("--cam_res",       type=int, nargs=2, default=None)
    p.add_argument("--fov",           type=int, default=None)
    p.add_argument("--light_scale",   type=float, default=None)
    p.add_argument("--chunk_size",       type=int, default=None)
    p.add_argument("--success_distance", type=float, default=None)
    p.add_argument("--output",           type=str, default="metrics_final.json")
    p.add_argument("--web_base_port",    type=int, default=None,
                   help="If set, worker w gets HTTP viewer on (web_base_port + w)")
    p.add_argument("--web_env_idx",      type=int, default=0,
                   help="Which env (0..N-1) in each parallel batch to broadcast")
    p.add_argument("--use_glb",          action="store_true", default=False,
                   help="Use GLB rendering path (auto-converts OBJ to GLB); default: OBJ-only")
    cli = p.parse_args()

    # Load YAML config and merge: CLI overrides > YAML
    cfg = load_config(cli.cfg)
    sc  = sim_constants(cfg)

    cli.episodes_file  = cli.episodes_file or cfg.EPISODES_FILE
    cli.model_path     = cli.model_path    or cfg.MODEL_PATH
    cli.model_gpu      = cli.model_gpu     if cli.model_gpu is not None else cfg.PARALLEL.MODEL_GPU
    cli.worker_gpus    = cli.worker_gpus   or cfg.PARALLEL.WORKER_GPUS
    cli.port           = cli.port          if cli.port is not None else cfg.PARALLEL.PORT
    cli.max_parallel   = cli.max_parallel  if cli.max_parallel is not None else cfg.PARALLEL.MAX_PARALLEL
    cli.max_concurrent = cli.max_concurrent if cli.max_concurrent is not None else cfg.PARALLEL.MAX_CONCURRENT
    cli.chunk_size     = cli.chunk_size    if cli.chunk_size is not None else cfg.PARALLEL.CHUNK_SIZE
    cli.max_steps      = cli.max_steps     if cli.max_steps is not None else cfg.TASK.MAX_STEPS
    cli.cam_res        = cli.cam_res       or list(sc["CAM_RES"])
    cli.fov            = cli.fov           if cli.fov is not None else sc["FOV"]
    cli.light_scale       = cli.light_scale   if cli.light_scale is not None else sc["LIGHT_SCALE"]
    cli.success_distance  = cli.success_distance if cli.success_distance is not None else sc["SUCCESS_DISTANCE"]
    cli.save_video        = cfg.PARALLEL.SAVE_VIDEO
    cli.video_output_dir  = cfg.PARALLEL.VIDEO_OUTPUT_DIR
    cli.video_fps         = cfg.PARALLEL.VIDEO_FPS

    # Generate params (from YAML only, no CLI override needed)
    gen = cfg.get("GENERATE", {})
    cli.generate_cfg = dict(gen) if gen else {}

    return cli


def main():
    args = parse_args()

    server_url   = f"http://127.0.0.1:{args.port}"
    max_conc     = args.max_concurrent or len(args.worker_gpus)

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_min = datetime.now().strftime("%Y%m%d_%H%M")
    model_tag = os.path.basename(args.model_path.rstrip("/"))
    data_tag = os.path.splitext(os.path.basename(args.episodes_file))[0]
    out_root = os.path.join(os.path.dirname(__file__), "output_parallel")
    # video root from config: <VIDEO_OUTPUT_DIR>/<data_tag>/<ts_min>/
    video_root = os.path.join(os.path.dirname(__file__), args.video_output_dir, data_tag, ts_min)
    run_dir = os.path.join(out_root, f"{ts}__{model_tag}__{data_tag}__P{args.max_parallel}__C{max_conc}")
    os.makedirs(run_dir, exist_ok=True)
    tqdm.write(f"[Orch] Run output dir → {run_dir}")

    # ----------------------------------------------------------------
    # 1. Load + group episodes
    # ----------------------------------------------------------------
    episodes = load_episodes(args.episodes_file)
    scenes   = group_by_scene(episodes, args.max_parallel)
    tqdm.write(f"[Orch] {len(scenes)} scenes  |  max {args.max_parallel} eps/scene  "
               f"|  model GPU {args.model_gpu}  |  worker GPUs {args.worker_gpus}")

    # ----------------------------------------------------------------
    # 2. Start model server
    # ----------------------------------------------------------------
    tmp_dir     = tempfile.mkdtemp(prefix="vln_orch_")
    server_log  = os.path.join(tmp_dir, "model_server.log")
    server_proc = start_model_server(args.model_gpu, args.port, args.model_path, server_log,
                                     chunk_size=args.chunk_size,
                                     generate_cfg=args.generate_cfg)
    try:
        wait_server_healthy(server_url)
    except RuntimeError:
        server_proc.terminate()
        raise

    # ----------------------------------------------------------------
    # 3. Dispatch workers
    # ----------------------------------------------------------------
    metrics_all = {}
    scene_items = list(scenes.items())
    n_workers   = len(args.worker_gpus)

    tqdm.write(f"[Orch] Dispatching {len(scene_items)} scenes "
               f"with max {max_conc} concurrent workers …")

    total_eps   = sum(len(v) for v in scenes.values())
    _BAR_FMT    = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    # Top-level bars: one for scenes, one for episodes
    scene_pbar = tqdm(
        total         = len(scene_items),
        desc          = "Scenes  ",
        unit          = "scene",
        position      = 0,
        dynamic_ncols = True,
        bar_format    = _BAR_FMT,
        file          = sys.stderr,
    )
    ep_pbar = tqdm(
        total         = total_eps,
        desc          = "Episodes",
        unit          = "ep",
        position      = 1,
        dynamic_ncols = True,
        bar_format    = _BAR_FMT,
        file          = sys.stderr,
    )
    # Per-slot step bars (one per concurrent worker slot)
    slot_pbars = [
        tqdm(
            total         = args.max_steps,
            desc          = f"Slot {s:>2}",
            unit          = "step",
            position      = 2 + s,
            dynamic_ncols = True,
            bar_format    = _BAR_FMT,
            file          = sys.stderr,
            leave         = False,
        )
        for s in range(max_conc)
    ]

    with ThreadPoolExecutor(max_workers=max_conc) as pool:
        futures   = {}  # fut → meta dict
        slot_meta = {}  # slot_idx → current worker meta

        def _submit(w_idx: int, scene_id: str, eps: list, slot_idx: int):
            gpu          = args.worker_gpus[w_idx % n_workers]
            prog_file    = os.path.join(tmp_dir, f"worker_{w_idx}_progress.json")
            log_file     = os.path.join(tmp_dir, f"worker_{w_idx}.log")
            scan         = os.path.basename(os.path.dirname(scene_id))
            tqdm.write(f"[Orch] Worker {w_idx} → {scan} ({len(eps)} eps) on GPU {gpu}")
            fut = pool.submit(
                run_worker,
                scene_id      = scene_id,
                episodes      = eps,
                gpu_id        = gpu,
                server_url    = server_url,
                tmp_dir       = tmp_dir,
                worker_idx    = w_idx,
                max_steps        = args.max_steps,
                cam_res          = args.cam_res,
                fov              = args.fov,
                light_scale      = args.light_scale,
                success_distance = args.success_distance,
                progress_file = prog_file,
                log_file      = log_file,
                web_port      = (args.web_base_port + w_idx) if args.web_base_port is not None else None,
                web_env_idx   = args.web_env_idx,
                web_output_dir= os.path.join(run_dir, f"web_viewer/worker_{w_idx}_{scan}")
                                if args.web_base_port is not None else None,
                video_output_dir= os.path.join(video_root, scan) if args.save_video else None,
                video_fps     = args.video_fps,
                use_glb       = args.use_glb,
            )
            futures[fut] = {
                "scene_id":  scene_id,
                "n_sched":   len(eps),
                "prog_file": prog_file,
                "done_seen": 0,
                "slot":      slot_idx,
                "finalized": False,
            }
            slot_meta[slot_idx] = futures[fut]
            # Reset this slot's step bar
            bar = slot_pbars[slot_idx]
            bar.reset(total=args.max_steps)
            bar.set_description_str(f"{scan[:10]:>10}")
            bar.refresh()

        # Seed up to max_conc workers
        next_scene_idx = 0
        for slot_idx in range(min(max_conc, len(scene_items))):
            _submit(next_scene_idx, *scene_items[next_scene_idx], slot_idx)
            next_scene_idx += 1

        pending = set(futures.keys())
        while pending:
            done_set, pending = wait(pending, timeout=1.0,
                                     return_when=FIRST_COMPLETED)

            # Poll heartbeat files for all active workers
            ep_live = 0
            for fut, meta in futures.items():
                if meta["finalized"]:
                    ep_live += meta["done_seen"]
                    continue
                prog = _safe_read_progress(meta["prog_file"])
                if prog is None:
                    ep_live += meta["done_seen"]
                    continue
                fresh_done = min(prog.get("done", meta["done_seen"]), meta["n_sched"])
                if fresh_done > meta["done_seen"]:
                    meta["done_seen"] = fresh_done
                ep_live += meta["done_seen"]

                # Update slot step bar
                slot = meta["slot"]
                bar  = slot_pbars[slot]
                step = prog.get("step", 0)
                max_s = prog.get("max_step", args.max_steps)
                bar.total = max_s
                bar.n     = step
                bar.set_postfix_str(
                    f"act={prog.get('active', '?')}/{prog.get('total', '?')} "
                    f"fwd={prog.get('fwd', 0)} "
                    f"L={prog.get('left', 0)} "
                    f"R={prog.get('right', 0)} "
                    f"stop={prog.get('stop', 0)}",
                    refresh=True,
                )

            # Update top-level episode bar
            if ep_live > ep_pbar.n:
                ep_pbar.update(ep_live - ep_pbar.n)

            # Handle finished workers
            for fut in done_set:
                meta   = futures[fut]
                scan   = os.path.basename(os.path.dirname(meta["scene_id"]))
                result = fut.result()
                metrics_all.update(result)
                meta["done_seen"]  = meta["n_sched"]
                meta["finalized"]  = True

                # Clear slot bar
                slot_pbars[meta["slot"]].set_description_str(f"{'(idle)':>10}")
                slot_pbars[meta["slot"]].n = 0
                slot_pbars[meta["slot"]].refresh()

                scene_pbar.update(1)
                scene_pbar.set_postfix_str(
                    f"last={scan}  total={len(metrics_all)}", refresh=True)
                tqdm.write(f"[Orch] Scene {scan} done → {len(result)} eps  "
                           f"(collected {len(metrics_all)}/{total_eps})")

                # Submit next scene into the freed slot
                if next_scene_idx < len(scene_items):
                    _submit(next_scene_idx, *scene_items[next_scene_idx], meta["slot"])
                    pending.add(list(futures.keys())[-1])
                    next_scene_idx += 1

    # Close all bars in reverse order to keep terminal tidy
    for bar in reversed(slot_pbars):
        bar.close()
    ep_pbar.close()
    scene_pbar.close()

    # ----------------------------------------------------------------
    # 4. Save + summarise
    # ----------------------------------------------------------------
    out_name = os.path.basename(args.output)
    out_path = os.path.join(run_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    tqdm.write(f"\n[Orch] Saved {len(metrics_all)} episode metrics → {out_path}")

    if metrics_all:
        keys = list(next(iter(metrics_all.values())).keys())
        avg  = {k: float(np.mean([m[k] for m in metrics_all.values()])) for k in keys}
        tqdm.write("\n=== Average Metrics ===")
        for k, v in avg.items():
            tqdm.write(f"  {k:<22}: {v:.4f}")
    else:
        avg = {}

    avg_path = os.path.join(run_dir, "avg_metrics.json")
    with open(avg_path, "w") as f:
        json.dump(avg, f, indent=2)
    tqdm.write(f"[Orch] Saved avg metrics → {avg_path}")

    cfg_path = os.path.join(run_dir, "run_config.json")
    with open(cfg_path, "w") as f:
        json.dump({
            "episodes_file": args.episodes_file,
            "model_path": args.model_path,
            "model_gpu": args.model_gpu,
            "worker_gpus": args.worker_gpus,
            "port": args.port,
            "max_parallel": args.max_parallel,
            "max_concurrent": max_conc,
            "max_steps": args.max_steps,
            "cam_res": args.cam_res,
            "fov": args.fov,
            "light_scale": args.light_scale,
            "output_name": out_name,
        }, f, indent=2)
    tqdm.write(f"[Orch] Saved run config → {cfg_path}")

    # ----------------------------------------------------------------
    # 5. Shutdown model server
    # ----------------------------------------------------------------
    tqdm.write("\n[Orch] Shutting down model server …")
    server_proc.terminate()
    try:
        server_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server_proc.kill()

    # Add worker-logs here!!!! can be deleted
    import shutil
    # Preserve worker / server logs for postmortem before deleting tmp_dir
    logs_dst = os.path.join(run_dir, "logs")
    os.makedirs(logs_dst, exist_ok=True)
    for fname in os.listdir(tmp_dir):
        if fname.endswith(".log") or fname.endswith("_progress.json"):
            try:
                shutil.copy2(os.path.join(tmp_dir, fname), os.path.join(logs_dst, fname))
            except Exception:
                pass
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tqdm.write(f"[Orch] Logs saved to {logs_dst}")

    tqdm.write("[Orch] Done.")


if __name__ == "__main__":
    main()
