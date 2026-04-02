"""
genark_config.py — Unified config loader for GenArk evaluation.

Usage:
    from genark_config import load_config
    cfg = load_config("configs/uninavid_r2r.yaml")

    cfg.SIMULATOR.CAMERA_HEIGHT   -> 1.25
    cfg.SIMULATOR.RGB_SENSOR.HFOV -> 120
    cfg.TASK.MAX_STEPS            -> 500
"""

import yaml
import numpy as np


class _CfgNode(dict):
    """Dict that also supports attribute access (recursive)."""

    def __getattr__(self, name):
        try:
            v = self[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")
        return v

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def _to_cfg(obj):
    if isinstance(obj, dict):
        return _CfgNode({k: _to_cfg(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_cfg(item) for item in obj]
    return obj


def load_config(path: str) -> _CfgNode:
    """Load a YAML config and return it as a nested attribute-access object."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_cfg(raw)


# ---------- convenience accessors (avoid repeating .SIMULATOR.xxx everywhere) ----------

def sim_constants(cfg) -> dict:
    """Return a flat dict of simulator/task constants used by env_worker / serial script."""
    sim = cfg.SIMULATOR
    return {
        "CAMERA_HEIGHT":      sim.CAMERA_HEIGHT,
        "STEP_MOVE":          sim.FORWARD_STEP_SIZE,
        "STEP_TURN":          np.deg2rad(sim.TURN_ANGLE),
        "ALLOW_SLIDING":      sim.ALLOW_SLIDING,
        "MAX_STEP_HEIGHT":    sim.MAX_STEP_HEIGHT,
        "CAM_RES":            (sim.RGB_SENSOR.WIDTH, sim.RGB_SENSOR.HEIGHT),
        "FOV":                sim.RGB_SENSOR.HFOV,
        "LIGHT_SCALE":        sim.LIGHT_SCALE,
        "MAX_STEPS":          cfg.TASK.MAX_STEPS,
        "SUCCESS_DISTANCE":   cfg.TASK.SUCCESS_DISTANCE,
    }
