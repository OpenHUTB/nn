#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render a leg model (e.g., Leg6Dof9Musc converted by MyoConverter) to PNG/GIF using MuJoCo Python bindings.

Examples:
  # Render a GIF (recommended)
  python examples/render_leg.py --xml ./models/mjc/Leg6Dof9Musc/leg6dof9musc_cvt3.xml --out leg.gif

  # Render a still image
  python examples/render_leg.py --xml ./models/mjc/Leg6Dof9Musc/leg6dof9musc_cvt3.xml --out leg.png

Notes:
  - Converted XML contains a keyframe that should be loaded to satisfy joint/muscle path constraints.
    MuJoCo doesn't load it by default, so we call mj_resetDataKeyframe(model, data, 0).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import mujoco
import imageio.v2 as imageio


# ----------------------------- helpers -----------------------------

def reset_to_keyframe_if_any(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Load keyframe 0 if present; otherwise do a normal reset."""
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def setup_camera(model: mujoco.MjModel, azimuth: float, elevation: float, distance_scale: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    extent = float(model.stat.extent) if float(model.stat.extent) > 0 else 1.0
    center = np.array(model.stat.center, dtype=float)

    cam.lookat[:] = center
    cam.distance = max(0.05, distance_scale * extent)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    return cam


def joint_name(model: mujoco.MjModel, jid: int) -> str:
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    return "" if n is None else str(n)


def pick_leg_joints(model: mujoco.MjModel) -> List[int]:
    """
    Heuristic:
      1) Prefer hinge joints whose names contain knee/ankle/hip (case-insensitive).
      2) If none found, fall back to the first hinge joint.
    """
    candidates: List[Tuple[int, int]] = []  # (priority, jid)
    keys = [
        ("knee", 0),
        ("ankle", 1),
        ("hip", 2),
        ("talus", 3),
        ("subtalar", 3),
    ]

    for jid in range(model.njnt):
        jtype = int(model.jnt_type[jid])
        if jtype != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        name = joint_name(model, jid).lower()
        for k, pri in keys:
            if k in name:
                candidates.append((pri, jid))
                break

    if candidates:
        candidates.sort(key=lambda x: x[0])
        # Keep up to 3 joints to animate (knee + ankle + hip is usually enough)
        return [jid for _, jid in candidates[:3]]

    # fallback: first hinge joint
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            return [jid]

    return []


def get_range_or_default(model: mujoco.MjModel, jid: int, default_lo=-0.5, default_hi=0.5) -> Tuple[float, float]:
    r = np.array(model.jnt_range[jid], dtype=float)
    if np.all(np.isfinite(r)) and (r[1] > r[0]):
        return float(r[0]), float(r[1])
    return float(default_lo), float(default_hi)


# ----------------------------- main render -----------------------------

def render_leg(
    xml_path: Path,
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    seconds: float,
    azimuth: float,
    elevation: float,
    distance_scale: float,
    joint_names: List[str],
) -> None:
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    reset_to_keyframe_if_any(model, data)

    # Choose joints
    jids: List[int] = []

    # user-specified joint names (optional)
    if joint_names:
        for nm in joint_names:
            try:
                jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm))
            except Exception:
                jid = -1
            if jid >= 0:
                jids.append(jid)

    # heuristic fallback
    if not jids:
        jids = pick_leg_joints(model)

    if not jids:
        print("[WARN] No hinge joint found. Will just render initial pose.")
    else:
        print("[INFO] Animating joints:")
        for jid in jids:
            print(f"  - id={jid:3d} name='{joint_name(model, jid)}' qposadr={int(model.jnt_qposadr[jid])}")

    cam = setup_camera(model, azimuth=azimuth, elevation=elevation, distance_scale=distance_scale)
    renderer = mujoco.Renderer(model, height=height, width=width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    # still image
    if suffix == ".png":
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        imageio.imwrite(out_path, rgb)
        print(f"[OK] Saved: {out_path}")
        return

    # gif
    if suffix != ".gif":
        raise ValueError(f"Unsupported output format '{suffix}'. Use .gif or .png")

    n_frames = max(1, int(round(seconds * fps)))
    frames = []

    # Precompute per-joint motion params
    motion = []
    for jid in jids:
        qadr = int(model.jnt_qposadr[jid])  # hinge => 1 dof
        lo, hi = get_range_or_default(model, jid)
        mid = 0.5 * (lo + hi)
        amp = 0.40 * (hi - lo)  # keep margin inside limits
        motion.append((qadr, mid, amp))

    for i in range(n_frames):
        t = i / fps

        # simple gait-like phase offsets between joints
        for k, (qadr, mid, amp) in enumerate(motion):
            phase = (k * math.pi / 3.0)  # 0, 60deg, 120deg
            val = mid + amp * math.sin(2.0 * math.pi * (t / max(1e-6, seconds)) + phase)
            data.qpos[qadr] = val

        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[OK] Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, required=True,
                    help="Path to MuJoCo XML (e.g. ./models/mjc/Leg6Dof9Musc/leg6dof9musc_cvt3.xml)")
    ap.add_argument("--out", type=str, default="leg.gif", help="Output file: .gif or .png")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=2.5)

    # camera
    ap.add_argument("--azimuth", type=float, default=95.0)
    ap.add_argument("--elevation", type=float, default=-15.0)
    ap.add_arg_
