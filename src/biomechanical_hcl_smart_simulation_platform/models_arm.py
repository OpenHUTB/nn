#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render an arm model (e.g., Arm26 converted by MyoConverter) to PNG/GIF using MuJoCo Python bindings.

Examples:
  # Render a GIF (recommended)
  python examples/render_arm.py --xml ./models/mjc/Arm26/arm26_cvt3.xml --out arm.gif

  # Render a still image
  python examples/render_arm.py --xml ./models/mjc/Arm26/arm26_cvt3.xml --out arm.png

Notes:
  - MyoConverter converted XML often contains a keyframe that must be loaded to satisfy joint/muscle path constraints.
    This script will automatically load keyframe 0 if present. (See README.)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import mujoco
import imageio.v2 as imageio


def _pick_first_hinge_joint(model: mujoco.MjModel) -> int | None:
    """Pick the first hinge joint as a simple 'elbow-like' joint."""
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            return jid
    return None


def _joint_id_by_name(model: mujoco.MjModel, name: str) -> int | None:
    try:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(jid) if jid >= 0 else None
    except Exception:
        return None


def _setup_camera(model: mujoco.MjModel, azimuth: float, elevation: float, distance_scale: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    # A reasonable default framing: focus model center, distance proportional to extent
    extent = float(model.stat.extent) if float(model.stat.extent) > 0 else 1.0
    center = np.array(model.stat.center, dtype=float)

    cam.lookat[:] = center
    cam.distance = max(0.05, distance_scale * extent)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    return cam


def _reset_to_keyframe_if_any(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # README: converted XML contains keyframe; MuJoCo doesn't load it by default.
    # Use mj_resetDataKeyframe(model, data, 0) to load keyframe 0.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def render(
    xml_path: Path,
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    seconds: float,
    joint_name: str | None,
    azimuth: float,
    elevation: float,
    distance_scale: float,
) -> None:
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    _reset_to_keyframe_if_any(model, data)

    # Choose joint
    jid = _joint_id_by_name(model, joint_name) if joint_name else None
    if jid is None:
        jid = _pick_first_hinge_joint(model)

    if jid is None:
        print("[WARN] No hinge joint found. Will just render the initial pose.")
    else:
        print(f"[INFO] Using joint id={jid}, name='{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)}'")

    # Joint qpos address (hinge = 1 DoF)
    qadr = int(model.jnt_qposadr[jid]) if jid is not None else None

    # Joint range (if not defined, fall back)
    if jid is not None:
        jrange = np.array(model.jnt_range[jid], dtype=float)
        if np.all(np.isfinite(jrange)) and (jrange[1] > jrange[0]):
            lo, hi = float(jrange[0]), float(jrange[1])
        else:
            lo, hi = -0.5, 0.5
        mid = 0.5 * (lo + hi)
        amp = 0.45 * (hi - lo)
    else:
        lo, hi, mid, amp = 0.0, 0.0, 0.0, 0.0

    cam = _setup_camera(model, azimuth=azimuth, elevation=elevation, distance_scale=distance_scale)
    renderer = mujoco.Renderer(model, height=height, width=width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".png":
        # still image
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        imageio.imwrite(out_path, rgb)
        print(f"[OK] Saved: {out_path}")
        return

    if suffix != ".gif":
        raise ValueError(f"Unsupported output format '{suffix}'. Use .gif or .png")

    n_frames = max(1, int(round(seconds * fps)))
    frames = []

    for i in range(n_frames):
        t = i / fps

        if jid is not None and qadr is not None:
            # simple sinusoidal motion within range
            angle = mid + amp * math.sin(2.0 * math.pi * (t / max(1e-6, seconds)))
            data.qpos[qadr] = angle
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        frames.append(rgb)

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[OK] Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, required=True, help="Path to MuJoCo XML (e.g. ./models/mjc/Arm26/arm26_cvt3.xml)")
    ap.add_argument("--out", type=str, default="arm.gif", help="Output file: .gif or .png")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--joint", type=str, default=None, help="Optional joint name to animate (otherwise first hinge joint)")
    ap.add_argument("--azimuth", type=float, default=110.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    ap.add_argument("--distance_scale", type=float, default=1.6)

    args = ap.parse_args()
    render(
        xml_path=Path(args.xml),
        out_path=Path(args.out),
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds=args.seconds,
        joint_name=args.joint,
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance_scale=args.distance_scale,
    )


if __name__ == "__main__":
    main()
