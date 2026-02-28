#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single entrypoint for BeatSVR bimanual evaluation visualization.

Outputs (same rollout, two views):
- evaluate/mainCamera.mp4  (GUI view)
- evaluate/envCamera.mp4   (env/perception view)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import imageio.v2 as imageio


# ---- Import Simulator (works both inside UITB repo and inside built simulator folder) ----
try:
    from uitb import Simulator  # type: ignore
except Exception:
    from simulator import Simulator  # type: ignore


# -----------------------------
# Video utilities
# -----------------------------
def _to_rgb_u8(frame: np.ndarray) -> np.ndarray:
    """Ensure uint8 RGB (H,W,3)."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def write_mp4(frames: List[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps)
    try:
        for f in frames:
            writer.append_data(_to_rgb_u8(f))
    finally:
        writer.close()


# -----------------------------
# Frame stack helpers (GUI + perception)
# -----------------------------
def pop_gui_frames(sim) -> List[np.ndarray]:
    """Fetch & clear GUI render stack (mainCamera). Requires render_mode='rgb_array_list'."""
    try:
        frames = sim.get_render_stack() or []
        sim.clear_render_stack()
        return list(frames)
    except Exception:
        return []


def pop_env_frames(sim) -> List[np.ndarray]:
    """
    Fetch & clear perception render stack (envCamera) when render_mode_perception='separate'.

    Some implementations return a dict: {module_name: frames}.
    We default to taking the first module's frames.
    """
    try:
        frames = sim.get_render_stack_perception()
        sim.clear_render_stack_perception()

        if frames is None:
            return []

        if isinstance(frames, dict):
            for _, v in frames.items():
                return list(v or [])
            return []

        return list(frames)
    except Exception:
        return []


# -----------------------------
# Core: evaluation + recording (same rollout, two views)
# -----------------------------
def run_eval_record(args: argparse.Namespace) -> None:
    sim_dir = Path(args.simulator_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = sim_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enable both stacks:
    # GUI frames -> get_render_stack()
    # Perception frames -> get_render_stack_perception() when "separate"
    sim = Simulator.get(
        str(sim_dir),
        render_mode="rgb_array_list",
        render_mode_perception="separate",
    )

    main_frames: List[np.ndarray] = []
    env_frames: List[np.ndarray] = []

    for ep in range(args.episodes):
        obs, info = sim.reset()

        # Grab initial frames (if any)
        main_frames += pop_gui_frames(sim)
        env_frames += pop_env_frames(sim)

        for _ in range(args.max_steps):
            # Replace this with your trained policy action if available
            action = sim.action_space.sample()

            obs, reward, terminated, truncated, info = sim.step(action)

            # Collect frames produced at this step
            main_frames += pop_gui_frames(sim)
            env_frames += pop_env_frames(sim)

            if terminated or truncated:
                break

    # Write outputs
    main_mp4 = out_dir / "mainCamera.mp4"
    env_mp4 = out_dir / "envCamera.mp4"

    if main_frames:
        write_mp4(main_frames, main_mp4, fps=args.fps)
        print(f"[OK] mainCamera saved: {main_mp4}")
    else:
        print("[WARN] mainCamera frames empty. GUI render stack not populated?")

    if env_frames:
        write_mp4(env_frames, env_mp4, fps=args.fps)
        print(f"[OK] envCamera saved: {env_mp4}")
    else:
        print("[WARN] envCamera frames empty. Perception separate render not populated?")


# -----------------------------
# CLI: single entrypoint
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="beatsvr_bimanual_eval",
        description="Single-entry evaluation recorder: bimanual + envCamera + mainCamera."
    )

    p.add_argument(
        "--simulator_dir",
        default=".",
        help="Built simulator folder (default: current dir)."
    )
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to record.")
    p.add_argument("--max_steps", type=int, default=2000, help="Max steps per episode.")
    p.add_argument("--fps", type=int, default=20, help="Output video fps.")
    p.add_argument(
        "--out_dir",
        default="evaluate",
        help="Output dir (default: <simulator_dir>/evaluate)."
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_eval_record(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
