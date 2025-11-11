#!/usr/bin/env python3
"""
Main entry (render-capable) for MOBL pointing demo.

"""

import os
import sys
import argparse
import shutil
import subprocess

def run_window(episodes: int, steps: int) -> None:
    # 不要设置 MUJOCO_GL=egl；用默认的窗口渲染
    from uitb import Simulator
    sim = Simulator.get("simulators/mobl_arms_index_pointing")
    for ep in range(episodes):
        obs, info = sim.reset()
        total = 0.0
        for t in range(steps):
            # 尝试弹窗渲染；如果环境不支持会抛错
            try:
                sim.render()
            except Exception as e:
                print("render warn:", e)
                raise
            obs, r, term, trunc, info = sim.step(sim.action_space.sample())
            total += float(r)
            if term or trunc:
                break
        print(f"[window] episode {ep+1}/{episodes} reward={total:.3f}")
    sim.close()
    print("DONE (window) ✅")

def run_record(outfile: str, episodes: int, steps: int, fps: int) -> None:
    # 用 Gymnasium 的 rgb_array 离屏渲染，并通过 ffmpeg 写 mp4
    os.environ.setdefault("MUJOCO_GL", "egl")
    if shutil.which("ffmpeg") is None:
        print("未找到 ffmpeg：请先安装 `sudo apt install -y ffmpeg`")
        sys.exit(1)
    try:
        import gymnasium as gym
    except Exception as e:
        print("未安装 gymnasium，请先安装：pip install gymnasium")
        print("错误：", e)
        sys.exit(1)

    env_id = "uitb:mobl_arms_index_pointing-v0"
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except TypeError:
        # 某些版本不接受 render_mode 关键字，退回默认创建
        env = gym.make(env_id)
    obs, info = env.reset()

    # 先拿一帧确定分辨率
    frame = env.render()
    h, w = frame.shape[0], frame.shape[1]

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        outfile,
    ]
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    import numpy as np  # 只为确保 tobytes 可用；若没装 numpy，uitb 依赖里一般会带
    for ep in range(episodes):
        obs, info = env.reset()
        total = 0.0
        for t in range(steps):
            frame = env.render()
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            proc.stdin.write(frame.tobytes())
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            total += float(r)
            if term or trunc:
                break
        print(f"[record] episode {ep+1}/{episodes} reward={total:.3f}")

    proc.stdin.close()
    proc.wait()
    env.close()
    print(f"DONE (record) → {outfile} ✅")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["window", "record", "auto"], default="record",
                   help="window=弹窗渲染; record=离屏录制MP4; auto=先window失败则record")
    p.add_argument("--out", default="assets/demo.mp4", help="record 模式输出 mp4 路径")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    # 依赖提示
    try:
        import uitb  # noqa
    except Exception as e:
        print("未检测到 uitb（user-in-the-box）。请按 README 安装依赖后再运行。")
        print("原始错误：", e)
        sys.exit(1)

    if args.mode == "window":
        run_window(args.episodes, args.steps)
    elif args.mode == "record":
        run_record(args.out, args.episodes, args.steps, args.fps)
    else:  # auto
        try:
            run_window(args.episodes, args.steps)
        except Exception:
            print("窗口渲染失败，自动切换到 record(EGL) 模式……")
            run_record(args.out, args.episodes, args.steps, args.fps)

if __name__ == "__main__":
    main()
