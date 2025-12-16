"""
UITB Bimanual Evaluation Multi-View Recorder
Record evaluation rollouts with:
- mainCamera (GUI camera): simulator.get_render_stack()
- envCamera  (env/perception camera): simulator.get_render_stack_perception()

Outputs:
  - mainCamera.mp4
  - envCamera.mp4
  (optional) mainCamera.gif / envCamera.gif

Requirements:
  pip install imageio pillow imageio-ffmpeg numpy

Example:
  python test/uitb_bimanual_eval_multiview.py \
    --simulator_dir ./ \
    --episodes 1 \
    --max_steps 2000 \
    --record_gif
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image

# 这里按你项目的实际导入路径调整：
# - 如果你是在 UITB 主仓库里：from uitb import Simulator
# - 如果你是在 simulator 独立包里：from simulator import Simulator
try:
    from uitb import Simulator  # type: ignore
except Exception:
    from simulator import Simulator  # type: ignore


def write_mp4(frames: list[np.ndarray], out_path: Path, fps: int = 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(out_path), fps=fps)
    try:
        for f in frames:
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            if f.ndim == 2:
                f = np.stack([f] * 3, axis=-1)
            if f.shape[-1] == 4:
                f = f[..., :3]
            w.append_data(f)
    finally:
        w.close()


def write_gif(frames: list[np.ndarray], out_path: Path, fps: int = 15, width: int = 640) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(out_path), mode="I", fps=fps, loop=0)
    try:
        for f in frames:
            f = f[..., :3] if f.shape[-1] == 4 else f
            im = Image.fromarray(f.astype(np.uint8))
            ow, oh = im.size
            if width and ow != width:
                nh = int(oh * width / ow)
                im = im.resize((width, nh), resample=Image.Resampling.LANCZOS)
            w.append_data(np.array(im))
    finally:
        w.close()


def pop_all_rendered_frames(sim) -> list[np.ndarray]:
    """Take and clear GUI render stack (main camera)."""
    try:
        frames = sim.get_render_stack()
        sim.clear_render_stack()
        return list(frames) if frames is not None else []
    except Exception:
        return []


def pop_all_perception_frames(sim) -> list[np.ndarray]:
    """Take and clear perception render stack (env camera when separate)."""
    try:
        frames = sim.get_render_stack_perception()
        sim.clear_render_stack_perception()
        # 有些实现会返回 dict（按 perception module 分开），这里兼容：
        if isinstance(frames, dict):
            # 常见：只有一个相机模块，就拿第一个
            for _, v in frames.items():
                return list(v) if v is not None else []
            return []
        return list(frames) if frames is not None else []
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--simulator_dir", required=True, help="Built simulator folder (contains simulator.py/config).")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--fps", type=int, default=20, help="MP4 fps")
    p.add_argument("--out_dir", default="evaluate", help="Output folder (relative or absolute).")
    p.add_argument("--record_gif", action="store_true", help="Also export GIFs.")
    args = p.parse_args()

    sim_dir = Path(args.simulator_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if Path(args.out_dir).is_absolute() else (sim_dir / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 关键：让环境把帧存到 render stack
    # - render_mode="rgb_array_list": GUI / 主相机帧会进 _render_stack
    # - render_mode_perception="separate": 感知相机帧会进 _render_stack_perception
    sim = Simulator.get(
        str(sim_dir),
        render_mode="rgb_array_list",
        render_mode_perception="separate",
    )

    main_frames: list[np.ndarray] = []
    env_frames: list[np.ndarray] = []

    for ep in range(args.episodes):
        obs, info = sim.reset()

        # reset 后先把可能的首帧取出来
        main_frames += pop_all_rendered_frames(sim)
        env_frames += pop_all_perception_frames(sim)

        for _ in range(args.max_steps):
            # 这里用随机动作；如果你要接 PPO/SB3，把 action 换成 model.predict(obs)
            action = sim.action_space.sample()

            obs, reward, terminated, truncated, info = sim.step(action)

            # 每一步把两路相机帧取出来（可能每步 0~n 帧，取决于你的实现）
            main_frames += pop_all_rendered_frames(sim)
            env_frames += pop_all_perception_frames(sim)

            if terminated or truncated:
                break

    # 输出命名（和你之前一致）
    main_mp4 = out_dir / "mainCamera.mp4"
    env_mp4 = out_dir / "envCamera.mp4"

    if main_frames:
        write_mp4(main_frames, main_mp4, fps=args.fps)
        print(f"[OK] mainCamera video: {main_mp4}")
    else:
        print("[WARN] mainCamera frames empty. (GUI render stack not populated)")

    if env_frames:
        write_mp4(env_frames, env_mp4, fps=args.fps)
        print(f"[OK] envCamera video: {env_mp4}")
    else:
        print("[WARN] envCamera frames empty. (perception separate render stack not populated)")

    # 可选：导出 GIF（便于 PR/报告）
    if args.record_gif:
        if main_frames:
            write_gif(main_frames, out_dir / "beatsvr_bimanual_eval_gui_camera.gif", fps=15, width=640)
        if env_frames:
            write_gif(env_frames, out_dir / "beatsvr_bimanual_eval_env_camera.gif", fps=15, width=640)
        print(f"[OK] gifs saved under: {out_dir}")


if __name__ == "__main__":
    main()
