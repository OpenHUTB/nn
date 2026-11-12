"""Command line interface: collect / train / eval / benchmark / demo / slam."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .dataset import CollectConfig, collect_dataset, load_dataset, save_dataset
from .evaluate import EvalResult, rollout
from .expert import PurePursuitExpert
from .models import PolicyRunner, build_model
from .sim import IMAGE_SIZE, CarSim, SimConfig
from .slam import GridMap, map_iou, run_slam
from .track import build_track
from .train import TrainConfig, dagger_train, train_policy
from .viz import MapView, composite_demo, render_map, save_gif

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
RUN_DIR = os.path.join(ROOT, "runs")
FIG_DIR = os.path.join(ROOT, "docs", "figures")
GIF_DIR = os.path.join(ROOT, "docs", "gifs")


def _ensure(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def cmd_collect(args) -> None:
    track = build_track()
    scenes = tuple(args.scenes.split(","))
    cfg = CollectConfig(laps=args.laps, scenes=scenes, seed=args.seed)
    t0 = time.time()

    def progress(steps, laps):
        print(f"\r  collecting: {steps} steps, {laps:.2f} laps", end="", flush=True)

    data = collect_dataset(track, cfg, progress=progress)
    print()
    out = args.out or os.path.join(DATA_DIR, "dataset.npz")
    save_dataset(out, data)
    print(f"saved {len(data['images'])} samples ({len(data['scenes'])} frames, scenes={scenes}) -> {out}")
    print(f"size {os.path.getsize(out) / 1e6:.1f} MB, took {time.time() - t0:.1f} s")


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
def _load_data(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"dataset not found: {path}. Run `python main.py collect` first.")
    return load_dataset(path)


def cmd_train(args) -> None:
    track = build_track()
    data = _load_data(args.data)
    cfg = TrainConfig(model=args.model, epochs=args.epochs, seed=args.seed)
    _ensure(RUN_DIR)

    if args.dagger:
        net, data, logs = dagger_train(track, data, cfg, iterations=args.dagger_iters, verbose=True)
        suffix = "dagger"
    else:
        net, logs = train_policy(data, cfg, verbose=True)
        suffix = "bc"

    import torch

    out = args.out or os.path.join(RUN_DIR, f"{args.model}_{suffix}.pt")
    torch.save({"model": args.model, "state_dict": net.state_dict(), "config": vars(cfg)}, out)
    hist_path = os.path.splitext(out)[0] + "_history.json"
    with open(hist_path, "w", encoding="utf-8") as fh:
        json.dump(logs, fh, indent=2, default=float)
    print(f"model saved to {out}")
    print(f"history saved to {hist_path}")


def load_policy(path: str) -> PolicyRunner:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = build_model(ckpt["model"], state_dim=int(ckpt.get("state_dim", 1)))
    net.load_state_dict(ckpt["state_dict"])
    return PolicyRunner(net)


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def _print_result(res: EvalResult) -> None:
    print(
        f"{res.policy:14s} {res.scene:5s} laps={res.laps_completed:4.2f} "
        f"success={'Y' if res.success else 'N'} coll={res.collisions:2d} "
        f"lat_rmse={res.lateral_rmse:4.2f} lat_max={res.lateral_max:4.2f} "
        f"v_mean={res.speed_mean:5.2f} lap={res.lap_time:5.1f}s infer={res.inference_ms:4.1f}ms"
    )


def cmd_eval(args) -> None:
    track = build_track()
    policy = None if args.expert else load_policy(args.model)
    name = "expert" if args.expert else os.path.basename(args.model)
    res, frames = rollout(
        policy,
        track,
        scene=args.scene,
        laps=args.laps,
        seed=args.seed,
        policy_name=name,
        record=args.record,
        max_steps=args.max_steps,
    )
    _print_result(res)
    if args.record and frames:
        _ensure(GIF_DIR)
        path = args.gif or os.path.join(GIF_DIR, f"eval_{name}_{args.scene}.gif")
        size = save_gif(frames, path, fps=10, colors=64)
        print(f"gif saved to {path} ({size / 1e6:.2f} MB, {len(frames)} frames)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res.to_dict(), fh, indent=2)


# ----------------------------------------------------------------------
# benchmark: architecture and algorithm comparison
# ----------------------------------------------------------------------
def cmd_benchmark(args) -> None:
    import torch

    _ensure(FIG_DIR, RUN_DIR, DATA_DIR)
    track = build_track()
    data = _load_data(args.data)

    scenarios = {
        "expert": None,
        "cnn_bc": None,
        "cnn_dagger": None,
        "mlp_bc": None,
    }
    histories = {}

    print("\n[1/4] behaviour cloning (CNN)")
    cnn, hist_cnn = train_policy(data, TrainConfig(model="cnn", epochs=args.epochs), verbose=False)
    torch.save({"model": "cnn", "state_dict": cnn.state_dict()}, os.path.join(RUN_DIR, "cnn_bc.pt"))
    histories["cnn_bc"] = hist_cnn
    scenarios["cnn_bc"] = PolicyRunner(cnn)

    print("[2/4] behaviour cloning (MLP)")
    mlp, hist_mlp = train_policy(data, TrainConfig(model="mlp", epochs=args.epochs), verbose=False)
    torch.save({"model": "mlp", "state_dict": mlp.state_dict()}, os.path.join(RUN_DIR, "mlp_bc.pt"))
    histories["mlp_bc"] = hist_mlp
    scenarios["mlp_bc"] = PolicyRunner(mlp)

    print("[3/4] DAgger (CNN + expert relabelling)")
    cnn_d, data_d, logs_d = dagger_train(
        track, data, TrainConfig(model="cnn", epochs=args.epochs), iterations=args.dagger_iters, verbose=False
    )
    torch.save({"model": "cnn", "state_dict": cnn_d.state_dict()}, os.path.join(RUN_DIR, "cnn_dagger.pt"))
    histories["cnn_dagger"] = logs_d
    scenarios["cnn_dagger"] = PolicyRunner(cnn_d)

    print("[4/4] closed-loop evaluation")
    results = []
    for name, policy in scenarios.items():
        for scene in args.scenes.split(","):
            res, _ = rollout(
                policy,
                track,
                scene=scene,
                laps=args.laps,
                seed=42,
                policy_name=name,
                max_steps=args.max_steps,
            )
            _print_result(res)
            results.append(res.to_dict())

    payload = {"results": results, "histories": histories, "config": {k: v for k, v in vars(args).items() if k != "func"}}
    with open(os.path.join(FIG_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)

    write_markdown_table(results, os.path.join(FIG_DIR, "results_table.md"))
    plot_histories(histories, os.path.join(FIG_DIR, "training_curves.png"))
    plot_comparison(results, os.path.join(FIG_DIR, "comparison.png"))
    print(f"\nsaved tables and figures to {FIG_DIR}")


def write_markdown_table(results: list[dict], path: str) -> None:
    header = (
        "| 策略 | 场景 | 完成圈数 | 成功率 | 碰撞次数 | 横向误差 RMSE (m) | 横向误差最大 (m) | "
        "平均速度 (m/s) | 单圈时间 (s) | 推理耗时 (ms) |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r['policy']} | {r['scene']} | {r['laps_completed']:.2f} | {'是' if r['success'] else '否'} | "
            f"{r['collisions']} | {r['lateral_rmse']:.2f} | {r['lateral_max']:.2f} | {r['speed_mean']:.2f} | "
            f"{r['lap_time']:.1f} | {r['inference_ms']:.2f} |"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n".join(rows) + "\n")


def plot_histories(histories: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, hist in histories.items():
        if name == "cnn_dagger":
            h = hist["dagger"][-1] if isinstance(hist, dict) else None
            if h is None:
                continue
            axes[0].plot(h["train_loss"], label="cnn_dagger(train)")
            axes[1].plot(h["val_loss"], label="cnn_dagger(val)")
        else:
            axes[0].plot(hist["train_loss"], label=f"{name}(train)")
            axes[1].plot(hist["val_loss"], label=f"{name}(val)")
    for ax, title in zip(axes, ("training loss", "validation loss")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_comparison(results: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = sorted({r["scene"] for r in results})
    policies = [p for p in ("expert", "cnn_bc", "cnn_dagger", "mlp_bc") if any(r["policy"] == p for r in results)]
    colors = {"expert": "#4c4c4c", "cnn_bc": "#1f77b4", "cnn_dagger": "#2ca02c", "mlp_bc": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    width = 0.8 / max(len(policies), 1)
    for i, policy in enumerate(policies):
        vals = [next((r["lateral_rmse"] for r in results if r["policy"] == policy and r["scene"] == s), 0) for s in scenes]
        axes[0].bar(np.arange(len(scenes)) + i * width, vals, width, label=policy, color=colors.get(policy))
        vals2 = [next((r["laps_completed"] for r in results if r["policy"] == policy and r["scene"] == s), 0) for s in scenes]
        axes[1].bar(np.arange(len(scenes)) + i * width, vals2, width, label=policy, color=colors.get(policy))
    axes[0].set_xticks(np.arange(len(scenes)) + 0.4 - width / 2, scenes)
    axes[1].set_xticks(np.arange(len(scenes)) + 0.4 - width / 2, scenes)
    axes[0].set_ylabel("lateral RMSE (m)")
    axes[1].set_ylabel("laps completed")
    axes[0].set_title("tracking accuracy")
    axes[1].set_title("completion")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# demo GIF
# ----------------------------------------------------------------------
def cmd_demo(args) -> None:
    track = build_track()
    policy = None if args.expert else load_policy(args.model)
    sim = CarSim(track, SimConfig(scene=args.scene, seed=args.seed, randomize_start=False))
    expert = PurePursuitExpert(track)
    st = sim.reset(args.offset)
    _ensure(GIF_DIR)

    frames = []
    fps = args.fps
    every = max(1, int(round(20 / fps)))
    steps = int(args.seconds * 20)
    for i in range(steps):
        if policy is None:
            action = expert.act(st)
        else:
            action = policy(sim.camera_image(), st.speed)
        if i % every == 0:
            chase = sim.render_chase(640, 360)
            ego = sim.render_front(320, 180)
            lines = [
                f"speed {st.speed:5.2f} m/s",
                f"steer {action[0]:+.2f}  thr {action[1]:+.2f}",
                f"lateral {st.lateral:+.2f} m",
                f"scene {args.scene}",
            ]
            frames.append(composite_demo(chase, ego, lines))
        st = sim.step(np.clip(action, -1, 1))
    path = args.out or os.path.join(GIF_DIR, f"demo_{'expert' if policy is None else 'nn'}_{args.scene}.gif")
    size = save_gif(frames, path, fps=fps, colors=64)
    print(f"gif saved to {path} ({size / 1e6:.2f} MB, {len(frames)} frames)")


# ----------------------------------------------------------------------
# SLAM demo
# ----------------------------------------------------------------------
def cmd_slam(args) -> None:
    track = build_track()
    policy = None if args.expert else load_policy(args.model)
    grid_holder = {}

    def render(grid, traj, pose, st):
        view = MapView(resolution=grid.resolution, origin=(grid.x_min, grid.y_min), width=grid.width, height=grid.height, scale=1)
        img = render_map(
            grid.log_odds,
            view,
            [tuple(p) for p in traj],
            (pose[0], pose[1], pose[2]),
            title=f"occupancy grid  |  steps {len(traj)}  |  scene {args.scene}",
        )
        return img

    metrics, frames, grid = run_slam(
        track,
        policy=policy,
        scene=args.scene,
        laps=args.laps,
        max_steps=args.max_steps,
        record_every=args.every,
        render=render,
    )
    print(
        f"SLAM: {metrics['steps']} steps, {metrics['laps']:.2f} laps, "
        f"ATE RMSE {metrics['ate_rmse']:.2f} m, max {metrics['ate_max']:.2f} m, "
        f"map IoU {map_iou(grid, track):.3f}, free/occupied cells {metrics['map_cells']}"
    )
    _ensure(GIF_DIR)
    if frames:
        path = args.out or os.path.join(GIF_DIR, "slam_mapping.gif")
        size = save_gif(frames, path, fps=args.fps, colors=64)
        print(f"gif saved to {path} ({size / 1e6:.2f} MB, {len(frames)} frames)")
    if args.json:
        payload = {k: v for k, v in metrics.items() if k not in ("trajectory", "estimated")}
        payload["map_iou"] = map_iou(grid, track)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=float)


# ----------------------------------------------------------------------
# keyboard teleoperation
# ----------------------------------------------------------------------
def cmd_teleop(args) -> None:
    import mujoco.viewer

    track = build_track()
    sim = CarSim(track, SimConfig(scene=args.scene, seed=0, randomize_start=False, allow_reverse=True))
    sim.reset(0.0)
    steer, throttle = 0.0, 0.0
    print("keyboard: A/D steer, W/S throttle, Q quit (focus the viewer window)")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            try:
                import msvcrt

                if msvcrt.kbhit():
                    key = msvcrt.getch().decode(errors="ignore").lower()
                    if key == "a":
                        steer = max(-1.0, steer - 0.15)
                    elif key == "d":
                        steer = min(1.0, steer + 0.15)
                    elif key == "w":
                        throttle = min(1.0, throttle + 0.2)
                    elif key == "s":
                        throttle = max(-1.0, throttle - 0.2)
                    elif key == "q":
                        break
                    elif key == " ":
                        steer, throttle = 0.0, 0.0
            except ImportError:  # Linux / macOS
                import select
                import termios
                import tty

                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1).lower()
                    steer = max(-1.0, steer - 0.15) if key == "a" else min(1.0, steer + 0.15) if key == "d" else steer
                    throttle = min(1.0, throttle + 0.2) if key == "w" else max(-1.0, throttle - 0.2) if key == "s" else throttle
                    if key == "q":
                        break
            sim.step([steer, throttle], substeps=1)
            viewer.sync()
            time.sleep(0.005)
    sim.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mujoco_car_nn", description="MuJoCo car + neural network driving demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="collect imitation-learning data with the expert")
    c.add_argument("--laps", type=float, default=5.0)
    c.add_argument("--scenes", type=str, default="day")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", type=str, default="")
    c.set_defaults(func=cmd_collect)

    t = sub.add_parser("train", help="behaviour cloning / DAgger")
    t.add_argument("--model", type=str, default="cnn", choices=["cnn", "mlp"])
    t.add_argument("--data", type=str, default=os.path.join(DATA_DIR, "dataset.npz"))
    t.add_argument("--epochs", type=int, default=24)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--dagger", action="store_true")
    t.add_argument("--dagger-iters", type=int, default=2)
    t.add_argument("--out", type=str, default="")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="closed-loop evaluation")
    e.add_argument("--model", type=str, default=os.path.join(RUN_DIR, "cnn_dagger.pt"))
    e.add_argument("--expert", action="store_true")
    e.add_argument("--scene", type=str, default="day", choices=["day", "dusk", "fog"])
    e.add_argument("--laps", type=float, default=2.0)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--max-steps", type=int, default=3000)
    e.add_argument("--record", action="store_true")
    e.add_argument("--gif", type=str, default="")
    e.add_argument("--json", type=str, default="")
    e.set_defaults(func=cmd_eval)

    b = sub.add_parser("benchmark", help="architecture / algorithm comparison")
    b.add_argument("--data", type=str, default=os.path.join(DATA_DIR, "dataset.npz"))
    b.add_argument("--epochs", type=int, default=20)
    b.add_argument("--dagger-iters", type=int, default=2)
    b.add_argument("--laps", type=float, default=1.0)
    b.add_argument("--scenes", type=str, default="day,dusk,fog")
    b.add_argument("--max-steps", type=int, default=2500)
    b.set_defaults(func=cmd_benchmark)

    d = sub.add_parser("demo", help="record a driving GIF")
    d.add_argument("--model", type=str, default=os.path.join(RUN_DIR, "cnn_dagger.pt"))
    d.add_argument("--expert", action="store_true")
    d.add_argument("--scene", type=str, default="day")
    d.add_argument("--seconds", type=float, default=14.0)
    d.add_argument("--fps", type=int, default=10)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--offset", type=float, default=0.0)
    d.add_argument("--out", type=str, default="")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("slam", help="occupancy mapping + scan matching demo")
    s.add_argument("--model", type=str, default=os.path.join(RUN_DIR, "cnn_dagger.pt"))
    s.add_argument("--expert", action="store_true")
    s.add_argument("--scene", type=str, default="day")
    s.add_argument("--laps", type=float, default=1.0)
    s.add_argument("--max-steps", type=int, default=2000)
    s.add_argument("--every", type=int, default=8)
    s.add_argument("--fps", type=int, default=10)
    s.add_argument("--out", type=str, default="")
    s.add_argument("--json", type=str, default="")
    s.set_defaults(func=cmd_slam)

    k = sub.add_parser("teleop", help="keyboard teleoperation in the MuJoCo viewer")
    k.add_argument("--scene", type=str, default="day")
    k.set_defaults(func=cmd_teleop)

    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ensure(DATA_DIR, RUN_DIR)
    args.func(args)
