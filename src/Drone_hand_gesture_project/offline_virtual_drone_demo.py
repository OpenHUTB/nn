import os
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_controller import DroneController


GESTURE_TO_COMMAND = {
    "open_palm": "takeoff",
    "victory": "forward",
    "pointing_up": "up",
    "thumb_up": "backward",
    "ok_sign": "hover",
    "pointing_down": "down",
    "closed_fist": "land",
    "thumb_down": "stop",
}


@dataclass
class GestureEvent:
    gesture: str
    duration: float
    intensity: float
    confidence: float


def get_demo_sequence():
    return [
        GestureEvent("open_palm", 2.8, 0.85, 0.94),
        GestureEvent("victory", 2.4, 0.78, 0.89),
        GestureEvent("pointing_up", 1.8, 0.72, 0.86),
        GestureEvent("ok_sign", 1.4, 0.92, 0.91),
        GestureEvent("thumb_up", 2.0, 0.68, 0.83),
        GestureEvent("pointing_down", 1.6, 0.60, 0.80),
        GestureEvent("victory", 1.8, 0.74, 0.87),
        GestureEvent("closed_fist", 2.6, 0.88, 0.93),
    ]


def run_offline_demo():
    controller = DroneController(simulation_mode=True)
    dt = 0.05
    sequence = get_demo_sequence()

    time_points = []
    positions = []
    batteries = []
    modes = []
    confidences = []
    gesture_names = []
    command_names = []
    command_spans = []

    current_time = 0.0
    for event in sequence:
        command = GESTURE_TO_COMMAND[event.gesture]
        controller.send_command(command, event.intensity)

        span_start = current_time
        steps = max(1, int(event.duration / dt))
        for _ in range(steps):
            controller.update_physics(dt)
            state = controller.get_state()

            time_points.append(current_time)
            positions.append(state["position"].copy())
            batteries.append(state["battery"])
            modes.append(state["mode"])
            confidences.append(event.confidence)
            gesture_names.append(event.gesture)
            command_names.append(command)
            current_time += dt
        command_spans.append((span_start, current_time, event.gesture, command))

        if command in {"forward", "backward", "up", "down"}:
            controller.send_command("hover", 1.0)
            for _ in range(int(0.8 / dt)):
                controller.update_physics(dt)
                state = controller.get_state()
                time_points.append(current_time)
                positions.append(state["position"].copy())
                batteries.append(state["battery"])
                modes.append(state["mode"])
                confidences.append(0.76)
                gesture_names.append("ok_sign")
                command_names.append("hover")
                current_time += dt

    return {
        "time": np.asarray(time_points, dtype=np.float32),
        "positions": np.asarray(positions, dtype=np.float32),
        "batteries": np.asarray(batteries, dtype=np.float32),
        "modes": modes,
        "confidences": np.asarray(confidences, dtype=np.float32),
        "gesture_names": gesture_names,
        "command_names": command_names,
        "spans": command_spans,
        "trajectory": controller.get_trajectory(),
    }


def save_offline_report(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    result = run_offline_demo()

    time_axis = result["time"]
    positions = result["positions"]
    batteries = result["batteries"]
    confidences = result["confidences"]
    trajectory = np.asarray(result["trajectory"], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    axes[0, 0].plot(time_axis, positions[:, 0], label="X", color="#0984e3")
    axes[0, 0].plot(time_axis, positions[:, 1], label="Y / Altitude", color="#00b894")
    axes[0, 0].plot(time_axis, positions[:, 2], label="Z", color="#d63031")
    axes[0, 0].set_title("Drone Position Timeline", weight="bold")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Position (m)")
    axes[0, 0].legend()

    if len(trajectory) > 0:
        axes[0, 1].plot(trajectory[:, 0], trajectory[:, 2], color="#6c5ce7", linewidth=2.0)
        axes[0, 1].scatter(trajectory[0, 0], trajectory[0, 2], color="#00cec9", s=70, label="Start")
        axes[0, 1].scatter(trajectory[-1, 0], trajectory[-1, 2], color="#d63031", s=70, label="End")
    axes[0, 1].set_title("Top-View Flight Trajectory", weight="bold")
    axes[0, 1].set_xlabel("X (m)")
    axes[0, 1].set_ylabel("Z (m)")
    axes[0, 1].legend()
    axes[0, 1].axis("equal")

    axes[1, 0].plot(time_axis, batteries, color="#fdcb6e", linewidth=2.2, label="Battery")
    axes[1, 0].plot(time_axis, confidences * 100, color="#e17055", linestyle="--", linewidth=1.8, label="Gesture confidence x100")
    axes[1, 0].set_title("Battery and Gesture Confidence", weight="bold")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend()

    mode_to_id = {mode: idx for idx, mode in enumerate(sorted(set(result["modes"])))}
    mode_ids = np.asarray([mode_to_id[m] for m in result["modes"]], dtype=np.float32)
    axes[1, 1].plot(time_axis, mode_ids, color="#2d3436", linewidth=1.8)
    axes[1, 1].set_yticks(list(mode_to_id.values()))
    axes[1, 1].set_yticklabels(list(mode_to_id.keys()))
    axes[1, 1].set_title("Drone Mode Transition", weight="bold")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Mode")

    fig.suptitle("Offline Gesture-to-Drone Virtual Simulation Report", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    report_path = os.path.join(output_dir, "offline_drone_simulation_report.png")
    fig.savefig(report_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(15, 4.8))
    palette = {
        "open_palm": "#55efc4",
        "victory": "#74b9ff",
        "pointing_up": "#ffeaa7",
        "thumb_up": "#fab1a0",
        "ok_sign": "#a29bfe",
        "pointing_down": "#fd79a8",
        "closed_fist": "#636e72",
    }
    for idx, (start, end, gesture, command) in enumerate(result["spans"]):
        ax2.barh(0, end - start, left=start, height=0.45, color=palette.get(gesture, "#b2bec3"), edgecolor="white")
        ax2.text((start + end) / 2, 0, f"{gesture}\n{command}", ha="center", va="center", fontsize=9, color="black")
    ax2.set_title("Gesture Command Timeline", weight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_yticks([])
    ax2.set_xlim(0, max(time_axis) + 0.5 if len(time_axis) else 1)
    fig2.tight_layout()
    timeline_path = os.path.join(output_dir, "offline_gesture_command_timeline.png")
    fig2.savefig(timeline_path, dpi=220, bbox_inches="tight")
    plt.close(fig2)

    return report_path, timeline_path


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "offline_demo_reports")
    report_path, timeline_path = save_offline_report(output_dir)
    print(f"✅ 离线无人机仿真报告已保存: {report_path}")
    print(f"✅ 离线手势命令时间线已保存: {timeline_path}")


if __name__ == "__main__":
    main()
