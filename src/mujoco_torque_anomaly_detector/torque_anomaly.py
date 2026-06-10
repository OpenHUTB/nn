"""MuJoCo robotic arm torque anomaly detector."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).with_name("sample_data") / "mujoco_arm_torque.csv"

def load_rows(path: Path = DATA) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]

def detect(rows: list[dict[str, float]]) -> list[dict[str, float | str]]:
    scored = []
    for r in rows:
        torque_norm = np.linalg.norm([r["shoulder_torque"], r["elbow_torque"], r["wrist_torque"]])
        vel_norm = np.linalg.norm([r["shoulder_vel"], r["elbow_vel"], r["wrist_vel"]])
        score = 0.46 * min(torque_norm / 78, 1.4) + 0.30 * min(vel_norm / 3.4, 1.3) + 0.24 * min(r["end_effector_error_m"] / 0.30, 1.5)
        label = "critical" if score > 0.82 else "warning" if score > 0.58 else "normal"
        scored.append({**r, "torque_norm": round(float(torque_norm), 3), "velocity_norm": round(float(vel_norm), 3), "anomaly_score": round(float(score), 4), "anomaly_level": label})
    return scored

def plot(scored: list[dict[str, float | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    t = [float(r["time_s"]) for r in scored]; score = [float(r["anomaly_score"]) for r in scored]
    torque = [float(r["torque_norm"]) for r in scored]; err = [float(r["end_effector_error_m"]) for r in scored]
    paths = []
    p = output / "mujoco_torque_anomaly_curve.png"
    plt.figure(figsize=(8,4.8)); plt.plot(t, score, marker="o", color="#eb5757"); plt.axhline(0.82, linestyle="--", color="#111827"); plt.xlabel("time (s)"); plt.ylabel("anomaly score"); plt.title("MuJoCo torque anomaly score"); plt.grid(True, linestyle="--", alpha=.3); plt.tight_layout(); plt.savefig(p, dpi=180); plt.close(); paths.append(p)
    p = output / "mujoco_torque_error_map.png"
    plt.figure(figsize=(7.2,5)); plt.scatter(torque, err, c=score, cmap="inferno", s=90); plt.colorbar(label="anomaly score"); plt.xlabel("torque norm"); plt.ylabel("end-effector error (m)"); plt.title("Torque-error anomaly map"); plt.grid(True, linestyle="--", alpha=.3); plt.tight_layout(); plt.savefig(p, dpi=180); plt.close(); paths.append(p)
    return paths

def run(output: Path) -> dict[str, object]:
    scored = detect(load_rows()); files = plot(scored, output)
    csv_path = output / "mujoco_torque_anomaly_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys())); writer.writeheader(); writer.writerows(scored)
    files.append(csv_path)
    scores = np.array([float(r["anomaly_score"]) for r in scored])
    report = {"source": "MuJoCo robotic arm torque log", "records": len(scored), "max_anomaly_score": round(float(scores.max()),4), "critical_frames": sum(r["anomaly_level"]=="critical" for r in scored), "generated_files": [p.name for p in files]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"); return report

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("docs/pr_assets/mujoco_torque_anomaly_detector")); args = parser.parse_args(); print(json.dumps(run(args.output), indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
