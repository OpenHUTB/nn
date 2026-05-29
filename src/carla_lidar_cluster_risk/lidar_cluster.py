"""CARLA LiDAR obstacle clustering and risk ranking."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).with_name("sample_data") / "carla_lidar_points.csv"

def generate_sample(path: Path = DATA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(426)
    centers = np.array([[10, 0.6], [18, -2.3], [7, 3.2], [26, 0.2]])
    rows = []
    for cid, c in enumerate(centers):
        pts = c + rng.normal(0, [0.9, 0.35], size=(24, 2))
        for x, y in pts:
            rows.append((round(float(x), 3), round(float(y), 3), cid))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["x_m","y_m","source_cluster"]); w.writerows(rows)

def load_points(path: Path = DATA) -> np.ndarray:
    if not path.exists(): generate_sample(path)
    with path.open(newline="", encoding="utf-8") as f:
        return np.array([[float(r["x_m"]), float(r["y_m"])] for r in csv.DictReader(f)])

def cluster(points: np.ndarray, radius: float = 1.8) -> list[np.ndarray]:
    remaining = set(range(len(points))); clusters = []
    while remaining:
        seed = remaining.pop(); group = {seed}; frontier = [seed]
        while frontier:
            i = frontier.pop()
            near = [j for j in list(remaining) if np.linalg.norm(points[i] - points[j]) <= radius]
            for j in near:
                remaining.remove(j); group.add(j); frontier.append(j)
        clusters.append(points[sorted(group)])
    return clusters

def analyze(points: np.ndarray) -> list[dict[str, float | int | str]]:
    rows = []
    for i, pts in enumerate(cluster(points)):
        center = pts.mean(axis=0); distance = float(np.linalg.norm(center)); lateral = abs(float(center[1]))
        risk = max(0, (30 - distance) / 30) * max(0, 1 - lateral / 4) * min(len(pts) / 20, 1.4)
        label = "danger" if risk > .50 else "watch" if risk > .25 else "safe"
        rows.append({"cluster_id": i, "points": len(pts), "center_x_m": round(float(center[0]),3), "center_y_m": round(float(center[1]),3), "distance_m": round(distance,3), "risk_score": round(float(risk),4), "risk_level": label})
    return sorted(rows, key=lambda r: float(r["risk_score"]), reverse=True)

def plot(points: np.ndarray, rows: list[dict[str, float | int | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True); paths=[]
    p=output/"carla_lidar_clusters.png"; plt.figure(figsize=(7.5,5.5)); plt.scatter(points[:,1], points[:,0], s=24, alpha=.65)
    for r in rows: plt.text(float(r["center_y_m"]), float(r["center_x_m"]), f'{r["cluster_id"]}:{r["risk_level"]}', ha="center")
    plt.xlabel("lateral y (m)"); plt.ylabel("forward x (m)"); plt.title("CARLA LiDAR obstacle clusters"); plt.grid(True, linestyle="--", alpha=.3); plt.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p)
    p=output/"carla_lidar_risk_rank.png"; plt.figure(figsize=(7,4.6)); plt.bar([str(r["cluster_id"]) for r in rows], [float(r["risk_score"]) for r in rows], color="#eb5757"); plt.xlabel("cluster id"); plt.ylabel("risk"); plt.title("Obstacle cluster risk ranking"); plt.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p)
    return paths

def run(output: Path) -> dict[str, object]:
    points=load_points(); rows=analyze(points); files=plot(points,rows,output)
    csv_path=output/"carla_lidar_cluster_risk.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f: w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    files.append(csv_path)
    report={"source":"CARLA LiDAR point cloud sample","points":int(len(points)),"clusters":len(rows),"danger_clusters":sum(r["risk_level"]=="danger" for r in rows),"max_risk":max(float(r["risk_score"]) for r in rows),"generated_files":[p.name for p in files]}
    (output/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); return report

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("docs/pr_assets/carla_lidar_cluster_risk")); args=parser.parse_args(); print(json.dumps(run(args.output),indent=2,ensure_ascii=False))
if __name__=="__main__": main()
