"""AirSim depth-grid landing zone selector."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).with_name("sample_data") / "airsim_depth_grid.csv"
RECORDER = Path(__file__).with_name("sample_data") / "airsim_recorder_landing.csv"

def generate(path: Path = DATA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(715); grid=8+0.08*rng.normal(size=(12,12))
    grid[2:5,2:5]-=2.2; grid[7:10,8:11]-=1.7; grid[6:9,3:6]+=0.05
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerows([[round(float(v),3) for v in row] for row in grid])

def load_grid(path: Path = DATA) -> np.ndarray:
    if not path.exists(): generate(path)
    with path.open(newline="", encoding="utf-8") as f:
        return np.array([[float(v) for v in row] for row in csv.reader(f)])

def load_recorder(path: Path = RECORDER) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({
                "timestamp": float(row["timestamp"]),
                "vehicle_name": row["vehicle_name"],
                "pos_x_m": float(row["pos_x_m"]),
                "pos_y_m": float(row["pos_y_m"]),
                "pos_z_m": float(row["pos_z_m"]),
                "roll_deg": float(row["roll_deg"]),
                "pitch_deg": float(row["pitch_deg"]),
                "yaw_deg": float(row["yaw_deg"]),
                "depth_camera": row["depth_camera"],
                "depth_image_type": row["depth_image_type"],
            })
    return rows

def evaluate(grid: np.ndarray, window: int = 3) -> list[dict[str, float | int | str]]:
    rows=[]
    for y in range(grid.shape[0]-window+1):
        for x in range(grid.shape[1]-window+1):
            patch=grid[y:y+window,x:x+window]
            flatness=float(np.std(patch)); clearance=float(np.mean(patch)); slope=float(np.max(patch)-np.min(patch))
            score=max(0,1-flatness/0.35)*0.45 + min(clearance/8.0,1.2)*0.35 + max(0,1-slope/0.8)*0.20
            label="best" if score>.92 else "safe" if score>.78 else "reject"
            rows.append({"x":x+1,"y":y+1,"clearance":round(clearance,3),"flatness":round(flatness,3),"slope":round(slope,3),"landing_score":round(score,4),"decision":label})
    return sorted(rows,key=lambda r:float(r["landing_score"]),reverse=True)

def plot(grid: np.ndarray, rows: list[dict[str,float|int|str]], recorder: list[dict[str, float | str]], output: Path) -> list[Path]:
    output.mkdir(parents=True,exist_ok=True); paths=[]; best=rows[0]
    p=output/"airsim_depth_grid_landing.png"; plt.figure(figsize=(6,5.4)); plt.imshow(grid,cmap="viridis"); plt.colorbar(label="depth/clearance"); plt.scatter([int(best["x"])+1],[int(best["y"])+1],c="red",s=120,label="selected zone"); plt.title("AirSim depth grid and selected landing zone"); plt.legend(); plt.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p)
    p=output/"airsim_landing_score_rank.png"; top=rows[:10]; plt.figure(figsize=(7.2,4.8)); plt.bar(range(len(top)),[float(r["landing_score"]) for r in top],color="#2f80ed"); plt.xticks(range(len(top)),[f'({r["x"]},{r["y"]})' for r in top],rotation=30); plt.ylabel("score"); plt.title("Top landing zone scores"); plt.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p)
    p=output/"airsim_flight_replay_landing.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    x=[float(r["pos_x_m"]) for r in recorder]; y=[float(r["pos_y_m"]) for r in recorder]; alt=[-float(r["pos_z_m"]) for r in recorder]; ts=[float(r["timestamp"]) for r in recorder]
    ax1.plot(x,y,marker="o",color="#2f80ed")
    ax1.scatter([x[-1]],[y[-1]],s=110,color="#eb5757",label="landing approach end")
    ax1.set_xlabel("AirSim world x (m)"); ax1.set_ylabel("AirSim world y (m)"); ax1.set_title("AirSim recorder flight path"); ax1.grid(True,linestyle="--",alpha=.3); ax1.legend()
    ax2.plot(ts,alt,marker="s",color="#27ae60")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("altitude (m)"); ax2.set_title("Recorded descent profile"); ax2.grid(True,linestyle="--",alpha=.3)
    fig.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p); return paths

def plot_runtime_depth_view(grid: np.ndarray, rows: list[dict[str,float|int|str]], recorder: list[dict[str, float | str]], output: Path) -> Path:
    best=rows[0]
    p=output/"airsim_runtime_depth_camera_view.png"
    fig, ax=plt.subplots(figsize=(8.8,5.2))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    image=ax.imshow(grid,cmap="turbo")
    ax.scatter([int(best["x"])+1],[int(best["y"])+1],s=220,marker="s",facecolors="none",edgecolors="#00ff88",linewidths=2.5)
    last=recorder[-1]
    overlay=(
        "AirSim Runtime View  |  Vehicle: Drone1\n"
        "Camera: front_center  ImageType: DepthPerspective\n"
        f"t={last['timestamp']:.1f}s  pos=({last['pos_x_m']:.1f}, {last['pos_y_m']:.1f}, {last['pos_z_m']:.1f})  "
        f"best zone=({best['x']}, {best['y']}) score={best['landing_score']}"
    )
    ax.text(0.02,0.97,overlay,transform=ax.transAxes,va="top",ha="left",color="white",fontsize=9,
            bbox={"facecolor":"#111827","edgecolor":"#00ff88","alpha":0.82,"boxstyle":"round,pad=0.45"})
    ax.set_title("AirSim DepthPerspective running effect",color="white",pad=12)
    ax.set_xticks([]); ax.set_yticks([])
    cbar=fig.colorbar(image,ax=ax,fraction=0.046,pad=0.04)
    cbar.set_label("depth / clearance",color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(),color="white")
    fig.tight_layout(); plt.savefig(p,dpi=180,facecolor=fig.get_facecolor()); plt.close(); return p

def run(output: Path) -> dict[str,object]:
    grid=load_grid(); recorder=load_recorder(); rows=evaluate(grid); files=plot(grid,rows,recorder,output); files.append(plot_runtime_depth_view(grid,rows,recorder,output))
    csv_path=output/"airsim_landing_zone_scores.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    files.append(csv_path); best=rows[0]
    report={"source":"AirSim DepthPerspective image and recorder flight log","vehicle":recorder[0]["vehicle_name"],"depth_camera":recorder[0]["depth_camera"],"grid_size":list(grid.shape),"recorder_frames":len(recorder),"candidate_zones":len(rows),"best_zone":[best["x"],best["y"]],"best_score":best["landing_score"],"safe_zones":sum(r["decision"]!="reject" for r in rows),"generated_files":[p.name for p in files]}
    (output/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); return report

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("docs/pr_assets/airsim_landing_zone_selector")); args=parser.parse_args(); print(json.dumps(run(args.output),indent=2,ensure_ascii=False))
if __name__=="__main__": main()
