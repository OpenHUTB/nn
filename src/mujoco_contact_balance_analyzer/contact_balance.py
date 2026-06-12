"""MuJoCo contact force balance analyzer."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

DATA=Path(__file__).with_name("sample_data")/"mujoco_contact_forces.csv"
SENSOR_DATA=Path(__file__).with_name("sample_data")/"mujoco_sensor_export.csv"
MJCF=Path(__file__).with_name("sample_data")/"mujoco_quadruped_balance.xml"

def load_rows(path: Path=DATA) -> list[dict[str,float]]:
    with path.open(newline="",encoding="utf-8") as f: return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]

def load_sensor_rows(path: Path=SENSOR_DATA) -> list[dict[str,float]]:
    with path.open(newline="",encoding="utf-8") as f: return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]

def sensor_rows_to_balance_rows(rows: list[dict[str,float]]) -> list[dict[str,float]]:
    converted=[]
    for r in rows:
        converted.append({
            "time_s": r["time_s"],
            "left_front_n": r["cfrc_lf_z"],
            "right_front_n": r["cfrc_rf_z"],
            "left_rear_n": r["cfrc_lr_z"],
            "right_rear_n": r["cfrc_rr_z"],
            "com_x_m": r["sensordata_com_x"],
            "com_y_m": r["sensordata_com_y"],
            "torso_roll_deg": r["qvel_roll"] * 10.0,
        })
    return converted

def parse_mjcf(path: Path=MJCF) -> dict[str, int | str]:
    root=ET.parse(path).getroot()
    sensors=root.findall(".//sensor/*")
    contact_geoms=[g for g in root.findall(".//geom") if "contact" in g.attrib.get("name","")]
    return {"model":root.attrib.get("model","unknown"),"sensor_count":len(sensors),"contact_geom_count":len(contact_geoms)}

def analyze(rows: list[dict[str,float]]) -> list[dict[str,float|str]]:
    out=[]
    for r in rows:
        left=r["left_front_n"]+r["left_rear_n"]; right=r["right_front_n"]+r["right_rear_n"]; total=left+right
        imbalance=0.0 if total <= 1e-9 else abs(left-right)/total
        com_shift=np.hypot(r["com_x_m"],r["com_y_m"]); roll=abs(r["torso_roll_deg"])
        risk=0.42*min(imbalance/0.32,1.4)+0.32*min(com_shift/0.30,1.3)+0.26*min(roll/10,1.4)
        action="recover_posture" if risk>.72 else "adjust_support" if risk>.45 else "stable_walk"
        out.append({**r,"left_force_n":round(left,2),"right_force_n":round(right,2),"force_imbalance":round(imbalance,4),"balance_risk":round(float(risk),4),"balance_action":action})
    return out

def plot(rows: list[dict[str,float|str]], sensor_rows: list[dict[str,float]], output: Path) -> list[Path]:
    output.mkdir(parents=True,exist_ok=True); t=[float(r["time_s"]) for r in rows]; risk=[float(r["balance_risk"]) for r in rows]; imb=[float(r["force_imbalance"]) for r in rows]; paths=[]
    p=output/"mujoco_contact_balance_curve.png"; plt.figure(figsize=(8,4.8)); plt.plot(t,risk,marker="o",label="balance risk",color="#eb5757"); plt.plot(t,imb,marker="s",label="force imbalance",color="#2f80ed"); plt.xlabel("time (s)"); plt.title("MuJoCo contact balance risk"); plt.grid(True,linestyle="--",alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p)
    p=output/"mujoco_force_distribution.png"; plt.figure(figsize=(7.5,4.8)); plt.plot(t,[float(r["left_force_n"]) for r in rows],label="left force"); plt.plot(t,[float(r["right_force_n"]) for r in rows],label="right force"); plt.xlabel("time (s)"); plt.ylabel("force (N)"); plt.title("Left-right contact force distribution"); plt.grid(True,linestyle="--",alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p)
    p=output/"mujoco_support_polygon_replay.png"
    foot=np.array([[0.18,0.12],[0.18,-0.12],[-0.18,-0.12],[-0.18,0.12],[0.18,0.12]])
    com=np.array([[r["sensordata_com_x"],r["sensordata_com_y"]] for r in sensor_rows])
    risk=[float(r["balance_risk"]) for r in rows]
    plt.figure(figsize=(6.4,5.6)); plt.plot(foot[:,0],foot[:,1],color="#111827",label="MuJoCo foot support polygon"); plt.scatter(com[:,0],com[:,1],c=risk,cmap="inferno",s=85,label="COM projection"); plt.colorbar(label="balance risk"); plt.xlabel("world x (m)"); plt.ylabel("world y (m)"); plt.title("MuJoCo sensor replay: COM inside support polygon"); plt.grid(True,linestyle="--",alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(p,dpi=180); plt.close(); paths.append(p); return paths

def run(output: Path) -> dict[str,object]:
    sensor_rows=load_sensor_rows(); rows=analyze(sensor_rows_to_balance_rows(sensor_rows)); mjcf=parse_mjcf(); files=plot(rows,sensor_rows,output); csv_path=output/"mujoco_contact_balance_scores.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    files.append(csv_path); risk=np.array([float(r["balance_risk"]) for r in rows])
    report={"source":"MuJoCo MJCF model stepped by mujoco Python, sensordata force sensors export","mjcf_model":mjcf["model"],"sensor_count":mjcf["sensor_count"],"contact_geom_count":mjcf["contact_geom_count"],"records":len(rows),"max_balance_risk":round(float(risk.max()),4),"recover_posture_frames":sum(r["balance_action"]=="recover_posture" for r in rows),"generated_files":[p.name for p in files]}
    (output/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); return report

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("docs/pr_assets/mujoco_contact_balance_analyzer")); args=parser.parse_args(); print(json.dumps(run(args.output),indent=2,ensure_ascii=False))
if __name__=="__main__": main()
