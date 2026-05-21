from __future__ import annotations
import sys, tempfile
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT))
from lidar_cluster import analyze, load_points, run
def test_clusters() -> None:
    rows=analyze(load_points()); assert len(rows)>=3; assert rows[0]["risk_level"]=="danger"
def test_exports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        m=run(Path(tmp)); assert m["danger_clusters"]>0; assert (Path(tmp)/"carla_lidar_clusters.png").exists()
if __name__=="__main__": test_clusters(); test_exports(); print("carla_lidar_cluster_risk tests passed")
