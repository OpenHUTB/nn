# CARLA LiDAR 障碍物聚类与风险排序模块

本项目使用 CARLA 风格 LiDAR 点云样例，对前方障碍物进行半径连通聚类，并依据距离、横向位置和点数密度计算危险程度。

## 主要内容
- 读取或生成 CARLA LiDAR 点云数据。
- 实现轻量级半径聚类算法。
- 计算每个障碍物簇的中心、距离和风险等级。
- 生成点云聚类图和障碍物风险排序图。

## 运行
```bash
python src/carla_lidar_cluster_risk/lidar_cluster.py --output docs/pr_assets/carla_lidar_cluster_risk
python src/carla_lidar_cluster_risk/tests/test_lidar_cluster.py
```
