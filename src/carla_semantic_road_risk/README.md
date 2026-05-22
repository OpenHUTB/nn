# CARLA 语义分割道路占用风险分析模块

本项目使用 CARLA `SemanticSegmentation` 相机导出的像素标签矩阵，分析道路可通行区域、车辆/行人占用和近场风险，并生成驾驶动作建议。

## 主要内容

- 使用 CARLA 语义分割相机标签数据 `carla_semantic_segmentation.csv`。
- 统计 road、lane_marking、vehicle、pedestrian 等语义像素。
- 计算可通行区域比例、近场障碍像素和综合风险分数。
- 输出 keep_speed / watch / slow_down 三类建议。
- 生成 CARLA 语义相机运行视图和风险热力图。

## 环境要求

- Python 3.7+
- NumPy
- pandas
- matplotlib
- OpenCV

## 数据字段说明

使用 CARLA `SemanticSegmentation` 相机导出的像素标签矩阵，包含 road、lane_marking、vehicle、pedestrian 等语义类别标签。

## 运行

```bash
python src/carla_semantic_road_risk/semantic_road_risk.py --output docs/pr_assets/carla_semantic_road_risk
python src/carla_semantic_road_risk/tests/test_semantic_road_risk.py
```
