# AirSim 深度图安全降落点选择模块

本项目使用 AirSim `DepthPerspective` 深度相机数据和 AirSim recorder 飞行记录，评估候选降落区域的净空、平整度和坡度，自动选择安全降落点。新增的 `airsim_settings.json` 记录了 AirSim 多旋翼与深度相机配置，`airsim_recorder_landing.csv` 记录了无人机下降过程中的位置、姿态和深度相机类型。

## 主要内容
- 读取 AirSim 深度图网格数据和 AirSim recorder 飞行轨迹。
- 使用滑动窗口评估候选降落区域。
- 计算 clearance、flatness、slope 和 landing_score。
- 输出 best / safe / reject 三类决策。
- 生成深度网格选点图、候选区域得分排序图、AirSim 下降轨迹回放图和 AirSim 深度相机运行效果图。

## 运行

```bash
python src/airsim_landing_zone_selector/landing_zone.py --output docs/pr_assets/airsim_landing_zone_selector
python src/airsim_landing_zone_selector/tests/test_landing_zone.py
```
