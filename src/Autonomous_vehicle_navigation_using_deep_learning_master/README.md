# Autonomous Vehicle Navigation Using Deep Learning

本项目基于深度学习实现自动驾驶汽车在CARLA仿真环境中的导航系统，支持自定义轨迹规划和行人动态模拟。

## 快速开始

### 环境要求
- **操作系统**: Ubuntu 20.04
- **仿真环境**: CARLA 0.9.13
- **Python**: 3.7
- **包管理**: Conda虚拟环境

### 安装步骤

1. **安装依赖包**:
```bash
conda create -n carla-env python=3.7
conda activate carla-env
pip install -r requirements.txt
```

2. **启动CARLA仿真器**:
```bash
./CarlaUE4.sh
```

3. **运行主程序**:
```bash
cd main
python main.py
```

4. **运行测试程序**:
```bash
cd test
python test_driving.py
```
## 项目结构

```
│  README.md
│  requirements.txt
├─agents
│  │  __init__.py
│  ├─navigation
│  │      basic_agent.py
│  │      behavior_agent.py
│  │      behavior_types.py
│  │      controller.py
│  │      global_route_planner.py
│  │      local_planner.py
│  │      __init__.py
│  └─tools
│          misc.py
│          __init__.py
├─main
│      car_env.py
│      config.py
│      config_manager.py
│      get_location.py
│      main.py
│      model_manager.py
│      route_visualizer.py
│      traffic_manager.py
│      trajectory_manager.py
│      vehicle_tracker.py
├─models
│      Braking___282.00max__282.00avg__282.00min__1679121006.model
│      Driving__6030.00max_6030.00avg_6030.00min__1679109656.model
└─test
        braking_dqn.py
        driving_dqn.py
        pedestrians_1.py
        pedestrians_2.py
        test_braking.py
        test_driving.py
```

## 核心功能

### 1. 自定义轨迹规划
使用 `get_location.py` 获取当前摄像头坐标，配置到 `config.py`:

```python
TRAJECTORIES = {
    "custom_trajectory": {
        "start": [x, y, z, yaw],  # 起点坐标和朝向
        "end": [x, y, z],         # 终点坐标
        "description": "自定义轨迹 - 城镇道路"
    }
}
```

### 2. 模型测试
- **刹车测试**: `test_braking.py` - 验证紧急制动性能
- **驾驶测试**: `test_driving.py` - 评估导航准确性

### 3. 行人模拟
- `pedestrians_1.py` - 随机行人生成（模式1）
- `pedestrians_2.py` - 随机行人生成（模式2）
## 配置说明

### 关键配置文件
`config.py` 包含所有可调整参数：
- 轨迹起点/终点坐标
- 深度学习模型参数
- 仿真环境设置

## 参考项目
本项目参考自: [varunpratap222/Autonomous-Vehicle-Navigation-Using-Deep-Learning](https://github.com/varunpratap222/Autonomous-Vehicle-Navigation-Using-Deep-Learning.git)

## 📝 注意事项
1. 确保CARLA仿真器已正确启动
2. 建议在独立的Conda环境中运行
3. 行人模拟模块需要额外计算资源

---

**温馨提示**: 运行前请确认CARLA版本为0.9.13，Python版本为3.7，以避免兼容性问题。