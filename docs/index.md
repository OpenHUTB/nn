title: 主页

# [神经网络](https://github.com/OpenHUTB/nn)

欢迎使用神经网络文档，该页面包含所有内容的索引。

## 目录

- [入门](#primary) - 基础算法与模型示例
- [感知](#perception) - 计算机视觉与感知系统
- [规划](#planning) - 路径规划与决策
- [控制](#control) - 控制算法与仿真
- [其他](#other)

---

## 入门 <span id="primary"></span>

- [__热身__](warmup.md) — 入门热身示例
- [__线性回归__](linear_regression.md) - 基础线性回归模型
- [__线性回归改进__](linear_regression_improved.md) - 线性回归优化版本
- [__线性回归修复__](linear_regression_fix.md) - 修复偏置未更新bug
- [__softmax回归__](softmax_regression.md) - 多分类softmax回归
- [__线性回归和softmax回归改进__](softmax_regression_improved.md) - 回归算法改进
- [__支持向量机__](svm.md) - SVM分类算法
- [__支持向量机改进__](svm_improved.md) - SVM优化版本
- [__简单神经网络__](simple_nn.md) - 基础神经网络
- [__卷积神经网络__](CNN.md) - CNN基础实现
- [__卷积神经网络改进__](cnn_keras_sequential_improved.md) - CNN优化版本
- [__循环神经网络__](RNN.md) - RNN基础实现
- [__循环神经网络改进__](poem_generation_rnn_improved.md) - RNN优化版本
- [__注意力机制__](attention.md) - 注意力机制原理与实现
- [__高斯混合__](gaussian_mixture.md) - 高斯混合模型
- [__高斯混合改进__](./chap11_gaussian_mixture/README.md) - 高斯混合优化
- [__受限玻尔兹曼机__](RBM.md) - RBM基础实现
- [__强化学习__](RL.md) - 强化学习基础
- [__机器人仿真__](ant_robot/机器人仿真系统.md) - MuJoCo机器人仿真
- [__强化学习工作空间__](rl_workspace.md) - 强化学习工作空间

[__基于深度学习置信度加权的自动驾驶雨天多传感器融合感知优化__](rain_sensor_fusion\DEMO_REPORT.md)

---

## 感知 <span id="perception"></span>

- [__车道线检测__](./lane_detection/README.md) - 基于 OpenCV 的 Carla 场景车道线检测（分步实现）
- [__carla_CAM__](./carla_CAM/README.md) - 使用类激活映射测试卷积神经网络
- [__交通标识识别__](./carla_traffic_sign_recognition/carla_traffic_sign_recognition.md) — 交通标识识别
- [__V2X路侧智能感知__](./edge_intelligence_V2X/README.md) - 基于YOLOv8n的V2X路侧智能感知系统优化与实现
- [__Carla多模态异常检测__](./carla_auto_vision_navigator.md) - 基于多模态融合的Carla非结构化场景异常检测自动驾驶
- [__目标检测__](./test/object_detection.md) - 目标检测与危险评估
- [__图像目标检测__](./image_object_detection/image_object_detection.md) - 多功能图像目标检测系统
- __跟踪__ - 目标跟踪
- [__路径追踪__](./test.md) - 路径追踪测试
- [__交通标识检测__](./traffic_sign_detection/README.md) - 交通标识目标检测
- [__td3_carracing__](./td3_carracing/README.md) - 基于 TD3 + CNN 的强化学习自动驾驶系统
- [__无人机飞行控制__](./UVA_flight_control_system.md) - 基于AirSim的无人机飞行控制系统
- [__人形机器人平衡控制__](./humanoid_balance/Humanoid_Balance.md) - 基于强化学习的人形机器人平衡控制仿真
- [__工程规范优化__](./improve/project.md) - 多场景仿真与控制优化项目
- [__人形机器人站立行走__](./mujoco_man/mujoco_manrun.md) - 基于 CPG + PD 的人形机器人稳定站立与行走仿真（MuJoCo）
- [__setup_tool模块汇报文档__](./setup_tool/report.md) - setup_tool 模块背景、改进内容、运行方式与效果总结
- [__机械臂自动抓取__](./robot_arm/README.md) - 基于Franka Panda的机械臂自动抓取仿真项目
- [__CARLA天气鲁棒性测试__](./carla_weather_robustness/README.md) - CARLA多天气条件下自动驾驶感知鲁棒性测试系统

- [__自动驾驶车辆语义分割__](./auto_drive_seg/README.md) - 基于 U-Net + Focal Loss 的 CARLA 街景 8 类语义分割（含数据增强、速度基准、概率热力图、类别频率分析）
- [__用户使用手势控制 Airsim 无人机__](./drone_hand_gesture/README.md) - 使用手势识别控制 Airsim 无人机飞行

- [__基于自监督学习与PPO强化学习的自动驾驶仿真项目__](./autonomous_driving/README.md) - 基于CARLA的SSL+RL自动驾驶仿真系统
- [__CARLA天气鲁棒性测试__](./carla_weather_robustness/README.md) - CARLA多天气条件下自动驾驶感知鲁棒性测试系统
- [__CARLA 模拟器中的物体与碰撞检测系统__](./carla_collision_detect/carla_collision_detect.md) - 物体与碰撞检测
  
## 规划 <span id="planning"></span>

[__导航__](#navigation)
- [__Carla YOLO规划器__](carla_yolo_planner.md) - Carla环境结合YOLO的自动驾驶路径规划方案
- [__人形机器人SAC强化学习步态优化__](./mujoco_running/running.md) - 基于CPG+PD+SAC残差强化学习的缓步稳定行走仿真
- [__人形机器人自主行走__](./mujoco_hci_sim/README.md) - 基于PPO强化学习的Humanoid人形机器人自主行走仿真
- [__人形机器人站立行走__](./mujoco_man/mujoco_manrun.md) - 基于 CPG + PD 的人形机器人稳定站立与行走仿真（MuJoCo）
- [__td3_carracing__](./td3_carracing/README.md) - 基于 TD3 + CNN 的 CarRacing 强化学习自动驾驶系统
- [__无人机路径学习__](./drone_path_learning/README.md) - 基于 AirSim + Gym + Stable-Baselines3 的无人机视觉导航强化学习项目
- [__机器人仿真(MuJoCo)__](ant_robot/机器人仿真系统.md)
- [__机械臂仿真系统__](arm_sim.md) - 基于MuJoCo的机械臂仿真与功能优化
- [__CARLA自动驾驶系统__](carla_ad_gym_rl/carla_ad_gym_rl.md) - CARLA 仿真与强化学习的自动驾驶智能决策系统
- [__CARLA天气鲁棒性测试__](./carla_weather_robustness/README.md) - CARLA多天气条件下自动驾驶感知鲁棒性测试系统

[__Carla YOLO规划器__](carla_yolo_planner.md) - Carla环境结合YOLO的自动驾驶路径规划方案

[__人形机器人SAC强化学习步态优化__](./mujoco_running/running.md) - 基于CPG+PD+SAC残差强化学习的缓步稳定行走仿真

[__人形机器人自主行走__](./mujoco_hci_sim/README.md) - 基于PPO强化学习的Humanoid人形机器人自主行走仿真

[__人形机器人站立行走__](./mujoco_man/mujoco_manrun.md) -  基于 CPG + PD 的人形机器人稳定站立与行走仿真（MuJoCo）

[__td3_carracing__](./td3_carracing/README.md) - 基于 TD3 + CNN 的 CarRacing 强化学习自动驾驶系统

[__机器人仿真(MuJoCo)__](ant_robot/机器人仿真系统.md)

[__机械臂仿真系统__](arm_sim.md) - 基于MuJoCo的机械臂仿真与功能优化

[__CARLA自动驾驶多场景仿真项目__](./DeFIX/docs/index.md)

[__自动驾驶系统__](./auto_drive_system/auto_drive_system_README) - 基于强化学习的自动驾驶系统

[__carla_2d_deeprl__](./carla_2d_deeprl/README.md) - 基于 CARLA 的极简 2D 深度强化学习自动驾驶环境。

[__PilotNet自动驾驶系统__](./pilotnet/index.md) - 基于NVIDIA PilotNet架构的端到端自动驾驶学习系统


## 控制  <span id="control"></span>

[PID](#pid)
- [__导航__](#navigation)
- [__自动驾驶系统__](./auto_drive_system/auto_drive_system_README) - 基于强化学习的自动驾驶系统
- [__CARLA自动驾驶多场景仿真项目__](./DeFIX/docs/index.md)
- [__carla_2d_deeprl__](./carla_2d_deeprl/README.md) - 基于 CARLA 的极简 2D 深度强化学习自动驾驶环境。
- [__无人机飞行控制__](./UVA_flight_control_system.md) - 基于AirSim的无人机飞行控制系统
- [__人形机器人平衡控制__](./humanoid_balance/Humanoid_Balance.md) - 基于强化学习的人形机器人平衡控制仿真
- [__工程规范优化__](./improve/project.md) - 多场景仿真与控制优化项目

# 其他  <span id="other"></span>
[ 驾驶事故视频识别 ](./carla_temporal_collage/index.md) - 基于 Temporal Collage Prompting 的 CARLA 驾驶事故视频识别系统

[__CARLA IMU 数据采集平台__](./carla_imu/carla_imu.md) — CARLA惯性测量单元数据采集与可视化驾驶平台开发汇报文档

[__人形机器人SAC强化学习步态优化__](./mujoco_running/running.md) - 基于CPG+PD+SAC残差强化学习的缓步稳定行走仿真

[__setup_tool模块汇报文档__](./setup_tool/report.md) - setup_tool 模块背景、改进内容、运行方式与效果总结

[__carla自动驾驶系统辅助监视器__](./autonomous_vehicle_monitor/monitor.md) -carla自动驾驶系统辅助器
