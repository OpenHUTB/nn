title: 主页

=======
# [神经网络](https://github.com/OpenHUTB/nn)

欢迎使用神经网络文档，该页面包含所有内容的索引。

* [__入门__](#primary)
* [__感知__](#perception)
* [__规划__](#planning)
* [__控制__](#control)
* [__其他__](#other)

---

## 入门 <span id="primary"></span>


[__热身__](warmup.md) — 入门热身示例

[__线性回归__](linear_regression.md)

[__线性回归改进__](linear_regression_improved.md)

[__线性回归修复__](linear_regression_fix.md) - 修复偏置未更新bug

[__softmax回归__](softmax_regression.md)

[__线性回归和softmax回归改进__](softmax_regression_improved.md)

[__支持向量机__](svm.md)

[__支持向量机改进__](svm_improved.md)

[__简单神经网络__](simple_nn.md)

[__卷积神经网络__](CNN.md)

[__卷积神经网络改进__](cnn_keras_sequential_improved.md)

[__循环神经网络__](RNN.md)

[__循环神经网络改进__](poem_generation_rnn_improved.md)

[__注意力机制__](attention.md)

[__高斯混合__](gaussian_mixture.md)

[__高斯混合改进__](./chap11_gaussian_mixture/README.md)

[__受限玻尔兹曼机__](RBM.md)

[__强化学习__](RL.md)



---
## 感知 <span id="perception"></span>

[__车道线检测__](./lane_detection/README.md) - 基于 OpenCV 的 Carla 场景车道线检测（分步实现）

[__carla_CAM__](./carla_CAM/README.md) - 使用类激活映射测试卷积神经网络

[__用户使用手势控制 Airsim 无人机__](./drone_hand_gesture/README.md) - 使用手势识别控制 Airsim 无人机飞行

[__V2X路侧智能感知__](./edge_intelligence_V2X/README.md) - 基于YOLOv8n的V2X路侧智能感知系统优化与实现

[__目标检测__](./test/object_detection.md) - 目标检测与危险评估

[__图像目标检测__](./image_object_detection/image_object_detection.md) - 多功能图像目标检测系统

[__路径追踪__](./test.md)

[__交通标识检测__](./traffic_sign_detection/README.md) - 目标检测


## 规划 <span id="planning"></span>

[__Carla YOLO规划器__](carla_yolo_planner.md) - Carla环境结合YOLO的自动驾驶路径规划方案

[__人形机器人SAC强化学习步态优化__](./mujoco_running/running.md) - 基于CPG+PD+SAC残差强化学习的缓步稳定行走仿真

[__人形机器人自主行走__](./mujoco_hci_sim/README.md) - 基于PPO强化学习的Humanoid人形机器人自主行走仿真

[__人形机器人站立行走__](./mujoco_man/mujoco_manrun.md) -  基于 CPG + PD 的人形机器人稳定站立与行走仿真（MuJoCo）

[__td3_carracing__](./td3_carracing/README.md) - 基于 TD3 + CNN 的 CarRacing 强化学习自动驾驶系统

[__机器人仿真(MuJoCo)__](ant_robot/机器人仿真系统.md)

[__机械臂仿真系统__](arm_sim.md) - 基于MuJoCo的机械臂仿真与功能优化

[__自动驾驶系统__](./auto_drive_system/auto_drive_system_README) - 基于强化学习的自动驾驶系统

## 控制  <span id="control"></span>

[__无人机飞行控制__](./UVA_flight_control_system.md) - 基于AirSim的无人机飞行控制系统

[__人形机器人平衡控制__](./humanoid_balance/Humanoid_Balance.md) - 基于强化学习的人形机器人平衡控制仿真

[__工程规范优化__](./improve/project.md) - 多场景仿真与控制优化项目


# 其他  <span id="other"></span>

[__CARLA IMU 数据采集平台__](./carla_imu/carla_imu.md) — CARLA惯性测量单元数据采集与可视化驾驶平台开发汇报文档

[__setup_tool模块汇报文档__](./setup_tool/report.md) - setup_tool 模块背景、改进内容、运行方式与效果总结



