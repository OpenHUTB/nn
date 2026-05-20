title: 主页

# [神经网络](https://github.com/OpenHUTB/nn)

欢迎使用神经网络文档，该页面包含所有内容的索引。

## 目录

- [入门](#primary) - 基础算法与模型示例
- [感知](#perception) - 计算机视觉与感知系统
- [规划](#planning) - 路径规划与决策
- [控制](#control) - 控制算法与仿真

---

## 入门 <span id="primary"></span>

- [__热身__](warmup.md) - 入门热身示例
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

---

## 感知 <span id="perception"></span>

- [__carla_CAM__](./carla_CAM/README.md) - 使用类激活映射测试卷积神经网络
- [__V2X路侧智能感知__](./edge_intelligence_V2X/README.md) - 基于YOLOv8n的V2X路侧智能感知系统优化与实现
- [__目标检测__](./test/object_detection.md) - 目标检测与危险评估
- [__图像目标检测__](./image_object_detection/image_object_detection.md) - 多功能图像目标检测系统
- __跟踪__ - 目标跟踪
- [__路径追踪__](./test.md) - 路径追踪测试
- [__交通标识检测__](./traffic_sign_detection/README.md) - 交通标识目标检测
- [__td3_carracing__](./td3_carracing/README.md) - 基于 TD3 + CNN 的 CarRacing 强化学习自动驾驶系统
- [__无人机飞行控制__](./UVA_flight_control_system.md) - 基于AirSim的无人机飞行控制系统
- [__人形机器人平衡控制__](./humanoid_balance/Humanoid_Balance.md) - 基于强化学习的人形机器人平衡控制仿真
- [__工程规范优化__](./improve/project.md) - 多场景仿真与控制优化项目
- [__人形机器人站立行走__](./mujoco_man/mujoco_manrun.md) - 基于 CPG + PD 的人形机器人稳定站立与行走仿真（MuJoCo）
- [__setup_tool模块汇报文档__](./setup_tool/report.md) - setup_tool 模块背景、改进内容、运行方式与效果总结
- [__机械臂自动抓取__](./robot_arm/README.md) - 基于Franka Panda的机械臂自动抓取仿真项目

## 规划 <span id="planning"></span>

[__导航__](#navigation)

## 控制  <span id="control"></span>

[PID](#pid)
