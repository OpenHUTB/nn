AirSim 深度强化学习无人机迷宫寻路系统

[项目简介]
基于 Microsoft AirSim API 与 Stable Baselines3 框架实现的无人机自动驾驶训练系统。
本项目摒弃了传统的规则算法，采用深度强化学习 (Deep Reinforcement Learning) 中的 PPO 算法。通过融合 32 线激光雷达 (LiDAR) 的距离信息与深度相机 (Depth Camera) 的视觉信息，训练无人机在虚拟迷宫环境中实现端到端的自主寻路与避障，具备从零经验自我学习并寻找出口的能力。

[核心功能]

1. 多模态传感器融合 (Sensor Fusion)
   视觉感知 (CNN)：利用前视深度相机捕获环境几何特征，通过卷积神经网络 (CNN) 提取墙壁纹理与空间结构，克服单一雷达数据的局限性。
   雷达感知 (1D-Scan)：将 3D 点云数据降维处理为 180 维的扇区距离向量，精准感知前方 -90° 至 +90° 的障碍物距离。
   决策融合：神经网络同时接收视觉与雷达输入，实现比单一传感器更鲁棒的避障决策。

2. PPO 深度强化学习策略
   算法内核：采用 Proximal Policy Optimization (PPO) 算法，兼顾样本效率与训练稳定性。
   稠密奖励机制 (Dense Reward)：设计了基于“距离终点增量”的引导奖励，配合碰撞惩罚与时间步惩罚，有效解决了稀疏奖励导致的迷失问题。
   持续学习能力：支持断点续训 (Checkpoint)，可加载之前的模型权重继续训练，逐步提升智能体智商。

3. 飞行控制与稳定性优化
   高度锁定 (Z-Axis Lock)：使用 moveByVelocityZBodyFrameAsync 接口强制锁定飞行高度（-1.5米），彻底消除重力漂移导致的触地问题。
   仿真加速：配置 AirSim 时钟加速 (ClockSpeed)，在保证物理仿真精度的前提下，将训练效率提升 5-10 倍。
   后台运行优化：解决了虚幻引擎后台降速问题，支持长时间无人值守训练。

[环境依赖]
在运行代码之前，请确保已安装以下 Python 库：
pip install airsim gymnasium stable-baselines3 shimmy opencv-python tensorboard

[项目结构]
custom_env.py     : 自定义 Gym 环境封装 (处理雷达/图像数据、计算奖励、重置环境)。
train.py          : 训练脚本 (定义 PPO 模型、配置超参数、执行训练循环)。
run_inference.py  : 推理/测试脚本 (加载训练好的模型，在环境中实际飞行测试)。
continue_train.py : (可选) 续训脚本，用于加载旧模型继续训练。

[配置文件]
关键步骤：为了开启时间加速并启用深度相机窗口，请务必使用以下配置覆盖你的 "文档\AirSim\settings.json" 文件。

{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 10,
  "ViewMode": "SpringArmChase",
  "Vehicles": {
    "Drone_1": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": 0,
      "Sensors": {
        "lidar_1": {
          "SensorType": 6,
          "Enabled": true,
          "Range": 40,
          "NumberOfChannels": 32,
          "PointsPerSecond": 60000,
          "RotationsPerSecond": 10,
          "VerticalFOVUpper": 10,
          "VerticalFOVLower": -10,
          "HorizontalFOVStart": -90,
          "HorizontalFOVEnd": 90,
          "X": 0, "Y": 0, "Z": -0.5,
          "DrawDebugPoints": false,
          "DataFrame": "SensorLocalFrame"
        }
      },
      "Cameras": {
        "front_center_custom": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 256,
              "Height": 144,
              "FOV_Degrees": 90
            },
            {
              "ImageType": 1,
              "Width": 256,
              "Height": 144,
              "FOV_Degrees": 90
            }
          ],
          "X": 0.5, "Y": 0, "Z": 0,
          "Pitch": 0, "Roll": 0, "Yaw": 0
        }
      }
    }
  },
  "SubWindows": [
    {
      "WindowID": 0,
      "CameraName": "front_center_custom",
      "ImageType": 1,
      "VehicleName": "Drone_1",
      "Visible": true
    },
    {
      "WindowID": 1,
      "CameraName": "front_center_custom",
      "ImageType": 0,
      "VehicleName": "Drone_1",
      "Visible": true
    }
  ]
}

[运行方式]
1. 启动 Unreal Engine (AirSim) 仿真环境，点击 Play。
2. 确保 custom_env.py 中的 EXIT_POS (出口坐标) 已根据迷宫实际情况修改。
3. 训练模型：
   python train.py
4. 测试模型 (训练完成后)：
   python run_inference.py

[可视化调试说明]
在训练或测试过程中：
AirSim 主窗口下方：会自动弹出两个子窗口，分别显示无人机视角的深度图 (Depth) 和彩色图 (RGB)，用于监控输入数据是否正常。
TensorBoard：训练日志保存在 ./airsim_logs/ 目录下，可使用 `tensorboard --logdir ./airsim_logs/` 查看奖励曲线 (ep_rew_mean) 变化。

[注意事项]
- 坐标转换：UE4 单位为厘米，AirSim 代码中单位为米，请注意转换 (除以100)。
- 性能优化：训练时请务必在 UE 编辑器偏好设置中取消 "Use Less CPU when in Background"。