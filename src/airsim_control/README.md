AirSim 深度强化学习无人机迷宫寻路系统

[项目简介]
基于 Microsoft AirSim API 与 Stable Baselines3 框架实现的无人机自动驾驶训练系统。
本项目摒弃了传统的规则算法，采用深度强化学习 (Deep Reinforcement Learning) 中的 PPO 算法。通过融合 32 线激光雷达 (LiDAR) 的距离信息与深度相机 (Depth Camera) 的视觉信息，训练无人机在虚拟迷宫环境中实现端到端的自主寻路与避障，具备从零经验自我学习并寻找出口的能力。

[核心功能]

1. 智能迷宫寻路 (DFS 深度优先)
    侧路优先策略：打破常规直行逻辑，采用 DFS (深度优先搜索) 思想，在遇到新岔路口时优先转弯探索，确保遍历所有路径。
    死胡同记忆与回溯：当检测到死路时，自动倒车并生成“虚拟墙（禁区）”封锁该区域，随后掉头返回主路。
    精准转向与纠偏：摒弃时间控制转向，采用闭环检测精准对准路口；若转向后对准墙壁，自动执行横向平移 (Side-Step) 进行修正。
    机身坐标系控制：使用 Body Frame 控制飞行，结合“急刹-原地决策”逻辑，彻底解决路口冲过头的问题。

2. 记忆与决策系统
    栅格化记忆地图：将世界划分为 1.5米 x 1.5米的栅格，实时记录“去过的地方”。
    射线检测评分：在路口决策时，向各个方向发射虚拟射线，检测路径深度与新旧程度，优先选择未探索区域。
    出口诱导机制：当雷达检测到极远距离（>15米）的开阔地时，判定为出口并全速冲刺。

3. 基础飞行保障
    慢速精细操控：巡航速度降至 1.0m/s，配合高灵敏度刹车逻辑，适应狭窄迷宫环境。
    高度硬限位 (Hard Clamp)：在 PID 控制基础上增加垂直速度硬锁（限制在 ±0.8m/s），无论误差多大，强制防止无人机“飞天”或失控。
    紧急避险：配备物理反推刹车逻辑，在距离障碍物 <1.0米 时强制反向推力，防止惯性撞墙。

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
代码运行时，会在 AirSim 窗口中绘制彩色小球，代表无人机的“思维过程”：

绿色球体：新路 (New Path) - 未探索的区域，优先级最高。
青色球体：岔路优先 (Priority) - 发现侧方新路口，强制优先探索。
蓝色方块：足迹 (Visited) - 已经走过的路径点。
红色球体：老路 (Old Path) - 探测到前方是走过的路，尽量避免。
黑色大球：死路/禁区 (Forbidden) - 已确认为死胡同，生成虚拟墙封锁。
黄色大球：出口 (Exit) - 检测到终点开阔地。

[手动操控 (辅助)]
虽然本系统设计为全自动运行，但在紧急情况下或使用手动模式代码时，可用以下按键：

[注意事项]
- 坐标转换：UE4 单位为厘米，AirSim 代码中单位为米，请注意转换 (除以100)。
- 性能优化：训练时请务必在 UE 编辑器偏好设置中取消 "Use Less CPU when in Background"。