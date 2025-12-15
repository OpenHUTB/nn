AirSim 无人机激光雷达系统：迷宫自主寻路与避障

[项目简介]
基于 Microsoft AirSim API 实现的无人机自动避障控制与环境 3D 点云扫描系统。
本项目利用 Python API 与 Microsoft AirSim 仿真环境交互，通过搭载的 32 线激光雷达（Lidar）获取环境深度数据。项目核心从简单的避障升级为具备记忆能力的自主路径规划系统，能够像人类一样探索迷宫、识别死胡同、并最终找到出口。

[核心功能]

1. 智能迷宫寻路
   - 直行优先策略：在多条路径可选时，优先保持直行，避免在老路中左右摇摆。
   - 死胡同记忆与封锁：利用 DFS (深度优先搜索) 思想，当检测到死路时，自动掉头并生成“虚拟墙（禁区）”，防止再次进入。
   - 出口诱导机制：当雷达检测到极远距离（>15米）的开阔地时，判定为出口并全速冲刺。
   - 机身坐标系控制：使用 Body Frame 控制飞行，彻底解决了无人机转向后方向感错乱的问题。

2. 记忆与决策系统
   - 栅格化记忆地图：将世界划分为 2米 x 2米的栅格，实时记录“去过的地方”。
   - 射线检测评分：在路口决策时，向各个方向发射虚拟射线，检测路径是否为“新路”，优先探索未知区域。
   - 防抖动冷却：决策后强制执行一段距离，防止在路口反复纠结。

3. 基础飞行保障
   - 紧急避险：配备物理反推刹车逻辑，在距离障碍物 <1.0米 时强制反向推力，防止惯性撞墙。
   - 高度锁定：通过 PID 控制器将飞行高度严格限制在设定值（如 -1.5米）。

[环境依赖]
在运行代码之前，请确保已安装以下 Python 库：
pip install airsim numpy keyboard

[配置文件]
关键步骤：为了确保雷达不被机身遮挡并获得最佳扫描效果，请务必使用以下配置覆盖你的 "文档\AirSim\settings.json" 文件。
注意：HorizontalFOV 必须设置为 -90 到 90，且 DataFrame 必须为 SensorLocalFrame。

{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone_1": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": 0,
      "Sensors": {
        "lidar_1": {
          "SensorType": 6,
          "Enabled": true,
          "Range": 100,
          "NumberOfChannels": 32,
          "PointsPerSecond": 100000,
          "RotationsPerSecond": 10,
          "VerticalFOVUpper": 20,
          "VerticalFOVLower": -45,
          "HorizontalFOVStart": -90,
          "HorizontalFOVEnd": 90,
          "X": 0, "Y": 0, "Z": -1.0,
          "DrawDebugPoints": true,
          "DataFrame": "SensorLocalFrame"
        }
      }
    }
  }
}

[运行方式]
1. 启动 Unreal Engine (AirSim) 仿真环境。
2. 确保无人机处于迷宫入口处。
3. 运行 Python 寻路脚本。

[可视化调试说明]
代码运行时，会在 AirSim 窗口中绘制彩色小球，代表无人机的“思维过程”：

- 绿色球体：新路 (New Path) - 未探索的区域，优先级最高。
- 蓝色方块：足迹 (Visited) - 已经走过的路径点。
- 红色球体：老路 (Old Path) - 探测到前方是走过的路，尽量避免。
- 黑色大球：死路/禁区 (Forbidden) - 已确认为死胡同，生成虚拟墙封锁。
- 黄色大球：出口 (Exit) - 检测到终点开阔地。

[手动操控 (辅助)]
虽然本系统设计为全自动运行，但在紧急情况下或使用手动模式代码时，可用以下按键：

Ctrl + C : 停止 (立即中断程序并降落)
W : 前进 (向机头方向水平移动)
S : 后退 (向后方水平移动)
A : 向左 (向左侧平移，不改变朝向)
D : 向右 (向右侧平移，不改变朝向)
Q : 左转 (原地逆时针旋转机头)
E : 右转 (原地顺时针旋转机头)
上箭头 : 上升 (垂直向上飞行)
下箭头 : 下降 (垂直向下飞行)
空格 : 急刹 (悬停，停止所有移动)