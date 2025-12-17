# 让一台装了激光雷达，IMU、GNSS和轮式底盘的机器人，能够在室外大范围、无明显人工特征的环境中实现：
# 1.高精度实时建图
# 2.可靠的全局定位
# 3.无人工干预的自主路径规划与导航

# 构建目录结构

```text
outdoor_ws/src/robot_description/
├── CMakeLists.txt
├── package.xml
├── urdf/
│   ├── robot.xacro           # 主机器人描述文件
│   ├── materials.xacro       # 颜色定义
│   └── sensors/              
│       ├── lidar.xacro       # LiDAR模型
│       └── imu.xacro         # IMU模型
├── launch/
│   ├── display.launch.py    # 显示模型
│   └── control.launch.py    # 控制启动
├── config/
│   ├── controllers.yaml     # ros2_control配置
│   └── robot_control.yaml   # 控制参数
└── meshes/                  # 可选：3D模型文件