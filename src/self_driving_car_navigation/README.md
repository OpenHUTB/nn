# 自动驾驶仿真学习环境（基于Carla与Python3.7）

## 项目概述
本项目构建了基于Carla 0.9.11仿真平台和Python 3.7的自动驾驶学习环境，适用于车辆控制、场景感知、路径规划等自动驾驶相关算法的学习与实践。通过VSCode开发环境实现代码编写、调试与运行的一体化流程，支持从基础车辆控制到复杂场景仿真的完整学习路径。

## 环境准备

### 依赖库安装
```bash
# 安装Python 3.7（建议使用虚拟环境）
# 安装Carla 0.9.11客户端
pip install carla==0.9.11
# 安装辅助依赖库
pip install numpy opencv-python matplotlib vscode-debugpy
```
- `carla==0.9.11`：自动驾驶仿真平台核心库，提供车辆、环境与传感器模拟
- `python 3.7`：项目开发与运行的Python版本
- `numpy`：数值计算支持
- `opencv-python`：图像数据处理
- `matplotlib`：数据可视化工具
- `vscode-debugpy`：VSCode调试支持

### 开发环境配置
1. 下载并安装[Carla 0.9.11官方发行版](https://github.com/carla-simulator/carla/releases/tag/0.9.11)
2. 安装[VSCode](https://code.visualstudio.com/)并配置Python 3.7解释器
3. 推荐插件：Python、Pylance、Code Runner（提升开发效率）

## 项目结构

| 文件名             | 功能描述                                                     |
|-------------------|--------------------------------------------------------------|
| `main.py`         | 核心程序入口，实现Carla客户端连接、世界初始化与主循环控制       |
| `vehicle_control.py`| 车辆控制模块，实现油门、刹车、转向等基础控制逻辑               |
| `scene_generation.py`| 场景生成工具，支持随机障碍物、天气变化与交通参与者生成         |
| `sensor_manager.py`| 传感器管理模块，处理摄像头、激光雷达等数据采集与解析           |
| `utils.py`        | 通用工具函数，包含坐标转换、数据可视化等辅助功能               |
| `config.yaml`     | 配置文件，存储仿真参数（如帧率、传感器类型、车辆模型等）       |
| `README.md`       | 项目说明文档                                                   |

## 核心功能

### 1. 基础车辆控制（main.py & vehicle_control.py）
- 客户端连接管理：自动连接Carla服务器，支持断开重连机制
- 多车辆控制：同时控制多辆自动驾驶车辆，实现编队行驶模拟
- 控制模式切换：支持手动控制（键盘）与自动控制（程序）模式切换
- 状态实时反馈：在VSCode终端输出车辆速度、位置等关键信息

```python
# 车辆控制逻辑示例
import carla

# 连接到Carla服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 获取车辆控制器
vehicle = world.get_actors().filter('*vehicle*')[0]
control = carla.VehicleControl()

# 设定前进指令（油门0.5，转向0）
control.throttle = 0.5
control.steer = 0.0
vehicle.apply_control(control)
```

### 2. 场景仿真与传感器（scene_generation.py & sensor_manager.py）
- 动态场景生成：支持随机天气（雨、雾、时间）、障碍物与交通灯配置
- 多传感器集成：摄像头（RGB/深度）、激光雷达、毫米波雷达数据采集
- 数据同步存储：传感器数据与车辆状态时间戳同步，便于离线分析
- VSCode调试支持：断点调试传感器数据处理流程，直观查看数据格式

```python
# 传感器配置示例
def setup_camera(world, vehicle):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    # 安装在车辆前方
    transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    # 定义数据回调函数
    camera.listen(lambda image: process_image(image))
    return camera
```

### 3. 学习任务支持
- 路径跟踪练习：预设参考路径，实现PID等控制算法跟踪
- 避障场景训练：生成动态障碍物，练习碰撞检测与规避逻辑
- 数据采集工具：批量采集不同场景下的传感器数据，用于模型训练

## 使用方法

1. 启动Carla服务器：
```bash
# 在Carla安装目录下执行
./CarlaUE4.sh  # Linux/Mac
CarlaUE4.exe   # Windows
```

2. 基础车辆控制示例：
```bash
python main.py --mode manual  # 手动控制模式
python main.py --mode auto    # 自动控制模式
```

3. 场景仿真运行：
```bash
python scene_generation.py --weather rain --obstacles 5
```

4. 传感器数据采集：
```bash
python sensor_manager.py --record --output ./data
```

### VSCode开发提示
- 按`F5`启动调试模式（需配置`.vscode/launch.json`）
- 使用Code Runner插件（右键`Run Code`）快速执行单文件
- 推荐使用VSCode的Jupyter插件进行分步调试与数据可视化

## 参数调整指南

| 参数               | 调整范围 | 效果说明                         |
|-------------------|----------|----------------------------------|
| `throttle_gain`   | 0.1~1.0 | 增大会提高加速响应，过大会导致打滑 |
| `sensor_fps`      | 10~60   | 提高值增加数据精度（增加计算量）  |
| `obstacle_density`| 0~20    | 增大会增加场景复杂度             |
| `simulation_delta_seconds`| 0.01~0.1 | 减小值提高仿真精度（降低运行速度）|

## 参考资料
- [Carla 0.9.11官方文档](https://carla.readthedocs.io/en/0.9.11/)
- [Python 3.7官方文档](https://docs.python.org/3.7/)
- [VSCode Python开发指南](https://code.visualstudio.com/docs/languages/python)
- [Carla自动驾驶教程](https://carla.readthedocs.io/en/latest/tutorials/)