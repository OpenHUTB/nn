# 无人机第三次作业 - 智能飞行控制系统

## 作业内容
基于 AirSim 无人机仿真平台，使用 Python 实现无人机自动起飞、**带碰撞检测的智能定点巡航**、**慢下降**功能，并新增 **RGB 相机拍照** 功能。

## 运行环境
- 操作系统：Windows 10/11 64位
- Python 版本：Python 3.10.11
- 仿真平台：AirSimNH / AirSim

## 依赖库
- airsim==1.8.1
- numpy>=1.21
- opencv-python>=4.5.0
- pynput>=1.8
- msgpack-rpc-python>=0.4.1

## 项目结构
```
drone-flight-sim/
├── main.py                  # 主程序入口
├── drone_controller.py     # 无人机核心控制模块（含相机控制）
├── collision_handler.py     # 碰撞检测与处理模块
├── flight_path.py           # 航点规划模块
├── config.py                # 配置文件（含相机配置）
├── utils.py                 # 工具函数
└── drone_images/            # 拍摄照片保存目录（自动创建）
```

## 功能实现

### 1. 自动连接与初始化
- 自动连接 AirSim 仿真环境
- 获取无人机控制权并解锁电机
- 初始化碰撞检测系统
- 初始化相机系统

### 2. 智能起飞控制
- 自动起飞至指定高度（默认3米）
- 起飞超时保护（10秒）
- 起飞状态验证与反馈

### 3. 定点巡航
- **智能碰撞检测**：
  - 实时监测碰撞事件
  - 自动过滤地面/道路接触（Road、Ground、Terrain 等）
  - 区分严重碰撞与正常地面接触
  - 碰撞后自动停止并进入应急流程

### 4. RGB 相机拍照功能（新增）
- **RGB 彩色图像拍摄**：
  - 捕获无人机视角的 RGB 彩色图像
  - 自动保存为 PNG 格式
  - 文件名包含时间戳、坐标、序号信息
- **深度图像拍摄**：
  - 以伪彩色方式保存深度信息（蓝色=近，红色=远）
- **分割图像拍摄**：
  - 将场景中不同物体用不同颜色标记
- **全景拍摄**：
  - 同时拍摄 RGB + 深度 + 分割三种图像
- **图片预览**：
  - 支持实时显示相机预览窗口

### 5. 安全降落系统
- **三重降落机制**：
  1. 正常降落：调用 AirSim 降落 API
  2. 重试机制：最多 3 次尝试
  3. 强制复位：降落失败时的最后保障
- 降落状态实时监控
- 高度检测与安全高度调整
- 降落完成后自动锁定电机

### 6. 慢速平稳降落
- **速度控制降落**：以 1m/s 的下降速度缓慢降落，避免冲击
- **下降过程监控**：实时显示当前高度，让降落过程可视化
- **渐进式着地**：从飞行高度逐步下降至着陆
- **电机柔和锁定**：着陆后平稳锁定电机，无抖动

## API 使用说明

### 相机控制 API

```python
# 创建无人机控制器
drone = DroneController()

# 设置图片保存目录（可选，默认保存到 drone_images 文件夹）
drone.set_output_dir("my_photos")

# 拍摄 RGB 彩色图像
drone.capture_image()

# 指定文件名保存
drone.capture_image(filename="my_photo.png")

# 拍摄并显示预览窗口
drone.capture_image(show_preview=True)

# 拍摄深度图像（伪彩色）
drone.capture_depth_image()

# 拍摄分割图像
drone.capture_segmentation_image()

# 同时拍摄 RGB + 深度 + 分割三种图像
drone.capture_all_cameras()
```

### 航点规划 API

```python
from flight_path import FlightPath

# 使用正方形路径
waypoints = FlightPath.square_path(size=15, height=-3)

# 使用矩形路径
waypoints = FlightPath.rectangle_path(width=20, length=10, altitude=-3)

# 使用自定义路径
waypoints = [(5, 0, -3), (5, -5, -3), (0, -5, -3), (0, 0, -3)]
```

## 配置参数

在 `config.py` 中可以修改以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TAKEOFF_HEIGHT` | -3 | 起飞高度（米） |
| `FLIGHT_VELOCITY` | 3 | 飞行速度（米/秒） |
| `MAX_FLIGHT_TIME` | 60 | 最大飞行时间（秒） |
| `COLLISION_COOLDOWN` | 1.0 | 碰撞冷却时间（秒） |
| `RGB_CAMERA_NAME` | "0" | RGB 相机名称 |

## 运行步骤

1. **启动仿真环境**
   - 启动 AirSimNH.exe
   - 选择"否(N)"进入四旋翼无人机模式
   - 等待仿真环境完全加载

2. **配置飞行路径**（可选）
   - 编辑 `main.py` 中的 `waypoints` 变量
   - 或使用 `flight_path.py` 中的预设路径
   - 在航点处自动拍照（已内置于代码中）

3. **运行程序**
   ```bash
   python main.py
   ```

## 照片存储

运行后拍摄的图片会自动保存到 `drone_images/` 目录下，文件命名格式：

- RGB 图像：`rgb_YYYYMMDD_HHMMSS_X_Y_n序号.png`
- 深度图像：`depth_YYYYMMDD_HHMMSS_X_Y.png`
- 分割图像：`seg_YYYYMMDD_HHMMSS_X_Y.png`

其中 `X`、`Y` 为拍照时的无人机坐标，`序号` 为该次运行的第 N 张照片。
