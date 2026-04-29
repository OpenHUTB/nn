# 智能飞行控制系统

## 作业内容
基于 AirSim 无人机仿真平台，使用 Python 实现无人机自动起飞、**带碰撞检测的智能定点巡航**、**慢下降**功能，**RGB 相机拍照**，**键盘手动控制**功能，新增**速度档位切换**和**一键返航**。

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
drone_flight_sim/
├── main.py                  # 主程序入口（支持两种飞行模式）
├── drone_controller.py     # 无人机核心控制模块（含相机控制和键盘控制）
├── collision_handler.py     # 碰撞检测与处理模块
├── flight_path.py           # 航点规划模块
├── keyboard_control.py      # 键盘控制模块（新增）
├── config.py                # 配置文件（含相机和键盘配置）
├── utils.py                 # 工具函数
└── drone_images/            # 拍摄照片保存目录（自动创建）
```

## 飞行模式

程序支持两种飞行模式，启动时会让你选择：

### 模式 1：自动航点飞行模式
- 无人机按照预设的航点列表自动飞行
- 在每个航点自动拍照
- 适合执行重复性巡检任务

### 模式 2：键盘手动控制模式
- 使用键盘实时控制无人机飞行
- 支持拍照等功能
- 适合手动探索和精确控制

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
- **智能碰撞检测与自动恢复**：
  - 实时监测碰撞事件
  - 自动过滤地面/道路接触（Road、Ground、Terrain 等）
  - 区分严重碰撞与正常地面接触
  - **碰撞后自动恢复**：
    - 自动尝试后退避障（最多3次）
    - 恢复成功后继续执行飞行任务
  - **手动接管机制**：
    - 自动恢复失败后提示用户手动接管
    - 切换到键盘控制模式让用户解决碰撞
    - 脱离困境后可继续降落
- **大范围航点飞行**：
  - 预设11个航点，覆盖更大飞行区域
  - 飞行高度5米，更安全的高度

### 4. RGB 相机拍照功能
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

### 5. 键盘手动控制功能（新增）
支持以下键盘控制：

| 按键 | 功能 |
|------|------|
| W | 前进 |
| S | 后退 |
| A | 向左横移 |
| D | 向右横移 |
| Q | 上升 |
| E | 下降 |
| 空格 | 悬停 |
| 1-5 | 切换速度档位（慢/中/快/很快/极速） |
| R | 一键返航 |
| P | 拍照 |
| T | 拍摄所有图像(RGB+深度+分割) |
| N | 拍摄深度图像 |
| B | 拍照预览模式 |
| L | 执行降落 |
| ESC | 紧急停止并退出 |

**特点**：
- 持续按键时无人机持续移动
- 释放按键后自动悬停并显示移动距离
- 支持组合按键实现斜向飞行

**速度档位（1-5键）**：
- 1档（慢速）：1 m/s
- 2档（中速）：2 m/s（默认）
- 3档（快速）：3 m/s
- 4档（很快）：5 m/s
- 5档（极速）：8 m/s

**一键返航（R键）**：
- 自动飞回起飞点位置
- 到达后自动降落
- 起飞时会自动记录返航点

### 6. 安全降落系统
- **三重降落机制**：
  1. 正常降落：调用 AirSim 降落 API
  2. 重试机制：最多 3 次尝试
  3. 强制复位：降落失败时的最后保障
- 降落状态实时监控
- 高度检测与安全高度调整
- 降落完成后自动锁定电机

### 7. 慢速平稳降落
- **速度控制降落**：以 1m/s 的下降速度缓慢降落，避免冲击
- **下降过程监控**：实时显示当前高度，让降落过程可视化
- **渐进式着地**：从飞行高度逐步下降至着陆
- **电机柔和锁定**：着陆后平稳锁定电机，无抖动

## API 使用说明

### 键盘控制 API

```python
from keyboard_control import KeyboardController, print_control_help

# 打印控制说明
print_control_help()

# 创建键盘控制器并启动
controller = KeyboardController(drone)
controller.start()
```

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

# 显示无人机状态
drone.get_telemetry()
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
| `TAKEOFF_HEIGHT` | -5 | 起飞高度（米） |
| `FLIGHT_VELOCITY` | 3 | 飞行速度（米/秒） |
| `MAX_FLIGHT_TIME` | 60 | 最大飞行时间（秒） |
| `COLLISION_COOLDOWN` | 1.0 | 碰撞冷却时间（秒） |
| `RGB_CAMERA_NAME` | "0" | RGB 相机名称 |
| `KEYBOARD_VELOCITY` | 2 | 键盘控制移动速度（米/秒） |
| `KEYBOARD_VELOCITY` | 2 | 键盘控制默认速度（米/秒） |
| `KEYBOARD_STEP` | 2 | 键盘控制位移步长（米） |

## 运行步骤

1. **启动仿真环境**
   - 启动 AirSimNH.exe
   - 选择"否(N)"进入四旋翼无人机模式
   - 等待仿真环境完全加载

2. **运行程序**
   ```
   python main.py
   ```

3. **选择飞行模式**
   - 输入 `1`：自动航点飞行模式
   - 输入 `2`：键盘手动控制模式

4. **键盘控制模式操作**
   - 按 W/S/A/D 控制水平移动
   - 按 Q/E 控制升降
   - 按 1-5 切换速度档位
   - 按 R 一键返航
   - 按 P 拍照
   - 按 ESC 或 L 退出并降落

## 照片存储

运行后拍摄的图片会自动保存到 `drone_images/` 目录下，文件命名格式：

- RGB 图像：`rgb_YYYYMMDD_HHMMSS_X_Y_n序号.png`
- 深度图像：`depth_YYYYMMDD_HHMMSS_X_Y.png`
- 分割图像：`seg_YYYYMMDD_HHMMSS_X_Y.png`

其中 `X`、`Y` 为拍照时的无人机坐标，`序号` 为该次运行的第 N 张照片。
