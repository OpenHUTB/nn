# 智能飞行控制系统

## 作业内容
基于 AirSim 无人机仿真平台，使用 Python 实现无人机自动起飞、**带碰撞检测的智能定点巡航**、**慢下降**功能，**RGB/深度/分割多模式相机拍照与处理**，**键盘手动控制**功能，**速度档位切换**、**一键返航**，以及 **CNN 深度图像碰撞预测与实时避障**。

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
- torch

## 项目结构
```
drone_flight_sim/
├── main.py                      # 主程序入口（支持两种飞行模式）
├── drone_controller.py          # 无人机核心控制模块
├── collision_handler.py         # 碰撞检测与处理模块(简单神经网络)
├── collision_predictor.py       # CNN 碰撞预测模块（实时推理）
├── flight_path.py               # 航点规划模块
├── keyboard_control.py          # 键盘控制模块
├── collision_data_collector.py  # 手动碰撞数据采集模块
├── auto_collision_collector.py  # 自动碰撞数据采集模块
├── train_collision_model.py     # CNN 碰撞预测模型训练
├── config.py                    # 配置文件
├── utils.py                     # 工具函数
├── drone_images/                # 拍摄照片保存目录（自动创建）
└── collision_dataset/           # 碰撞数据采集保存目录（自动创建）
```

## 飞行模式

程序支持两种飞行模式，启动时会让你选择：

### 模式 1：自动航点飞行模式
- 无人机按照预设的航点列表自动飞行
- 在每个航点自动拍照
- 实时 CNN 碰撞预测与自动避障
- 适合执行重复性巡检任务

### 模式 2：键盘手动控制模式
- 使用键盘实时控制无人机飞行
- 支持拍照等功能
- 按 M 键开启/关闭 CNN 碰撞预警
- 适合手动探索和精确控制

## 功能实现

### 1. 自动连接与初始化
- 自动连接 AirSim 仿真环境
- 获取无人机控制权并解锁电机
- 初始化碰撞检测系统
- 初始化相机系统
- 自动加载 CNN 碰撞预测模型

### 2. 智能起飞控制
- 自动起飞至指定高度（默认5米）
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

### 4. CNN 碰撞预测与实时避障（新增）
- **模型**：使用 CollisionCNN 对深度图像进行二分类（安全 vs 危险）
- **训练**：`python train_collision_model.py`
- **性能**：测试准确率 91.80%（301 个样本）
- **航点模式**：飞行循环中每 0.5 秒用深度图推理碰撞风险
  - 🟢 安全：正常飞行
  - 🟡 预警：自动减速
  - 🔴 危险：连续 3 次触发后自动后退+上升避障
- **键盘模式**：按 **M 键**开启/关闭 CNN 碰撞预警（独立线程监控）
- **避障策略**：后退 3m + 上升 2m，避障后自动恢复飞向目标
- 模型文件：`collision_model.pth`

### 5. RGB 相机拍照功能（键盘快捷键操作）

**支持键盘快捷键**：

| 按键 | 功能 | 说明 |
|------|------|------|
| P | RGB拍照 | 拍摄彩色图像，自动保存 |
| T | 全景拍照 | 一次性拍摄 RGB + 深度 + 分割三种图像 |
| N | 深度图像 | 拍摄深度图（伪彩色：蓝=近，红=远） |
| B | 实时预览 | 打开相机预览窗口，实时查看无人机视角 |

**拍照功能详情**：
- **RGB拍照（P键）**：拍摄无人机视角的彩色照片，自动保存到 `drone_images/` 目录，文件名包含时间戳和位置信息
- **全景拍照（T键）**：同时获取 RGB 彩色图、深度图、分割图三种图像，适合需要完整数据的场景
- **深度图像（N键）**：拍摄深度图，使用伪彩色显示（JET色彩表：蓝色表示近，红色表示远），可用于测距和避障
- **实时预览（B键）**：打开相机实时预览窗口，可以直观看到无人机视角，适合探索环境时使用

**图片保存**：
- 保存位置：`drone_images/` 目录（自动创建）
- RGB图像：`rgb_时间戳_X_Y_n序号.png`
- 深度图像：`depth_时间戳_X_Y.png`
- 分割图像：`seg_时间戳_X_Y.png`
- 全景图像：`all_时间戳_X_Y_rgb/depth/seg.png`

### 6. 键盘手动控制功能

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
| L | 执行降落 |
| M | 切换 CNN 碰撞预警（开/关） |
| ESC | 紧急停止并退出 |
| O | 一键环绕（飞矩形轨迹） |

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

### 7. 安全降落系统
- **三重降落机制**：
  1. 正常降落：调用 AirSim 降落 API
  2. 重试机制：最多 3 次尝试
  3. 强制复位：降落失败时的最后保障
- 降落状态实时监控
- 高度检测与安全高度调整
- 降落完成后自动锁定电机

### 8. 慢速平稳降落
- **速度控制降落**：以 1m/s 的下降速度缓慢降落，避免冲击
- **下降过程监控**：实时显示当前高度，让降落过程可视化
- **渐进式着地**：从飞行高度逐步下降至着陆
- **电机柔和锁定**：着陆后平稳锁定电机，无抖动

## 配置参数

在 `config.py` 中可以修改以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TAKEOFF_HEIGHT` | -5 | 起飞高度（米） |
| `FLIGHT_VELOCITY` | 3 | 飞行速度（米/秒） |
| `MAX_FLIGHT_TIME` | 60 | 最大飞行时间（秒） |
| `COLLISION_COOLDOWN` | 1.0 | 碰撞冷却时间（秒） |
| `RGB_CAMERA_NAME` | "0" | RGB 相机名称 |
| `KEYBOARD_VELOCITY` | 2 | 键盘控制默认速度（米/秒） |
| `KEYBOARD_STEP` | 2 | 键盘控制位移步长（米） |
| `ENABLE_CNN_PREDICTION` | True | 是否启用 CNN 碰撞预测 |
| `CNN_RISK_THRESHOLD` | 0.7 | 碰撞风险阈值 |
| `CNN_PREDICTION_INTERVAL` | 0.5 | CNN 推理间隔（秒） |
| `CNN_CONSECUTIVE_WARNING_THRESHOLD` | 3 | 连续预警触发避障次数 |
| `CNN_AVOID_BACK_DISTANCE` | 3.0 | 避障后退距离（米） |
| `CNN_AVOID_RISE_HEIGHT` | 2.0 | 避障上升高度（米） |

## 碰撞数据采集

### 自动采集（推荐）

```bash
python auto_collision_collector.py
```

**飞行模式**：

| 模式 | 说明 | 碰撞率 |
|------|------|--------|
| 1. 螺旋飞行 | 螺旋向外扩大飞行 | 中 |
| 2. 随机飞行 | 飞向随机目标点 | 中 |
| 3. 折线飞行 | 高速直线折返 | 高 |
| 4. 撞墙模式 | 专门朝障碍物飞行 | 最高 |

**采集原理**：
- 安全样本(label=0)：飞行过程中定期自动采集
- 危险样本(label=1)：碰撞事件发生时自动采集并标注
- 碰撞后自动恢复，继续飞行采集

### 手动采集

```bash
python collision_data_collector.py
```

| 按键 | 功能 |
|------|------|
| W/S/A/D | 飞行控制 |
| Q/E | 上升/下降 |
| 0 | 设置安全标签 |
| 1 | 设置危险标签 |
| C | 采集当前样本 |

数据保存：`collision_dataset/depth/` + `collision_dataset/labels.csv`

## 碰撞预测模型

### 训练

```bash
python train_collision_model.py
```

### 评估

```bash
python train_collision_model.py --eval
```

### 模型架构

```
输入: 深度图像 (64x64 灰度)
    ↓
Conv2D(1→16) + BN + ReLU + MaxPool
    ↓
Conv2D(16→32) + BN + ReLU + MaxPool
    ↓
Conv2D(32→64) + BN + ReLU + MaxPool
    ↓
Conv2D(64→128) + BN + ReLU + MaxPool
    ↓
Flatten → Dense(256) → Dropout → Dense(64) → Dense(1)
    ↓
输出: 碰撞风险概率 [0~1]
```

### 性能
- 数据集：`collision_dataset/`（301 个样本，安全: 278, 危险: 23）
- 数据增强：过采样平衡类别
- 测试准确率：91.80%
- 模型文件：`collision_model.pth`

## API 参考

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

# 正方形路径
waypoints = FlightPath.square_path(size=15, height=-3)

# 矩形路径
waypoints = FlightPath.rectangle_path(width=20, length=10, altitude=-3)

# 三角形路径
waypoints = FlightPath.triangle_path(size=15, height=-5)

# 自定义路径
waypoints = [(5, 0, -3), (5, -5, -3), (0, -5, -3), (0, 0, -3)]
```

### CNN 碰撞预测 API

```python
from collision_predictor import CollisionPredictor

# 创建预测器
predictor = CollisionPredictor(risk_threshold=0.7)

# 从深度图像预测
result = predictor.predict(depth_image_numpy)
# result = {'probability': 0.85, 'is_dangerous': True, 'risk_level': 'danger'}

# 从 AirSim 直接获取深度图并预测
result = predictor.predict_from_airsim(client, camera_name="0")
```

## 运行步骤

1. **启动仿真环境**
   - 启动 AirSimNH.exe
   - 选择"否(N)"进入四旋翼无人机模式
   - 等待仿真环境完全加载

2. **运行程序**
   ```bash
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
   - 按 M 切换 CNN 碰撞预警
   - 按 P/T/N/B 进行拍照操作
   - 按 ESC 或 L 退出并降落

## 照片存储

运行后拍摄的图片会自动保存到 `drone_images/` 目录下，文件命名格式：

- RGB 图像：`rgb_YYYYMMDD_HHMMSS_X_Y_n序号.png`
- 深度图像：`depth_YYYYMMDD_HHMMSS_X_Y.png`
- 分割图像：`seg_YYYYMMDD_HHMMSS_X_Y.png`

其中 `X`、`Y` 为拍照时的无人机坐标，`序号` 为该次运行的第 N 张照片。
