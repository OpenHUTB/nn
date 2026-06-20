## CARLA中的交通标志检测与车辆控制

本项目在CARLA模拟器中实现了一个智能驾驶系统，集成了实时交通标志检测、障碍物避让和自适应车辆控制功能。系统使用YOLOv8物体检测模型识别交通标志（停车标志、限速标志），并结合红绿灯状态和前方障碍物检测，实现安全的自动驾驶行为。通过Pygame实时显示驾驶员视角摄像头画面和HUD信息（车速、障碍物距离、检测到的标志）。

## 主要功能

### 1. 交通标志检测
- 使用YOLOv8轻量级模型进行实时目标检测
- 支持识别停车标志（STOP）和限速标志（Speed Limit）
- 自动适配GPU/CPU加速推理
- 置信度阈值0.5过滤低质量检测结果

### 2. 智能车辆控制
- **障碍物检测与避让**：检测前方50米范围内、35度视角内的其他车辆，根据距离分级刹车（轻刹/重刹/满刹）
- **红绿灯响应**：自动识别红灯并平滑减速停车
- **停车标志处理**：检测到STOP标志时减速停车，等待2秒后继续
- **限速标志跟随**：根据限速标志自动调整车速，保持在规定速度±5km/h范围内
- **动态安全距离**：根据车速动态计算安全距离（基础10米 + 车速×0.6）

### 3. 自动路径跟随
- 基于地图路点（Waypoint）的自动转向控制
- 平滑滤波算法防止方向盘抖动
- 转向角度限制（最大±0.7）确保稳定性
- 每帧转向变化量限制（±0.07）避免剧烈摆动

### 4. 实时可视化
- Pygame窗口显示800×600分辨率的驾驶员视角
- HUD信息显示：
  - 当前车速（km/h）
  - 前方障碍物距离（米）
  - 检测到的交通标志类型

## 项目结构
maintaining_sign_boards/
  ├── main.py # 主脚本：集成CARLA模拟、YOLO检测、车辆控制和可视化 
  └── README.md # 项目说明文档

## 核心模块说明

### `main.py` 关键函数

- **`process_image()`**：将CARLA摄像头原始BGRA数据转换为RGB格式
- **`detect_traffic_signs()`**：使用YOLOv8检测交通标志，返回标志类别、置信度和边界框
- **`get_steering_angle()`**：计算车辆到目标路点的转向角度
- **`detect_front_obstacle()`**：检测前方障碍物并返回最近距离
- **`control_vehicle_based_on_sign()`**：综合交通标志、红绿灯、障碍物信息进行车辆控制决策
- **`spawn_dynamic_elements()`**：在地图上生成限速标志（20/40/60 km/h）和停车标志
- **`main()`**：主循环，整合所有模块实现完整的自动驾驶流程

## 运行环境

### Python依赖

确保你的Python环境安装了以下内容：

```bash
pip install pygame numpy torch ultralytics carla
```

**注意**：`carla`包需要与CARLA模拟器版本匹配。如果使用CARLA 0.9.13+，建议从CARLA官方提供的Python API安装

方法1：使用CARLA自带的Python API
```bash
cd <CARLA_ROOT>/PythonAPI/carla/dist pip install carla-<version>.whl
```


方法2：直接安装（可能版本不匹配）
```bash
pip install carla
```

### 附加需求

1. **CARLA模拟器**（版本 ≥ 0.9.13）：从[CARLA GitHub](https://github.com/carla-simulator/carla)下载
   
2. **CUDA**（可选）：如果有NVIDIA GPU，YOLOv8推理速度会显著提升
   - 需要安装对应版本的CUDA和cuDNN
   - PyTorch会自动检测并使用GPU

3. **Python版本**：≥ 3.7（推荐3.8-3.11）

4. **YOLOv8模型**：首次运行时会自动下载`yolov8n.pt`模型文件（约6MB）

## 运行方式

### 步骤1：启动CARLA模拟器

```bash
Windows
<Carla安装目录>\CarlaUE4.exe

Linux
<Carla安装目录>/CarlaUE4.sh

```
等待模拟器完全加载（看到城市地图和UI界面）。

### 步骤2：运行Python脚本

```bash
请使用你实际保存的目录
cd D:\pythoncode\nn\src\maintaining_sign_boards
python main.py
```
### 步骤3：观察运行效果

- Pygame窗口会弹出，显示车辆摄像头的实时画面
- HUD左上角显示：
  - **Speed**：当前车速
  - **Obstacle**：前方障碍物距离（红色=有障碍，绿色=无障碍）
  - **Sign**：检测到的交通标志
- 控制台会输出详细的检测和控车日志
- 按`ESC`键或关闭窗口退出程序

## 配置说明

### 可调整参数

在`main.py`中可以修改以下参数：

```python
# YOLO检测置信度阈值（第30行）
conf=0.5 # 降低可检测更多标志，但可能增加误检

# 障碍物检测范围（第63行）
max_distance=50.0 # 最大检测距离（米） angle_threshold=35.0 # 检测角度范围（度）

# 安全距离公式（第130行）
safe_distance = current_speed * 0.6 + 10.0 # 调整系数改变刹车灵敏度

# 默认油门（第340行）
final_control.throttle = 0.3 # 无标志时的基础速度

# 转向控制参数（第319-333行）
raw_steer = angle * 0.7 # 转向灵敏度 steer_delta = np.clip(steer_delta, -0.07, 0.07) # 每帧最大转向变化 steer = np.clip(steer, -0.7, 0.7) # 最大转向角度
```


## 技术细节

### 图像处理流程
1. CARLA摄像头输出BGRA格式原始数据
2. 转换为NumPy数组并提取BGR通道
3. BGR转RGB供YOLO模型使用
4. YOLOv8推理（640×640输入尺寸）
5. 解析检测结果并过滤置信度>0.5的目标

### 控制优先级
车辆控制决策按以下优先级执行：
1. **障碍物避让**（最高优先级）：立即分级刹车
2. **红灯停车**：平滑减速至停止
3. **STOP标志**：减速停车并等待2秒
4. **限速标志**：调整车速至限制值
5. **默认巡航**：油门0.3 + 自动转向

### 性能优化
- 预计算共用变量（车速、障碍物距离）避免重复调用
- YOLO模型仅在摄像头图像可用时运行
- 帧率限制为30FPS保证稳定性
- 使用轻量级YOLOv8n模型平衡速度和精度

## 常见问题

### Q: 提示"Connection refused"错误
A: 确保CARLA模拟器已启动并在localhost:2000端口监听。

### Q: YOLO检测速度慢
A: 
- 检查是否启用GPU：`torch.cuda.is_available()`应返回`True`
- 降低输入分辨率：修改`imgsz=640`为更小值（如480）
- 使用更快的模型：替换为`yolov8n.pt`（已使用）

### Q: 车辆无法生成
A: 
- 检查固定坐标是否在道路上
- 查看控制台日志确认是否切换到随机生成点
- 确保CARLA地图已完全加载

### Q: 检测不到交通标志
A: 
- 降低置信度阈值（第30行的`conf=0.5`改为0.3）
- 确保地图上有生成的标志（查看控制台"生成了限速XX标志"日志）
- 调整摄像头位置和FOV（第280-284行）

## 扩展方向

- 添加更多交通标志类型（让行、禁止通行等）
- 集成车道线检测和保持
- 实现多车辆协同避障
- 添加行人检测和避让
- 使用强化学习优化控制策略

## 许可证

本项目仅供学习和研究使用。CARLA模拟器遵循MIT许可证。
