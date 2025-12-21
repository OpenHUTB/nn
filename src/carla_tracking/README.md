# CARLA_YOLO_ObjectTracking

## 项目概述
本项目实现了基于YOLOv5算法和CARLA仿真器的自动驾驶目标检测与跟踪功能。通过CARLA仿真环境生成动态交通场景，利用YOLOv5模型实时检测车辆、摩托车、公交车、卡车等交通参与者，结合内置优化版SORT跟踪算法实现目标的连续追踪，并通过OpenCV实时可视化检测与跟踪结果。

## 核心功能
- 基于CARLA仿真器构建真实交通场景
- 集成YOLOv5模型实现多类别交通目标检测
- 优化版SORT跟踪算法实现目标连续追踪，支持ID分配与轨迹预测
- 深度相机数据融合，实现目标距离估计与速度计算
- 实时可视化检测结果，包括边界框、置信度、跟踪ID、距离和速度信息

## 安装步骤
### 操作步骤：

1. 下载并安装CARLA仿真器：
```plaintext
https://github.com/carla-simulator/carla/releases
```
选择适合操作系统的版本，解压至任意路径（如D:\CARLA）

2. 克隆或下载本项目代码：
```bash
git clone <项目仓库地址>
cd CARLA_YOLO_ObjectTracking
```

3. 安装Python依赖库：
```bash
# 安装PyTorch（支持CUDA加速）
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# 安装YOLOv5依赖
pip install ultralytics

# 安装其他依赖
pip install numpy opencv-python scipy carla

# 若使用CPU版本PyTorch
pip install torch torchvision
```

4. 安装CARLA Python API：
```bash
# 进入CARLA安装目录下的PythonAPI路径
cd D:\CARLA\PythonAPI\carla\dist
# 安装对应版本的egg文件（需与Python版本匹配）
easy_install carla-<版本号>-py3.7-<平台>.egg
```

5. 下载YOLOv5预训练模型：
将模型文件（yolov5s.pt/yolov5su.pt/yolov5m.pt/yolov5mu.pt/yolov5x.pt）放置在指定路径（默认路径为`D:\yolo\`，可在代码`load_detection_model`函数中修改）

## 使用说明
### 基本运行
```bash
python main.py
```

### 可选参数
```bash
# 选择检测模型
python main.py --model yolov5mu

# 调整置信度阈值
python main.py --conf-thres 0.2

# 调整NPC车辆数量
python main.py --npc-count 50

<<<<<<< HEAD
# 禁用深度相机
python main.py --use-depth False
```

### 交互控制
- 按 `q` 键：退出程序
- 按 `r` 键：重新生成NPC车辆

## 核心算法说明
### 1. 优化版SORT跟踪算法
- 基于卡尔曼滤波器的运动预测，考虑目标距离动态调整过程噪声
- 结合匈牙利算法实现检测框与跟踪轨迹的最优匹配
- 引入深度信息加权的IOU计算，提升远距离目标跟踪稳定性
- 动态IOU阈值调整，根据目标距离自适应匹配严格程度

### 2. 目标距离与速度估计
- 利用深度相机数据计算目标距离
- 通过距离变化率估算目标相对速度
- 距离历史平滑处理，降低测量噪声影响

### 3. NPC生成策略
- 优先在主车辆周围10-50米范围内生成NPC
- 避免NPC车辆过度拥挤（最小间距8米）
- 分阶段生成策略，确保目标数量充足
- 交通管理器参数优化，实现更真实的车辆行为

## 可视化说明
可视化窗口将显示以下信息：
- 目标边界框（颜色随距离变化：红色<15m，黄色15-30m，绿色30-50m）
- 目标类别、置信度、跟踪ID
- 目标距离（单位：米）和相对速度（单位：米/秒）
- 实时FPS、帧数、跟踪数量等统计信息

2. 目标跟踪：
- 采用优化版SORT跟踪算法，核心改进包括：
  - 卡尔曼滤波器参数优化，基于目标距离动态调整过程噪声，适配车辆运动特性
  - 引入历史位置平滑处理，远距离目标采用更高平滑权重，减少跟踪抖动
  - 基于目标距离动态调整IOU匹配阈值，增强不同距离目标的跟踪稳定性
  - 基于距离的尺寸过滤策略，适应透视变换影响
  - 对远距离目标延长保留时间，减少频繁消失与重现
- 跟踪核心参数：max_age（最大消失帧数，默认8）、min_hits（最小命中数，默认3）、iou_threshold（基础IOU匹配阈值，默认0.4）
- 跟踪结果包含持续的目标ID、距离信息和速度估计，支持目标短暂消失后重新出现的ID匹配

3. CARLA交互：
- 自动清理仿真环境中的残留车辆和静态车辆，保证场景纯净
- 主车辆开启自动驾驶模式，相机优化挂载于主车辆前方（x=1.8m，z=1.6m，俯角-5°），更适合前方车辆检测
- 自动生成NPC车辆并开启自动驾驶，构建动态交通场景（数量可通过--npc-count配置，默认15辆）
- 主车辆生成优化：优先选择交通密度高的位置生成，提升场景真实性
- spectator视角自动跟随主车辆（后上方8m，z=10m，俯角-35°），方便观察全局场景
- 同步模式运行，固定帧率0.05s/帧（20FPS），保证检测稳定性
- 深度相机与RGB相机精确同步，提供目标距离信息用于优化跟踪

## 项目结构
- main.py：主程序脚本，包含以下核心组件：
  - KalmanFilter：卡尔曼滤波器实现，用于目标运动状态预测与更新，支持基于距离的参数动态调整
  - Track：跟踪目标实体类，维护单目标的边界框、ID、生命周期、历史位置、距离及速度等信息
  - Sort：SORT跟踪器核心实现，处理多目标匹配与跟踪状态管理，集成深度信息优化
  - 工具函数：包含边界框绘制（支持类别差异化颜色）、相机投影矩阵构建、CARLA环境清理等功能
  - CARLA交互模块：负责服务器连接、车辆生成（含智能选址）、相机配置及同步控制
  - 检测与跟踪主逻辑：集成YOLOv5检测与SORT跟踪，实现端到端流程
- YOLOv5预训练模型（yolov5s.pt/yolov5m.pt/yolov5x.pt）：需放置在指定路径


## 常见问题
### CARLA连接失败：
- 确保CARLA仿真器已启动，且端口与脚本参数一致（默认2000）
- 检查CARLA版本与Python API版本是否匹配
- 关闭防火墙或添加端口例外规则

### 模型加载错误：
- 确认模型文件已下载并放在正确路径（默认`D:\yolo\`）
- 检查模型文件名与代码中`model_paths`字典配置一致
- 网络问题导致模型自动下载失败时，可手动下载并放置到指定路径

### 可视化窗口无响应：
- 确保OpenCV版本兼容（推荐4.5.x系列）
- 检查是否存在未处理的异常导致主循环中断
- 尝试重新启动CARLA仿真器和程序

## 参考文档
- [CARLA官方文档](https://carla.readthedocs.io/)
- [YOLOv5官方文档](https://docs.ultralytics.com/yolov5/)
- [SORT跟踪算法论文](https://arxiv.org/abs/1602.00763)