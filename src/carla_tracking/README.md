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