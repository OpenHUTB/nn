# Traffic Sign Detection

本模块实现了基于 YOLOv8 的实时交通标志检测系统，可在 CARLA 模拟器中实现自动驾驶车辆的交通标志识别与行为控制。

## 功能特性

- 🚦 **实时交通标志检测**：使用 YOLOv8 模型检测 STOP 标志、限速标志等
- 🚗 **车辆行为控制**：根据检测结果自动调整车辆速度和制动
- 🚦 **交通灯识别**：识别红绿灯状态并做出相应反应
- 🖥️ **实时可视化**：使用 Pygame 显示驾驶员视角画面
- 📊 **日志记录**：终端输出检测结果和车辆动作日志

## 项目结构

```
traffic_sign_detection/
├── Main.py           # 主脚本：CARLA仿真与YOLO检测集成
├── config.py         # 配置文件
├── vehicle_control.py # 车辆控制模块
├── detection.py      # YOLO检测模块
└── utils.py          # 工具函数
```

## 环境要求

### 依赖安装
```bash
pip install carla pygame numpy torch ultralytics
```

### 额外要求
- **CARLA Simulator** (版本 ≥ 0.9.13)：从 [CARLA GitHub](https://github.com/carla-simulator/carla) 下载
- **CUDA** (可选)：GPU 加速推理
- **Python ≥ 3.7**

## 预训练模型

| 模型 | 文件 | 来源 | 用途 |
|------|------|------|------|
| YOLOv8n | yolov8n.pt | Ultralytics | 实时交通标志检测 |

加载方式：
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

## 工作原理

### 1. CARLA 设置
- 连接到 CARLA 服务器
- 生成自动驾驶车辆、随机交通和标志
- 挂载 RGB 摄像头到车辆

### 2. YOLOv8 推理
- 摄像头每一帧图像传入 YOLOv8 模型
- 检测标志的边界框和类别标签

### 3. 车辆控制
- 根据限速标志调整车速
- 遇到 STOP 标志完全制动
- 遵守红绿灯状态

### 4. 可视化
- Pygame 显示驾驶员视角
- 终端记录检测和动作事件

## 使用说明

### 启动 CARLA 模拟器
```bash
./CarlaUE4.sh
```

### 运行主脚本
```bash
cd docs/traffic_sign_detection
python Main.py
```

### 停止仿真
仿真运行 2 分钟后自动停止，或按 `Ctrl+C` 强制退出。

## 代码示例

### 主循环流程
```python
import carla
from ultralytics import YOLO
import pygame

# 初始化
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

model = YOLO("yolov8n.pt")

# 主循环
while True:
    # 获取摄像头图像
    image = camera.get_image()
    
    # YOLO 检测
    results = model(image)
    
    # 解析检测结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = box.cls  # 类别
            conf = box.conf  # 置信度
            
            # 根据检测结果控制车辆
            if cls == STOP_SIGN_CLASS:
                brake()
            elif cls == SPEED_LIMIT_CLASS:
                set_speed(limit)
    
    # 更新显示
    update_display(image)
```

## 配置说明

### config.py 主要参数
```python
# 摄像头配置
CAMERA_FOV = 90
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 车辆控制参数
MAX_SPEED = 50  # km/h
STOP_DISTANCE = 5  # 停车距离(米)
BRAKE_INTENSITY = 1.0

# 检测阈值
CONFIDENCE_THRESHOLD = 0.5
```

## 注意事项

1. **模型精度**：默认使用 yolov8n 小型模型，如需更高精度可替换为 yolov8s.pt、yolov8m.pt 等
2. **CARLA 资产**：确保 CARLA 世界包含脚本引用的标志资产
3. **类别适配**：YOLO 默认类别可能需要微调或重新训练以提高标志分类精度

## 扩展功能

- [ ] 添加自定义交通标志数据集训练
- [ ] 支持更多标志类型（让行、禁止通行等）
- [ ] 结合语义分割实现更精确的场景理解
- [ ] 集成路径规划算法

## 参考资料

- [CARLA Simulator Documentation](https://carla.readthedocs.io/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)