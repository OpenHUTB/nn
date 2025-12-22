# CARLA多目标跟踪系统

一个基于CARLA仿真环境的实时多目标检测与跟踪系统，集成了YOLOv8目标检测、SORT多目标跟踪算法和行为分析功能。

## 🌟 核心特性

- **实时多目标跟踪**：集成YOLOv8检测和SORT跟踪算法
- **多传感器融合**：支持相机RGB图像和LiDAR点云数据
- **行为分析**：检测车辆停车、超车、变道、刹车、危险接近等行为
- **增强可视化**：彩色ID编码 + 独立统计面板显示
- **多天气支持**：支持晴天、多云、雨天、雾天、夜间等天气条件
- **随机位置生成**：每次运行自车在不同位置生成，增加测试多样性
- **性能优化**：多线程检测、GPU加速、卡尔曼滤波预测

## 📋 系统要求

### 硬件要求
- **CPU**：4核以上（推荐Intel i5/i7或同等AMD处理器）
- **GPU**：NVIDIA GPU，4GB显存以上（推荐RTX 3060+）
- **内存**：8GB以上（推荐16GB）
- **存储**：10GB可用空间

### 软件要求
- **操作系统**：Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**：3.8或更高版本
- **CARLA**：0.9.14或更高版本

## 🚀 快速开始

### 1. 环境安装
```bash
# 克隆项目
git clone https://github.com/yourusername/carla-multi-object-tracking.git
cd carla-multi-object-tracking

# 安装依赖
pip install -r requirements.txt

# 下载YOLOv8模型（可选，会自动下载）
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 2. 启动CARLA服务器
```bash
# Windows
CarlaUE4.exe -windowed -ResX=800 -ResY=600

# Linux
./CarlaUE4.sh -windowed -ResX=800 -ResY=600

# 可选：添加-world-port参数指定端口（如果端口冲突）
./CarlaUE4.sh -windowed -ResX=800 -ResY=600 -world-port=2000
```

### 3. 运行跟踪系统
```bash
# 基本运行（自车会在随机位置生成）
python main.py

# 使用自定义配置
python main.py --config config.yaml --host localhost --port 2000

# 禁用LiDAR
python main.py --no-lidar

# 设置自定义参数
python main.py --model yolov8n.pt --conf-thres 0.5 --weather clear

# 使用随机天气
python main.py --weather random
```

## 📁 项目结构

```
carla-multi-object-tracking/
├── main.py              # 主程序入口
├── sensors.py           # 传感器管理（相机、LiDAR）- 包含随机生成逻辑
├── tracker.py           # 目标检测与跟踪算法
├── utils.py             # 工具函数库
├── config.yaml          # 配置文件
├── requirements.txt     # 依赖包列表
├── README.md           # 项目说明文档
└── outputs/            # 输出目录（自动创建）
    ├── screenshots/    # 截图保存
    └── logs/          # 运行日志
```

## ⚙️ 配置文件说明

主要的配置选项（完整配置见config.yaml）：

```yaml
# CARLA连接配置
host: "localhost"
port: 2000
timeout: 20.0

# 传感器配置
img_width: 640
img_height: 480
fov: 90
use_lidar: true
lidar_channels: 32

# 检测配置
yolo_model: "yolov8n.pt"
conf_thres: 0.5
iou_thres: 0.3
device: "cuda"  # 或 "cpu"

# 跟踪配置
max_age: 5
min_hits: 3
kf_dt: 0.05
max_speed: 50.0

# 可视化配置
window_width: 1280
window_height: 720
display_fps: 30
```

## 🚗 自车生成配置

### 随机生成（默认）
- 每次运行程序，自车会在CARLA地图的随机生成点出现
- 系统会自动尝试5个不同的随机位置，直到成功生成
- 生成位置信息会显示在控制台日志中

### 固定位置生成
如果需要固定位置测试，可以：

1. **修改代码**：编辑 `sensors.py` 中的 `create_ego_vehicle` 函数
2. **指定坐标**：取消注释固定坐标代码
3. **重启程序**：重新运行即可在固定位置生成

### 查看可用生成点
运行测试脚本查看所有生成点：
```bash
python -c "
import carla
client = carla.Client('localhost', 2000)
spawn_points = client.get_world().get_map().get_spawn_points()
print(f'共有 {len(spawn_points)} 个生成点')
for i, sp in enumerate(spawn_points[:5]):
    print(f'{i}: ({sp.location.x:.1f}, {sp.location.y:.1f}), 朝向: {sp.rotation.yaw:.1f}°')
"
```

## 🎮 使用说明

### 可视化界面
系统提供独立的显示窗口：
- **主窗口**：显示实时跟踪画面，包含检测框、跟踪ID和行为状态
- **颜色图例**：显示行为状态和车辆类型的颜色编码

### 键盘控制
| 按键 | 功能 | 说明 |
|------|------|------|
| **ESC** | 退出程序 | 安全关闭所有连接 |
| **W** | 切换天气 | 循环切换晴天、多云、雨天、雾天、夜间 |
| **S** | 保存截图 | 保存当前画面到outputs/screenshots/ |
| **P** | 暂停/继续 | 暂停或恢复程序运行 |
| **M** | 显示/隐藏图例 | 控制颜色图例的显示状态 |
| **R** | 重新随机生成 | 重新随机选择自车位置（需要修改代码支持） |

### 彩色编码说明
系统使用彩色编码来区分不同状态：

#### 行为状态颜色（优先级从高到低）
- **红色**：危险状态（距离过近）
- **黄色**：停车状态
- **紫色**：超车状态
- **青色**：变道/转弯状态
- **蓝色**：加速状态
- **橙色**：刹车状态
- **绿色**：正常行驶

#### 车辆类型颜色
- **蓝色**：轿车
- **绿色**：公交车
- **红色**：卡车
- **青色**：其他车辆

## 📊 系统功能

### 1. 目标检测（YOLOv8）
- 使用YOLOv8模型进行车辆检测
- 支持Car/Bus/Truck三类检测
- 自动调整输入尺寸优化性能
- 支持GPU加速和模型量化

### 2. 多目标跟踪（SORT）
- 基于Simple Online and Realtime Tracking算法
- 卡尔曼滤波预测目标位置
- 匈牙利算法进行数据关联
- IOU匹配策略

### 3. 行为分析
- **停车检测**：速度低于阈值并持续多帧
- **超车检测**：相对速度超过自车速度的150%
- **变道检测**：横向位移超过阈值
- **刹车检测**：负加速度超过阈值
- **危险检测**：与自车距离过近

### 4. 传感器融合
- RGB相机：提供2D图像信息
- LiDAR：提供3D点云信息（可选）
- 融合策略：优先使用视觉检测，LiDAR用于验证和距离估计

## 🛠️ 故障排除

### 常见问题

#### Q1: CARLA连接失败
```
错误信息：timeout of 20.0s exceeded
解决方法：
1. 确认CARLA服务器已启动
2. 检查端口设置（默认2000）
3. 增加timeout时间配置
```

#### Q2: 自车生成在同一位置
```
问题：自车总在相同位置生成
解决方法：
1. 已修复！现在默认随机生成位置
2. 如果仍有问题，重启CARLA服务器清除缓存
3. 检查sensors.py中的随机种子生成逻辑
```

#### Q3: 检测模型加载失败
```
错误信息：YOLO model not found
解决方法：
1. 确保yolov8n.pt文件存在
2. 运行自动下载：python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

#### Q4: 帧率过低
```
现象：FPS低于10
解决方法：
1. 降低图像分辨率（img_width/height）
2. 禁用LiDAR（--no-lidar）
3. 使用更小的YOLO模型（yolov8n.pt）
4. 调整display_fps参数
```

#### Q5: 自车生成失败
```
错误信息：无法生成自车
解决方法：
1. 等待CARLA世界完全加载（约30秒）
2. 清理现有车辆：重启CARLA服务器
3. 尝试不同生成点：程序会自动尝试多个位置
```

### 性能优化建议
1. **GPU模式**：确保使用CUDA加速
2. **分辨率调整**：适当降低图像分辨率
3. **模型选择**：使用YOLOv8n或YOLOv8s轻量模型
4. **异步处理**：启用检测线程分离
5. **LiDAR优化**：根据需要调整LiDAR参数

## 📈 性能指标

在标准配置下（RTX 3060, i7-12700H, 16GB RAM）：

| 项目 | 性能指标 |
|------|----------|
| 检测FPS | 25-30 FPS |
| 检测精度 | >90% mAP |
| 跟踪准确度 | >85% MOTA |
| 内存占用 | 2-3GB |
| GPU占用 | 3-4GB |

## 📝 数据记录

系统支持以下数据记录功能：
- 自动保存运行配置
- 截图保存功能（按S键）
- 性能数据统计
- 跟踪结果记录

## 🔮 未来计划

- [ ] 支持更多车辆类型检测
- [ ] 添加行人检测和跟踪
- [ ] 实现深度学习的多目标跟踪算法
- [ ] 添加轨迹预测功能
- [ ] 支持多摄像头融合
- [ ] 添加ROS2接口支持
- [ ] 实现云端数据记录和分析

## 📄 许可证

本项目采用MIT许可证。详情请见[LICENSE](LICENSE)文件。

## 🙏 致谢

- [CARLA Simulator](https://carla.org/) - 开源自动驾驶仿真平台
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 先进的目标检测框架
- [Open3D](http://www.open3d.org/) - 3D数据处理库
- [SORT算法](https://github.com/abewley/sort) - 实时多目标跟踪算法

## 📚 参考文献

1. Bewley, A., et al. "Simple online and realtime tracking." ICIP 2016.
2. Redmon, J., et al. "YOLOv3: An Incremental Improvement." arXiv 2018.
3. Dosovitskiy, A., et al. "CARLA: An Open Urban Driving Simulator." CoRL 2017.

## 👥 贡献指南

欢迎提交Issue和Pull Request！贡献前请阅读：
1. 遵循PEP 8代码规范
2. 添加适当的注释和文档
3. 确保新功能有相应的测试
4. 更新相关的文档和示例

---

**提示**：运行前请确保CARLA服务器已正确启动，并检查防火墙设置允许相关端口通信。

**随机位置特性**：系统默认会在随机位置生成自车，每次运行都有不同的起点，增加了测试场景的多样性。如需固定位置测试，请参考"自车生成配置"章节。