CARLA无人车障碍物识别系统
项目概述
这是一个基于CARLA仿真环境的无人车障碍物检测系统，集成了多种传感器数据处理能力，包括RGB视觉检测、深度图像分析和激光雷达点云处理。系统使用YOLO算法进行实时障碍物检测，并提供可视化和数据记录功能。
主要功能
1. 障碍物检测
基于YOLO算法的实时物体检测
支持多种障碍物类型：车辆、行人、自行车、摩托车、公交车、卡车、交通标识等
可配置置信度阈值和检测参数
风险等级评估（高、中、低风险）
传感器数据处理
RGB相机: 视觉图像采集和处理
深度相机: 距离测量和3D定位
语义分割相机: 场景理解和语义标注
激光雷达: 3D点云数据采集和处理
数据记录和分析
实时检测数据记录
性能统计分析
障碍物行为分析
驾驶建议生成
4. 可视化界面
实时检测结果显示
多传感器数据可视化
深度图和激光雷达点云显示
交互式操作界面
系统架构
复制
carla_obstacle_detection/
├── src/                          # 源代码
│   ├── carla_obstacle_detection.py  # 主程序
│   ├── obstacle_detector.py         # 障碍物检测器
│   ├── sensor_manager.py            # 传感器管理器
│   ├── data_logger.py               # 数据记录器
│   └── visualizer.py                # 可视化器
├── config/                        # 配置文件
│   └── sensor_config.yaml          # 传感器配置
├── data/                          # 数据文件
│   ├── test_images/               # 测试图像
│   └── models/                    # 模型文件
├── tests/                         # 测试代码
│   └── test_detector.py           # 检测器测试
├── examples/                      # 示例代码
│   └── demo.py                    # 演示程序
├── output/                        # 输出结果
└── logs/                          # 日志文件
安装要求
Python依赖
复制
pip install numpy opencv-python torch torchvision
pip install PyYAML carla
pip install matplotlib seaborn  # 用于可视化
CARLA仿真环境
CARLA 0.9.13 或更高版本
Python 3.7+
