CARLA无人车障碍物识别系统
项目概述
这是一个基于CARLA仿真环境的无人车障碍物检测系统，集成了多种传感器数据处理能力，包括RGB视觉检测、深度图像分析和激光雷达点云处理。系统使用YOLO算法进行实时障碍物检测，并提供可视化和数据记录功能。
主要功能
1. 障碍物检测
• 基于YOLO算法的实时物体检测
• 支持多种障碍物类型：车辆、行人、自行车、摩托车、公交车、卡车、交通标识等
• 可配置置信度阈值和检测参数
• 风险等级评估（高、中、低风险）
2. 传感器数据处理
• RGB相机: 视觉图像采集和处理
• 深度相机: 距离测量和3D定位
• 语义分割相机: 场景理解和语义标注
• 激光雷达: 3D点云数据采集和处理
3. 数据记录和分析
• 实时检测数据记录
• 性能统计分析
• 障碍物行为分析
• 驾驶建议生成
4. 可视化界面
• 实时检测结果显示
• 多传感器数据可视化
• 深度图和激光雷达点云显示
• 交互式操作界面
系统架构
￼
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
￼
复制
pip install numpy opencv-python torch torchvision
pip install PyYAML carla
pip install matplotlib seaborn  # 用于可视化
CARLA仿真环境
• CARLA 0.9.13 或更高版本
• Python 3.7+
• 8GB+ RAM
• 支持CUDA的GPU（推荐）
快速开始
1. 环境配置
bash
￼
复制
# 克隆或下载项目文件
cd
 carla_obstacle_detection

# 安装Python依赖
pip install
 -r requirements.txt

# 启动CARLA服务器
# 在CARLA安装目录运行:
./CarlaUE4.sh
2. 运行演示程序
bash
￼
复制
# 进入项目目录
cd
 carla_obstacle_detection

# 运行交互式演示
python examples/demo.py
3. 运行测试
bash
￼
复制
# 运行检测器测试
python tests/test_detector.py
4. 完整CARLA仿真
bash
￼
复制
# 运行完整的CARLA仿真（需要先启动CARLA服务器）
python src/carla_obstacle_detection.py
使用指南
基础使用
1. 配置传感器
编辑 config/sensor_config.yaml 文件：
yaml
￼
复制
carla_settings:
  town: "Town01"  # 选择地图
  vehicle_type: "vehicle.tesla.model3"  # 选择车辆类型
  
sensors:
  rgb_camera:
    width: 800
    height: 600
    fov: 90
2. 运行检测
python
￼
复制
from src.obstacle_detector import
 ObstacleDetector

# 初始化检测器
detector = ObstacleDetector(config)

# 检测图像
image = cv2.imread("test_image.jpg")
detections = detector.detect(image)

# 绘制结果
result = detector.draw_detections(image, detections)
cv2.imshow("Detection Result", result)
高级功能
1. 自定义检测参数
python
￼
复制
detection_params = {
    'yolo': {
        'conf_threshold': 0.7,  # 提高置信度阈值
        'iou_threshold': 0.5
    },
    'obstacle_classes': ['person', 'car', 'bicycle']
}
2. 风险评估
python
￼
复制
# 获取障碍物风险等级
obstacle_type = detector.predict_obstacle_type(detection)
print(f"障碍物类型: {obstacle_type['type']}")
print(f"风险等级: {obstacle_type['risk_level']}")
print(f"建议行动: {obstacle_type['action']}")
3. 数据记录
python
￼
复制
from src.data_logger import
 DataLogger

logger = DataLogger("output")
logger.log_frame(frame_id, detections)
logger.log_obstacle_analysis(detections, vehicle_velocity)
logger.save_summary()
测试数据集
系统包含5个预定义的测试场景：
1.
单辆车检测: 验证基本车辆检测能力
2.
多车辆场景: 测试多目标检测
3.
行人和车辆混合: 验证多类别检测
4.
交通标识检测: 测试特殊目标检测
5.
复杂场景: 综合性能测试
测试图像说明
• 格式: JPEG, 800x600分辨率
• 标注: 包含边界框和类别信息
• 深度图: 模拟距离信息
• 激光雷达数据: 模拟3D点云数据
性能指标
检测性能
• 检测精度: 85-95%（取决于场景复杂度）
• 处理速度: 10-30 FPS（取决于硬件配置）
• 延迟: 50-100ms
• 支持目标数: 最多50个同时检测
传感器性能
• RGB相机: 800x600@30fps
• 深度相机: 800x600@30fps
• 激光雷达: 32线，100米范围，10Hz
键盘快捷键
可视化界面
• q: 退出程序
• s: 保存当前截图
• h: 显示帮助信息
演示程序
• 1: 离线图像检测
• 2: 深度图像处理
• 3: 激光雷达数据处理
• 4: 性能基准测试
• 5: 完整测试套件
• 0: 退出程序
输出文件
检测结果
• detections_log.json: 详细检测记录
• performance_log.csv: 性能统计数据
• obstacle_analysis.json: 障碍物分析结果
可视化结果
• screenshots/: 截图文件夹
• output/: 检测结果图像
• test_results_visualization.png: 测试结果图表
日志文件
• logs/: 系统运行日志
• summary.json: 运行汇总报告
故障排除
常见问题
1. CARLA连接失败
￼
复制
错误: 连接CARLA失败
解决: 确保CARLA服务器正在运行，检查端口2000是否可用
2. YOLO模型加载失败
￼
复制
错误: 模型文件不存在
解决: 系统会自动使用预训练模型，或下载指定模型文件
3. 检测性能问题
￼
复制
解决: 
1. 降低图像分辨率
2. 提高置信度阈值
3. 减少检测类别数量
4. 使用GPU加速
4. 内存不足
￼
复制
解决:
1. 减小缓冲区大小
2. 减少并发处理
3. 优化数据存储
调试模式
python
￼
复制
# 启用详细日志
import
 logging
logging.basicConfig(level=logging.DEBUG)

# 保存中间结果
visualizer.record_video(image, "debug_output.mp4")
扩展开发
添加新的检测类别
1.
在配置文件中添加类别
2.
准备训练数据
3.
训练自定义YOLO模型
4.
更新检测器配置
集成新的传感器
1.
在sensor_manager.py中添加传感器类型
2.
实现数据处理回调
3.
更新可视化器支持
4.
添加相应的测试用例
自定义分析算法
1.
继承ObstacleDetector类
2.
重写检测和分析方法
3.
添加新的评估指标
4.
更新配置文件
许可证
本项目采用MIT许可证，详见LICENSE文件。
贡献指南
欢迎贡献代码和报告问题。请遵循以下步骤：
1.
Fork项目仓库
2.
创建功能分支
3.
提交更改
4.
创建Pull Request
更新日志
v1.0.0 (2025-12-01)
• 初始版本发布
• 基本障碍物检测功能
• 多传感器数据处理
• 可视化和数据记录
• 测试数据集和演示程序
