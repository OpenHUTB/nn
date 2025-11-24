CARLA无人车障碍物识别系统
3. 数据记录
python
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