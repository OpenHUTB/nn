#!/usr/bin/env python3
"""
测试版 - CARLA多目标跟踪系统（ROS版）
简化版本用于测试ROS封装
"""

import sys
import os
import time
import argparse

# ======================== ROS支持 ========================
try:
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String, Float32
    from cv_bridge import CvBridge
    import cv2
    import numpy as np
    ROS_AVAILABLE = True
    print("✅ ROS模块导入成功")
except ImportError as e:
    ROS_AVAILABLE = False
    print(f"⚠️  ROS模块导入失败: {e}")

# ======================== 简单日志类 ========================
class SimpleLogger:
    def info(self, msg): print(f"[INFO] {msg}")
    def warning(self, msg): print(f"[WARN] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")

logger = SimpleLogger()

# ======================== 模拟传感器类 ========================
class MockSensorManager:
    def __init__(self):
        logger.info("初始化模拟传感器管理器")
    
    def get_image(self):
        # 创建一个简单的测试图像
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, 'CARLA Tracking ROS', (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img
    
    def get_detections(self):
        # 模拟检测结果
        return [
            {'id': 1, 'bbox': [100, 100, 200, 200], 'class': 'vehicle'},
            {'id': 2, 'bbox': [300, 150, 400, 250], 'class': 'pedestrian'}
        ]

# ======================== 模拟跟踪器类 ========================
class MockTracker:
    def __init__(self):
        logger.info("初始化模拟多目标跟踪器")
        self.track_id = 0
    
    def update(self, detections):
        self.track_id += 1
        return [{'id': self.track_id, 'detection': d} for d in detections]

# ======================== ROS发布器类 ========================
class ROSPublisher:
    def __init__(self):
        if ROS_AVAILABLE:
            self.bridge = CvBridge()
            self.image_pub = rospy.Publisher('/carla/camera', Image, queue_size=10)
            self.detection_pub = rospy.Publisher('/carla/detections', String, queue_size=10)
            self.status_pub = rospy.Publisher('/carla/status', String, queue_size=10)
            logger.info("ROS发布器初始化完成")
        else:
            logger.warning("ROS不可用，发布器未初始化")
    
    def publish_image(self, cv_image):
        if ROS_AVAILABLE and hasattr(self, 'image_pub'):
            try:
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                self.image_pub.publish(ros_image)
                return True
            except Exception as e:
                logger.warning(f"发布图像失败: {e}")
        return False
    
    def publish_detection(self, detections):
        if ROS_AVAILABLE and hasattr(self, 'detection_pub'):
            det_str = str(detections)
            self.detection_pub.publish(det_str)
            return True
        return False
    
    def publish_status(self, status):
        if ROS_AVAILABLE and hasattr(self, 'status_pub'):
            self.status_pub.publish(status)
            return True
        return False

# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(description='CARLA多目标跟踪系统ROS测试版')
    parser.add_argument('--mode', choices=['simulation', 'test'], default='test',
                       help='运行模式：simulation（仿真）或 test（测试）')
    parser.add_argument('--duration', type=int, default=10,
                       help='运行时长（秒）')
    parser.add_argument('--rate', type=float, default=2.0,
                       help='发布频率（Hz）')
    parser.add_argument('--no-ros', action='store_true',
                       help='禁用ROS功能')
    
    args = parser.parse_args()
    
    # 初始化ROS
    if ROS_AVAILABLE and not args.no_ros:
        rospy.init_node('carla_tracking_system')
        logger.info("ROS节点初始化: carla_tracking_system")
    
    # 创建组件
    sensor_manager = MockSensorManager()
    tracker = MockTracker()
    ros_publisher = ROSPublisher() if not args.no_ros else None
    
    logger.info(f"启动CARLA多目标跟踪系统ROS测试版")
    logger.info(f"模式: {args.mode}, 时长: {args.duration}秒, 频率: {args.rate}Hz")
    logger.info(f"ROS支持: {'启用' if ros_publisher else '禁用'}")
    
    # 主循环
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < args.duration:
            # 获取数据
            image = sensor_manager.get_image()
            detections = sensor_manager.get_detections()
            tracks = tracker.update(detections)
            
            # 发布ROS消息
            if ros_publisher:
                ros_publisher.publish_image(image)
                ros_publisher.publish_detection(detections)
                ros_publisher.publish_status(f"Frame {frame_count}: {len(tracks)} tracks")
            
            # 显示信息
            logger.info(f"帧 {frame_count}: 检测到 {len(detections)} 个目标, 跟踪 {len(tracks)} 个轨迹")
            
            frame_count += 1
            time.sleep(1.0 / args.rate)
            
    except KeyboardInterrupt:
        logger.info("收到退出信号")
    except Exception as e:
        logger.error(f"运行错误: {e}")
    
    logger.info(f"系统运行完成，共处理 {frame_count} 帧")
    logger.info(f"平均帧率: {frame_count / args.duration:.2f} Hz")
    logger.info("系统关闭")

if __name__ == '__main__':
    main()
