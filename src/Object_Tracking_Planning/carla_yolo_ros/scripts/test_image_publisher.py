#!/usr/bin/env python3
"""
测试图像发布器 - 为ROS节点提供测试图像
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

class TestImagePublisher:
    def __init__(self):
        rospy.init_node('test_image_publisher', anonymous=True)
        
        # 参数
        self.publish_rate = rospy.get_param('~publish_rate', 5)  # Hz
        self.image_width = rospy.get_param('~width', 640)
        self.image_height = rospy.get_param('~height', 480)
        
        # 发布者
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
        self.status_pub = rospy.Publisher('/test/status', String, queue_size=10)
        
        self.bridge = CvBridge()
        self.frame_count = 0
        
        rospy.loginfo(f"Test Image Publisher initialized: {self.image_width}x{self.image_height} @ {self.publish_rate}Hz")
    
    def create_test_image(self):
        """创建测试图像"""
        # 创建黑色背景
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # 添加一些模拟对象
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        objects = ['Car', 'Person', 'Bicycle', 'Traffic Light']
        
        for i, (color, obj) in enumerate(zip(colors, objects)):
            # 计算位置
            x = 100 + i * 120
            y = 100 + (i % 2) * 150
            
            # 绘制对象
            cv2.rectangle(image, (x, y), (x + 80, y + 60), color, 2)
            cv2.putText(image, obj, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 添加检测标记
            cv2.circle(image, (x + 40, y + 30), 5, color, -1)
        
        # 添加帧计数和水印
        cv2.putText(image, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "Test Image for YOLO Detection", (10, self.image_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        self.frame_count += 1
        return image
    
    def publish_status(self, message):
        """发布状态"""
        from std_msgs.msg import String
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
    
    def run(self):
        """主循环"""
        rate = rospy.Rate(self.publish_rate)
        
        rospy.loginfo("Starting test image publishing...")
        
        while not rospy.is_shutdown():
            try:
                # 创建测试图像
                image = self.create_test_image()
                
                # 转换为ROS消息
                ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = "test_camera"
                
                # 发布图像
                self.image_pub.publish(ros_image)
                
                # 发布状态
                self.publish_status(f"Published frame {self.frame_count}")
                
                rospy.logdebug(f"Published test image frame {self.frame_count}")
                
                rate.sleep()
                
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr(f"Error publishing image: {e}")
                time.sleep(1.0)
        
        rospy.loginfo("Test Image Publisher stopped")

def main():
    publisher = TestImagePublisher()
    publisher.run()

if __name__ == '__main__':
    main()
