#!/usr/bin/env python
# carla_ros_node.py - ROS节点主文件
import rospy
import sys
import os
import signal
import threading
from carla_core import CarlaTrackingCore
from publisher import CarlaPublisher
from subscriber import CarlaSubscriber

class CarlaROSNode:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('carla_tracking_node', anonymous=True)
        rospy.on_shutdown(self.shutdown)
        
        # 获取参数
        config_path = rospy.get_param('~config_path', os.path.join(os.path.dirname(__file__), '../config/carla_config.json'))
        npc_count = rospy.get_param('~npc_count', 20)
        target_speed = rospy.get_param('~target_speed', 30.0)
        
        # 初始化核心模块
        self.core = CarlaTrackingCore(config_path)
        # 更新参数
        self.core.config['npc']['count'] = npc_count
        self.core.config['vehicle']['target_speed'] = target_speed
        
        # 初始化发布器和订阅器
        self.publisher = CarlaPublisher()
        self.subscriber = CarlaSubscriber(self.core)
        
        # 线程控制
        self.running = False
        self.thread = None
        
        # 日志
        rospy.loginfo("CARLA ROS节点初始化完成")

    def start(self):
        """启动节点"""
        # 初始化CARLA
        self.core.initialize()
        
        # 启动运行线程
        self.running = True
        self.thread = threading.Thread(target=self.run_loop)
        self.thread.start()
        
        # 保持节点运行
        rospy.spin()

    def run_loop(self):
        """运行循环"""
        rate = rospy.Rate(30)  # 30Hz
        while self.running and not rospy.is_shutdown():
            try:
                # 单步运行CARLA
                self.core.step()
                
                # 发布结果
                self.publisher.update(self.core.detection_results)
                
                # 日志（每100帧）
                if self.core.frame_count % 100 == 0:
                    rospy.loginfo(f"帧数: {self.core.frame_count} | FPS: {self.core.detection_results['fps']:.1f} | 跟踪目标数: {len(self.core.detection_results['tracks'])}")
                
                rate.sleep()
            except Exception as e:
                rospy.logerr(f"运行循环出错: {e}")

    def shutdown(self):
        """关闭节点"""
        rospy.loginfo("关闭CARLA ROS节点...")
        self.running = False
        if self.thread:
            self.thread.join()
        self.core.cleanup()
        rospy.loginfo("节点已关闭")

if __name__ == '__main__':
    try:
        node = CarlaROSNode()
        node.start()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"节点启动失败: {e}")
        sys.exit(1)
