import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import threading
import time

class ROSHandler(threading.Thread):
    def __init__(self, stabilizer):
        super().__init__(daemon=True)
        self.stabilizer = stabilizer
        self.running = True
        self.has_ros = False
        self.rospy = None
        self.Twist = None

        # 尝试初始化ROS
        try:
            import rospy
            from geometry_msgs.msg import Twist
            self.rospy = rospy
            self.Twist = Twist
            self.has_ros = True
            
            # 初始化ROS节点
            if not self.rospy.core.is_initialized():
                self.rospy.init_node('humanoid_cmd_vel_listener', anonymous=True)
            
            # 订阅/cmd_vel话题
            self.sub = self.rospy.Subscriber(
                "/cmd_vel", 
                self.Twist, 
                self._cmd_vel_callback, 
                queue_size=1, 
                tcp_nodelay=True
            )
            print("[ROS提示] 已启动/cmd_vel话题监听")
        except ImportError:
            print("[ROS提示] 未检测到ROS环境，跳过/cmd_vel话题监听")
        except Exception as e:
            print(f"[ROS提示] ROS初始化失败：{e}")
            self.has_ros = False

    def _cmd_vel_callback(self, msg):
        """处理/cmd_vel话题消息"""
        # 提取速度和转向指令
        linear_x = msg.linear.x  # 前进/后退（-1~1）
        angular_z = msg.angular.z  # 左转/右转（-1~1）
        
        # 设置到机器人控制器
        self.stabilizer.set_velocity(linear_x)
        self.stabilizer.set_turn_rate(angular_z)
        
        # 打印日志
        gait_mode = self.stabilizer.gait_mode
        print(f"[ROS指令] 速度={linear_x:.2f} | 转向={angular_z:.2f} | 步态={gait_mode}")

    def run(self):
        """ROS监听线程主循环（修复spin_once错误）"""
        if not self.has_ros:
            return
        
        # ROS 1兼容的周期监听（替代spin_once）
        rate = self.rospy.Rate(100)  # 100Hz刷新频率
        while self.running and not self.rospy.is_shutdown():
            try:
                rate.sleep()  # 非阻塞式等待，替代spin_once
            except Exception as e:
                print(f"[ROS提示] 监听线程异常：{e}")
                break

    def stop(self):
        """停止监听线程"""
        self.running = False
        if self.has_ros:
            self.rospy.signal_shutdown("Simulation stopped")
        print("[ROS提示] 监听线程已停止")
