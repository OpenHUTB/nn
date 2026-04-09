#!/usr/bin/env python3
"""
手势控制节点 - 简化版本，避免复杂的机器学习依赖
"""
import rclpy
from rclpy.node import Node
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class SimpleGestureDetector:
    """简化的手势检测器，避免复杂的依赖"""
    def __init__(self):
        self.gesture_commands = {
            "open_palm": "takeoff",
            "closed_fist": "land",
            "pointing_up": "up",
            "pointing_down": "down",
            "victory": "forward",
            "thumb_up": "backward",
            "thumb_down": "stop",
            "ok_sign": "hover",
        }
    
    def detect_gestures(self, image, simulation_mode=False):
        """模拟手势检测"""
        import numpy as np
        import cv2
        
        # 在图像上绘制虚拟手势信息
        height, width = image.shape[:2]
        cv2.putText(image, "虚拟手势检测模式", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "按 'q' 退出", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 模拟手势
        gestures = list(self.gesture_commands.keys())
        import random
        gesture = random.choice(gestures)
        confidence = random.uniform(0.7, 0.95)
        
        cv2.putText(image, f"手势: {gesture}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"置信度: {confidence:.2f}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image, gesture, confidence, None
    
    def get_command(self, gesture):
        return self.gesture_commands.get(gesture, "none")

class SimpleDroneController:
    """简化的无人机控制器"""
    def __init__(self, simulation_mode=True):
        self.simulation_mode = simulation_mode
        self.position = [0.0, 0.0, 0.0]
        self.battery = 100.0
        self.armed = False
        self.mode = "DISARMED"
    
    def send_command(self, command, intensity=1.0):
        print(f"[仿真] 执行命令: {command}, 强度: {intensity}")
        
        if command == "takeoff":
            self.armed = True
            self.mode = "TAKEOFF"
            self.position[1] = 2.0  # 起飞到2米高度
        elif command == "land":
            self.mode = "LAND"
            self.position[1] = 0.0
            self.armed = False
        elif command == "up":
            self.position[1] += 0.5 * intensity
        elif command == "down":
            self.position[1] = max(0, self.position[1] - 0.5 * intensity)
        elif command == "forward":
            self.position[2] += 0.5 * intensity
        elif command == "backward":
            self.position[2] -= 0.5 * intensity
        elif command == "left":
            self.position[0] -= 0.5 * intensity
        elif command == "right":
            self.position[0] += 0.5 * intensity
        elif command == "hover":
            self.mode = "HOVER"
        elif command == "stop":
            self.mode = "STOP"
    
    def get_state(self):
        return {
            'position': self.position,
            'battery': self.battery,
            'armed': self.armed,
            'mode': self.mode
        }

from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time

class GestureControlNode(Node):
    def __init__(self):
        super().__init__('gesture_control_node')
        
        # ROS参数
        self.declare_parameter('simulation_mode', True)
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('command_cooldown', 1.5)
        
        # 获取参数
        simulation_mode = self.get_parameter('simulation_mode').value
        camera_id = self.get_parameter('camera_id').value
        self.command_cooldown = self.get_parameter('command_cooldown').value
        
        # 使用简化的组件
        self.gesture_detector = SimpleGestureDetector()
        self.drone_controller = SimpleDroneController(simulation_mode)
        
        # ROS发布器
        self.gesture_pub = self.create_publisher(String, 'gesture_detected', 10)
        self.command_pub = self.create_publisher(String, 'gesture_command', 10)
        self.velocity_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # 定时器
        self.gesture_timer = self.create_timer(0.5, self.gesture_callback)
        self.status_timer = self.create_timer(2.0, self.status_callback)
        
        # 状态
        self.current_gesture = "none"
        self.gesture_confidence = 0.0
        self.last_command_time = time.time()
        
        self.get_logger().info('🎮 手势控制节点已启动 (简化版本)')
        self.get_logger().info(f'📊 模式: {"仿真" if simulation_mode else "真实"}')
        self.get_logger().info('💡 提示: 这是简化版本，用于测试ROS2包结构')
    
    def gesture_callback(self):
        """手势检测回调"""
        try:
            # 创建虚拟图像
            import numpy as np
            import cv2
            
            # 创建虚拟摄像头图像
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
            cv2.putText(frame, "虚拟摄像头模式", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "手势指令:", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
            cv2.putText(frame, "按 'q' 退出", (50, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 检测手势
            processed_frame, gesture, confidence, _ = \
                self.gesture_detector.detect_gestures(frame, simulation_mode=True)
            
            # 更新状态
            self.current_gesture = gesture
            self.gesture_confidence = confidence
            
            # 发布手势
            gesture_msg = String()
            gesture_msg.data = f"{gesture}:{confidence:.2f}"
            self.gesture_pub.publish(gesture_msg)
            
            # 处理命令
            current_time = time.time()
            if current_time - self.last_command_time > self.command_cooldown:
                if confidence > 0.7:
                    command = self.gesture_detector.get_command(gesture)
                    if command != "none":
                        # 执行命令
                        intensity = min(max(confidence, 0.5), 1.0)
                        self.drone_controller.send_command(command, intensity)
                        
                        # 发布命令
                        command_msg = String()
                        command_msg.data = command
                        self.command_pub.publish(command_msg)
                        
                        # 发布速度指令
                        if command in ['forward', 'backward', 'up', 'down', 'left', 'right']:
                            twist = Twist()
                            speed = 0.3 * intensity
                            
                            if command == 'forward':
                                twist.linear.x = speed
                            elif command == 'backward':
                                twist.linear.x = -speed
                            elif command == 'up':
                                twist.linear.z = speed
                            elif command == 'down':
                                twist.linear.z = -speed
                            elif command == 'left':
                                twist.linear.y = speed
                            elif command == 'right':
                                twist.linear.y = -speed
                            
                            self.velocity_pub.publish(twist)
                        
                        self.last_command_time = current_time
                        self.get_logger().info(f'🎯 手势: {gesture} -> 命令: {command}')
            
            # 显示图像
            cv2.imshow('Gesture Control (Virtual Mode)', processed_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'手势检测错误: {e}')
    
    def status_callback(self):
        """状态更新回调"""
        state = self.drone_controller.get_state()
        self.get_logger().info(
            f'📊 状态 | 位置: ({state["position"][0]:.1f}, {state["position"][1]:.1f}, {state["position"][2]:.1f}) | '
            f'电池: {state["battery"]:.1f}% | 模式: {state["mode"]}'
        )
    
    def destroy_node(self):
        """清理资源"""
        import cv2
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GestureControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 收到中断信号，正在关闭...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
