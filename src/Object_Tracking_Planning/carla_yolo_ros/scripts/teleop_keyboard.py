#!/usr/bin/env python3
"""
键盘遥操作节点 - 用于控制CARLA车辆
"""

import rospy
import sys
import select
import termios
import tty
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class TeleopKeyboard:
    def __init__(self):
        rospy.init_node('teleop_keyboard', anonymous=True)
        
        # 控制参数
        self.linear_speed = rospy.get_param('~linear_speed', 0.5)
        self.angular_speed = rospy.get_param('~angular_speed', 0.5)
        
        # 发布者
        self.cmd_pub = rospy.Publisher('/carla/control_cmd', Twist, queue_size=10)
        self.instruction_pub = rospy.Publisher('/teleop/instructions', String, queue_size=10)
        
        # 保存终端设置
        self.settings = termios.tcgetattr(sys.stdin)
        
        rospy.loginfo("Teleop Keyboard Node Initialized")
        rospy.loginfo("Control keys:")
        rospy.loginfo("  W: Forward")
        rospy.loginfo("  S: Backward")
        rospy.loginfo("  A: Left")
        rospy.loginfo("  D: Right")
        rospy.loginfo("  Space: Stop")
        rospy.loginfo("  Q: Quit")
    
    def get_key(self):
        """获取键盘输入"""
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def publish_instruction(self, message):
        """发布指令说明"""
        msg = String()
        msg.data = message
        self.instruction_pub.publish(msg)
    
    def run(self):
        """主循环"""
        try:
            self.publish_instruction("Use WASD to control, Space to stop, Q to quit")
            
            while not rospy.is_shutdown():
                key = self.get_key()
                
                # 创建控制消息
                twist = Twist()
                
                if key == 'w' or key == 'W':
                    # 前进
                    twist.linear.x = self.linear_speed
                    twist.angular.z = 0.0
                    self.publish_instruction("Moving FORWARD")
                    
                elif key == 's' or key == 'S':
                    # 后退
                    twist.linear.x = -self.linear_speed
                    twist.angular.z = 0.0
                    self.publish_instruction("Moving BACKWARD")
                    
                elif key == 'a' or key == 'A':
                    # 左转
                    twist.linear.x = self.linear_speed * 0.5
                    twist.angular.z = self.angular_speed
                    self.publish_instruction("Turning LEFT")
                    
                elif key == 'd' or key == 'D':
                    # 右转
                    twist.linear.x = self.linear_speed * 0.5
                    twist.angular.z = -self.angular_speed
                    self.publish_instruction("Turning RIGHT")
                    
                elif key == ' ':
                    # 停止
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.publish_instruction("STOPPING")
                    
                elif key == 'q' or key == 'Q':
                    # 退出
                    rospy.loginfo("Quitting teleop")
                    break
                
                else:
                    # 无效按键
                    continue
                
                # 发布控制命令
                self.cmd_pub.publish(twist)
                rospy.loginfo(f"Command: linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}")
                
        except Exception as e:
            rospy.logerr(f"Error in teleop: {e}")
        
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            
            # 发送停止命令
            stop_twist = Twist()
            stop_twist.linear.x = 0.0
            stop_twist.angular.z = 0.0
            self.cmd_pub.publish(stop_twist)
            
            rospy.loginfo("Teleop Keyboard Node Shutdown")

def main():
    teleop = TeleopKeyboard()
    teleop.run()

if __name__ == '__main__':
    main()
