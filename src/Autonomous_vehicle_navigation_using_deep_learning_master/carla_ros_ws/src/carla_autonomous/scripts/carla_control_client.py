#!/usr/bin/env python3
"""
CARLA控制客户端 - 修复版
"""

import rospy
import sys
import select
import tty
import termios
from std_msgs.msg import String
from geometry_msgs.msg import Twist

# 尝试导入自定义服务，如果失败使用标准服务
try:
    from carla_autonomous.srv import StartEpisode, Reset, Stop
    USE_CUSTOM_SERVICES = True
except ImportError:
    from std_srvs.srv import Empty, Trigger
    USE_CUSTOM_SERVICES = False
    rospy.logwarn("使用标准ROS服务")

class CarlaControlClient:
    def __init__(self):
        rospy.init_node('carla_control_client', anonymous=True)
        
        # 控制发布器
        self.control_pub = rospy.Publisher('/carla/control_cmd', Twist, queue_size=10)
        
        # 等待服务可用
        rospy.loginfo("等待服务...")
        
        if USE_CUSTOM_SERVICES:
            rospy.wait_for_service('/carla/start_episode')
            rospy.wait_for_service('/carla/reset')
            rospy.wait_for_service('/carla/stop')
            
            self.start_episode = rospy.ServiceProxy('/carla/start_episode', StartEpisode)
            self.reset = rospy.ServiceProxy('/carla/reset', Reset)
            self.stop = rospy.ServiceProxy('/carla/stop', Stop)
        else:
            rospy.wait_for_service('/carla/start_episode')
            rospy.wait_for_service('/carla/reset')
            rospy.wait_for_service('/carla/stop')
            
            self.start_episode = rospy.ServiceProxy('/carla/start_episode', Trigger)
            self.reset = rospy.ServiceProxy('/carla/reset', Empty)
            self.stop = rospy.ServiceProxy('/carla/stop', Trigger)
        
        rospy.loginfo("CARLA控制客户端已启动")
    
    def get_key(self):
        """获取单个按键"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
            else:
                key = None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def manual_control(self):
        """手动控制模式"""
        rospy.loginfo("进入手动控制模式")
        rospy.loginfo("使用键盘控制车辆:")
        rospy.loginfo("  W/S: 前进/后退")
        rospy.loginfo("  A/D: 左转/右转")
        rospy.loginfo("  Space: 刹车")
        rospy.loginfo("  Q: 退出手动模式")
        
        print("\n开始控制（按Q退出）...")
        
        try:
            while not rospy.is_shutdown():
                key = self.get_key()
                
                if key is None:
                    continue
                
                control_msg = Twist()
                
                if key == 'w' or key == 'W':
                    control_msg.linear.x = 0.5  # 前进
                    print("前进", end='\r')
                elif key == 's' or key == 'S':
                    control_msg.linear.x = -0.3  # 后退
                    print("后退", end='\r')
                elif key == 'a' or key == 'A':
                    control_msg.angular.z = 0.5  # 左转
                    print("左转", end='\r')
                elif key == 'd' or key == 'D':
                    control_msg.angular.z = -0.5  # 右转
                    print("右转", end='\r')
                elif key == ' ':
                    control_msg.linear.z = 1.0  # 刹车
                    print("刹车", end='\r')
                elif key == 'q' or key == 'Q':
                    print("退出手动模式")
                    break
                else:
                    continue
                
                self.control_pub.publish(control_msg)
                
                # 添加延迟防止过度发布
                rospy.sleep(0.05)
        
        except Exception as e:
            rospy.logerr(f"手动控制出错: {e}")
    
    def start_autonomous_episode(self):
        """启动自主驾驶episode"""
        try:
            response = self.start_episode()
            if USE_CUSTOM_SERVICES:
                if response.success:
                    rospy.loginfo(f"启动成功: {response.message}")
                else:
                    rospy.logwarn(f"启动失败: {response.message}")
            else:
                if response.success:
                    rospy.loginfo(f"启动成功: {response.message}")
                else:
                    rospy.logwarn(f"启动失败: {response.message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"服务调用失败: {e}")
    
    def reset_environment(self):
        """重置环境"""
        try:
            if USE_CUSTOM_SERVICES:
                response = self.reset()
                rospy.loginfo("环境重置成功")
            else:
                self.reset()
                rospy.loginfo("环境重置成功")
        except rospy.ServiceException as e:
            rospy.logerr(f"重置服务调用失败: {e}")
    
    def stop_all(self):
        """停止所有"""
        try:
            response = self.stop()
            if USE_CUSTOM_SERVICES:
                if response.success:
                    rospy.loginfo(f"停止成功: {response.message}")
                else:
                    rospy.logwarn(f"停止失败: {response.message}")
            else:
                if response.success:
                    rospy.loginfo(f"停止成功: {response.message}")
                else:
                    rospy.logwarn(f"停止失败: {response.message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"停止服务调用失败: {e}")
    
    def run(self):
        """运行客户端"""
        print("\n" + "="*50)
        print("CARLA自动驾驶控制客户端")
        print("="*50)
        print("\n控制命令:")
        print("  1: 启动自主驾驶")
        print("  2: 手动控制")
        print("  3: 重置环境")
        print("  4: 停止")
        print("  0: 退出")
        print("="*50)
        
        try:
            while not rospy.is_shutdown():
                try:
                    command = input("\n请输入命令: ")
                    
                    if command == '1':
                        self.start_autonomous_episode()
                    elif command == '2':
                        self.manual_control()
                    elif command == '3':
                        self.reset_environment()
                    elif command == '4':
                        self.stop_all()
                    elif command == '0':
                        rospy.loginfo("退出客户端")
                        break
                    else:
                        print("未知命令，请输入 0-4")
                        
                except EOFError:
                    break
                except Exception as e:
                    rospy.logerr(f"命令处理出错: {e}")
                    
        except KeyboardInterrupt:
            rospy.loginfo("客户端已停止")
        except Exception as e:
            rospy.logerr(f"客户端运行出错: {e}")

if __name__ == '__main__':
    try:
        client = CarlaControlClient()
        client.run()
    except rospy.ROSInterruptException:
        pass