#!/usr/bin/env python3
"""
CARLA控制客户端 - 用于手动控制或外部控制
"""

import rospy
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from carla_autonomous.srv import StartEpisode, Reset, Stop

class CarlaControlClient:
    def __init__(self):
        rospy.init_node('carla_control_client')
        
        # 服务代理
        self.start_episode = rospy.ServiceProxy('/carla/start_episode', StartEpisode)
        self.reset = rospy.ServiceProxy('/carla/reset', Reset)
        self.stop = rospy.ServiceProxy('/carla/stop', Stop)
        
        # 控制发布器
        self.control_pub = rospy.Publisher('/carla/control_cmd', Twist, queue_size=10)
        
        rospy.loginfo("CARLA控制客户端已启动")
    
    def manual_control(self):
        """手动控制模式"""
        rospy.loginfo("进入手动控制模式")
        rospy.loginfo("使用键盘控制车辆:")
        rospy.loginfo("  W/S: 前进/后退")
        rospy.loginfo("  A/D: 左转/右转")
        rospy.loginfo("  Space: 刹车")
        rospy.loginfo("  Q: 退出")
        
        try:
            import curses
            stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            stdscr.keypad(True)
            
            while not rospy.is_shutdown():
                key = stdscr.getch()
                
                control_msg = Twist()
                
                if key == ord('w'):
                    control_msg.linear.x = 0.5  # 前进
                elif key == ord('s'):
                    control_msg.linear.x = -0.3  # 后退
                elif key == ord('a'):
                    control_msg.angular.z = 0.5  # 左转
                elif key == ord('d'):
                    control_msg.angular.z = -0.5  # 右转
                elif key == ord(' '):
                    control_msg.linear.z = 1.0  # 刹车
                elif key == ord('q'):
                    break
                else:
                    continue
                
                self.control_pub.publish(control_msg)
                
        except ImportError:
            rospy.logwarn("curses模块未安装，无法使用键盘控制")
        except Exception as e:
            rospy.logerr(f"手动控制出错: {e}")
        finally:
            try:
                curses.nocbreak()
                stdscr.keypad(False)
                curses.echo()
                curses.endwin()
            except:
                pass
    
    def start_autonomous_episode(self):
        """启动自主驾驶episode"""
        try:
            response = self.start_episode()
            if response.success:
                rospy.loginfo(f"启动成功: {response.message}")
            else:
                rospy.logwarn(f"启动失败: {response.message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"服务调用失败: {e}")
    
    def reset_environment(self):
        """重置环境"""
        try:
            response = self.reset()
            rospy.loginfo("环境重置成功")
        except rospy.ServiceException as e:
            rospy.logerr(f"重置服务调用失败: {e}")
    
    def stop_all(self):
        """停止所有"""
        try:
            response = self.stop()
            rospy.loginfo("停止成功")
        except rospy.ServiceException as e:
            rospy.logerr(f"停止服务调用失败: {e}")
    
    def run(self):
        """运行客户端"""
        rospy.loginfo("控制命令:")
        rospy.loginfo("  1: 启动自主驾驶")
        rospy.loginfo("  2: 手动控制")
        rospy.loginfo("  3: 重置环境")
        rospy.loginfo("  4: 停止")
        rospy.loginfo("  0: 退出")
        
        try:
            while not rospy.is_shutdown():
                command = input("请输入命令: ")
                
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
                    rospy.logwarn("未知命令")
                    
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