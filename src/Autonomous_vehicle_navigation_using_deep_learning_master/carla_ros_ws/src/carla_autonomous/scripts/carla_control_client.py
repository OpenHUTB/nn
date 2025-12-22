#!/usr/bin/env python3
"""
CARLA控制客户端 - 修复版
"""

import rospy
import sys
import select
import tty
import termios
import time
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
        rospy.loginfo("  W: 前进")
        rospy.loginfo("  S: 后退")
        rospy.loginfo("  A: 左转")
        rospy.loginfo("  D: 右转")
        rospy.loginfo("  Space: 刹车")
        rospy.loginfo("  Q: 退出")
        rospy.loginfo("注意: 按住键持续控制，松开自动回正")
        
        print("\n开始控制（按Q退出）...")
        
        # 使用一个简单的循环，每次获取一个字符
        # 这种方法在大多数系统上都能工作
        import select
        
        control_msg = Twist()
        
        # 设置终端为非阻塞模式
        import termios
        import tty
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(fd)
            
            # 状态变量
            current_throttle = 0.0  # 当前油门
            current_steer = 0.0     # 当前转向
            current_brake = 0.0     # 当前刹车
            
            while not rospy.is_shutdown():
                # 检查是否有按键
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                
                if rlist:
                    # 有按键
                    key = sys.stdin.read(1).lower()
                    
                    if key == 'w':
                        current_throttle = 0.5
                        current_brake = 0.0
                        print("前进", end='\r')
                    elif key == 's':
                        current_throttle = -0.5
                        current_brake = 0.0
                        print("后退", end='\r')
                    elif key == 'a':
                        current_steer = -0.5
                        print("左转", end='\r')
                    elif key == 'd':
                        current_steer = 0.5
                        print("右转", end='\r')
                    elif key == ' ':
                        current_throttle = 0.0
                        current_steer = 0.0
                        current_brake = 1.0
                        print("刹车", end='\r')
                    elif key == 'q':
                        print("\n退出")
                        break
                    else:
                        # 未知按键，忽略
                        continue
                else:
                    # 没有按键，转向自动回正
                    current_steer = 0.0
                    # 油门保持当前状态
                    print("保持...", end='\r')
                
                # 设置控制消息
                control_msg.linear.x = current_throttle
                control_msg.angular.z = current_steer
                control_msg.linear.z = current_brake
                
                # 发布控制消息
                self.control_pub.publish(control_msg)
        
        except Exception as e:
            rospy.logerr(f"手动控制出错: {e}")
        finally:
            # 恢复终端设置
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            
            # 确保车辆停止
            control_msg.linear.x = 0.0
            control_msg.angular.z = 0.0
            control_msg.linear.z = 1.0
            try:
                self.control_pub.publish(control_msg)
            except:
                pass
            print("\n退出手动模式")
            
            # 确保车辆停止
            control_msg.linear.x = 0.0
            control_msg.angular.z = 0.0
            control_msg.linear.z = 1.0
            self.control_pub.publish(control_msg)
            print("\n退出手动模式")
    
    def manual_control_incremental(self):
        """增量控制模式 - 更精确的控制"""
        rospy.loginfo("进入增量控制模式")
        rospy.loginfo("使用键盘控制车辆:")
        rospy.loginfo("  W: 增加前进油门")
        rospy.loginfo("  S: 增加后退油门")
        rospy.loginfo("  A: 增加左转")
        rospy.loginfo("  D: 增加右转")
        rospy.loginfo("  E: 减少油门")
        rospy.loginfo("  Space: 急刹车")
        rospy.loginfo("  R: 重置控制")
        rospy.loginfo("  Q: 退出")
        
        print("\n开始控制（按Q退出）...")
        
        control_msg = Twist()
        control_msg.linear.x = 0.0
        control_msg.angular.z = 0.0
        control_msg.linear.z = 0.0
        
        # 控制增量
        throttle_inc = 0.1
        steer_inc = 0.1
        
        try:
            while not rospy.is_shutdown():
                key = self.get_key()
                
                if key is not None:
                    if key == 'w' or key == 'W':
                        # 增加前进油门
                        control_msg.linear.x += throttle_inc
                        control_msg.linear.x = min(1.0, control_msg.linear.x)
                        control_msg.linear.z = 0.0
                        print(f"前进油门: {control_msg.linear.x:.2f}")
                    elif key == 's' or key == 'S':
                        # 增加后退油门（负值）
                        control_msg.linear.x -= throttle_inc
                        control_msg.linear.x = max(-1.0, control_msg.linear.x)
                        control_msg.linear.z = 0.0
                        print(f"后退油门: {control_msg.linear.x:.2f}")
                    elif key == 'a' or key == 'A':
                        # 左转
                        control_msg.angular.z -= steer_inc
                        control_msg.angular.z = min(-1.0, control_msg.angular.z)
                        print(f"左转: {control_msg.angular.z:.2f}")
                    elif key == 'd' or key == 'D':
                        # 右转
                        control_msg.angular.z += steer_inc
                        control_msg.angular.z = max(1.0, control_msg.angular.z)
                        print(f"右转: {control_msg.angular.z:.2f}")
                    elif key == 'e' or key == 'E':
                        # 减少油门（绝对值）
                        if control_msg.linear.x > 0:
                            control_msg.linear.x -= throttle_inc
                            control_msg.linear.x = max(0, control_msg.linear.x)
                        elif control_msg.linear.x < 0:
                            control_msg.linear.x += throttle_inc
                            control_msg.linear.x = min(0, control_msg.linear.x)
                        print(f"油门: {control_msg.linear.x:.2f}")
                    elif key == ' ':
                        # 急刹车
                        control_msg.linear.z = 1.0
                        control_msg.linear.x = 0.0
                        print("急刹车")
                    elif key == 'r' or key == 'R':
                        # 重置
                        control_msg.linear.x = 0.0
                        control_msg.angular.z = 0.0
                        control_msg.linear.z = 0.0
                        print("重置控制")
                    elif key == 'q' or key == 'Q':
                        # 退出
                        control_msg.linear.x = 0.0
                        control_msg.angular.z = 0.0
                        control_msg.linear.z = 1.0
                        self.control_pub.publish(control_msg)
                        print("\n退出增量控制")
                        break
                
                # 持续发布控制消息
                self.control_pub.publish(control_msg)
                time.sleep(0.05)
        
        except Exception as e:
            rospy.logerr(f"增量控制出错: {e}")
        finally:
            # 确保车辆停止
            control_msg.linear.x = 0.0
            control_msg.angular.z = 0.0
            control_msg.linear.z = 1.0
            self.control_pub.publish(control_msg)
    
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
        print("  2: 手动控制（按键控制）")
        print("  3: 增量控制（精确控制）")
        print("  4: 重置环境")
        print("  5: 停止")
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
                        self.manual_control_incremental()
                    elif command == '4':
                        self.reset_environment()
                    elif command == '5':
                        self.stop_all()
                    elif command == '0':
                        rospy.loginfo("退出客户端")
                        break
                    else:
                        print("未知命令，请输入 0-5")
                        
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