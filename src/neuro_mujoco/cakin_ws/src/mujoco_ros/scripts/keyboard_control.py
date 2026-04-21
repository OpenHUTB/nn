#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32MultiArray
from pynput import keyboard
import sys
import argparse

# 全局变量
pub = None
delta_pos = None
joint_num = 0
step_size = 0.05  # 每次按键的角度增量（弧度）

def on_press(key):
    """键盘按下事件处理"""
    global delta_pos
    try:
        # 将 delta_pos 重置为全零
        delta_pos = [0.0] * joint_num

        # 根据按键设置对应的关节增量
        if key.char == '1':
            delta_pos[0] = step_size
        elif key.char == 'q':
            delta_pos[0] = -step_size
        elif key.char == '2':
            delta_pos[1] = step_size
        elif key.char == 'w':
            delta_pos[1] = -step_size
        elif key.char == '3' and joint_num > 2:
            delta_pos[2] = step_size
        elif key.char == 'e' and joint_num > 2:
            delta_pos[2] = -step_size
        elif key.char == '4' and joint_num > 3:
            delta_pos[3] = step_size
        elif key.char == 'r' and joint_num > 3:
            delta_pos[3] = -step_size
        elif key.char == '5' and joint_num > 4:
            delta_pos[4] = step_size
        elif key.char == 't' and joint_num > 4:
            delta_pos[4] = -step_size
        elif key.char == '6' and joint_num > 5:
            delta_pos[5] = step_size
        elif key.char == 'y' and joint_num > 5:
            delta_pos[5] = -step_size
        elif key.char == '7' and joint_num > 6:
            delta_pos[6] = step_size
        elif key.char == 'u' and joint_num > 6:
            delta_pos[6] = -step_size
        elif key.char == '8' and joint_num > 7:
            delta_pos[7] = step_size
        elif key.char == 'i' and joint_num > 7:
            delta_pos[7] = -step_size

        # 如果有增量，则发布消息
        if any(d != 0 for d in delta_pos):
            msg = Float32MultiArray(data=delta_pos)
            pub.publish(msg)
            rospy.loginfo(f"发布关节增量: {delta_pos}")

    except AttributeError:
        # 处理特殊按键（如方向键），这里我们忽略
        pass

def on_release(key):
    """键盘释放事件处理"""
    # 当按键释放时，发布一个全零的增量，让机器人停止在当前位置
    if key == keyboard.Key.esc:
        # 按下ESC键退出
        rospy.loginfo("检测到ESC键，退出键盘控制...")
        return False

def main():
    global pub, joint_num, delta_pos
    rospy.init_node('keyboard_controller', anonymous=True)
    
    # 使用 argparse 接收命令行参数
    parser = argparse.ArgumentParser(description='ROS Keyboard Controller for MuJoCo')
    parser.add_argument('joint_num', type=int, help='Number of joints to control (e.g., 7 for Franka)')
    parser.add_argument('--step', type=float, default=0.05, help='Step size in radians for each key press.')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:]) # 处理ROS参数

    joint_num = args.joint_num
    step_size = args.step
    delta_pos = [0.0] * joint_num

    pub = rospy.Publisher('joint_position_delta', Float32MultiArray, queue_size=10)
    
    rospy.loginfo("="*50)
    rospy.loginfo("      ROS 键盘控制器已启动      ")
    rospy.loginfo(f"控制关节数: {joint_num}, 步长: {step_size} rad")
    rospy.loginfo("按键映射:")
    rospy.loginfo("  关节1: '1' (增加) / 'q' (减少)")
    rospy.loginfo("  关节2: '2' (增加) / 'w' (减少)")
    rospy.loginfo("  关节3: '3' (增加) / 'e' (减少)")
    rospy.loginfo("  关节4: '4' (增加) / 'r' (减少)")
    rospy.loginfo("  关节5: '5' (增加) / 't' (减少)")
    rospy.loginfo("  关节6: '6' (增加) / 'y' (减少)")
    rospy.loginfo("  关节7: '7' (增加) / 'u' (减少)")
    rospy.loginfo("  关节8: '8' (增加) / 'i' (减少)")
    rospy.loginfo("  退出: 按 'ESC' 键")
    rospy.loginfo("="*50)

    # 启动键盘监听
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass