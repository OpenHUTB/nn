#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""键盘遥控节点：把按键转成 /uav/cmd_vel（geometry_msgs/Twist）。

按键（按住生效，松开自动归零）:
    w / s   前后（+x / -x）
    a / d   左右（+y / -y）
    r / f   升降（+z / -z）
    q / e   偏航（+yaw / -yaw）
    空格    悬停（速度清零）
    Ctrl+C  退出
"""

import sys
import termios
import tty
import select

import rospy
from geometry_msgs.msg import Twist

HELP = """
------------------------------
  w/s : 前进 / 后退
  a/d : 左移 / 右移
  r/f : 上升 / 下降
  q/e : 左转 / 右转
  空格: 悬停（清零）
  Ctrl+C 退出
------------------------------
"""

KEYMAP = {
    "w": ("linear", "x", 1.0),
    "s": ("linear", "x", -1.0),
    "a": ("linear", "y", 1.0),
    "d": ("linear", "y", -1.0),
    "r": ("linear", "z", 1.0),
    "f": ("linear", "z", -1.0),
    "q": ("angular", "z", 1.0),
    "e": ("angular", "z", -1.0),
}


def get_key(timeout=0.1):
    """非阻塞读一个字符；没有输入返回空串。"""
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    if not dr:
        return ""
    return sys.stdin.read(1)


def main():
    rospy.init_node("uav_teleop_key")

    topic = rospy.get_param("~cmd_topic", "/uav/cmd_vel")
    speed = float(rospy.get_param("~speed", 1.0))          # m/s
    yaw_rate = float(rospy.get_param("~yaw_rate", 0.8))    # rad/s
    rate_hz = float(rospy.get_param("~rate", 20.0))

    pub = rospy.Publisher(topic, Twist, queue_size=1)
    rate = rospy.Rate(rate_hz)

    if not sys.stdin.isatty():
        rospy.logwarn("stdin 不是终端，键盘读取会失效；请在交互式终端里运行")
        return

    old = termios.tcgetattr(sys.stdin)
    print(HELP)
    cmd = Twist()

    try:
        tty.setcbreak(sys.stdin.fileno())
        while not rospy.is_shutdown():
            key = get_key(1.0 / rate_hz)
            if key:
                if key == " ":
                    cmd = Twist()
                    rospy.loginfo("悬停")
                elif key in KEYMAP:
                    axis, comp, sign = KEYMAP[key]
                    cmd = Twist()
                    value = sign * (yaw_rate if axis == "angular" else speed)
                    vec = getattr(cmd, axis)
                    setattr(vec, comp, value)
                    rospy.loginfo("cmd %s.%s = %.2f", axis, comp, value)
                elif key in ("\x03",):
                    break
            pub.publish(cmd)
            rate.sleep()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        pub.publish(Twist())
        print("\n已退出，速度已归零")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
