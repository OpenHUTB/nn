#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""状态监控节点：订阅 odom，定时打印飞行状态，可另存 CSV。

输出示例:
    [  12.3s] pos=( 3.21, -1.05,  1.50)  yaw= 12.4°  v= 1.02 m/s
"""

import math
import os

import rospy
from nav_msgs.msg import Odometry


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class Monitor(object):
    def __init__(self):
        self.topic = rospy.get_param("~odom_topic", "/uav/odom")
        self.period = float(rospy.get_param("~period", 0.5))
        self.csv_path = rospy.get_param("~csv", "")

        self.t0 = None
        self.rows = []
        self.latest = None

        self.sub = rospy.Subscriber(self.topic, Odometry, self.on_odom, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(self.period), self.on_timer)

    def on_odom(self, msg):
        if self.t0 is None:
            self.t0 = msg.header.stamp
        self.latest = msg
        if self.csv_path:
            p = msg.pose.pose.position
            self.rows.append([msg.header.stamp.to_sec(), p.x, p.y, p.z,
                              yaw_from_quat(msg.pose.pose.orientation)])

    def on_timer(self, _evt):
        m = self.latest
        if m is None:
            rospy.loginfo_throttle(5.0, "等待 %s ...", self.topic)
            return
        p = m.pose.pose.position
        v = m.twist.twist.linear
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        t = (m.header.stamp - self.t0).to_sec()
        rospy.loginfo("[%6.1fs] pos=(%6.2f,%6.2f,%6.2f) yaw=%6.1f°  v=%.2f m/s",
                      t, p.x, p.y, p.z,
                      math.degrees(yaw_from_quat(m.pose.pose.orientation)), speed)

    def save(self):
        if not self.csv_path or not self.rows:
            return
        d = os.path.dirname(os.path.abspath(self.csv_path))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(self.csv_path, "w") as f:
            f.write("t,x,y,z,yaw\n")
            for r in self.rows:
                f.write(",".join("%.4f" % x for x in r) + "\n")
        rospy.loginfo("已保存 %d 行到 %s", len(self.rows), self.csv_path)


def main():
    rospy.init_node("uav_state_monitor")
    mon = Monitor()
    rospy.on_shutdown(mon.save)
    rospy.spin()


if __name__ == "__main__":
    main()
