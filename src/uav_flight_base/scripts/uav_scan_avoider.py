#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""激光雷达自主避障：感知 + 控制，完全不依赖相机/图形界面。

控制律（反应式避障）:
    front = 正前方 ±front_fov 内的最小距离
    front < warn_dist  → 转向最近空隙方向，减速
    front < stop_dist  → 原地转
    否则               → 直行

同时把 (t, x, y, yaw, front, left, right) 记到 CSV，便于事后画轨迹图。
"""

import csv
import math
import os

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ScanAvoider(object):
    def __init__(self):
        self.scan_topic = rospy.get_param("~scan_topic", "/uav/scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/uav/odom")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

        self.v_norm = rospy.get_param("~v_norm", 0.9)
        self.v_slow = rospy.get_param("~v_slow", 0.35)
        self.w_max = rospy.get_param("~w_max", 0.9)
        self.warn_dist = rospy.get_param("~warn_dist", 1.6)
        self.stop_dist = rospy.get_param("~stop_dist", 0.8)
        self.front_fov = rospy.get_param("~front_fov", 0.6)   # ±弧度
        self.record = rospy.get_param("~csv", "")

        self.scan = None
        self.pose = None
        self.t0 = None
        self.rows = []
        self.last_log = rospy.Time.now()
        self._csv = None
        self._writer = None
        if self.record:
            d = os.path.dirname(os.path.abspath(self.record))
            if d and not os.path.isdir(d):
                os.makedirs(d)
            self._csv = open(self.record, "w", newline="")
            self._writer = csv.writer(self._csv)
            self._writer.writerow(["t", "x", "y", "z", "yaw", "front", "left", "right"])
            self._csv.flush()

        self.pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.on_scan, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.on_odom, queue_size=1)
        rospy.loginfo("激光避障启动: %s", self.scan_topic)

    def on_scan(self, msg):
        self.scan = msg

    def on_odom(self, msg):
        self.pose = msg

    def sector_min(self, lo, hi):
        s = self.scan
        if s is None:
            return float("inf")
        best = float("inf")
        n = len(s.ranges)
        for i, r in enumerate(s.ranges):
            a = s.angle_min + i * s.angle_increment
            if lo <= a <= hi and r == r:          # 过滤 NaN
                best = min(best, r)
        return best

    def run(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            cmd = Twist()
            if self.scan is None:
                self.pub.publish(cmd)
                r.sleep()
                continue

            front = self.sector_min(-self.front_fov, self.front_fov)
            left = self.sector_min(0.3, 1.4)
            right = self.sector_min(-1.4, -0.3)

            if front < self.stop_dist:
                # 太近：原地转向空隙大的一侧
                cmd.angular.z = self.w_max if left > right else -self.w_max
                cmd.linear.x = 0.0
            elif front < self.warn_dist:
                # 有点近：朝空隙转，同时减速
                err = (left - right) / max(left + right, 1e-3)
                cmd.angular.z = max(-self.w_max, min(self.w_max, 1.2 * err))
                cmd.linear.x = self.v_slow
            else:
                cmd.linear.x = self.v_norm
                cmd.angular.z = 0.0

            self.pub.publish(cmd)
            self.record_row(front, left, right)
            r.sleep()
        self.save()

    def record_row(self, front, left, right):
        if self.pose is None:
            return
        if self.t0 is None:
            self.t0 = self.pose.header.stamp
        p = self.pose.pose.pose.position
        yaw = yaw_from_quat(self.pose.pose.pose.orientation)
        t = (self.pose.header.stamp - self.t0).to_sec()
        row = [round(t, 3), round(p.x, 4), round(p.y, 4), round(p.z, 4),
               round(yaw, 4), round(min(front, 99), 3),
               round(min(left, 99), 3), round(min(right, 99), 3)]
        self.rows.append(row)
        if self._writer is not None:
            self._writer.writerow(row)
            self._csv.flush()     # 实时落盘，进程被杀也不丢数据

    def save(self):
        if self._csv is not None:
            try:
                self._csv.close()
            except Exception:
                pass
            self._csv = None
        if self.record:
            rospy.loginfo("轨迹已保存到 %s（%d 行）", self.record, len(self.rows))


def main():
    rospy.init_node("uav_scan_avoider")
    node = ScanAvoider()
    rospy.on_shutdown(node.save)
    node.run()


if __name__ == "__main__":
    main()
