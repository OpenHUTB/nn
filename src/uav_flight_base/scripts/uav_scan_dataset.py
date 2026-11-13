#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务2（神经网络版）第一步：采集 (激光扫描, 速度指令) 数据。

一边运行专家避障策略（uav_scan_avoider），一边记录它发出的 cmd_vel 与
当时的 360 线激光扫描，得到行为克隆数据集。

产物: ~/policy_data/scan_dataset.csv
"""

import csv
import os

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

N = 90          # 采样维度（360 线降采样到 90）
RMAX = 3.5      # 距离截断


class Recorder(object):
    def __init__(self):
        out = rospy.get_param("~out_dir", os.path.expanduser("~/policy_data"))
        if not os.path.isdir(out):
            os.makedirs(out)
        self.path = os.path.join(out, "scan_dataset.csv")
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["r%d" % i for i in range(N)] + ["v", "w"])

        self.scan = None
        self.cmd = None
        self.count = 0

        rospy.Subscriber("/uav/scan", LaserScan, self.on_scan, queue_size=1)
        rospy.Subscriber("/cmd_vel", Twist, self.on_cmd, queue_size=1)
        rospy.Timer(rospy.Duration(0.1), self.tick)
        rospy.loginfo("采集到 %s", self.path)

    def on_scan(self, msg):
        self.scan = msg

    def on_cmd(self, msg):
        self.cmd = msg

    def tick(self, _evt):
        if self.scan is None or self.cmd is None:
            return
        r = np.array(self.scan.ranges, dtype=np.float32)
        r[~np.isfinite(r)] = RMAX
        r = np.clip(r, 0.0, RMAX)
        idx = np.linspace(0, len(r) - 1, N).astype(int)
        feat = r[idx]
        self.w.writerow(list(np.round(feat, 3)) +
                        [round(self.cmd.linear.x, 3), round(self.cmd.angular.z, 3)])
        self.f.flush()
        self.count += 1
        if self.count % 100 == 0:
            rospy.loginfo("已记录 %d 条", self.count)


def main():
    rospy.init_node("uav_scan_dataset")
    Recorder()
    rospy.spin()


if __name__ == "__main__":
    main()
