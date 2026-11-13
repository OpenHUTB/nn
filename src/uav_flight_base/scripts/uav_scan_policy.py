#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务2（神经网络版）第三步：用训练好的网络接管飞行，并记录评估指标。

订阅 /uav/scan -> 网络 -> /cmd_vel
同时把 (t, v, w, min_range) 写到 ~/policy_data/eval_<tag>.csv
"""

import csv
import os
import sys

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

N = 90
RMAX = 3.5


class ScanPolicy(object):
    def __init__(self, weights, tag):
        z = np.load(weights)
        self.W1, self.b1 = z["W1"], z["b1"]
        self.W2, self.b2 = z["W2"], z["b2"]
        self.xm, self.xs = z["xm"], z["xs"]
        self.ym, self.ys = z["ym"], z["ys"]

        out = os.path.dirname(weights) or os.path.expanduser("~/policy_data")
        self.log = open(os.path.join(out, "eval_%s.csv" % tag), "w", newline="")
        self.w = csv.writer(self.log)
        self.w.writerow(["t", "v", "w", "min_range"])
        self.t0 = rospy.Time.now()
        self.count = 0
        self.min_all = 9.9

        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/uav/scan", LaserScan, self.on_scan, queue_size=1)
        rospy.loginfo("扫描策略已加载: %s  tag=%s", weights, tag)

    def on_scan(self, msg):
        r = np.array(msg.ranges, dtype=np.float32)
        r[~np.isfinite(r)] = RMAX
        r = np.clip(r, 0.0, RMAX)
        mr = float(r.min())
        self.min_all = min(self.min_all, mr)

        idx = np.linspace(0, len(r) - 1, N).astype(int)
        feat = (r[idx] - self.xm) / self.xs
        hh = np.maximum(feat @ self.W1 + self.b1, 0.0)
        o = hh @ self.W2 + self.b2
        out = o * self.ys + self.ym

        cmd = Twist()
        cmd.linear.x = float(np.clip(out[0], 0.0, 0.9))
        cmd.angular.z = float(np.clip(out[1], -1.0, 1.0))
        self.pub.publish(cmd)

        self.count += 1
        if self.count % 5 == 0:
            self.w.writerow([round((rospy.Time.now() - self.t0).to_sec(), 2),
                             round(cmd.linear.x, 3), round(cmd.angular.z, 3), round(mr, 3)])
            self.log.flush()
        if self.count % 100 == 0:
            rospy.loginfo("推理 %d 帧  v=%.2f w=%.2f min_range=%.2f",
                          self.count, cmd.linear.x, cmd.angular.z, mr)


def main():
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/policy_data/scan_policy.npz")
    tag = sys.argv[2] if len(sys.argv) > 2 else "run"
    rospy.init_node("uav_scan_policy")
    ScanPolicy(path, tag)
    rospy.spin()


if __name__ == "__main__":
    main()
