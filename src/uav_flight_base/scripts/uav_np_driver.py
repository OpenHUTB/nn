#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务4：用 NumPy 网络实时飞行（纯图像输入，不使用任何真值）。"""

import os
import sys

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

W, H = 32, 24


class NpDriver(object):
    def __init__(self, weights):
        z = np.load(weights)
        self.W1, self.b1 = z["W1"], z["b1"]
        self.W2, self.b2 = z["W2"], z["b2"]
        self.ym, self.ys = z["y_mean"], z["y_std"]
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/uav_cam/image_raw", Image, self.on_img, queue_size=1)
        self.count = 0
        self.busy = False
        rospy.loginfo("NumPy 策略已加载: %s", weights)

    def on_img(self, msg):
        if self.busy:
            return
        self.busy = True
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            a = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
            gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (W, H)).astype(np.float32).ravel() / 255.0
            h = np.maximum(small @ self.W1 + self.b1, 0.0)
            o = h @ self.W2 + self.b2
            out = o * self.ys + self.ym
            cmd = Twist()
            cmd.linear.x = float(np.clip(out[0], 0.0, 0.8))
            cmd.angular.z = float(np.clip(out[1], -0.9, 0.9))
            self.pub.publish(cmd)
            self.count += 1
            if self.count % 100 == 0:
                rospy.loginfo("推理 %d 帧  v=%.2f w=%.2f", self.count, cmd.linear.x, cmd.angular.z)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "推理失败: %s", e)
        finally:
            self.busy = False


def main():
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/bc_data/weights.npz")
    rospy.init_node("uav_np_driver")
    NpDriver(path)
    rospy.spin()


if __name__ == "__main__":
    main()
