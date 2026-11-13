#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""颜色目标检测：在相机图像里找出红色目标桶。

不依赖 OpenCV —— 用 NumPy 自己做 RGB→HSV 转换，方便只有基础依赖的环境。

发布:
    /uav/perception/target   geometry_msgs/Point
        x = 归一化横向偏差 (-1 左, +1 右, 0 居中)
        y = 目标面积占画面的比例 (0~1，可用来估计远近)
        z = 1 表示检测到，0 表示丢失

订阅:
    /uav_cam/image_raw       sensor_msgs/Image
"""

import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image


def rgb_to_hsv(rgb):
    """rgb: (H, W, 3) uint8 -> h(0-360), s(0-1), v(0-1)"""
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.max(a, axis=-1)
    mn = np.min(a, axis=-1)
    d = mx - mn + 1e-6

    h = np.zeros_like(mx)
    mask = mx == r
    h[mask] = (60 * ((g - b) / d) % 360)[mask]
    mask = mx == g
    h[mask] = (60 * ((b - r) / d) + 120)[mask]
    mask = mx == b
    h[mask] = (60 * ((r - g) / d) + 240)[mask]

    s = d / (mx + 1e-6)
    return h, s, mx


class ColorDetector(object):
    def __init__(self):
        self.topic_in = rospy.get_param("~image_topic", "/uav_cam/image_raw")
        self.topic_out = rospy.get_param("~target_topic", "/uav/perception/target")
        # 红色在 HSV 色环上跨 0 度
        self.h_lo = rospy.get_param("~hue_low", 345.0)
        self.h_hi = rospy.get_param("~hue_high", 15.0)
        self.s_min = rospy.get_param("~sat_min", 0.45)
        self.v_min = rospy.get_param("~val_min", 0.20)
        self.min_area = rospy.get_param("~min_area", 0.0015)   # 占画面比例
        self.stride = int(rospy.get_param("~stride", 2))       # 降采样倍数，省 CPU
        self.max_rate = float(rospy.get_param("~rate", 5.0))   # 最高处理频率

        self.last_proc = rospy.Time(0)

        self.pub = rospy.Publisher(self.topic_out, Point, queue_size=1)
        self.sub = rospy.Subscriber(self.topic_in, Image, self.on_image, queue_size=1)
        rospy.loginfo("颜色检测启动: %s -> %s", self.topic_in, self.topic_out)

    def on_image(self, msg):
        now = rospy.Time.now()
        if self.max_rate > 0 and (now - self.last_proc).to_sec() < 1.0 / self.max_rate:
            return
        self.last_proc = now

        w, h, step = msg.width, msg.height, msg.step
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding.lower()
        try:
            if enc in ("rgb8", "bgr8"):
                img = raw.reshape(h, step)[:, :w * 3].reshape(h, w, 3)
                if enc == "bgr8":
                    img = img[:, :, ::-1]
            elif enc in ("mono8", "8uc1"):
                return
            else:
                return
        except ValueError:
            return

        s = max(1, self.stride)
        if s > 1:
            img = np.ascontiguousarray(img[::s, ::s])
            w, h = img.shape[1], img.shape[0]

        hh, ss, vv = rgb_to_hsv(img)
        mask = ((hh >= self.h_lo) | (hh <= self.h_hi)) & (ss >= self.s_min) & (vv >= self.v_min)
        area = float(mask.mean())

        out = Point()
        out.z = 0.0
        if area >= self.min_area:
            ys, xs = np.nonzero(mask)
            cx = float(xs.mean())
            out.x = (cx - w / 2.0) / (w / 2.0)   # -1..1
            out.y = area
            out.z = 1.0
        self.pub.publish(out)


def main():
    rospy.init_node("uav_color_detector")
    ColorDetector()
    rospy.spin()


if __name__ == "__main__":
    main()
