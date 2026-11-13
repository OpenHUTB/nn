#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取一帧 Gazebo 相机图像并存盘（用于无图形界面环境下的验证）。

用法:
    rosrun uav_flight_base uav_camera_capture.py --topic /uav_cam/image_raw --out /tmp/frame.ppm --timeout 20
"""

import argparse
import sys

import numpy as np
import rospy
from sensor_msgs.msg import Image


def save_ppm(path, arr):
    h, w, _ = arr.shape
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(arr.tobytes())


def save_png(path, arr):
    try:
        from PIL import Image as PILImage
    except ImportError:
        return False
    PILImage.fromarray(arr).save(path)
    return True


class Grabber(object):
    def __init__(self, topic, out, timeout):
        self.out = out
        self.got = False
        self.sub = rospy.Subscriber(topic, Image, self.cb, queue_size=1)
        self.deadline = rospy.Time.now() + rospy.Duration(timeout)

    def cb(self, msg):
        if self.got:
            return
        w, h, step = msg.width, msg.height, msg.step
        enc = msg.encoding.lower()
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if enc in ("rgb8", "bgr8"):
            img = raw.reshape(h, step)[:, :w * 3].reshape(h, w, 3)
            if enc == "bgr8":
                img = img[:, :, ::-1]
        elif enc in ("mono8", "8uc1"):
            img = np.repeat(raw.reshape(h, step)[:, :w][:, :, None], 3, axis=2)
        else:
            rospy.logwarn("暂不支持的编码: %s", msg.encoding)
            return
        img = np.ascontiguousarray(img)
        base = self.out.rsplit(".", 1)[0]
        save_ppm(base + ".ppm", img)
        ok = save_png(base + ".png", img)
        rospy.loginfo("已保存 %s%s (%dx%d, %s)", base + ".ppm",
                      " 和 .png" if ok else "", w, h, msg.encoding)
        self.got = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/uav_cam/image_raw")
    ap.add_argument("--out", default="/tmp/uav_frame.ppm")
    ap.add_argument("--timeout", type=float, default=20.0)
    args, _ = ap.parse_known_args(sys.argv)

    rospy.init_node("uav_camera_capture", anonymous=True)
    g = Grabber(args.topic, args.out, args.timeout)
    r = rospy.Rate(5)
    while not rospy.is_shutdown() and not g.got:
        if rospy.Time.now() > g.deadline:
            rospy.logerr("超时未收到图像: %s", args.topic)
            return 1
        r.sleep()
    return 0


if __name__ == "__main__":
    sys.exit(main())
