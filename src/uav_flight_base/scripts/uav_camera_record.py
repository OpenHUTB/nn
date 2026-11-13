#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按固定间隔录制若干帧相机图像，用于生成演示 GIF。

用法:
    rosrun uav_flight_base uav_camera_record.py --outdir /tmp/seq --count 24 --interval 0.25
"""

import argparse
import os
import sys

import numpy as np
import rospy
from sensor_msgs.msg import Image


def to_rgb(msg):
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
        return None
    return np.ascontiguousarray(img)


def save_ppm(path, arr):
    h, w, _ = arr.shape
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(arr.tobytes())


class Recorder(object):
    def __init__(self, outdir, count, interval):
        self.outdir = outdir
        self.count = count
        self.interval = rospy.Duration(interval)
        self.next_at = None
        self.saved = 0
        self.latest = None
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        self.sub = rospy.Subscriber("/uav_cam/image_raw", Image, self.cb, queue_size=1)

    def cb(self, msg):
        self.latest = msg

    def spin(self):
        r = rospy.Rate(30)
        self.next_at = rospy.Time.now()
        while not rospy.is_shutdown() and self.saved < self.count:
            if self.latest is not None and rospy.Time.now() >= self.next_at:
                img = to_rgb(self.latest)
                if img is not None:
                    p = os.path.join(self.outdir, "frame_%03d.ppm" % self.saved)
                    save_ppm(p, img)
                    self.saved += 1
                    self.next_at = rospy.Time.now() + self.interval
            r.sleep()
        rospy.loginfo("录制完成: %d 帧 -> %s", self.saved, self.outdir)
        return 0 if self.saved else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/seq")
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--interval", type=float, default=0.25)
    args, _ = ap.parse_known_args(sys.argv)
    rospy.init_node("uav_camera_record", anonymous=True)
    return Recorder(args.outdir, args.count, args.interval).spin()


if __name__ == "__main__":
    sys.exit(main())
