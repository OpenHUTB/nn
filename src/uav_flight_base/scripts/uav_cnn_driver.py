#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务4 第三步：用训练好的 CNN 实时飞行（只吃图像，不用任何真值）。"""

import os
import sys

import cv2
import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from uav_train_cnn import SmallCNN

W, H = 64, 48


class CnnDriver(object):
    def __init__(self, model_path):
        self.net = SmallCNN()
        self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.net.eval()
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/uav_cam/image_raw", Image, self.on_img, queue_size=1)
        self.busy = False
        self.count = 0
        rospy.loginfo("CNN 推理已加载: %s", model_path)

    def on_img(self, msg):
        if self.busy:
            return
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            a = buf.reshape(msg.height, msg.step)[:, :msg.width * 3]
            a = a.reshape(msg.height, msg.width, 3)
            gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (W, H)).astype(np.float32) / 255.0
            t = torch.from_numpy(small)[None, None, :, :]
            with torch.no_grad():
                out = self.net(t)[0].numpy()
            cmd = Twist()
            cmd.linear.x = float(max(0.0, min(0.8, out[0])))
            cmd.angular.z = float(max(-0.9, min(0.9, out[1])))
            self.pub.publish(cmd)
            self.count += 1
            if self.count % 100 == 0:
                rospy.loginfo("推理 %d 帧  v=%.2f w=%.2f", self.count, cmd.linear.x, cmd.angular.z)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "推理失败: %s", e)


def main():
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/bc_data/model.pt")
    rospy.init_node("uav_cnn_driver")
    CnnDriver(path)
    rospy.spin()


if __name__ == "__main__":
    main()
