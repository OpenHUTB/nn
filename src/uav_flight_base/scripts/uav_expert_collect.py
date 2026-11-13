#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务4 第一步：采集"专家"数据（行为克隆）。

专家策略不看图像，用 Gazebo 真值位姿算朝向目标的偏航误差，输出
(v, w) 控制量；同时把相机图像存盘。训练时只用图像 → 控制量，
推理时也不需要真值，实现真正的"端到端"。

产物:
    ~/bc_data/images/000000.png ...   64x48 灰度图
    ~/bc_data/labels.csv              frame,v,w
"""

import math
import os

import rospy
import cv2
import numpy as np
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

W, H = 64, 48
GOAL = (2.6, 0.3)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Collector(object):
    def __init__(self):
        self.out = rospy.get_param("~out_dir", os.path.expanduser("~/bc_data"))
        self.every = int(rospy.get_param("~every", 3))     # 每 3 帧存一张
        self.v_max = rospy.get_param("~v_max", 0.7)
        self.model_name = rospy.get_param("~model_name", "uav")

        self.img_dir = os.path.join(self.out, "images")
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir)

        self.labels = open(os.path.join(self.out, "labels.csv"), "w")
        self.labels.write("frame,v,w\n")

        self.count = 0
        self.seen = 0
        self.pose = None
        self.img = None
        self.latest = None

        rospy.Subscriber("/uav_cam/image_raw", Image, self.on_img, queue_size=1)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.on_states, queue_size=1)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.loginfo("采集开始 -> %s", self.out)

    def on_img(self, msg):
        self.img = msg

    def on_states(self, msg):
        try:
            i = msg.name.index(self.model_name)
        except ValueError:
            return
        p = msg.pose[i]
        self.pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def control(self, x, y, yaw):
        dx, dy = GOAL[0] - x, GOAL[1] - y
        dist = math.hypot(dx, dy)
        err = math.atan2(dy, dx) - yaw
        while err > math.pi:
            err -= 2 * math.pi
        while err < -math.pi:
            err += 2 * math.pi
        w = max(-0.8, min(0.8, 1.3 * err))
        v = self.v_max * max(0.0, 1.0 - abs(err) / 1.4)
        if dist < 0.5:
            v, w = 0.0, 0.0
        return v, w

    def run(self):
        r = rospy.Rate(15)
        while not rospy.is_shutdown():
            if self.img is None or self.pose is None:
                r.sleep()
                continue

            m = self.img
            buf = np.frombuffer(m.data, dtype=np.uint8)
            small = None
            try:
                if m.encoding.lower() in ("rgb8", "bgr8"):
                    a = buf.reshape(m.height, m.step)[:, :m.width * 3]
                    a = a.reshape(m.height, m.width, 3)
                    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
                    small = cv2.resize(gray, (W, H))
            except Exception:
                small = None

            v, w = self.control(*self.pose)
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.pub.publish(cmd)

            self.seen += 1
            if small is not None and self.seen % self.every == 0:
                name = "%06d.png" % self.count
                cv2.imwrite(os.path.join(self.img_dir, name), small)
                self.labels.write("%s,%.3f,%.3f\n" % (name, v, w))
                self.labels.flush()
                self.count += 1
                if self.count % 50 == 0:
                    rospy.loginfo("已采集 %d 帧", self.count)
            r.sleep()

        self.labels.close()
        rospy.loginfo("采集结束，共 %d 帧", self.count)


def main():
    rospy.init_node("uav_expert_collect")
    Collector().run()


if __name__ == "__main__":
    main()
