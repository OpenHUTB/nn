#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉伺服：让相机始终对准红色目标，并逐步靠近。

控制律:
    yaw     : PID(横向偏差)，把目标拉回画面中央
    forward : 与目标面积误差成比例，接近到 target_area 就停
    丢失目标: 超过 search_after 秒 → 左右小幅扫视（±sweep_limit），不做整圈旋转

启动后前 warmup 秒保持静止，避免相机还没出图就开始乱转。
"""

import rospy
from geometry_msgs.msg import Point, Twist


class VisualServo(object):
    def __init__(self):
        self.topic_target = rospy.get_param("~target_topic", "/uav/perception/target")
        self.topic_cmd = rospy.get_param("~cmd_topic", "/cmd_vel")

        # PID
        self.kp = rospy.get_param("~kp", 1.1)
        self.kd = rospy.get_param("~kd", 0.25)
        self.yaw_max = rospy.get_param("~yaw_max", 0.9)

        # 前进
        self.v_max = rospy.get_param("~v_max", 1.0)
        self.target_area = rospy.get_param("~target_area", 0.10)
        self.stop_gain = rospy.get_param("~stop_gain", 4.0)

        # 搜索
        self.warmup = rospy.get_param("~warmup", 4.0)
        self.search_after = rospy.get_param("~search_after", 1.5)
        self.search_rate = rospy.get_param("~search_rate", 0.35)
        self.sweep_limit = rospy.get_param("~sweep_limit", 1.2)   # 弧度，约 ±69°

        self.t_start = rospy.Time.now()
        self.last_seen = rospy.Time.now()
        self.prev_err = 0.0
        self.prev_t = rospy.Time.now()
        self.latest = None
        self.sweep_dir = 1.0
        self.swept = 0.0

        self.pub = rospy.Publisher(self.topic_cmd, Twist, queue_size=1)
        self.sub = rospy.Subscriber(self.topic_target, Point, self.on_target, queue_size=1)
        self.rate = rospy.Rate(20)
        rospy.loginfo("视觉伺服启动（warmup %.1fs）", self.warmup)

    def on_target(self, msg):
        self.latest = msg
        if msg.z > 0.5:
            self.last_seen = rospy.Time.now()
            self.swept = 0.0

    def run(self):
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = max(1e-3, (now - self.prev_t).to_sec())
            self.prev_t = now

            cmd = Twist()

            # 1) 暖机：不动
            if (now - self.t_start).to_sec() < self.warmup:
                self.pub.publish(cmd)
                self.rate.sleep()
                continue

            seen = self.latest is not None and self.latest.z > 0.5
            lost = (now - self.last_seen).to_sec()

            if seen:
                err = self.latest.x
                area = self.latest.y
                d_err = (err - self.prev_err) / dt
                self.prev_err = err

                yaw = max(-self.yaw_max, min(self.yaw_max, self.kp * err + self.kd * d_err))
                v = max(0.0, min(self.v_max, self.stop_gain * (self.target_area - area)))
                if abs(err) > 0.5:
                    v *= 0.3
                cmd.angular.z = yaw
                cmd.linear.x = v
            else:
                self.prev_err = 0.0
                if lost > self.search_after:
                    # 2) 左右扫视，扫到边界就换向
                    cmd.angular.z = self.search_rate * self.sweep_dir
                    self.swept += self.search_rate * dt
                    if self.swept > self.sweep_limit:
                        self.sweep_dir *= -1.0
                        self.swept = 0.0
                    rospy.loginfo_throttle(2.0, "搜索中，方向 %+.0f", self.sweep_dir)
                else:
                    # 刚丢失：先停一下，别乱动
                    cmd.angular.z = 0.0
                    cmd.linear.x = 0.0

            self.pub.publish(cmd)
            self.rate.sleep()


def main():
    rospy.init_node("uav_visual_servo")
    VisualServo().run()


if __name__ == "__main__":
    main()
