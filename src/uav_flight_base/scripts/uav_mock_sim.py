#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小无人机运动学仿真（无需 Gazebo）。

订阅 /uav/cmd_vel，按一阶惯性模型积分出位置与姿态，
发布 /uav/odom（nav_msgs/Odometry）和 map->base_link 的 TF。

用途：
    1) 没有模拟器时也能验证遥控、控制器、可视化链路
    2) 给后续 SLAM / 端到端模块提供稳定的 odom 源
"""

import math

import rospy
import tf2_ros
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


class MockUav(object):
    def __init__(self):
        self.topic_cmd = rospy.get_param("~cmd_topic", "/uav/cmd_vel")
        self.topic_odom = rospy.get_param("~odom_topic", "/uav/odom")
        self.rate_hz = float(rospy.get_param("~rate", 50.0))
        self.tau = float(rospy.get_param("~time_constant", 0.35))   # 一阶惯性
        self.max_speed = float(rospy.get_param("~max_speed", 5.0))
        self.z_min = float(rospy.get_param("~z_min", 0.2))
        self.z_max = float(rospy.get_param("~z_max", 30.0))

        self.x = float(rospy.get_param("~init_x", 0.0))
        self.y = float(rospy.get_param("~init_y", 0.0))
        self.z = float(rospy.get_param("~init_z", 1.5))
        self.yaw = float(rospy.get_param("~init_yaw", 0.0))

        self.vx = self.vy = self.vz = 0.0
        self.wz = 0.0

        self.cmd = Twist()
        self.sub = rospy.Subscriber(self.topic_cmd, Twist, self.on_cmd, queue_size=1)
        self.pub = rospy.Publisher(self.topic_odom, Odometry, queue_size=1)
        self.br = tf2_ros.TransformBroadcaster()
        self.last = rospy.Time.now()

    def on_cmd(self, msg):
        self.cmd = msg

    def clip(self, v):
        return max(-self.max_speed, min(self.max_speed, v))

    def step(self, dt):
        ax = (self.cmd.linear.x - self.vx) / self.tau
        ay = (self.cmd.linear.y - self.vy) / self.tau
        az = (self.cmd.linear.z - self.vz) / self.tau
        aw = (self.cmd.angular.z - self.wz) / self.tau

        self.vx = self.clip(self.vx + ax * dt)
        self.vy = self.clip(self.vy + ay * dt)
        self.vz = self.clip(self.vz + az * dt)
        self.wz = self.wz + aw * dt

        # 机体速度转到世界坐标系
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (self.vx * cy - self.vy * sy) * dt
        self.y += (self.vx * sy + self.vy * cy) * dt
        self.z = max(self.z_min, min(self.z_max, self.z + self.vz * dt))
        self.yaw += self.wz * dt

    def publish(self, now):
        qx, qy, qz, qw = yaw_to_quat(self.yaw)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = self.z
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = self.vy
        odom.twist.twist.linear.z = self.vz
        odom.twist.twist.angular.z = self.wz
        self.pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = "map"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.translation.z = self.z
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.br.sendTransform(tf)

    def spin(self):
        r = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = max(1e-4, (now - self.last).to_sec())
            self.last = now
            self.step(min(dt, 0.1))
            self.publish(now)
            r.sleep()


def main():
    rospy.init_node("uav_mock_sim")
    MockUav().spin()


if __name__ == "__main__":
    main()
