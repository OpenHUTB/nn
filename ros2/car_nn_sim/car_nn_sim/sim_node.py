#!/usr/bin/env python3
"""ROS 2 (Humble) simulation node for the MuJoCo neural-network car."""

import os
import sys

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan

_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from mujoco_car_nn.sim import LIDAR_RANGE, CarSim, SimConfig  # noqa: E402
from mujoco_car_nn.track import build_track  # noqa: E402


class CarSimNode(Node):
    def __init__(self) -> None:
        super().__init__("car_nn_sim")
        self.declare_parameter("scene", "day")
        self.declare_parameter("policy", "")
        self.declare_parameter("offset", 0.0)
        self.declare_parameter("rate", 20.0)

        scene = self.get_parameter("scene").value
        policy_path = self.get_parameter("policy").value
        rate = float(self.get_parameter("rate").value)
        offset = float(self.get_parameter("offset").value)

        track = build_track()
        self.sim = CarSim(track, SimConfig(scene=scene, seed=0, randomize_start=False))
        self.st = self.sim.reset(offset)
        self.policy = None
        if policy_path:
            from mujoco_car_nn.cli import load_policy

            self.policy = load_policy(policy_path)
            self.get_logger().info(f"driving with policy: {policy_path}")

        self.pub_img = self.create_publisher(Image, "/car/camera/image_raw", 1)
        self.pub_scan = self.create_publisher(LaserScan, "/car/scan", 1)
        self.pub_odom = self.create_publisher(Odometry, "/car/odom", 1)
        self.create_subscription(Twist, "/car/cmd_vel", self.on_cmd, 1)
        self.cmd = [0.0, 0.0]
        self.timer = self.create_timer(1.0 / rate, self.step)

    def on_cmd(self, msg: Twist) -> None:
        self.cmd[0] = float(np.clip(msg.angular.z, -1.0, 1.0))
        self.cmd[1] = float(np.clip(msg.linear.x / 8.0, -1.0, 1.0))

    def step(self) -> None:
        stamp = self.get_clock().now().to_msg()
        img = self.sim.camera_image()
        scan = self.sim.lidar()

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera_link"
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "rgb8"
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        self.pub_img.publish(msg)

        scan_msg = LaserScan()
        scan_msg.header.stamp = stamp
        scan_msg.header.frame_id = "laser_link"
        scan_msg.angle_min = -np.pi
        scan_msg.angle_max = np.pi
        scan_msg.angle_increment = 2 * np.pi / len(scan)
        scan_msg.range_min = 0.05
        scan_msg.range_max = LIDAR_RANGE
        scan_msg.ranges = scan.astype(np.float32).tolist()
        self.pub_scan.publish(scan_msg)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.st.x
        odom.pose.pose.position.y = self.st.y
        odom.pose.pose.orientation.z = float(np.sin(self.st.yaw / 2))
        odom.pose.pose.orientation.w = float(np.cos(self.st.yaw / 2))
        odom.twist.twist.linear.x = self.st.speed
        odom.twist.twist.angular.z = self.st.yaw_rate
        self.pub_odom.publish(odom)

        action = self.policy(img, self.st.speed) if self.policy is not None else np.array(self.cmd, dtype=np.float32)
        self.st = self.sim.step(action)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CarSimNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
