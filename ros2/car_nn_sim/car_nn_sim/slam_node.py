#!/usr/bin/env python3
"""ROS 2 SLAM node: lidar + odometry -> occupancy grid + map->base_link TF."""

import os
import sys

import numpy as np

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from mujoco_car_nn.slam import GridMap, scan_match  # noqa: E402


class SlamNode(Node):
    def __init__(self) -> None:
        super().__init__("car_slam")
        self.grid = GridMap()
        self.pose = (0.0, 0.0, 0.0)
        self.tf = TransformBroadcaster(self)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_map = self.create_publisher(OccupancyGrid, "/map", qos)
        self.create_subscription(LaserScan, "/car/scan", self.on_scan, 1)
        self.create_timer(1.0, self.publish_map)

    def on_scan(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        local = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])
        matched, _ = scan_match(local, self.grid, self.pose)
        self.pose = matched
        self.grid.update(self.pose, ranges, angles)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(self.pose[0])
        t.transform.translation.y = float(self.pose[1])
        t.transform.rotation.z = float(np.sin(self.pose[2] / 2))
        t.transform.rotation.w = float(np.cos(self.pose[2] / 2))
        self.tf.sendTransform(t)

    def publish_map(self) -> None:
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = self.grid.resolution
        msg.info.width = self.grid.width
        msg.info.height = self.grid.height
        msg.info.origin.position.x = self.grid.x_min
        msg.info.origin.position.y = self.grid.y_min
        prob = self.grid.probability()
        data = np.full(prob.shape, -1, dtype=np.int8)
        data[prob > 0.6] = 100
        data[prob < 0.4] = 0
        msg.data = data.ravel().tolist()
        self.pub_map.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
