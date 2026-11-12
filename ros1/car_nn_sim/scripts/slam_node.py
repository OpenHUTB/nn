#!/usr/bin/env python3
"""ROS 1 SLAM node: lidar + odometry -> occupancy grid + map->base_link TF."""

import os
import sys

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from mujoco_car_nn.slam import GridMap, scan_match  # noqa: E402
from mujoco_car_nn.track import build_track  # noqa: E402


class SlamNode:
    def __init__(self) -> None:
        rospy.init_node("car_slam")
        self.grid = GridMap()
        self.pose = None
        self.publish_rate = rospy.get_param("~publish_rate", 1.0)
        self.pub_map = rospy.Publisher("/map", OccupancyGrid, queue_size=1, latch=True)
        self.tf = TransformBroadcaster()
        self.sub_scan = rospy.Subscriber("/car/scan", LaserScan, self.on_scan, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish_map)

    def on_scan(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        if self.pose is None:
            self.pose = (0.0, 0.0, 0.0)
        matched, _ = scan_match(np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)]), self.grid, self.pose)
        self.pose = matched
        self.grid.update(self.pose, ranges, angles)
        self._broadcast()

    def _broadcast(self) -> None:
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(self.pose[0])
        t.transform.translation.y = float(self.pose[1])
        t.transform.rotation.z = float(np.sin(self.pose[2] / 2))
        t.transform.rotation.w = float(np.cos(self.pose[2] / 2))
        self.tf.sendTransform(t)

    def publish_map(self, _event=None) -> None:
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
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


if __name__ == "__main__":
    try:
        node = SlamNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
