#!/usr/bin/env python3
"""ROS 1 simulation node for the MuJoCo neural-network car.

Publishes ``/car/camera/image_raw``, ``/car/scan`` and ``/car/odom`` and
consumes ``/car/cmd_vel`` (linear.x in m/s -> throttle, angular.z in
[-1, 1] -> normalised steering).  When the ``~policy`` parameter points at
a checkpoint the node drives itself with the neural network instead of
waiting for teleoperation commands.
"""

import os
import sys

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan

# Make the repository's python package importable when running from the
# catkin workspace (repo_root/src/mujoco_car_nn).
_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from mujoco_car_nn.sim import LIDAR_RANGE, CarSim, SimConfig  # noqa: E402
from mujoco_car_nn.track import build_track  # noqa: E402


def to_image_msg(img: np.ndarray, stamp) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = "camera_link"
    msg.height, msg.width = img.shape[:2]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = img.tobytes()
    return msg


def to_scan_msg(ranges: np.ndarray, stamp) -> LaserScan:
    msg = LaserScan()
    msg.header.stamp = stamp
    msg.header.frame_id = "laser_link"
    msg.angle_min = -np.pi
    msg.angle_max = np.pi
    msg.angle_increment = 2 * np.pi / len(ranges)
    msg.range_min = 0.05
    msg.range_max = LIDAR_RANGE
    msg.ranges = ranges.astype(np.float32).tolist()
    return msg


def to_odom_msg(st, stamp) -> Odometry:
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = "odom"
    msg.child_frame_id = "base_link"
    msg.pose.pose.position.x = st.x
    msg.pose.pose.position.y = st.y
    msg.pose.pose.orientation.z = float(np.sin(st.yaw / 2))
    msg.pose.pose.orientation.w = float(np.cos(st.yaw / 2))
    msg.twist.twist.linear.x = st.speed
    msg.twist.twist.angular.z = st.yaw_rate
    return msg


class CarSimNode:
    """Simulation node: sensors out, ``/car/cmd_vel`` or policy in."""

    def __init__(self) -> None:
        rospy.init_node("car_nn_sim")
        self.scene = rospy.get_param("~scene", "day")
        policy_path = rospy.get_param("~policy", "")
        self.offset = float(rospy.get_param("~offset", 0.0))
        self.rate = float(rospy.get_param("~rate", 20.0))

        track = build_track()
        self.sim = CarSim(
            track, SimConfig(scene=self.scene, seed=int(rospy.get_param("~seed", 0)), randomize_start=False)
        )
        self.st = self.sim.reset(self.offset)
        self.policy = _load_policy(policy_path) if policy_path else None
        if self.policy is not None:
            rospy.loginfo("driving with policy %s", policy_path)

        self.pub_img = rospy.Publisher("/car/camera/image_raw", Image, queue_size=1)
        self.pub_scan = rospy.Publisher("/car/scan", LaserScan, queue_size=1)
        self.pub_odom = rospy.Publisher("/car/odom", Odometry, queue_size=1)
        self.cmd = [0.0, 0.0]
        rospy.Subscriber("/car/cmd_vel", Twist, self.on_cmd, queue_size=1)

    def on_cmd(self, msg: Twist) -> None:
        self.cmd[0] = float(np.clip(msg.angular.z, -1.0, 1.0))
        self.cmd[1] = float(np.clip(msg.linear.x / 8.0, -1.0, 1.0))

    def step(self) -> None:
        stamp = rospy.Time.now()
        img = self.sim.camera_image()
        scan = self.sim.lidar()
        self.pub_img.publish(to_image_msg(img, stamp))
        self.pub_scan.publish(to_scan_msg(scan, stamp))
        self.pub_odom.publish(to_odom_msg(self.st, stamp))

        action = self.policy(img, self.st.speed) if self.policy is not None else np.array(self.cmd, dtype=np.float32)
        self.st = self.sim.step(action)

    def spin(self) -> None:
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()


def _load_policy(path: str):
    from mujoco_car_nn.cli import load_policy

    return load_policy(path)


def main() -> None:
    CarSimNode().spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
