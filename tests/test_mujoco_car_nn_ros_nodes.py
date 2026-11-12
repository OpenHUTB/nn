"""Runtime smoke test of the ROS 1 / ROS 2 nodes without a ROS installation.

The node classes are executed against minimal stubs of ``rospy``/``rclpy``
and the message packages.  This catches attribute typos, wrong message
fields and control-loop mistakes on a machine that has no ROS, which is
exactly the situation of the CI runner.  Run with::

    python -m pytest tests/test_ros_nodes.py -q
"""

import importlib.util
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


class Msg:
    """Permissive stand-in for a ROS message: any attribute can be set or read."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        child = Msg()
        object.__setattr__(self, name, child)
        return child


def _msg_class(name):
    return type(name, (Msg,), {})


class _Publisher:
    def __init__(self, *args, **kwargs):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _Subscriber:
    def __init__(self, topic, msg_type, callback, **kwargs):
        self.topic = topic
        self.callback = callback


class _Rate:
    def __init__(self, hz):
        self.hz = hz

    def sleep(self):
        pass


def _install_ros1_stubs():
    rospy = types.ModuleType("rospy")
    params = {"~scene": "day", "~policy": "", "~offset": 0.0, "~rate": 20.0, "~seed": 0}

    rospy.init_node = lambda *a, **k: None
    rospy.get_param = lambda name, default=None: params.get(name, default)
    rospy.loginfo = lambda *a, **k: None
    rospy.Publisher = _Publisher
    rospy.Subscriber = _Subscriber
    rospy.Rate = _Rate
    rospy.is_shutdown = lambda: True
    rospy.Time = types.SimpleNamespace(now=lambda: 0.0)
    rospy.ROSInterruptException = RuntimeError

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.Twist = _msg_class("Twist")
    geometry_msgs_msg.TransformStamped = _msg_class("TransformStamped")
    geometry_msgs.msg = geometry_msgs_msg

    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = _msg_class("Odometry")
    nav_msgs_msg.OccupancyGrid = _msg_class("OccupancyGrid")
    nav_msgs.msg = nav_msgs_msg

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = _msg_class("Image")
    sensor_msgs_msg.LaserScan = _msg_class("LaserScan")
    sensor_msgs.msg = sensor_msgs_msg

    tf2_ros = types.ModuleType("tf2_ros")

    class _Broadcaster:
        def __init__(self, *a, **k):
            self.sent = []

        def sendTransform(self, t):
            self.sent.append(t)

    tf2_ros.TransformBroadcaster = _Broadcaster

    return {
        "rospy": rospy,
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs_msg,
        "nav_msgs": nav_msgs,
        "nav_msgs.msg": nav_msgs_msg,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "tf2_ros": tf2_ros,
    }


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with_stubs(stubs, fn):
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_ros1_sim_node_runs():
    stubs = _install_ros1_stubs()
    path = os.path.join(ROOT, "ros1", "car_nn_sim", "scripts", "sim_node.py")

    def run():
        module = _load(path, "ros1_sim_node")
        node = module.CarSimNode()
        idle = module.Twist()
        idle.angular.z = 0.0
        idle.linear.x = 0.0
        node.on_cmd(idle)
        for _ in range(3):
            node.step()
        assert len(node.pub_img.messages) == 3
        assert len(node.pub_scan.messages) == 3
        assert len(node.pub_odom.messages) == 3
        img = node.pub_img.messages[-1]
        assert (img.height, img.width, img.encoding) == (64, 64, "rgb8")
        assert len(img.data) == 64 * 64 * 3
        scan = node.pub_scan.messages[-1]
        assert len(scan.ranges) == 64
        assert scan.range_max == module.LIDAR_RANGE
        odom = node.pub_odom.messages[-1]
        assert odom.header.frame_id == "odom"
        assert abs(odom.pose.pose.orientation.w) > 0.0
        # a teleop command must actually move the car
        cmd = module.Twist()
        cmd.angular.z = 0.0
        cmd.linear.x = 6.0
        node.on_cmd(cmd)
        for _ in range(20):
            node.step()
        assert node.st.speed > 0.5, "teleop command should accelerate the car"
        node.sim.close()

    _with_stubs(stubs, run)


def test_ros2_sim_node_runs():
    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_qos = types.ModuleType("rclpy.qos")

    class _Param:
        def __init__(self, value):
            self.value = value

    params = {"scene": "day", "policy": "", "offset": 0.0, "rate": 20.0}

    class Node:
        def __init__(self, name):
            self.name = name
            self.publishers = []
            self.timers = []

        def declare_parameter(self, name, value=None):
            params.setdefault(name, value)

        def get_parameter(self, name):
            return _Param(params.get(name))

        def get_logger(self):
            return types.SimpleNamespace(info=lambda *a, **k: None)

        def create_publisher(self, msg_type, topic, qos):
            pub = _Publisher()
            self.publishers.append((topic, pub))
            return pub

        def create_subscription(self, topic, msg_type, callback, qos):
            self.subscription = (topic, callback)
            return None

        def create_timer(self, period, callback):
            self.timers.append(callback)
            return None

        def get_clock(self):
            return types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_msg=lambda: 0.0))

        def destroy_node(self):
            pass

    rclpy_node.Node = Node
    rclpy.node = rclpy_node
    rclpy.init = lambda *a, **k: None
    rclpy.spin = lambda node: None
    rclpy.shutdown = lambda: None

    rclpy_qos.QoSProfile = lambda **k: None
    rclpy_qos.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL=1)

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = _msg_class("Image")
    sensor_msgs_msg.LaserScan = _msg_class("LaserScan")
    sensor_msgs.msg = sensor_msgs_msg

    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = _msg_class("Odometry")
    nav_msgs_msg.OccupancyGrid = _msg_class("OccupancyGrid")
    nav_msgs.msg = nav_msgs_msg

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.Twist = _msg_class("Twist")
    geometry_msgs_msg.TransformStamped = _msg_class("TransformStamped")
    geometry_msgs.msg = geometry_msgs_msg

    tf2_ros = types.ModuleType("tf2_ros")

    class _Broadcaster:
        def __init__(self, *a, **k):
            self.sent = []

        def sendTransform(self, t):
            self.sent.append(t)

    tf2_ros.TransformBroadcaster = _Broadcaster

    stubs = {
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "nav_msgs": nav_msgs,
        "nav_msgs.msg": nav_msgs_msg,
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs_msg,
        "tf2_ros": tf2_ros,
    }
    path = os.path.join(ROOT, "ros2", "car_nn_sim", "car_nn_sim", "sim_node.py")

    def run():
        module = _load(path, "ros2_sim_node")
        node = module.CarSimNode()
        assert len(node.publishers) == 3
        for _ in range(3):
            node.step()
        topics = [t for t, _ in node.publishers]
        assert "/car/camera/image_raw" in topics and "/car/scan" in topics and "/car/odom" in topics
        img = node.publishers[0][1].messages[-1]
        assert len(img.data) == 64 * 64 * 3
        node.sim.close()

    _with_stubs(stubs, run)


if __name__ == "__main__":
    test_ros1_sim_node_runs()
    print("ok  ros1")
    test_ros2_sim_node_runs()
    print("ok  ros2")
