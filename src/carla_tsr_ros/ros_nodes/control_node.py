#!/usr/bin/env python3
"""控制节点：订阅 /perception/traffic_signs，检测到 stop sign 时发布刹车指令"""
import json
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class ControlNode:
    def __init__(self):
        rospy.init_node('control_node', anonymous=False)

        self.stop_area_ratio = rospy.get_param('~stop_distance', 0.3)
        self.brake = rospy.get_param('~brake_value', 1.0)
        self.throttle = rospy.get_param('~throttle_value', 0.0)

        self.cmd_pub = rospy.Publisher('/vehicle/control_cmd', Twist, queue_size=10)
        rospy.Subscriber('/perception/traffic_signs', String, self.on_detections, queue_size=10)

        rospy.loginfo("Control node ready")

    def on_detections(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            rospy.logwarn("Invalid JSON in detection message")
            return

        detections = data.get('detections', [])
        cmd = Twist()

        stop_triggered = False
        for d in detections:
            if d['class_name'] == 'stop sign':
                x1, y1, x2, y2 = d['bbox']
                area_ratio = ((x2 - x1) * (y2 - y1)) / (640.0 * 480.0)
                if area_ratio >= self.stop_area_ratio:
                    stop_triggered = True
                    rospy.logwarn_throttle(2.0,
                        f"STOP sign close (area={area_ratio:.2f}), braking")

        if stop_triggered:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            cmd.linear.x = 1.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


if __name__ == '__main__':
    try:
        ControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
