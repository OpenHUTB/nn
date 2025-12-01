#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np

class MujocoCtrlPublisher:
    def __init__(self):
        rospy.init_node("mujoco_ctrl_publisher", anonymous=True)
        self.pub = rospy.Publisher("/mujoco/ctrl_cmd", Float32MultiArray, queue_size=10)
        self.rate = rospy.Rate(10)
        self.model_nu = 8  # ant 模型 nu=8，若你的模型不同请修改
        rospy.loginfo("控制指令发布者已启动，发布 /mujoco/ctrl_cmd")

    def run(self):
        while not rospy.is_shutdown():
            # 正弦波控制指令
            ctrl_cmd = np.sin(rospy.get_time() * 2.0) * 0.3
            msg = Float32MultiArray()
            msg.data = [ctrl_cmd] * self.model_nu
            self.pub.publish(msg)
            rospy.loginfo(f"发布指令：{[round(x,3) for x in msg.data[:5]]}...")
            self.rate.sleep()

if __name__ == "__main__":
    try:
        publisher = MujocoCtrlPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
