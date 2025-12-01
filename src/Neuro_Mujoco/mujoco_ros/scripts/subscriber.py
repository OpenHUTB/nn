#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

class MujocoStateSubscriber:
    def __init__(self):
        rospy.init_node("mujoco_state_subscriber", anonymous=True)
        # 订阅关节状态
        rospy.Subscriber("/mujoco/joint_states", JointState, self.joint_state_cb)
        # 订阅基座姿态
        rospy.Subscriber("/mujoco/pose", PoseStamped, self.pose_cb)
        rospy.loginfo("="*50)
        rospy.loginfo("Mujoco 状态订阅者已启动")
        rospy.loginfo("订阅话题：/mujoco/joint_states、/mujoco/pose")
        rospy.loginfo("="*50)

    def joint_state_cb(self, msg):
        # 打印关节状态（位置+速度）
        rospy.loginfo("【关节状态】")
        rospy.loginfo(f"  关节名称：{msg.name}")
        rospy.loginfo(f"  关节位置：{[round(x,3) for x in msg.position[:5]]}...")
        rospy.loginfo(f"  关节速度：{[round(x,3) for x in msg.velocity[:5]]}...")

    def pose_cb(self, msg):
        # 打印基座姿态（位置+四元数）
        rospy.loginfo("【基座姿态】")
        rospy.loginfo(f"  位置：x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}")
        rospy.loginfo(f"  姿态：qx={msg.pose.orientation.x:.3f}, qy={msg.pose.orientation.y:.3f}, qz={msg.pose.orientation.z:.3f}, qw={msg.pose.orientation.w:.3f}")

if __name__ == "__main__":
    try:
        subscriber = MujocoStateSubscriber()
        rospy.spin()  # 阻塞等待消息
    except rospy.ROSInterruptException:
        rospy.loginfo("状态订阅者退出")
