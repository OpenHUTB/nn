import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
import numpy as np
# 导入仓库中的感知模块
from my-test-repo.main.BasicWithEndEffectorPosition import BasicWithEndEffectorPosition
from my-test-repo.mobl_arms_bimanual.sp import MoblArmsSimulator

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        # 初始化仿真器（假设模型路径已知）
        self.model_path = "~/ros2_ws/src/my-test-repo/mobl_arms_bimanual/bm_model.xml"
        self.simulator = MoblArmsSimulator(model_path=self.model_path)
        
        # 初始化感知模块（配置末端执行器，参考仓库定义）
        self.end_effector = [["site", "right_hand_site"], ["site", "left_hand_site"]]  # 从bm_model.xml中获取的末端执行器名称
        self.perception = BasicWithEndEffectorPosition(
            model=self.simulator.model,
            data=self.simulator.data,
            bm_model=self.simulator.bimanual_model,
            end_effector=self.end_effector
        )
        
        # 创建ROS话题发布者
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.end_effector_pub = self.create_publisher(PointStamped, '/end_effector_position', 10)
        
        # 定时器（10Hz发布感知数据）
        self.timer = self.create_timer(0.1, self.publish_perception_data)
        self.get_logger().info("感知模块启动成功")

    def publish_perception_data(self):
        # 1. 发布关节状态（角度、速度）
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        # 从仿真器获取关节名称和角度（参考仓库中_joint_id映射）
        joint_names = ["shoulder_elv_r", "shoulder_elv_l", "elbow_flexion_r", "elbow_flexion_l"]
        joint_msg.name = joint_names
        joint_msg.position = [self.simulator.data.qpos[self.simulator._get_joint_id(name)] for name in joint_names]
        joint_msg.velocity = [self.simulator.data.qvel[self.simulator._get_joint_id(name)] for name in joint_names]
        self.joint_state_pub.publish(joint_msg)
        
        # 2. 发布末端执行器位置（从感知模块获取）
        # 假设感知模块输出包含末端执行器坐标（需根据仓库代码调整）
        end_pos = self.perception._get_end_effector_positions()  # 需在BasicWithEndEffectorPosition中实现该方法
        for i, pos in enumerate(end_pos):
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = f"end_effector_{i}"
            point_msg.point.x = pos[0]
            point_msg.point.y = pos[1]
            point_msg.point.z = pos[2]
            self.end_effector_pub.publish(point_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()