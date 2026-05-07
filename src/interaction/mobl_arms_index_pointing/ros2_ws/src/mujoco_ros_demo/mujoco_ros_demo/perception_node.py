#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
import numpy as np


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        # 1. 初始化数据存储变量
        self.finger_joint_angles = None  # 手指关节角度
        self.finger_tip_pos = None       # 手指末端3D坐标 [x,y,z]
        self.target_ball_pos = None      # 目标小球3D坐标 [x,y,z]
        
        # 2. 创建订阅者
        self.joint_sub = self.create_subscription(
            Float64MultiArray,
            '/perception/finger_joint_angles',  # 关节角度话题（Float64MultiArray）
            self.joint_angle_callback,
            10
        )
        self.finger_tip_sub = self.create_subscription(
            PointStamped,
            '/perception/finger_tip_pos',       # 手指末端位置话题（PointStamped）
            self.finger_tip_callback,
            10
        )
        self.target_sub = self.create_subscription(
            PointStamped,
            '/perception/target_ball_pos',      # 目标小球位置话题（PointStamped）
            self.target_callback,
            10
        )
        
        # 3. 创建发布者（输出处理结果）
        self.result_pub = self.create_publisher(
            Float64MultiArray,  # 格式：[距离, 命中标记(1/0)]
            '/perception/result',
            10
        )
        
        # 4. 初始化定时器（周期性处理数据，10Hz）
        self.timer = self.create_timer(0.1, self.process_data)
        
        self.get_logger().info("感知模块已启动，等待仿真数据...")

    def joint_angle_callback(self, msg):
        """接收手指关节角度数据"""
        self.finger_joint_angles = msg.data
        self.get_logger().debug(f"收到关节角度：{self.finger_joint_angles}")

    def finger_tip_callback(self, msg):
        """接收手指末端位置数据"""
        self.finger_tip_pos = [msg.point.x, msg.point.y, msg.point.z]
        self.get_logger().debug(f"收到手指末端位置：{self.finger_tip_pos}")

    def target_callback(self, msg):
        """接收目标小球位置数据"""
        self.target_ball_pos = [msg.point.x, msg.point.y, msg.point.z]
        self.get_logger().debug(f"收到目标小球位置：{self.target_ball_pos}")

    def analyze_finger_pose(self):
        """解析手指姿态（伸直/弯曲）"""
        if self.finger_joint_angles is None:
            return "未知"
        
        # 假设关节角度绝对值<0.1为伸直，否则为弯曲（可根据模型调整阈值）
        avg_angle = np.mean(np.abs(self.finger_joint_angles))
        return "伸直" if avg_angle < 0.1 else "弯曲"

    def calculate_distance(self):
        """计算手指末端与目标小球的欧式距离"""
        if self.finger_tip_pos is None or self.target_ball_pos is None:
            return None
        
        finger = np.array(self.finger_tip_pos)
        target = np.array(self.target_ball_pos)
        distance = np.linalg.norm(finger - target)
        return round(distance, 4)

    def process_data(self):
        """周期性处理所有数据并发布结果"""
        # 1. 检查数据完整性
        if None in [self.finger_joint_angles, self.finger_tip_pos, self.target_ball_pos]:
            self.get_logger().warn("等待完整仿真数据...")
            return
        
        # 2. 解析手指姿态
        finger_pose = self.analyze_finger_pose()
        
        # 3. 计算空间距离
        distance = self.calculate_distance()
        hit_mark = 1.0 if distance < 0.05 else 0.0  # 距离<5cm判定为命中
        
        # 4. 打印日志（可视化调试）
        self.get_logger().info(f"=== 感知结果 ===")
        self.get_logger().info(f"食指姿态：{finger_pose}")
        self.get_logger().info(f"手指末端位置：{self.finger_tip_pos}")
        self.get_logger().info(f"目标小球位置：{self.target_ball_pos}")
        self.get_logger().info(f"空间距离：{distance} m | 命中标记：{int(hit_mark)}")
        
        # 5. 发布处理结果
        result_msg = Float64MultiArray()
        result_msg.data = [distance, hit_mark]
        self.result_pub.publish(result_msg)


def main(args=None):
    rclpy.init(args=args)
    perception_node = PerceptionNode()
    
    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info("感知模块已停止")
    finally:
        if rclpy.ok():  # 仅当上下文有效时销毁节点
            perception_node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()