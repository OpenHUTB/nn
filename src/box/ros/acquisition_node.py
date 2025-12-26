# 导入ROS 2核心库
import rclpy
from rclpy.node import Node
# 导入关节消息类型
from sensor_msgs.msg import JointState
# 导入文件操作相关库
import csv
from datetime import datetime
import os

class ArmDataAcquisitionNode(Node):
    """数据获取模块节点：订阅关节数据并保存为CSV"""
    def __init__(self):
        super().__init__('arm_data_acquisition_node')
        
        # 创建数据保存目录（带时间戳，避免重名）
        self.save_dir = os.path.expanduser(f"~/ros2_arm_data/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建CSV文件并写入表头
        self.csv_file_path = os.path.join(self.save_dir, 'arm_joint_data.csv')
        with open(self.csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['时间戳(秒)', '关节名', '关节角度(弧度)'])
        
        # 创建订阅者：订阅/arm/joint_states话题，回调函数处理数据
        self.joint_subscriber = self.create_subscription(
            JointState,
            '/arm/joint_states',
            self.joint_data_callback,
            10  # 队列大小
        )
        
        # 日志提示：节点启动成功
        self.get_logger().info(f"数据获取模块已启动！数据保存路径：{self.csv_file_path}")

    def joint_data_callback(self, msg):
        """订阅回调函数：处理接收到的关节数据"""
        # 计算时间戳（秒，精确到小数点后2位）
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        timestamp = round(timestamp, 2)
        
        # 将每个关节的角度写入CSV文件
        with open(self.csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for joint_name, joint_angle in zip(msg.name, msg.position):
                writer.writerow([timestamp, joint_name, round(joint_angle, 2)])
        
        # 日志输出：确认数据保存
        self.get_logger().info(f"保存数据：时间戳={timestamp}，joint2角度={round(msg.position[1], 2)}")

def main(args=None):
    """主函数：启动数据获取节点"""
    rclpy.init(args=args)
    node = ArmDataAcquisitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()