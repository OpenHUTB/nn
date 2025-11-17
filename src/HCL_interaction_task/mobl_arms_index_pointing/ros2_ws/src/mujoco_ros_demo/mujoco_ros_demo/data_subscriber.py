import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')
        
        # 订阅关节角度话题
        self.subscription = self.create_subscription(
            Float64MultiArray, '/mujoco/joint_angles', self.listener_callback, 10)
        self.subscription  # 防止未使用变量警告
        self.get_logger().info('Data subscriber started. Waiting for joint angles...')

    def listener_callback(self, msg):
        # 处理收到的数据：计算平均值
        joint_angles = msg.data
        avg_angle = sum(joint_angles) / len(joint_angles) if joint_angles else 0.0
        
        # 打印结果
        self.get_logger().info(
            f'Received joint angles: {joint_angles}\n'
            f'Average angle: {avg_angle:.2f}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = DataSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()