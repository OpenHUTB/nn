import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray  # 用数组发布关节角度
import mujoco
import numpy as np

class MujocoPublisher(Node):
    def __init__(self):
        super().__init__('mujoco_publisher')
        
        # 1. 加载MuJoCo模型（从参数获取路径）
        self.declare_parameter('model_path', 'config/humanoid.xml')
        model_path = self.get_parameter('model_path').value
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 2. 创建发布器（发布关节角度）
        self.joint_pub = self.create_publisher(
            Float64MultiArray, '/mujoco/joint_angles', 10)
        
        # 3. 定时运行仿真（50Hz）
        self.timer = self.create_timer(0.5, self.run_simulation)
        self.get_logger().info('MuJoCo publisher started. Publishing joint angles...')

    def run_simulation(self):
        # 运行一步仿真
        mujoco.mj_step(self.model, self.data)
        
        # 提取关节角度（简化：取前6个关节角度）
        joint_angles = self.data.qpos[:6].tolist()  # qpos是关节位置数组
        
        # 打包成ROS消息并发布
        msg = Float64MultiArray()
        msg.data = joint_angles
        self.joint_pub.publish(msg)
        self.get_logger().debug(f'Published: {joint_angles}')  # 调试信息

def main(args=None):
    rclpy.init(args=args)
    node = MujocoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()