#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Float64MultiArray, Float64
import sys
import os
import threading

# ========== 关键：添加simulator.py所在目录到Python路径 ==========
# 使用绝对路径（已验证存在）
simulator_dir = '/home/lbxlb/HCL_interaction_task/mobl_arms_index_pointing'
sys.path.append(simulator_dir)
print(f"Added path: {simulator_dir}")
print(f"Path exists: {os.path.exists(simulator_dir)}")
print(f"simulator.py exists: {os.path.exists(os.path.join(simulator_dir, 'simulator.py'))}")

try:
    from simulator import Simulator  # 导入你的仿真器
except ImportError as e:
    print(f"Failed to import Simulator: {e}")
    sys.exit(1)

class DataAcquisitionNode(Node):
    def __init__(self):
        super().__init__('data_acquisition_node')
        
        # 发布器定义（对应仿真动态数据）
        self.finger_joint_pub = self.create_publisher(
            Float64MultiArray, '/perception/finger_joint_angles', 10)
        self.target_ball_pub = self.create_publisher(
            PointStamped, '/perception/target_ball_pos', 10)
        self.finger_tip_pub = self.create_publisher(
            PointStamped, '/perception/finger_tip_pos', 10)
        self.sim_time_pub = self.create_publisher(
            Float64, '/perception/sim_time', 10)
        
        # ========== 初始化仿真器（使用正确的绝对路径） ==========
        self.config_path = os.path.join(simulator_dir, 'config.yaml')
        self.model_path = os.path.join(simulator_dir, 'simulation.xml')
        
        self.get_logger().info(f"Loading config: {self.config_path}")
        self.get_logger().info(f"Loading model: {self.model_path}")
        
        try:
            self.sim = Simulator(config_path=self.config_path, model_path=self.model_path)
            self.get_logger().info("Simulator initialized successfully")
        except Exception as e:
            self.get_logger().fatal(f"Simulator init failed: {str(e)}")
            sys.exit(1)
        
        # 启动仿真线程（独立线程运行仿真，不阻塞ROS）
        self.sim_thread = threading.Thread(target=self.run_simulation)
        self.sim_thread.daemon = True
        self.sim_thread.start()
        
        # 定时发布数据（10Hz，与仿真频率匹配）
        self.timer = self.create_timer(1/10, self.publish_realtime_data)
        self.get_logger().info('ROS Data Acquisition Node Started!')

    def run_simulation(self):
        """独立线程运行仿真循环"""
        self.get_logger().info('Simulation started (gesture control enabled)')
        while self.sim.step():
            pass
        self.sim.close()
        self.get_logger().info('Simulation finished')

    def publish_realtime_data(self):
        """直接读取仿真数据并发布"""
        try:
            # 1. 发布手指关节角度（前5个关节）
            joint_msg = Float64MultiArray()
            joint_msg.data = self.sim.data.qpos[:5].tolist()
            self.finger_joint_pub.publish(joint_msg)
            
            # 2. 发布目标小球位置
            target_body_id = self.sim.model.body('target').id  
            ball_msg = PointStamped()
            ball_msg.header.stamp = self.get_clock().now().to_msg()
            ball_msg.header.frame_id = 'sim_world'
            ball_msg.point.x = self.sim.data.xpos[target_body_id][0]
            ball_msg.point.y = self.sim.data.xpos[target_body_id][1]
            ball_msg.point.z = self.sim.data.xpos[target_body_id][2]
            self.target_ball_pub.publish(ball_msg)
            
            # 3. 发布手指末端位置
            tip_msg = PointStamped()
            tip_msg.header.stamp = self.get_clock().now().to_msg()
            tip_msg.header.frame_id = 'sim_world'
            tip_msg.point.x = self.sim.data.xpos[self.sim.finger_tip_body_id][0]
            tip_msg.point.y = self.sim.data.xpos[self.sim.finger_tip_body_id][1]
            tip_msg.point.z = self.sim.data.xpos[self.sim.finger_tip_body_id][2]
            self.finger_tip_pub.publish(tip_msg)
            
            # 4. 发布仿真时间
            time_msg = Float64()
            time_msg.data = self.sim.data.time
            self.sim_time_pub.publish(time_msg)
            
        except Exception as e:
            self.get_logger().error(f"Data publish failed: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = DataAcquisitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if hasattr(node, 'sim'):
            node.sim.close()
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()