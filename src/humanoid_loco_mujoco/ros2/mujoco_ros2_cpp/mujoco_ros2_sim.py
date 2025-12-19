import rclpy
from rclpy.node import Node
import numpy as np
import mujoco
import mujoco.viewer
from my_interfaces.srv import RobotCtrl
from my_interfaces.msg import RobotState
import time
current_action='stop'
class MujocoRos2Sim(Node):
    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info("节点已启动：%s!" % name)
        # 加载数据,模型,轨迹数据
        self.model=mujoco.MjModel.from_xml_path("RobotH.xml")
        self.data=mujoco.MjData(self.model)
        self.data_walk=np.load("walk.npz")
        self.data_squat=np.load("squat.npz")
        # 创建viewer
        self.viewer=mujoco.viewer.launch_passive(self.model, self.data)

        #创建服务用来控制model运动
        self.service=self.create_service(RobotCtrl, 'robot_control', self.control_callback)
        #创建话题用来发布当前状态
        self.pub=self.create_publisher(RobotState, 'robot_state', 10)

    def control_callback(self, request, response):

        self.get_logger().info(f"Received command: {request.command}")
        if request.command == "squat":
            current_action="squat"
        elif request.command == "walk":
            current_action="walk"
        elif request.command == "stop":
            current_action="stop"
        response.result=True
        return response
    
    def publish_status(self):
        # 发布当前状态
        msg=RobotState()
        msg.action=current_action
        self.pub.publish(msg)

    def run_sim(self):
        while True:
            # 控制动作
            if current_action=="squat":
                self.data.qpos[0]=self.data_squat["qpos"][0]
                self.data.qvel[0]=self.data_squat["qvel"][0]
            elif current_action=="walk":
                self.data.qpos[0]=self.data_walk["qpos"][0]
                self.data.qvel[0]=self.data_walk["qvel"][0]
            elif current_action=="stop":
                self.data.qpos[0]=0
                self.data.qvel[0]=0
            # 模拟一步
            mujoco.mj_step(self.model, self.data)
            # 实时渲染
            self.viewer.sync()
            # 发布状态
            self.publish_status()



def main(args=None):
    rclpy.init(args=args)
    sim=MujocoRos2Sim("mujoco_ros2_sim")
    sim.run_sim()
    rclpy.spin(sim)
    sim.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()