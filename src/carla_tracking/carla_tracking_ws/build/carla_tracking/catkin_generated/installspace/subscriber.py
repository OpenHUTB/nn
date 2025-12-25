# subscriber.py - ROS订阅器
import rospy
from std_msgs.msg import Float32, Bool, String
from geometry_msgs.msg import Twist

class CarlaSubscriber:
    def __init__(self, core):
        """
        Args:
            core: CarlaTrackingCore实例，用于控制核心逻辑
        """
        self.core = core
        self.logger = rospy.loginfo
        
        # 订阅控制指令
        self.vel_sub = rospy.Subscriber('/carla/control/cmd_vel', Twist, self.vel_callback)
        self.speed_sub = rospy.Subscriber('/carla/control/target_speed', Float32, self.speed_callback)
        self.reset_sub = rospy.Subscriber('/carla/control/reset_npc', Bool, self.reset_callback)
        self.config_sub = rospy.Subscriber('/carla/control/config', String, self.config_callback)
        
        # 控制状态
        self.manual_control = False
        self.target_speed = 30.0
        self.vel_cmd = Twist()

    def vel_callback(self, msg):
        """速度控制回调（手动控制）"""
        self.vel_cmd = msg
        self.manual_control = True
        # 转换为CARLA车辆控制
        control = carla.VehicleControl()
        control.throttle = max(0.0, min(1.0, msg.linear.x))
        control.brake = max(0.0, min(1.0, -msg.linear.x))
        control.steer = max(-1.0, min(1.0, msg.angular.z))
        self.core.vehicle.apply_control(control)

    def speed_callback(self, msg):
        """目标速度回调"""
        self.target_speed = msg.data
        self.core.controller.set_target_speed(self.target_speed)
        self.logger(f"更新目标速度为: {self.target_speed} km/h")

    def reset_callback(self, msg):
        """重置NPC回调"""
        if msg.data:
            self.logger("重置NPC车辆...")
            self.core.npc_manager.destroy_all_npcs()
            self.core.npc_manager.spawn_npcs(self.core.world, 
                                             count=self.core.config['npc']['count'], 
                                             ego_vehicle=self.core.vehicle)

    def config_callback(self, msg):
        """更新配置回调"""
        try:
            config = json.loads(msg.data)
            self.core.config_manager._update_config(self.core.config, config)
            self.logger(f"更新配置: {config}")
        except Exception as e:
            rospy.logerr(f"更新配置失败: {e}")

    def reset_manual_control(self):
        """重置手动控制（恢复自动模式）"""
        self.manual_control = False
        self.vel_cmd = Twist()
