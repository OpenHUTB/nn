# drone_controller.py
"""无人机核心控制类"""

import airsim
import time
from config import FlightConfig
from collision_handler import CollisionHandler

class DroneController:
    def __init__(self):
        print("🔌 正在连接 AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("✅ 连接成功！")
        
        print("🎮 获取控制权...")
        self.client.enableApiControl(True)
        print("🔓 解锁电机...")
        self.client.armDisarm(True)
        print("✅ 初始化完成")
        
        self.collision_handler = CollisionHandler(self.client)
    
    def takeoff(self):
        """起飞"""
        print("🚀 起飞中...")
        
        start_time = time.time()
        self.client.takeoffAsync().join()
        
        if time.time() - start_time > FlightConfig.TAKEOFF_TIMEOUT:
            print("❌ 起飞超时！")
            return False
        
        print(f"📈 上升到 {abs(FlightConfig.TAKEOFF_HEIGHT)} 米...")
        self.client.moveToZAsync(FlightConfig.TAKEOFF_HEIGHT, 1).join()
        
        pos = self.get_position()
        print(f"✅ 起飞完成，当前位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})")
        return True
    
    def fly_to_position(self, x, y, z, velocity=None):
        """飞向目标点"""
        if velocity is None:
            velocity = FlightConfig.FLIGHT_VELOCITY
        
        print(f"✈️  正在飞往: ({x}, {y}, {z})")
        start_pos = self.get_position()
        print(f"   起始位置: ({start_pos.x_val:.1f}, {start_pos.y_val:.1f}, {start_pos.z_val:.1f})")
        
        # 开始飞行
        self.client.moveToPositionAsync(x, y, z, velocity)
        
        start_time = time.time()
        while time.time() - start_time < FlightConfig.MAX_FLIGHT_TIME:
            # 检查碰撞
            is_serious, collision = self.collision_handler.check_collision()
            if is_serious:
                self.client.cancelLastTask()
                self.client.hoverAsync().join()
                return False
            
            # 检查是否到达
            if self.is_at_position(x, y, z):
                print(f"📍 成功到达目标点 ({x}, {y}, {z})")
                return True
            
            time.sleep(0.1)
        
        print("❌ 飞行超时！")
        return False
    
    def safe_land(self):
        """安全降落"""
        print("\n🛬 开始安全降落流程...")
        
        for attempt in range(FlightConfig.LANDING_MAX_ATTEMPTS):
            try:
                print(f"   尝试 {attempt + 1}/{FlightConfig.LANDING_MAX_ATTEMPTS}: 稳定无人机...")
                self.client.hoverAsync().join()
                time.sleep(1)
                
                pos = self.get_position()
                current_height = -pos.z_val
                print(f"   当前位置: ({pos.x_val:.1f}, {pos.y_val:.1f})")
                print(f"   当前高度: {current_height:.2f}m")
                
                if current_height < 0.3:
                    print("✅ 无人机已经在地面")
                    return True
                
                # 执行降落
                print("   执行降落命令...")
                self.client.landAsync().join()
                
                # 等待降落完成
                for _ in range(int(FlightConfig.LANDING_MAX_WAIT / FlightConfig.LANDING_CHECK_INTERVAL)):
                    time.sleep(FlightConfig.LANDING_CHECK_INTERVAL)
                    current_z = self.get_position().z_val
                    if current_z >= 0:
                        print("✅ 降落成功！")
                        return True
                
            except Exception as e:
                print(f"❌ 降落尝试 {attempt + 1} 失败: {e}")
                time.sleep(1)
        
        # 最后手段：复位
        print("⚠️  常规降落失败，尝试复位...")
        try:
            self.client.reset()
            time.sleep(2)
            print("✅ 已复位")
            return True
        except:
            return False
    
    def emergency_stop(self):
        """紧急停止"""
        print("🚨 执行紧急停止！")
        try:
            self.client.cancelLastTask()
            self.client.hoverAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("✅ 已紧急停止")
        except Exception as e:
            print(f"⚠️  紧急停止异常: {e}")
    
    def get_position(self):
        """获取当前位置"""
        return self.client.getMultirotorState().kinematics_estimated.position
    
    def is_at_position(self, x, y, z):
        """判断是否到达目标点"""
        pos = self.get_position()
        distance = ((pos.x_val - x)**2 + (pos.y_val - y)**2 + (pos.z_val - z)**2)**0.5
        return distance < FlightConfig.ARRIVAL_TOLERANCE
    
    def cleanup(self):
        """清理资源"""
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("✅ 资源清理完成")
        except:
            pass