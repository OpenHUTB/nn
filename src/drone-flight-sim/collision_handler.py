# collision_handler.py
"""碰撞检测和处理模块"""

import time
from config import FlightConfig, GROUND_OBJECTS

class CollisionHandler:
    def __init__(self, client):
        self.client = client
        self.collision_count = 0
        self.last_collision_time = 0
        self.is_collided = False
    
    def check_collision(self):
        """检测碰撞，返回 (是否严重碰撞, 碰撞信息)"""
        collision_info = self.client.simGetCollisionInfo()
        
        if not collision_info.has_collided:
            return False, None
        
        current_time = time.time()
        if current_time - self.last_collision_time < FlightConfig.COLLISION_COOLDOWN:
            return False, None  # 冷却期内忽略
        
        self.last_collision_time = current_time
        self.collision_count += 1
        
        # 获取当前高度
        drone_pos = self.client.getMultirotorState().kinematics_estimated.position
        current_height = -drone_pos.z_val
        
        # 判断是否为地面接触
        is_ground = (
            current_height < FlightConfig.GROUND_HEIGHT_THRESHOLD or
            any(keyword in collision_info.object_name for keyword in GROUND_OBJECTS)
        )
        
        if is_ground:
            print(f"⚠️  检测到与 {collision_info.object_name} 接触（高度: {current_height:.2f}m），忽略")
            return False, None
        
        # 严重碰撞
        print(f"\n💥 严重碰撞发生！")
        print(f"   碰撞位置: ({collision_info.position.x_val:.2f}, "
              f"{collision_info.position.y_val:.2f}, {collision_info.position.z_val:.2f})")
        print(f"   碰撞物体: {collision_info.object_name}")
        print(f"   当前高度: {current_height:.2f}m")
        print(f"   碰撞次数: {self.collision_count}")
        
        return True, collision_info
    
    def reset_collision_state(self):
        """重置碰撞状态"""
        self.is_collided = False