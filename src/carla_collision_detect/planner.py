import carla
import math
import numpy as np
import time

class LanePlanner:
    def __init__(self, ego_vehicle, world):
        self.ego_vehicle = ego_vehicle
        self.world = world
        self.map = self.world.get_map()
        
        self.is_changing_lane = False
        self.target_lane_id = None
        self.target_lane_dir = None
        self.cooldown_end_time = 0.0
        
        # 🌟 PID 控制器状态变量 (防抖防歪)
        self.lat_error_sum = 0.0
        self.prev_lat_error = 0.0
        
        # 🌟 PID 参数调优 (柔和且精准，保证变道后平滑居中)
        self.Kp = 1.2   
        self.Ki = 0.02  
        self.Kd = 0.2   

    def get_lateral_control(self, current_dist, current_speed_kmh):
        """
        核心：全时车道保持 (LKA) + 自动变道 (ALC)
        """
        trigger_dist = 35.0
        current_time = time.time()
        curr_wp = self.map.get_waypoint(self.ego_vehicle.get_location())
        
        # 1. 障碍物触发变道
        if not self.is_changing_lane and current_dist < trigger_dist and current_speed_kmh > 10 and current_time > self.cooldown_end_time:
            left_wp = curr_wp.get_left_lane()
            right_wp = curr_wp.get_right_lane()
            
            can_change_right = curr_wp.lane_change in [carla.LaneChange.Right, carla.LaneChange.Both]
            can_change_left = curr_wp.lane_change in [carla.LaneChange.Left, carla.LaneChange.Both]
            
            if right_wp and right_wp.lane_type == carla.LaneType.Driving and can_change_right:
                print("\033[92m🚀 执行【右侧】变道...\033[0m")
                self.is_changing_lane = True
                self.target_lane_id = right_wp.lane_id
                self.target_lane_dir = 'right'
                self.lat_error_sum = 0.0  # 清空历史误差
                
            elif left_wp and left_wp.lane_type == carla.LaneType.Driving and can_change_left:
                print("\033[92m🚀 执行【左侧】变道...\033[0m")
                self.is_changing_lane = True
                self.target_lane_id = left_wp.lane_id
                self.target_lane_dir = 'left'
                self.lat_error_sum = 0.0  # 清空历史误差
                
        return self._calculate_steer(curr_wp)

    def _calculate_steer(self, current_wp):
        ego_trans = self.ego_vehicle.get_transform()
        ego_loc = ego_trans.location
        
        # ==========================================
        # 🛑 变道状态监控与切换
        # ==========================================
        if self.is_changing_lane:
            if current_wp.lane_id == self.target_lane_id:
                ego_yaw = ego_trans.rotation.yaw
                lane_yaw = current_wp.transform.rotation.yaw
                diff = abs((ego_yaw - lane_yaw) % 360)
                if diff > 180: diff = 360 - diff
                
                if diff < 5.0:
                    self.is_changing_lane = False
                    self.cooldown_end_time = time.time() + 3.0 
                    print("✅ 变道完成")

        # ==========================================
        # 🎯 寻找追踪目标点
        # ==========================================
        track_lane_id = self.target_lane_id if self.is_changing_lane else current_wp.lane_id
        
        fwd_wps = current_wp.next(5.0)
        if not fwd_wps: return 0.0 
        fwd_wp = fwd_wps[0]
        
        target_wp = fwd_wp
        if fwd_wp.lane_id != track_lane_id:
            if self.is_changing_lane:
                if self.target_lane_dir == 'right':
                    temp = fwd_wp.get_right_lane()
                    if temp: target_wp = temp
                else:
                    temp = fwd_wp.get_left_lane()
                    if temp: target_wp = temp

        target_loc = target_wp.transform.location
        
        ego_yaw = ego_trans.rotation.yaw
        
        # 2. 算出目标点在世界里的绝对角度 (利用最基础的 arctan2)
        target_yaw = math.degrees(math.atan2(target_loc.y - ego_loc.y, target_loc.x - ego_loc.x))
        
        angle_diff = (target_yaw - ego_yaw) % 360
        if angle_diff > 180:
            angle_diff -= 360

        steer_value = angle_diff / 25.0
        
        # 限制方向盘打死幅度，防止侧翻
        steer_value = np.clip(steer_value, -0.8, 0.8)

        return steer_value