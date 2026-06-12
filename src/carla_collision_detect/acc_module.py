import time

class ACCController:
    def __init__(self, headway_time=1.5, safe_buffer=6.0):
        """
        :param headway_time: 安全时距（秒），即保持多少秒的车距
        :param safe_buffer: 最小停止安全缓冲区（米）
        """
        self.headway_time = headway_time
        self.safe_buffer = safe_buffer
        self.prev_distance = None
        self.last_time = None

    def update_target_speed(self, current_dist, ego_speed_kmh, original_target_speed):
        """
        根据前方动态障碍物的距离和速度，动态调整巡航目标车速
        """
        current_time = time.time()
        if current_dist == float('inf') or self.last_time is None:
            self.prev_distance = current_dist
            self.last_time = current_time
            self.smoothed_lead_speed_kmh = None
            self.final_steady_speed = None
            return original_target_speed

        dt = current_time - self.last_time
        if dt <= 0: 
            dt = 0.05

        ego_speed_ms = ego_speed_kmh / 3.6

        # 1. 终极过滤：抹平视觉框抖动带来的“假速度”
        lead_speed_kmh = 0.0
        if self.prev_distance is not None and self.prev_distance != float('inf'):
            relative_velocity = (current_dist - self.prev_distance) / dt
            raw_lead_speed_kmh = (ego_speed_ms + relative_velocity) * 3.6
            
            # ⚠️ 极强滤波：98% 信任历史速度，只给 2% 的权重给当前帧，彻底无视视觉框闪烁！
            if getattr(self, 'smoothed_lead_speed_kmh', None) is None:
                self.smoothed_lead_speed_kmh = raw_lead_speed_kmh
            else:
                self.smoothed_lead_speed_kmh = 0.02 * raw_lead_speed_kmh + 0.98 * self.smoothed_lead_speed_kmh
                
            lead_speed_kmh = max(0.0, self.smoothed_lead_speed_kmh)
            
        self.lead_speed_kmh = lead_speed_kmh

        # 更新历史状态
        self.prev_distance = current_dist
        self.last_time = current_time

        # 2. 核心死区逻辑：你要的“死死咬住前车速度”
        desired_safe_dist = (ego_speed_ms * self.headway_time) + self.safe_buffer
        distance_error = current_dist - desired_safe_dist

        # ⚠️ 只要前后车距误差在 ±5 米之内，完全放弃距离微调！直接复制前车速度！
        if abs(distance_error) < 5.0:
            raw_target_speed = lead_speed_kmh
        else:
            # 只有距离实在差得太远才进行极微弱的修正 (系数从 1.5 暴降到 0.2)
            raw_target_speed = lead_speed_kmh + (distance_error * 0.2)

        raw_target_speed = max(0.0, min(original_target_speed, raw_target_speed))

        # 3. 强制缓变输出：阻止底层 PI 控制器发癫
        if getattr(self, 'final_steady_speed', None) is None:
            self.final_steady_speed = ego_speed_kmh
            
        # 限制系统每秒钟最多只能改变 3.0 km/h 的目标车速
        max_accel_step = 3.0 * dt 
        
        if raw_target_speed > self.final_steady_speed + max_accel_step:
            self.final_steady_speed += max_accel_step
        elif raw_target_speed < self.final_steady_speed - max_accel_step:
            self.final_steady_speed -= max_accel_step
        else:
            self.final_steady_speed = raw_target_speed

        return self.final_steady_speed