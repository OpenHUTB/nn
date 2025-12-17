"""
车辆跟踪和视角控制模块 - 平滑跟随车辆，获取车辆状态
"""

import carla
import math
import time
import numpy as np
import config as cfg

class VehicleTracker:
    """车辆跟踪器 - 负责视角控制和状态获取"""
    
    def __init__(self, world):
        self.world = world
        self.spectator = world.get_spectator()
        
        # 平滑跟随参数
        self.smooth_factor = cfg.SMOOTH_FOLLOW_FACTOR
        self.min_smooth_factor = cfg.MIN_SMOOTH_FACTOR
        self.max_smooth_factor = cfg.MAX_SMOOTH_FACTOR
        self.adaptive_smoothing = cfg.SMOOTH_FACTOR_ADAPTIVE
        self.distance_threshold = cfg.DISTANCE_THRESHOLD
        
        # 插值参数
        self.use_interpolation = cfg.USE_SMOOTH_INTERPOLATION
        self.interpolation_steps = cfg.INTERPOLATION_STEPS
        
        # 状态追踪
        self.last_camera_transform = None
        self.last_vehicle_transform = None
        self.last_update_time = time.time()
        self.frame_count = 0
        
        # 预测参数
        self.prediction_enabled = True
        self.velocity_history = []
        self.max_velocity_history = 10
        
        # 视角参数
        self.target_height = cfg.TOP_DOWN_HEIGHT
        self.pitch_angle = cfg.TOP_DOWN_PITCH
        
        print(f"📐 车辆跟踪器初始化完成，平滑系数: {self.smooth_factor}")
    
    def set_top_down_view(self, vehicle, height=None):
        """设置俯视视角"""
        if vehicle is None:
            return False
        
        try:
            if height is not None:
                self.target_height = height
            
            transform = vehicle.get_transform()
            location = transform.location
            
            # 设置相机在车辆正上方
            camera_location = carla.Location(
                x=location.x,
                y=location.y,
                z=location.z + self.target_height
            )
            
            # 设置俯视角度
            camera_rotation = carla.Rotation(
                pitch=self.pitch_angle,
                yaw=transform.rotation.yaw,
                roll=0.0
            )
            
            camera_transform = carla.Transform(camera_location, camera_rotation)
            self.spectator.set_transform(camera_transform)
            self.last_camera_transform = camera_transform
            self.last_vehicle_transform = transform
            
            print(f"📐 设置俯视视角，高度: {self.target_height}m")
            return True
            
        except Exception as e:
            print(f"❌ 设置俯视视角失败: {e}")
            return False
    
    def smooth_follow_vehicle(self, vehicle, height=None):
        """平滑跟随车辆（每帧调用）- 改进版本"""
        if vehicle is None:
            return False
        
        try:
            current_time = time.time()
            time_delta = current_time - self.last_update_time
            
            # 限制最小时间间隔，避免计算过于频繁
            if time_delta < 0.001:
                return True
            
            self.last_update_time = current_time
            self.frame_count += 1
            
            # 获取车辆当前状态
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            # 获取车辆速度用于预测
            vehicle_velocity = vehicle.get_velocity()
            speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
            
            # 更新速度历史
            self.velocity_history.append(speed)
            if len(self.velocity_history) > self.max_velocity_history:
                self.velocity_history.pop(0)
            
            if height is not None:
                self.target_height = height
            
            # 计算目标相机位置（车辆正上方）
            target_location = carla.Location(
                x=vehicle_location.x,
                y=vehicle_location.y,
                z=vehicle_location.z + self.target_height
            )
            
            # 目标相机旋转（保持俯视，yaw跟随车辆）
            target_rotation = carla.Rotation(
                pitch=self.pitch_angle,
                yaw=vehicle_rotation.yaw,
                roll=0.0
            )
            
            # 自适应平滑系数
            effective_smooth_factor = self._calculate_adaptive_smooth_factor(
                vehicle_location, target_location, speed, time_delta
            )
            
            # 预测目标位置（如果启用）
            if self.prediction_enabled and len(self.velocity_history) > 1:
                avg_speed = np.mean(self.velocity_history[-3:]) if len(self.velocity_history) >= 3 else speed
                target_location = self._predict_target_position(
                    target_location, vehicle_rotation, avg_speed, time_delta
                )
            
            # 计算平滑移动
            if self.last_camera_transform:
                if self.use_interpolation:
                    # 使用多步插值
                    smooth_transform = self._multi_step_interpolation(
                        self.last_camera_transform,
                        target_location,
                        target_rotation,
                        effective_smooth_factor
                    )
                else:
                    # 使用单步平滑
                    smooth_loc = self._lerp_location(
                        self.last_camera_transform.location,
                        target_location,
                        effective_smooth_factor
                    )
                    
                    smooth_rot = self._lerp_rotation(
                        self.last_camera_transform.rotation,
                        target_rotation,
                        effective_smooth_factor
                    )
                    
                    smooth_transform = carla.Transform(smooth_loc, smooth_rot)
            else:
                # 第一次直接设置
                smooth_transform = carla.Transform(target_location, target_rotation)
            
            # 设置相机
            self.spectator.set_transform(smooth_transform)
            self.last_camera_transform = smooth_transform
            self.last_vehicle_transform = vehicle_transform
            
            # 每100帧输出一次调试信息
            if cfg.DEBUG_MODE and self.frame_count % 100 == 0:
                print(f"[视角] 帧: {self.frame_count}, "
                      f"平滑系数: {effective_smooth_factor:.3f}, "
                      f"速度: {speed:.2f}m/s, "
                      f"时差: {time_delta:.3f}s")
            
            return True
            
        except Exception as e:
            if cfg.DEBUG_MODE:
                print(f"⚠️ 视角更新失败: {e}")
            return False
    
    def _calculate_adaptive_smooth_factor(self, vehicle_loc, target_loc, speed, time_delta):
        """计算自适应平滑系数"""
        base_factor = self.smooth_factor
        
        if not self.adaptive_smoothing:
            return base_factor
        
        # 计算当前位置与目标位置的距离
        if self.last_camera_transform:
            current_loc = self.last_camera_transform.location
            distance = math.sqrt(
                (target_loc.x - current_loc.x)**2 +
                (target_loc.y - current_loc.y)**2 +
                (target_loc.z - current_loc.z)**2
            )
            
            # 根据距离调整平滑系数
            if distance > self.distance_threshold:
                # 距离较远，使用较大的平滑系数快速接近
                factor = min(self.max_smooth_factor, 
                           base_factor * (1.0 + distance / self.distance_threshold * 0.5))
            else:
                # 距离较近，使用较小的平滑系数保持平滑
                factor = max(self.min_smooth_factor, 
                           base_factor * (distance / self.distance_threshold))
            
            # 根据速度调整
            if speed > 5.0:  # 高速时减小平滑系数，反应更快
                factor = min(factor * 0.8, self.max_smooth_factor)
            
            # 根据时间间隔调整
            if time_delta > 0.05:  # 帧间隔较大时增加平滑系数
                factor = min(factor * 1.2, self.max_smooth_factor)
            
            return max(self.min_smooth_factor, min(self.max_smooth_factor, factor))
        
        return base_factor
    
    def _predict_target_position(self, target_loc, vehicle_rot, speed, time_delta):
        """预测目标位置"""
        if speed < 0.1:  # 速度太慢不预测
            return target_loc
        
        # 预测未来0.2秒的位置
        prediction_time = 0.2
        angle_rad = math.radians(vehicle_rot.yaw)
        
        predicted_x = target_loc.x + speed * math.cos(angle_rad) * prediction_time
        predicted_y = target_loc.y + speed * math.sin(angle_rad) * prediction_time
        
        return carla.Location(
            x=predicted_x,
            y=predicted_y,
            z=target_loc.z
        )
    
    def _multi_step_interpolation(self, current_transform, target_loc, target_rot, smooth_factor):
        """多步插值，实现更平滑的移动"""
        current_loc = current_transform.location
        current_rot = current_transform.rotation
        
        # 计算每一步的插值比例
        step_factor = smooth_factor / self.interpolation_steps
        
        intermediate_loc = current_loc
        intermediate_rot = current_rot
        
        for step in range(self.interpolation_steps):
            # 逐步插值
            intermediate_loc = self._lerp_location(
                intermediate_loc, target_loc, step_factor
            )
            
            intermediate_rot = self._lerp_rotation(
                intermediate_rot, target_rot, step_factor
            )
        
        return carla.Transform(intermediate_loc, intermediate_rot)
    
    def _lerp_location(self, loc1, loc2, t):
        """改进的线性插值位置（指数平滑）"""
        # 使用指数平滑：exp(-t) 而不是线性
        alpha = 1.0 - math.exp(-t * 10.0)  # 调整系数控制平滑度
        
        return carla.Location(
            x=loc1.x + (loc2.x - loc1.x) * alpha,
            y=loc1.y + (loc2.y - loc1.y) * alpha,
            z=loc1.z + (loc2.z - loc1.z) * alpha
        )
    
    def _lerp_rotation(self, rot1, rot2, t):
        """改进的线性插值旋转（处理角度环绕）"""
        def lerp_angle(a1, a2, t):
            # 使用球形线性插值(SLERP)的思路
            diff = ((a2 - a1 + 180) % 360) - 180
            
            # 使用更平滑的插值函数
            smooth_t = math.sin(t * math.pi / 2)  # 使用sin函数实现缓入效果
            
            return a1 + diff * smooth_t
        
        return carla.Rotation(
            pitch=lerp_angle(rot1.pitch, rot2.pitch, t),
            yaw=lerp_angle(rot1.yaw, rot2.yaw, t),
            roll=lerp_angle(rot1.roll, rot2.roll, t)
        )
    
    def get_vehicle_state(self, vehicle):
        """获取车辆状态信息"""
        if vehicle is None:
            return None
        
        try:
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            control = vehicle.get_control()
            
            # 计算速度
            speed_3d = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_2d = math.sqrt(velocity.x**2 + velocity.y**2)
            
            # 计算加速度（简化版本）
            acceleration = 0.0
            if self.last_vehicle_transform:
                time_diff = time.time() - self.last_update_time
                if time_diff > 0:
                    last_speed = self._calculate_speed_from_transform(
                        self.last_vehicle_transform, transform, time_diff
                    )
                    acceleration = (speed_2d - last_speed) / time_diff
            
            state = {
                'x': transform.location.x,
                'y': transform.location.y,
                'z': transform.location.z,
                'heading': transform.rotation.yaw,
                'pitch': transform.rotation.pitch,
                'roll': transform.rotation.roll,
                'speed_3d': speed_3d,
                'speed_2d': speed_2d,
                'acceleration': acceleration,
                'velocity_x': velocity.x,
                'velocity_y': velocity.y,
                'velocity_z': velocity.z,
                'throttle': control.throttle,
                'steer': control.steer,
                'brake': control.brake,
                'hand_brake': control.hand_brake,
                'reverse': control.reverse
            }
            
            return state
            
        except Exception as e:
            print(f"❌ 获取车辆状态失败: {e}")
            return None
    
    def _calculate_speed_from_transform(self, prev_transform, curr_transform, time_diff):
        """从两个变换计算速度"""
        prev_loc = prev_transform.location
        curr_loc = curr_transform.location
        
        distance = math.sqrt(
            (curr_loc.x - prev_loc.x)**2 +
            (curr_loc.y - prev_loc.y)**2
        )
        
        return distance / time_diff if time_diff > 0 else 0.0
    
    def calculate_progress(self, x, y, route_points):
        """计算行驶进度"""
        if not route_points or len(route_points) < 2:
            return "进度: N/A"
        
        # 计算到起点和终点的距离
        start_point = route_points[0]
        end_point = route_points[-1]
        
        dist_to_start = math.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)
        dist_to_end = math.sqrt((x - end_point[0])**2 + (y - end_point[1])**2)
        
        # 计算总路线长度（估算）
        total_distance = 0
        for i in range(len(route_points) - 1):
            x1, y1, _ = route_points[i]
            x2, y2, _ = route_points[i+1]
            total_distance += math.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # 计算进度百分比
        if total_distance > 0:
            traveled = max(0, total_distance - dist_to_end)
            progress = min(100, (traveled / total_distance) * 100)
        else:
            progress = 0
        
        return f"进度: {progress:.1f}% | 距起点: {dist_to_start:.1f}m | 距终点: {dist_to_end:.1f}m"
    
    def update_smooth_factor(self, factor):
        """更新平滑系数"""
        if 0 < factor <= 1:
            self.smooth_factor = factor
            print(f"🔄 平滑系数更新为: {factor}")
        else:
            print(f"❌ 无效的平滑系数: {factor}，保持为: {self.smooth_factor}")
    
    def reset(self):
        """重置跟踪器状态"""
        self.velocity_history = []
        self.frame_count = 0
        self.last_update_time = time.time()
        print("🔄 车辆跟踪器已重置")
