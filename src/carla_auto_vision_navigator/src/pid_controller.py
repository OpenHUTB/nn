import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIDController:
    def __init__(self, kp=1.0, ki=0.1, kd=0.05):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数

        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, target_value, current_value):
        """计算PID输出"""
        current_time = time.time()
        dt = current_time - self.last_time if current_time - self.last_time > 0 else 0.001

        # 计算误差
        error = target_value - current_value

        # 比例项
        proportional = self.kp * error

        # 积分项（防积分饱和）
        self.integral += error * dt
        self.integral = max(min(self.integral, 10.0), -10.0)
        integral = self.ki * self.integral

        # 微分项
        derivative = self.kd * (error - self.last_error) / dt

        # 总输出
        output = proportional + integral + derivative

        # 更新状态
        self.last_error = error
        self.last_time = current_time

        return output

    def control_vehicle(self, vehicle, action):
        """控制车辆执行决策动作"""
        # 获取车辆当前状态
        current_speed = vehicle.get_velocity()
        current_speed_kmh = 3.6 * (current_speed.x ** 2 + current_speed.y ** 2 + current_speed.z ** 2) ** 0.5

        # 速度控制（PID）
        speed_output = self.compute(action["speed"], current_speed_kmh)

        # 构造车辆控制指令
        control = carla.VehicleControl()
        control.throttle = max(min(speed_output / 100, 1.0), 0.0) if speed_output > 0 else 0.0
        control.brake = max(min(-speed_output / 100, 1.0), 0.0) if speed_output < 0 else action["brake"]
        control.steer = action["steer"]
        control.hand_brake = True if action["speed"] == 0 else False

        # 执行控制
        vehicle.apply_control(control)
        logger.info(f"当前速度：{current_speed_kmh:.1f}km/h | 目标速度：{action['speed']}km/h | 转向：{action['steer']}")