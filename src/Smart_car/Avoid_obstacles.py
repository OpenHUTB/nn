import time
import random

# 模拟超声波传感器（返回障碍物距离，单位：cm）
class UltrasonicSensor:
    def get_distance(self):
        # 随机生成10-200cm的距离（模拟真实环境波动）
        return random.uniform(10, 200)

# 模拟无人车执行器（控制方向和速度）
class CarController:
    def __init__(self):
        self.speed = 0  # 0-100（速度百分比）
        self.direction = "forward"  # forward/left/right

    def set_speed(self, speed):
        self.speed = max(0, min(100, speed))  # 速度限制在0-100
        print(f"当前速度：{self.speed}%")

    def set_direction(self, direction):
        self.direction = direction
        print(f"当前方向：{self.direction}")

# 避障核心逻辑（安全距离阈值：30cm）
def obstacle_avoidance(car, sensor, safe_distance=30):
    while True:
        distance = sensor.get_distance()
        print(f"检测到障碍物距离：{distance:.1f}cm")

        if distance > safe_distance:
            # 无危险：直行，高速行驶
            car.set_direction("forward")
            car.set_speed(80)
        else:
            # 有危险：减速→随机左转/右转避障
            car.set_speed(30)
            turn_dir = random.choice(["left", "right"])
            car.set_direction(turn_dir)
            print(f"触发避障！转向{turn_dir}")
            time.sleep(1.5)  # 保持转向1.5秒
            car.set_direction("forward")  # 避障后回正

        time.sleep(0.5)  # 每0.5秒检测一次

# 主程序运行
if __name__ == "__main__":
    sensor = UltrasonicSensor()
    car = CarController()
    print("无人车避障系统启动...")
    try:
        obstacle_avoidance(car, sensor)
    except KeyboardInterrupt:
        print("\n系统停止")
        car.set_speed(0)  # 紧急停车
