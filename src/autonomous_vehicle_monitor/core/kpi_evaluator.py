import cv2
import numpy as np
import time

class KPIEvaluator:
    def __init__(self):
        # 基础计时
        self.start_time = time.time()

        # 安全指标
        self.collision_count = 0
        self.lane_departure_count = 0
        self.total_frames = 0
        self.lane_keep_frames = 0

        # 驾驶平顺性
        self.hard_accel_count = 0
        self.hard_decel_count = 0
        self.prev_speed = 0.0

        # 速度与效率
        self.speed_history = []

        # 碰撞预警 TTC
        self.min_ttc = 999.0

    def update(self, vehicle, front_distance, is_in_lane):
        """ 每帧调用一次，更新所有KPI """
        self.total_frames += 1

        # 车道保持
        if is_in_lane:
            self.lane_keep_frames += 1

        # 获取当前速度 km/h
        vel = vehicle.get_velocity()
        speed = np.sqrt(vel.x ** 2 + vel.y ** 2) * 3.6
        self.speed_history.append(speed)

        # 急加速 / 急减速
        delta = speed - self.prev_speed
        if delta > 3.0:
            self.hard_accel_count += 1
        if delta < -3.0:
            self.hard_decel_count += 1
        self.prev_speed = speed

        # 更新最小 TTC
        if speed > 2 and front_distance < 50:
            ttc = front_distance / (speed / 3.6 + 0.01)
            if ttc < self.min_ttc:
                self.min_ttc = ttc

    def add_collision(self):
        """ 碰撞时调用 """
        self.collision_count += 1

    def get_stats(self):
        """ 返回所有指标字典 """
        duration = time.time() - self.start_time
        avg_speed = np.mean(self.speed_history) if self.speed_history else 0
        efficiency = (avg_speed / 50.0) * 100 if avg_speed > 0 else 0
        lane_keep_rate = (self.lane_keep_frames / self.total_frames) * 100 if self.total_frames else 0

        return {
            "time": round(duration, 2),
            "collision": self.collision_count,
            "lane_rate": round(lane_keep_rate, 2),
            "min_ttc": round(self.min_ttc, 2),
            "hard_accel": self.hard_accel_count,
            "hard_decel": self.hard_decel_count,
            "avg_speed": round(avg_speed, 2),
            "efficiency": round(efficiency, 2),
        }

    def save_report(self, path="kpi_report.txt"):
        """ 保存报告到文件 """
        s = self.get_stats()
        with open(path, "w", encoding="utf-8") as f:
            f.write("===== CARLA 自动驾驶 KPI 安全评估报告 =====\n")
            f.write(f"运行时长：{s['time']} s\n")
            f.write(f"碰撞次数：{s['collision']}\n")
            f.write(f"车道保持率：{s['lane_rate']} %\n")
            f.write(f"最小TTC：{s['min_ttc']} s\n")
            f.write(f"急加速次数：{s['hard_accel']}\n")
            f.write(f"急减速次数：{s['hard_decel']}\n")
            f.write(f"平均速度：{s['avg_speed']} km/h\n")
            f.write(f"通行效率：{s['efficiency']} %\n")

    def draw_on_screen(self, img, x=1300, y=60):
        """ 在画面右上角绘制KPI """
        s = self.get_stats()
        lines = [
            "KPI Safety Report",
            f"Collision: {s['collision']}",
            f"Lane Keep: {s['lane_rate']}%",
            f"TTC: {s['min_ttc']}s",
            f"Hard Accel: {s['hard_accel']}",
            f"Hard Decel: {s['hard_decel']}",
            f"Avg Speed: {s['avg_speed']} km/h",
            f"Efficiency: {s['efficiency']}%"
        ]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (x, y + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)