"""
仿真数据记录器 - 自动保存每次运行的完整数据到CSV文件
"""

import csv
import os
from datetime import datetime


class DataLogger:
    """仿真数据记录器，自动保存CSV日志"""

    def __init__(self, log_dir="logs"):
        # 创建日志目录
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 生成文件名：logs/yyyy-mm-dd_HH-MM-SS.csv
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filepath = os.path.join(log_dir, f"{timestamp}.csv")

        # 打开文件并写入表头
        self.file = open(self.filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.headers = [
            'Time_s', 'Time_str',
            'Speed_kmh', 'Throttle', 'Brake', 'Steer',
            'Detected_Count', 'Detected_Objects',
            'Nearest_Object', 'Nearest_Distance_m',
            'Warning', 'Is_Emergency',
            'Traffic_Light'
        ]
        self.writer.writerow(self.headers)

        self.frame_count = 0
        self.max_speed = 0
        self.total_warnings = 0
        self.total_emergencies = 0

        print(f"📝 Data log started: {self.filepath}")

    def record(self, elapsed, speed, throttle, brake, steer,
               signs, warning_text, is_emergency,
               traffic_light_state, nearest_obs):
        """记录一帧数据"""
        self.frame_count += 1

        # 统计
        if speed > self.max_speed:
            self.max_speed = round(speed, 1)
        if is_emergency:
            self.total_emergencies += 1
        if warning_text:
            self.total_warnings += 1

        # 时间
        m = int(elapsed // 60)
        s = int(elapsed % 60)
        time_str = f"{m:02d}:{s:02d}"

        # 检测物体摘要
        detected_str = "; ".join([f"{label}({conf:.0%})" for label, conf, _ in signs]) if signs else ""
        nearest_str = ""
        nearest_dist = ""
        if nearest_obs:
            label, conf, bbox, dist = nearest_obs
            nearest_str = label
            nearest_dist = f"{dist:.1f}"

        # 交通灯
        tl_str = ""
        if traffic_light_state is not None:
            tl_str = str(traffic_light_state).split('.')[-1]

        # 写入一行
        self.writer.writerow([
            f"{elapsed:.1f}", time_str,
            f"{speed:.1f}", f"{throttle:.2f}", f"{brake:.2f}", f"{steer:.3f}",
            len(signs), detected_str,
            nearest_str, nearest_dist,
            warning_text or "", is_emergency,
            tl_str
        ])

    def close_and_report(self):
        """关闭文件并打印统计报告"""
        self.file.close()

        # 统计报告
        duration = self.frame_count / 30  # 假设30fps
        print("\n" + "=" * 50)
        print("📊  DATA LOGGING REPORT")
        print("=" * 50)
        print(f"  File:      {self.filepath}")
        print(f"  Frames:    {self.frame_count}")
        print(f"  Duration:  {duration:.0f}s ({duration/60:.1f}min)")
        print(f"  Max Speed: {self.max_speed:.1f} km/h")
        print(f"  Warnings:  {self.total_warnings}")
        print(f"  Emergencies: {self.total_emergencies}")
        print("=" * 50)
