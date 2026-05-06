import csv
import time
import carla

class BlackBox:
    def __init__(self, filename="blackbox.csv"):
        self.filename = filename
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "time", "speed", "steer", "throttle", "brake",
            "accel_x", "accel_y", "accel_z"
        ])
        self.start_time = time.time()

    def record(self, vehicle):
        t = time.time() - self.start_time
        v = vehicle.get_velocity()
        speed = (3.6 * (v.x**2 + v.y**2 + v.z**2)**0.5)  # km/h
        control = vehicle.get_control()
        accel = vehicle.get_acceleration()

        self.writer.writerow([
            round(t, 2),
            round(speed, 2),
            round(control.steer, 3),
            round(control.throttle, 3),
            round(control.brake, 3),
            round(accel.x, 2),
            round(accel.y, 2),
            round(accel.z, 2)
        ])

    def close(self):
        self.file.close()
        print(f"✅ 黑匣子数据已保存：{self.filename}")