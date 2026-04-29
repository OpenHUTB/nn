import cv2
import carla
import numpy as np

class VirtualDashboard:
    def __init__(self):
        self.width = 320
        self.height = 240

    def get_gear_text(self, v):
        if v == 0:
            return "P Parking"
        elif v > 0:
            return "D Drive"
        else:
            return "R Reverse"

    def render(self, vehicle):
        dash = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        ctrl = vehicle.get_control()
        vel = vehicle.get_velocity()
        speed = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # 基础文字
        cv2.putText(dash, "Virtual Dashboard", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(dash, f"Speed: {speed:.1f} km/h", (20,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(dash, f"Steer: {ctrl.steer:.2f}", (20,95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(dash, f"Throttle: {ctrl.throttle:.2f}", (20,125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(dash, f"Brake: {ctrl.brake:.2f}", (20,155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # 档位
        gear = self.get_gear_text(ctrl.gear)
        cv2.putText(dash, f"Gear: {gear}", (20,185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,150,0), 2)

        return dash