import carla
import cv2

class TrafficLightMonitor:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.light_state = "None"
        self.color = (255,255,255)

    def update(self):
        light = self.vehicle.get_traffic_light()
        if light is None:
            self.light_state = "No Traffic Light"
            self.color = (150,150,150)
            return

        state = light.get_state()
        if state == carla.TrafficLightState.Red:
            self.light_state = "RED"
            self.color = (0,0,255)
        elif state == carla.TrafficLightState.Yellow:
            self.light_state = "YELLOW"
            self.color = (0,255,255)
        elif state == carla.TrafficLightState.Green:
            self.light_state = "GREEN"
            self.color = (0,255,0)
        else:
            self.light_state = "Off"
            self.color = (100,100,100)

    def render(self, canvas, x,y):
        # 信号灯色块
        cv2.rectangle(canvas, (x,y), (x+120, y+60), self.color, -1)
        cv2.putText(canvas, f"Light: {self.light_state}", (x+10, y+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        return canvas