import json
import time

class Recorder:
    def __init__(self):
        self.frames = []
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        self.frames = []

    def record_frame(self, vehicle, npcs):
        data = {
            "time": time.time() - self.start_time,
            "ego": {
                "x": vehicle.get_transform().location.x,
                "y": vehicle.get_transform().location.y,
                "z": vehicle.get_transform().location.z,
                "pitch": vehicle.get_transform().rotation.pitch,
                "yaw": vehicle.get_transform().rotation.yaw,
                "roll": vehicle.get_transform().rotation.roll,
            },
            "npcs": []
        }
        for npc in npcs:
            if npc.is_alive:
                t = npc.get_transform()
                data["npcs"].append({
                    "id": npc.id,
                    "x": t.location.x, "y": t.location.y, "z": t.location.z,
                    "pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll
                })
        self.frames.append(data)

    def save(self, path="recording.json"):
        with open(path, "w") as f:
            json.dump(self.frames, f)
        print(f"✅ 场景录制完成：{path}")