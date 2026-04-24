import carla
import json

class Player:
    def __init__(self, world, vehicle, npcs):
        self.world = world
        self.ego = vehicle
        self.npc_dict = {n.id: n for n in npcs}
        self.frames = []
        self.index = 0

    def load(self, path="recording.json"):
        with open(path, "r") as f:
            self.frames = json.load(f)
        print(f"✅ 回放加载完成：{len(self.frames)} 帧")

    def play_frame(self):
        if self.index >= len(self.frames):
            return False
        data = self.frames[self.index]
        t = carla.Transform(
            carla.Location(data["ego"]["x"], data["ego"]["y"], data["ego"]["z"]),
            carla.Rotation(data["ego"]["pitch"], data["ego"]["yaw"], data["ego"]["roll"])
        )
        self.ego.set_transform(t)
        for npc_data in data["npcs"]:
            nid = npc_data["id"]
            if nid in self.npc_dict:
                npc = self.npc_dict[nid]
                t = carla.Transform(
                    carla.Location(npc_data["x"], npc_data["y"], npc_data["z"]),
                    carla.Rotation(npc_data["pitch"], npc_data["yaw"], npc_data["roll"])
                )
                npc.set_transform(t)
        self.index += 1
        return True