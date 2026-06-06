import random
import carla
import time

class HazardManager:
    def __init__(self, world, vehicle, npc_manager, kpi):
        self.world = world
        self.ego = vehicle
        self.npc_manager = npc_manager
        self.kpi = kpi
        self.blueprint_lib = world.get_blueprint_library()
        self.last_trigger = 0
        self.cooldown = 8.0
        self.active_hazard = None

    def tick(self):
        now = time.time()
        if now - self.last_trigger < self.cooldown:
            return
        if random.random() < 0.006:
            self.trigger_random_hazard()

    def trigger_random_hazard(self):
        self.last_trigger = time.time()
        typ = random.choice([
            "sudden_brake",
            "pedestrian_jump",
            "car_cut_in",
            "obstacle"
        ])
        print(f"\n⚠️ 触发危险场景: {typ}")

        if typ == "sudden_brake":
            self.trigger_sudden_brake()
        elif typ == "pedestrian_jump":
            self.trigger_ghost_pedestrian()
        elif typ == "car_cut_in":
            self.trigger_cut_in()
        elif typ == "obstacle":
            self.trigger_static_obstacle()

    def trigger_sudden_brake(self):
        for v in self.npc_manager.vehicles:
            if not v.is_alive:
                continue
            d = v.get_location().distance(self.ego.get_location())
            if 8 < d < 22:
                v.apply_control(carla.VehicleControl(throttle=0, brake=1.0))
                self.active_hazard = "sudden_brake"
                return

    def trigger_ghost_pedestrian(self):
        bp = self.blueprint_lib.find('walker.pedestrian.0001')
        loc = self.ego.get_location()
        offset = carla.Location(x=15, y=random.choice([-1.5, 1.5]), z=1)
        trans = carla.Transform(loc + offset)
        try:
            ped = self.world.spawn_actor(bp, trans)
            ctrl = self.world.spawn_actor(self.blueprint_lib.find('controller.ai.walker'), carla.Transform(), attach_to=ped)
            ctrl.start()
            ctrl.go_to_location(loc + carla.Location(x=-5, y=0))
            self.active_hazard = "pedestrian_jump"
        except:
            pass

    def trigger_cut_in(self):
        bp = random.choice(self.blueprint_lib.filter('vehicle.*'))
        loc = self.ego.get_location()
        trans = carla.Transform(
            loc + carla.Location(x=12, y=3.2, z=0.5),
            self.ego.get_transform().rotation
        )
        try:
            car = self.world.spawn_actor(bp, trans)
            car.set_autopilot(True)
            self.npc_manager.vehicles.append(car)
            self.active_hazard = "car_cut_in"
        except:
            pass

    def trigger_static_obstacle(self):
        bp = self.blueprint_lib.find('static.prop.container')
        loc = self.ego.get_location()
        trans = carla.Transform(loc + carla.Location(x=18, y=0, z=0.2))
        try:
            self.world.spawn_actor(bp, trans)
            self.active_hazard = "obstacle"
        except:
            pass

    def get_active_hazard(self):
        return self.active_hazard

    def clear(self):
        self.active_hazard = None