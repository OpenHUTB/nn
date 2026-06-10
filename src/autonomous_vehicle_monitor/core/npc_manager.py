import carla
import random

class NpcManager:
    def __init__(self, world, bp_lib, spawn_points):
        self.world = world
        self.bp_lib = bp_lib
        self.spawn_points = spawn_points
        self.vehicles = []
        self.walkers = []
        self.all_actors = []

    def spawn_all(self, num_vehicles=15, num_walkers=20):
        self.spawn_vehicles(num_vehicles)
        self.spawn_walkers(num_walkers)
        self.start_walkers()

    def spawn_vehicles(self, num):
        for _ in range(num):
            v_bp = random.choice(self.bp_lib.filter('vehicle.*'))
            spawn = random.choice(self.spawn_points)
            try:
                npc = self.world.spawn_actor(v_bp, spawn)
                npc.set_autopilot(True)
                self.vehicles.append(npc)
                self.all_actors.append(npc)
            except:
                continue

    def spawn_walkers(self, num):
        walker_bps = self.bp_lib.filter('walker.pedestrian.*')
        for _ in range(num):
            try:
                loc = self.world.get_random_location_from_navigation()
                if not loc: continue
                walker = self.world.try_spawn_actor(random.choice(walker_bps), carla.Transform(loc))
                if walker:
                    self.walkers.append(walker)
                    self.all_actors.append(walker)
            except:
                continue

    def start_walkers(self):
        controller_bp = self.bp_lib.find('controller.ai.walker')
        for w in self.walkers:
            try:
                ctrl = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=w)
                self.all_actors.append(ctrl)
                ctrl.start()
                ctrl.go_to_location(self.world.get_random_location_from_navigation())
            except:
                continue

    def destroy_all(self):
        for a in self.all_actors:
            try:
                if a.is_alive: a.destroy()
            except:
                pass