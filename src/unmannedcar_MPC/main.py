#!/usr/bin/env python3

import carla
import config as Config
import math
from drawer import PyGameDrawer
from sync_pygame import SyncPyGame
from mpc import MPC


class Main():

    def __init__(self):
        # setup world
        self.client = carla.Client(Config.CARLA_SERVER, 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(Config.WORLD_NAME)
        self.map = self.world.get_map()

        # spawn ego
        ego_spawn_point = self.map.get_spawn_points()[100]
        bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        self.ego = self.world.spawn_actor(bp, ego_spawn_point)

        # init game and drawer
        self.game = SyncPyGame(self)
        self.drawer = PyGameDrawer(self)
        self.mpc = MPC(self.drawer, self.ego)

        # start game loop
        self.game.game_loop(self.world, self.on_tick)

    def on_tick(self):
        # generate reference path (global frame)
        lookahead = 5
        wp = self.map.get_waypoint(self.ego.get_location())
        path = []

        for _ in range(lookahead):
            _wps = wp.next(1)
            if len(_wps) == 0:
                break
            wp = _wps[0]
            path.append(wp.transform.location)

        # get forward speed
        velocity = self.ego.get_velocity()
        speed_m_s = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        dt = 1 / Config.PYGAME_FPS

        # generate control signal
        control = carla.VehicleControl()
        control.throttle = 0.6
        control.steer = self.mpc.run_step(path, speed_m_s, dt)

        # apply control signal
        self.ego.apply_control(control)

if __name__ == '__main__':
    Main()