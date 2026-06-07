#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import carla
import logging
import random
import time

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class TrafficGeneratorNode:
    def __init__(self):
        rospy.init_node('traffic_generator_node', anonymous=False)
        self.host = rospy.get_param('~host', '192.168.133.1')
        self.port = rospy.get_param('~port', 2000)
        self.n_vehicles = rospy.get_param('~number_of_vehicles', 30)
        self.n_walkers = rospy.get_param('~number_of_walkers', 10)
        
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.client = None
        
        self.connect_carla()
        self.setup_traffic()

    def connect_carla(self):
        rospy.loginfo(f"Traffic: Connecting to {self.host}...")
        try:
            self.client = carla.Client(self.host, self.port, worker_threads=1)
            self.client.set_timeout(30.0)
            self.world = self.client.get_world()
        except Exception as e:
            rospy.logerr(f"Connection Error: {e}")
            exit(1)

    def setup_traffic(self):
        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.5)
        tm.set_synchronous_mode(True)
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # 1. 生成车辆
        # 直接使用 filter 函数，只获取车辆
        bps = self.world.get_blueprint_library().filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= self.n_vehicles: break
            bp = random.choice(bps)
            if bp.has_attribute('role_name'): bp.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(bp, transform).then(SetAutopilot(FutureActor, True, tm.get_port())))
        
        for response in self.client.apply_batch_sync(batch, True):
            if not response.error: self.vehicles_list.append(response.actor_id)
            
        rospy.loginfo(f"Spawned {len(self.vehicles_list)} vehicles")

        # 2. 生成行人
        bps_w = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        batch = []
        for i in range(self.n_walkers):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                sp = carla.Transform(loc)
                batch.append(SpawnActor(random.choice(bps_w), sp))
        
        results = self.client.apply_batch_sync(batch, True)
        for r in results:
            if not r.error: self.walkers_list.append({"id": r.actor_id})
            
        # 3. 生成控制器
        batch = [SpawnActor(self.world.get_blueprint_library().find('controller.ai.walker'), carla.Transform(), w["id"]) for w in self.walkers_list]
        results = self.client.apply_batch_sync(batch, True)
        
        for i, r in enumerate(results):
            if not r.error: 
                self.walkers_list[i]["con"] = r.actor_id
                # 【重要修改】必须把控制器(r.actor_id)放在偶数位，行人ID放在奇数位
                # 这样下面的 loop 才能正确调用 start()
                self.all_id.extend([r.actor_id, self.walkers_list[i]["id"]])
                
        self.world.tick()
        
        # 4. 启动逻辑
        all_actors = self.world.get_actors(self.all_id)
        for i in range(0, len(self.all_id), 2):
            # 这里调用 start()，所以 list 的第 0, 2, 4... 项必须是控制器(Controller)
            all_actors[i].start()
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            all_actors[i].set_max_speed(1.5)

    def run(self):
        try:
            while not rospy.is_shutdown():
                self.world.tick()
        except Exception as e:
            rospy.logerr(f"Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        rospy.loginfo("Cleaning up...")
        # 退出前必须关闭同步模式，否则服务器会死锁
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        
        if self.client:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list + self.all_id])

if __name__ == '__main__':
    TrafficGeneratorNode().run()
