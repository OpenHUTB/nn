
"""
交通管理器-生成行人和车辆
"""

import sys
import os
import glob

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from numpy import random

class TrafficManager:
    """交通管理器"""
    
    def __init__(self, client, world):
        self.client = client
        self.world = world
        self.tm_port = 8000
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        
    def generate_traffic(self, num_vehicles=15, num_walkers=20, safe_mode=True):
        """生成交通流"""
        print(f"生成交通: {num_vehicles}辆车, {num_walkers}个行人")
        
        # 获取交通管理器
        traffic_manager = self.client.get_trafficmanager(self.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        
        # 生成车辆
        self._spawn_vehicles(traffic_manager, num_vehicles, safe_mode)
        
        # 生成行人
        if num_walkers > 0:
            self._spawn_walkers_simple(num_walkers)
        
        print(f"完成: {len(self.vehicles_list)}辆车, {len(self.walkers_list)}个行人")
        return True
    
    def _spawn_vehicles(self, traffic_manager, num_vehicles, safe_mode):
        """生成车辆"""
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        
        if safe_mode:
            # 过滤危险车辆
            safe_vehicles = [
                'vehicle.audi.a2',
                'vehicle.audi.tt',
                'vehicle.bmw.grandtourer',
                'vehicle.dodge.charger_police',
                'vehicle.jeep.wrangler_rubicon',
                'vehicle.lincoln.mkz2017',
                'vehicle.mercedes-benz.coupe',
                'vehicle.mini.cooperst',
                'vehicle.nissan.micra',
                'vehicle.nissan.patrol',
                'vehicle.seat.leon',
                'vehicle.tesla.model3',
                'vehicle.toyota.prius',
                'vehicle.volkswagen.t2'
            ]
            blueprints = [bp for bp in blueprints if any(safe in bp.id for safe in safe_vehicles)]
        
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        batch = []
        for i in range(min(num_vehicles, len(spawn_points))):
            blueprint = random.choice(blueprints)
            batch.append(
                carla.command.SpawnActor(blueprint, spawn_points[i])
                .then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port()))
            )
        
        # 批量生成
        for response in self.client.apply_batch_sync(batch):
            if not response.error:
                self.vehicles_list.append(response.actor_id)
    
    def _spawn_walkers_simple(self, num_walkers):
        """简化的行人生成"""
        walker_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        
        # 生成行人
        batch = []
        walker_speeds = []
        
        for i in range(num_walkers):
            spawn_location = self.world.get_random_location_from_navigation()
            if spawn_location and spawn_location.z < 5:  # 确保位置合理
                walker_bp = random.choice(walker_bps)
                
                # 设置为非无敌
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                
                # 设置速度
                speed = 1.0  # 默认速度
                if walker_bp.has_attribute('speed'):
                    speed = walker_bp.get_attribute('speed').recommended_values[1]
                
                walker_speeds.append(speed)
                batch.append(carla.command.SpawnActor(walker_bp, carla.Transform(spawn_location)))
        
        # 生成行人实体
        walker_ids = []
        results = self.client.apply_batch_sync(batch)
        
        for i, response in enumerate(results):
            if not response.error:
                walker_id = response.actor_id
                walker_ids.append(walker_id)
                self.walkers_list.append({
                    "id": walker_id,
                    "speed": walker_speeds[i] if i < len(walker_speeds) else 1.0
                })
        
        # 生成行人控制器
        batch = []
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        
        for walker_info in self.walkers_list:
            batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), walker_info["id"]))
        
        results = self.client.apply_batch_sync(batch)
        
        for i, response in enumerate(results):
            if not response.error and i < len(self.walkers_list):
                self.walkers_list[i]["con"] = response.actor_id
                self.all_id.append(response.actor_id)
                self.all_id.append(self.walkers_list[i]["id"])
        
        # 初始化控制器
        all_actors = self.world.get_actors(self.all_id)
        
        for i in range(0, len(self.all_id), 2):
            if i < len(all_actors):
                controller = all_actors[i]
                walker = all_actors[i+1] if i+1 < len(all_actors) else None
                
                if controller and walker:
                    try:
                        controller.start()
                        
                        # 设置目的地
                        target = self.world.get_random_location_from_navigation()
                        if target:
                            controller.go_to_location(target)
                        
                        # 设置速度
                        walker_index = i // 2
                        if walker_index < len(self.walkers_list):
                            controller.set_max_speed(self.walkers_list[walker_index]["speed"])
                    except:
                        pass
    
    def cleanup(self):
        """清理交通"""
        print("清理交通...")
        
        # 销毁车辆
        if self.vehicles_list:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        
        # 销毁行人
        if self.all_id:
            # 先停止控制器
            all_actors = self.world.get_actors(self.all_id)
            for i in range(0, len(self.all_id), 2):
                if i < len(all_actors):
                    all_actors[i].stop()
            
            # 销毁所有行人相关actor
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])