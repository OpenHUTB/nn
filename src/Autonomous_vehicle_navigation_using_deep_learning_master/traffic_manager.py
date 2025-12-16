"""
交通管理器 - 封装generate_traffic.py功能，生成和管理交通流
"""

import sys
import os
import glob
import time
import logging
from numpy import random

# 添加CARLA路径
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import VehicleLightState as vls

class TrafficManager:
    """交通管理器 - 负责生成和控制交通流"""
    
    def __init__(self, client=None, host='localhost', port=2000):
        """
        初始化交通管理器
        
        Args:
            client: 可选的CARLA客户端对象
            host: CARLA服务器主机
            port: CARLA服务器端口
        """
        if client:
            self.client = client
            self.world = client.get_world()
        else:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
        
        self.tm_port = 8000
        self.traffic_manager = None
        self.vehicles_list = []
        self.walkers_list = []
        self.all_actors = []
        self.all_id = []
        
        self.is_synchronous = False
        self.synchronous_master = False
        
        print("🚦 交通管理器初始化完成")
    
    def generate_traffic(self, 
                         num_vehicles=20, 
                         num_walkers=30,
                         safe_mode=True,
                         hybrid_mode=True,
                         sync_mode=False,
                         respawn_vehicles=False):
        """
        生成交通流
        
        Args:
            num_vehicles: 车辆数量
            num_walkers: 行人数量
            safe_mode: 安全模式（避免事故倾向车辆）
            hybrid_mode: 混合物理模式
            sync_mode: 同步模式
            respawn_vehicles: 是否重生休眠车辆
            
        Returns:
            bool: 是否成功
        """
        print("\n" + "="*50)
        print("生成交通流")
        print("="*50)
        print(f"车辆: {num_vehicles}辆")
        print(f"行人: {num_walkers}个")
        print(f"安全模式: {'开启' if safe_mode else '关闭'}")
        print(f"混合模式: {'开启' if hybrid_mode else '关闭'}")
        print(f"同步模式: {'开启' if sync_mode else '关闭'}")
        
        try:
            # 获取交通管理器
            self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
            
            # 配置交通管理器
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            
            if respawn_vehicles:
                self.traffic_manager.set_respawn_dormant_vehicles(True)
            
            if hybrid_mode:
                self.traffic_manager.set_hybrid_physics_mode(True)
                self.traffic_manager.set_hybrid_physics_radius(70.0)
            
            # 设置仿真模式
            self.is_synchronous = sync_mode
            settings = self.world.get_settings()
            
            if sync_mode:
                self.traffic_manager.set_synchronous_mode(True)
                if not settings.synchronous_mode:
                    self.synchronous_master = True
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                else:
                    self.synchronous_master = False
                
                self.world.apply_settings(settings)
                print("✅ 同步模式已启用")
            
            # 生成车辆
            self._spawn_vehicles(num_vehicles, safe_mode)
            
            # 生成行人
            self._spawn_walkers(num_walkers)
            
            # 配置交通管理器参数
            self.traffic_manager.global_percentage_speed_difference(30.0)
            
            print(f"\n✅ 交通流生成完成!")
            print(f"  生成车辆: {len(self.vehicles_list)}辆")
            print(f"  生成行人: {len(self.walkers_list)}个")
            
            return True
            
        except Exception as e:
            print(f"❌ 生成交通流失败: {e}")
            return False
    
    def _spawn_vehicles(self, num_vehicles, safe_mode):
        """生成车辆"""
        print("🚗 生成车辆...")
        
        # 获取车辆蓝图
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        
        if safe_mode:
            # 过滤掉不安全或特殊车辆
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
        
        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        
        # 获取生成点
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        if num_vehicles > len(spawn_points):
            print(f"⚠️ 请求的车辆数({num_vehicles})超过生成点数({len(spawn_points)})")
            num_vehicles = len(spawn_points)
        
        # 批量生成车辆
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= num_vehicles:
                break
            
            blueprint = random.choice(blueprints)
            
            # 设置随机颜色
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            
            # 设置驾驶员ID
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            
            blueprint.set_attribute('role_name', 'autopilot')
            
            # 添加到批量命令
            batch.append(carla.command.SpawnActor(blueprint, transform)
                        .then(carla.command.SetAutopilot(
                            carla.command.FutureActor, 
                            True, 
                            self.traffic_manager.get_port())))
        
        # 执行批量命令
        for response in self.client.apply_batch_sync(batch, self.synchronous_master):
            if response.error:
                logging.error(f"生成车辆失败: {response.error}")
            else:
                self.vehicles_list.append(response.actor_id)
        
        print(f"✅ 生成 {len(self.vehicles_list)} 辆车辆")
    
    def _spawn_walkers(self, num_walkers):
        """生成行人"""
        print("🚶 生成行人...")
        
        if num_walkers <= 0:
            return
        
        # 获取行人蓝图
        walker_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        
        # 获取随机位置
        spawn_points = []
        for i in range(num_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        
        # 生成行人
        batch = []
        walker_speeds = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_bps)
            
            # 设置为非无敌
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            
            # 设置速度
            speed = 0.0
            if walker_bp.has_attribute('speed'):
                speed = walker_bp.get_attribute('speed').recommended_values[1]  # 正常行走速度
            
            walker_speeds.append(speed)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        
        # 执行批量命令
        results = self.client.apply_batch_sync(batch, True)
        
        for i in range(len(results)):
            if results[i].error:
                logging.error(f"生成行人失败: {results[i].error}")
            else:
                self.walkers_list.append({"id": results[i].actor_id})
        
        # 生成行人控制器
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        
        for i in range(len(self.walkers_list)):
            batch.append(carla.command.SpawnActor(
                walker_controller_bp, 
                carla.Transform(), 
                self.walkers_list[i]["id"]))
        
        results = self.client.apply_batch_sync(batch, True)
        
        for i in range(len(results)):
            if results[i].error:
                logging.error(f"生成行人控制器失败: {results[i].error}")
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
                self.all_id.append(results[i].actor_id)
                self.all_id.append(self.walkers_list[i]["id"])
        
        # 获取所有行人actor
        all_actors = self.world.get_actors(self.all_id)
        
        # 初始化控制器
        for i in range(0, len(self.all_id), 2):
            # 启动控制器
            all_actors[i].start()
            # 设置随机目标
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # 设置最大速度
            all_actors[i].set_max_speed(float(walker_speeds[int(i/2)]))
        
        print(f"✅ 生成 {len(self.walkers_list)} 个行人")
    
    def update(self):
        """更新交通管理器（用于同步模式）"""
        if self.is_synchronous and self.synchronous_master:
            self.world.tick()
        elif self.is_synchronous:
            self.world.wait_for_tick()
    
    def set_vehicle_lights(self, enabled=True):
        """设置车辆灯光"""
        if not self.vehicles_list:
            return
        
        try:
            all_vehicle_actors = self.world.get_actors(self.vehicles_list)
            for actor in all_vehicle_actors:
                self.traffic_manager.update_vehicle_lights(actor, enabled)
            
            print(f"✅ 车辆灯光 {'开启' if enabled else '关闭'}")
        except Exception as e:
            print(f"⚠️ 设置车辆灯光失败: {e}")
    
    def set_global_speed_limit(self, percentage=30.0):
        """设置全局速度限制百分比"""
        if self.traffic_manager:
            self.traffic_manager.global_percentage_speed_difference(percentage)
            print(f"✅ 设置全局速度限制: {percentage}%")
    
    def cleanup(self):
        """清理所有生成的交通"""
        print("\n🧹 清理交通流...")
        
        try:
            # 停止同步模式
            if self.is_synchronous and self.synchronous_master:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            
            # 销毁车辆
            if self.vehicles_list:
                print(f"销毁 {len(self.vehicles_list)} 辆车辆...")
                self.client.apply_batch([
                    carla.command.DestroyActor(x) for x in self.vehicles_list
                ])
            
            # 销毁行人
            if self.all_id:
                print(f"销毁 {len(self.walkers_list)} 个行人...")
                
                # 先停止控制器
                all_actors = self.world.get_actors(self.all_id)
                for i in range(0, len(self.all_id), 2):
                    all_actors[i].stop()
                
                # 销毁所有actor
                self.client.apply_batch([
                    carla.command.DestroyActor(x) for x in self.all_id
                ])
            
            # 清空列表
            self.vehicles_list = []
            self.walkers_list = []
            self.all_id = []
            
            print("✅ 交通流清理完成")
            
        except Exception as e:
            print(f"❌ 清理交通流失败: {e}")
    
    def get_traffic_info(self):
        """获取交通信息"""
        return {
            'num_vehicles': len(self.vehicles_list),
            'num_walkers': len(self.walkers_list),
            'is_synchronous': self.is_synchronous,
            'tm_port': self.tm_port
        }
    
    def __del__(self):
        """析构函数，确保清理"""
        if self.vehicles_list or self.walkers_list:
            self.cleanup()
