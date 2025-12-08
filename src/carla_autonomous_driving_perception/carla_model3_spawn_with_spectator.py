import carla
import pygame
import time
import random  # 新增：用于随机选生成点

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.load_world('Town05')
bp_lib = world.get_blueprint_library()

# 新增：主动生成 Tesla Model3 车辆（解决空列表问题）
model3_bp = bp_lib.find('vehicle.tesla.model3')  # 筛选 Model3 蓝图
spawn_points = world.get_map().get_spawn_points()  # 先获取生成点
vehicle = world.spawn_actor(model3_bp, random.choice(spawn_points))  # 生成 Model3

world.wait_for_tick()
# 后续筛选逻辑可保留（兼容多辆车场景）
actor_list = world.get_actors().filter('*model3*')
vehicle_list = []
for vehicle in actor_list:
    vehicle_list.append(vehicle)
vehicle = vehicle_list[0]

def set_spectator(world):
    spectator = world.get_spectator()
    transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-8, z=3)), vehicle.get_transform().rotation)
    spectator.set_transform(transform)

set_spectator(world)
# 最终获取生成点（供后续使用）
spawn_points = world.get_map().get_spawn_points()