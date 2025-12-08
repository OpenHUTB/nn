import carla
import pygame
import time
import random  # 用于随机选生成点/随机选NPC车辆蓝图

# 1. 连接CARLA服务器并加载地图
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.load_world('Town05')
bp_lib = world.get_blueprint_library()

# 2. 生成主角车辆（Tesla Model3）
model3_bp = bp_lib.find('vehicle.tesla.model3')  # 筛选Model3蓝图
spawn_points = world.get_map().get_spawn_points()  # 获取地图合法生成点
vehicle = world.spawn_actor(model3_bp, random.choice(spawn_points))  # 生成Model3
world.wait_for_tick()

# 3. 生成200辆NPC车辆（交通流）
print("开始生成NPC交通车辆...")
for i in range(200):
    # 随机选择车辆蓝图（过滤所有vehicle类型）
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    # 尝试生成NPC车辆（try_spawn_actor避免生成点冲突导致报错）
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    # 避免循环过快导致CARLA处理不及时
    time.sleep(0.01)
print(f"NPC车辆生成完成，当前地图车辆总数：{len(world.get_actors().filter('*vehicle*'))}")

# 4. 启动所有NPC车辆的自动驾驶（交通流动）
print("启动所有NPC车辆自动驾驶...")
for v in world.get_actors().filter('*vehicle*'):
    v.set_autopilot(True)

# 5. 关闭主角Model3车辆的自动驾驶（确保手动控制/后续自定义逻辑）
vehicle.set_autopilot(False)
print("已关闭主角Model3车辆的自动驾驶")

# 6. 筛选所有Model3车辆（兼容多辆车场景）
actor_list = world.get_actors().filter('*model3*')
vehicle_list = []
for vehicle in actor_list:
    vehicle_list.append(vehicle)
vehicle = vehicle_list[0]

# 7. 设置观众视角（后上方追尾视角）
def set_spectator(world):
    spectator = world.get_spectator()
    # 视角位置：主角车后方8米、上方3米，跟随车辆朝向
    transform = carla.Transform(
        vehicle.get_transform().transform(carla.Location(x=-8, z=3)),
        vehicle.get_transform().rotation
    )
    spectator.set_transform(transform)

set_spectator(world)
print("视角已切换至主角Model3车辆后上方追尾视角")

# 8. 最终获取生成点（供后续扩展使用）
spawn_points = world.get_map().get_spawn_points()
print(f"地图Town05合法生成点总数：{len(spawn_points)}")

# 保持运行，便于查看效果
print("\n程序运行中，按Ctrl+C退出...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n程序退出，清理资源...")
    # 可选：销毁生成的车辆（避免CARLA残留 Actors）
    for v in world.get_actors().filter('*vehicle*'):
        if v.is_alive:
            v.destroy()
    print("资源清理完成")