import carla
import pygame
import time
import random

# 自定义线性插值函数（兼容Python 3.7+，解决视角抖动核心）
def lerp(a, b, t):
    """线性插值：从a到b平滑过渡，t∈[0,1]（0=取a，1=取b，越小越平滑）"""
    return a + t * (b - a)

# 1. 连接CARLA服务器并启用同步模式（稳定数据获取，减少抖动）
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)  # 延长超时时间，适配多NPC生成
world = client.load_world('Town05')
# 启用同步模式，保证帧率稳定、数据无滞后
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1/30  # 固定30帧/秒
world.apply_settings(settings)

bp_lib = world.get_blueprint_library()

# 2. 生成主角车辆（Tesla Model3）
model3_bp = bp_lib.find('vehicle.tesla.model3')
spawn_points = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(model3_bp, random.choice(spawn_points))
world.tick()  # 同步帧，获取最新车辆数据

# 3. 生成500辆NPC车辆（分批生成，避免CARLA卡顿/崩溃）
npc_count = 500  # 新增：NPC数量从200增至500
print(f"开始生成{npc_count}辆NPC交通车辆...")
for i in range(npc_count):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    # 每生成100辆同步一次，减轻CARLA压力
    if i % 100 == 0:
        world.tick()
        time.sleep(0.05)
    else:
        time.sleep(0.01)
# 统计实际生成的车辆数（生成点冲突可能略少）
actual_npc_count = len(world.get_actors().filter('*vehicle*')) - 1  # 减主角车
print(f"NPC车辆生成完成，实际生成：{actual_npc_count}辆（含主角车共{len(world.get_actors().filter('*vehicle*'))}辆）")

# 4. 启动所有NPC+主角车自动驾驶
print("启动所有NPC车辆+主角Model3自动驾驶...")
for v in world.get_actors().filter('*vehicle*'):
    v.set_autopilot(True)
print("主角Model3车辆已启用自动驾驶，将与NPC同步行驶")

# 5. 筛选主角Model3车辆
actor_list = world.get_actors().filter('*model3*')
vehicle = actor_list[0] if actor_list else None
if not vehicle:
    raise Exception("主角Model3车辆生成失败！")

# 6. 平滑视角函数（核心解决抖动：插值过渡+实时跟随）
def set_spectator_smooth(world, vehicle, last_transform=None):
    """
    平滑更新主角车后上方视角，避免抖动
    :param last_transform: 上一帧视角，用于插值过渡
    :return: 当前帧视角（供下一帧插值）
    """
    spectator = world.get_spectator()
    # 目标视角：主角车后方8米、上方3米，轻微偏移避免遮挡
    vehicle_tf = vehicle.get_transform()
    target_tf = carla.Transform(
        vehicle_tf.transform(carla.Location(x=-8, z=3, y=0.5)),
        vehicle_tf.rotation
    )
    # 首次调用直接设置视角
    if last_transform is None:
        spectator.set_transform(target_tf)
        return target_tf
    # 插值平滑过渡（t=0.1，越小视角越稳，0.05~0.2为宜）
    smooth_loc = carla.Location(
        x=lerp(last_transform.location.x, target_tf.location.x, 0.1),
        y=lerp(last_transform.location.y, target_tf.location.y, 0.1),
        z=lerp(last_transform.location.z, target_tf.location.z, 0.1)
    )
    smooth_rot = carla.Rotation(
        pitch=lerp(last_transform.rotation.pitch, target_tf.rotation.pitch, 0.1),
        yaw=lerp(last_transform.rotation.yaw, target_tf.rotation.yaw, 0.1),
        roll=lerp(last_transform.rotation.roll, target_tf.rotation.roll, 0.1)
    )
    smooth_tf = carla.Transform(smooth_loc, smooth_rot)
    spectator.set_transform(smooth_tf)
    return smooth_tf

# 初始化视角
last_spectator_tf = set_spectator_smooth(world, vehicle)
print("视角已切换至主角Model3车辆后上方（平滑跟随，无抖动）")

# 7. 主循环：稳定帧率+平滑视角
print("\n程序运行中，主角车与500辆NPC同步行驶，按Ctrl+C退出...")
clock = pygame.time.Clock()  # 精准控制帧率
try:
    while True:
        world.tick()  # 同步CARLA帧，数据无滞后
        # 平滑更新视角
        last_spectator_tf = set_spectator_smooth(world, vehicle, last_spectator_tf)
        clock.tick(30)  # 严格30帧/秒，避免帧率波动导致抖动
except KeyboardInterrupt:
    print("\n程序退出，清理资源...")
    # 恢复CARLA默认设置（避免影响后续使用）
    settings.synchronous_mode = False
    world.apply_settings(settings)
    # 销毁所有生成的车辆
    for v in world.get_actors().filter('*vehicle*'):
        if v.is_alive:
            v.destroy()
    print("资源清理完成，CARLA设置已恢复！")