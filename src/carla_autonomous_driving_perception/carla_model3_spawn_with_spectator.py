import carla
import pygame
import time
import random
from threading import Lock

# 自定义线性插值函数（适配同步帧）
def lerp(a, b, t):
    """线性插值：t值根据同步帧率调整（30帧下0.15更稳定）"""
    return a + t * (b - a)

# 1. 连接CARLA服务器并配置强同步模式
client = carla.Client('localhost', 2000)
client.set_timeout(15.0)
world = client.load_world('Town05')

# 启用严格同步模式（关键：固定帧间隔，禁用异步更新）
settings = world.get_settings()
settings.synchronous_mode = True  # 客户端控制帧推进
settings.fixed_delta_seconds = 1/30  # 30帧/秒（与后续tick频率一致）
settings.no_rendering_mode = False  # 启用渲染
world.apply_settings(settings)

# 2. 初始化同步锁与帧数据缓存（确保线程安全）
frame_lock = Lock()
latest_snapshot = None  # 存储当前帧的Actor快照（含车辆状态）

# 绑定帧同步回调：每帧更新车辆状态快照
def on_world_tick(snapshot):
    global latest_snapshot
    with frame_lock:
        latest_snapshot = snapshot  # 缓存当前帧的所有Actor状态
world.on_tick(on_world_tick)

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# 3. 生成主角车辆（Tesla Model3）
model3_bp = bp_lib.find('vehicle.tesla.model3')
# 确保生成点有效（避免初始位置异常导致抖动）
vehicle = None
for _ in range(5):
    try:
        vehicle = world.spawn_actor(model3_bp, random.choice(spawn_points))
        print(f"主角车辆生成成功（ID: {vehicle.id}）")
        break
    except:
        time.sleep(0.5)
if not vehicle:
    raise Exception("主角车辆生成失败，请重启CARLA服务器")

# 4. 生成NPC车辆（减少至100辆，确保同步性能）
npc_count = 100  # 500辆会导致同步延迟，100辆是性能与效果的平衡
print(f"开始生成{npc_count}辆NPC车辆...")
for i in range(npc_count):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    if 'tesla' in vehicle_bp.id:  # 避免与主角车混淆
        continue
    # 尝试生成（避开主角车位置）
    spawn_point = random.choice(spawn_points)
    if spawn_point.location.distance(vehicle.get_location()) < 20:
        continue
    world.try_spawn_actor(vehicle_bp, spawn_point)
    # 每生成20辆同步一次，确保服务器不卡顿
    if i % 20 == 0:
        world.tick()
        time.sleep(0.1)

# 统计实际生成数量
all_vehicles = world.get_actors().filter('*vehicle*')
actual_npc_count = len(all_vehicles) - 1
print(f"NPC生成完成 | 实际数量: {actual_npc_count}辆（总车辆: {len(all_vehicles)}）")

# 5. 启动所有车辆自动驾驶（绑定交通管理器同步端口）
tm = client.get_trafficmanager(8000)
tm.set_synchronous_mode(True)  # 交通管理器也启用同步模式
for v in all_vehicles:
    v.set_autopilot(True, tm.get_port())  # 所有车辆通过TM控制，确保行为同步

# 6. 平滑视角函数（基于当前帧快照数据）
def set_spectator_smooth(last_transform=None):
    """
    基于当前帧快照更新视角，彻底避免异步抖动
    数据来源：on_world_tick缓存的latest_snapshot（当前帧精确状态）
    """
    spectator = world.get_spectator()
    with frame_lock:
        if not latest_snapshot:
            return last_transform  # 等待第一帧数据
        # 从当前帧快照中获取主角车的精确状态（而非实时查询）
        vehicle_snapshot = latest_snapshot.find(vehicle.id)
        if not vehicle_snapshot:
            return last_transform
        vehicle_tf = vehicle_snapshot.get_transform()  # 这是当前帧的精确位置
    
    # 目标视角：车后8米、上方3米，轻微右偏
    target_tf = carla.Transform(
        vehicle_tf.transform(carla.Location(x=-8, z=3, y=0.5)),
        vehicle_tf.rotation
    )
    
    # 首次调用直接设置
    if last_transform is None:
        spectator.set_transform(target_tf)
        return target_tf
    
    # 插值平滑（t=0.15适配30帧，同步模式下更稳定）
    smooth_loc = carla.Location(
        x=lerp(last_transform.location.x, target_tf.location.x, 0.15),
        y=lerp(last_transform.location.y, target_tf.location.y, 0.15),
        z=lerp(last_transform.location.z, target_tf.location.z, 0.15)
    )
    smooth_rot = carla.Rotation(
        pitch=lerp(last_transform.rotation.pitch, target_tf.rotation.pitch, 0.15),
        yaw=lerp(last_transform.rotation.yaw, target_tf.rotation.yaw, 0.15),
        roll=lerp(last_transform.rotation.roll, target_tf.rotation.roll, 0.15)
    )
    smooth_tf = carla.Transform(smooth_loc, smooth_rot)
    spectator.set_transform(smooth_tf)
    return smooth_tf

# 7. 主循环（严格按帧推进）
print("\n程序运行中（强同步模式），按Ctrl+C退出...")
print("镜头基于当前帧数据更新，已解决异步抖动问题")
last_spectator_tf = None
clock = pygame.time.Clock()

try:
    # 先推进一帧获取初始快照
    world.tick()
    last_spectator_tf = set_spectator_smooth()
    
    while True:
        # 推进一帧（触发on_world_tick更新快照）
        world.tick()
        # 基于当前帧快照更新视角（确保数据时序一致）
        last_spectator_tf = set_spectator_smooth(last_spectator_tf)
        # 严格控制客户端帧率（与服务器帧间隔一致）
        clock.tick(30)

except KeyboardInterrupt:
    print("\n用户中断，清理资源...")
finally:
    # 恢复CARLA默认设置（关键：避免影响后续使用）
    settings.synchronous_mode = False
    tm.set_synchronous_mode(False)
    world.apply_settings(settings)
    # 销毁所有车辆
    for v in all_vehicles:
        if v.is_alive:
            v.destroy()
    print("资源清理完成，同步模式已关闭")