import carla
from datetime import datetime
from utils import calculate_vehicle_speed_kmh
import vehicle_status_gui as vsg  # 新增导入GUI模块

# 全局变量：仅保留碰撞状态标记
collision_occurred = False

def create_collision_sensor(world: carla.World, vehicle: carla.Vehicle):
    """
    创建碰撞传感器并绑定到车辆
    :param world: CARLA World对象
    :param vehicle: 被监测的车辆Actor
    :return: 碰撞传感器Actor
    """
    collision_bp = world.get_blueprint_library().find("sensor.other.collision")
    collision_sensor = world.spawn_actor(
        collision_bp,
        carla.Transform(),
        attach_to=vehicle
    )
    # 绑定碰撞回调函数
    collision_sensor.listen(lambda event: on_collision(event, vehicle))
    print("🔍 碰撞传感器已启动")
    return collision_sensor

def on_collision(event, vehicle: carla.Vehicle):
    """
    碰撞事件回调处理：记录碰撞信息、直接更新GUI碰撞车速
    :param event: 碰撞事件对象
    :param vehicle: 发生碰撞的车辆Actor
    """
    global collision_occurred
    collision_occurred = True

    # 1. 计算碰撞时车速并直接更新GUI的碰撞车速
    collision_speed = calculate_vehicle_speed_kmh(vehicle)
    vsg.update_vehicle_status("collision_speed", collision_speed)  # 直接更新GUI

    # 2. 记录碰撞详细日志到文件
    collision_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    collision_actor_type = event.other_actor.type_id  # 碰撞对象类型
    collision_loc = vehicle.get_transform().location  # 碰撞位置
    loc_str = f"({collision_loc.x:.2f}, {collision_loc.y:.2f}, {collision_loc.z:.2f})"

    # 写入碰撞日志文件
    with open("collision_logs.txt", "a", encoding="utf-8") as f:
        log_line = (
            f"[{collision_time}] 发生碰撞 | "
            f"碰撞对象：{collision_actor_type} | "
            f"位置：{loc_str} | "
            f"碰撞车速：{collision_speed} km/h\n"
        )
        f.write(log_line)

    # 控制台输出碰撞信息
    print(f"\n🚨 碰撞检测：车速 {collision_speed} km/h | 碰撞对象：{collision_actor_type}")

def get_collision_occurred() -> bool:
    """读取当前碰撞状态"""
    global collision_occurred
    return collision_occurred

def reset_collision_occurred():
    """重置碰撞状态"""
    global collision_occurred
    collision_occurred = False

def stop_collision_monitor():
    """停止碰撞监测，重置全局状态"""
    global collision_occurred
    collision_occurred = False
    print("\n🛑 碰撞监测已停止")