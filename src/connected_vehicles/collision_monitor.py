import carla
import time
import threading
from datetime import datetime

# 全局缓存：存储碰撞事件
collision_cache = []
# 锁：保证多线程操作缓存的安全性
cache_lock = threading.Lock()
# 控制线程退出的标志
stop_flag = False
# 防抖缓存（记录最近碰撞的关键信息，避免重复）
debounce_cache = {}
# 防抖时间（同一位置+同一对象，500ms内只记录一次）
DEBOUNCE_TIME = 0.5


def create_collision_sensor(world, vehicle):
    """
    为指定车辆创建碰撞传感器，监听碰撞事件并缓存（5秒批量写入文件）
    :param world: CARLA的world对象
    :param vehicle: 需要挂载传感器的车辆Actor
    :return: 碰撞传感器Actor
    """
    # 获取碰撞传感器蓝图
    bp_lib = world.get_blueprint_library()
    collision_bp = bp_lib.find('sensor.other.collision')

    # 在车辆中心位置生成碰撞传感器（附着到车辆）
    collision_sensor = world.spawn_actor(
        collision_bp,
        carla.Transform(),  # 传感器位置（车辆中心）
        attach_to=vehicle  # 绑定到目标车辆
    )

    # 绑定碰撞事件回调函数
    collision_sensor.listen(_on_collision)

    # 启动定时写入文件的线程（5秒执行一次）
    write_thread = threading.Thread(target=_write_collision_to_file, daemon=True)
    write_thread.start()

    print("🔍 碰撞监测传感器已启动（碰撞数据将每5秒写入文件）")
    return collision_sensor


def _on_collision(event):
    """碰撞事件回调函数：缓存碰撞数据（增加防抖去重）"""
    try:
        # 收集碰撞核心信息
        collision_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 精确到毫秒
        current_timestamp = time.time()  # 用于防抖的时间戳
        vehicle_id = event.actor.id
        vehicle_type = event.actor.type_id

        # 碰撞对象信息（修复ID为0的问题）
        other_actor = event.other_actor
        if other_actor and other_actor.id != 0:
            other_actor_id = other_actor.id
            other_actor_type = other_actor.type_id
        else:
            # 补充识别：从碰撞事件的物理信息中提取对象类型
            other_actor_id = "未知(ID:0)"
            other_actor_type = event.normal_impulse.__str__().split()[-1] if hasattr(event, 'normal_impulse') else "环境/未知物体"

        # 碰撞位置（简化：保留1位小数，减少精度导致的重复）
        location = event.actor.get_transform().location
        location_str = f"X:{location.x:.1f}, Y:{location.y:.1f}, Z:{location.z:.1f}"

        # 防抖关键Key：车辆ID + 碰撞对象类型 + 简化后的位置
        debounce_key = f"{vehicle_id}_{other_actor_type}_{location_str}"

        # 防抖逻辑：500ms内同一Key不重复记录
        with cache_lock:
            last_record_time = debounce_cache.get(debounce_key, 0)
            if current_timestamp - last_record_time < DEBOUNCE_TIME:
                return  # 跳过重复记录
            # 更新防抖缓存的时间戳
            debounce_cache[debounce_key] = current_timestamp

        # 构造碰撞数据字典
        collision_data = {
            "time": collision_time,
            "vehicle_id": vehicle_id,
            "vehicle_type": vehicle_type,
            "other_actor_id": other_actor_id,
            "other_actor_type": other_actor_type,
            "location": location_str
        }

        # 加锁写入缓存
        with cache_lock:
            collision_cache.append(collision_data)

    except Exception as e:
        # 异常不影响主程序，仅缓存错误信息
        with cache_lock:
            collision_cache.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": f"碰撞数据收集失败: {str(e)}"
            })


def _write_collision_to_file():
    """定时（5秒）将缓存的碰撞数据写入文件"""
    global stop_flag
    file_path = "collision_logs.txt"  # 碰撞日志文件路径

    while not stop_flag:
        time.sleep(5)  # 每5秒执行一次

        # 加锁读取并清空缓存
        with cache_lock:
            if not collision_cache:
                continue  # 无数据则跳过
            current_data = collision_cache.copy()
            collision_cache.clear()

        # 写入文件（追加模式，避免覆盖）
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                # 写入批次分隔符
                f.write(f"\n===== 批次更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                for data in current_data:
                    if "error" in data:
                        # 错误日志格式
                        f.write(f"[错误] {data['time']} | {data['error']}\n")
                    else:
                        # 正常碰撞日志格式
                        f.write(
                            f"[碰撞] {data['time']} | "
                            f"自车ID:{data['vehicle_id']}({data['vehicle_type']}) | "
                            f"碰撞对象ID:{data['other_actor_id']}({data['other_actor_type']}) | "
                            f"位置:{data['location']}\n"
                        )
        except Exception as e:
            # 写文件失败时，将数据回写缓存（避免丢失）
            with cache_lock:
                collision_cache.extend(current_data)
            print(f"⚠️ 碰撞日志写入失败: {str(e)}（数据已回写缓存）")


def stop_collision_monitor():
    """停止监测（主程序退出时调用，确保剩余数据写入文件）"""
    global stop_flag
    stop_flag = True
    # 强制写入剩余缓存数据
    with cache_lock:
        if collision_cache:
            file_path = "collision_logs.txt"
            try:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(f"\n===== 程序退出批次: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                    for data in collision_cache:
                        if "error" in data:
                            f.write(f"[错误] {data['time']} | {data['error']}\n")
                        else:
                            f.write(
                                f"[碰撞] {data['time']} | "
                                f"自车ID:{data['vehicle_id']}({data['vehicle_type']}) | "
                                f"碰撞对象ID:{data['other_actor_id']}({data['other_actor_type']}) | "
                                f"位置:{data['location']}\n"
                            )
                collision_cache.clear()
            except Exception as e:
                print(f"⚠️ 程序退出时写入剩余碰撞数据失败: {str(e)}")