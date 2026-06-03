import carla
import math

def get_vehicle_traffic_light(world: carla.World, vehicle: carla.Vehicle) -> carla.TrafficLight:
    """
    （兼容版）获取车辆当前接近的红绿灯
    :param world: CARLA World对象
    :param vehicle: 车辆Actor
    :return: 红绿灯Actor（无则返回None）
    """
    # 获取所有红绿灯
    traffic_lights = world.get_actors().filter("traffic.traffic_light")
    if not traffic_lights:
        return None

    vehicle_loc = vehicle.get_transform().location
    min_distance = float('inf')
    target_light = None

    # 遍历红绿灯，找到距离最近且处于红灯状态的
    for light in traffic_lights:
        if light.get_state() != carla.TrafficLightState.Red:
            continue  # 只检测红灯

        # 计算车辆与红绿灯的距离
        light_loc = light.get_transform().location
        distance = math.sqrt(
            (vehicle_loc.x - light_loc.x)**2 +
            (vehicle_loc.y - light_loc.y)**2 +
            (vehicle_loc.z - light_loc.z)**2
        )

        # 只检测距离<50米的红绿灯（避免跨路口误判）
        if distance < 50 and distance < min_distance:
            min_distance = distance
            target_light = light

    return target_light

def get_traffic_light_state(light: carla.TrafficLight) -> str:
    """
    获取红绿灯状态（字符串格式）
    :param light: 红绿灯Actor
    :return: red/yellow/green/off
    """
    if not light:
        return "off"
    state = light.get_state()
    if state == carla.TrafficLightState.Red:
        return "red"
    elif state == carla.TrafficLightState.Yellow:
        return "yellow"
    elif state == carla.TrafficLightState.Green:
        return "green"
    else:
        return "off"

def check_red_light_violation(world: carla.World, vehicle: carla.Vehicle) -> bool:
    """
    （兼容版）检测车辆是否闯红灯
    :param world: CARLA World对象
    :param vehicle: 车辆Actor
    :return: True=闯红灯，False=未闯红灯
    """
    light = get_vehicle_traffic_light(world, vehicle)
    if light and get_traffic_light_state(light) == "red":
        # 车辆在红灯时仍在移动（速度>0）则判定为闯红灯
        from utils import calculate_vehicle_speed_kmh
        speed = calculate_vehicle_speed_kmh(vehicle)
        if speed > 0:
            print(f"\n🚦 闯红灯检测：车速 {speed} km/h | 位置 {vehicle.get_transform().location}")
            return True
    return False