import carla

def set_all_traffic_lights(world, green_time=10.0, red_time=10.0, yellow_time=2.0):
    """
    统一设置地图中所有红绿灯的切换时间
    :param world: CARLA world 对象
    :param green_time: 绿灯时长
    :param red_time: 红灯时长
    :param yellow_time: 黄灯时长
    """
    try:
        traffic_lights = world.get_actors().filter('traffic.traffic_light*')
        for tl in traffic_lights:
            tl.set_green_time(green_time)
            tl.set_red_time(red_time)
            tl.set_yellow_time(yellow_time)
    except:
        print("⚠️ 红绿灯设置失败")

def get_current_light_state(car):
    """
    获取车辆前方红绿灯状态
    :return: 红灯/绿灯/黄灯/无
    """
    light_state = "无"
    try:
        tl = car.get_traffic_light()
        if tl:
            dis = car.get_transform().location.distance(tl.get_transform().location)
            if dis < 20:
                state = tl.get_state()
                if state == carla.TrafficLightState.Red:
                    light_state = "红灯"
                elif state == carla.TrafficLightState.Green:
                    light_state = "绿灯"
                elif state == carla.TrafficLightState.Yellow:
                    light_state = "黄灯"
    except:
        light_state = "无"
    return light_state