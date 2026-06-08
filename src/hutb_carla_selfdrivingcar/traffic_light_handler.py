import carla
import math

class TrafficLightHandler:
    """交通灯识别和响应"""

    def __init__(self, world):
        self.world = world
        self.traffic_lights = self._find_traffic_lights()
        print(f"[TrafficLight] 找到 {len(self.traffic_lights)} 个交通灯")

    def _find_traffic_lights(self):
        """查找场景中的所有交通灯"""
        actors = self.world.get_actors()
        traffic_lights = actors.filter('traffic.traffic_light')
        return list(traffic_lights)

    def get_traffic_light_state(self, traffic_light):
        """获取交通灯状态"""
        state = traffic_light.get_state()
        # CARLA 交通灯状态: Green, Yellow, Red
        return state

    def is_red_or_yellow(self, traffic_light):
        """检查交通灯是否为红灯或黄灯"""
        state = traffic_light.get_state()
        return state == carla.TrafficLightState.Red or state == carla.TrafficLightState.Yellow

    def get_nearby_traffic_lights(self, ego_vehicle, radius=50, forward_only=True):
        """获取附近的交通灯"""
        ego_location = ego_vehicle.get_location()
        ego_velocity = ego_vehicle.get_velocity()
        nearby_lights = []

        for tl in self.traffic_lights:
            tl_location = tl.get_location()
            distance = ego_location.distance(tl_location)

            if distance < radius:
                # 如果只检查前方交通灯
                if forward_only:
                    # 计算车辆前进方向
                    ego_transform = ego_vehicle.get_transform()
                    forward_vector = ego_transform.get_forward_vector()

                    # 计算交通灯相对于车辆的方向向量
                    direction_to_tl = carla.Vector3D(
                        x=tl_location.x - ego_location.x,
                        y=tl_location.y - ego_location.y,
                        z=0
                    )

                    # 计算点积，判断是否在前方
                    dot_product = forward_vector.x * direction_to_tl.x + forward_vector.y * direction_to_tl.y

                    if dot_product > 0:  # 交通灯在前方
                        nearby_lights.append({
                            'traffic_light': tl,
                            'distance': distance,
                            'state': tl.get_state(),
                            'location': tl_location
                        })
                else:
                    nearby_lights.append({
                        'traffic_light': tl,
                        'distance': distance,
                        'state': tl.get_state(),
                        'location': tl_location
                    })

        # 按距离排序
        nearby_lights.sort(key=lambda x: x['distance'])
        return nearby_lights

    def should_stop_for_traffic_light(self, ego_vehicle, stop_distance=8):
        """判断是否需要为交通灯停车"""
        nearby_lights = self.get_nearby_traffic_lights(ego_vehicle, radius=50, forward_only=True)

        for light_info in nearby_lights:
            tl = light_info['traffic_light']
            distance = light_info['distance']
            state = light_info['state']

            # 如果是红灯或黄灯，且距离小于停车距离
            if self.is_red_or_yellow(tl) and distance < stop_distance:
                return True, distance, state

            # 如果是黄灯且距离较近，需要停车
            if state == carla.TrafficLightState.Yellow and distance < stop_distance * 2:
                return True, distance, state

        return False, 0, None

    def get_stop_control(self):
        """获取停车控制命令"""
        return carla.VehicleControl(throttle=0.0, brake=1.0)

    def update_traffic_light_states(self):
        """更新交通灯状态信息（用于调试）"""
        states = {}
        for tl in self.traffic_lights:
            state = tl.get_state()
            state_name = str(state).split('.')[-1]
            states[tl.id] = state_name
        return states