import carla
import time
import logging
from config import CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT, UNSTRUCTURED_SCENES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarlaClient:
    def __init__(self):
        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.spectator = None
        self.anomaly_actors = []  # 存储异常场景Actor（如落石、施工围挡）

    def connect(self):
        """连接Carla服务器"""
        try:
            self.client = carla.Client(CARLA_HOST, CARLA_PORT)
            self.client.set_timeout(CARLA_TIMEOUT)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
            logger.info(f"成功连接Carla，当前地图：{self.map.name}")
            return True
        except Exception as e:
            logger.error(f"连接Carla失败：{e}")
            return False

    def load_unstructured_scene(self, town_name="town07"):
        """加载非结构化场景地图"""
        if town_name not in UNSTRUCTURED_SCENES:
            logger.error(f"不支持的非结构化场景：{town_name}")
            return False

        # 加载地图
        self.world = self.client.load_world(town_name)
        self.map = self.world.get_map()

        # 设置天气
        weather_config = UNSTRUCTURED_SCENES[town_name]["weather"]
        if weather_config == "RAINY":
            self.world.set_weather(carla.WeatherParameters.RainyNight)
        elif weather_config == "FOGGY":
            self.world.set_weather(carla.WeatherParameters.Foggy)

        logger.info(f"加载非结构化场景：{town_name}，天气：{weather_config}")
        return True

    def spawn_vehicle(self, spawn_point_idx=0):
        """生成自动驾驶车辆"""
        # 获取车辆蓝图
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("color", "255,0,0")

        # 获取生成点（优先非结构化道路）
        spawn_points = self.map.get_spawn_points()
        if spawn_point_idx >= len(spawn_points):
            spawn_point_idx = 0
        spawn_point = spawn_points[spawn_point_idx]

        # 生成车辆
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(False)  # 关闭自动巡航，手动控制

        # 设置 spectator 视角跟随车辆
        self.spectator = self.world.get_spectator()
        self._update_spectator()

        logger.info(f"生成车辆：{self.vehicle.id}")
        return self.vehicle

    def spawn_anomaly_actors(self, town_name="town07"):
        """生成非结构化场景异常Actor"""
        anomaly_types = UNSTRUCTURED_SCENES[town_name]["anomaly_types"]
        blueprint_library = self.world.get_blueprint_library()

        # 生成不同类型异常
        for anomaly_type in anomaly_types:
            if anomaly_type == "fallen_tree":
                bp = blueprint_library.find("static.prop.tree_fallen")
                transform = carla.Transform(carla.Location(x=120, y=45, z=0.5), carla.Rotation(yaw=90))
            elif anomaly_type == "pothole":
                bp = blueprint_library.find("static.prop.pothole")
                transform = carla.Transform(carla.Location(x=110, y=40, z=0.1))
            elif anomaly_type == "construction":
                bp = blueprint_library.find("static.prop.barrier")
                transform = carla.Transform(carla.Location(x=130, y=50, z=0.5))
            else:
                continue

            actor = self.world.spawn_actor(bp, transform)
            self.anomaly_actors.append(actor)
            logger.info(f"生成异常Actor：{anomaly_type}（ID：{actor.id}）")

    def _update_spectator(self):
        """更新 spectator 视角"""
        if self.vehicle and self.spectator:
            transform = self.vehicle.get_transform()
            self.spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=5, x=-10),
                carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
            ))

    def clean_up(self):
        """清理所有Actor"""
        # 清理车辆
        if self.vehicle:
            self.vehicle.destroy()
            logger.info("销毁车辆Actor")

        # 清理异常Actor
        for actor in self.anomaly_actors:
            actor.destroy()
        logger.info("销毁所有异常Actor")

        self.client = None
        self.world = None
        self.vehicle = None


if __name__ == "__main__":
    # 测试客户端
    client = CarlaClient()
    if client.connect():
        client.load_unstructured_scene("town07")
        client.spawn_vehicle()
        client.spawn_anomaly_actors("town07")
        time.sleep(10)
        client.clean_up()