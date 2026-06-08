import carla
from datetime import datetime
import logging
from utils import calculate_vehicle_speed_kmh, safe_update_dict
from config import COLLISION_SENSOR_BP, COLLISION_LOG_FILE

logger = logging.getLogger(__name__)

class CollisionMonitor:
    def __init__(self):
        self.collision_occurred = False
        self.collision_sensor = None

    def create_collision_sensor(self, world: carla.World, vehicle: carla.Vehicle) -> None:
        """
        创建碰撞传感器并绑定到车辆
        :param world: CARLA World对象
        :param vehicle: 被监测的车辆Actor
        :raises RuntimeError: 传感器生成失败时抛出
        """
        try:
            collision_bp = world.get_blueprint_library().find(COLLISION_SENSOR_BP)
            self.collision_sensor = world.spawn_actor(
                collision_bp,
                carla.Transform(),
                attach_to=vehicle
            )
            # 绑定碰撞回调函数
            self.collision_sensor.listen(lambda event: self.on_collision(event, vehicle))
            logger.info("碰撞传感器已启动")
        except Exception as e:
            raise RuntimeError(f"创建碰撞传感器失败：{e}")

    def on_collision(self, event, vehicle: carla.Vehicle) -> None:
        """碰撞事件回调处理：记录日志+更新碰撞状态"""
        self.collision_occurred = True
        collision_speed = calculate_vehicle_speed_kmh(vehicle)

        # 1. 记录碰撞详细日志
        collision_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        collision_actor_type = event.other_actor.type_id
        collision_loc = vehicle.get_transform().location
        loc_str = f"({collision_loc.x:.2f}, {collision_loc.y:.2f}, {collision_loc.z:.2f})"

        log_line = (
            f"[{collision_time}] 发生碰撞 | "
            f"碰撞对象：{collision_actor_type} | "
            f"位置：{loc_str} | "
            f"碰撞车速：{collision_speed} km/h"
        )
        # 写入日志文件
        try:
            with open(COLLISION_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
            logger.warning(log_line)
        except IOError as e:
            logger.error(f"写入碰撞日志失败：{e}")

        # 2. 对外暴露碰撞车速
        from vehicle_status_gui import update_vehicle_status
        update_vehicle_status("collision_speed", collision_speed)

    def get_collision_occurred(self) -> bool:
        """读取当前碰撞状态"""
        return self.collision_occurred

    def reset_collision_occurred(self) -> None:
        """重置碰撞状态"""
        self.collision_occurred = False

    def destroy_sensor(self) -> None:
        """销毁碰撞传感器"""
        if self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None
            logger.info("碰撞传感器已销毁")

    def stop(self) -> None:
        """停止碰撞监测"""
        self.reset_collision_occurred()
        self.destroy_sensor()
        logger.info("碰撞监测已停止")