import sys
import time
import signal
import keyboard
from pathlib import Path
import carla
import logging
from utils import calculate_vehicle_speed_kmh, debounce_check
from collision_monitor import CollisionMonitor
from environment_controller import env_controller
from traffic_light_controller import tl_controller
from vehicle_status_gui import gui_instance, create_status_window, stop_gui, update_vehicle_status
from config import (
    CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT,
    MAX_SPEED_KMH, SPAWN_POINT_OFFSET, STEER_ANGLE, BRAKE_INTENSITY,
    WEATHER_LIST
)

# 初始化日志
logger = logging.getLogger(__name__)

class CarlaDriver:
    def __init__(self):
        self.exit_flag = False
        self.car = None
        self.collision_monitor = CollisionMonitor()
        self.current_weather_idx = 0
        # 防抖标记
        self.w_key_triggered = [False]
        self.c_key_triggered = [False]
        # 注册退出信号
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

    def handle_exit(self, _sig, _frame) -> None:
        """退出信号处理：优雅释放资源"""
        if self.exit_flag:
            return
        self.exit_flag = True
        logger.warning("程序终止信号触发")
        self.cleanup_resources()
        sys.exit(0)

    def cleanup_resources(self) -> None:
        """清理所有CARLA资源和GUI"""
        # 停止GUI
        stop_gui()
        # 停止碰撞监测
        self.collision_monitor.stop()
        # 销毁车辆
        if self.car:
            self.car.destroy()
            logger.info("车辆Actor已销毁")
        logger.info("资源清理完成")

    def init_carla(self) -> carla.World:
        """初始化CARLA连接和车辆"""
        # 添加CARLA PythonAPI路径
        BASE_DIR = Path(__file__).parent
        sys.path.append(str(BASE_DIR / "PythonAPI" / "carla" / "dist"))

        # 连接CARLA服务器
        try:
            client = carla.Client(CARLA_HOST, CARLA_PORT)
            client.set_timeout(CARLA_TIMEOUT)
            world = client.get_world()
            logger.info(f"成功连接CARLA服务器：{CARLA_HOST}:{CARLA_PORT}")
        except ConnectionRefusedError:
            raise ConnectionRefusedError(
                "连接CARLA失败！请确认：\n"
                f"1. CARLA模拟器已启动（端口{CARLA_PORT}）\n"
                f"2. 服务器地址正确（{CARLA_HOST}:{CARLA_PORT}）"
            )

        # 生成车辆
        carla_map = world.get_map()
        spawn_point = carla_map.get_spawn_points()[0]
        spawn_point.location -= spawn_point.get_forward_vector() * SPAWN_POINT_OFFSET

        try:
            car_bp = world.get_blueprint_library().filter("vehicle")[0]
            self.car = world.spawn_actor(car_bp, spawn_point)
            logger.info("车辆Actor生成成功")
        except Exception as e:
            raise RuntimeError(f"生成车辆失败：{e}")

        # 初始化碰撞传感器
        self.collision_monitor.create_collision_sensor(world, self.car)

        # 初始化天气
        env_controller.set_weather(world, WEATHER_LIST[self.current_weather_idx])

        # 启动GUI
        create_status_window()

        return world

    def print_operation_guide(self) -> None:
        """打印操作说明"""
        guide = """
========================================
操作说明：
↑：前进 | ↓：倒车 | ←：左转 | →：右转 
空格键：急刹 | C：模拟碰撞 | ESC：退出
W键：循环切换天气（晴天→雨天→雾天→夜间→晴天...）
📊 实时监测：车速 | 天气 | 能见度 | 碰撞状态 | 红绿灯违规
========================================
        """
        print(guide)
        logger.info("操作说明已打印")

    def control_vehicle(self) -> carla.VehicleControl:
        """车辆控制逻辑"""
        ctrl = carla.VehicleControl()
        ctrl.hand_brake = False
        ctrl.gear = 1

        # 急刹逻辑
        if keyboard.is_pressed("space"):
            ctrl.brake = BRAKE_INTENSITY
            ctrl.hand_brake = True
            ctrl.throttle = 0.0
            current_speed = calculate_vehicle_speed_kmh(self.car)
            logger.info(f"急刹触发！当前车速：{current_speed} km/h")
        else:
            # 前进/倒车
            if keyboard.is_pressed("up"):
                ctrl.throttle = 1.0
                ctrl.reverse = False
            elif keyboard.is_pressed("down"):
                ctrl.throttle = 1.0
                ctrl.reverse = True
                ctrl.gear = -1
            else:
                ctrl.throttle = 0.0

            # 转向
            if keyboard.is_pressed("left"):
                ctrl.steer = -STEER_ANGLE
            elif keyboard.is_pressed("right"):
                ctrl.steer = STEER_ANGLE
            else:
                ctrl.steer = 0.0

            # 普通刹车
            ctrl.brake = BRAKE_INTENSITY if keyboard.is_pressed("s") else 0.0

        # 限速
        current_speed = calculate_vehicle_speed_kmh(self.car)
        if current_speed > MAX_SPEED_KMH:
            ctrl.throttle = 0.2

        return ctrl

    def update_spectator(self, world: carla.World) -> None:
        """视角跟随车辆"""
        trans = self.car.get_transform()
        cam_loc = trans.location - trans.get_forward_vector() * 10 + carla.Location(z=4)
        cam_rot = trans.rotation
        cam_rot.pitch = -20
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))

    def main_loop(self, world: carla.World) -> None:
        """主循环：车辆控制+状态监测"""
        print_counter = 0
        red_light_violation_flag = False

        while not self.exit_flag:
            # 1. 计算并更新车速
            current_speed = calculate_vehicle_speed_kmh(self.car)
            update_vehicle_status("speed", current_speed)

            # 2. 定期打印状态
            print_counter += 1
            if print_counter % 20 == 0:
                env_state = env_controller.get_current_environment_state()
                env_info = f"天气：{env_state['weather_type']} | 能见度：{env_state['visibility']}%"
                collision_speed = gui_instance.vehicle_status["collision_speed"]
                collision_info = f"碰撞车速：{collision_speed} km/h"
                print(f"\r速度：{current_speed:.1f} km/h | {env_info} | {collision_info} | 闯红灯：否", end="")

                # 更新GUI的天气、能见度
                update_vehicle_status("weather", env_state['weather_type'])
                update_vehicle_status("visibility", env_state['visibility'])

            # 3. 车辆控制
            ctrl = self.control_vehicle()
            self.car.apply_control(ctrl)

            # 4. 视角跟随
            self.update_spectator(world)

            # 5. 碰撞状态处理
            collision_occurred = self.collision_monitor.get_collision_occurred()
            update_vehicle_status("collision_occurred", collision_occurred)
            if collision_occurred:
                self.collision_monitor.reset_collision_occurred()

            # 6. 模拟碰撞
            if debounce_check(keyboard.is_pressed("c"), self.c_key_triggered):
                if current_speed > 0:
                    update_vehicle_status("collision_speed", current_speed)
                    update_vehicle_status("collision_occurred", True)
                    logger.warning(f"模拟碰撞：车速{current_speed} km/h")
                else:
                    logger.warning("车辆静止，无法模拟碰撞")

            # 7. 切换天气
            if debounce_check(keyboard.is_pressed("w"), self.w_key_triggered):
                self.current_weather_idx = (self.current_weather_idx + 1) % len(WEATHER_LIST)
                env_controller.set_weather(world, WEATHER_LIST[self.current_weather_idx])

            # 8. 红绿灯检测
            red_light_violation = tl_controller.check_red_light_violation(world, self.car)
            update_vehicle_status("red_light_violation", red_light_violation)

            if red_light_violation and not red_light_violation_flag:
                red_light_violation_flag = True
                logger.warning("检测到闯红灯行为！")
            elif not red_light_violation:
                red_light_violation_flag = False

            # 9. 退出检测
            if keyboard.is_pressed("esc") and not self.exit_flag:
                self.exit_flag = True
                break

            time.sleep(0.01)

    def run(self) -> None:
        """启动主程序"""
        try:
            # 初始化CARLA
            world = self.init_carla()
            # 打印操作说明
            self.print_operation_guide()
            # 主循环
            self.main_loop(world)
        except ConnectionRefusedError as e:
            logger.error(e)
            sys.exit(1)
        except Exception as e:
            logger.error(f"程序异常：{e}", exc_info=True)
        finally:
            self.cleanup_resources()
            logger.info("程序正常退出")

if __name__ == "__main__":
    driver = CarlaDriver()
    driver.run()