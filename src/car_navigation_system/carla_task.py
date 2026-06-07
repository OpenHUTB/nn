# --------------------------
# 简化修复版：确保车辆正确生成
# --------------------------

import logging
import carla
import time
import numpy as np
import cv2
import math
from collections import deque
import random

class Config:
    """配置参数类"""
    # 车辆参数
    TARGET_SPEED = 30.0  # km/h
    WAYPOINT_DISTANCE = 5.0  # 路点距离
    
    # 控制参数
    THROTTLE_MAX = 0.6
    THROTTLE_NORMAL = 0.3
    BRAKE_STRONG = 0.3
    STEER_MAX = 0.5
    
    # 相机参数
    CAMERA_IMAGE_SIZE_X = 640
    CAMERA_IMAGE_SIZE_Y = 480
    CAMERA_FOV = 90
    CAMERA_LOCATION_X = -8.0
    CAMERA_LOCATION_Z = 6.0
    CAMERA_PITCH = -20.0
    
    # 世界参数
    WEATHER_CLOUDINESS = 30.0
    WEATHER_PRECIPITATION = 0.0
    WEATHER_SUN_ALTITUDE = 70.0
    
    # NPC参数
    NPC_VEHICLE_COUNT = 2
    
    # CARLA参数
    CARLA_HOST = 'localhost'
    CARLA_PORT = 2000
    TIMEOUT = 10.0
    FIXED_DELTA_SECONDS = 0.05

class SimpleController:
    """简单但可靠的控制逻辑"""

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self.target_speed = Config.TARGET_SPEED  # km/h
        self.waypoint_distance = Config.WAYPOINT_DISTANCE
        self.last_waypoint = None

    def get_control(self):
        """基于路点的简单控制"""
        # 获取车辆状态
        location = self.vehicle.get_location()
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        # 计算速度
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6  # km/h

        # 获取路点
        waypoint = self.map.get_waypoint(location, project_to_road=True)

        if not waypoint:
            # 如果没有找到路点，返回保守控制
            return Config.THROTTLE_NORMAL, 0.0, 0.0

        # 获取下一个路点
        next_waypoints = waypoint.next(self.waypoint_distance)

        if not next_waypoints:
            # 如果没有下一个路点，使用当前路点
            target_waypoint = waypoint
        else:
            target_waypoint = next_waypoints[0]

        self.last_waypoint = target_waypoint

        # 计算转向
        vehicle_yaw = math.radians(transform.rotation.yaw)
        target_loc = target_waypoint.transform.location

        # 计算相对位置
        dx = target_loc.x - location.x
        dy = target_loc.y - location.y

        local_x = dx * math.cos(vehicle_yaw) + dy * math.sin(vehicle_yaw)
        local_y = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)

        if abs(local_x) < 0.1:
            steer = 0.0
        else:
            angle = math.atan2(local_y, local_x)
            steer = max(-Config.STEER_MAX, min(Config.STEER_MAX, angle / 1.0))
        # 速度控制
        if speed < self.target_speed * 0.8:
            throttle, brake = Config.THROTTLE_MAX, 0.0
        elif speed > self.target_speed * 1.2:
            throttle, brake = 0.0, Config.BRAKE_STRONG
        else:
            throttle, brake = 0.3, 0.0

        return throttle, brake, steer


class SimpleDrivingSystem:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.controller = None
        self.camera_image = None
        # 日志系统
        self.setup_logger()

    def setup_logger(self):
        #设置日志系统
        self.logger = logging.getLogger('CARLA_Driving')
        self.logger.setLevel(logging.INFO)
    
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
    def connect(self):
        """连接到CARLA服务器"""
        self.logger.info("正在连接到CARLA服务器...")

        try:
            # 尝试多种连接方式
            self.client = carla.Client(Config.CARLA_HOST, Config.CARLA_PORT)
            self.client.set_timeout(Config.TIMEOUT)

            # 检查可用地图
            available_maps = self.client.get_available_maps()
            self.logger.info(f"可用地图: {available_maps}")

            # 加载地图
            self.world = self.client.load_world('Town01')
            self.logger.info("地图加载成功")

            # 设置同步模式
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = Config.FIXED_DELTA_SECONDS  # 先使用异步模式确保连接
            self.world.apply_settings(settings)
            
            self.logger.info("连接成功！")
            return True

        except Exception as e:
            self.logger.info(f"连接失败: {e}")
            self.logger.info("请确保:")
            self.logger.info("1. CARLA服务器正在运行")
            self.logger.info("2. 服务器端口为2000")
            self.logger.info("3. 地图Town01可用")
            return False

    def spawn_vehicle(self):
        """生成车辆 - 简化版本"""
        self.logger.info("正在生成车辆...")

        try:
            # 获取蓝图库
            blueprint_library = self.world.get_blueprint_library()

            # 选择车辆蓝图
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            if not vehicle_bp:
                self.logger.info("未找到特斯拉蓝图，尝试其他车辆...")
                vehicle_bp = blueprint_library.filter('vehicle.*')[0]

            vehicle_bp.set_attribute('color', '255,0,0')  # 红色

            # 获取出生点
            spawn_points = self.world.get_map().get_spawn_points()
            self.logger.info(f"找到 {len(spawn_points)} 个出生点")

            if not spawn_points:
                self.logger.info("没有可用的出生点！")
                return False

            # 选择第一个出生点
            spawn_point = spawn_points[0]

            # 尝试生成车辆
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if not self.vehicle:
                self.logger.info("无法生成车辆，尝试清理现有车辆...")
                # 清理现有车辆
                for actor in self.world.get_actors().filter('vehicle.*'):
                    actor.destroy()
                time.sleep(0.5)

                # 再次尝试
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if self.vehicle:
                self.logger.info(f"车辆生成成功！ID: {self.vehicle.id}")
                self.logger.info(f"位置: {spawn_point.location}")

                # 禁用自动驾驶
                self.vehicle.set_autopilot(False)

                return True
            else:
                self.logger.info("车辆生成失败")
                return False

        except Exception as e:
            self.logger.info(f"生成车辆时出错: {e}")
            return False

    def setup_camera(self):
        """设置相机"""
        self.logger.info("正在设置相机...")

        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')

            # 设置相机属性
            camera_bp.set_attribute('image_size_x', str(Config.CAMERA_IMAGE_SIZE_X))
            camera_bp.set_attribute('image_size_y', str(Config.CAMERA_IMAGE_SIZE_Y))
            camera_bp.set_attribute('fov', str(Config.CAMERA_FOV))
            camera_transform = carla.Transform(
                carla.Location(x=Config.CAMERA_LOCATION_X, z=Config.CAMERA_LOCATION_Z),
                carla.Rotation(pitch=Config.CAMERA_PITCH)
            )

            # 生成相机
            self.camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )

            # 设置回调函数
            self.camera.listen(lambda image: self.camera_callback(image))

            self.logger.info("相机设置成功")
            return True

        except Exception as e:
            self.logger.info(f"设置相机时出错: {e}")
            return False

    def camera_callback(self, image):
        """相机数据回调"""
        try:
            # 转换图像数据
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            self.camera_image = array[:, :, :3]  # RGB通道
        except:
            pass

    def setup_controller(self):
        """设置控制器"""
        self.controller = SimpleController(self.world, self.vehicle)
        self.logger.info("控制器设置完成")

    def run(self):
        """主运行循环"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("简化自动驾驶系统")
        self.logger.info("=" * 50)

        # 连接服务器
        if not self.connect():
            return

        # 生成车辆
        if not self.spawn_vehicle():
            return

        # 设置相机
        if not self.setup_camera():
            # 即使相机失败也继续运行
            self.logger.info("警告：相机设置失败，继续运行...")

        # 设置控制器
        self.setup_controller()

        # 等待一会儿让系统稳定
        self.logger.info("系统初始化中...")
        time.sleep(2.0)

        # 设置天气
        weather = carla.WeatherParameters(
            cloudiness=Config.WEATHER_CLOUDINESS,
            precipitation=Config.WEATHER_PRECIPITATION,
            sun_altitude_angle=Config.WEATHER_SUN_ALTITUDE
        )
        self.world.set_weather(weather)

        # 生成一些NPC车辆
        self.spawn_npc_vehicles(Config.NPC_VEHICLE_COUNT)

        self.logger.info("\n系统准备就绪！")
        self.logger.info("控制指令:")
        self.logger.info("  q - 退出程序")
        self.logger.info("  r - 重置车辆")
        self.logger.info("  s - 紧急停止")
        self.logger.info("\n开始自动驾驶...\n")

        frame_count = 0
        running = True

        try:
            while running:
                # 获取车辆状态
                velocity = self.vehicle.get_velocity()
                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6

                # 获取控制指令
                throttle, brake, steer = self.controller.get_control()

                # 应用控制
                control = carla.VehicleControl(
                    throttle=float(throttle),
                    brake=float(brake),
                    steer=float(steer),
                    hand_brake=False,
                    reverse=False
                )
                self.vehicle.apply_control(control)

                # 更新显示
                if self.camera_image is not None:
                    display_img = self.camera_image.copy()

                    # 添加状态信息
                    cv2.putText(display_img, f"Speed: {speed:.1f} km/h",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Throttle: {throttle:.2f}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Steer: {steer:.2f}",
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Frame: {frame_count}",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)

                    cv2.imshow('Autonomous Driving - Simple Version', display_img)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("正在退出...")
                    running = False
                elif key == ord('r'):
                    self.reset_vehicle()
                elif key == ord('s'):
                    # 紧急停止
                    self.vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, brake=1.0, hand_brake=True
                    ))
                    self.logger.info("紧急停止")

                frame_count += 1

                # 每100帧显示一次状态
                if frame_count % 100 == 0:
                    self.logger.info(f"运行中... 帧数: {frame_count}, 速度: {speed:.1f} km/h")

                self.world.tick()

        except KeyboardInterrupt:
            self.logger.info("\n用户中断")
        except Exception as e:
            self.logger.info(f"运行错误: {e}")
        finally:
            self.cleanup()

    def spawn_npc_vehicles(self, count=2):
        """生成NPC车辆（简化）"""
        self.logger.info(f"正在生成 {count} 辆NPC车辆...")

        try:
            blueprint_library = self.world.get_blueprint_library()
            spawn_points = self.world.get_map().get_spawn_points()

            npc_vehicles = []

            for i in range(min(count, len(spawn_points))):
                # 跳过主车辆的出生点
                if i == 0:
                    continue

                try:
                    # 随机选择车辆类型
                    vehicle_bps = list(blueprint_library.filter('vehicle.*'))
                    if vehicle_bps:
                        vehicle_bp = random.choice(vehicle_bps)

                        # 生成NPC
                        npc = self.world.try_spawn_actor(vehicle_bp, spawn_points[i])

                        if npc:
                            npc.set_autopilot(True)
                            npc_vehicles.append(npc)
                            self.logger.info(f"生成NPC车辆 {len(npc_vehicles)}")
                except:
                    pass

            self.logger.info(f"成功生成 {len(npc_vehicles)} 辆NPC车辆")

        except Exception as e:
            self.logger.info(f"生成NPC车辆时出错: {e}")

    def reset_vehicle(self):
        """重置车辆位置"""
        self.logger.info("重置车辆...")

        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            new_spawn_point = random.choice(spawn_points)
            self.vehicle.set_transform(new_spawn_point)
            self.logger.info(f"车辆已重置到新位置: {new_spawn_point.location}")

            # 等待重置完成
            time.sleep(0.5)

    def cleanup(self):
        """清理资源"""
        self.logger.info("\n正在清理资源...")

        if self.camera:
            try:
                self.camera.stop()
                self.camera.destroy()
            except:
                pass

        if self.vehicle:
            try:
                self.vehicle.destroy()
            except:
                pass

        # 等待销毁完成
        time.sleep(1.0)

        cv2.destroyAllWindows()
        self.logger.info("清理完成")


def main():
    """主函数"""
    print("自动驾驶系统 - 简化版本")
    print("确保CARLA服务器正在运行...")

    system = SimpleDrivingSystem()
    system.run()


if __name__ == "__main__":
    main()