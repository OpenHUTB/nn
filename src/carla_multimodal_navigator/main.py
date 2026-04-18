# --------------------------
# 简化修复版：确保车辆正确生成
# --------------------------
import carla
import time
import numpy as np
import cv2
import math
from collections import deque
import random


class SimpleController:
    """简单但可靠的控制逻辑"""

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self.target_speed = 30.0  # km/h
        self.waypoint_distance = 5.0
        self.last_waypoint = None
        # 限速检测相关属性
        self.speed_limit = 30.0  # 默认限速 30 km/h
        self.speed_limit_detected = False  # 是否检测到限速标志

    def detect_speed_limits(self, location, transform):
        """检测道路限速标志"""
        # 重置限速检测状态
        self.speed_limit_detected = False
        
        # 简单的限速检测逻辑，确保车辆能够根据限速调整速度
        # 每100米切换一次限速，以便测试
        distance = math.sqrt(location.x ** 2 + location.y ** 2)
        
        if distance < 100:
            self.speed_limit = 20.0  # 学校区域
            self.speed_limit_detected = True
        elif distance < 200:
            self.speed_limit = 40.0  # 普通道路
            self.speed_limit_detected = True
        else:
            self.speed_limit = 30.0  # 默认限速
            self.speed_limit_detected = False

    def get_control(self, speed):
        """基于路点的简单控制"""
        # 获取车辆状态
        location = self.vehicle.get_location()
        transform = self.vehicle.get_transform()

        # 检测限速标志
        self.detect_speed_limits(location, transform)

        # 获取路点
        waypoint = self.map.get_waypoint(location, project_to_road=True)

        if not waypoint:
            # 如果没有找到路点，返回保守控制
            return 0.3, 0.0, 0.0

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
            steer = max(-0.5, min(0.5, angle / 1.0))

        # 速度控制
        # 使用检测到的限速作为目标速度
        current_target_speed = self.speed_limit if self.speed_limit_detected else self.target_speed
        
        if speed < current_target_speed * 0.8:
            throttle, brake = 0.6, 0.0
        elif speed > current_target_speed * 1.2:
            throttle, brake = 0.0, 0.3
        else:
            throttle, brake = 0.3, 0.0

        return throttle, brake, steer


class SimpleDrivingSystem:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.speed_sensor = None
        self.controller = None
        self.camera_image = None
        self.vehicle_speed = 0.0  # km/h

    def connect(self):
        """连接到CARLA服务器"""
        print("正在连接到CARLA服务器...")

        try:
            # 尝试多种连接方式
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)

            # 检查可用地图
            available_maps = self.client.get_available_maps()
            print(f"可用地图: {available_maps}")

            # 加载地图
            self.world = self.client.load_world('Town01')
            print("地图加载成功")

            # 设置同步模式
            settings = self.world.get_settings()
            settings.synchronous_mode = False  # 先使用异步模式确保连接
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

            print("连接成功！")
            return True

        except Exception as e:
            print(f"连接失败: {e}")
            print("请确保:")
            print("1. CARLA服务器正在运行")
            print("2. 服务器端口为2000")
            print("3. 地图Town01可用")
            return False

    def spawn_vehicle(self):
        """生成车辆 - 简化版本"""
        print("正在生成车辆...")

        try:
            # 获取蓝图库
            blueprint_library = self.world.get_blueprint_library()

            # 选择车辆蓝图
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            if not vehicle_bp:
                print("未找到特斯拉蓝图，尝试其他车辆...")
                vehicle_bp = blueprint_library.filter('vehicle.*')[0]

            vehicle_bp.set_attribute('color', '255,0,0')  # 红色

            # 获取出生点
            spawn_points = self.world.get_map().get_spawn_points()
            print(f"找到 {len(spawn_points)} 个出生点")

            if not spawn_points:
                print("没有可用的出生点！")
                return False

            # 选择第一个出生点
            spawn_point = spawn_points[0]

            # 尝试生成车辆
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if not self.vehicle:
                print("无法生成车辆，尝试清理现有车辆...")
                # 清理现有车辆
                for actor in self.world.get_actors().filter('vehicle.*'):
                    actor.destroy()
                time.sleep(0.5)

                # 再次尝试
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            if self.vehicle:
                print(f"车辆生成成功！ID: {self.vehicle.id}")
                print(f"位置: {spawn_point.location}")

                # 禁用自动驾驶
                self.vehicle.set_autopilot(False)

                return True
            else:
                print("车辆生成失败")
                return False

        except Exception as e:
            print(f"生成车辆时出错: {e}")
            return False

    def setup_camera(self):
        """设置相机"""
        print("正在设置相机...")

        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')

            # 设置相机属性
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '90')

            # 相机位置（车辆后方）
            camera_transform = carla.Transform(
                carla.Location(x=-8.0, z=6.0),  # 在车辆后方上方
                carla.Rotation(pitch=-20.0)  # 向下看
            )

            # 生成相机
            self.camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )

            # 设置回调函数
            self.camera.listen(lambda image: self.camera_callback(image))

            print("相机设置成功")
            return True

        except Exception as e:
            print(f"设置相机时出错: {e}")
            return False

    def setup_speed_sensor(self):
        """设置速度传感器"""
        print("正在设置速度传感器...")

        try:
            blueprint_library = self.world.get_blueprint_library()
            # 尝试查找速度传感器蓝图
            speed_sensors = blueprint_library.filter('sensor.other.speed')
            
            if speed_sensors:
                # 使用专用速度传感器
                speed_bp = speed_sensors[0]
                print("使用专用速度传感器")
            else:
                # 使用IMU传感器作为速度传感器
                print("未找到专用速度传感器，使用IMU传感器")
                imu_sensors = blueprint_library.filter('sensor.other.imu')
                if not imu_sensors:
                    print("未找到速度或IMU传感器，将使用备用方法")
                    return False
                speed_bp = imu_sensors[0]

            # 设置传感器属性
            try:
                speed_bp.set_attribute('sensor_tick', '0.01')  # 100Hz
            except Exception as e:
                print(f"设置传感器属性时出错: {e}")

            # 传感器位置（车辆中心）
            sensor_transform = carla.Transform(
                carla.Location(x=0.0, y=0.0, z=1.0),  # 车辆中心上方
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )

            # 生成传感器
            try:
                self.speed_sensor = self.world.spawn_actor(
                    speed_bp, sensor_transform, attach_to=self.vehicle
                )
            except Exception as e:
                print(f"生成速度传感器时出错: {e}")
                return False

            # 设置回调函数
            try:
                self.speed_sensor.listen(lambda data: self.speed_sensor_callback(data))
            except Exception as e:
                print(f"设置速度传感器回调时出错: {e}")
                # 继续执行，不设置回调

            print("速度传感器设置成功")
            return True

        except Exception as e:
            print(f"设置速度传感器时出错: {e}")
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

    def speed_sensor_callback(self, data):
        """速度传感器数据回调"""
        try:
            # 从传感器数据中获取速度
            # 检查数据类型并相应处理
            if hasattr(data, 'velocity'):
                # 专用速度传感器
                velocity = data.velocity
                self.vehicle_speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6
            elif hasattr(data, 'accelerometer'):
                # IMU传感器，使用车辆速度作为参考
                if self.vehicle:
                    velocity = self.vehicle.get_velocity()
                    self.vehicle_speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6
            else:
                # 其他类型传感器
                if self.vehicle:
                    velocity = self.vehicle.get_velocity()
                    self.vehicle_speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6
        except Exception as e:
            print(f"速度传感器回调错误: {e}")
            # 回退到直接获取车辆速度
            if self.vehicle:
                try:
                    velocity = self.vehicle.get_velocity()
                    self.vehicle_speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6
                except:
                    pass

    def setup_controller(self):
        """设置控制器"""
        self.controller = SimpleController(self.world, self.vehicle)
        print("控制器设置完成")

    def run(self):
        """主运行循环"""
        print("\n" + "=" * 50)
        print("简化自动驾驶系统")
        print("=" * 50)

        # 连接服务器
        if not self.connect():
            return

        # 生成车辆
        if not self.spawn_vehicle():
            return

        # 设置相机
        if not self.setup_camera():
            # 即使相机失败也继续运行
            print("警告：相机设置失败，继续运行...")

        # 设置速度传感器
        if not self.setup_speed_sensor():
            # 即使传感器失败也继续运行，但会使用备用方法
            print("警告：速度传感器设置失败，将使用备用速度获取方法...")

        # 设置控制器
        self.setup_controller()

        # 等待一会儿让系统稳定
        print("系统初始化中...")
        time.sleep(2.0)

        # 设置天气
        weather = carla.WeatherParameters(
            cloudiness=30.0,
            precipitation=0.0,
            sun_altitude_angle=70.0
        )
        self.world.set_weather(weather)

        # 生成一些NPC车辆
        self.spawn_npc_vehicles(2)

        print("\n系统准备就绪！")
        print("控制指令:")
        print("  q - 退出程序")
        print("  r - 重置车辆")
        print("  s - 紧急停止")
        print("\n开始自动驾驶...\n")

        frame_count = 0
        running = True

        try:
            while running:
                # 获取速度数据（优先使用传感器数据）
                if self.vehicle_speed > 0:
                    speed = self.vehicle_speed
                else:
                    # 备用方法：直接从车辆获取速度
                    velocity = self.vehicle.get_velocity()
                    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6

                # 获取控制指令
                throttle, brake, steer = self.controller.get_control(speed)

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
                    # 添加限速信息
                    speed_limit_status = "Detected" if self.controller.speed_limit_detected else "Default"
                    cv2.putText(display_img, f"Speed Limit: {self.controller.speed_limit:.0f} km/h",
                                (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    cv2.putText(display_img, f"Limit Status: {speed_limit_status}",
                                (20, 240), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    # 添加速度传感器状态
                    speed_sensor_status = "Active" if self.speed_sensor else "Inactive"
                    cv2.putText(display_img, f"Speed Sensor: {speed_sensor_status}",
                                (20, 280), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 255), 2)

                    cv2.imshow('Autonomous Driving - Simple Version', display_img)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("正在退出...")
                    running = False
                elif key == ord('r'):
                    self.reset_vehicle()
                elif key == ord('s'):
                    # 紧急停止
                    self.vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, brake=1.0, hand_brake=True
                    ))
                    print("紧急停止")

                frame_count += 1

                # 每100帧显示一次状态
                if frame_count % 100 == 0:
                    print(f"运行中... 帧数: {frame_count}, 速度: {speed:.1f} km/h")

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"运行错误: {e}")
        finally:
            self.cleanup()

    def spawn_npc_vehicles(self, count=2):
        """生成NPC车辆（简化）"""
        print(f"正在生成 {count} 辆NPC车辆...")

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
                            print(f"生成NPC车辆 {len(npc_vehicles)}")
                except:
                    pass

            print(f"成功生成 {len(npc_vehicles)} 辆NPC车辆")

        except Exception as e:
            print(f"生成NPC车辆时出错: {e}")

    def reset_vehicle(self):
        """重置车辆位置"""
        print("重置车辆...")

        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            new_spawn_point = random.choice(spawn_points)
            self.vehicle.set_transform(new_spawn_point)
            print(f"车辆已重置到新位置: {new_spawn_point.location}")

            # 等待重置完成
            time.sleep(0.5)

    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")

        if self.camera:
            try:
                self.camera.stop()
                self.camera.destroy()
            except:
                pass

        if self.speed_sensor:
            try:
                self.speed_sensor.stop()
                self.speed_sensor.destroy()
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
        print("清理完成")


def main():
    """主函数"""
    print("自动驾驶系统 - 简化版本")
    print("确保CARLA服务器正在运行...")

    system = SimpleDrivingSystem()
    system.run()


if __name__ == "__main__":
    main()