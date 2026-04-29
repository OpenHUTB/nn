# --------------------------
# 简化修复版：确保车辆正确生成
# --------------------------

import carla
import time
import numpy as np
import cv2
import math
import random


class SimpleController:
    """简单但可靠的控制逻辑"""

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self.target_speed = 50.0  # km/h
        self.waypoint_distance = 5.0
        self.last_waypoint = None
        self.manual_reverse = False  # 倒车模式标志

    def get_control(self):
        """基于路点的简单控制"""
        # 获取车辆状态
        location = self.vehicle.get_location()
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        # 计算速度
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6  # km/h

        # 检查是否在倒车模式
        if self.manual_reverse:
            # 倒车模式：直接返回倒车控制
            return 0.3, 0.0, 0.0, True  # throttle, brake, steer, reverse

        # 获取路点
        waypoint = self.map.get_waypoint(location, project_to_road=True)

        if not waypoint:
            return 0.3, 0.0, 0.0, False

        # 获取下一个路点
        next_waypoints = waypoint.next(self.waypoint_distance)

        if not next_waypoints:
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
        if speed < self.target_speed * 0.8:
            throttle, brake = 0.6, 0.0
        elif speed > self.target_speed * 1.2:
            throttle, brake = 0.0, 0.3
        else:
            throttle, brake = 0.3, 0.0

        return throttle, brake, steer, False

    def toggle_reverse(self):
        """切换倒车模式"""
        self.manual_reverse = not self.manual_reverse
        if self.manual_reverse:
            print("进入倒车模式")
        else:
            print("退出倒车模式，恢复前进")


class SimpleDrivingSystem:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicles = []  # 存储多辆车辆
        self.cameras = []   # 存储多台相机
        self.controllers = []  # 存储多个控制器
        self.camera_images = []  # 存储多个相机图像
        self.current_vehicle_index = 0  # 当前选中的车辆索引

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

    def spawn_vehicle(self, count=3):
        """生成多辆车辆"""
        print(f"正在生成 {count} 辆车辆...")

        try:
            # 获取蓝图库
            blueprint_library = self.world.get_blueprint_library()

            # 获取出生点
            spawn_points = self.world.get_map().get_spawn_points()
            print(f"找到 {len(spawn_points)} 个出生点")

            if not spawn_points:
                print("没有可用的出生点！")
                return False

            # 清理现有车辆
            print("清理现有车辆...")
            for actor in self.world.get_actors().filter('vehicle.*'):
                actor.destroy()
            time.sleep(0.5)

            # 车辆颜色
            colors = ['255,0,0', '0,255,0', '0,0,255']  # 红、绿、蓝

            for i in range(count):
                if i >= len(spawn_points):
                    print(f"出生点不足，只生成 {i} 辆车辆")
                    break

                # 选择车辆蓝图
                vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
                if not vehicle_bp:
                    print("未找到特斯拉蓝图，尝试其他车辆...")
                    vehicle_bp = blueprint_library.filter('vehicle.*')[0]

                # 设置车辆颜色
                if i < len(colors):
                    vehicle_bp.set_attribute('color', colors[i])
                else:
                    # 随机颜色
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    vehicle_bp.set_attribute('color', f'{r},{g},{b}')

                # 选择出生点
                spawn_point = spawn_points[i]

                # 尝试生成车辆
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

                if vehicle:
                    print(f"车辆 {i+1} 生成成功！ID: {vehicle.id}")
                    print(f"位置: {spawn_point.location}")

                    # 禁用自动驾驶
                    vehicle.set_autopilot(False)

                    # 添加到车辆列表
                    self.vehicles.append(vehicle)
                else:
                    print(f"车辆 {i+1} 生成失败")

            if self.vehicles:
                return True
            else:
                print("所有车辆生成失败")
                return False

        except Exception as e:
            print(f"生成车辆时出错: {e}")
            return False

    def setup_camera(self):
        """为所有车辆设置相机"""
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

            for i, vehicle in enumerate(self.vehicles):
                # 生成相机
                camera = self.world.spawn_actor(
                    camera_bp, camera_transform, attach_to=vehicle
                )

                # 设置回调函数，使用闭包来捕获索引
                def make_callback(index):
                    def callback(image):
                        self.camera_callback(image, index)
                    return callback

                # 为每个相机设置回调
                camera.listen(make_callback(i))

                # 添加到相机列表
                self.cameras.append(camera)
                # 初始化相机图像
                self.camera_images.append(None)

            print(f"成功设置 {len(self.cameras)} 台相机")
            return True

        except Exception as e:
            print(f"设置相机时出错: {e}")
            return False

    def camera_callback(self, image, index):
        """相机数据回调"""
        try:
            # 转换图像数据
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            if index < len(self.camera_images):
                self.camera_images[index] = array[:, :, :3]  # RGB通道
        except:
            pass

    def setup_controller(self):
        """为所有车辆设置控制器"""
        for vehicle in self.vehicles:
            controller = SimpleController(self.world, vehicle)
            self.controllers.append(controller)
        print(f"成功设置 {len(self.controllers)} 个控制器")

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
        print("  r - 重置当前车辆")
        print("  s - 紧急停止当前车辆")
        print("  x - 切换当前车辆的倒车/前进模式（速度为0时生效）")
        print("  1/2/3 - 切换到对应车辆的视角")
        print("\n开始自动驾驶...\n")

        frame_count = 0
        running = True

        try:
            while running:
                # 控制所有车辆
                for i, (vehicle, controller) in enumerate(zip(self.vehicles, self.controllers)):
                    # 获取控制指令
                    throttle, brake, steer, reverse = controller.get_control()

                    # 应用控制
                    control = carla.VehicleControl(
                        throttle=float(throttle),
                        brake=float(brake),
                        steer=float(steer),
                        hand_brake=False,
                        reverse=reverse
                    )
                    vehicle.apply_control(control)

                # 获取当前车辆状态
                current_vehicle = self.vehicles[self.current_vehicle_index]
                current_controller = self.controllers[self.current_vehicle_index]
                velocity = current_vehicle.get_velocity()
                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6

                # 更新显示
                if self.current_vehicle_index < len(self.camera_images) and self.camera_images[self.current_vehicle_index] is not None:
                    display_img = self.camera_images[self.current_vehicle_index].copy()

                    # 添加状态信息
                    cv2.putText(display_img, f"Vehicle: {self.current_vehicle_index + 1}/{len(self.vehicles)}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Speed: {speed:.1f} km/h",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    # 获取当前车辆的控制指令
                    throttle, brake, steer, reverse = current_controller.get_control()
                    cv2.putText(display_img, f"Throttle: {throttle:.2f}",
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Steer: {steer:.2f}",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    cv2.putText(display_img, f"Frame: {frame_count}",
                                (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)
                    
                    # 显示倒车状态
                    if current_controller.manual_reverse:
                        cv2.putText(display_img, "REVERSE MODE",
                                    (20, 240), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 2)  # 红色显示

                    cv2.imshow('Autonomous Driving - Simple Version', display_img)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("正在退出...")
                    running = False
                elif key == ord('r'):
                    self.reset_vehicle(self.current_vehicle_index)
                elif key == ord('s'):
                    # 紧急停止当前车辆
                    current_vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, brake=1.0, hand_brake=True
                    ))
                    print(f"紧急停止车辆 {self.current_vehicle_index + 1}")
                elif key == ord('x'):
                    # 切换当前车辆的倒车模式（只在速度接近0时允许切换）
                    if speed < 1.0:  # 速度小于1km/h时允许切换
                        current_controller.toggle_reverse()
                    else:
                        print("请先减速到接近停止（速度<1km/h）再切换倒车模式")
                elif key == ord('1'):
                    # 切换到第一辆车
                    if len(self.vehicles) >= 1:
                        self.current_vehicle_index = 0
                        print("切换到车辆 1")
                elif key == ord('2'):
                    # 切换到第二辆车
                    if len(self.vehicles) >= 2:
                        self.current_vehicle_index = 1
                        print("切换到车辆 2")
                elif key == ord('3'):
                    # 切换到第三辆车
                    if len(self.vehicles) >= 3:
                        self.current_vehicle_index = 2
                        print("切换到车辆 3")

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

    def reset_vehicle(self, index):
        """重置指定车辆位置"""
        if 0 <= index < len(self.vehicles):
            print(f"重置车辆 {index + 1}...")
            spawn_points = self.world.get_map().get_spawn_points()
            if spawn_points:
                new_spawn_point = random.choice(spawn_points)
                self.vehicles[index].set_transform(new_spawn_point)
                print(f"车辆 {index + 1} 已重置到新位置: {new_spawn_point.location}")

                # 等待重置完成
                time.sleep(0.5)
        else:
            print(f"车辆索引 {index} 无效")

    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")

        # 清理所有相机
        for camera in self.cameras:
            if camera:
                try:
                    camera.stop()
                    camera.destroy()
                except:
                    pass

        # 清理所有车辆
        for vehicle in self.vehicles:
            if vehicle:
                try:
                    vehicle.destroy()
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