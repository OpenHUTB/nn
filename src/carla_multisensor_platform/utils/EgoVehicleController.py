import math
import pygame
import queue
import cv2
import carla
import logging

class EgoVehicleController:
    def __init__(self) -> None:
        self.controller = None
        self.cruise_speed = 30.0  # 巡航速度 km/h
        self.cruise_enabled = False
    
    def toggle_cruise(self):
        """切换定速巡航"""
        self.cruise_enabled = not self.cruise_enabled
        status = "开启" if self.cruise_enabled else "关闭"
        print(f"定速巡航 {status}")
    
    def setup_ego_vehicle(self, ego_vehicle):
        """配置主车辆初始控制参数"""
        self.controller = carla.VehicleControl()
        self.controller.throttle = 0.5
        self.controller.steer = 0.0
        ego_vehicle.apply_control(self.controller)
        return self.controller

    def update_ego_vehicle(self, ego_vehicle, control, obstacle_distance=None):
        """
        更新车辆运动状态（带碰撞避免）
        
        Args:
          ego_vehicle: CARLA主车辆对象
          control: carla.VehicleControl 控制对象
          obstacle_distance: 前方障碍物距离（米），None表示无检测
            - 使用示例：
              # 从传感器获取距离
              distance = lidar_sensor.get_distance()
              controller.update_ego_vehicle(vehicle, control, distance)
            - 阈值说明：
              < 3.0m: 紧急刹车
              3.0-6.0m: 减速慢行  
              6.0-10.0m: 轻微减速
              > 10.0m: 正常巡航或手动控制
    
        Returns:
          None (直接应用控制到车辆)
        """
        # 获取当前速度和位置
        transform = ego_vehicle.get_transform()
        velocity = ego_vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

        # ========== 自动避障逻辑 ==========
        if obstacle_distance is not None:
            if obstacle_distance < 3.0:
                # 紧急刹车
                control.throttle = 0.0
                control.brake = 1.0
                print(f"⚠️ 紧急刹车！距离障碍物 {obstacle_distance:.1f} 米")
                ego_vehicle.apply_control(control)
                return
            elif obstacle_distance < 6.0:
                # 减速慢行
                control.throttle = 0.2
                control.brake = 0.3
                print(f"⚠️ 减速慢行，距离障碍物 {obstacle_distance:.1f} 米")
                ego_vehicle.apply_control(control)
                return
            elif obstacle_distance < 10.0:
                # 轻微减速
                control.throttle = 0.35
                control.brake = 0.1
                print(f"⚠️ 注意前方，距离障碍物 {obstacle_distance:.1f} 米")
                ego_vehicle.apply_control(control)
                return

        # ========== 定速巡航模式 ==========
        if self.cruise_enabled:
            speed_error = self.cruise_speed - speed
            if speed_error > 0:
                control.throttle = min(0.5, 0.1 + speed_error / 100)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = min(0.5, abs(speed_error) / 100)
        else:
            # ========== 原有速度控制 ==========
            if speed < 30.0:
                control.throttle = 0.5
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = 0.1

        # ========== 车道保持逻辑 ==========
        waypoint = ego_vehicle.get_world().get_map().get_waypoint(transform.location)
        if waypoint:
            next_waypoint = waypoint.next(5.0)[0]
            if next_waypoint:
                next_location = next_waypoint.transform.location
                angle = math.atan2(next_location.y - transform.location.y,
                                next_location.x - transform.location.x)
                angle = math.degrees(angle) - transform.rotation.yaw
                angle = (angle + 180) % 360 - 180
                control.steer = max(-0.5, min(0.5, angle / 90.0))

        ego_vehicle.apply_control(control)


class KeyboardController:
    def get_obstacle_distance_from_sensors(ego_vehicle):
      """
        从车辆传感器获取最近障碍物距离（示例实现）
    
        使用方式：
          # 在主循环中调用
          distance = get_obstacle_distance_from_sensors(ego_vehicle)
          controller.update_ego_vehicle(ego_vehicle, control, distance)
    
        Args:
          ego_vehicle: CARLA车辆对象
         
        Returns:
          float: 最近障碍物的距离（米），如果没有检测到障碍物则返回None
      """
      try:
        # 方法1：使用碰撞传感器（需要已添加）
        # collision_sensor = ego_vehicle.get_sensor_by_type(carla.SensorType.COLLISION)
        
        # 方法2：使用雷达/LiDAR（示例）
        # for sensor in ego_vehicle.get_sensors():
        #     if sensor.type_id.startswith('sensor.lidar'):
        #         points = sensor.get_point_cloud()
        #         if points:
        #             min_distance = min(p.distance for p in points)
        #             return min_distance
        
        # 方法3：简易模拟（用于测试，实际使用时删除）
        import random
        return random.choice([None, 2.5, 5.0, 8.0, 15.0])
        
      except Exception as e:
        print(f"传感器读取失败: {e}")
        return None
    def __init__(self):
        self.controller = carla.VehicleControl()
        self.is_reverse = False
        
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("CARLA Keyboard Controller Mode")
        self.screen = pygame.display.set_mode((320, 240))
        self.clock = pygame.time.Clock()
        
    def update(self, keys):
        self.controller.throttle = 0.0
        self.controller.brake = 0.0
        self.controller.steer = 0.0

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.controller.throttle = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.controller.brake = 1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.controller.steer = -0.5
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.controller.steer = 0.5
        if keys[pygame.K_SPACE]:
            self.controller.hand_brake = True
        else:
            self.controller.hand_brake = False
        if keys[pygame.K_r]:
            self.is_reverse = not self.is_reverse

        self.controller.reverse = self.is_reverse

    def run(self, ego_vehicle, image_queue):
        cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
        self.ego_vehicle = ego_vehicle
        try:
            while True:
                self.ego_vehicle.get_world().tick()
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                self.update(keys)
                self.ego_vehicle.apply_control(self.controller)

                self.draw_keyboard_state(self.screen, keys)
                pygame.display.flip()

                try:
                    image_frame = image_queue.get(timeout=1.0)
                    cv2.imshow('Camera', image_frame[1])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return
                except queue.Empty:
                    print('No camera image received.')

                self.clock.tick(60)
        except Exception as e:
            logging.error(e)

        finally:
            cv2.destroyAllWindows()
            pygame.quit()
            print('Keyboard controller mode exited.')
            
    def draw_keyboard_state(self, screen, keys):
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        GRAY = (180, 180, 180)
        BLACK = (0, 0, 0)

        font = pygame.font.SysFont(None, 30)
        screen.fill(BLACK)

        key_map = {
            "UP": (160, 50),
            "LEFT": (100, 100),
            "DOWN": (160, 100),
            "RIGHT": (220, 100),
            "SPACE": (100, 170),
            "REVERSE": (220, 170)
        }

        for key, pos in key_map.items():
            if key == "UP":
                pressed = keys[pygame.K_UP] or keys[pygame.K_w]
            elif key == "DOWN":
                pressed = keys[pygame.K_DOWN] or keys[pygame.K_s]
            elif key == "LEFT":
                pressed = keys[pygame.K_LEFT] or keys[pygame.K_a]
            elif key == "RIGHT":
                pressed = keys[pygame.K_RIGHT] or keys[pygame.K_d]
            elif key == "SPACE":
                pressed = keys[pygame.K_SPACE]
            elif key == "REVERSE":
                pressed = keys[pygame.K_r]
            else:
                pressed = False

            color = GREEN if pressed else GRAY
            rect = pygame.Rect(pos[0], pos[1], 50, 40)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, WHITE, rect, 2)
            label = font.render(key, True, WHITE)
            screen.blit(label, (pos[0] + 5, pos[1] + 10))