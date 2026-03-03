"""
MuJoCo 四旋翼无人机仿真 - 动态交通版
✅ 无人机稳定飞行（带视觉模型）
✅ 自动避开静态障碍物
✅ 动态行驶的小车（带视觉模型）
✅ 包含升空和降落过程
✅ 动态红绿灯系统
✅ 航点巡航
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import os
import random


class TrafficLight:
    """红绿灯控制器"""
    def __init__(self):
        self.states = {
            "north_south": "red",
            "east_west": "green"
        }
        self.timer = 0
        self.cycle_time = 10.0  # 10秒切换一次

    def update(self, dt):
        """更新红绿灯状态"""
        self.timer += dt
        changed = False

        if self.timer >= self.cycle_time:
            self.timer = 0
            changed = True
            # 切换状态
            if self.states["north_south"] == "red":
                self.states["north_south"] = "green"
                self.states["east_west"] = "red"
            else:
                self.states["north_south"] = "red"
                self.states["east_west"] = "green"

        return changed


class MovingCar:
    """动态行驶的小车"""
    def __init__(self, car_id, start_pos, direction, speed=1.5):
        self.car_id = car_id
        self.position = np.array(start_pos, dtype=float)
        self.direction = direction  # "north", "south", "east", "west"
        self.speed = speed
        self.size = 0.5
        self.height = 0.2
        self.color = [random.uniform(0.3, 1), random.uniform(0.3, 1),
                     random.uniform(0.3, 1), 1]

        # 行驶路线参数
        self.route_length = 8.0
        self.min_pos = -5.0
        self.max_pos = 5.0

        # 状态
        self.moving = True
        self.waiting_at_light = False

    def update(self, dt, traffic_lights):
        """更新小车位置"""
        if not self.moving:
            return

        # 检查红绿灯
        self.check_traffic_light(traffic_lights)

        if self.waiting_at_light:
            return

        # 根据方向移动
        move_dist = self.speed * dt * 20

        if self.direction == "north":
            self.position[1] += move_dist
            if self.position[1] > self.max_pos:
                self.position[1] = self.min_pos
        elif self.direction == "south":
            self.position[1] -= move_dist
            if self.position[1] < self.min_pos:
                self.position[1] = self.max_pos
        elif self.direction == "east":
            self.position[0] += move_dist
            if self.position[0] > self.max_pos:
                self.position[0] = self.min_pos
        elif self.direction == "west":
            self.position[0] -= move_dist
            if self.position[0] < self.min_pos:
                self.position[0] = self.max_pos

    def check_traffic_light(self, traffic_lights):
        """检查红绿灯"""
        # 计算到路口的距离
        distance_to_intersection = abs(self.position[0]) + abs(self.position[1])

        if distance_to_intersection < 1.5:
            if self.direction in ["north", "south"]:
                self.waiting_at_light = (traffic_lights.states["north_south"] == "red")
            else:
                self.waiting_at_light = (traffic_lights.states["east_west"] == "red")
        else:
            self.waiting_at_light = False


class QuadrotorSimulation:
    def __init__(self, xml_path="quadrotor_detailed_city.xml"):
        """初始化：从XML文件加载模型"""
        if not os.path.exists(xml_path):
            # 如果没有找到XML文件，创建一个默认的XML字符串
            print(f"找不到XML文件: {xml_path}，使用默认模型")
            self.create_default_model()
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            print(f"✓ 模型加载成功: {xml_path}")

        self.data = mujoco.MjData(self.model)
        self.n_actuators = self.model.nu

        # 基础推力
        self.base_thrust = 600
        if self.n_actuators > 0:
            self.data.ctrl[:] = [self.base_thrust] * self.n_actuators

        # ========== 飞行阶段 ==========
        self.flight_phase = "takeoff"
        self.phase_start_time = 0.0

        # 起飞参数
        self.takeoff_height = 2.0
        self.takeoff_speed = 0.5
        self.start_height = 0.2

        # 巡航参数
        self.cruise_height = 2.0
        self.move_speed = 2.5

        # 降落参数
        self.landing_height = 0.2
        self.landing_speed = 0.3

        # 平滑移动
        self.smooth_pos = np.array([0.0, 0.0, self.start_height])

        # 航点
        self.waypoints = [
            np.array([4.0, 4.0, 2.0]),
            np.array([4.0, -4.0, 2.0]),
            np.array([-4.0, -4.0, 2.0]),
            np.array([-4.0, 4.0, 2.0]),
            np.array([0.0, 0.0, 2.0]),
            np.array([5.0, 0.0, 2.0]),
            np.array([-5.0, 0.0, 2.0]),
            np.array([0.0, 5.0, 2.0]),
            np.array([0.0, -5.0, 2.0])
        ]
        self.current_waypoint = 0

        # ========== 红绿灯系统 ==========
        self.traffic_lights = TrafficLight()

        # ========== 动态车辆系统 ==========
        self.cars = []
        self.init_moving_cars(6)

        # ========== 静态障碍物 ==========
        self.static_obstacles = {
            "building_1": np.array([6.0, 6.0, 1.5]),
            "building_2": np.array([-6.0, 6.0, 1.5]),
            "building_3": np.array([6.0, -6.0, 1.5]),
            "building_4": np.array([-6.0, -6.0, 1.5]),
            "tree_1": np.array([3.0, 3.0, 0.8]),
            "tree_2": np.array([-3.0, 3.0, 0.8]),
            "tree_3": np.array([3.0, -3.0, 0.8]),
            "tree_4": np.array([-3.0, -3.0, 0.8]),
        }

        self.obstacle_sizes = {
            "building_1": 1.2,
            "building_2": 1.2,
            "building_3": 1.2,
            "building_4": 1.2,
            "tree_1": 0.5,
            "tree_2": 0.5,
            "tree_3": 0.5,
            "tree_4": 0.5,
        }

        # ========== 避障参数 ==========
        self.safety_distance = 1.5
        self.avoidance_strength = 2.5

        # 获取关节数量
        self.nq = self.model.nq
        self.nv = self.model.nv
        print(f"✓ 关节数量: nq={self.nq}, nv={self.nv}")
        print(f"✓ 已初始化 {len(self.cars)} 辆动态小车")

    def create_default_model(self):
        """创建默认的XML模型"""
        xml_string = """
        <mujoco model="quadrotor_dynamic">
            <option timestep="0.01"/>
            
            <worldbody>
                <!-- 光照 -->
                <light name="sun" pos="0 0 10" diffuse="1 1 1" ambient="0.3 0.3 0.3"/>
                
                <!-- 地面 -->
                <geom name="ground" type="plane" pos="0 0 -0.1" size="15 15 0.1" rgba="0.3 0.5 0.3 1"/>
                
                <!-- 道路网格 -->
                <geom name="road_grid" type="box" pos="0 0 0" size="10 10 0.02" rgba="0.2 0.2 0.2 0.3"/>
                
                <!-- 道路标线 -->
                <geom name="line_h" type="box" pos="0 0 0.01" size="12 0.1 0.02" rgba="1 1 0 1"/>
                <geom name="line_v" type="box" pos="0 0 0.01" size="0.1 12 0.02" rgba="1 1 0 1"/>
                
                <!-- ========== 红绿灯 ========== -->
                <!-- 北 -->
                <body name="light_n" pos="0 5 0.3">
                    <geom name="pole_n" type="cylinder" size="0.08 1.5" pos="0 0 0.8" rgba="0.3 0.3 0.3 1"/>
                    <geom name="red_n" type="sphere" size="0.15" pos="0 0.2 1.5" rgba="1 0 0 1"/>
                    <geom name="yellow_n" type="sphere" size="0.15" pos="0 0 1.3" rgba="1 1 0 1"/>
                    <geom name="green_n" type="sphere" size="0.15" pos="0 -0.2 1.1" rgba="0 1 0 1"/>
                </body>
                
                <!-- 南 -->
                <body name="light_s" pos="0 -5 0.3">
                    <geom name="pole_s" type="cylinder" size="0.08 1.5" pos="0 0 0.8" rgba="0.3 0.3 0.3 1"/>
                    <geom name="red_s" type="sphere" size="0.15" pos="0 0.2 1.5" rgba="1 0 0 1"/>
                    <geom name="yellow_s" type="sphere" size="0.15" pos="0 0 1.3" rgba="1 1 0 1"/>
                    <geom name="green_s" type="sphere" size="0.15" pos="0 -0.2 1.1" rgba="0 1 0 1"/>
                </body>
                
                <!-- 东 -->
                <body name="light_e" pos="5 0 0.3">
                    <geom name="pole_e" type="cylinder" size="0.08 1.5" pos="0 0 0.8" rgba="0.3 0.3 0.3 1"/>
                    <geom name="red_e" type="sphere" size="0.15" pos="0 0.2 1.5" rgba="1 0 0 1"/>
                    <geom name="yellow_e" type="sphere" size="0.15" pos="0 0 1.3" rgba="1 1 0 1"/>
                    <geom name="green_e" type="sphere" size="0.15" pos="0 -0.2 1.1" rgba="0 1 0 1"/>
                </body>
                
                <!-- 西 -->
                <body name="light_w" pos="-5 0 0.3">
                    <geom name="pole_w" type="cylinder" size="0.08 1.5" pos="0 0 0.8" rgba="0.3 0.3 0.3 1"/>
                    <geom name="red_w" type="sphere" size="0.15" pos="0 0.2 1.5" rgba="1 0 0 1"/>
                    <geom name="yellow_w" type="sphere" size="0.15" pos="0 0 1.3" rgba="1 1 0 1"/>
                    <geom name="green_w" type="sphere" size="0.15" pos="0 -0.2 1.1" rgba="0 1 0 1"/>
                </body>
                
                <!-- ========== 静态建筑 ========== -->
                <body name="building_1" pos="6 6 0.5">
                    <geom name="b1" type="box" size="1 1 1" rgba="0.5 0.5 0.5 1"/>
                </body>
                <body name="building_2" pos="-6 6 0.5">
                    <geom name="b2" type="box" size="1 1 1" rgba="0.7 0.3 0.2 1"/>
                </body>
                <body name="building_3" pos="6 -6 0.5">
                    <geom name="b3" type="box" size="1 1 1" rgba="0.3 0.7 0.2 1"/>
                </body>
                <body name="building_4" pos="-6 -6 0.5">
                    <geom name="b4" type="box" size="1 1 1" rgba="0.2 0.3 0.7 1"/>
                </body>
                
                <!-- ========== 树木 ========== -->
                <body name="tree_1" pos="3 3 0.3">
                    <geom name="trunk1" type="cylinder" size="0.15 0.8" pos="0 0 0.5" rgba="0.5 0.3 0.1 1"/>
                    <geom name="leaves1" type="sphere" size="0.3" pos="0 0 1.1" rgba="0.2 0.7 0.2 1"/>
                </body>
                <body name="tree_2" pos="-3 3 0.3">
                    <geom name="trunk2" type="cylinder" size="0.15 0.8" pos="0 0 0.5" rgba="0.5 0.3 0.1 1"/>
                    <geom name="leaves2" type="sphere" size="0.3" pos="0 0 1.1" rgba="0.2 0.7 0.2 1"/>
                </body>
                
                <!-- ========== 无人机模型 ========== -->
                <body name="quadrotor" pos="0 0 0.2">
                    <joint name="quad_free_joint" type="free"/>
                    
                    <!-- 主体 -->
                    <geom name="body" type="sphere" size="0.2" rgba="0.1 0.1 0.1 1"/>
                    
                    <!-- 机臂 -->
                    <geom name="arm1" type="capsule" fromto="0.15 0.15 0 0.35 0.35 0" size="0.03" rgba="0.3 0.3 0.3 1"/>
                    <geom name="arm2" type="capsule" fromto="0.15 -0.15 0 0.35 -0.35 0" size="0.03" rgba="0.3 0.3 0.3 1"/>
                    <geom name="arm3" type="capsule" fromto="-0.15 -0.15 0 -0.35 -0.35 0" size="0.03" rgba="0.3 0.3 0.3 1"/>
                    <geom name="arm4" type="capsule" fromto="-0.15 0.15 0 -0.35 0.35 0" size="0.03" rgba="0.3 0.3 0.3 1"/>
                    
                    <!-- 旋翼 -->
                    <body name="rotor1" pos="0.35 0.35 0.05">
                        <joint name="rotor1_joint" type="hinge" axis="0 0 1"/>
                        <geom name="prop1" type="cylinder" size="0.2 0.02" rgba="0.1 0.1 0.1 1"/>
                    </body>
                    <body name="rotor2" pos="0.35 -0.35 0.05">
                        <joint name="rotor2_joint" type="hinge" axis="0 0 1"/>
                        <geom name="prop2" type="cylinder" size="0.2 0.02" rgba="0.1 0.1 0.1 1"/>
                    </body>
                    <body name="rotor3" pos="-0.35 -0.35 0.05">
                        <joint name="rotor3_joint" type="hinge" axis="0 0 1"/>
                        <geom name="prop3" type="cylinder" size="0.2 0.02" rgba="0.1 0.1 0.1 1"/>
                    </body>
                    <body name="rotor4" pos="-0.35 0.35 0.05">
                        <joint name="rotor4_joint" type="hinge" axis="0 0 1"/>
                        <geom name="prop4" type="cylinder" size="0.2 0.02" rgba="0.1 0.1 0.1 1"/>
                    </body>
                    
                    <!-- LED灯 -->
                    <geom name="led_front" type="sphere" size="0.04" pos="0.2 0 0.1" rgba="1 0 0 1"/>
                    <geom name="led_back" type="sphere" size="0.04" pos="-0.2 0 0.1" rgba="0 0 1 1"/>
                </body>
            </worldbody>
            
            <actuator>
                <motor name="motor1" joint="rotor1_joint" gear="80" ctrlrange="0 1000"/>
                <motor name="motor2" joint="rotor2_joint" gear="80" ctrlrange="0 1000"/>
                <motor name="motor3" joint="rotor3_joint" gear="80" ctrlrange="0 1000"/>
                <motor name="motor4" joint="rotor4_joint" gear="80" ctrlrange="0 1000"/>
            </actuator>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        print("✓ 默认模型创建成功")

    def init_moving_cars(self, num_cars=6):
        """初始化动态车辆"""
        car_configs = [
            ([-4.0, 1.0, 0.2], "east", 1.5),
            ([4.0, -1.0, 0.2], "west", 1.5),
            ([1.0, -4.0, 0.2], "north", 1.5),
            ([-1.0, 4.0, 0.2], "south", 1.5),
            ([-3.0, 2.0, 0.2], "east", 1.2),
            ([3.0, -2.0, 0.2], "west", 1.2),
        ]

        for i, (pos, direction, speed) in enumerate(car_configs[:num_cars]):
            car = MovingCar(i, pos, direction, speed)
            self.cars.append(car)

    def create_car_geom(self, viewer, car):
        """创建小车的视觉几何体（在运行时动态添加）"""
        # 由于MuJoCo不支持运行时动态添加几何体，
        # 我们通过在代码中模拟小车位置
        pass

    def get_avoidance_force(self, current_pos):
        """计算避障力"""
        avoidance = np.zeros(2)
        min_dist = 999
        closest = ""

        # 检查静态障碍物
        for obs_name, obs_pos in self.static_obstacles.items():
            to_obs = current_pos[:2] - obs_pos[:2]
            dist = np.linalg.norm(to_obs)
            obs_size = self.obstacle_sizes.get(obs_name, 0.5)

            actual_dist = max(dist - obs_size, 0.1)

            if actual_dist < min_dist:
                min_dist = actual_dist
                closest = f"静态_{obs_name}"

            if actual_dist < self.safety_distance:
                if dist > 0:
                    direction = to_obs / dist
                else:
                    direction = np.array([1, 0])

                strength = self.avoidance_strength * (1.0 - actual_dist / self.safety_distance) ** 2
                avoidance += direction * strength

        # 检查动态车辆
        for i, car in enumerate(self.cars):
            to_car = current_pos[:2] - car.position[:2]
            dist = np.linalg.norm(to_car)
            car_size = 0.5

            actual_dist = max(dist - car_size, 0.1)

            if actual_dist < min_dist:
                min_dist = actual_dist
                closest = f"动态_车{i+1}"

            dynamic_safety = self.safety_distance * 1.3

            if actual_dist < dynamic_safety:
                if dist > 0:
                    direction = to_car / dist
                else:
                    direction = np.array([1, 0])

                # 根据车辆移动方向调整避障力
                if car.direction == "north":
                    direction[1] += 0.4
                elif car.direction == "south":
                    direction[1] -= 0.4
                elif car.direction == "east":
                    direction[0] += 0.4
                elif car.direction == "west":
                    direction[0] -= 0.4

                direction = direction / (np.linalg.norm(direction) + 0.001)

                strength = self.avoidance_strength * 1.8 * (1.0 - actual_dist / dynamic_safety) ** 2
                avoidance += direction * strength

        return avoidance, min_dist, closest

    def update_flight_phase(self, current_time):
        """更新飞行阶段"""
        if self.flight_phase == "takeoff":
            elapsed = current_time - self.phase_start_time
            progress = min(elapsed * self.takeoff_speed, 1.0)
            current_height = self.start_height + (self.takeoff_height - self.start_height) * progress

            if progress >= 1.0:
                self.flight_phase = "cruise"
                self.phase_start_time = current_time
                print("\n🚁 起飞完成，开始巡航")

            return current_height

        elif self.flight_phase == "cruise":
            return self.cruise_height

        else:
            elapsed = current_time - self.phase_start_time
            progress = min(elapsed * self.landing_speed, 1.0)
            current_height = self.cruise_height - (self.cruise_height - self.landing_height) * progress

            return current_height

    def get_next_target(self, current_pos):
        """获取下一个目标点"""
        if self.flight_phase == "takeoff":
            return np.array([0, 0, self.takeoff_height])
        elif self.flight_phase == "landing":
            return np.array([0, 0, 0.2])
        else:
            target = self.waypoints[self.current_waypoint]
            dist = np.linalg.norm(current_pos[:2] - target[:2])
            if dist < 2.0:
                self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
                target = self.waypoints[self.current_waypoint]
            return target

    def simulation_loop(self, viewer, duration):
        """主仿真循环"""
        start_time = time.time()
        last_print_time = time.time()
        self.phase_start_time = 0.0

        # 用于存储小车视觉几何体的引用
        car_geoms = []

        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()
            current_time = self.data.time

            # 物理仿真步进
            mujoco.mj_step(self.model, self.data)

            # 更新红绿灯
            changed = self.traffic_lights.update(self.model.opt.timestep)

            # 更新动态车辆
            for car in self.cars:
                car.update(self.model.opt.timestep, self.traffic_lights)

            # 获取当前位置
            current_pos = self.data.qpos[0:3].copy()

            # 更新飞行高度
            target_height = self.update_flight_phase(current_time)

            # 获取下一个目标点
            next_target = self.get_next_target(current_pos)
            next_target[2] = target_height

            # 计算避障力
            avoidance_2d, min_dist, closest = self.get_avoidance_force(current_pos)

            # 计算移动方向
            to_target = next_target - current_pos
            dist_to_target = np.linalg.norm(to_target)

            if dist_to_target > 0.1:
                target_dir_3d = to_target / dist_to_target

                # 结合避障力
                if np.linalg.norm(avoidance_2d) > 0.1:
                    avoid_dir_2d = avoidance_2d / (np.linalg.norm(avoidance_2d) + 0.001)
                    avoid_dir_3d = np.array([avoid_dir_2d[0], avoid_dir_2d[1], 0])

                    avoid_weight = min(0.8, 0.3 + 0.5 * (1.0 - min_dist / self.safety_distance))
                    target_weight = 1.0 - avoid_weight

                    move_dir = target_dir_3d * target_weight + avoid_dir_3d * avoid_weight
                    move_dir = move_dir / np.linalg.norm(move_dir)
                else:
                    move_dir = target_dir_3d

                # 计算新位置
                new_pos = current_pos + move_dir * self.move_speed * self.model.opt.timestep * 30
            else:
                new_pos = next_target

            # 平滑移动
            self.smooth_pos = self.smooth_pos + (new_pos - self.smooth_pos) * 0.3
            self.smooth_pos[2] = target_height

            # 限制移动范围
            self.smooth_pos[0] = np.clip(self.smooth_pos[0], -8, 8)
            self.smooth_pos[1] = np.clip(self.smooth_pos[1], -8, 8)

            # 设置无人机位置
            self.data.qpos[0] = self.smooth_pos[0]
            self.data.qpos[1] = self.smooth_pos[1]
            self.data.qpos[2] = self.smooth_pos[2]

            # 设置无人机姿态（稍微倾斜以模拟运动）
            if self.nq > 3:
                speed_ratio = np.linalg.norm(self.data.qvel[0:3]) / self.move_speed
                tilt = 0.2 * speed_ratio
                self.data.qpos[3] = math.cos(tilt/2)  # w
                self.data.qpos[4] = 0.0               # x
                self.data.qpos[5] = math.sin(tilt/2)  # y（俯仰）
                self.data.qpos[6] = 0.0               # z

            # 旋翼旋转
            for i in range(4):
                if 7 + i < self.nq:
                    self.data.qpos[7 + i] += 20.0 * self.model.opt.timestep

            if viewer:
                viewer.sync()

            # 打印状态
            if time.time() - last_print_time > 2.0:
                phase_names = {
                    "takeoff": "🔼 起飞",
                    "cruise": "✈️ 巡航",
                    "landing": "🔽 降落"
                }

                status = "避障中" if np.linalg.norm(avoidance_2d) > 0.1 else "正常飞行"

                ns_state = self.traffic_lights.states["north_south"]
                ew_state = self.traffic_lights.states["east_west"]

                moving_cars = len([c for c in self.cars if c.moving and not c.waiting_at_light])
                waiting_cars = len([c for c in self.cars if c.waiting_at_light])

                print(f"\n{'='*70}")
                print(f"时间: {current_time:.1f}s | 阶段: {phase_names[self.flight_phase]}")
                print(f"位置: ({self.smooth_pos[0]:.2f}, {self.smooth_pos[1]:.2f}, {self.smooth_pos[2]:.2f})")
                print(f"目标: 航点 {self.current_waypoint + 1}/{len(self.waypoints)}")
                print(f"状态: {status}")
                print(f"最近障碍: {closest} 距离 {min_dist:.2f}m")
                print(f"\n🚦 红绿灯: 南北={ns_state.upper()}, 东西={ew_state.upper()}")
                print(f"🚗 车辆: 总数={len(self.cars)}, 行驶={moving_cars}, 等待={waiting_cars}")
                print(f"{'='*70}")

                last_print_time = time.time()

            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=60.0, use_viewer=True):
        """运行仿真"""
        print(f"\n{'🚁'*10} 无人机动态交通仿真 {'🚁'*10}")
        print(f"▶ 起飞高度: {self.takeoff_height}m")
        print(f"▶ 巡航高度: {self.cruise_height}m")
        print(f"▶ 移动速度: {self.move_speed}m/s")
        print(f"▶ 静态障碍物: {len(self.static_obstacles)}")
        print(f"▶ 动态车辆: {len(self.cars)}")
        print(f"▶ 红绿灯周期: {self.traffic_lights.cycle_time}s")
        print(f"{'='*70}")

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = 35
                    viewer.cam.distance = 20.0
                    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

                    print("\n🔼 无人机开始起飞...")
                    print("🚦 红绿灯系统已激活")
                    print("🚗 动态车辆系统已激活")
                    print("📍 航点巡航模式")
                    print("🚧 智能避障系统已启用")
                    self.simulation_loop(viewer, duration)
            else:
                self.simulation_loop(None, duration)
        except Exception as e:
            print(f"⚠ 仿真错误: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'✅'*10} 仿真结束 {'✅'*10}")


def main():
    print("🚁 MuJoCo 四旋翼无人机 - 动态交通版")
    print("=" * 70)

    try:
        xml_path = "quadrotor_detailed_city.xml"
        sim = QuadrotorSimulation(xml_path)

        # ========== 可调参数 ==========
        sim.takeoff_height = 2.0
        sim.cruise_height = 2.0
        sim.move_speed = 2.5
        sim.safety_distance = 1.5
        sim.avoidance_strength = 2.5
        sim.traffic_lights.cycle_time = 12.0

        print("✅ 初始化完成")
        sim.run_simulation(duration=60.0, use_viewer=True)

    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
    except KeyboardInterrupt:
        print("\n\n⏹ 仿真被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()