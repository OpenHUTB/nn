"""
MuJoCo 四旋翼无人机仿真 - 手动控制版
✅ 手动键盘控制无人机
✅ 动态行驶的小车
✅ 红绿灯系统
✅ 静态障碍物
✅ 实时状态显示
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import os
import random
from pynput import keyboard


class ManualControl:
    """手动控制系统"""
    def __init__(self):
        # 控制状态
        self.forward = 0.0
        self.backward = 0.0
        self.left = 0.0
        self.right = 0.0
        self.up = 0.0
        self.down = 0.0
        self.yaw_left = 0.0
        self.yaw_right = 0.0

        # 速度设置
        self.speed = 2.0
        self.yaw_speed = 1.0

        # 控制模式
        self.mode = "position"  # position 或 velocity

        # 键盘监听
        self.listener = None
        self.running = True

        # 控制说明
        self.instructions = """
        🎮 手动控制系统
        ================
        W/S : 前进/后退
        A/D : 左移/右移
        ↑/↓ : 上升/下降
        ←/→ : 左转/右转
        Space : 急停
        M : 切换控制模式
        ESC : 退出
        
        当前模式: POSITION
        """

    def start(self):
        """启动键盘监听"""
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
        print(self.instructions)

    def stop(self):
        """停止键盘监听"""
        self.running = False
        if self.listener:
            self.listener.stop()

    def on_press(self, key):
        """按键按下处理"""
        try:
            # 字母键
            if hasattr(key, 'char') and key.char:
                if key.char == 'w' or key.char == 'W':
                    self.forward = 1.0
                elif key.char == 's' or key.char == 'S':
                    self.backward = 1.0
                elif key.char == 'a' or key.char == 'A':
                    self.left = 1.0
                elif key.char == 'd' or key.char == 'D':
                    self.right = 1.0
                elif key.char == 'm' or key.char == 'M':
                    self.toggle_mode()

            # 特殊键
            elif key == keyboard.Key.up:
                self.up = 1.0
            elif key == keyboard.Key.down:
                self.down = 1.0
            elif key == keyboard.Key.left:
                self.yaw_left = 1.0
            elif key == keyboard.Key.right:
                self.yaw_right = 1.0
            elif key == keyboard.Key.space:
                self.emergency_stop()
            elif key == keyboard.Key.esc:
                return False  # 停止监听

        except AttributeError:
            pass

    def on_release(self, key):
        """按键释放处理"""
        try:
            if hasattr(key, 'char') and key.char:
                if key.char == 'w' or key.char == 'W':
                    self.forward = 0.0
                elif key.char == 's' or key.char == 'S':
                    self.backward = 0.0
                elif key.char == 'a' or key.char == 'A':
                    self.left = 0.0
                elif key.char == 'd' or key.char == 'D':
                    self.right = 0.0

            elif key == keyboard.Key.up:
                self.up = 0.0
            elif key == keyboard.Key.down:
                self.down = 0.0
            elif key == keyboard.Key.left:
                self.yaw_left = 0.0
            elif key == keyboard.Key.right:
                self.yaw_right = 0.0

        except AttributeError:
            pass

    def toggle_mode(self):
        """切换控制模式"""
        if self.mode == "position":
            self.mode = "velocity"
            print("\n🔄 切换到速度模式")
        else:
            self.mode = "position"
            print("\n🔄 切换到位置模式")

    def emergency_stop(self):
        """急停"""
        self.forward = 0.0
        self.backward = 0.0
        self.left = 0.0
        self.right = 0.0
        self.up = 0.0
        self.down = 0.0
        self.yaw_left = 0.0
        self.yaw_right = 0.0
        print("\n🛑 急停！")

    def get_velocity(self):
        """获取速度指令"""
        vx = (self.right - self.left) * self.speed
        vy = (self.forward - self.backward) * self.speed
        vz = (self.up - self.down) * self.speed
        yaw = (self.yaw_right - self.yaw_left) * self.yaw_speed

        return np.array([vx, vy, vz, yaw])


class TrafficLight:
    """红绿灯控制器"""
    def __init__(self):
        self.states = {
            "north_south": "red",
            "east_west": "green"
        }
        self.timer = 0
        self.cycle_time = 10.0

    def update(self, dt):
        self.timer += dt
        changed = False

        if self.timer >= self.cycle_time:
            self.timer = 0
            changed = True
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
        self.direction = direction
        self.speed = speed
        self.size = 0.5
        self.height = 0.2
        self.color = [random.uniform(0.3, 1), random.uniform(0.3, 1),
                     random.uniform(0.3, 1), 1]

        self.min_pos = -6.0
        self.max_pos = 6.0
        self.moving = True
        self.waiting_at_light = False

    def update(self, dt, traffic_lights):
        if not self.moving:
            return

        self.check_traffic_light(traffic_lights)

        if self.waiting_at_light:
            return

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
        dist_to_intersection = abs(self.position[0]) + abs(self.position[1])

        if dist_to_intersection < 1.5:
            if self.direction in ["north", "south"]:
                self.waiting_at_light = (traffic_lights.states["north_south"] == "red")
            else:
                self.waiting_at_light = (traffic_lights.states["east_west"] == "red")
        else:
            self.waiting_at_light = False


class QuadrotorSimulation:
    def __init__(self):
        """初始化"""
        self.create_default_model()
        self.data = mujoco.MjData(self.model)

        # 手动控制系统
        self.manual_control = ManualControl()

        # 无人机状态
        self.drone_pos = np.array([0.0, 0.0, 0.2])
        self.drone_vel = np.zeros(3)
        self.drone_yaw = 0.0

        # 红绿灯系统
        self.traffic_lights = TrafficLight()

        # 动态车辆
        self.cars = []
        self.init_moving_cars(6)

        # 静态障碍物
        self.static_obstacles = {
            "building_1": np.array([6.0, 6.0, 1.5]),
            "building_2": np.array([-6.0, 6.0, 1.5]),
            "building_3": np.array([6.0, -6.0, 1.5]),
            "building_4": np.array([-6.0, -6.0, 1.5]),
            "tree_1": np.array([4.0, 4.0, 0.8]),
            "tree_2": np.array([-4.0, 4.0, 0.8]),
            "tree_3": np.array([4.0, -4.0, 0.8]),
            "tree_4": np.array([-4.0, -4.0, 0.8]),
        }

        self.obstacle_sizes = {
            "building_1": 1.2, "building_2": 1.2, "building_3": 1.2, "building_4": 1.2,
            "tree_1": 0.5, "tree_2": 0.5, "tree_3": 0.5, "tree_4": 0.5,
        }

        # 飞行边界
        self.bounds = {
            'x': [-8, 8],
            'y': [-8, 8],
            'z': [0.2, 3.0]
        }

    def create_default_model(self):
        """创建默认的XML模型 - 修复了actuator错误"""
        xml_string = """
        <mujoco model="quadrotor_manual">
            <option timestep="0.01"/>
            
            <worldbody>
                <light name="sun" pos="0 0 10" diffuse="1 1 1" ambient="0.3 0.3 0.3"/>
                
                <!-- 地面 -->
                <geom name="ground" type="plane" pos="0 0 -0.1" size="15 15 0.1" rgba="0.3 0.5 0.3 1"/>
                
                <!-- 道路 -->
                <geom name="road_h" type="box" pos="0 0 0" size="12 2 0.05" rgba="0.2 0.2 0.2 1"/>
                <geom name="road_v" type="box" pos="0 0 0" size="2 12 0.05" rgba="0.2 0.2 0.2 1"/>
                
                <!-- 道路标线 -->
                <geom name="line_h" type="box" pos="0 0 0.02" size="12 0.1 0.02" rgba="1 1 0 1"/>
                <geom name="line_v" type="box" pos="0 0 0.02" size="0.1 12 0.02" rgba="1 1 0 1"/>
                
                <!-- 红绿灯 -->
                <body name="light_n" pos="0 6 0.3">
                    <geom name="pole_n" type="cylinder" size="0.1 2" pos="0 0 1" rgba="0.3 0.3 0.3 1"/>
                    <geom name="red_n" type="sphere" size="0.2" pos="0 0.25 1.8" rgba="1 0 0 1"/>
                    <geom name="yellow_n" type="sphere" size="0.2" pos="0 0 1.5" rgba="1 1 0 1"/>
                    <geom name="green_n" type="sphere" size="0.2" pos="0 -0.25 1.2" rgba="0 1 0 1"/>
                </body>
                
                <body name="light_s" pos="0 -6 0.3">
                    <geom name="pole_s" type="cylinder" size="0.1 2" pos="0 0 1" rgba="0.3 0.3 0.3 1"/>
                    <geom name="red_s" type="sphere" size="0.2" pos="0 0.25 1.8" rgba="1 0 0 1"/>
                    <geom name="yellow_s" type="sphere" size="0.2" pos="0 0 1.5" rgba="1 1 0 1"/>
                    <geom name="green_s" type="sphere" size="0.2" pos="0 -0.25 1.2" rgba="0 1 0 1"/>
                </body>
                
                <!-- 无人机 -->
                <body name="quadrotor" pos="0 0 0.2">
                    <joint name="quad_free_joint" type="free"/>
                    
                    <!-- 主体 -->
                    <geom name="body" type="sphere" size="0.2" rgba="0.1 0.1 0.1 1"/>
                    
                    <!-- 机臂 -->
                    <geom name="arm1" type="capsule" fromto="0.2 0.2 0 0.4 0.4 0" size="0.03" rgba="0.3 0.3 0.3 1"/>
                    <geom name="arm2" type="capsule" fromto="0.2 -0.2 0 0.4 -0.4 0" size="0.03" rgba="0.3 0.3 0.3 1"/>
                    <geom name="arm3" type="capsule" fromto="-0.2 -0.2 0 -0.4 -0.4 0" size="0.03" rgba="0.3 0.3 0.3 1"/>
                    <geom name="arm4" type="capsule" fromto="-0.2 0.2 0 -0.4 0.4 0" size="0.03" rgba="0.3 0.3 0.3 1"/>
                    
                    <!-- 旋翼 -->
                    <body name="rotor1" pos="0.4 0.4 0.05">
                        <joint name="rotor1_joint" type="hinge" axis="0 0 1"/>
                        <geom name="prop1" type="cylinder" size="0.25 0.02" rgba="0.1 0.1 0.1 1"/>
                    </body>
                    <body name="rotor2" pos="0.4 -0.4 0.05">
                        <joint name="rotor2_joint" type="hinge" axis="0 0 1"/>
                        <geom name="prop2" type="cylinder" size="0.25 0.02" rgba="0.1 0.1 0.1 1"/>
                    </body>
                    <body name="rotor3" pos="-0.4 -0.4 0.05">
                        <joint name="rotor3_joint" type="hinge" axis="0 0 1"/>
                        <geom name="prop3" type="cylinder" size="0.25 0.02" rgba="0.1 0.1 0.1 1"/>
                    </body>
                    <body name="rotor4" pos="-0.4 0.4 0.05">
                        <joint name="rotor4_joint" type="hinge" axis="0 0 1"/>
                        <geom name="prop4" type="cylinder" size="0.25 0.02" rgba="0.1 0.1 0.1 1"/>
                    </body>
                    
                    <!-- LED灯 -->
                    <geom name="led_front" type="sphere" size="0.05" pos="0.25 0 0.1" rgba="1 0 0 1"/>
                    <geom name="led_back" type="sphere" size="0.05" pos="-0.25 0 0.1" rgba="0 0 1 1"/>
                </body>
                
                <!-- 静态建筑 -->
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
            </worldbody>
            
            <actuator>
                <motor name="motor1" joint="rotor1_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
                <motor name="motor2" joint="rotor2_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
                <motor name="motor3" joint="rotor3_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
                <motor name="motor4" joint="rotor4_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
            </actuator>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        print("✓ 模型创建成功")

    def init_moving_cars(self, num_cars=6):
        """初始化动态车辆"""
        car_configs = [
            ([-5.0, 1.5, 0.2], "east", 1.5),
            ([5.0, -1.5, 0.2], "west", 1.5),
            ([1.5, -5.0, 0.2], "north", 1.5),
            ([-1.5, 5.0, 0.2], "south", 1.5),
            ([-4.0, 2.5, 0.2], "east", 1.2),
            ([4.0, -2.5, 0.2], "west", 1.2),
        ]

        for i, (pos, direction, speed) in enumerate(car_configs[:num_cars]):
            car = MovingCar(i, pos, direction, speed)
            self.cars.append(car)

    def check_collision(self, pos):
        """检查是否与障碍物碰撞"""
        for obs_name, obs_pos in self.static_obstacles.items():
            dist = np.linalg.norm(pos[:2] - obs_pos[:2])
            obs_size = self.obstacle_sizes.get(obs_name, 0.5)
            if dist < obs_size + 0.3 and abs(pos[2] - obs_pos[2]) < 1.0:
                return True, obs_name
        return False, None

    def simulation_loop(self, viewer, duration):
        """主仿真循环"""
        start_time = time.time()
        last_print_time = time.time()

        # 启动手动控制
        self.manual_control.start()

        while (viewer and viewer.is_running()) and (time.time() - start_time) < duration:
            step_start = time.time()

            # 物理仿真步进
            mujoco.mj_step(self.model, self.data)

            # 更新红绿灯
            self.traffic_lights.update(self.model.opt.timestep)

            # 更新动态车辆
            for car in self.cars:
                car.update(self.model.opt.timestep, self.traffic_lights)

            # 获取当前无人机位置
            current_pos = self.data.qpos[0:3].copy()

            # 获取手动控制指令
            vel = self.manual_control.get_velocity()

            # 更新位置（速度控制）
            new_pos = current_pos + vel[:3] * self.model.opt.timestep * 50

            # 边界检查
            new_pos[0] = np.clip(new_pos[0], self.bounds['x'][0], self.bounds['x'][1])
            new_pos[1] = np.clip(new_pos[1], self.bounds['y'][0], self.bounds['y'][1])
            new_pos[2] = np.clip(new_pos[2], self.bounds['z'][0], self.bounds['z'][1])

            # 碰撞检测
            collision, obs = self.check_collision(new_pos)
            if collision:
                print(f"\n⚠ 接近障碍物: {obs}")
                # 不更新位置，保持原位置
            else:
                # 更新无人机位置
                self.data.qpos[0] = new_pos[0]
                self.data.qpos[1] = new_pos[1]
                self.data.qpos[2] = new_pos[2]

            # 更新偏航
            self.drone_yaw += vel[3] * self.model.opt.timestep * 20

            # 设置无人机姿态
            self.data.qpos[3] = math.cos(self.drone_yaw / 2)  # w
            self.data.qpos[4] = 0.0                           # x
            self.data.qpos[5] = 0.0                           # y
            self.data.qpos[6] = math.sin(self.drone_yaw / 2)  # z

            # 旋翼旋转
            for i in range(4):
                self.data.qpos[7 + i] += 30.0 * self.model.opt.timestep

            viewer.sync()

            # 打印状态
            if time.time() - last_print_time > 0.5:
                ns_state = self.traffic_lights.states["north_south"]
                ew_state = self.traffic_lights.states["east_west"]

                moving_cars = len([c for c in self.cars if c.moving and not c.waiting_at_light])
                waiting_cars = len([c for c in self.cars if c.waiting_at_light])

                # 清除上一行
                print('\033[2K\033[1G', end='')

                print(f"\r📍 位置: ({self.data.qpos[0]:.1f}, {self.data.qpos[1]:.1f}, {self.data.qpos[2]:.1f}) "
                      f"| 速度: {np.linalg.norm(vel[:3]):.1f} "
                      f"| 🚦 {ns_state.upper()}/{ew_state.upper()} "
                      f"| 🚗 {moving_cars}/{waiting_cars} "
                      f"| 模式: {self.manual_control.mode.upper()}", end='')

                last_print_time = time.time()

            # 控制仿真速率
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=300.0):
        """运行仿真"""
        print(f"\n{'🚁'*10} 无人机手动控制仿真 {'🚁'*10}")
        print(f"▶ 控制方式: 键盘")
        print(f"▶ 飞行边界: X[-8,8] Y[-8,8] Z[0.2,3.0]")
        print(f"▶ 静态障碍物: {len(self.static_obstacles)}")
        print(f"▶ 动态车辆: {len(self.cars)}")
        print(f"{'='*60}")

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.azimuth = -45
                viewer.cam.elevation = 30
                viewer.cam.distance = 15.0
                viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

                print("\n🔄 手动控制已激活")
                print("🎮 使用 WASD 移动，方向键上升/下降/转向")
                print("⏱️ 仿真运行中...")

                self.simulation_loop(viewer, duration)

        except Exception as e:
            print(f"⚠ 仿真错误: {e}")
        finally:
            self.manual_control.stop()


def main():
    print("🚁 MuJoCo 四旋翼无人机 - 手动控制版")
    print("=" * 60)

    try:
        sim = QuadrotorSimulation()
        sim.run_simulation(duration=300.0)

    except KeyboardInterrupt:
        print("\n\n⏹ 仿真被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()