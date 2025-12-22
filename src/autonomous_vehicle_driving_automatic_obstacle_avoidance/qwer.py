# qwer.py
import mujoco
import glfw
import numpy as np
import csv
import os
from datetime import datetime
import imageio
from typing import List

# -------------------------- 全局配置（已优化避障参数）--------------------------
TIME_STEP = 0.01
MAX_STEPS = 2000
NSTEP = 5

# 激光雷达参数
LIDAR_MOUNT_POS = np.array([0.5, 0, 0.2])
LIDAR_MAX_DISTANCE = 8.0
LIDAR_NUM_RAYS = 360

# 无人车参数（优化后，避免碰撞）
MAX_STEER_ANGLE = 0.6  # 增大转向角
MAX_DRIVE_TORQUE = 15.0  # 增大驱动力矩
SAFE_SPEED = 1.8  # 提高直行速度
AVOID_SPEED = 1.0  # 提高避让速度
OBSTACLE_THRESHOLD = 4.0  # 提前触发避让
AVOID_STEER_ANGLE = 0.5  # 增大避让转向角
TORQUE_GAIN = 8.0  # 提高加速响应

# 执行器索引
DRIVE_ACTUATOR_IDS = [0, 1]
STEER_ACTUATOR_IDS = [2, 3]

# -------------------------- MuJoCo XML（精细化车辆和障碍物模型）--------------------------
SCENE_XML = """
<mujoco model="autonomous_vehicle">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <option gravity="0 0 -9.81" timestep="0.01"/>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.6 0.7 0.9" rgb2="0.3 0.4 0.6" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.8 0.8 0.8" rgb2="0.9 0.9 0.9" width="300" height="300" mark="edge" markrgb="0.5 0.5 0.5"/>
    <material name="groundplane" texture="grid" texrepeat="5 5" texuniform="true"/>
  </asset>

  <actuator>
    <motor name="left_drive" joint="left_drive" gear="50" ctrllimited="true" ctrlrange="-10 10"/>
    <motor name="right_drive" joint="right_drive" gear="50" ctrllimited="true" ctrlrange="-10 10"/>
    <motor name="left_steer" joint="left_steer" gear="1" ctrllimited="true" ctrlrange="-28.6 28.6"/>
    <motor name="right_steer" joint="right_steer" gear="1" ctrllimited="true" ctrlrange="-28.6 28.6"/>
  </actuator>

  <worldbody>
    <geom name="ground" type="plane" size="20 20 0.1" material="groundplane" friction="0.8"/>
    <light directional="true" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 10" dir="0 0 -1"/>
    <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="5 -5 5" dir="-1 1 -1"/>

    <!-- 精细化车辆模型 -->
    <body name="vehicle" pos="0 0 0.3">
      <joint name="vehicle_free" type="free"/>

      <!-- 车身主体 -->
      <geom name="body_base" type="box" size="0.6 0.3 0.15" rgba="0.2 0.4 0.8 1" mass="1.5"/>

      <!-- 车顶 -->
      <geom name="body_top" type="box" pos="0 0 0.25" size="0.5 0.25 0.1" rgba="0.3 0.5 0.9 1" mass="0.3"/>

      <!-- 前挡风玻璃 -->
      <geom name="windshield" type="box" pos="0.4 0 0.25" size="0.1 0.25 0.1" rgba="0.7 0.9 1 0.7" mass="0.1"/>

      <!-- 前保险杠 -->
      <geom name="front_bumper" type="box" pos="0.6 0 0.1" size="0.05 0.3 0.05" rgba="0.1 0.1 0.1 1" mass="0.1"/>

      <!-- 后保险杠 -->
      <geom name="rear_bumper" type="box" pos="-0.6 0 0.1" size="0.05 0.3 0.05" rgba="0.1 0.1 0.1 1" mass="0.1"/>

      <!-- 左前轮 -->
      <body name="left_front_wheel" pos="0.4 -0.3 0.1">
        <joint name="left_steer" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom name="lf_tire" type="cylinder" size="0.1 0.05" rgba="0.1 0.1 0.1 1" mass="0.15" friction="0.8"/>
        <geom name="lf_rim" type="cylinder" size="0.07 0.06" rgba="0.7 0.7 0.7 1" mass="0.05"/>
        <geom name="lf_hubcap" type="cylinder" size="0.05 0.07" rgba="0.9 0.9 0.9 1" mass="0.01"/>
      </body>

      <!-- 右前轮 -->
      <body name="right_front_wheel" pos="0.4 0.3 0.1">
        <joint name="right_steer" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom name="rf_tire" type="cylinder" size="0.1 0.05" rgba="0.1 0.1 0.1 1" mass="0.15" friction="0.8"/>
        <geom name="rf_rim" type="cylinder" size="0.07 0.06" rgba="0.7 0.7 0.7 1" mass="0.05"/>
        <geom name="rf_hubcap" type="cylinder" size="0.05 0.07" rgba="0.9 0.9 0.9 1" mass="0.01"/>
      </body>

      <!-- 左后轮 -->
      <body name="left_rear_wheel" pos="-0.4 -0.3 0.1">
        <joint name="left_drive" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom name="lr_tire" type="cylinder" size="0.1 0.05" rgba="0.1 0.1 0.1 1" mass="0.15" friction="0.8"/>
        <geom name="lr_rim" type="cylinder" size="0.07 0.06" rgba="0.7 0.7 0.7 1" mass="0.05"/>
        <geom name="lr_hubcap" type="cylinder" size="0.05 0.07" rgba="0.9 0.9 0.9 1" mass="0.01"/>
      </body>

      <!-- 右后轮 -->
      <body name="right_rear_wheel" pos="-0.4 0.3 0.1">
        <joint name="right_drive" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom name="rr_tire" type="cylinder" size="0.1 0.05" rgba="0.1 0.1 0.1 1" mass="0.15" friction="0.8"/>
        <geom name="rr_rim" type="cylinder" size="0.07 0.06" rgba="0.7 0.7 0.7 1" mass="0.05"/>
        <geom name="rr_hubcap" type="cylinder" size="0.05 0.07" rgba="0.9 0.9 0.9 1" mass="0.01"/>
      </body>

      <!-- 激光雷达传感器 -->
      <body name="lidar_sensor" pos="0.5 0 0.35">
        <geom name="lidar_body" type="cylinder" size="0.05 0.03" rgba="0.2 0.2 0.2 1" mass="0.1"/>
        <geom name="lidar_top" type="cylinder" pos="0 0 0.04" size="0.06 0.01" rgba="0.1 0.1 0.1 1" mass="0.05"/>
      </body>
    </body>

    <!-- 精细化障碍物模型 -->
    <!-- 障碍物1：带标签的箱子 -->
    <body name="obstacle1" pos="7 0.5 0.3">
      <joint name="obstacle1_free" type="free"/>
      <geom name="box_main" type="box" size="0.4 0.4 0.3" rgba="0.8 0.2 0.2 1" mass="8" contype="1" conaffinity="1" friction="0.6"/>
      <geom name="box_label" type="box" pos="0 0 0.31" size="0.3 0.3 0.01" rgba="1 1 1 1" mass="0.1"/>
      <geom name="box_handle1" type="cylinder" pos="0.35 0.2 0.15" size="0.02 0.1" rgba="0.6 0.6 0.6 1" mass="0.1"/>
      <geom name="box_handle2" type="cylinder" pos="0.35 -0.2 0.15" size="0.02 0.1" rgba="0.6 0.6 0.6 1" mass="0.1"/>
    </body>

    <!-- 障碍物2：带标签的油桶 -->
    <body name="obstacle2" pos="10 -0.6 0.2">
      <joint name="obstacle2_free" type="free"/>
      <geom name="barrel_body" type="cylinder" size="0.3 0.2" rgba="0.8 0.4 0.2 1" mass="7" contype="1" conaffinity="1" friction="0.5"/>
      <geom name="barrel_top" type="cylinder" pos="0 0 0.21" size="0.25 0.01" rgba="0.9 0.9 0.9 1" mass="0.2"/>
      <geom name="barrel_bottom" type="cylinder" pos="0 0 -0.21" size="0.25 0.01" rgba="0.7 0.7 0.7 1" mass="0.2"/>
      <geom name="barrel_label" type="box" pos="0 0 0" size="0.31 0.05 0.15" rgba="1 1 0 0.8" mass="0.1"/>
    </body>

    <!-- 新增障碍物3：锥形路障（使用多个几何体组合模拟） -->
    <body name="obstacle3" pos="12 0.8 0.0">
      <joint name="obstacle3_free" type="free"/>
      <!-- 锥形主体用多个圆柱体堆叠模拟 -->
      <geom name="cone_section1" type="cylinder" pos="0 0 0.05" size="0.25 0.05" rgba="0.9 0.6 0.1 1" mass="1.5" contype="1" conaffinity="1" friction="0.7"/>
      <geom name="cone_section2" type="cylinder" pos="0 0 0.15" size="0.2 0.05" rgba="0.9 0.6 0.1 1" mass="1.2" contype="1" conaffinity="1" friction="0.7"/>
      <geom name="cone_section3" type="cylinder" pos="0 0 0.25" size="0.15 0.05" rgba="0.9 0.6 0.1 1" mass="0.9" contype="1" conaffinity="1" friction="0.7"/>
      <geom name="cone_section4" type="cylinder" pos="0 0 0.35" size="0.1 0.05" rgba="0.9 0.6 0.1 1" mass="0.6" contype="1" conaffinity="1" friction="0.7"/>
      <geom name="cone_section5" type="cylinder" pos="0 0 0.45" size="0.05 0.05" rgba="0.9 0.6 0.1 1" mass="0.3" contype="1" conaffinity="1" friction="0.7"/>
      <!-- 条纹装饰 -->
      <geom name="cone_stripes1" type="cylinder" pos="0 0 0.1" size="0.26 0.01" rgba="1 1 1 1" mass="0.05"/>
      <geom name="cone_stripes2" type="cylinder" pos="0 0 0.3" size="0.16 0.01" rgba="1 1 1 1" mass="0.05"/>
      <!-- 底座 -->
      <geom name="cone_base" type="cylinder" pos="0 0 -0.05" size="0.3 0.05" rgba="0.2 0.2 0.2 1" mass="1.0"/>
    </body>

    <site name="target" pos="15 0 0.5" size="0.1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""


# -------------------------- 激光雷达（已匹配mj_ray参数）--------------------------
class LidarSensor:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.angles = np.linspace(0, 2 * np.pi, LIDAR_NUM_RAYS, endpoint=False)
        self.geomgroup = np.zeros(6, dtype=np.uint8)
        self.geomgroup[0] = 1  # 只检测第0组的geom

    def rotate_vector(self, vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = quat
        rot_mat = np.array([
            [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
        ])
        return np.dot(rot_mat, vec)

    def get_distance_data(self) -> List[float]:
        distances = [LIDAR_MAX_DISTANCE] * LIDAR_NUM_RAYS
        vehicle_pos = self.data.qpos[:3]
        vehicle_quat = self.data.qpos[3:7]

        for i, angle in enumerate(self.angles):
            local_dir = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
            local_dir = local_dir / np.linalg.norm(local_dir)
            world_dir = self.rotate_vector(local_dir, vehicle_quat)
            world_dir = world_dir / np.linalg.norm(world_dir)

            ray_start = vehicle_pos + self.rotate_vector(LIDAR_MOUNT_POS, vehicle_quat)
            ray_start = ray_start.astype(np.float64)

            # 匹配mj_ray参数格式
            pnt = ray_start.reshape(3, 1)
            vec = world_dir.reshape(3, 1)
            geomid = np.zeros(1, dtype=np.int32)

            detected_dist = mujoco.mj_ray(
                self.model, self.data,
                pnt=pnt,
                vec=vec,
                geomgroup=self.geomgroup,
                flg_static=1,
                bodyexclude=0,
                geomid=geomid
            )

            if geomid[0] != -1 and detected_dist <= LIDAR_MAX_DISTANCE:
                distances[i] = detected_dist

        return distances


# -------------------------- 可视化类 --------------------------
class Visualizer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        self.window = glfw.create_window(1280, 720, "无人车仿真", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self.cam.distance = 10.0
        self.cam.azimuth = -45.0
        self.cam.elevation = -30.0
        self.cam.lookat = [5.0, 0.0, 0.5]

    def render(self):
        if glfw.window_should_close(self.window):
            return False
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        viewport = mujoco.MjrRect(0, 0, 1280, 720)
        mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        return True

    def close(self):
        glfw.destroy_window(self.window)
        glfw.terminate()


# -------------------------- 日志类 --------------------------
class DataLogger:
    def __init__(self):
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.fields = ['step', 'time', 'x', 'y', 'speed', 'steer', 'torque', 'collision']
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(self.fields)

    def log(self, step, time, x, y, speed, steer, torque, collision):
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [step, round(time, 3), round(x, 3), round(y, 3), round(speed, 3), round(steer, 3), round(torque, 3),
                 1 if collision else 0])


# -------------------------- 决策算法类 --------------------------
class RulePolicy:
    def __init__(self):
        self.front_rays = slice(350, 360)
        self.front_ext = slice(0, 10)
        self.left_rays = slice(270, 350)
        self.right_rays = slice(10, 90)

    def get_speed(self, vel):
        return np.linalg.norm(vel[:2])

    def decide(self, lidar_data, vel):
        lidar = np.array(lidar_data)
        current_speed = self.get_speed(vel)

        front_dist = np.min(np.concatenate([lidar[self.front_rays], lidar[self.front_ext]]))
        left_dist = np.min(lidar[self.left_rays])
        right_dist = np.min(lidar[self.right_rays])

        if front_dist >= OBSTACLE_THRESHOLD:
            steer = 0.0
            target_speed = SAFE_SPEED
        else:
            steer = AVOID_STEER_ANGLE if left_dist > right_dist else -AVOID_STEER_ANGLE
            target_speed = AVOID_SPEED

        torque = TORQUE_GAIN * (target_speed - current_speed)
        torque = np.clip(torque, -MAX_DRIVE_TORQUE, MAX_DRIVE_TORQUE)
        return steer, torque


# -------------------------- 仿真环境类（已整合自动录制功能）--------------------------
class VehicleEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.lidar = LidarSensor(self.model, self.data)
        self.vis = Visualizer(self.model, self.data)
        self.logger = DataLogger()
        self.policy = RulePolicy()

        # 初始化帧缓存用于GIF录制
        self.frames = []
        self.step_count = 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        return {
            'pos': self.data.qpos[:3],
            'vel': self.data.qvel[:3],
            'lidar': self.lidar.get_distance_data(),
            'collision': self.data.ncon > 0
        }

    def step(self, steer, torque):
        for idx in STEER_ACTUATOR_IDS:
            self.data.ctrl[idx] = np.clip(steer, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)
        for idx in DRIVE_ACTUATOR_IDS:
            self.data.ctrl[idx] = torque

        mujoco.mj_step(self.model, self.data, nstep=NSTEP)
        self.step_count += 1
        return self.get_state()

    def run(self):
        state = self.reset()
        print("仿真启动（按ESC关闭），正在自动录制GIF...")
        print("=" * 50)

        # 主循环 - 持续运行直到用户按下ESC键
        while True:
            # 1. 决策：根据激光雷达和速度生成控制指令
            steer, torque = self.policy.decide(state['lidar'], state['vel'])
            # 2. 执行动作：更新车辆状态
            state = self.step(steer, torque)
            # 3. 记录日志
            self.logger.log(self.step_count, self.step_count * TIME_STEP, state['pos'][0], state['pos'][1],
                            self.policy.get_speed(state['vel']), steer, torque, state['collision'])
            # 4. 渲染画面
            if not self.vis.render():
                print("\n用户按下ESC键，仿真结束")
                break

            # -------------------------- 录制当前帧 --------------------------
            try:
                width, height = glfw.get_framebuffer_size(self.vis.window)
                if width > 0 and height > 0:
                    buffer = np.zeros((height, width, 3), dtype=np.uint8)
                    # 创建正确的 MjrRect 对象
                    viewport = mujoco.MjrRect(0, 0, width, height)
                    mujoco.mjr_readPixels(buffer, None, viewport, self.vis.context)
                    buffer = np.flipud(buffer)
                    self.frames.append(buffer)
            except Exception as e:
                print(f"录制帧时出错: {e}")
            # -------------------------------------------------------------------

            # 打印进度（每100步）
            if self.step_count % 100 == 0:
                print(
                    f"Step: {self.step_count:4d} | 速度: {self.policy.get_speed(state['vel']):.2f}m/s | 碰撞: {state['collision']}")

        # -------------------------- 结束录制并保存 --------------------------
        try:
            if self.frames:
                # 使用imageio v3 API保存GIF，调整duration参数延长播放时间
                gif_path = 'simulation.gif'
                imageio.mimsave(gif_path, self.frames, duration=0.2, loop=0)
                print(f"GIF动图已保存至：{gif_path}（项目根目录）")
            else:
                print("没有帧数据可保存为GIF")
        except Exception as e:
            print(f"GIF保存失败: {e}")
        finally:
            self.vis.close()
        # -------------------------------------------------------------------


if __name__ == "__main__":
    env = VehicleEnv()
    env.run()
