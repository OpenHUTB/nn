import numpy as np
from collections import deque
from utils import quat_to_euler_xyz

class SensorSimulator:
    """传感器模拟模块（IMU+足底力，独立可复用）"""
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # 开关
        self.enable_sensor_simulation = True
        
        # IMU参数
        self.imu_angle_noise = 0.01        # 欧拉角噪声(rad)
        self.imu_vel_noise = 0.05          # 角速度噪声(rad/s)
        self.imu_delay_frames = 2          # 延迟帧数
        
        # 足底力参数
        self.foot_force_noise = 0.3        # 力噪声(N)
        self.foot_force_offset = 0.1       # 零漂
        self.foot_contact_threshold = 1.5  # 接触判定阈值
        
        # 缓存（模拟延迟）
        self.imu_data_buffer = deque(maxlen=self.imu_delay_frames)
        self.foot_data_buffer = deque(maxlen=self.imu_delay_frames)
        
        # 存储当前数据
        self.current_sensor_data = {}

    def simulate_imu(self):
        """模拟带噪声+延迟的IMU数据"""
        # 真实数据
        true_quat = self.data.qpos[3:7].astype(np.float64).copy()
        true_euler = quat_to_euler_xyz(true_quat)
        true_ang_vel = self.data.qvel[3:6].astype(np.float64).copy()
        
        # 添加噪声
        noisy_euler = true_euler + np.random.normal(0, self.imu_angle_noise, 3)
        noisy_ang_vel = true_ang_vel + np.random.normal(0, self.imu_vel_noise, 3)
        
        # 限幅
        noisy_euler = np.clip(noisy_euler, -np.pi/2, np.pi/2)
        noisy_ang_vel = np.clip(noisy_ang_vel, -5.0, 5.0)
        
        # 缓存
        self.imu_data_buffer.append({
            "euler": noisy_euler,
            "ang_vel": noisy_ang_vel,
            "true_euler": true_euler,
            "true_ang_vel": true_ang_vel
        })
        
        # 返回延迟后的数据
        if len(self.imu_data_buffer) < self.imu_delay_frames:
            return {"euler": true_euler, "ang_vel": true_ang_vel, "true_euler": true_euler, "true_ang_vel": true_ang_vel}
        return self.imu_data_buffer[0]

    def simulate_foot_force(self):
        """模拟带噪声+零漂的足底力数据"""
        # 真实接触力
        left_force = self._get_foot_force("left")
        right_force = self._get_foot_force("right")
        
        # 添加噪声和零漂
        noisy_left = left_force + np.random.normal(0, self.foot_force_noise) + self.foot_force_offset
        noisy_right = right_force + np.random.normal(0, self.foot_force_noise) + self.foot_force_offset
        
        # 限幅
        noisy_left = max(0.0, noisy_left)
        noisy_right = max(0.0, noisy_right)
        
        # 接触判定
        left_contact = 1 if noisy_left > self.foot_contact_threshold else 0
        right_contact = 1 if noisy_right > self.foot_contact_threshold else 0
        
        # 缓存
        self.foot_data_buffer.append({
            "left_force": noisy_left,
            "right_force": noisy_right,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "true_left_force": left_force,
            "true_right_force": right_force
        })
        
        # 返回延迟后的数据
        if len(self.foot_data_buffer) < self.imu_delay_frames:
            return {
                "left_force": left_force, "right_force": right_force,
                "left_contact": 1 if left_force > self.foot_contact_threshold else 0,
                "right_contact": 1 if right_force > self.foot_contact_threshold else 0,
                "true_left_force": left_force, "true_right_force": right_force
            }
        return self.foot_data_buffer[0]

    def _get_foot_force(self, side):
        """获取单只脚的真实接触力"""
        import mujoco
        foot_geoms = ["foot1_" + side, "foot2_" + side]
        total_force = 0.0
        for geom_name in foot_geoms:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, geom_id, force)
            total_force += np.linalg.norm(force[:3])
        return total_force

    def get_sensor_data(self, gait_mode="NORMAL"):
        """获取最终传感器数据（模拟/真实可切换）"""
        if not self.enable_sensor_simulation:
            # 关闭模拟：返回真实数据
            imu_data = {
                "euler": quat_to_euler_xyz(self.data.qpos[3:7]),
                "ang_vel": self.data.qvel[3:6].copy(),
                "true_euler": quat_to_euler_xyz(self.data.qpos[3:7]),
                "true_ang_vel": self.data.qvel[3:6].copy()
            }
            foot_data = {
                "left_force": self._get_foot_force("left"),
                "right_force": self._get_foot_force("right"),
                "left_contact": 1 if self._get_foot_force("left") > self.foot_contact_threshold else 0,
                "right_contact": 1 if self._get_foot_force("right") > self.foot_contact_threshold else 0,
                "true_left_force": self._get_foot_force("left"),
                "true_right_force": self._get_foot_force("right")
            }
        else:
            # 开启模拟：返回带噪声+延迟的数据
            imu_data = self.simulate_imu()
            foot_data = self.simulate_foot_force()
        
        # 整合数据
        self.current_sensor_data = {
            "imu": imu_data,
            "foot": foot_data,
            "time": self.data.time,
            "gait_mode": gait_mode
        }
        return self.current_sensor_data

    def print_sensor_data(self):
        """打印当前传感器数据"""
        if not self.current_sensor_data:
            print("[传感器] 暂无数据")
            return
        imu = self.current_sensor_data["imu"]
        foot = self.current_sensor_data["foot"]
        print("\n=== 传感器数据 ===")
        print(f"时间: {self.current_sensor_data['time']:.2f}s | 模拟: {'开启' if self.enable_sensor_simulation else '关闭'} | 步态: {self.current_sensor_data['gait_mode']}")
        print(f"IMU姿态: {imu['euler'][0]:.3f}/{imu['euler'][1]:.3f}/{imu['euler'][2]:.3f}rad (真实: {imu['true_euler'][0]:.3f}/{imu['true_euler'][1]:.3f}/{imu['true_euler'][2]:.3f})")
        print(f"左脚力: {foot['left_force']:.2f}N (真实: {foot['true_left_force']:.2f}N) | 接触: {foot['left_contact']}")
        print(f"右脚力: {foot['right_force']:.2f}N (真实: {foot['true_right_force']:.2f}N) | 接触: {foot['right_contact']}")
        print("==================\n")
