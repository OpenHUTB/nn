import mujoco
import mujoco.viewer
import time
import numpy as np

# 优化后的 XML 模型 - 增强稳定性
model_xml = """
<mujoco model="humanoid_stable_standing">
    <compiler angle="degree" inertiafromgeom="true"/>
    <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="texplane" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1"/>
        <material name="mat_skin" rgba="0.8 0.6 0.4 1"/> 
        <material name="mat_cloth" rgba="0.1 0.7 0.1 1"/>
        <material name="mat_foot" rgba="0.3 0.3 0.3 1"/>
    </asset>

    <worldbody>
        <light pos="3 3 5" dir="0 0 -1" castshadow="true"/>
        <light pos="-2 2 3" dir="0 0 -1" castshadow="true"/>
        <light pos="0 5 2" dir="0 -1 0" castshadow="false"/>
        <geom name="floor" pos="0 0 0" size="10 10 .5" type="plane" material="matplane" condim="6" friction="1.5 0.2 0.02"/>

        <body name="torso" pos="0 0 0.85">
            <freejoint name="root"/>
            <!-- 躯干 - 增加宽度和重量使重心更稳 -->
            <geom name="torso_geom" type="capsule" fromto="0 -.1 0 0 .1 0" size="0.1" mass="15.0" material="mat_cloth"/>
            <geom name="head" type="sphere" pos="0 0 0.25" size="0.11" mass="2.5" material="mat_skin"/> 

            <body name="l_arm" pos="0 0.18 0.1">
                <joint name="l_shoulder" type="hinge" axis="0 1 0" range="-90 90" damping="8" stiffness="20"/>
                <geom name="l_hand_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.045" mass="2.0" material="mat_skin"/>
            </body>

            <body name="r_arm" pos="0 -0.18 0.1">
                <joint name="r_shoulder" type="hinge" axis="0 1 0" range="-90 90" damping="8" stiffness="20"/>
                <geom name="r_hand_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.045" mass="2.0" material="mat_skin"/>
            </body>

            <body name="l_leg" pos="0 0.1 -0.05">
                <joint name="l_hip" type="hinge" axis="0 1 0" range="-30 30" damping="30" stiffness="50"/>
                <geom name="l_thigh" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.07" mass="5.0" material="mat_skin"/>
                <body name="l_shin" pos="0 0 -0.35">
                    <joint name="l_knee" type="hinge" axis="0 1 0" range="0 60" damping="25" stiffness="40"/>
                    <geom name="l_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.30" size="0.06" mass="3.5" material="mat_skin"/>
                    <body name="l_foot" pos="0 0 -0.30">
                        <joint name="l_ankle" type="hinge" axis="0 1 0" range="-15 15" damping="40" stiffness="60"/>
                        <geom name="l_foot_geom" type="box" size="0.14 0.06 0.035" pos="0.07 0 -0.005" mass="2.0" material="mat_foot"/>
                    </body>
                </body>
            </body>

            <body name="r_leg" pos="0 -0.1 -0.05">
                <joint name="r_hip" type="hinge" axis="0 1 0" range="-30 30" damping="30" stiffness="50"/>
                <geom name="r_thigh" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.07" mass="5.0" material="mat_skin"/>
                <body name="r_shin" pos="0 0 -0.35">
                    <joint name="r_knee" type="hinge" axis="0 1 0" range="0 60" damping="25" stiffness="40"/>
                    <geom name="r_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.30" size="0.06" mass="3.5" material="mat_skin"/>
                    <body name="r_foot" pos="0 0 -0.30">
                        <joint name="r_ankle" type="hinge" axis="0 1 0" range="-15 15" damping="40" stiffness="60"/>
                        <geom name="r_foot_geom" type="box" size="0.14 0.06 0.035" pos="0.07 0 -0.005" mass="2.0" material="mat_foot"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <!-- 髋关节 - 增强刚度 -->
        <position name="p_l_hip" joint="l_hip" kp="1200" ctrlrange="-30 30"/>
        <position name="p_r_hip" joint="r_hip" kp="1200" ctrlrange="-30 30"/>
        <!-- 膝关节 -->
        <position name="p_l_knee" joint="l_knee" kp="1000" ctrlrange="0 60"/>
        <position name="p_r_knee" joint="r_knee" kp="1000" ctrlrange="0 60"/>
        <!-- 踝关节 - 增强脚踝控制 -->
        <position name="p_l_ankle" joint="l_ankle" kp="800" ctrlrange="-15 15"/>
        <position name="p_r_ankle" joint="r_ankle" kp="800" ctrlrange="-15 15"/>
        <!-- 肩关节 -->
        <position name="p_l_shoulder" joint="l_shoulder" kp="300" ctrlrange="-90 90"/>
        <position name="p_r_shoulder" joint="r_shoulder" kp="300" ctrlrange="-90 90"/>
    </actuator>
</mujoco>
"""

class StableController:
    """稳定站立和行走控制器"""
    
    def __init__(self, model):
        self.model = model
        self.step_phase = 0.0
        self.step_duration = 1.5  # 更慢的步态更稳定
        self.walk_speed = 0.4      # 降低速度
        
        self.hip_swing = 20.0      # 减小摆动幅度
        self.knee_bend = 35.0      # 减小弯曲
        self.ankle_pitch = 8.0
        
        # 增强的 PID 平衡控制
        self.roll_pid = PID(2.0, 0.1, 1.2, -5, 5)
        self.pitch_pid = PID(2.5, 0.15, 1.5, -8, 8)
        self.yaw_pid = PID(1.5, 0.05, 0.8, -3, 3)
        
        # 重心控制
        self.com_pid = PID(1.0, 0.02, 0.5, -10, 10)
        
        self.vel_integral = 0.0
        self.com_target = 0.0  # 目标重心位置
        
    def compute_control(self, data, dt):
        """计算控制信号"""
        t = data.time
        ctrl = np.zeros(self.model.nu)
        
        # 计算质心位置和偏差
        com_pos = self.get_com(data)
        com_error = com_pos[0] - self.com_target
        
        # 步态相位（仅当行走时启用）
        walk_enabled = t > 1.0  # 先站立1秒再开始行走
        
        if walk_enabled:
            self.step_phase = (t % self.step_duration) / self.step_duration
            phase_rad = 2 * np.pi * self.step_phase
            
            # 髋关节控制
            hip_angle = self.hip_swing * np.sin(phase_rad)
            ctrl[0] = hip_angle - com_error * 2
            ctrl[1] = -hip_angle - com_error * 2
            
            # 膝关节控制
            left_knee = 0.0
            right_knee = 0.0
            
            if self.step_phase > 0.5:
                swing_progress = (self.step_phase - 0.5) * 2
                left_knee = self.knee_bend * np.sin(swing_progress * np.pi)
                right_knee = 8.0
            else:
                swing_progress = self.step_phase * 2
                left_knee = 8.0
                right_knee = self.knee_bend * np.sin(swing_progress * np.pi)
            
            ctrl[2] = np.clip(left_knee, 0, 60)
            ctrl[3] = np.clip(right_knee, 0, 60)
            
            # 踝关节控制
            left_ankle = 0.0
            right_ankle = 0.0
            
            if self.step_phase > 0.5:
                if self.step_phase > 0.9:
                    left_ankle = -self.ankle_pitch * (self.step_phase - 0.9) / 0.1
                elif self.step_phase > 0.8:
                    left_ankle = self.ankle_pitch
            else:
                if self.step_phase > 0.4:
                    right_ankle = -self.ankle_pitch * (self.step_phase - 0.4) / 0.1
                elif self.step_phase > 0.3:
                    right_ankle = self.ankle_pitch
            
            ctrl[4] = np.clip(left_ankle, -15, 15)
            ctrl[5] = np.clip(right_ankle, -15, 15)
            
            # 手臂控制
            ctrl[6] = -hip_angle * 1.0
            ctrl[7] = hip_angle * 1.0
            
            # 速度维持
            actual_vel = float(data.qvel[0])
            vel_error = self.walk_speed - actual_vel
            self.vel_integral += vel_error * dt
            vel_correction = 0.3 * vel_error + 0.05 * self.vel_integral
            vel_correction = np.clip(vel_correction, -5, 5)
            
            ctrl[0] += vel_correction
            ctrl[1] += vel_correction
        else:
            # 站立模式：保持直立
            ctrl[0] = -com_error * 3  # 左髋
            ctrl[1] = -com_error * 3  # 右髋
            ctrl[2] = 5.0   # 左膝微屈
            ctrl[3] = 5.0   # 右膝微屈
            ctrl[4] = 0.0   # 左踝中立
            ctrl[5] = 0.0   # 右踝中立
            ctrl[6] = 0.0   # 左臂下垂
            ctrl[7] = 0.0   # 右臂下垂
        
        # 姿态平衡控制（始终启用）
        try:
            # 获取躯干姿态
            torso_quat = data.xquat[0:4]
            roll, pitch, yaw = self.quat_to_euler(torso_quat)
            
            # 增强的 PID 控制
            roll_correction = self.roll_pid.update(-roll, dt)
            pitch_correction = self.pitch_pid.update(-pitch, dt)
            yaw_correction = self.yaw_pid.update(-yaw, dt)
            
            # 应用姿态修正
            ctrl[0] += roll_correction * 4 + yaw_correction * 2
            ctrl[1] -= roll_correction * 4 + yaw_correction * 2
            ctrl[4] += pitch_correction * 3
            ctrl[5] += pitch_correction * 3
            
        except Exception:
            pass
        
        # 输出限制
        return np.clip(ctrl, 
                      [-30, -30, 0, 0, -15, -15, -90, -90],
                      [30, 30, 60, 60, 15, 15, 90, 90])
    
    def get_com(self, data):
        """计算质心位置"""
        # 简化计算：使用躯干位置作为质心近似
        return np.array([data.qpos[0], data.qpos[1], data.qpos[2]])
    
    def quat_to_euler(self, quat):
        """四元数转欧拉角"""
        w = float(quat[0])
        x = float(quat[1])
        y = float(quat[2])
        z = float(quat[3])
        
        # Roll
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return float(roll), float(pitch), float(yaw)


class PID:
    """增强的 PID 控制器"""
    
    def __init__(self, kp, ki, kd, output_min=-np.inf, output_max=np.inf):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.integral = 0.0
        self.last_error = 0.0
        self.last_output = 0.0
    
    def update(self, error, dt):
        if dt <= 0:
            return 0.0
        
        error = float(error)
        
        # 比例项
        p_term = self.kp * error
        
        # 积分项（带防饱和）
        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100)
        i_term = self.ki * self.integral
        
        # 微分项（带滤波）
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative
        
        # 计算输出
        output = p_term + i_term + d_term
        output = np.clip(output, self.output_min, self.output_max)
        
        # 防积分饱和
        if abs(output) >= abs(self.output_max) * 0.9:
            self.integral -= error * dt * 0.5
        
        self.last_error = error
        self.last_output = output
        return float(output)
    
    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_output = 0.0


def main():
    print("正在加载模型...")
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    
    # 初始化控制器
    controller = StableController(model)
    
    # 计算正确的初始高度
    initial_height = 0.35 + 0.30 + 0.035 + 0.05
    data.qpos[2] = initial_height
    
    # 设置初始姿态为稳定站立
    # 髋关节中立，膝关节微屈5度，踝关节中立
    if len(data.qpos) > 7:
        for i in range(7, min(15, len(data.qpos))):
            data.qpos[i] = 0.0
        # 膝关节微屈增加稳定性
        if len(data.qpos) > 9:
            data.qpos[9] = 5.0   # 左膝
            data.qpos[10] = 5.0  # 右膝
    
    # 重置速度为零
    data.qvel[:] = 0.0
    
    print(f"模型信息:")
    print(f"  - 关节数量: {model.nq}")
    print(f"  - 控制通道: {model.nu}")
    print(f"  - 初始高度: {initial_height:.3f}m")
    
    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 优化相机视角
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 75.0
        viewer.cam.elevation = -10.0
        viewer.cam.lookat = [0, 0, 0.8]
        
        last_sync = time.time()
        last_time = data.time
        stability_counter = 0
        step_count = 0
        
        print("\n" + "="*50)
        print("人形机器人稳定控制系统")
        print("="*50)
        print("\n初始化站立...")
        print("前1秒: 稳定站立")
        print("1秒后: 开始慢速行走")
        print("\n稳定性增强:")
        print("  - 增强的 PID 姿态控制")
        print("  - 质心位置调节")
        print("  - 踝关节主动稳定")
        print("  - 膝关节微屈缓冲")
        print("\n相机控制:")
        print("  - 鼠标左键: 旋转视角")
        print("  - 鼠标右键: 平移")
        print("  - 滚轮: 缩放")
        print("\n按 Ctrl+C 退出\n")
        
        # 等待初始稳定
        viewer.sync()
        time.sleep(0.5)
        
        while viewer.is_running():
            step_start = time.time()
            
            # 计算时间步长
            dt = data.time - last_time
            if dt <= 0:
                dt = model.opt.timestep
            last_time = data.time
            
            # 获取控制信号
            try:
                ctrl = controller.compute_control(data, dt)
                data.ctrl[:] = ctrl
            except Exception as e:
                if step_count % 200 == 0:
                    print(f"控制计算警告: {e}")
                continue
            
            # 步进仿真
            mujoco.mj_step(model, data)
            step_count += 1
            
            # 获取状态
            torso_height = data.qpos[2]
            torso_roll = data.qvel[3] if len(data.qvel) > 3 else 0
            torso_pitch = data.qvel[4] if len(data.qvel) > 4 else 0
            
            # 摔倒检测
            is_fallen = torso_height < 0.3 or abs(torso_roll) > 2.0 or abs(torso_pitch) > 2.0
            
            if is_fallen:
                stability_counter += 1
                if stability_counter > 30:
                    print(f"⚠️ 摔倒恢复... (高度={torso_height:.2f}, 步数={step_count})")
                    mujoco.mj_resetData(model, data)
                    data.qpos[2] = initial_height
                    data.qvel[:] = 0.0
                    if len(data.qpos) > 9:
                        data.qpos[9] = 5.0
                        data.qpos[10] = 5.0
                    controller.roll_pid.reset()
                    controller.pitch_pid.reset()
                    controller.yaw_pid.reset()
                    controller.com_pid.reset()
                    controller.vel_integral = 0.0
                    stability_counter = 0
                    step_count = 0
                    viewer.sync()
                    time.sleep(0.2)
            else:
                stability_counter = max(0, stability_counter - 1)
            
            # 状态输出
            if step_count > 0 and step_count % 300 == 0:
                status = "🚶 行走中" if step_count > 1500 else "🧍 站立中"
                print(f"{status} | 步数: {step_count:4d} | 高度: {torso_height:.2f}m | "
                      f"速度: {data.qvel[0]:.2f}m/s | X: {data.qpos[0]:.2f}m")
            
            # 同步渲染
            if time.time() - last_sync > 1/60:
                viewer.sync()
                last_sync = time.time()
            
            # 实时控制
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)
        
        viewer.close()
        print("\n演示结束")

if __name__ == "__main__":
    main()