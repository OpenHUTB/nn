import mujoco
import mujoco.viewer
import time
import numpy as np

# 修改后的 XML 模型：修正了手臂位置，增加了脚掌，优化了动力学参数
model_xml = """
<mujoco model="humanoid_stable_walking">
    <compiler angle="degree" inertiafromgeom="true"/>
    <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="texplane" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1"/>
        <material name="mat_skin" rgba="0.8 0.6 0.4 1"/> 
        <material name="mat_cloth" rgba="0.1 0.7 0.1 1"/>
    </asset>

    <worldbody>
        <light pos="0 0 4.0" dir="0 0 -1" castshadow="true"/>
        <geom name="floor" pos="0 0 0" size="0 0 .5" type="plane" material="matplane" condim="3" friction="1.0 0.05 0.0001"/>

        <body name="torso" pos="0 0 1.25">
            <freejoint name="root"/>
            <geom name="torso_geom" type="capsule" fromto="0 -.07 0 0 .07 0" size="0.07" material="mat_cloth"/>
            <geom name="head" type="sphere" pos="0 0 0.19" size="0.09" material="mat_skin"/> 

            <body name="l_arm" pos="0 0.13 0.05">
                <joint name="l_shoulder" type="hinge" axis="0 1 0" range="-90 90" damping="2"/>
                <geom name="l_hand_geom" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03" material="mat_skin"/>
            </body>

            <body name="r_arm" pos="0 -0.13 0.05">
                <joint name="r_shoulder" type="hinge" axis="0 1 0" range="-90 90" damping="2"/>
                <geom name="r_hand_geom" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03" material="mat_skin"/>
            </body>

            <body name="l_leg" pos="0 0.1 0">
                <joint name="l_hip" type="hinge" axis="0 1 0" range="-40 40" damping="20"/>
                <geom name="l_thigh" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" material="mat_skin"/>
                <body name="l_shin" pos="0 0 -0.35">
                    <joint name="l_knee" type="hinge" axis="0 1 0" range="0 80" damping="15"/>
                    <geom name="l_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" material="mat_skin"/>
                    <body name="l_foot" pos="0 0 -0.35">
                        <geom name="l_foot_geom" type="box" size="0.08 0.04 0.02" pos="0.04 0 0" material="mat_skin"/>
                    </body>
                </body>
            </body>

            <body name="r_leg" pos="0 -0.1 0">
                <joint name="r_hip" type="hinge" axis="0 1 0" range="-40 40" damping="20"/>
                <geom name="r_thigh" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" material="mat_skin"/>
                <body name="r_shin" pos="0 0 -0.35">
                    <joint name="r_knee" type="hinge" axis="0 1 0" range="0 80" damping="15"/>
                    <geom name="r_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" material="mat_skin"/>
                    <body name="r_foot" pos="0 0 -0.35">
                        <geom name="r_foot_geom" type="box" size="0.08 0.04 0.02" pos="0.04 0 0" material="mat_skin"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <position name="p_l_hip" joint="l_hip" kp="500" ctrlrange="-40 40"/>
        <position name="p_r_hip" joint="r_hip" kp="500" ctrlrange="-40 40"/>
        <position name="p_l_knee" joint="l_knee" kp="400" ctrlrange="0 80"/>
        <position name="p_r_knee" joint="r_knee" kp="400" ctrlrange="0 80"/>
        <position name="p_l_shoulder" joint="l_shoulder" kp="100" ctrlrange="-90 90"/>
        <position name="p_r_shoulder" joint="r_shoulder" kp="100" ctrlrange="-90 90"/>
    </actuator>
</mujoco>
"""

def main():
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    
    # 初始高度调整，确保脚着地
    data.qpos[2] = 0.85 

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_sync = time.time()
        while viewer.is_running():
            step_start = time.time()

            t = data.time
            freq = 0.8  # 稍微放慢频率更稳定
            
            # --- 步态逻辑 ---
            # 基础目标角度
            walk_cycle = np.sin(2 * np.pi * freq * t)
            target_angle = walk_cycle * 20
            
            # 髋部控制
            data.ctrl[0] = target_angle   # 左髋
            data.ctrl[1] = -target_angle  # 右髋
            
            # 膝盖控制：迈步时弯曲，支撑时伸直（防止腿软）
            data.ctrl[2] = 20 if target_angle > 5 else 2 
            data.ctrl[3] = 20 if target_angle < -5 else 2
            
            # 手部控制：摆臂动作
            data.ctrl[4] = -target_angle * 1.2 # 摆臂方向通常与同侧腿相反
            data.ctrl[5] = target_angle * 1.2
            
            mujoco.mj_step(model, data)

            # 摔倒自动重置 (检测躯干高度)
            if data.qpos[2] < 0.45:
                mujoco.mj_resetData(model, data)
                data.qpos[2] = 0.85
                # 随机给一个小推力防止死循环原地摔
                data.qvel[0] = 0.1

            # 控制渲染频率
            if time.time() - last_sync > 1/60:
                viewer.sync()
                last_sync = time.time()

            # 保持实时率
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()
