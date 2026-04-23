import mujoco
import mujoco.viewer
import time
import numpy as np

# 深度稳定优化的 XML
model_xml = """
<mujoco model="humanoid_stable_walking">
    <compiler angle="degree" inertiafromgeom="true"/>
    <option timestep="0.001" integrator="RK4" gravity="0 0 -9.81"/>
    
    <asset>
        <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="texplane" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1"/>
        <material name="mat_skin" rgba="0.8 0.6 0.4 1"/> 
    </asset>

    <worldbody>
        <light pos="0 0 4.0" dir="0 0 -1" castshadow="true"/>
        <geom name="floor" pos="0 0 0" size="0 0 .5" type="plane" material="matplane" condim="3" friction="1.5 0.1 0.001"/>
        
        <body name="torso" pos="0 0 1.1">
            <freejoint name="root"/>
            <geom name="torso_geom" type="capsule" fromto="0 -.07 0 0 .07 0" size="0.07" rgba="0.1 0.7 0.1 1"/>
            <geom name="head" type="sphere" pos="0 0 0.22" size="0.09" material="mat_skin"/> 
            
            <body name="l_leg" pos="0 0.1 0">
                <joint name="l_hip" type="hinge" axis="0 1 0" range="-40 40" damping="15"/>
                <geom name="l_thigh" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" material="mat_skin"/>
                <body name="l_shin" pos="0 0 -0.35">
                    <joint name="l_knee" type="hinge" axis="0 1 0" range="0 80" damping="10"/>
                    <geom name="l_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" material="mat_skin"/>
                </body>
            </body>

            <body name="r_leg" pos="0 -0.1 0">
                <joint name="r_hip" type="hinge" axis="0 1 0" range="-40 40" damping="15"/>
                <geom name="r_thigh" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" material="mat_skin"/>
                <body name="r_shin" pos="0 0 -0.35">
                    <joint name="r_knee" type="hinge" axis="0 1 0" range="0 80" damping="10"/>
                    <geom name="r_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" material="mat_skin"/>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <position name="p_l_hip" joint="l_hip" kp="250" ctrlrange="-40 40"/>
        <position name="p_r_hip" joint="r_hip" kp="250" ctrlrange="-40 40"/>
        <position name="p_l_knee" joint="l_knee" kp="200" ctrlrange="0 80"/>
        <position name="p_r_knee" joint="r_knee" kp="200" ctrlrange="0 80"/>
    </actuator>
</mujoco>
"""

def main():
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    data.qpos[2] = 1.1 # 初始高度离地

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_sync = time.time()
        while viewer.is_running():
            step_start = time.time()

            # --- 平滑步态控制 (闭环逻辑原型) ---
            t = data.time
            freq = 1.0
            
            # 正弦波给出目标角度
            target_angle = np.sin(2 * np.pi * freq * t) * 25
            
            data.ctrl[0] = target_angle   # 左髋
            data.ctrl[1] = -target_angle  # 右髋
            data.ctrl[2] = 15 if target_angle > 5 else 0 # 迈腿时膝盖微弯
            data.ctrl[3] = 15 if target_angle < -5 else 0

            mujoco.mj_step(model, data)

            # 摔倒自动重置
            if data.qpos[2] < 0.6:
                mujoco.mj_resetData(model, data)
                data.qpos[2] = 1.1

            # 解决画面闪烁：限制同步频率
            if time.time() - last_sync > 1/60:
                viewer.sync()
                last_sync = time.time()

            # 保持实时率
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()
