import mujoco
import mujoco.viewer
import time
import numpy as np

# 1. 升级版模型定义：让 MuJoCo 小人拥有肤色和头部，视觉对标 PyBullet
model_xml = """
<mujoco model="humanoid">
    <compiler angle="degree" inertiafromgeom="true"/>
    <option timestep="0.005" integrator="RK4" />
    <visual>
        <map fogstart="3" fogend="10" force="0.1" znear="0.01"/>
        <quality shadowsize="4096"/>
    </visual>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="texplane" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1"/>
        <material name="mat_skin" rgba="0.8 0.6 0.4 1"/> 
        <material name="mat_torso" rgba="0.1 0.7 0.1 1"/> 
    </asset>
    <worldbody>
        <light pos="0 0 4.0" dir="0 0 -1" castshadow="true"/>
        <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="matplane" condim="3"/>
        <body name="torso" pos="0 0 1.25">
            <freejoint name="root"/>
            <geom name="torso_geom" type="capsule" fromto="0 -.07 0 0 .07 0" size="0.07" material="mat_torso"/>
            <geom name="head" type="sphere" pos="0 0 0.22" size="0.09" material="mat_skin"/> 
            <body name="l_leg" pos="0 0.1 0">
                <joint name="l_hip" type="hinge" axis="0 1 0" range="-30 30"/>
                <geom name="l_thigh" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" material="mat_skin"/>
                <body name="l_shin" pos="0 0 -0.35">
                    <joint name="l_knee" type="hinge" axis="0 1 0" range="0 60"/>
                    <geom name="l_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" material="mat_skin"/>
                </body>
            </body>
            <body name="r_leg" pos="0 -0.1 0">
                <joint name="r_hip" type="hinge" axis="0 1 0" range="-30 30"/>
                <geom name="r_thigh" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" material="mat_skin"/>
                <body name="r_shin" pos="0 0 -0.35">
                    <joint name="r_knee" type="hinge" axis="0 1 0" range="0 60"/>
                    <geom name="r_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" material="mat_skin"/>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor joint="l_hip" gear="100"/><motor joint="r_hip" gear="100"/>
        <motor joint="l_knee" gear="50"/><motor joint="r_knee" gear="50"/>
    </actuator>
</mujoco>
"""

# 2. 运行逻辑：确保缩进完全正确
def main():
    try:
        model = mujoco.MjModel.from_xml_string(model_xml)
        data = mujoco.MjData(model)
        print("--- MuJoCo 仿真环境启动成功 ---")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                # 随机动作
                data.ctrl[:] = np.random.uniform(-0.5, 0.5, model.nu)
                mujoco.mj_step(model, data)
                
                # 如果摔倒自动重置
                if data.qpos[2] < 0.4:
                    mujoco.mj_resetData(model, data)
                    data.qpos[2] = 1.25
                
                viewer.sync()
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except Exception as e:
        print(f"程序运行过程中发生错误: {e}")

if __name__ == "__main__":
    main()
