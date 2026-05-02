
import mujoco
import mujoco.viewer
import numpy as np
import time

# =========================
# 第五次作业：高级稳定步行系统
# =========================

model_xml = """
<mujoco model="humanoid_fifth_assignment">

    <compiler angle="degree" inertiafromgeom="true"/>

    <option timestep="0.0015"
            integrator="RK4"
            gravity="0 0 -9.81"
            viscosity="0.1"
            density="1000"/>

    <default>
        <joint damping="8" stiffness="15" armature="0.05"/>
        <geom friction="1.8 0.2 0.1" condim="6"/>
    </default>

    <asset>
        <texture type="skybox" builtin="gradient"
                 rgb1="0.4 0.6 0.9"
                 rgb2="0 0 0"
                 width="512"
                 height="512"/>

        <texture name="ground_tex"
                 builtin="checker"
                 rgb1="0.2 0.3 0.4"
                 rgb2="0.1 0.15 0.2"
                 width="512"
                 height="512"/>

        <material name="ground_mat"
                  texture="ground_tex"
                  reflectance="0.25"/>

        <material name="skin" rgba="0.85 0.65 0.45 1"/>
        <material name="cloth" rgba="0.15 0.45 0.9 1"/>
        <material name="foot" rgba="0.2 0.2 0.2 1"/>
    </asset>

    <worldbody>

        <light pos="3 3 6" dir="0 0 -1" castshadow="true"/>

        <geom name="floor"
              type="plane"
              size="15 15 0.2"
              material="ground_mat"/>

        <!-- 主躯干 -->
        <body name="torso" pos="0 0 0.88">

            <freejoint name="root"/>

            <geom name="torso_geom"
                  type="capsule"
                  fromto="0 -0.12 0 0 0.12 0"
                  size="0.11"
                  mass="16"
                  material="cloth"/>

            <!-- 胸部 -->
            <geom type="sphere"
                  pos="0 0 0.10"
                  size="0.09"
                  mass="3"
                  material="cloth"/>

            <!-- 头 -->
            <geom name="head"
                  type="sphere"
                  pos="0 0 0.28"
                  size="0.11"
                  mass="2.5"
                  material="skin"/>

            <!-- 左臂 -->
            <body name="left_arm" pos="0 0.20 0.10">
                <joint name="l_shoulder"
                       type="hinge"
                       axis="0 1 0"
                       range="-90 90"
                       damping="12"/>

                <geom type="capsule"
                      fromto="0 0 0 0 0 -0.32"
                      size="0.045"
                      mass="2"
                      material="skin"/>

                <!-- 左手 -->
                <body name="left_hand" pos="0 0 -0.32">
                    <geom type="sphere"
                          size="0.05"
                          mass="0.8"
                          material="skin"/>
                </body>
            </body>

            <!-- 右臂 -->
            <body name="right_arm" pos="0 -0.20 0.10">
                <joint name="r_shoulder"
                       type="hinge"
                       axis="0 1 0"
                       range="-90 90"
                       damping="12"/>

                <geom type="capsule"
                      fromto="0 0 0 0 0 -0.32"
                      size="0.045"
                      mass="2"
                      material="skin"/>

                <!-- 右手 -->
                <body name="right_hand" pos="0 0 -0.32">
                    <geom type="sphere"
                          size="0.05"
                          mass="0.8"
                          material="skin"/>
                </body>
            </body>

            <!-- 左腿 -->
            <body name="left_leg" pos="0 0.11 -0.05">

                <joint name="l_hip"
                       type="hinge"
                       axis="0 1 0"
                       range="-35 35"
                       damping="35"/>

                <geom type="capsule"
                      fromto="0 0 0 0 0 -0.38"
                      size="0.075"
                      mass="5.5"
                      material="skin"/>

                <body name="left_shin" pos="0 0 -0.38">

                    <joint name="l_knee"
                           type="hinge"
                           axis="0 1 0"
                           range="0 70"
                           damping="45"/>

                    <geom type="capsule"
                          fromto="0 0 0 0 0 -0.34"
                          size="0.065"
                          mass="4"
                          material="skin"/>

                    <body name="left_foot" pos="0 0 -0.34">

                        <joint name="l_ankle"
                               type="hinge"
                               axis="0 1 0"
                               range="-18 18"
                               damping="60"/>

                        <geom name="l_foot_geom"
                              type="box"
                              size="0.16 0.08 0.035"
                              pos="0.07 0 -0.015"
                              mass="2.8"
                              material="foot"/>
                    </body>
                </body>
            </body>

            <!-- 右腿 -->
            <body name="right_leg" pos="0 -0.11 -0.05">

                <joint name="r_hip"
                       type="hinge"
                       axis="0 1 0"
                       range="-35 35"
                       damping="35"/>

                <geom type="capsule"
                      fromto="0 0 0 0 0 -0.38"
                      size="0.075"
                      mass="5.5"
                      material="skin"/>

                <body name="right_shin" pos="0 0 -0.38">

                    <joint name="r_knee"
                           type="hinge"
                           axis="0 1 0"
                           range="0 70"
                           damping="45"/>

                    <geom type="capsule"
                          fromto="0 0 0 0 0 -0.34"
                          size="0.065"
                          mass="4"
                          material="skin"/>

                    <body name="right_foot" pos="0 0 -0.34">

                        <joint name="r_ankle"
                               type="hinge"
                               axis="0 1 0"
                               range="-18 18"
                               damping="60"/>

                        <geom name="r_foot_geom"
                              type="box"
                              size="0.16 0.08 0.035"
                              pos="0.07 0 -0.015"
                              mass="2.8"
                              material="foot"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <!-- 执行器 -->
    <actuator>

        <position joint="l_hip" kp="1700" ctrlrange="-35 35"/>
        <position joint="r_hip" kp="1700" ctrlrange="-35 35"/>

        <position joint="l_knee" kp="1400" ctrlrange="0 70"/>
        <position joint="r_knee" kp="1400" ctrlrange="0 70"/>

        <position joint="l_ankle" kp="1200" ctrlrange="-18 18"/>
        <position joint="r_ankle" kp="1200" ctrlrange="-18 18"/>

        <position joint="l_shoulder" kp="500" ctrlrange="-90 90"/>
        <position joint="r_shoulder" kp="500" ctrlrange="-90 90"/>

    </actuator>

</mujoco>
"""


# =========================
# PID 控制器
# =========================
class PID:

    def __init__(self, kp, ki, kd, out_min=-100, out_max=100):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.out_min = out_min
        self.out_max = out_max

        self.integral = 0
        self.last_error = 0

    def update(self, error, dt):

        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)

        derivative = (error - self.last_error) / dt

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.last_error = error

        return np.clip(output, self.out_min, self.out_max)


# =========================
# 高级生物控制器
# =========================
class AdvancedBioController:

    def __init__(self, model):

        self.model = model

        self.step_period = 1.1
        self.walk_amplitude = 18

        # 姿态稳定 PID
        self.pitch_pid = PID(7.5, 0.6, 2.5, -12, 12)
        self.height_pid = PID(150, 5, 40, -8, 8)

    def get_pitch(self, q):

        sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
        return np.arcsin(np.clip(sinp, -1, 1))

    def compute(self, data, dt):

        ctrl = np.zeros(self.model.nu)

        t = data.time

        # 四元数
        q = data.qpos[3:7]

        pitch = self.get_pitch(q)

        # 周期步态
        phase = (t % self.step_period) / self.step_period

        # 正弦步态
        walk_wave = np.sin(2 * np.pi * phase)

        # 左右支撑切换
        left_support = phase < 0.5

        # 髋关节
        hip_swing = self.walk_amplitude * walk_wave

        if left_support:
            ctrl[0] = hip_swing
            ctrl[1] = -5
        else:
            ctrl[0] = -5
            ctrl[1] = -hip_swing

        # 膝关节控制
        knee_wave = 30 * np.abs(walk_wave)

        ctrl[2] = knee_wave if not left_support else 6
        ctrl[3] = knee_wave if left_support else 6

        # 踝关节平衡控制
        balance = self.pitch_pid.update(-pitch, dt)

        ctrl[4] = balance
        ctrl[5] = balance

        # 手臂反向摆动
        arm_wave = -hip_swing * 1.3

        ctrl[6] = arm_wave
        ctrl[7] = -arm_wave

        # 高度恢复
        body_height = data.qpos[2]

        if body_height < 0.72:
            ctrl[2] += 10
            ctrl[3] += 10

        return ctrl


# =========================
# 主函数
# =========================
def main():

    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)

    controller = AdvancedBioController(model)

    # 初始高度
    data.qpos[2] = 0.84

    print("=" * 60)
    print("高级稳定步行机器人")
    print("优化内容：")
    print("1. 增强步态稳定性")
    print("2. 增加双臂摆动")
    print("3. 增加姿态 PID 控制")
    print("4. 增加恢复平衡功能")
    print("5. 提高摩擦与抗摔能力")
    print("=" * 60)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -18

        while viewer.is_running():

            start = time.time()

            dt = model.opt.timestep

            # 控制输出
            data.ctrl[:] = controller.compute(data, dt)

            # 物理步进
            mujoco.mj_step(model, data)

            # 输出状态
            if int(data.time * 100) % 100 == 0:

                print(
                    f"时间:{data.time:.2f}s | "
                    f"高度:{data.qpos[2]:.3f}m | "
                    f"前进距离:{data.qpos[0]:.3f}m"
                )

            viewer.sync()

            elapsed = time.time() - start

            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
