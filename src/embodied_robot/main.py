import mujoco
import time
from mujoco import viewer


def main():
    # 定义简单的双轮机器人模型
    robot_xml = """
    <mujoco model="simple_robot">
        <option timestep="0.01"/>
        <worldbody>
            <!-- 地面 -->
            <body name="ground">
                <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
            </body>

            <!-- 机器人主体 -->
            <body name="base" pos="0 0 0.1">
                <inertial pos="0 0 0" mass="1.0" diaginertia="0.1 0.1 0.1"/>
                <geom type="box" size="0.2 0.1 0.1" rgba="0.2 0.4 0.8 1"/>

                <!-- 左轮 -->
                <body name="left_wheel" pos="0 -0.15 0">
                    <joint name="left_joint" type="hinge" axis="0 1 0"/>
                    <geom type="cylinder" size="0.05 0.08" rgba="0.5 0.5 0.5 1"/>
                    <inertial pos="0 0 0" mass="0.2" diaginertia="0.01 0.01 0.01"/>
                </body>

                <!-- 右轮 -->
                <body name="right_wheel" pos="0 0.15 0">
                    <joint name="right_joint" type="hinge" axis="0 1 0"/>
                    <geom type="cylinder" size="0.05 0.08" rgba="0.5 0.5 0.5 1"/>
                    <inertial pos="0 0 0" mass="0.2" diaginertia="0.01 0.01 0.01"/>
                </body>
            </body>
        </worldbody>

        <!-- 电机执行器 -->
        <actuator>
            <motor name="left_motor" joint="left_joint" gear="100"/>
            <motor name="right_motor" joint="right_joint" gear="100"/>
        </actuator>
    </mujoco>
    """

    # 加载模型
    model = mujoco.MjModel.from_xml_string(robot_xml)
    data = mujoco.MjData(model)

    # 初始化可视化器
    viewer_instance = viewer.launch_passive(model, data)

    # 设置直行速度（左右轮相同速度）
    forward_speed = 1.0
    data.ctrl[0] = forward_speed  # 左轮速度
    data.ctrl[1] = forward_speed  # 右轮速度

    # 运行仿真10秒
    total_time = 10.0
    steps = int(total_time / model.opt.timestep)

    start_time = time.time()
    for _ in range(steps):
        # 执行仿真步
        mujoco.mj_step(model, data)

        # 更新可视化
        viewer_instance.sync()

        # 维持实时速率
        elapsed = time.time() - start_time
        expected = _ * model.opt.timestep
        if elapsed < expected:
            time.sleep(expected - elapsed)

    print(f"仿真完成，机器人最终位置: {data.body('base').xpos}")
    viewer_instance.close()


if __name__ == "__main__":
    main()

