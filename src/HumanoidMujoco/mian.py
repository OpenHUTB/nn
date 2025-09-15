# 标准库
import time

# 第三方库
import mujoco
from mujoco import viewer


def main():
    """
    主函数：使用嵌入式XML模型运行MuJoCo物理模拟

    模型定义直接以字符串形式嵌入代码中，无需外部文件
    """
    # 嵌入式XML模型定义 - 单摆系统
    xml_model = """
    <mujoco model="pendulum">
        <option timestep="0.01" gravity="0 0 -9.81"/>

        <default>
            <joint armature="0.1" damping="0.5"/>
            <geom conaffinity="0" condim="3" friction="1 0.1 0.1"/>
        </default>

        <worldbody>
            <!-- 地面 -->
            <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>

            <!-- 光源 -->
            <light pos="0 0 3" dir="0 0 -1"/>

            <!-- 摆锤系统 -->
            <body name="pendulum" pos="0 0 2">
                <joint name="hinge" type="hinge" axis="0 1 0"/>
                <geom name="rod" type="capsule" fromto="0 0 0 0 0 -1" size="0.05" rgba="0.8 0.2 0.2 1"/>
                <geom name="bob" type="sphere" size="0.1" pos="0 0 -1" rgba="0.2 0.6 0.8 1"/>
            </body>
        </worldbody>
    </mujoco>
    """

    # 从XML字符串加载模型（无需外部文件）
    model = mujoco.MjModel.from_xml_string(xml_model)

    # 初始化模拟数据
    data = mujoco.MjData(model)

    # 设置初始角度，让摆锤有一个初始偏角
    data.qpos[0] = 1.0  # 约57度

    # 启动可视化窗口
    with viewer.launch_passive(model, data) as v:
        # 运行20秒模拟（60步/秒 × 20秒）
        for _ in range(1200):
            # 推进模拟一步
            mujoco.mj_step(model, data)

            # 打印摆锤角度信息
            print(f"时间: {data.time:.2f}, 摆角: {data.qpos[0]:.2f} rad")

            # 更新可视化
            v.sync()

            # 控制可视化帧率
            time.sleep(0.01)


if __name__ == "__main__":
    main()
