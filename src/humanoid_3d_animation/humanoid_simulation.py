import mujoco
import mujoco.viewer as viewer
import os
import time
import math
import threading  # 新增：用于监听控制台输入（实现重置指令）
import numpy as np

def create_humanoid_xml(file_path):
    """
    自动创建humanoid.xml文件并写入模型代码
    优化点：XML内容格式化，增加注释，提升可读性
    """
    xml_content = """<mujoco model="simple_humanoid">
  <!-- 编译器设置：角度单位为弧度，从几何形状推导惯性 -->
  <compiler angle="radian" inertiafromgeom="true"/>
  <!-- 仿真参数：时间步长0.005s，重力加速度9.81m/s²（z轴负方向） -->
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <!-- 可视化全局设置：默认相机视角 -->
  <visual>
    <global azimuth="135" elevation="-30" perspective="0.01"/>
  </visual>

  <!-- 世界体：包含灯光、地面和人形机器人 -->
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>

    <!-- 骨盆（根节点）：包含自由关节，允许六自由度运动 -->
    <body name="pelvis" pos="0 0 1.0">
      <joint name="root" type="free"/>
      <geom name="pelvis_geom" type="capsule" size="0.1" fromto="0 0 0 0 0 0.2" rgba="0.5 0.5 0.9 1"/>

      <!-- 躯干 -->
      <body name="torso" pos="0 0 0.2">
        <geom name="torso_geom" type="capsule" size="0.1" fromto="0 0 0 0 0 0.3" rgba="0.5 0.5 0.9 1"/>

        <!-- 头部 -->
        <body name="head" pos="0 0 0.3">
          <geom name="head_geom" type="sphere" size="0.15" pos="0 0 0" rgba="0.8 0.5 0.5 1"/>
        </body>

        <!-- 左手臂：肩关节+肘关节 -->
        <body name="left_arm" pos="0.15 0 0.15">
          <joint name="left_shoulder" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <geom name="left_upper_arm" type="capsule" size="0.05" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          <body name="left_forearm" pos="0 0 0.2">
            <joint name="left_elbow" type="hinge" axis="1 0 0" range="-1.57 0"/>
            <geom name="left_forearm_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          </body>
        </body>

        <!-- 右手臂：肩关节+肘关节 -->
        <body name="right_arm" pos="-0.15 0 0.15">
          <joint name="right_shoulder" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <geom name="right_upper_arm" type="capsule" size="0.05" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          <body name="right_forearm" pos="0 0 0.2">
            <joint name="right_elbow" type="hinge" axis="1 0 0" range="-1.57 0"/>
            <geom name="right_forearm_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          </body>
        </body>

        <!-- 左腿部：髋关节+膝关节 -->
        <body name="left_leg" pos="0.05 0 -0.2">
          <joint name="left_hip" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <geom name="left_thigh" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1"/>
          <body name="left_calf" pos="0 0 -0.3">
            <joint name="left_knee" type="hinge" axis="1 0 0" range="0 1.57"/>
            <geom name="left_calf_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1"/>
          </body>
        </body>

        <!-- 右腿部：髋关节+膝关节 -->
        <body name="right_leg" pos="-0.05 0 -0.2">
          <joint name="right_hip" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <geom name="right_thigh" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1"/>
          <body name="right_calf" pos="0 0 -0.3">
            <joint name="right_knee" type="hinge" axis="1 0 0" range="0 1.57"/>
            <geom name="right_calf_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 执行器：添加阻尼和电机控制（新增电机，原仅阻尼无法主动控制） -->
  <actuator>
    <!-- 手臂关节：阻尼+电机 -->
    <motor name="left_shoulder_motor" joint="left_shoulder" ctrlrange="-1.57 1.57" gear="10"/>
    <damping joint="left_shoulder" damping="0.1"/>
    <motor name="right_shoulder_motor" joint="right_shoulder" ctrlrange="-1.57 1.57" gear="10"/>
    <damping joint="right_shoulder" damping="0.1"/>
    <motor name="left_elbow_motor" joint="left_elbow" ctrlrange="-1.57 0" gear="10"/>
    <damping joint="left_elbow" damping="0.1"/>
    <motor name="right_elbow_motor" joint="right_elbow" ctrlrange="-1.57 0" gear="10"/>
    <damping joint="right_elbow" damping="0.1"/>

    <!-- 腿部关节：阻尼+电机 -->
    <motor name="left_hip_motor" joint="left_hip" ctrlrange="-1.57 1.57" gear="10"/>
    <damping joint="left_hip" damping="0.1"/>
    <motor name="right_hip_motor" joint="right_hip" ctrlrange="-1.57 1.57" gear="10"/>
    <damping joint="right_hip" damping="0.1"/>
    <motor name="left_knee_motor" joint="left_knee" ctrlrange="0 1.57" gear="10"/>
    <damping joint="left_knee" damping="0.1"/>
    <motor name="right_knee_motor" joint="right_knee" ctrlrange="0 1.57" gear="10"/>
    <damping joint="right_knee" damping="0.1"/>
  </actuator>
</mujoco>"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"✅ 已自动在 {file_path} 创建humanoid.xml文件！")

def get_joint_ctrl_id(model, joint_name):
    """
    根据关节名称获取对应的控制索引（替代硬编码索引，提升鲁棒性）
    参数：
        model: MuJoCo的MjModel对象
        joint_name: 关节名称（如"left_shoulder"）
    返回：
        控制索引（int），若不存在返回-1
    """
    # 先获取电机执行器的ID（对应actuator中的motor）
    motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_motor")
    if motor_id == -1:
        # 若没有电机，尝试获取阻尼执行器ID（兼容旧版）
        motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
    return motor_id

def print_robot_state(data, joint_names, interval=1.0):
    """
    周期性打印机器人关节状态（位置、控制信号）
    参数：
        data: MuJoCo的MjData对象
        joint_names: 需要打印的关节名称列表
        interval: 打印时间间隔（秒）
    """
    current_time = data.time
    if not hasattr(print_robot_state, "last_print_time"):
        print_robot_state.last_print_time = 0.0  # 初始化上次打印时间

    if current_time - print_robot_state.last_print_time >= interval:
        print(f"\n===== 机器人状态（时间：{current_time:.2f}s）=====")
        for name in joint_names:
            # 获取关节ID和控制索引
            joint_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ctrl_id = get_joint_ctrl_id(data.model, name)
            if joint_id != -1 and ctrl_id != -1:
                # 根关节是自由关节（7个自由度），普通关节的qpos索引偏移7位
                qpos_index = 7 + joint_id  # 自由关节占前7个qpos
                if qpos_index < len(data.qpos):
                    print(f"关节 {name}: 位置 = {data.qpos[qpos_index]:.2f} rad, 控制信号 = {data.ctrl[ctrl_id]:.2f}")
        print_robot_state.last_print_time = current_time

def reset_robot(model, data):
    """
    重置机器人到初始状态
    参数：
        model: MuJoCo的MjModel对象
        data: MuJoCo的MjData对象
    """
    mujoco.mj_resetData(model, data)  # 重置动力学数据
    data.qpos[0:7] = [0, 0, 1.0, 1, 0, 0, 0]  # 重置根关节位置（x,y,z,四元数）
    print("\n🔄 机器人已重置到初始状态！")

def input_listener(reset_flag):
    """
    后台线程：监听控制台输入，输入'r'则设置重置标记
    参数：
        reset_flag: 共享的布尔列表（用于跨线程传递标记，列表是可变对象）
    """
    while True:
        user_input = input().strip().lower()
        if user_input == 'r':
            reset_flag[0] = True
        elif user_input == 'q':
            print("📤 收到退出指令，仿真将结束...")
            break

def run_humanoid_simulation():
    """
    优化后的仿真主函数：修复API兼容问题，用控制台输入实现重置
    """
    # 优化：使用用户目录拼接路径，避免硬编码用户名（更通用）
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    model_path = os.path.join(desktop_path, "humanoid.xml")

    # 打印路径信息
    print(f"===== 模型文件路径 =====")
    print(f"模型文件完整路径：{model_path}")
    print(f"========================")

    # 检查并创建文件
    if not os.path.exists(model_path):
        create_humanoid_xml(model_path)
    else:
        print("ℹ️ humanoid.xml文件已存在，无需重新创建！")

    # 加载模型：直接读取内容，用字符串加载（彻底解决中文路径问题）
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            xml_content = f.read()
        print("✅ Python内置函数已成功读取文件，权限正常！")
    except Exception as e:
        print(f"❌ Python读取文件失败，权限/路径问题：{e}")
        return

    # 加载MuJoCo模型
    try:
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)
        print("✅ 从字符串加载模型成功！开始启动仿真...")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 定义需要控制的关节名称
    joint_names = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_hip", "right_hip",
        "left_knee", "right_knee"
    ]

    # 新增：共享重置标记（用列表实现跨线程可变对象）
    reset_flag = [False]
    # 启动后台线程监听控制台输入
    input_thread = threading.Thread(target=input_listener, args=(reset_flag,), daemon=True)
    input_thread.start()

    # 运行仿真可视化
    with viewer.launch_passive(model, data) as v:
        # 相机跟随设置（跟随骨盆位置）
        pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_body_id != -1:
            v.cam.trackbodyid = pelvis_body_id  # 跟踪骨盆体
        v.cam.distance = 2.0  # 相机距离跟随目标的距离
        v.cam.azimuth = 45    # 相机方位角
        v.cam.elevation = -20 # 相机仰角

        print("\n📌 仿真操作提示：")
        print("  - 在控制台输入 'r' 并回车，重置机器人到初始状态")
        print("  - 在控制台输入 'q' 并回车，退出仿真")
        print("  - 关闭可视化窗口也可退出仿真")

        print("🚀 仿真开始...")

        while v.is_running():
            # 检查重置标记：如果为True，执行重置并重置标记
            if reset_flag[0]:
                reset_robot(model, data)
                reset_flag[0] = False  # 重置标记置为False

            # ========== 关节主动运动控制（用关节名称获取索引） ==========
            t = data.time  # 仿真累计时间
            # 1. 手臂运动：左肩关节和右肩关节做相反的正弦运动（2Hz频率）
            left_shoulder_id = get_joint_ctrl_id(model, "left_shoulder")
            right_shoulder_id = get_joint_ctrl_id(model, "right_shoulder")
            if left_shoulder_id != -1:
                data.ctrl[left_shoulder_id] = math.sin(t * 2) * 1.0  # 左肩关节
            if right_shoulder_id != -1:
                data.ctrl[right_shoulder_id] = -math.sin(t * 2) * 1.0  # 右肩关节（反向）

            # 2. 肘部运动：跟随肩部运动，幅度更小
            left_elbow_id = get_joint_ctrl_id(model, "left_elbow")
            right_elbow_id = get_joint_ctrl_id(model, "right_elbow")
            if left_elbow_id != -1:
                data.ctrl[left_elbow_id] = math.sin(t * 2) * 0.5  # 左肘部
            if right_elbow_id != -1:
                data.ctrl[right_elbow_id] = -math.sin(t * 2) * 0.5  # 右肘部（反向）

            # 3. 腿部运动：左髋和右髋做余弦运动（2Hz频率，和手臂同步）
            left_hip_id = get_joint_ctrl_id(model, "left_hip")
            right_hip_id = get_joint_ctrl_id(model, "right_hip")
            if left_hip_id != -1:
                data.ctrl[left_hip_id] = math.cos(t * 2) * 0.8  # 左髋
            if right_hip_id != -1:
                data.ctrl[right_hip_id] = -math.cos(t * 2) * 0.8  # 右髋（反向）

            # 4. 膝盖运动：跟随髋部运动，幅度稍小
            left_knee_id = get_joint_ctrl_id(model, "left_knee")
            right_knee_id = get_joint_ctrl_id(model, "right_knee")
            if left_knee_id != -1:
                data.ctrl[left_knee_id] = math.cos(t * 2) * 0.6  # 左膝盖
            if right_knee_id != -1:
                data.ctrl[right_knee_id] = -math.cos(t * 2) * 0.6  # 右膝盖（反向）
            # ================================================

            # 执行仿真步
            mujoco.mj_step(model, data)
            # 更新可视化
            v.sync()
            # 控制仿真速度（使用模型时间步长，更匹配物理仿真）
            time.sleep(model.opt.timestep)

            # 周期性打印机器人状态（每1秒打印一次）
            print_robot_state(data, joint_names, interval=1.0)

        print("\n🏁 仿真结束！")

if __name__ == "__main__":
    run_humanoid_simulation()