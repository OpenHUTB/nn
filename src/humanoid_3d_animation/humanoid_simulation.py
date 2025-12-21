import mujoco
import mujoco.viewer as viewer
import os
import time
# 新增：导入数学库，用于生成周期性的正弦/余弦运动信号
import math

def create_humanoid_xml(file_path):
    """自动创建humanoid.xml文件并写入模型代码"""
    xml_content = """<mujoco model="simple_humanoid">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <visual>
    <global azimuth="135" elevation="-30" perspective="0.01"/>
  </visual>

  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>

    <body name="pelvis" pos="0 0 1.0">
      <joint name="root" type="free"/>
      <geom name="pelvis_geom" type="capsule" size="0.1" fromto="0 0 0 0 0 0.2" rgba="0.5 0.5 0.9 1"/>

      <body name="torso" pos="0 0 0.2">
        <geom name="torso_geom" type="capsule" size="0.1" fromto="0 0 0 0 0 0.3" rgba="0.5 0.5 0.9 1"/>

        <body name="head" pos="0 0 0.3">
          <geom name="head_geom" type="sphere" size="0.15" pos="0 0 0" rgba="0.8 0.5 0.5 1"/>
        </body>

        <body name="left_arm" pos="0.15 0 0.15">
          <joint name="left_shoulder" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <geom name="left_upper_arm" type="capsule" size="0.05" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          <body name="left_forearm" pos="0 0 0.2">
            <joint name="left_elbow" type="hinge" axis="1 0 0" range="-1.57 0"/>
            <geom name="left_forearm_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          </body>
        </body>

        <body name="right_arm" pos="-0.15 0 0.15">
          <joint name="right_shoulder" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <geom name="right_upper_arm" type="capsule" size="0.05" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          <body name="right_forearm" pos="0 0 0.2">
            <joint name="right_elbow" type="hinge" axis="1 0 0" range="-1.57 0"/>
            <geom name="right_forearm_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" rgba="0.5 0.9 0.5 1"/>
          </body>
        </body>

        <body name="left_leg" pos="0.05 0 -0.2">
          <joint name="left_hip" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <geom name="left_thigh" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1"/>
          <body name="left_calf" pos="0 0 -0.3">
            <joint name="left_knee" type="hinge" axis="1 0 0" range="0 1.57"/>
            <geom name="left_calf_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.5 1"/>
          </body>
        </body>

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

  <actuator>
    <damping joint="left_shoulder" damping="0.1"/>
    <damping joint="right_shoulder" damping="0.1"/>
    <damping joint="left_elbow" damping="0.1"/>
    <damping joint="right_elbow" damping="0.1"/>
    <damping joint="left_hip" damping="0.1"/>
    <damping joint="right_hip" damping="0.1"/>
    <damping joint="left_knee" damping="0.1"/>
    <damping joint="right_knee" damping="0.1"/>
  </actuator>
</mujoco>"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"已自动在 {file_path} 创建humanoid.xml文件！")

def run_humanoid_simulation():
    # 直接写死桌面路径（替换成你的实际用户名，这里保留你的用户名）
    # 注意：路径中的反斜杠用双反斜杠，或单反斜杠加r前缀
    model_path = r"C:\Users\欧阳志威1\Desktop\humanoid.xml"  # r前缀表示原始字符串，避免转义

    # 打印路径
    print(f"===== 模型文件路径 =====")
    print(f"模型文件完整路径：{model_path}")
    print(f"========================")

    # 检查并创建文件
    if not os.path.exists(model_path):
        create_humanoid_xml(model_path)
    else:
        print("humanoid.xml文件已存在，无需重新创建！")

    # 加载模型（新增：用Python内置的open函数测试是否能读取文件）
    try:
        # 先测试Python是否能读取该文件（排除系统权限问题）
        with open(model_path, "r", encoding="utf-8") as f:
            f.read()
        print("Python内置函数已成功读取文件，权限正常！")
    except Exception as e:
        print(f"Python读取文件失败，权限/路径问题：{e}")
        return

    # 加载MuJoCo模型
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print("模型加载成功！开始启动仿真...")
    except Exception as e:
        print(f"MuJoCo加载模型失败：{e}")
        # 备选方案：从字符串加载模型（绕过文件读取）
        print("尝试从字符串直接加载模型...")
        with open(model_path, "r", encoding="utf-8") as f:
            xml_str = f.read()
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        print("从字符串加载模型成功！开始启动仿真...")

    # 运行仿真
    with viewer.launch_passive(model, data) as v:
        sim_steps = 10000
        step_interval = 0.01
        print("仿真开始，按窗口关闭按钮退出...")
        for _ in range(sim_steps):
            # ========== 新增：关节主动运动控制核心代码 ==========
            # 1. 手臂运动：左肩关节和右肩关节做相反的正弦运动（周期性摆动）
            # data.time：仿真累计时间（秒），驱动周期性运动
            # math.sin(data.time * 2)：2Hz频率的正弦波，取值[-1,1]
            # * 1.0：运动幅度（可调整，越大摆动越剧烈）
            data.ctrl[0] = math.sin(data.time * 2) * 1.0  # 左肩关节（对应actuator索引0）
            data.ctrl[1] = -math.sin(data.time * 2) * 1.0 # 右肩关节（对应actuator索引1，负号表示反向）

            # 2. 肘部运动：跟随肩部运动，幅度更小
            data.ctrl[2] = math.sin(data.time * 2) * 0.5  # 左肘部（索引2）
            data.ctrl[3] = -math.sin(data.time * 2) * 0.5 # 右肘部（索引3）

            # 3. 腿部运动：左髋和右髋做余弦运动（与正弦波相位差90°，运动更协调）
            data.ctrl[4] = math.cos(data.time * 1) * 0.8  # 左髋（索引4）
            data.ctrl[5] = -math.cos(data.time * 1) * 0.8 # 右髋（索引5）

            # 4. 膝盖运动：跟随髋部运动，幅度稍小
            data.ctrl[6] = math.cos(data.time * 1) * 0.6  # 左膝盖（索引6）
            data.ctrl[7] = -math.cos(data.time * 1) * 0.6 # 右膝盖（索引7）
            # ================================================

            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(step_interval)
        print("仿真结束！")

if __name__ == "__main__":
    run_humanoid_simulation()