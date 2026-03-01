import mujoco
import mujoco.viewer
import numpy as np
import glfw
import time
import threading

# 加载模型
model = mujoco.MjModel.from_xml_string("""
<mujoco model="optimized_robotic_arm_3_finger_gripper">
  <compiler angle="radian" inertiafromgeom="true" meshdir="assets/" texturedir="textures/"/>
  <option timestep="0.001" gravity="0 0 -9.81" iterations="100" solver="Newton" cone="elliptic"/>
  <size njmax="500" nconmax="100"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
  </visual>
  <default>
    <joint armature="0.1" damping="1.5" limited="true" solimplimit="0 0.99 0.001" solreflimit="0.001 0.5"/>
    <geom condim="6" friction="1.0 0.5 0.001" solmix="1.0" solref="0.02 1" margin="0.001" gap="0"/>
    <motor ctrllimited="true" ctrlrange="-2 2" forcelimited="true" forcerange="-100 100"/>
    <site size="0.005" rgba="1 0 0 0.8" type="sphere"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.1 0.2 0.3" width="1024" height="1024"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="300" height="300" mark="edge" markrgb="0.8 0.8 0.8"/>
    <texture name="metal" type="cube" builtin="flat" mark="cross" width="256" height="256" rgb1="0.5 0.5 0.5" rgb2="0.7 0.7 0.7" markrgb="1 1 1"/>
    <material name="grid_mat" texture="grid" texrepeat="20 20" texuniform="true" reflectance="0.2"/>
    <material name="metal_mat" texture="metal" texuniform="true" reflectance="0.5" specular="1" shininess="0.5"/>
    <material name="arm_mat" rgba="0.6 0.6 0.7 1" specular="0.5" shininess="0.3"/>
    <material name="gripper_mat" rgba="0.3 0.3 0.8 1" specular="0.8" shininess="0.5"/>
    <material name="finger_mat" rgba="0.2 0.5 0.8 1" specular="0.5" shininess="0.4"/>
    <material name="red_mat" rgba="1 0.2 0.2 1"/>
    <material name="green_mat" rgba="0.2 0.8 0.2 1"/>
    <material name="blue_mat" rgba="0.2 0.5 1 1"/>
    <material name="yellow_mat" rgba="1 0.8 0.2 1"/>
    <material name="purple_mat" rgba="0.7 0.2 1 1"/>
  </asset>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="3 3 0.1" material="grid_mat" condim="3"/>
    <light directional="true" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3" pos="0 0 4" dir="0 0 -1"/>
    <light directional="true" diffuse="0.5 0.5 0.6" specular="0.1 0.1 0.1" pos="3 3 2" dir="-1 -1 -1"/>
    <light directional="true" diffuse="0.4 0.4 0.5" specular="0.05 0.05 0.05" pos="-3 -3 2" dir="1 1 -1"/>
    <body name="base" pos="0 0 0.05">
      <geom name="base_geom" type="cylinder" size="0.15 0.08" material="metal_mat" mass="5.0"/>
      <site name="base_site" pos="0 0 0" size="0.01"/>
      <body name="shoulder" pos="0 0 0.1">
        <joint name="joint1" type="hinge" pos="0 0 0" axis="0 0 1" range="-3.1416 3.1416" damping="2.0" armature="0.2"/>
        <geom name="shoulder_geom" type="cylinder" size="0.1 0.08" material="arm_mat" mass="2.0"/>
        <site name="shoulder_site" pos="0 0 0" size="0.008"/>
        <body name="upper_arm" pos="0 0 0.09">
          <joint name="joint2" type="hinge" pos="0 0 0" axis="0 1 0" range="-2.0 2.0" damping="1.8" armature="0.15"/>
          <geom name="upper_arm_geom" type="capsule" fromto="0 0 0 0 0 0.35" size="0.075" material="arm_mat" mass="1.5"/>
          <site name="upper_arm_site" pos="0 0 0.175" size="0.008"/>
          <body name="forearm" pos="0 0 0.35">
            <joint name="joint3" type="hinge" pos="0 0 0" axis="0 1 0" range="-2.5 0.5" damping="1.5" armature="0.1"/>
            <geom name="forearm_geom" type="capsule" fromto="0 0 0 0 0 0.3" size="0.065" material="arm_mat" mass="1.2"/>
            <site name="forearm_site" pos="0 0 0.15" size="0.008"/>
            <body name="wrist1" pos="0 0 0.3">
              <joint name="joint4" type="hinge" pos="0 0 0" axis="0 0 1" range="-2.0 2.0" damping="1.0" armature="0.05"/>
              <geom name="wrist1_geom" type="capsule" fromto="0 0 0 0 0 0.15" size="0.055" material="arm_mat" mass="0.5"/>
              <site name="wrist1_site" pos="0 0 0.075" size="0.008"/>
              <body name="wrist2" pos="0 0 0.15">
                <joint name="joint5" type="hinge" pos="0 0 0" axis="0 1 0" range="-1.8 1.8" damping="0.8" armature="0.04"/>
                <geom name="wrist2_geom" type="capsule" fromto="0 0 0 0.12 0 0" size="0.045" material="arm_mat" mass="0.3"/>
                <site name="wrist2_site" pos="0.06 0 0" size="0.008"/>
                <body name="wrist3" pos="0.12 0 0">
                  <joint name="joint6" type="hinge" pos="0 0 0" axis="1 0 0" range="-1.8 1.8" damping="0.6" armature="0.03"/>
                  <geom name="wrist3_geom" type="cylinder" size="0.04 0.03" material="arm_mat" mass="0.2"/>
                  <site name="ee_site" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
                  <body name="gripper_base" pos="0.05 0 0" euler="0 0 0">
                    <geom name="gripper_base_geom" type="cylinder" size="0.045 0.035" material="gripper_mat" mass="0.15"/>
                    <site name="grip_center" pos="0 0 0" size="0.008" rgba="0 1 0 1"/>
                    <body name="finger1_base" pos="0.04 0 0">
                      <joint name="finger1_slide" type="slide" axis="0 0 1" range="-0.02 0.06" damping="0.4" stiffness="50"/>
                      <geom name="finger1_base_geom" type="box" size="0.018 0.01 0.035" material="finger_mat" mass="0.04"/>
                      <body name="finger1_proximal" pos="0 0 0.035">
                        <joint name="finger1_hinge1" type="hinge" axis="0 1 0" range="-0.2 1.5" damping="0.3" stiffness="30" armature="0.01"/>
                        <geom name="finger1_proximal_geom" type="box" size="0.015 0.008 0.028" material="finger_mat" mass="0.025"/>
                        <body name="finger1_distal" pos="0 0 0.028">
                          <joint name="finger1_hinge2" type="hinge" axis="0 1 0" range="0 1.2" damping="0.2" stiffness="20" armature="0.005"/>
                          <geom name="finger1_distal_geom" type="box" size="0.012 0.006 0.022" material="finger_mat" mass="0.018"/>
                          <site name="finger1_tip" pos="0 0 0.022" size="0.006" rgba="1 0.5 0 1"/>
                        </body>
                      </body>
                    </body>
                    <body name="finger2_base" pos="-0.02 0.035 0" euler="0 0 2.094">
                      <joint name="finger2_slide" type="slide" axis="0 0 1" range="-0.02 0.06" damping="0.4" stiffness="50"/>
                      <geom name="finger2_base_geom" type="box" size="0.018 0.01 0.035" material="finger_mat" mass="0.04"/>
                      <body name="finger2_proximal" pos="0 0 0.035">
                        <joint name="finger2_hinge1" type="hinge" axis="0 1 0" range="-0.2 1.5" damping="0.3" stiffness="30" armature="0.01"/>
                        <geom name="finger2_proximal_geom" type="box" size="0.015 0.008 0.028" material="finger_mat" mass="0.025"/>
                        <body name="finger2_distal" pos="0 0 0.028">
                          <joint name="finger2_hinge2" type="hinge" axis="0 1 0" range="0 1.2" damping="0.2" stiffness="20" armature="0.005"/>
                          <geom name="finger2_distal_geom" type="box" size="0.012 0.006 0.022" material="finger_mat" mass="0.018"/>
                          <site name="finger2_tip" pos="0 0 0.022" size="0.006" rgba="1 0.5 0 1"/>
                        </body>
                      </body>
                    </body>
                    <body name="finger3_base" pos="-0.02 -0.035 0" euler="0 0 -2.094">
                      <joint name="finger3_slide" type="slide" axis="0 0 1" range="-0.02 0.06" damping="0.4" stiffness="50"/>
                      <geom name="finger3_base_geom" type="box" size="0.018 0.01 0.035" material="finger_mat" mass="0.04"/>
                      <body name="finger3_proximal" pos="0 0 0.035">
                        <joint name="finger3_hinge1" type="hinge" axis="0 1 0" range="-0.2 1.5" damping="0.3" stiffness="30" armature="0.01"/>
                        <geom name="finger3_proximal_geom" type="box" size="0.015 0.008 0.028" material="finger_mat" mass="0.025"/>
                        <body name="finger3_distal" pos="0 0 0.028">
                          <joint name="finger3_hinge2" type="hinge" axis="0 1 0" range="0 1.2" damping="0.2" stiffness="20" armature="0.005"/>
                          <geom name="finger3_distal_geom" type="box" size="0.012 0.006 0.022" material="finger_mat" mass="0.018"/>
                          <site name="finger3_tip" pos="0 0 0.022" size="0.006" rgba="1 0.5 0 1"/>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="test_sphere" pos="0.2 0.1 0.1">
      <joint type="free" name="sphere_joint"/>
      <geom name="sphere_geom" type="sphere" size="0.06" material="red_mat" mass="0.25" friction="0.7 0.3 0.01"/>
      <site name="sphere_site" pos="0 0 0" size="0.008"/>
    </body>
    <body name="test_box" pos="0.1 -0.1 0.05">
      <joint type="free" name="box_joint"/>
      <geom name="box_geom" type="box" size="0.05 0.07 0.04" material="green_mat" mass="0.35" friction="0.8 0.4 0.01"/>
      <site name="box_site" pos="0 0 0" size="0.008"/>
    </body>
    <body name="test_cylinder" pos="0.3 0.0 0.1">
      <joint type="free" name="cylinder_joint"/>
      <geom name="cylinder_geom" type="cylinder" size="0.04 0.09" material="blue_mat" mass="0.3" friction="0.75 0.35 0.01"/>
      <site name="cylinder_site" pos="0 0 0" size="0.008"/>
    </body>
    <body name="test_capsule" pos="-0.1 0.2 0.12">
      <joint type="free" name="capsule_joint"/>
      <geom name="capsule_geom" type="capsule" fromto="0 -0.04 0 0 0.04 0" size="0.03" material="yellow_mat" mass="0.2" friction="0.65 0.25 0.01"/>
      <site name="capsule_site" pos="0 0 0" size="0.008"/>
    </body>
    <body name="test_ellipsoid" pos="0.0 -0.2 0.18">
      <joint type="free" name="ellipsoid_joint"/>
      <geom name="ellipsoid_geom" type="ellipsoid" size="0.05 0.07 0.04" material="purple_mat" mass="0.28" friction="0.6 0.2 0.01"/>
      <site name="ellipsoid_site" pos="0 0 0" size="0.008"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor_joint1" joint="joint1" gear="80" ctrlrange="-2.5 2.5" forcerange="-150 150"/>
    <motor name="motor_joint2" joint="joint2" gear="70" ctrlrange="-2.5 2.5" forcerange="-120 120"/>
    <motor name="motor_joint3" joint="joint3" gear="60" ctrlrange="-2.5 2.5" forcerange="-100 100"/>
    <motor name="motor_joint4" joint="joint4" gear="40" ctrlrange="-2.0 2.0" forcerange="-80 80"/>
    <motor name="motor_joint5" joint="joint5" gear="40" ctrlrange="-2.0 2.0" forcerange="-60 60"/>
    <motor name="motor_joint6" joint="joint6" gear="30" ctrlrange="-2.0 2.0" forcerange="-40 40"/>
    <motor name="motor_finger1_slide" joint="finger1_slide" gear="50" ctrlrange="-0.5 0.5" forcerange="-80 80"/>
    <motor name="motor_finger1_hinge1" joint="finger1_hinge1" gear="35" ctrlrange="-0.5 1.5" forcerange="-60 60"/>
    <motor name="motor_finger1_hinge2" joint="finger1_hinge2" gear="25" ctrlrange="0 1.2" forcerange="-40 40"/>
    <motor name="motor_finger2_slide" joint="finger2_slide" gear="50" ctrlrange="-0.5 0.5" forcerange="-80 80"/>
    <motor name="motor_finger2_hinge1" joint="finger2_hinge1" gear="35" ctrlrange="-0.5 1.5" forcerange="-60 60"/>
    <motor name="motor_finger2_hinge2" joint="finger2_hinge2" gear="25" ctrlrange="0 1.2" forcerange="-40 40"/>
    <motor name="motor_finger3_slide" joint="finger3_slide" gear="50" ctrlrange="-0.5 0.5" forcerange="-80 80"/>
    <motor name="motor_finger3_hinge1" joint="finger3_hinge1" gear="35" ctrlrange="-0.5 1.5" forcerange="-60 60"/>
    <motor name="motor_finger3_hinge2" joint="finger3_hinge2" gear="25" ctrlrange="0 1.2" forcerange="-40 40"/>
  </actuator>
  <sensor>
    <jointpos name="sensor_joint1" joint="joint1"/>
    <jointpos name="sensor_joint2" joint="joint2"/>
    <jointpos name="sensor_joint3" joint="joint3"/>
    <jointpos name="sensor_joint4" joint="joint4"/>
    <jointpos name="sensor_joint5" joint="joint5"/>
    <jointpos name="sensor_joint6" joint="joint6"/>
    <jointpos name="sensor_finger1_slide" joint="finger1_slide"/>
    <jointpos name="sensor_finger1_hinge1" joint="finger1_hinge1"/>
    <jointpos name="sensor_finger1_hinge2" joint="finger1_hinge2"/>
    <jointpos name="sensor_finger2_slide" joint="finger2_slide"/>
    <jointpos name="sensor_finger2_hinge1" joint="finger2_hinge1"/>
    <jointpos name="sensor_finger2_hinge2" joint="finger2_hinge2"/>
    <jointpos name="sensor_finger3_slide" joint="finger3_slide"/>
    <jointpos name="sensor_finger3_hinge1" joint="finger3_hinge1"/>
    <jointpos name="sensor_finger3_hinge2" joint="finger3_hinge2"/>
    <force name="sensor_finger1_force" site="finger1_tip"/>
    <force name="sensor_finger2_force" site="finger2_tip"/>
    <force name="sensor_finger3_force" site="finger3_tip"/>
    <touch name="sensor_finger1_touch" site="finger1_tip"/>
    <touch name="sensor_finger2_touch" site="finger2_tip"/>
    <touch name="sensor_finger3_touch" site="finger3_tip"/>
    <framepos name="sensor_gripper_pos" objtype="site" objname="grip_center"/>
    <framequat name="sensor_gripper_quat" objtype="site" objname="grip_center"/>
    <framepos name="sensor_sphere_pos" objtype="body" objname="test_sphere"/>
    <framepos name="sensor_box_pos" objtype="body" objname="test_box"/>
    <framepos name="sensor_cylinder_pos" objtype="body" objname="test_cylinder"/>
    <framepos name="sensor_capsule_pos" objtype="body" objname="test_capsule"/>
    <framepos name="sensor_ellipsoid_pos" objtype="body" objname="test_ellipsoid"/>
    <rangefinder name="sensor_gripper_range" site="grip_center" cutoff="0.5"/>
  </sensor>
  <equality>
    <joint name="equal_finger_slide_12" joint1="finger1_slide" joint2="finger2_slide" polycoef="0 1 0 0 0"/>
    <joint name="equal_finger_slide_13" joint1="finger1_slide" joint2="finger3_slide" polycoef="0 1 0 0 0"/>
    <joint name="equal_finger_hinge1_12" joint1="finger1_hinge1" joint2="finger2_hinge1" polycoef="0 1 0 0 0"/>
    <joint name="equal_finger_hinge1_13" joint1="finger1_hinge1" joint2="finger3_hinge1" polycoef="0 1 0 0 0"/>
    <joint name="equal_finger_hinge2_12" joint1="finger1_hinge2" joint2="finger2_hinge2" polycoef="0 1 0 0 0"/>
    <joint name="equal_finger_hinge2_13" joint1="finger1_hinge2" joint2="finger3_hinge2" polycoef="0 1 0 0 0"/>
  </equality>
</mujoco>
""")

# 创建数据实例
data = mujoco.MjData(model)

# 控制参数设置
JOINT_SPEED = 0.05  # 关节运动速度
GRIPPER_SPEED = 0.03  # 抓取器运动速度
CONTROL_DIM = model.nu  # 控制维度

# 初始化控制指令
ctrl = np.zeros(CONTROL_DIM)
running = True
reset_flag = False

# 按键状态跟踪
key_states = {}

# 按键映射字典
key_mapping = {
    # 机械臂关节控制 (数字键1-6对应关节1-6，方向键控制正负方向)
    glfw.KEY_1: {'axis': 0, 'dir': 1},  # 1键: 关节1 +
    glfw.KEY_Q: {'axis': 0, 'dir': -1},  # Q键: 关节1 -
    glfw.KEY_2: {'axis': 1, 'dir': 1},  # 2键: 关节2 +
    glfw.KEY_W: {'axis': 1, 'dir': -1},  # W键: 关节2 -
    glfw.KEY_3: {'axis': 2, 'dir': 1},  # 3键: 关节3 +
    glfw.KEY_E: {'axis': 2, 'dir': -1},  # E键: 关节3 -
    glfw.KEY_4: {'axis': 3, 'dir': 1},  # 4键: 关节4 +
    glfw.KEY_R: {'axis': 3, 'dir': -1},  # R键: 关节4 -
    glfw.KEY_5: {'axis': 4, 'dir': 1},  # 5键: 关节5 +
    glfw.KEY_T: {'axis': 4, 'dir': -1},  # T键: 关节5 -
    glfw.KEY_6: {'axis': 5, 'dir': 1},  # 6键: 关节6 +
    glfw.KEY_Y: {'axis': 5, 'dir': -1},  # Y键: 关节6 -

    # 抓取器控制
    glfw.KEY_SPACE: {'axis': 6, 'dir': 1},  # 空格键: 手指伸出/张开
    glfw.KEY_LEFT_SHIFT: {'axis': 6, 'dir': -1},  # Shift键: 手指缩回/闭合
    glfw.KEY_Z: {'axis': 7, 'dir': 1},  # Z键: 手指第一节 +
    glfw.KEY_X: {'axis': 7, 'dir': -1},  # X键: 手指第一节 -
    glfw.KEY_C: {'axis': 8, 'dir': 1},  # C键: 手指第二节 +
    glfw.KEY_V: {'axis': 8, 'dir': -1},  # V键: 手指第二节 -
}


def print_controls():
    """打印控制说明"""
    print("\n=== 机械臂键盘控制说明 ===")
    print("机械臂关节控制 (1-6关节，Q-Y对应反向):")
    print("  1/Q: 基座旋转 (关节1)")
    print("  2/W: 肩部俯仰 (关节2)")
    print("  3/E: 肘部运动 (关节3)")
    print("  4/R: 腕部旋转1 (关节4)")
    print("  5/T: 腕部旋转2 (关节5)")
    print("  6/Y: 腕部旋转3 (关节6)")
    print("\n抓取器控制:")
    print("  空格: 手指伸出/张开")
    print("  Shift: 手指缩回/闭合")
    print("  Z/X: 手指第一节关节")
    print("  C/V: 手指第二节关节")
    print("\n其他控制:")
    print("  ESC: 退出程序")
    print("  R: 重置机械臂位置 (注意：是字母R，不是数字键)")
    print("==========================\n")


def check_keypress():
    """检查键盘输入（独立线程运行）"""
    global running, reset_flag, ctrl, key_states

    # 初始化GLFW
    if not glfw.init():
        print("无法初始化GLFW")
        return

    # 创建一个隐藏窗口用于捕获键盘输入
    window = glfw.create_window(1, 1, "Key Capture", None, None)
    if not window:
        glfw.terminate()
        print("无法创建GLFW窗口")
        return

    glfw.make_context_current(window)

    # 键盘回调函数
    def key_callback(window, key, scancode, action, mods):
        global running, reset_flag, ctrl

        # 退出程序
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            running = False

        # 重置机械臂 (使用字母键R，避免和数字键4冲突)
        if key == glfw.KEY_R and action == glfw.PRESS and mods == 0:
            reset_flag = True
            print("机械臂已重置到初始位置！")

        # 更新按键状态
        if key in key_mapping:
            if action == glfw.PRESS:
                key_states[key] = True
            elif action == glfw.RELEASE:
                key_states[key] = False

    glfw.set_key_callback(window, key_callback)

    # 键盘检查循环
    while running:
        glfw.poll_events()

        # 重置控制指令
        ctrl = np.zeros(CONTROL_DIM)

        # 处理所有按下的按键
        for key, pressed in key_states.items():
            if pressed and key in key_mapping:
                info = key_mapping[key]
                axis = info['axis']
                direction = info['dir']

                # 根据控制轴设置速度
                if axis < 6:  # 机械臂关节
                    ctrl[axis] = direction * JOINT_SPEED
                else:  # 抓取器
                    ctrl[axis] = direction * GRIPPER_SPEED

        time.sleep(0.001)

    glfw.destroy_window(window)
    glfw.terminate()


def main():
    """主函数"""
    global running, reset_flag

    # 打印控制说明
    print_controls()

    # 启动键盘监听线程
    key_thread = threading.Thread(target=check_keypress, daemon=True)
    key_thread.start()

    # 创建viewer
    viewer = mujoco.viewer.launch(model, data)

    try:
        # 仿真循环
        while running and viewer.is_running():
            # 检查重置标志
            if reset_flag:
                mujoco.mj_resetData(model, data)
                reset_flag = False
                ctrl = np.zeros(CONTROL_DIM)

            # 设置控制指令
            data.ctrl[:] = ctrl

            # 运行仿真步
            mujoco.mj_step(model, data)

            # 控制帧率
            time.sleep(0.001)

    finally:
        running = False
        viewer.close()
        key_thread.join()


if __name__ == "__main__":
    main()