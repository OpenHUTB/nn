# MuJoCo 3.4.0 带自动复位的3自由度机械臂精准取放（增加进度条显示）

import sys
import mujoco
import mujoco.viewer
import time
import numpy as np
import datetime

# 速度配置（单位：秒）
SPEED_CONFIG = {
    "slow":   {"joint": 3.0, "grip": 1.5, "pause": 0.02},
    "medium": {"joint": 2.0, "grip": 1.0, "pause": 0.001},
    "fast":   {"joint": 1.0, "grip": 0.5, "pause": 0.0005}
}


def robot_arm_auto_reset_demo():
    # 纯MuJoCo 3.4.0原生语法，无任何高版本扩展标签
    robot_xml = """
<mujoco model="3-DOF Robot Arm with Auto Reset">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual>
    <global azimuth="30" elevation="-25"/>  <!-- 清晰3D视角 -->
  </visual>
  <asset>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="yellow" rgba="0.8 0.7 0.2 1"/>
    <material name="gray" rgba="0.5 0.5 0.5 1"/>
    <material name="blue" rgba="0.2 0.4 0.8 1"/>
    <material name="green" rgba="0.2 0.8 0.2 1"/>
  </asset>

  <!-- 世界体定义 -->
  <worldbody>
    <!-- 固定监控相机 -->
    <camera name="monitor_camera" pos="1.8 1.8 1.2" xyaxes="1 0 0 0 1 0"/>
    <!-- 工作平台 -->
    <geom name="workbench" type="plane" size="2 2 0.1" pos="0 0 -0.05" material="gray"/>
    <!-- 待抓取目标：蓝色球体（易抓取，不易滚落） -->
    <body name="target_ball" pos="0.9 0.6 0.0">
      <geom name="target_geom" type="sphere" size="0.08" pos="0 0 0" material="blue"/>
      <joint name="target_joint" type="free"/>
    </body>
    <!-- 目标放置区域：绿色圆形标记 -->
    <geom name="place_area" type="cylinder" size="0.15 0.01" pos="-0.9 0.6 0.0" material="green"/>
    <!-- 3自由度机械臂 -->
    <body name="robot_base" pos="0 0 0.0">
      <geom name="base_geom" type="cylinder" size="0.18 0.1" pos="0 0 0" material="yellow"/>
      <joint name="base_joint" type="free"/>

      <!-- 关节1：基座旋转（Z轴） -->
      <body name="arm_1" pos="0 0 0.1">
        <geom name="arm1_geom" type="cylinder" size="0.08 0.6" pos="0 0 0.3" material="yellow"/>
        <joint name="joint1_rotate" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14" damping="0.03"/>

        <!-- 关节2：大臂俯仰（Y轴） -->
        <body name="arm_2" pos="0 0 0.6">
          <geom name="arm2_geom" type="cylinder" size="0.07 0.5" pos="0 0 0.25" material="yellow"/>
          <joint name="joint2_pitch" type="hinge" axis="0 1 0" pos="0 0 0" range="-1.5 1.5" damping="0.03"/>

          <!-- 关节3：小臂伸缩（X轴） -->
          <body name="arm_3" pos="0 0 0.5">
            <geom name="arm3_geom" type="cylinder" size="0.06 0.4" pos="0.2 0 0" material="yellow"/>
            <joint name="joint3_telescope" type="slide" axis="1 0 0" pos="0 0 0" range="0 0.4" damping="0.03"/>

            <!-- 平行夹爪 -->
            <body name="gripper_base" pos="0.4 0 0">
              <geom name="gripper_base_geom" type="box" size="0.07 0.07 0.07" pos="0 0 0" material="red"/>

              <!-- 左夹爪 -->
              <body name="gripper_left" pos="0 0.07 0">
                <geom name="gripper_left_geom" type="box" size="0.05 0.04 0.05" pos="0 0 0" material="red"/>
                <joint name="gripper_left_joint" type="hinge" axis="0 0 1" pos="0 -0.07 0" range="-0.4 0" damping="0.02"/>
              </body>

              <!-- 右夹爪 -->
              <body name="gripper_right" pos="0 -0.07 0">
                <geom name="gripper_right_geom" type="box" size="0.05 0.04 0.05" pos="0 0 0" material="red"/>
                <joint name="gripper_right_joint" type="hinge" axis="0 0 1" pos="0 0.07 0" range="0 0.4" damping="0.02"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 执行器配置（MuJoCo 3.4.0 完全兼容） -->
  <actuator>
    <!-- 关节位置控制（高精度） -->
    <position name="joint1_act" joint="joint1_rotate" kp="1100" kv="100"/>
    <position name="joint2_act" joint="joint2_pitch" kp="1100" kv="100"/>
    <position name="joint3_act" joint="joint3_telescope" kp="1100" kv="100"/>

    <!-- 夹爪速度控制（软接触，防损坏） -->
    <velocity name="gripper_left_act" joint="gripper_left_joint" kv="40" ctrlrange="-0.3 0"/>
    <velocity name="gripper_right_act" joint="gripper_right_joint" kv="40" ctrlrange="0 0.3"/>
  </actuator>
</mujoco>
    """

    # 加载模型（确保零XML错误）
    try:
        model = mujoco.MjModel.from_xml_string(robot_xml)
        data = mujoco.MjData(model)
        print("✅ 3自由度机械臂模型加载成功，启动仿真...")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 获取执行器索引
    joint_idxs = {
        "joint1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint1_act"),
        "joint2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint2_act"),
        "joint3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint3_act")
    }
    left_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_left_act")
    right_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_right_act")

    # ---------------------- 模块化功能函数 ----------------------
    def joint_move(joint_name, target_val, duration, viewer, step_desc):
        """单关节精准移动，带进度条显示"""
        print(f"\n🔧 {step_desc}")
        idx = joint_idxs[joint_name]
        start_val = data.ctrl[idx]
        start_time = time.time()
        bar_length = 20

        while (time.time() - start_time) < duration and viewer.is_running():
            progress = (time.time() - start_time) / duration
            current_val = start_val + progress * (target_val - start_val)
            data.ctrl[idx] = current_val

            # 进度条显示
            filled = int(progress * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r{joint_name} 进度：|{bar}| {progress * 100:.1f}% | 当前值：{current_val:.2f}", end="")
            
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
        print()
        return True

    def gripper_close(viewer, desc="目标"):
        """软接触闭合夹爪，带进度条显示"""
        print(f"\n🔧 闭合夹爪抓取{desc}")
        grip_speed = -0.25
        close_duration = 1.0
        start_time = time.time()
        bar_length = 20

        while (time.time() - start_time) < close_duration and viewer.is_running():
            progress = (time.time() - start_time) / close_duration
            data.ctrl[left_grip_idx] = grip_speed
            data.ctrl[right_grip_idx] = -grip_speed

            # 进度条显示
            filled = int(progress * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r夹爪闭合进度：|{bar}| {progress * 100:.1f}%", end="")
            
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

        data.ctrl[left_grip_idx] = 0
        data.ctrl[right_grip_idx] = 0
        print(f"\n✅ {desc} 抓取完成，夹爪锁定")
        return True

    def gripper_open(viewer, desc="目标"):
        """张开夹爪放置目标"""
        print(f"\n🔧 张开夹爪放置{desc}")
        open_duration = 0.8
        start_time = time.time()

        while (time.time() - start_time) < open_duration and viewer.is_running():
            data.ctrl[left_grip_idx] = 0.25
            data.ctrl[right_grip_idx] = -0.25
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

        data.ctrl[left_grip_idx] = 0
        data.ctrl[right_grip_idx] = 0
        print(f"✅ {desc} 放置完成，夹爪复位")
        return True

    def robot_auto_reset(viewer):
        """机械臂自动复位到初始位置"""
        print("\n\n🔧 开始机械臂自动复位")
        joint_move("joint2", 0.0, 1.5, viewer, "复位：抬升大臂")
        joint_move("joint3", 0.0, 1.5, viewer, "复位：收缩小臂")
        joint_move("joint1", 0.0, 2.0, viewer, "复位：基座回正")
        print("✅ 机械臂已完成自动复位，准备下一次抓取")
        return True

    def target_auto_reset(viewer):
        """目标物体自动重置到原位"""
        print("\n🔧 目标物体自动重置中...")
        target_ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
        target_qpos = np.array([0.9, 0.6, 0.0, 1, 0, 0, 0])
        data.qpos[7:14] = target_qpos
        data.qvel[6:12] = 0
        mujoco.mj_step(model, data)
        print("✅ 目标物体已重置到原位")

    def grab_and_place(viewer, retry_max=2, speed="medium"):
        """完整取放流程（含自动重试、连续失败提醒、速度控制）"""
        retry_count = 0
        success = False
        consecutive_fails = 0
        
        # 获取速度参数
        joint_duration = SPEED_CONFIG[speed]["joint"]
        grip_duration = SPEED_CONFIG[speed]["grip"]
        pause_time = SPEED_CONFIG[speed]["pause"]

        while retry_count < retry_max and not success:
            print(f"\n\n===== 开始第 {retry_count + 1} 次抓取尝试 =====")
            try:
                # 阶段1：对准目标
                joint_move("joint1", 0.0, joint_duration, viewer, "步骤1：旋转基座对准蓝色目标")
                joint_move("joint2", -0.7, joint_duration, viewer, "步骤2：俯仰大臂接近目标")
                joint_move("joint3", 0.35, joint_duration, viewer, "步骤3：伸缩小臂对准目标")

                # 阶段2：抓取目标
                gripper_close(viewer, "蓝色球体")

                # 阶段3：抬升并转移目标
                joint_move("joint2", 0.0, joint_duration, viewer, "步骤4：抬升目标脱离平台")
                joint_move("joint1", 3.14, joint_duration, viewer, "步骤5：旋转基座对准绿色放置区域")
                joint_move("joint2", -0.7, joint_duration, viewer, "步骤6：降低目标接近放置区域")

                # 阶段4：放置目标
                gripper_open(viewer, "蓝色球体")

                success = True
                consecutive_fails = 0
                print(f"\n\n🎉 第 {retry_count+1} 次抓取尝试成功！")
            except Exception as e:
                retry_count += 1
                consecutive_fails += 1
                print(f"\n❌ 第 {retry_count} 次抓取失败：{e}")
                
                if consecutive_fails >= 3:
                    print("\n" + "="*50)
                    print("⚠️⚠️⚠️ 连续失败次数过多！")
                    print("请检查：")
                    print("  1. 目标物体是否在抓取范围内？")
                    print("  2. 夹爪是否正常工作？")
                    print("  3. 机械臂关节是否卡住？")
                    print("="*50)
                    input("按 Enter 键继续尝试...")
                    consecutive_fails = 0
                
                robot_auto_reset(viewer)

        if not success:
            print(f"\n❌ 已达到最大重试次数（{retry_max}次），抓取失败")
        return success

    # ---------------------- 启动主流程 ----------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n📌 开始带自动复位的机械臂精准取放流程...")
        print("-" * 60)

        # ========== 多次抓取配置 ==========
        # 从命令行读取抓取次数，默认 5 次
        if len(sys.argv) > 1:
            total_rounds = int(sys.argv[1])
        else:
            total_rounds = 5
        print(f"📌 本次将抓取 {total_rounds} 次")
        
        # 读取速度参数
        speed = "medium"
        if len(sys.argv) > 2:
            speed_arg = sys.argv[2].lower()
            if speed_arg in SPEED_CONFIG:
                speed = speed_arg
                print(f"📌 速度模式: {speed} (slow/medium/fast)")
            else:
                print(f"⚠️ 速度参数 '{speed_arg}' 无效，使用默认 'medium'")
        
        success_count = 0

        for round_num in range(1, total_rounds + 1):
            print(f"\n{'='*40}")
            print(f"🔄 第 {round_num}/{total_rounds} 次抓取")
            print(f"{'='*40}")

            grab_success = grab_and_place(viewer, speed=speed)

            if grab_success:
                success_count += 1
                print(f"✅ 第 {round_num} 次抓取成功")
            else:
                print(f"❌ 第 {round_num} 次抓取失败")

            robot_auto_reset(viewer)
            target_auto_reset(viewer)

            if round_num < total_rounds:
                print("\n⏸ 准备下一次抓取...")
                for _ in range(30):
                    if not viewer.is_running():
                        break
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.05)

        # ========== 输出统计结果 ==========
        print(f"\n{'='*50}")
        print(f"🎉 所有抓取完成！")
        print(f"📊 统计结果：成功 {success_count}/{total_rounds} 次")
        print(f"📈 成功率：{success_count/total_rounds*100:.1f}%")
        print(f"{'='*50}")

        print("\n\n📌 流程结束，保持可视化5秒...")
        start_hold = time.time()
        while (time.time() - start_hold) < 5 and viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    # 保存日志到文件
    try:
        with open("grasp_log.txt", "a", encoding="gbk") as f:
            f.write(f"[{datetime.datetime.now()}] 抓取{total_rounds}次, 成功{success_count}次, 成功率{success_count/total_rounds*100:.1f}%, 速度:{speed}\n")
        print("📝 日志已保存到 grasp_log.txt")
    except Exception as e:
        print(f"⚠️ 日志保存失败：{e}")

    print("\n\n🎉 3自由度机械臂自动复位取放演示完毕！")


if __name__ == "__main__":
    robot_arm_auto_reset_demo()