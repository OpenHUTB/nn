# MuJoCo 3.4.0 带自动复位的3自由度机械臂精准取放（无传感器，零XML错误）
import mujoco
import mujoco.viewer
import time
import numpy as np


def robot_arm_auto_reset_demo():
    # 纯MuJoCo 3.4.0原生语法，无任何高版本扩展标签
    # robot_xml = """
    # """

    # 加载模型（确保零XML错误）
    try:
        model = mujoco.MjModel.from_xml_path("robot_arm.xml")
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
        """单关节精准移动"""
        print(f"\n🔧 {step_desc}")
        idx = joint_idxs[joint_name]
        start_val = data.ctrl[idx]
        start_time = time.time()

        while (time.time() - start_time) < duration and viewer.is_running():
            progress = (time.time() - start_time) / duration
            current_val = start_val + progress * (target_val - start_val)
            data.ctrl[idx] = current_val

            print(f"\r{joint_name} 进度：{progress * 100:.1f}% | 当前值：{current_val:.2f}", end="")
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
        print()  # 换行
        return True

    def gripper_close(viewer, desc="目标"):
        """软接触闭合夹爪"""
        print(f"\n🔧 闭合夹爪抓取{desc}")
        grip_speed = -0.25
        close_duration = 1.0
        start_time = time.time()

        while (time.time() - start_time) < close_duration and viewer.is_running():
            progress = (time.time() - start_time) / close_duration
            data.ctrl[left_grip_idx] = grip_speed
            data.ctrl[right_grip_idx] = -grip_speed

            print(f"\r夹爪闭合进度：{progress * 100:.1f}%", end="")
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
        # 步骤1：抬升大臂
        joint_move("joint2", 0.0, 1.5, viewer, "复位：抬升大臂")
        # 步骤2：收缩小臂
        joint_move("joint3", 0.0, 1.5, viewer, "复位：收缩小臂")
        # 步骤3：旋转基座回正
        joint_move("joint1", 0.0, 2.0, viewer, "复位：基座回正")
        print("✅ 机械臂已完成自动复位，准备下一次抓取")
        return True

    def grab_and_place(viewer, retry_max=2):
        """完整取放流程（含自动重试）"""
        retry_count = 0
        success = False

        while retry_count < retry_max and not success:
            print(f"\n\n===== 开始第 {retry_count + 1} 次抓取尝试 =====")
            try:
                # 阶段1：对准目标
                joint_move("joint1", 0.0, 2.0, viewer, "步骤1：旋转基座对准蓝色目标")
                joint_move("joint2", -0.7, 2.0, viewer, "步骤2：俯仰大臂接近目标")
                joint_move("joint3", 0.35, 2.0, viewer, "步骤3：伸缩小臂对准目标")

                # 阶段2：抓取目标
                gripper_close(viewer, "蓝色球体")

                # 阶段3：抬升并转移目标
                joint_move("joint2", 0.0, 1.5, viewer, "步骤4：抬升目标脱离平台")
                joint_move("joint1", 3.14, 2.5, viewer, "步骤5：旋转基座对准绿色放置区域")
                joint_move("joint2", -0.7, 1.5, viewer, "步骤6：降低目标接近放置区域")

                # 阶段4：放置目标
                gripper_open(viewer, "蓝色球体")

                # 抓取成功，退出重试循环
                success = True
                print("\n\n🎉 第 {retry_count+1} 次抓取尝试成功！")
            except Exception as e:
                retry_count += 1
                print(f"\n❌ 第 {retry_count} 次抓取失败：{e}，准备重试...")
                robot_auto_reset(viewer)

        if not success:
            print(f"\n❌ 已达到最大重试次数（{retry_max}次），抓取失败")
        return success

    # ---------------------- 启动主流程 ----------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n📌 开始带自动复位的机械臂精准取放流程...")
        print("-" * 60)

        # 执行完整取放流程
        grab_success = grab_and_place(viewer)

        # 无论成功与否，最终执行自动复位
        if grab_success:
            robot_auto_reset(viewer)
        else:
            print("\n🔧 强制执行机械臂自动复位")
            robot_auto_reset(viewer)

        # 保持可视化查看结果
        print("\n\n📌 流程结束，保持可视化5秒...")
        start_hold = time.time()
        while (time.time() - start_hold) < 5 and viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    print("\n\n🎉 3自由度机械臂自动复位取放演示完毕！")


if __name__ == "__main__":
    robot_arm_auto_reset_demo()