import os
import shutil
import time
import numpy as np
import mujoco
# 路径
MODEL_PATH = r"C:\Users\龙忠梁\Downloads\mujoco-3.3.7-windows-x86_64\model\robotic_arm\arm_with_gripper.xml"

def resolve_model_path(path: str) -> str:
    """处理包含中文字符的路径"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    # 检查路径是否包含非ASCII字符
    if any(ord(ch) > 127 for ch in path):
        temp_dir = r"C:\temp_mujoco_project"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "arm_with_gripper.xml")
        shutil.copy2(path, temp_path)
        print(f"已将模型复制到临时路径: {temp_path}")
        return temp_path
    return path

def set_initial_pose(model, data):
    """设置机械臂的初始姿态（立起来）"""
    # 关节初始位置（弧度）
    initial_qpos = np.zeros(model.nq)

    # 设置6个主要关节的初始位置
    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    initial_angles = [0.0, 0.785, -1.57, 0.0, 0.0, 0.0]

    for i, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            qpos_addr = model.jnt_qposadr[joint_id]
            initial_qpos[qpos_addr] = initial_angles[i]

    # 设置抓取器初始为半张开状态
    finger_joints = ["finger_1_joint", "finger_2_joint", "finger_3_joint",
                     "finger_1_proximal_joint", "finger_2_proximal_joint", "finger_3_proximal_joint",
                     "finger_1_distal_joint", "finger_2_distal_joint", "finger_3_distal_joint"]

    for joint_name in finger_joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            qpos_addr = model.jnt_qposadr[joint_id]
            if "slide" in joint_name:  # 滑动关节初始位置
                initial_qpos[qpos_addr] = 0.02  # 稍微张开
            elif "proximal" in joint_name:  # 近端关节
                initial_qpos[qpos_addr] = 0.3  # 轻微弯曲
            elif "distal" in joint_name:  # 远端关节
                initial_qpos[qpos_addr] = 0.2  # 轻微弯曲

    data.qpos[:] = initial_qpos
    mujoco.mj_forward(model, data)

def move_arm_to_workspace(model, data, viewer):
    """将机械臂移动到工作空间中央位置"""
    print("正在移动机械臂到工作位置...")

    # 控制机械臂移动到合适的工作位置
    steps = 300
    for i in range(steps):
        # 简单的控制：保持抓取器位置，移动机械臂到合适位置
        if model.nu >= 6:
            # 设置机械臂关节的控制信号
            # 调整机械臂姿态，使其处于更好的抓取位置
            data.ctrl[0] = 0.0  # joint_1
            data.ctrl[1] = 0.5  # joint_2 - 肩部
            data.ctrl[2] = -0.3  # joint_3 - 肘部
            data.ctrl[3] = 0.0  # joint_4
            data.ctrl[4] = 0.2  # joint_5
            data.ctrl[5] = 0.0  # joint_6

        mujoco.mj_step(model, data)

        # 同步视图
        if viewer is not None:
            viewer.sync()

        time.sleep(model.opt.timestep * 0.5)

    print("✓ 机械臂已移动到工作位置")

def control_gripper_demo(model, data, viewer):
    """演示爪勾的抓取功能，修正控制逻辑确保三个手指都向中心点弯曲"""

    print("=== 爪勾抓取功能演示 ===")
    print("修正控制逻辑，确保三个手指都向中心点弯曲")

    # 首先，打印所有抓取器相关关节的信息，以便调试
    print("\n抓取器关节信息:")
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        if joint_id != -1:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name and "finger" in joint_name:
                print(f"  控制通道 {i}: 控制关节 '{joint_name}' (ID: {joint_id})")

    # 演示1：直接控制所有抓取器电机，观察效果
    print("\n1. 测试所有抓取器控制通道...")

    # 先清零所有控制
    data.ctrl[:] = 0.0

    # 测试张开：使用负值
    print("   测试张开动作 (使用负控制信号)...")
    for i in range(300):
        # 设置所有抓取器控制为负值
        for j in range(6, model.nu):  # 抓取器控制从索引6开始
            data.ctrl[j] = -1.0

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)

    # 测试闭合：使用正值
    print("   测试闭合动作 (使用正控制信号)...")
    for i in range(300):
        # 设置所有抓取器控制为正值
        for j in range(6, model.nu):  # 抓取器控制从索引6开始
            data.ctrl[j] = 1.0

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)

    # 重置为张开状态
    print("   重置为张开状态...")
    for i in range(300):
        for j in range(6, model.nu):
            data.ctrl[j] = -1.0

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)

    # 演示2：尝试分别控制三个手指
    print("\n2. 分别控制三个手指...")

    # 创建控制通道到手指的映射
    control_to_finger = {}
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        if joint_id != -1:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                # 检查属于哪个手指
                if "finger_1" in joint_name:
                    control_to_finger[i] = 1
                elif "finger_2" in joint_name:
                    control_to_finger[i] = 2
                elif "finger_3" in joint_name:
                    control_to_finger[i] = 3

    # 手指1闭合
    print("   手指1闭合...")
    for i in range(200):
        for ctrl_idx, finger_num in control_to_finger.items():
            if finger_num == 1:
                data.ctrl[ctrl_idx] = 1.5  # 正值闭合
            else:
                data.ctrl[ctrl_idx] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)

    # 手指2闭合
    print("   手指2闭合...")
    for i in range(200):
        for ctrl_idx, finger_num in control_to_finger.items():
            if finger_num == 2:
                data.ctrl[ctrl_idx] = 1.5  # 正值闭合
            else:
                data.ctrl[ctrl_idx] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)

    # 手指3闭合
    print("   手指3闭合...")
    for i in range(200):
        for ctrl_idx, finger_num in control_to_finger.items():
            if finger_num == 3:
                data.ctrl[ctrl_idx] = 1.5  # 正值闭合
            else:
                data.ctrl[ctrl_idx] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)

    # 所有手指张开
    print("   所有手指张开...")
    for i in range(300):
        for j in range(6, model.nu):
            data.ctrl[j] = -1.0  # 负值张开

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)

    # 演示3：尝试不同的控制策略
    print("\n3. 尝试不同的控制策略...")

    # 策略1：区分滑动关节和旋转关节
    print("   策略1: 区分滑动关节和旋转关节...")

    # 首先识别不同类型的关节
    slide_ctrl_indices = []
    proximal_ctrl_indices = []
    distal_ctrl_indices = []

    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        if joint_id != -1:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                if "slide" in joint_name or ("joint" in joint_name and "proximal" not in joint_name and "distal" not in joint_name):
                    slide_ctrl_indices.append(i)
                elif "proximal" in joint_name:
                    proximal_ctrl_indices.append(i)
                elif "distal" in joint_name:
                    distal_ctrl_indices.append(i)

    print(f"   找到 {len(slide_ctrl_indices)} 个滑动关节控制")
    print(f"   找到 {len(proximal_ctrl_indices)} 个近端关节控制")
    print(f"   找到 {len(distal_ctrl_indices)} 个远端关节控制")

    # 完全张开
    print("   完全张开...")
    for i in range(300):
        for idx in slide_ctrl_indices:
            data.ctrl[idx] = -2.0  # 负值可能使滑动关节向外
        for idx in proximal_ctrl_indices:
            data.ctrl[idx] = -1.5  # 负值可能使旋转关节伸直
        for idx in distal_ctrl_indices:
            data.ctrl[idx] = -1.0  # 负值可能使远端关节伸直

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)

    # 完全闭合（尝试不同的组合）
    print("   尝试闭合 (组合1)...")
    for i in range(300):
        for idx in slide_ctrl_indices:
            data.ctrl[idx] = 2.0  # 正值可能使滑动关节向内
        for idx in proximal_ctrl_indices:
            data.ctrl[idx] = 1.5  # 正值可能使旋转关节弯曲
        for idx in distal_ctrl_indices:
            data.ctrl[idx] = 1.0  # 正值可能使远端关节弯曲

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)

    # 尝试相反的组合
    print("   尝试闭合 (组合2 - 相反符号)...")
    for i in range(300):
        for idx in slide_ctrl_indices:
            data.ctrl[idx] = -2.0  # 负值可能使滑动关节向内
        for idx in proximal_ctrl_indices:
            data.ctrl[idx] = -1.5  # 负值可能使旋转关节弯曲
        for idx in distal_ctrl_indices:
            data.ctrl[idx] = -1.0  # 负值可能使远端关节弯曲

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)

    # 固定机械臂主体，只控制夹爪动作
    print("   固定机械臂主体，只控制夹爪动作...")
    # 锁定前6个关节（机械臂主体）
    for j in range(6):
        data.ctrl[j] = 0.0  # 固定机械臂主体不动
    
    # 增加多次抓取循环演示
    print("   增加多次抓取循环演示...")
    for cycle in range(5):  # 执行5次抓取循环
        print(f"   抓取循环 {cycle+1} 开始...")
        
        # 闭合抓取器 - 更强的控制信号
        print(f"     闭合抓取器 (循环 {cycle+1})...")
        for i in range(300):  # 增加迭代次数
            # 固定机械臂主体不动
            for j in range(6):
                data.ctrl[j] = 0.0
            
            # 控制夹爪闭合
            for j in range(6, model.nu):
                data.ctrl[j] = 2.0  # 更大的正值确保完全闭合

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * 0.5)
        
        # 等待一段时间
        print(f"     保持抓取状态 (循环 {cycle+1})...")
        for i in range(150):  # 增加等待时间
            # 固定机械臂主体不动
            for j in range(6):
                data.ctrl[j] = 0.0
                
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * 0.5)
        
        # 张开抓取器 - 更强的控制信号
        print(f"     张开抓取器 (循环 {cycle+1})...")
        for i in range(300):  # 增加迭代次数
            # 固定机械臂主体不动
            for j in range(6):
                data.ctrl[j] = 0.0
            
            # 控制夹爪张开
            for j in range(6, model.nu):
                data.ctrl[j] = -2.0  # 更大的负值确保完全张开

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * 0.5)
        
        # 等待一段时间
        print(f"     保持张开状态 (循环 {cycle+1})...")
        for i in range(150):  # 增加等待时间
            # 固定机械臂主体不动
            for j in range(6):
                data.ctrl[j] = 0.0
                
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * 0.5)
    
    print("   多次抓取循环演示完成！")
    
    # 最终回到张开状态
    print("   最终回到张开状态...")
    for i in range(300):
        for j in range(6, model.nu):
            data.ctrl[j] = -1.0  # 使用负值

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)

    print("\n✓ 爪勾抓取功能演示完成!")
    print("\n观察结果:")
    print("1. 如果所有手指都正确向中心弯曲，说明控制信号符号正确")
    print("2. 如果有手指向外翻，可能需要检查模型中的关节轴设置")
    print("3. 如果所有手指都上下运动，说明滑动关节轴可能是(0,0,1)而不是水平方向")

def main():
    """主函数"""
    # 加载模型
    model_path = resolve_model_path(MODEL_PATH)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print("开始机械臂爪勾抓取功能演示")
    print("修正控制逻辑，测试不同的控制策略")

    # 启动可视化
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机视角，专注于抓取器
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.8
        viewer.cam.lookat[:] = [0.25, 0.0, 0.15]

        print("\n=== 阶段1：初始化机械臂 ===")
        # 设置机械臂初始姿态
        set_initial_pose(model, data)

        # 给一些时间让机械臂稳定
        for _ in range(150):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        print("✓ 机械臂已初始化")

        print("\n=== 阶段2：爪勾抓取功能演示 ===")
        # 演示爪勾的抓取功能
        control_gripper_demo(model, data, viewer)

        print("\n=== 演示完成 ===")
        print("请在可视化窗口中观察三个手指的运动方向")
        print("根据观察结果，可能需要调整控制信号的符号或模型设置")
        print("关闭窗口以结束程序")

        # 保持窗口打开，让用户观察
        print("\n保持张开状态展示中...")
        for i in range(300):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * 0.5)

        # 保持窗口打开
        while viewer.is_running:
            time.sleep(0.1)

if __name__ == "__main__":
    main()