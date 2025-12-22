import numpy as np
import math
import time
import mujoco as mj
from mujoco import viewer

# ====================== 全局配置参数 ======================
# MuJoCo模型路径
MUJOCO_MODEL_PATH = "robot_arm.xml"
# 机械臂关节名
JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3"]
# 目标物体名
TARGET_BODY_NAME = "target_ball"
# 仿真目标位置（可调整）
SIM_TARGET_POS = np.array([1.5, 1.0, 0.5])
# 全局标志：是否已执行移动逻辑
has_moved = False

# ====================== MuJoCo机械臂控制类（适配原生UI）=====================
class MuJoCoArmController:
    def __init__(self, model_path):
        # 加载模型和数据
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        # 获取关节和目标物体索引
        self.joint_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self.target_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, TARGET_BODY_NAME)
        # 设置目标物体初始位置
        self.set_target_pos(SIM_TARGET_POS)
        print("✅ MuJoCo机械臂控制器初始化完成")

    def set_target_pos(self, pos):
        """设置目标物体位置"""
        self.data.body(TARGET_BODY_NAME).xpos = pos
        print(f"🎯 目标物体位置已设置为：{np.round(pos, 3)}")

    def get_joint_angles(self):
        """获取当前关节角度"""
        return [self.data.joint(name).qpos[0] for name in JOINT_NAMES]

    def set_joint_target(self, target_angles, kp=200, kd=10):
        """PD控制关节到目标角度"""
        current_angles = np.array(self.get_joint_angles())
        current_vel = np.array([self.data.joint(name).qvel[0] for name in JOINT_NAMES])
        torque = kp * (target_angles - current_angles) - kd * current_vel
        for i, joint_id in enumerate(self.joint_ids):
            self.data.ctrl[joint_id] = torque[i]

    def inverse_kinematics(self, target_pos):
        """数值逆运动学求解关节角度（兼容版本，改用手动映射避免mj_inverse问题）"""
        print("ℹ️ 使用兼容版逆运动学（坐标映射角度）")
        # 手动坐标到角度的映射（可根据模型调整比例）
        angle0 = target_pos[0] * np.pi / 4  # joint0：绕z轴旋转
        angle1 = target_pos[1] * np.pi / 4  # joint1：绕x轴旋转
        angle2 = -target_pos[1] * np.pi / 4 # joint2：绕x轴旋转
        angle3 = 0.0                        # joint3：固定角度
        return np.array([angle0, angle1, angle2, angle3])

    def move_to_target(self, target_pos):
        """移动机械臂到目标位置"""
        global has_moved
        if has_moved:
            return
        print(f"\n📢 开始移动到目标位置：{np.round(target_pos, 3)}")
        target_angles = self.inverse_kinematics(target_pos)
        # 逐步逼近目标角度
        current_angles = np.array(self.get_joint_angles())
        step = 0.02
        while np.linalg.norm(current_angles - target_angles) > 0.02:
            current_angles = np.clip(current_angles + step * np.sign(target_angles - current_angles), -np.pi, np.pi)
            self.set_joint_target(current_angles)
            mj.mj_step(self.model, self.data)
            time.sleep(0.01)
        self.set_joint_target(target_angles)
        self.close_gripper()
        has_moved = True
        print("✅ 机械臂已到达目标位置并完成抓取")

    def close_gripper(self):
        """闭合夹爪（模拟抓取）"""
        print("🤖 夹爪闭合，抓取目标")
        # 目标物体随夹爪移动（兼容：检查arm4是否存在）
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "arm4")
        if body_id != -1:
            gripper_pos = self.data.body("arm4").xpos
            self.data.body(TARGET_BODY_NAME).xpos = gripper_pos + np.array([0, 0.1, 0])
        time.sleep(0.5)

# ====================== 主程序（兼容所有MuJoCo版本，移除callback参数）=====================
def main():
    global has_moved
    # 初始化控制器
    arm_controller = MuJoCoArmController(MUJOCO_MODEL_PATH)
    # 设置目标物体位置
    arm_controller.set_target_pos(SIM_TARGET_POS)

    # 打印操作说明
    print("\n=====================================")
    print("📋 操作说明：")
    print("  - 程序启动后会自动移动机械臂到目标位置并抓取")
    print("  - 在Viewer中可通过鼠标调整视角（拖拽/滚轮）")
    print("  - 按ESC键或关闭窗口退出")
    print("=====================================\n")

    # 先执行机械臂移动逻辑（在启动Viewer前完成核心运动）
    arm_controller.move_to_target(SIM_TARGET_POS)

    # 启动MuJoCo原生可视化Viewer（兼容版本：仅传入model和data，不使用callback）
    # 方式1：使用viewer.launch（简单启动，部分版本支持）
    try:
        viewer.launch(arm_controller.model, arm_controller.data)
    except Exception as e:
        # 方式2：如果launch报错，改用手动循环（最兼容的方式）
        print(f"⚠️ 简易启动失败，使用手动循环模式：{e}")
        # 创建viewer实例
        v = viewer.Viewer(arm_controller.model, arm_controller.data)
        while True:
            # 持续步进到下一帧
            mj.mj_step(arm_controller.model, arm_controller.data)
            # 更新viewer画面
            v.sync()
            # 短暂休眠，控制帧率
            time.sleep(0.01)
            # 检查是否关闭窗口
            if not v.is_running():
                break
        v.close()

if __name__ == "__main__":
    main()