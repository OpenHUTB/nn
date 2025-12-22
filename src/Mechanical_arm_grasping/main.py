import pybullet as p
import pybullet_data
import time
import numpy as np


class ArmElevatorControllerPyBullet:
    def __init__(self):
        # 连接PyBullet模拟器（GUI模式，显示界面）
        self.physics_client = p.connect(p.GUI)
        # 设置模型搜索路径（关键：确保能找到内置模型）
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # 关闭重力（避免机械臂倾倒，专注升降控制；若需要真实物理效果可开启）
        p.setGravity(0, 0, 0)

        # 加载地面和KUKA IIWA机械臂（内置模型，必存在，无需额外配置）
        self.plane_id = p.loadURDF("plane.urdf")  # 加载地面
        # 机械臂初始位姿：坐标(0,0,0)，姿态（无旋转）
        self.arm_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )

        # 定义升降关节：选择KUKA IIWA的第1个关节（索引0，可实现垂直方向升降/旋转，适配升降逻辑）
        self.elevator_joint_index = 0
        # 获取关节信息（限位、当前位置）
        joint_info = p.getJointInfo(self.arm_id, self.elevator_joint_index)
        self.joint_min = joint_info[8]  # 关节运动下限
        self.joint_max = joint_info[9]  # 关节运动上限
        self.current_pos = p.getJointState(self.arm_id, self.elevator_joint_index)[0]  # 当前位置

        # 打印关节初始化信息
        print(f"升降关节初始化完成：")
        print(f"关节索引：{self.elevator_joint_index}")
        print(f"当前位置：{self.current_pos:.3f}")
        print(f"运动范围：[{self.joint_min:.3f}, {self.joint_max:.3f}]")

    def move_elevator(self, target_pos, speed=0.05):
        """
        驱动升降关节运动到目标位置
        :param target_pos: 目标位置（需在关节限位范围内）
        :param speed: 运动速度（正数，越小越慢）
        """
        # 校验目标位置合法性
        if target_pos < self.joint_min or target_pos > self.joint_max:
            raise ValueError(f"目标位置超出关节范围！允许范围：[{self.joint_min:.3f}, {self.joint_max:.3f}]")

        print(f"\n开始升降运动：当前位置 {self.current_pos:.3f} → 目标位置 {target_pos:.3f}")
        # 循环控制，直到接近目标位置（误差小于0.001）
        while abs(self.current_pos - target_pos) > 0.001:
            # 计算运动步长（根据目标位置判断升降方向）
            step = speed if target_pos > self.current_pos else -speed
            # 更新当前位置（防止超出限位）
            self.current_pos = np.clip(self.current_pos + step, self.joint_min, self.joint_max)
            # 发送位置指令给关节（位置控制模式）
            p.setJointMotorControl2(
                bodyUniqueId=self.arm_id,
                jointIndex=self.elevator_joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.current_pos
            )
            # 步进物理仿真（更新场景状态）
            p.stepSimulation()
            # 小幅延时，模拟真实运动节奏
            time.sleep(0.01)
            # 获取模拟器中关节的实际位置（反馈同步）
            self.current_pos = p.getJointState(self.arm_id, self.elevator_joint_index)[0]
            # 实时刷新显示当前位置
            print(f"实时位置：{self.current_pos:.3f}", end='\r')

        print(f"\n升降运动完成！最终位置：{self.current_pos:.3f}")

    def disconnect(self):
        """断开与PyBullet模拟器的连接"""
        p.disconnect(self.physics_client)
        print("\n已断开与PyBullet模拟器的连接")


# ------------------- 主执行程序 -------------------
if __name__ == "__main__":
    # 1. 初始化机械臂升降控制器
    arm_controller = ArmElevatorControllerPyBullet()

    try:
        # 2. 执行升降动作序列
        arm_controller.move_elevator(target_pos=arm_controller.joint_max * 0.6, speed=0.03)  # 上升（接近上限）
        time.sleep(1)  # 停顿1秒
        arm_controller.move_elevator(target_pos=arm_controller.joint_min * 0.6, speed=0.02)  # 下降（接近下限）
        time.sleep(1)  # 停顿1秒
        arm_controller.move_elevator(target_pos=0)  # 回到初始中间位置
    finally:
        # 3. 无论是否出错，最终断开连接
        arm_controller.disconnect()