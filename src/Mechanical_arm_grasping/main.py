import pybullet as p
import pybullet_data
import time
import numpy as np
from typing import Optional, Tuple

# ------------------- 配置常量（便于统一修改） -------------------
SIMULATION_GRAVITY: Tuple[float, float, float] = (0, 0, 0)  # 仿真重力，(0,0,-9.8)为真实重力
ARM_MODEL_PATH: str = "kuka_iiwa/model.urdf"  # 机械臂模型路径
ARM_BASE_POSITION: Tuple[float, float, float] = (0, 0, 0)  # 机械臂初始位置
ARM_BASE_ORIENTATION: Tuple[float, float, float] = (0, 0, 0)  # 机械臂初始姿态（欧拉角）
ELEVATOR_JOINT_INDEX: int = 0  # 升降关节索引
MOVE_SPEED_DEFAULT: float = 0.03  # 默认升降速度
POSITION_TOLERANCE: float = 0.001  # 位置误差容忍度（到达该误差即认为运动完成）
DELAY_STEP: float = 0.01  # 仿真步进延时


class ArmElevatorController:
    """机械臂升降关节控制器（面向对象封装，职责单一）"""

    def __init__(self):
        """初始化模拟器连接、机械臂模型和关节信息"""
        self.physics_client: Optional[int] = None
        self.arm_id: Optional[int] = None
        self.plane_id: Optional[int] = None

        # 关节相关参数
        self.elevator_joint_index: int = ELEVATOR_JOINT_INDEX
        self.joint_min: float = 0.0
        self.joint_max: float = 0.0
        self.current_pos: float = 0.0

        # 初始化流程
        self._connect_simulator()
        self._load_scene()
        self._init_joint_info()
        self._print_init_info()

    def _connect_simulator(self) -> None:
        """私有方法：连接PyBullet模拟器（封装初始化细节）"""
        try:
            self.physics_client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(*SIMULATION_GRAVITY)
            print("✅ 成功连接PyBullet模拟器")
        except Exception as e:
            raise RuntimeError(f"❌ 连接模拟器失败：{str(e)}")

    def _load_scene(self) -> None:
        """私有方法：加载地面和机械臂模型（封装场景加载逻辑）"""
        try:
            # 加载地面
            self.plane_id = p.loadURDF("plane.urdf")
            # 加载机械臂
            base_orientation = p.getQuaternionFromEuler(ARM_BASE_ORIENTATION)
            self.arm_id = p.loadURDF(
                ARM_MODEL_PATH,
                basePosition=ARM_BASE_POSITION,
                baseOrientation=base_orientation
            )
            print("✅ 成功加载场景（地面+机械臂）")
        except Exception as e:
            self.disconnect()  # 加载失败时自动断开连接
            raise RuntimeError(f"❌ 加载场景失败：{str(e)}")

    def _init_joint_info(self) -> None:
        """私有方法：初始化升降关节的限位和当前位置"""
        if self.arm_id is None:
            raise RuntimeError("❌ 机械臂未加载，无法初始化关节信息")

        # 获取关节基础信息
        joint_info = p.getJointInfo(self.arm_id, self.elevator_joint_index)
        self.joint_min = joint_info[8]
        self.joint_max = joint_info[9]
        # 获取当前关节位置
        self.current_pos = p.getJointState(self.arm_id, self.elevator_joint_index)[0]

    def _print_init_info(self) -> None:
        """打印初始化信息（格式化输出，更易读）"""
        print("\n=" * 40)
        print("📌 机械臂升降关节初始化信息")
        print("=" * 40)
        print(f"关节索引：{self.elevator_joint_index}")
        print(f"当前位置：{self.current_pos:.3f}")
        print(f"运动范围：[{self.joint_min:.3f}, {self.joint_max:.3f}]")
        print(f"默认速度：{MOVE_SPEED_DEFAULT}")
        print(f"位置误差容忍度：{POSITION_TOLERANCE}")
        print("=" * 40 + "\n")

    def _check_target_pos_valid(self, target_pos: float) -> bool:
        """私有方法：校验目标位置是否合法（返回布尔值，便于后续扩展）"""
        if self.joint_min <= target_pos <= self.joint_max:
            return True
        print(f"❌ 目标位置 {target_pos:.3f} 超出关节范围：[{self.joint_min:.3f}, {self.joint_max:.3f}]")
        return False

    def move_elevator(self, target_pos: float, speed: Optional[float] = None) -> None:
        """
        驱动升降关节运动到目标位置（公开方法，对外提供核心功能）
        :param target_pos: 目标位置（需在关节限位范围内）
        :param speed: 运动速度，默认使用MOVE_SPEED_DEFAULT
        :return: None
        """
        # 处理默认速度
        move_speed = speed if speed is not None else MOVE_SPEED_DEFAULT
        # 校验目标位置
        if not self._check_target_pos_valid(target_pos):
            return

        # 打印运动开始信息
        print(f"\n🚀 开始升降运动：当前位置 {self.current_pos:.3f} → 目标位置 {target_pos:.3f}（速度：{move_speed}）")

        # 闭环控制关节运动
        while abs(self.current_pos - target_pos) > POSITION_TOLERANCE:
            # 计算运动步长（方向+大小）
            step = move_speed if target_pos > self.current_pos else -move_speed
            # 更新当前位置（防止超出限位）
            self.current_pos = np.clip(self.current_pos + step, self.joint_min, self.joint_max)
            # 发送位置控制指令


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
            # 步进仿真
            p.stepSimulation()
            time.sleep(DELAY_STEP)
            # 同步模拟器中的实际关节位置
            self.current_pos = p.getJointState(self.arm_id, self.elevator_joint_index)[0]
            # 实时刷新显示（清除当前行，更整洁）
            print(f"🔍 实时位置：{self.current_pos:.3f}", end='\r')

        # 运动完成提示
        print(f"\n✅ 升降运动完成！最终位置：{self.current_pos:.3f}")

    def move_elevator_relative(self, delta_pos: float, speed: Optional[float] = None) -> None:
        """
        相对运动：基于当前位置升降指定距离（新增功能，提升易用性）
        :param delta_pos: 相对位移（正数=上升，负数=下降）
        :param speed: 运动速度
        :return: None
        """
        target_pos = self.current_pos + delta_pos
        self.move_elevator(target_pos, speed)

    def disconnect(self) -> None:
        """断开模拟器连接（容错处理，避免重复断开）"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
            print("\n🔌 已断开与PyBullet模拟器的连接")


# ------------------- 主执行逻辑（解耦，便于测试） -------------------
def main():
    """主函数：执行升降动作序列"""
    arm_controller = None
    try:
        # 初始化控制器
        arm_controller = ArmElevatorController()

        # 执行升降动作序列
        print("\n" + "-" * 50)
        print("📝 执行升降动作序列1：上升到上限60%")
        print("-" * 50)
        arm_controller.move_elevator(target_pos=arm_controller.joint_max * 0.6)
        time.sleep(1)

        print("\n" + "-" * 50)
        print("📝 执行升降动作序列2：下降到下限60%")
        print("-" * 50)
        arm_controller.move_elevator(target_pos=arm_controller.joint_min * 0.6, speed=0.02)
        time.sleep(1)

        print("\n" + "-" * 50)
        print("📝 执行升降动作序列3：相对上升0.5")
        print("-" * 50)
        arm_controller.move_elevator_relative(delta_pos=0.5, speed=0.04)
        time.sleep(1)

        print("\n" + "-" * 50)
        print("📝 执行升降动作序列4：回到初始位置0")
        print("-" * 50)
        arm_controller.move_elevator(target_pos=0)

    except Exception as e:
        print(f"\n❌ 程序执行出错：{str(e)}")
    finally:
        # 确保无论是否出错，都断开连接
        if arm_controller is not None:
            arm_controller.disconnect()


if __name__ == "__main__":
    # 启动程序
    print("🚀 启动机械臂升降控制系统...")
    main()
    print("\n🎉 程序正常结束")
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
