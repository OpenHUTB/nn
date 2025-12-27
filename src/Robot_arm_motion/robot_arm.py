import mujoco
import mujoco.viewer as viewer
import numpy as np
import time
import sys
import os

"""
Franka Panda 机械臂自动抓取仿真 v1.1
基于MuJoCo实现的基础抓取控制器
"""

# ========== 路径适配 ==========
SCENE_PATH = os.path.join(os.path.dirname(__file__),
                          "mujoco_menagerie-main",
                          "franka_emika_panda",
                          "grab_scene.xml")

if not os.path.exists(SCENE_PATH):
    print(f"❌ 场景文件不存在：{SCENE_PATH}")
    sys.exit(1)


# ========== 智能抓取控制器 ==========
class PandaAutoGrab:
    # 【优化1】定义状态机阶段的具名常量，提升代码可读性
    PHASE_MOVE_TO_INIT = 0
    PHASE_DETECT_CUBE = 1
    PHASE_MOVE_TO_CUBE_ABOVE = 2
    PHASE_OPEN_GRIPPER = 3
    PHASE_MOVE_TO_GRAB_HEIGHT = 4
    PHASE_CLOSE_GRIPPER = 5
    PHASE_LIFT_CUBE = 6
    PHASE_MOVE_TO_PLACE_ABOVE = 7
    PHASE_MOVE_TO_PLACE_HEIGHT = 8
    PHASE_RELEASE_CUBE = 9
    PHASE_MOVE_BACK_FROM_PLACE = 10
    PHASE_MOVE_BACK_TO_INIT = 11
    PHASE_FINISHED = 12

    def __init__(self):
        """初始化Franka Panda机械臂抓取控制器，加载模型和初始化参数"""
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.running = True
        self.step_counter = 0
        self.current_phase = 0
        self.grab_complete = False

        # 机械臂参数
        self.ee_body_id = self.model.body("hand").id
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]
        self.gripper_joint_names = ["finger_joint1", "finger_joint2"]

        # 雅克比矩阵
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

        # 抓取参数
        self.cube_body_id = self.model.body("cube").id
        self.target_place_pos = np.array([0.3, 0.0, 0.1])
        self.gripper_open_pos = 0.04
        self.gripper_close_pos = 0.005
        self.safe_lift_height = 0.15
        self.grab_height = 0.05

        # PD控制参数
        self.PD_KP = 250  # 比例增益
        self.PD_KD = 100  # 微分增益
        self.TORQUE_LIMIT = 20  # 力矩限制

        # 雅克比伪逆参数
        self.JACOBIAN_DAMPING = 0.01  # 雅克比伪逆的阻尼系数

        # 关节速度参数
        self.JOINT_VEL_LIMIT = 0.5  # 关节速度上限

        # 位置控制参数
        self.POS_TOLERANCE = 0.003  # 末端执行器位置误差容忍阈值

        # 夹爪控制参数
        self.GRIPPER_WAIT_STEPS = 100  # 夹爪动作完成所需的等待步数

        # 位置坐标参数
        self.INIT_EE_POS = np.array([0.4, 0.0, 0.2])  # 末端执行器初始目标位置
        self.LIFT_HEIGHT_INCREMENT = 0.05  # 抓取后额外抬升的高度增量

        # 相机视角参数
        self.CAM_AZIMUTH = 70  # 相机方位角
        self.CAM_ELEVATION = -25  # 相机仰角
        self.CAM_DISTANCE = 1.8  # 相机距离
        self.CAM_LOOKAT = np.array([0.4, 0.0, 0.1])  # 相机注视点

        # 仿真控制参数
        self.SIMULATION_SLEEP = 1 / 200  # 仿真循环的休眠时间

        # 打印模型信息
        print("=" * 50)
        print("📌 模型Body列表：", [self.model.body(i).name for i in range(min(self.model.nbody, 10))])
        print("📌 模型Joint列表：", [self.model.joint(i).name for i in range(min(self.model.njnt, 10))])
        print("=" * 50)

    def get_ee_pos(self) -> np.ndarray:
        """获取末端执行器位置

        Returns:
            np.ndarray: 末端执行器的三维位置坐标[x, y, z]
        """
        return self.data.xpos[self.ee_body_id].copy()

    def get_cube_pos(self) -> np.ndarray:
        """获取立方体位置

        Returns:
            np.ndarray: 立方体的三维位置坐标[x, y, z]
        """
        return self.data.xpos[self.cube_body_id].copy()

    def _compute_jacobian(self) -> np.ndarray:
        """计算末端执行器的位置雅克比矩阵

        Returns:
            np.ndarray: 3×7的位置雅克比矩阵（仅包含机械臂7个关节的分量）
        """
        mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, self.get_ee_pos(), self.ee_body_id)
        return self.jacp[:, self.joint_ids]

    def _move_step(self, target: np.ndarray, speed: float = 0.3) -> bool:
        """单步位置控制：基于雅克比伪逆实现末端执行器的位置跟踪

        Args:
            target (np.ndarray): 末端执行器的目标位置，形状为(3,)的三维坐标[x, y, z]
            speed (float): 移动速度系数，控制机械臂的运动速度

        Returns:
            bool: 若到达目标位置返回True，否则返回False
        """
        ee_pos = self.get_ee_pos()
        error = target - ee_pos
        error_norm = np.linalg.norm(error)

        if error_norm < self.POS_TOLERANCE:
            return True  # 到达目标

        jacobian = self._compute_jacobian()
        jacobian_pinv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + self.JACOBIAN_DAMPING * np.eye(3))

        joint_vel_cmd = speed * jacobian_pinv @ error
        joint_vel_cmd = np.clip(joint_vel_cmd, -self.JOINT_VEL_LIMIT, self.JOINT_VEL_LIMIT)

        torque = np.zeros(7)
        for i in range(7):
            angle_error = joint_vel_cmd[i] * 0.1
            torque[i] = self.PD_KP * angle_error - self.PD_KD * self.data.qvel[self.joint_ids[i]]
            torque[i] = np.clip(torque[i], -self.TORQUE_LIMIT, self.TORQUE_LIMIT)

        for i in range(7):
            self.data.ctrl[self.joint_ids[i]] = torque[i]

        return False

    def _gripper_step(self, pos: float) -> None:
        """单步夹爪位置控制，设置夹爪的目标开合位置

        Args:
            pos (float): 夹爪目标位置，0.04为完全打开，0.005为闭合抓取
        """
        for j_name in self.gripper_joint_names:
            j_id = self.model.joint(j_name).id
            self.data.ctrl[j_id] = pos

    def _grab_phase_machine(self) -> None:
        """抓取状态机：按阶段执行机械臂的抓取、移动、放置等一系列动作

        状态机分为12个阶段，从初始位置移动→识别立方体→抓取→放置→返回，
        每个阶段完成后自动切换到下一个阶段，直到抓取任务完成。
        """
        # 【优化2】使用具名常量替代数字，使逻辑更清晰
        if self.current_phase == self.PHASE_MOVE_TO_INIT:
            if self._move_step(self.INIT_EE_POS):
                print("\n✅ 到达初始位置")
                self.current_phase = self.PHASE_DETECT_CUBE
                self.step_counter = 0

        elif self.current_phase == self.PHASE_DETECT_CUBE:
            self.cube_pos = self.get_cube_pos()
            print(f"\n🎯 识别到立方体位置：{np.round(self.cube_pos, 3)}")
            self.current_phase = self.PHASE_MOVE_TO_CUBE_ABOVE

        elif self.current_phase == self.PHASE_MOVE_TO_CUBE_ABOVE:
            if self._move_step(self.cube_pos + np.array([0, 0, self.safe_lift_height]), speed=0.4):
                print("\n✅ 到达立方体上方")
                self.current_phase = self.PHASE_OPEN_GRIPPER
                self.step_counter = 0

        elif self.current_phase == self.PHASE_OPEN_GRIPPER:
            if self.step_counter == 0:
                self._gripper_step(self.gripper_open_pos)
                print("\n✋ 打开夹爪")
            if self.step_counter > self.GRIPPER_WAIT_STEPS:
                self.current_phase = self.PHASE_MOVE_TO_GRAB_HEIGHT
                self.step_counter = 0
            self.step_counter += 1

        elif self.current_phase == self.PHASE_MOVE_TO_GRAB_HEIGHT:
            if self._move_step(self.cube_pos + np.array([0, 0, self.grab_height]), speed=0.2):
                print("\n✅ 下降到抓取高度")
                self.current_phase = self.PHASE_CLOSE_GRIPPER
                self.step_counter = 0

        elif self.current_phase == self.PHASE_CLOSE_GRIPPER:
            if self.step_counter == 0:
                self._gripper_step(self.gripper_close_pos)
                print("\n🤏 闭合夹爪抓取")
            if self.step_counter > self.GRIPPER_WAIT_STEPS:
                self.current_phase = self.PHASE_LIFT_CUBE
                self.step_counter = 0
            self.step_counter += 1

        elif self.current_phase == self.PHASE_LIFT_CUBE:
            lift_target = self.cube_pos + np.array([0, 0, self.safe_lift_height + self.LIFT_HEIGHT_INCREMENT])
            if self._move_step(lift_target, speed=0.3):
                print("\n✅ 抬升立方体")
                self.current_phase = self.PHASE_MOVE_TO_PLACE_ABOVE
                self.step_counter = 0

        elif self.current_phase == self.PHASE_MOVE_TO_PLACE_ABOVE:
            if self._move_step(self.target_place_pos + np.array([0, 0, self.safe_lift_height]), speed=0.4):
                print("\n✅ 到达放置点上方")
                self.current_phase = self.PHASE_MOVE_TO_PLACE_HEIGHT
                self.step_counter = 0

        elif self.current_phase == self.PHASE_MOVE_TO_PLACE_HEIGHT:
            if self._move_step(self.target_place_pos + np.array([0, 0, self.grab_height]), speed=0.2):
                print("\n✅ 下降到放置高度")
                self.current_phase = self.PHASE_RELEASE_CUBE
                self.step_counter = 0

        elif self.current_phase == self.PHASE_RELEASE_CUBE:
            if self.step_counter == 0:
                self._gripper_step(self.gripper_open_pos)
                print("\n🫳 释放立方体")
            if self.step_counter > self.GRIPPER_WAIT_STEPS:
                self.current_phase = self.PHASE_MOVE_BACK_FROM_PLACE
                self.step_counter = 0
            self.step_counter += 1

        elif self.current_phase == self.PHASE_MOVE_BACK_FROM_PLACE:
            if self._move_step(self.target_place_pos + np.array([0, 0, self.safe_lift_height]), speed=0.3):
                print("\n✅ 撤离机械臂")
                self.current_phase = self.PHASE_MOVE_BACK_TO_INIT
                self.step_counter = 0

        elif self.current_phase == self.PHASE_MOVE_BACK_TO_INIT:
            if self._move_step(self.INIT_EE_POS, speed=0.4):
                print("\n✅ 返回初始位置")
                self.current_phase = self.PHASE_FINISHED

        elif self.current_phase == self.PHASE_FINISHED:
            if not self.grab_complete:
                print("\n" + "=" * 50)
                print("✅ 智能抓取任务完成！")
                print("=" * 50)
                self.grab_complete = True

    def _init_camera(self) -> None:
        """初始化Viewer的相机视角"""
        self.viewer.cam.azimuth = self.CAM_AZIMUTH
        self.viewer.cam.elevation = self.CAM_ELEVATION
        self.viewer.cam.distance = self.CAM_DISTANCE
        self.viewer.cam.lookat = self.CAM_LOOKAT

    def run(self):
        """单线程仿真主循环"""
        self.viewer = viewer.launch_passive(self.model, self.data)
        self._init_camera()

        print("\n🚀 仿真已启动，开始自动抓取...")
        print("💡 关闭Viewer窗口可退出程序")

        try:
            while self.viewer.is_running():
                if self.running and not self.grab_complete:
                    self._grab_phase_machine()
                else:
                    for i in range(7):
                        self.data.ctrl[self.joint_ids[i]] = 0

                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                time.sleep(self.SIMULATION_SLEEP)
        except KeyboardInterrupt:
            print("\n⚠️ 检测到Ctrl+C，正在退出仿真...")

        self.running = False
        self.viewer.close()
        print("\n👋 仿真结束")


# ========== 主函数 ==========
if __name__ == "__main__":
    try:
        panda = PandaAutoGrab()
        panda.run()
    except Exception as e:
        print(f"\n❌ 程序错误：{e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)