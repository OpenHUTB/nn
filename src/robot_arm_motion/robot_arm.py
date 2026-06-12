import mujoco
import mujoco.viewer as viewer
import numpy as np
import time
import sys
import os
import logging

# 配置logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
Franka Panda 机械臂自动抓取仿真 v1.2
基于MuJoCo实现的基础抓取控制器，增加了场景随机化以测试鲁棒性
"""

# ========== 路径适配 ==========
SCENE_PATH = os.path.join(os.path.dirname(__file__),
                          "mujoco_menagerie-main",
                          "franka_emika_panda",
                          "grab_scene.xml")


# ========== 智能抓取控制器 ==========
class PandaAutoGrab:
    """
    Franka Panda 机械臂智能抓取控制器 (基于MuJoCo)

    该类实现了一个完整的、基于视觉的机械臂抓取和放置任务。
    它通过一个状态机来编排一系列动作，并使用基于雅可比伪逆的操作空间控制
    来精确地移动机械臂末端执行器。

    核心算法:
    - **状态机**: 采用阶段式状态机 (`_grab_phase_machine`) 来管理抓取流程。
    - **运动学控制**: 使用雅可比矩阵的伪逆将末端执行器的笛卡尔空间速度
      指令转换为关节空间的速度指令，实现高精度的位置跟踪。
    - **PD控制器**: 在关节空间使用PD控制器将速度指令转换为力矩输出。

    主要属性:
    - model, data: MuJoCo的核心数据结构，分别表示模型和数据。
    - current_phase: 当前状态机所处的阶段。
    - 各类参数: 如PD增益、速度限制、位置容差等，均定义为类属性以便调整。
    """
    # 状态机阶段常量
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
    PHASE_VERIFY_GRASP = 5.5

    def __init__(self):
        """初始化Franka Panda机械臂抓取控制器，加载模型和初始化参数"""
        self.model = None
        self.data = None
        self.viewer = None
        self.running = True
        self.step_counter = 0
        self.current_phase = 0
        self.grab_complete = False
        self.max_steps_per_phase = 5000 # 添加超时保护

        # 尝试加载模型文件
        try:
            self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
            self.data = mujoco.MjData(self.model)
        except FileNotFoundError:
            logging.error(f"场景文件不存在，请检查路径: {SCENE_PATH}")
            sys.exit(1)
        except mujoco.FatalError as e:
            logging.error(f"加载MJCF模型时发生致命错误: {e}")
            logging.error("请检查XML文件的语法和引用的资源（如mesh, texture）是否正确。")
            sys.exit(1)

        # 【优化1】增加场景随机化
        # 在立方体初始位置上添加一个小范围的随机偏移
        cube_body_id = self.model.body("cube").id
        # 注意：在新版 MuJoCo 中，使用 body_jntadr 来获取物体在 qpos 中的起始索引
        # 由于 "cube" 是一个 free joint，它在 qpos 中有6个值 (x, y, z, qw, qx, qy, qz)
        # 我们只修改前3个值来改变其位置
        qpos_start_idx = self.model.body_jntadr[cube_body_id]
        random_offset = np.array([np.random.uniform(-0.05, 0.05),
                                  np.random.uniform(-0.05, 0.05),
                                  0])
        self.data.qpos[qpos_start_idx: qpos_start_idx + 3] += random_offset
        logging.info(f"已为立方体添加随机初始位置偏移: {np.round(random_offset, 3)}")

        # 机械臂参数
        self.ee_body_id = self.model.body("hand").id
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]
        self.arm_actuator_ids = [self.model.actuator(f"actuator{i}").id for i in range(1, 8)]
        self.gripper_actuator_ids = []
        for i in range(self.model.nu): 
            act_name = self.model.actuator(i).name
            if 'finger' in act_name.lower() or 'gripper' in act_name.lower():
                self.gripper_actuator_ids.append(i)
                logging.info(f"✅ 自动发现夹爪执行器: {act_name} (ID: {i})")

        if not self.gripper_actuator_ids:
            fallback_names = ["actuator8", "actuator9", "gripper_finger1_actuator", "gripper_finger2_actuator"]
            for name in fallback_names:
                try:
                    id = self.model.actuator(name).id
                    self.gripper_actuator_ids.append(id)
                    logging.info(f"✅ 回退发现夹爪执行器: {name} (ID: {id})")
                except:
                    pass

        # 雅克比矩阵
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

        # 抓取参数
        self.cube_body_id = self.model.body("cube").id
        self.target_place_pos = np.array([0.4, -0.2, 0.05])  
        self.gripper_open_ctrl = 255.0  
        self.gripper_close_ctrl = 0.0   
        self.safe_lift_height = 0.15
        self.grab_height = 0.11  
        self.cube_half_size = 0.02
        self.target_quat = np.array([0, 1, 0, 0]) 
        self.grasp_retries = 0
        self.max_grasp_retries = 3


        # PD控制参数
        self.PD_KP = 300  
        self.PD_KD = 80  
        self.TORQUE_LIMIT = 40 

        # 雅克比伪逆参数
        self.JACOBIAN_DAMPING = 0.01 

        # 关节速度参数
        self.JOINT_VEL_LIMIT = 0.5  # 关节速度上限

        # 位置控制参数
        self.POS_TOLERANCE = 0.003  # 末端执行器位置误差容忍阈值

        # 夹爪控制参数
        self.GRIPPER_WAIT_STEPS = 200  # 夹爪动作完成所需的等待步数

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
        logging.info("=" * 50)
        logging.info("📌 模型Body列表: %s", [self.model.body(i).name for i in range(min(self.model.nbody, 10))])
        logging.info("📌 模型Joint列表: %s", [self.model.joint(i).name for i in range(min(self.model.njnt, 10))])
        logging.info("📌 模型Actuator列表: %s", [self.model.actuator(i).name for i in range(min(self.model.nu, 10))])
        logging.info("=" * 50)

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

    def _compute_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        """计算末端执行器的位置雅克比矩阵

        Returns:
            np.ndarray: 3×7的位置雅克比矩阵（仅包含机械臂7个关节的分量）
        """
        mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, self.get_ee_pos(), self.ee_body_id)
        return self.jacp[:, self.joint_ids], self.jacr[:, self.joint_ids]

    def _move_step(self, target_pos: np.ndarray, target_quat: np.ndarray = None, speed: float = 0.3) -> bool:
        """单步位置控制：基于雅克比伪逆实现末端执行器的位置跟踪

        Args:
            target (np.ndarray): 末端执行器的目标位置，形状为(3,)的三维坐标[x, y, z]
            speed (float): 移动速度系数，控制机械臂的运动速度

        Returns:
            bool: 若到达目标位置返回True，否则返回False
        """
        ee_pos = self.get_ee_pos()
        pos_err = target_pos - ee_pos
        dist = np.linalg.norm(pos_err)

        ori_err = np.zeros(3)
        if target_quat is not None:
            ee_quat = self.data.xquat[self.ee_body_id]
            mujoco.mju_subQuat(ori_err, target_quat, ee_quat)

        if dist < self.POS_TOLERANCE and (target_quat is None or np.linalg.norm(ori_err) < 0.05):
            return True

        jp, jr = self._compute_jacobian()

        if target_quat is not None:
            # 6D 控制
            jacobian = np.vstack([jp, jr])
            task_err = np.concatenate([pos_err, ori_err])
            damping = self.JACOBIAN_DAMPING * np.eye(6)
        else:
            jacobian = jp
            task_err = pos_err
            damping = self.JACOBIAN_DAMPING * np.eye(3)

        jacobian_pinv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping)

        adaptive_speed = speed * min(1.0, dist / 0.05)
        joint_vel_cmd = adaptive_speed * jacobian_pinv @ task_err
        joint_vel_cmd = np.clip(joint_vel_cmd, -self.JOINT_VEL_LIMIT, self.JOINT_VEL_LIMIT)

        dt = self.model.opt.timestep

        for i in range(7):
            angle_error = joint_vel_cmd[i] * dt
            torque = self.PD_KP * angle_error - self.PD_KD * self.data.qvel[self.joint_ids[i]]
            torque = np.clip(torque, -self.TORQUE_LIMIT, self.TORQUE_LIMIT)
            self.data.ctrl[self.arm_actuator_ids[i]] = torque
        return False

    def _gripper_step(self, pos: float) -> None:
        """单步夹爪位置控制，设置夹爪的目标开合位置

        Args:
            pos (float): 夹爪目标位置，0.04为完全打开，0.005为闭合抓取
        """
        if self.gripper_actuator_ids:
            for actuator_id in self.gripper_actuator_ids:
                ctrlrange = self.model.actuator_ctrlrange[actuator_id]
                final_ctrl = np.clip(pos, ctrlrange[0], ctrlrange[1])

                if actuator_id < len(self.data.ctrl):
                    self.data.ctrl[actuator_id] = final_ctrl
                else:
                    logging.error(f"❌ 执行器ID {actuator_id} 超出 ctrl 数组范围!")
        else:

            pass

    def _grab_phase_machine(self) -> None:
        """抓取状态机：按阶段执行机械臂的抓取、移动、放置等一系列动作"""
        if self.current_phase in [self.PHASE_MOVE_TO_CUBE_ABOVE, self.PHASE_MOVE_TO_GRAB_HEIGHT]:
            self.cube_pos = self.get_cube_pos()

        if hasattr(self, '_phase_start_step') and (self.step_counter - self._phase_start_step) > self.max_steps_per_phase:
            logging.warning(f"⚠️ 阶段 {self.current_phase} 超时，强制进入下一阶段。")
            self._advance_phase()

        if self.current_phase == self.PHASE_MOVE_TO_INIT:
            self._gripper_step(self.gripper_open_ctrl) 
            if self._move_step(self.INIT_EE_POS, self.target_quat):
                logging.info("✅ 到达初始位置")
                self._advance_phase()
            else:
                self._set_phase_start()

        elif self.current_phase == self.PHASE_DETECT_CUBE:
            self.cube_pos = self.get_cube_pos()
            logging.info("🎯 识别到立方体位置: %s", np.round(self.cube_pos, 3))
            self._advance_phase()

        elif self.current_phase == self.PHASE_MOVE_TO_CUBE_ABOVE:
            target = self.cube_pos + np.array([0, 0, self.safe_lift_height])
            if self._move_step(target, self.target_quat, speed=0.4):
                logging.info("✅ 到达立方体上方")
                self._advance_phase()
            else:
                self._set_phase_start()

        elif self.current_phase == self.PHASE_OPEN_GRIPPER:
            if self.step_counter == 0:
                self._gripper_step(self.gripper_open_ctrl)
                logging.info("✋ 打开夹爪")
            if self.step_counter > self.GRIPPER_WAIT_STEPS:
                self._advance_phase()
            self.step_counter += 1
            self._set_phase_start_if_unset()

        elif self.current_phase == self.PHASE_MOVE_TO_GRAB_HEIGHT:
            grab_target_z = self.cube_pos[2] + self.grab_height
            target = np.array([self.cube_pos[0], self.cube_pos[1], grab_target_z])
            if self._move_step(target, self.target_quat, speed=0.5):
                logging.info("✅ 下降到抓取高度: %.3f (立方体中心 Z: %.3f)", grab_target_z, self.cube_pos[2])
                self._advance_phase()
            else:
                self._set_phase_start()

        elif self.current_phase == self.PHASE_CLOSE_GRIPPER:
            if self.step_counter == 0:
                self._gripper_step(self.gripper_close_ctrl)
                logging.info("🤏 闭合夹爪抓取")
            if self.step_counter > self.GRIPPER_WAIT_STEPS:
                self.current_phase = self.PHASE_VERIFY_GRASP
                self.step_counter = 0
            self.step_counter += 1
            self._set_phase_start_if_unset()

        elif self.current_phase == self.PHASE_VERIFY_GRASP:
            has_contact = False
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                body1 = self.model.body(self.model.geom_bodyid[contact.geom1]).name
                body2 = self.model.body(self.model.geom_bodyid[contact.geom2]).name
                if ("finger" in body1 and "cube" in body2) or ("finger" in body2 and "cube" in body1):
                    has_contact = True
                    break

            if has_contact:
                logging.info("✅ 抓取确认成功")
                self.current_phase = self.PHASE_LIFT_CUBE
                self.grasp_retries = 0
            else:
                self.grasp_retries += 1
                if self.grasp_retries < self.max_grasp_retries:
                    logging.warning("⚠️ 抓取失败，重试中 (%d/%d)", self.grasp_retries, self.max_grasp_retries)
                    self.current_phase = self.PHASE_OPEN_GRIPPER
                else:
                    logging.error("❌ 多次抓取失败，放弃任务")
                    self.current_phase = self.PHASE_MOVE_BACK_TO_INIT
            self.step_counter = 0
            self._set_phase_start()

        elif self.current_phase == self.PHASE_LIFT_CUBE:
            lift_target = self.get_cube_pos() + np.array([0, 0, self.safe_lift_height])
            if self._move_step(lift_target, self.target_quat, speed=0.3):
                logging.info("✅ 抬升立方体")
                self._advance_phase()
            else:
                self._set_phase_start()

        elif self.current_phase == self.PHASE_MOVE_TO_PLACE_ABOVE:
            target = self.target_place_pos + np.array([0, 0, self.safe_lift_height])
            if self._move_step(target, self.target_quat, speed=0.4):
                logging.info("✅ 到达放置点上方")
                self._advance_phase()
            else:
                self._set_phase_start()

        elif self.current_phase == self.PHASE_MOVE_TO_PLACE_HEIGHT:
            target = self.target_place_pos + np.array([0, 0, self.grab_height])
            if self._move_step(target, self.target_quat, speed=0.5):
                logging.info("✅ 下降到放置高度")
                elf._advance_phase()
            else:
                self._set_phase_start()

        elif self.current_phase == self.PHASE_RELEASE_CUBE:
            if self.step_counter == 0:
                self._gripper_step(self.gripper_open_ctrl)
                logging.info("🫳 释放立方体")
            if self.step_counter > self.GRIPPER_WAIT_STEPS:
                sself._advance_phase()
            self.step_counter += 1
            self._set_phase_start_if_unset()

        elif self.current_phase == self.PHASE_MOVE_BACK_FROM_PLACE:
            target = self.target_place_pos + np.array([0, 0, self.safe_lift_height])
            if self._move_step(target, self.target_quat, speed=0.3):
                logging.info("✅ 撤离机械臂")
                self._advance_phase()
            else:
                self._set_phase_start()

        elif self.current_phase == self.PHASE_MOVE_BACK_TO_INIT:
            if self._move_step(self.INIT_EE_POS, self.target_quat, speed=0.4):
                logging.info("✅ 返回初始位置")
                self._advance_phase()
            else:
                self._set_phase_start()

        elif self.current_phase == self.PHASE_FINISHED:
            if not self.grab_complete:
                logging.info("=" * 50)
                logging.info("✅ 智能抓取任务完成！")
                logging.info("=" * 50)
                self.grab_complete = True
            for i in range(7):
                self.data.ctrl[self.joint_ids[i]] = 0

    def _advance_phase(self):
        """进入下一阶段"""
        self.current_phase += 1
        self.step_counter = 0

    def _set_phase_start(self):
        """记录当前阶段开始的步数"""
        if not hasattr(self, '_phase_start_step'):
            self._phase_start_step = self.step_counter

    def _set_phase_start_if_unset(self):
        """如果未设置，则记录当前阶段开始的步数"""
        if not hasattr(self, '_phase_start_step'):
            self._set_phase_start()

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

        logging.info("🚀 仿真已启动，开始自动抓取...")
        logging.info("💡 关闭Viewer窗口可退出程序")

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
                self.step_counter += 1 
        except KeyboardInterrupt:
            logging.warning("⚠️ 检测到Ctrl+C，正在退出仿真...")

        self.running = False
        self.viewer.close()
        logging.info("👋 仿真结束")


# ========== 主函数 ==========
if __name__ == "__main__":
    try:
        panda = PandaAutoGrab()
        panda.run()
    except Exception as e:
        logging.error(f"❌ 程序发生未处理的错误: {e}", exc_info=True)
        sys.exit(1)