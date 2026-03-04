import mujoco
import mujoco.viewer
import numpy as np
import time
import logging
import os
import threading
import sys

# 修复循环导入
try:
    from core.kinematics import RoboticArmKinematics
    from core.arm_functions import ArmFunctions
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.kinematics import RoboticArmKinematics
    from core.arm_functions import ArmFunctions

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MuJoCoArmSim:
    """集成所有功能的MuJoCo机械臂仿真类"""

    def __init__(self, model_path="model/six_axis_arm.xml"):
        # 路径校验
        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")

        # 加载模型
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            logger.info(f"成功加载MuJoCo模型：{self.model_path}")
            logger.info(f"MuJoCo版本：{mujoco.__version__}")
        except Exception as e:
            logger.error(f"加载模型失败：{e}")
            raise

        # 初始化核心模块
        self.kinematics = RoboticArmKinematics()
        self.arm_functions = ArmFunctions(self.kinematics)

        # 缓存ID
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.joint_ids = self._get_ids(mujoco.mjtObj.mjOBJ_JOINT, self.joint_names)
        self.actuator_ids = self._get_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, [f"act{i + 1}" for i in range(6)])

        # 仿真参数
        self.fps = 30
        self.dt = 1.0 / self.fps

        # 关节角滤波缓存
        self.last_joint_angles = None
        self.filter_alpha = 0.2

        # 功能开关和状态
        self.running = True
        self.current_trajectory_index = 0  # 轨迹执行索引
        self.manual_control_key = 'stop'  # 手动控制按键
        self.moving_target_pos = [0.1, 0.0, 0.3]  # 初始跟随目标点
        self.link_geom_names = [  # 机械臂连杆几何名称
            "base_geom", "link1_geom", "link2_geom", "link3_geom",
            "link4_geom", "link5_geom", "end_effector"
        ]

        # 启动手动控制线程（非阻塞读取键盘输入）
        self.manual_thread = threading.Thread(target=self._manual_control_listener, daemon=True)
        self.manual_thread.start()

    def _get_ids(self, obj_type, names):
        """批量获取MuJoCo对象ID"""
        ids = []
        for name in names:
            obj_id = mujoco.mj_name2id(self.model, obj_type, name)
            if obj_id == -1:
                raise ValueError(f"未找到对象：{name}")
            ids.append(obj_id)
        return ids

    def set_joint_angles(self, joint_angles):
        """设置关节角（低版本适配）"""
        if len(joint_angles) != 6:
            raise ValueError(f"需输入6个关节角，当前输入{len(joint_angles)}个")

        joint_angles = self.kinematics._clip_joint_angles(joint_angles)
        joint_radians = np.radians(joint_angles)

        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = joint_radians[i]

    def _filter_joint_angles(self, raw_angles):
        """关节角低通滤波"""
        normalized_angles = []
        for angle in raw_angles:
            angle = angle % 360
            if angle > 180:
                angle -= 360
            normalized_angles.append(angle)

        if self.last_joint_angles is None:
            self.last_joint_angles = normalized_angles
            return normalized_angles

        filtered_angles = []
        for i in range(6):
            filtered = self.filter_alpha * normalized_angles[i] + (1 - self.filter_alpha) * self.last_joint_angles[i]
            if abs(filtered) > 180:
                filtered = self.last_joint_angles[i]
                logger.warning(f"关节{self.joint_names[i]}检测到异常值，已滤波")
            filtered_angles.append(filtered)

        self.last_joint_angles = filtered_angles
        return filtered_angles

    def get_joint_angles(self):
        """获取当前关节角（滤波+防溢出）"""
        raw_radians = [self.data.joint(jid).qpos[0] for jid in self.joint_ids]
        raw_angles = np.degrees(raw_radians).tolist()
        filtered_angles = self._filter_joint_angles(raw_angles)
        return [round(angle, 2) for angle in filtered_angles]

    def _init_viewer(self, viewer):
        """初始化Viewer视角"""
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -15
        viewer.cam.lookat = np.array([0.2, 0, 0.5])
        logger.info("Viewer初始化完成")

    def _manual_control_listener(self):
        """手动控制键盘监听线程（非阻塞）"""
        logger.info("===== 手动控制说明 =====")
        logger.info("按键格式：j1+ / j1- / j2+ / j2- ... / j6+ / j6- / stop")
        logger.info("输入示例：j1+ → 关节1增加1度 | stop → 停止手动控制")
        logger.info("========================")

        while self.running:
            try:
                key = input("请输入控制按键：").strip()
                self.manual_control_key = key
            except:
                continue

    def _update_moving_target(self):
        """更新移动目标点（正弦运动，演示跟随效果）"""
        # 目标点在x-z平面做正弦运动
        t = time.time()
        self.moving_target_pos[0] = 0.1 + 0.05 * math.sin(t)
        self.moving_target_pos[2] = 0.3 + 0.05 * math.cos(t)

    def run_simulation(self, mode="trajectory", duration=30.0):
        """
        运行仿真（支持多模式）
        :param mode: 运行模式 - trajectory:轨迹规划 | manual:手动控制 | follow:目标跟随
        :param duration: 仿真时长（秒）
        """
        start_time = time.time()
        frame_count = 0

        # 初始化模式
        if mode == "trajectory":
            # 生成从初始位姿到目标位姿的轨迹
            initial_joints = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
            target_joints = self.kinematics.inverse_kinematics([0.15, 0.0, 0.35, 0, 0, 0])
            self.arm_functions.generate_linear_trajectory(initial_joints, target_joints, num_points=100)
            logger.info(f"启动轨迹规划模式，轨迹点数量：{len(self.arm_functions.trajectory_points)}")

        elif mode == "manual":
            logger.info("启动手动控制模式（按提示输入按键控制关节）")

        elif mode == "follow":
            logger.info("启动目标点跟随模式（目标点做正弦运动）")

        else:
            raise ValueError(f"无效模式：{mode}，支持：trajectory/manual/follow")

        # 启动Viewer
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        try:
            self._init_viewer(viewer)

            # 仿真主循环
            while viewer.is_running() and (time.time() - start_time) < duration:
                frame_start = time.time()

                # 1. 根据模式更新关节角
                current_joints = self.get_joint_angles()
                new_joints = current_joints

                if mode == "trajectory":
                    # 轨迹规划模式
                    next_point, self.current_trajectory_index = self.arm_functions.get_next_trajectory_point(
                        self.current_trajectory_index
                    )
                    if next_point is not None:
                        new_joints = next_point

                elif mode == "manual":
                    # 手动控制模式
                    new_joints = self.arm_functions.manual_joint_control(
                        current_joints, self.manual_control_key, step=1.0
                    )
                    self.manual_control_key = 'stop'  # 单次按键生效

                elif mode == "follow":
                    # 目标跟随模式
                    self._update_moving_target()
                    new_joints = self.arm_functions.follow_moving_target(
                        current_joints, self.moving_target_pos
                    )

                # 2. 设置新关节角
                self.set_joint_angles(new_joints)

                # 3. 执行仿真步
                mujoco.mj_step(self.model, self.data)

                # 4. 碰撞检测
                has_collision, collision_pairs = self.arm_functions.check_collision(
                    self.model, self.data, self.link_geom_names
                )

                # 5. 打印状态
                if frame_count % 5 == 0:
                    try:
                        current_pose = self.kinematics.forward_kinematics(current_joints)
                        collision_info = "【碰撞】" if has_collision else ""
                        print(f"\r仿真时长：{time.time() - start_time:.1f}s | 末端位姿：{current_pose} {collision_info}",
                              end="")
                    except Exception as e:
                        print(f"\r仿真时长：{time.time() - start_time:.1f}s | 状态异常：{e}", end="")

                # 6. 同步Viewer
                viewer.sync()

                # 7. 帧率控制
                frame_elapsed = time.time() - frame_start
                time.sleep(max(0, self.dt - frame_elapsed))

                frame_count += 1
        finally:
            self.running = False
            viewer.close()
            logger.info(f"\n仿真结束：总帧数{frame_count}，平均帧率{frame_count / (time.time() - start_time):.1f}FPS")


def main():
    """主函数（支持选择运行模式）"""
    try:
        sim = MuJoCoArmSim()

        # 选择运行模式
        print("请选择仿真模式：")
        print("1 - 轨迹规划模式（机械臂沿平滑轨迹运动）")
        print("2 - 手动控制模式（键盘控制单个关节）")
        print("3 - 目标跟随模式（跟随移动的目标点）")
        mode_choice = input("输入模式编号（1/2/3）：").strip()

        mode_map = {"1": "trajectory", "2": "manual", "3": "follow"}
        mode = mode_map.get(mode_choice, "trajectory")

        # 运行仿真
        sim.run_simulation(mode=mode, duration=30.0)
    except Exception as e:
        logger.error(f"仿真失败：{e}")
        raise


if __name__ == "__main__":
    main()