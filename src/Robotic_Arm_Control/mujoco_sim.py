import mujoco
import mujoco.viewer
import numpy as np
import time
import logging
import os

# 修复循环导入
try:
    from core.kinematics import RoboticArmKinematics
except ImportError as e:
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.kinematics import RoboticArmKinematics

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MuJoCoArmSim:
    """MuJoCo机械臂仿真类（防关节溢出版）"""

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

        # 初始化运动学
        self.kinematics = RoboticArmKinematics()

        # 缓存ID
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.joint_ids = self._get_ids(mujoco.mjtObj.mjOBJ_JOINT, self.joint_names)
        self.actuator_ids = self._get_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, [f"act{i + 1}" for i in range(6)])

        # 仿真参数
        self.fps = 30
        self.dt = 1.0 / self.fps

        # 关节角滤波缓存（防数值溢出）
        self.last_joint_angles = None
        self.filter_alpha = 0.2  # 低通滤波系数

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

        # 先裁剪关节角
        joint_angles = self.kinematics._clip_joint_angles(joint_angles)
        joint_radians = np.radians(joint_angles)

        # 设置控制器输入
        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = joint_radians[i]

    def _filter_joint_angles(self, raw_angles):
        """关节角低通滤波：去除异常值，防溢出"""
        # 1. 归一化角度到 [-180, 180]
        normalized_angles = []
        for angle in raw_angles:
            angle = angle % 360
            if angle > 180:
                angle -= 360
            normalized_angles.append(angle)

        # 2. 低通滤波（避免突变）
        if self.last_joint_angles is None:
            self.last_joint_angles = normalized_angles
            return normalized_angles

        filtered_angles = []
        for i in range(6):
            filtered = self.filter_alpha * normalized_angles[i] + (1 - self.filter_alpha) * self.last_joint_angles[i]
            # 检测异常值（超过限位2倍则使用上一帧值）
            if abs(filtered) > 180:
                filtered = self.last_joint_angles[i]
                logger.warning(f"关节{self.joint_names[i]}检测到异常值{normalized_angles[i]:.2f}度，已使用上一帧值")
            filtered_angles.append(filtered)

        self.last_joint_angles = filtered_angles
        return filtered_angles

    def get_joint_angles(self):
        """获取当前关节角（滤波+防溢出）"""
        # 读取原始关节角（弧度转角度）
        raw_radians = [self.data.joint(jid).qpos[0] for jid in self.joint_ids]
        raw_angles = np.degrees(raw_radians).tolist()

        # 滤波处理
        filtered_angles = self._filter_joint_angles(raw_angles)

        # 四舍五入，减少精度问题
        return [round(angle, 2) for angle in filtered_angles]

    def _init_viewer(self, viewer):
        """初始化Viewer视角"""
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -15
        viewer.cam.lookat = np.array([0.2, 0, 0.5])
        logger.info("Viewer初始化完成")

    def run_simulation(self, target_pose=None, duration=10.0):
        """运行仿真（防溢出版）"""
        start_time = time.time()
        frame_count = 0

        # 目标位姿预处理
        if target_pose is not None:
            logger.info(f"目标末端位姿：{target_pose}（米/度）")
            try:
                # 更安全的初始关节角
                initial_joints = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
                target_joints = self.kinematics.inverse_kinematics(
                    target_pose, initial_joints=initial_joints
                )
                logger.info(f"逆解关节角：{target_joints}（度）")
                self.set_joint_angles(target_joints)
            except Exception as e:
                logger.error(f"逆解失败：{e}")
                # 安全默认关节角
                default_joints = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
                logger.info(f"使用安全默认关节角：{default_joints}")
                self.set_joint_angles(default_joints)

        # 启动Viewer
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        try:
            self._init_viewer(viewer)

            # 仿真主循环
            while viewer.is_running() and (time.time() - start_time) < duration:
                frame_start = time.time()

                # 执行仿真步
                mujoco.mj_step(self.model, self.data)

                # 每5帧打印状态（减少计算量）
                if frame_count % 5 == 0:
                    try:
                        current_joints = self.get_joint_angles()
                        current_pose = self.kinematics.forward_kinematics(current_joints)
                        print(f"\r仿真时长：{time.time() - start_time:.1f}s | 末端位姿：{current_pose}", end="")
                    except Exception as e:
                        # 即使单帧出错，也不终止仿真
                        logger.warning(f"帧{frame_count}计算失败：{e}")
                        print(f"\r仿真时长：{time.time() - start_time:.1f}s | 状态异常（已忽略）", end="")

                # 同步Viewer
                viewer.sync()

                # 帧率控制
                frame_elapsed = time.time() - frame_start
                time.sleep(max(0, self.dt - frame_elapsed))

                frame_count += 1
        finally:
            viewer.close()

        # 仿真结束
        avg_fps = frame_count / (time.time() - start_time)
        logger.info(f"\n仿真结束：总帧数{frame_count}，平均帧率{avg_fps:.1f}FPS")


def main():
    """主函数"""
    try:
        sim = MuJoCoArmSim()
        # 极保守的目标位姿（绝对在工作空间内）
        target_pose = [0.1, 0.0, 0.3, 0, 0, 0]
        sim.run_simulation(target_pose=target_pose, duration=10.0)
    except Exception as e:
        logger.error(f"仿真失败：{e}")
        raise


if __name__ == "__main__":
    main()