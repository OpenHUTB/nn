import mujoco
import mujoco.viewer
import numpy as np
import time
import logging
import os
import threading
import sys
import queue

# 导入核心类
try:
    from core.base_arm import BaseRoboticArm, BaseMuJoCoSim
    from core.arm_extensions import PIDController, TrajectoryManager, TargetVisualizer
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from core.base_arm import BaseRoboticArm, BaseMuJoCoSim
    from core.arm_extensions import PIDController, TrajectoryManager, TargetVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RoboticArmSim(BaseMuJoCoSim):
    """集成所有功能的机械臂仿真类"""

    def __init__(self, model_path="model/six_axis_arm.xml", config_path="config/arm_config.yaml"):
        # 初始化基础类
        super().__init__(model_path)
        self.arm = BaseRoboticArm(config_path)

        # 初始化扩展功能
        self.pid = PIDController()
        self.trajectory_mgr = TrajectoryManager(self.arm)
        self.target_vis = TargetVisualizer(self)

        # 运行状态
        self.running = True
        self.key_queue = queue.Queue()
        self.current_traj_idx = 0
        self.mode = "trajectory"
        self.duration = 30.0

        # 手动控制线程
        self.manual_thread = None

    def _manual_control_listener(self):
        """手动控制监听"""
        logger.info("\n===== 增强版手动控制说明 =====")
        logger.info("基础控制：j1+ / j1- / j2+ / j2- ... / j6+ / j6- (步长1度)")
        logger.info("轨迹控制：save_traj (保存当前轨迹) | load_traj (加载轨迹)")
        logger.info("目标控制：set_target x,y,z (设置目标点，如 set_target 0.2,0,0.4)")
        logger.info("退出控制：stop | quit")
        logger.info("==============================\n")

        while self.running:
            try:
                key = input("请输入控制指令：").strip()
                if key == 'quit':
                    self.running = False
                    break
                self.key_queue.put(key)
            except:
                continue

    def _process_manual_key(self, key, current_joints):
        """处理手动控制指令"""
        new_joints = current_joints.copy()

        # 基础关节控制
        joint_map = {'j1+': 0, 'j1-': 0, 'j2+': 1, 'j2-': 1, 'j3+': 2, 'j3-': 2,
                     'j4+': 3, 'j4-': 3, 'j5+': 4, 'j5-': 4, 'j6+': 5, 'j6-': 5}

        if key in joint_map:
            idx = joint_map[key]
            step = 1.0 if '+' in key else -1.0
            new_joints[idx] += step
            new_joints = self.arm._clip_joint_angles(new_joints)

        # 轨迹控制
        elif key == 'save_traj':
            self.trajectory_mgr.save_trajectory()

        elif key == 'load_traj':
            if self.trajectory_mgr.load_trajectory():
                self.current_traj_idx = 0

        # 目标点控制
        elif key.startswith('set_target'):
            try:
                pos_str = key.split(' ')[1]
                pos = [float(x) for x in pos_str.split(',')]
                if len(pos) == 3:
                    self.target_vis.update_target(pos)
                    logger.info(f"目标点已更新：{pos}")
            except:
                logger.error("目标点格式错误，示例：set_target 0.2,0,0.4")

        return new_joints

    def _update_moving_target(self):
        """更新移动目标点"""
        t = time.time()
        self.target_vis.update_target([
            0.1 + 0.05 * np.sin(t),
            0.0 + 0.03 * np.cos(t),
            0.3 + 0.04 * np.sin(t / 2)
        ])

    def run_simulation(self, mode="trajectory", duration=30.0):
        """运行仿真（集成所有增强功能）"""
        self.mode = mode
        self.duration = duration
        start_time = time.time()
        frame_count = 0

        # 初始化模式
        if mode == "trajectory":
            # 生成默认轨迹
            start_joints = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
            target_joints = self.arm.inverse_kinematics([0.15, 0.0, 0.35, 0, 0, 0], dt=self.dt)
            self.trajectory_mgr.generate_trajectory(start_joints, target_joints, 100)

        elif mode == "manual":
            self.manual_thread = threading.Thread(target=self._manual_control_listener, daemon=True)
            self.manual_thread.start()

        elif mode == "follow":
            logger.info("目标跟随模式启动，目标点做三维运动")

        # 启动Viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 初始化Viewer
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -15
            viewer.cam.lookat = np.array([0.2, 0, 0.5])

            while viewer.is_running() and (time.time() - start_time) < duration and self.running:
                frame_start = time.time()
                current_joints = self.get_joint_angles()
                new_joints = current_joints

                # 模式处理
                if mode == "trajectory":
                    # 轨迹模式：使用PID控制
                    if self.current_traj_idx < len(self.trajectory_mgr.trajectory):
                        target_joints = self.trajectory_mgr.trajectory[self.current_traj_idx]
                        # 速度限制
                        new_joints = self.arm._clip_joint_speed(current_joints, target_joints, self.dt)
                        # PID控制输出
                        pid_output = self.pid.compute(current_joints, new_joints)
                        for i, act_id in enumerate(self.actuator_ids):
                            self.data.ctrl[act_id] = pid_output[i]
                        self.current_traj_idx += 1

                elif mode == "manual":
                    # 手动模式
                    try:
                        key = self.key_queue.get_nowait()
                        new_joints = self._process_manual_key(key, current_joints)
                        self.set_joint_angles(new_joints)
                    except queue.Empty:
                        pass

                elif mode == "follow":
                    # 跟随模式
                    self._update_moving_target()
                    new_joints = self.arm.inverse_kinematics(
                        self.target_vis.target_pos + [0, 0, 0],
                        initial_joints=current_joints,
                        dt=self.dt
                    )
                    # 速度限制
                    new_joints = self.arm._clip_joint_speed(current_joints, new_joints, self.dt)
                    self.set_joint_angles(new_joints)

                # 执行仿真步
                mujoco.mj_step(self.model, self.data)

                # 碰撞检测
                has_collision, _ = self.check_collision()

                # 渲染目标点
                self.target_vis.render(viewer)

                # 打印状态
                if frame_count % 5 == 0:
                    current_pose = self.arm.forward_kinematics(current_joints)
                    collision_info = "【碰撞】" if has_collision else ""
                    print(f"\r仿真时长：{time.time() - start_time:.1f}s | 末端位姿：{current_pose} {collision_info}",
                          end="")

                # 同步Viewer
                viewer.sync()

                # 帧率控制
                frame_elapsed = time.time() - frame_start
                time.sleep(max(0, self.dt - frame_elapsed))
                frame_count += 1

        logger.info(f"\n仿真结束：总帧数{frame_count}，平均帧率{frame_count / (time.time() - start_time):.1f}FPS")
        self.running = False


def main():
    """主函数"""
    try:
        sim = RoboticArmSim()

        # 模式选择
        print("\n========================")
        print("      增强版机械臂仿真系统      ")
        print("========================")
        print("新增功能：PID控制、目标可视化、轨迹保存/加载、速度限制")
        print("请选择仿真模式：")
        print("1 - 轨迹规划模式（PID精准控制）")
        print("2 - 手动控制模式（增强指令）")
        print("3 - 目标跟随模式（三维目标点）")
        print("========================")

        while True:
            mode_choice = input("输入模式编号（1/2/3）：").strip()
            if mode_choice in ["1", "2", "3"]:
                break
            print("无效输入！请输入 1、2 或 3")

        mode_map = {"1": "trajectory", "2": "manual", "3": "follow"}
        sim.run_simulation(mode=mode_map[mode_choice], duration=30.0)

    except Exception as e:
        logger.error(f"仿真失败：{e}")
        raise


if __name__ == "__main__":
    main()