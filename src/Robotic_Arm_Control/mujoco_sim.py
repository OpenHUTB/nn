import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from core.kinematics import RoboticArmKinematics


class MuJoCoArmSim:
    def __init__(self, model_path="model/six_axis_arm.xml"):
        # 修复：处理相对路径，确保能找到XML文件
        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MuJoCo模型文件不存在：{self.model_path}")

        # 加载MuJoCo模型（兼容不同版本）
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
        except Exception as e:
            raise RuntimeError(f"加载MuJoCo模型失败：{e}")

        self.data = mujoco.MjData(self.model)

        # 初始化运动学模块
        self.kinematics = RoboticArmKinematics()

        # 关节名称映射（MuJoCo关节名 → 索引）
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.joint_ids = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"找不到关节：{name}")
            self.joint_ids.append(jid)

        self.actuator_ids = []
        for i in range(6):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act{i + 1}")
            if aid == -1:
                raise ValueError(f"找不到执行器：act{i + 1}")
            self.actuator_ids.append(aid)

    def set_joint_angles(self, joint_angles):
        """设置机械臂关节角（单位：度）"""
        if len(joint_angles) != 6:
            raise ValueError("需输入6个关节角")
        # 转换为弧度（MuJoCo默认弧度）
        joint_radians = np.radians(joint_angles)
        # 设置执行器目标位置
        for i, act_id in enumerate(self.actuator_ids):
            self.data.actuator_target[act_id] = joint_radians[i]

    def get_joint_angles(self):
        """获取当前关节角（单位：度）"""
        joint_radians = []
        for jid in self.joint_ids:
            # 兼容不同MuJoCo版本的qpos访问方式
            try:
                joint_radians.append(self.data.joint(jid).qpos[0])
            except:
                joint_radians.append(self.data.qpos[jid])
        return np.degrees(joint_radians).round(2).tolist()

    def run_simulation(self, target_pose=None, duration=10.0):
        """运行仿真：可指定目标末端位姿，或手动控制"""
        # 初始化Viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 修复：兼容MuJoCo viewer的相机设置方式
            viewer.cam.distance = 2.0  # 相机距离
            viewer.cam.azimuth = 45  # 相机方位角
            viewer.cam.elevation = -15  # 相机仰角
            # 移除lookat，改用内置的cam.lookat（viewer层面设置，非XML）
            viewer.cam.lookat = np.array([0.2, 0, 0.5])

            start_time = time.time()
            # 如果指定目标位姿，先解算关节角
            if target_pose is not None:
                print(f"\n目标末端位姿：{target_pose} (m/度)")
                try:
                    target_joints = self.kinematics.inverse_kinematics(target_pose)
                    print(f"解算得到关节角：{target_joints} (度)")
                    self.set_joint_angles(target_joints)
                except ValueError as e:
                    print(f"逆解失败：{e}")

            # 仿真循环
            print("\n开始仿真（按窗口关闭按钮退出）...")
            while viewer.is_running() and (time.time() - start_time) < duration:
                step_start = time.time()

                # 一步仿真
                mujoco.mj_step(self.model, self.data)

                # 实时打印关节角和末端位姿（优化显示格式）
                current_joints = self.get_joint_angles()
                try:
                    current_pose = self.kinematics.forward_kinematics(current_joints)
                    print(f"\r关节角：{current_joints} | 末端位姿：{current_pose}", end="")
                except:
                    print(f"\r关节角：{current_joints} | 末端位姿：计算失败", end="")

                # 同步Viewer
                viewer.sync()

                # 控制仿真帧率（60Hz）
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    try:
        # 初始化仿真
        sim = MuJoCoArmSim()
        # 示例：指定目标末端位姿（x=0.2m, y=0.1m, z=0.4m，姿态0）
        target_pose = [0.2, 0.1, 0.4, 0, 0, 0]
        # 运行仿真（持续10秒）
        sim.run_simulation(target_pose=target_pose, duration=10.0)
    except Exception as e:
        print(f"\n运行出错：{e}")
        # 打印详细路径，帮助排查
        print(f"当前工作目录：{os.getcwd()}")
        print(f"模型文件期望路径：{os.path.abspath('model/six_axis_arm.xml')}")