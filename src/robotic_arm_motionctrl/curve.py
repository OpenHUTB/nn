import mujoco
import mujoco.viewer
import numpy as np
import time


class SCurveTrajectoryController:
    def __init__(self, model_path="arm6dof_final.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 6 关节
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.act_names = [f"motor_{j}" for j in self.joint_names]

        self.act_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            for a in self.act_names
        ]

        # PD 控制器
        self.kp = 60.0
        self.kd = 7.0

        # S 曲线参数
        self.total_time = 6.0      # 总运动时间
        self.Ta = 1.2             # 加速时间
        self.Td = 1.2             # 减速时间
        self.dt_ctrl = 0.01       # 控制周期

    def pd_control(self, idx, target):
        curr_q = self.data.joint(self.joint_names[idx]).qpos[0]
        curr_dq = self.data.joint(self.joint_names[idx]).qvel[0]
        err = target - curr_q
        tau = self.kp * err - self.kd * curr_dq
        self.data.ctrl[self.act_id[idx]] = np.clip(tau, -10, 10)

    def s_curve(self, t, q0, q1):
        t_total = self.total_time
        Ta = self.Ta
        Td = self.Td
        Tv = t_total - Ta - Td

        if Tv < 0:
            Tv = 0

        if t < 0:
            return q0
        elif t < Ta:
            tau = t / Ta
            s = 3*tau**2 - 2*tau**3
            return q0 + (q1 - q0) * s
        elif t < Ta + Tv:
            return q0 + (q1 - q0) * 1.0
        elif t < t_total:
            tau = (t - Ta - Tv) / Td
            s = 3*tau**2 - 2*tau**3
            return q1 - (q1 - q0) * s
        else:
            return q1

    def run(self):
        # 起点 & 终点
        q_start = np.array([0.0,  0.2,  -0.5,  0.0,  0.0,  0.0])
        q_end   = np.array([0.6,  1.4,  -0.9,  0.2,  0.3,  0.1])

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -25
            viewer.cam.lookat = [0.2, 0, 0.25]

            print("✅ S 曲线速度规划启动")
            print("▶  运动平滑无冲击，工业级轨迹")

            t = 0.0
            while viewer.is_running():
                q_target = self.s_curve(t, q_start, q_end)

                for i in range(6):
                    self.pd_control(i, q_target[i])

                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(self.dt_ctrl)
                t += self.dt_ctrl

                # 到达终点后反向运动
                if t >= self.total_time:
                    q_start, q_end = q_end, q_start
                    t = 0.0


if __name__ == "__main__":
    ctrl = SCurveTrajectoryController()
    ctrl.run()