# -*- coding: utf-8 -*-
import mujoco
from mujoco import viewer
import time
import numpy as np
import os

from ant_config import AntConfig
from locomotion import AntLocomotion
from perception import AntPerception
from planner import AntPlanner


class AntEnvManager:
    """
    Ant 环境管理模块。
    集成配置、感知、规划、动态目标更新和运动控制，并运行仿真。
    """

    def __init__(self):
        self.config = AntConfig()
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"❌ 找不到模型文件: {self.config.model_path}")

        self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
        self.data = mujoco.MjData(self.model)

        self.locomotion = AntLocomotion(self.config, self.model)
        self.perception = AntPerception(self.config, self.model)
        self.planner = AntPlanner(self.config)

        self.root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.config.root_joint_name
        )
        self.root_qpos_adr = self.model.jnt_qposadr[self.root_id]
        self.root_qvel_adr = self.model.jnt_dofadr[self.root_id]

        # 目标小球 geom id。小球是视觉目标，不参与碰撞；运行时更新 model.geom_pos 来显示动态运动。
        self.ball_geom_ids = []
        for name in self.config.ball_geom_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"找不到目标小球 geom: {name}")
            self.ball_geom_ids.append(geom_id)

    def _compute_dynamic_ball_targets(self, t):
        """
        计算两个动态小球当前的目标坐标。
        两个小球整体位于 y 轴方向，并沿 x 轴做小范围正弦往复运动。
        """
        targets = []
        axis = int(self.config.ball_motion_axis)
        amp = float(self.config.ball_move_amplitude)
        omega = 2.0 * np.pi * float(self.config.ball_move_frequency)
        phases = list(self.config.ball_move_phases)

        for i, base_pos in enumerate(self.config.ball_base_targets):
            pos = np.asarray(base_pos, dtype=float).copy()
            phase = phases[i] if i < len(phases) else 0.0
            pos[axis] += amp * np.sin(omega * t + phase)
            targets.append(pos)

        return targets

    def _update_dynamic_targets(self):
        """
        同步更新：
        1. planner 中用于追踪判定的动态目标坐标；
        2. ant.xml 中两个可见小球的 geom 位置。
        """
        dynamic_targets = self._compute_dynamic_ball_targets(self.data.time)
        self.planner.update_targets(dynamic_targets)

        for geom_id, target_pos in zip(self.ball_geom_ids, dynamic_targets):
            visual_pos = np.asarray(target_pos, dtype=float).copy()
            visual_pos[2] = self.config.ball_visual_z
            self.model.geom_pos[geom_id] = visual_pos

        # 更新静态 world geom 的全局位置，保证 viewer 和 data.geom_xpos 使用最新坐标。
        mujoco.mj_forward(self.model, self.data)

    def _apply_soft_start(self):
        """
        前 0.5 秒轻微锁定根节点高度和姿态速度，避免开局摔落后长时间抽搐。
        """
        t = self.data.time
        st = self.config.stabilization_time
        if t < st:
            q = self.root_qpos_adr
            v = self.root_qvel_adr
            self.data.qpos[q + 2] = self.config.initial_z
            self.data.qvel[v:v + 6] *= 0.2

    def _apply_assisted_turning(self, steer):
        """
        将 Planner 的转向指令转换为根节点 z 轴角速度。

        原来的腿部开环差速在当前 Ant 模型中转向不明显，
        机器人会一直沿 y 方向直线走。这里加入温和的偏航辅助：
        - steer > 0：逆时针转向；
        - steer < 0：顺时针转向；
        - steer = 0：逐步阻尼偏航角速度。
        """
        if not getattr(self.config, "enable_yaw_assist", False):
            return

        v = self.root_qvel_adr
        target_yaw_rate = float(
            np.clip(
                self.config.yaw_assist_gain * steer,
                -self.config.max_yaw_rate,
                self.config.max_yaw_rate,
            )
        )
        alpha = float(np.clip(getattr(self.config, "yaw_rate_smoothing", 0.35), 0.0, 1.0))

        # freejoint 的 6 个速度自由度通常为 3 个平移速度 + 3 个角速度，
        # 最后一维对应 z 轴偏航角速度。
        yaw_vel_adr = v + 5
        self.data.qvel[yaw_vel_adr] = (1.0 - alpha) * self.data.qvel[yaw_vel_adr] + alpha * target_yaw_rate

    def run(self):
        print("🚀 Ant y方向动态小球追踪 + 绕墙仿真启动...")
        print("📍 目标布置：近球 y=3，远球 y=6，两个小球沿 x 轴小范围往复移动。")
        print("执行顺序：动态近球 → 绕过中间墙体 → 动态远球 → 停止")

        with viewer.launch_passive(self.model, self.data) as v:
            mujoco.mj_resetData(self.model, self.data)
            self._update_dynamic_targets()

            while v.is_running():
                step_start = time.time()

                # 先更新小球位置，再感知和规划，保证机器人追踪的是当前动态目标。
                self._update_dynamic_targets()
                current_target = self.planner.get_current_target()
                obs_info = self.perception.sense_environment(self.data, current_target)

                speed, steer = self.planner.get_next_action(obs_info)

                dt = self.model.opt.timestep
                ctrl = self.locomotion.compute_control(self.data, dt, speed, steer)
                self.data.ctrl[:] = ctrl

                self._apply_soft_start()
                self._apply_assisted_turning(steer)

                mujoco.mj_step(self.model, self.data)
                v.sync()

                elapsed = time.time() - step_start
                if elapsed < self.model.opt.timestep:
                    time.sleep(self.model.opt.timestep - elapsed)


if __name__ == "__main__":
    env = AntEnvManager()
    try:
        env.run()
    except KeyboardInterrupt:
        print("\n🛑 用户停止仿真")
