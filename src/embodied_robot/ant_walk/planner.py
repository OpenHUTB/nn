# -*- coding: utf-8 -*-
import numpy as np


class AntPlanner:
    """
    Ant 决策与路径规划模块。

    当前任务：
    1. 两个小球位于机器人前方 y 轴方向，并沿 x 轴小范围往复移动；
    2. 机器人先追踪近处动态小球；
    3. 接触近球后，切换为追踪远处动态小球；
    4. 追踪远球时遇到墙体则转向绕行；
    5. 接触远球后，任务完成，机器人停止行动。
    """

    def __init__(self, config):
        self.config = config

        # 初始化为小球中心位置；运行时 env_manager.py 会持续更新为动态位置。
        self.patrol_targets = [np.asarray(p, dtype=float) for p in self.config.ball_targets]
        self.target_names = list(getattr(self.config, "ball_target_names", ["near_ball", "far_ball"]))

        self.current_target_idx = 0
        self.mission_complete = False

        self.base_speed = float(getattr(self.config, "forward_force", 1.35))
        self.is_stuck = False
        self.last_pos = np.zeros(3)
        self.stuck_counter = 0

    @staticmethod
    def _normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def update_targets(self, dynamic_targets):
        """
        由 env_manager.py 在每个仿真步刷新两个动态小球的位置。
        """
        self.patrol_targets = [np.asarray(p, dtype=float) for p in dynamic_targets]

    def _avoid_vector_to_steer(self, avoid_vec, robot_forward_yaw):
        """
        将全局避障向量转换为机器人前进方向下的转向误差。

        当前 Ant 步态的自然前进方向是机体局部 +Y，
        因此不能直接用 robot_yaw，而要使用 perception.py 返回的
        robot_forward_yaw = robot_yaw + robot_forward_yaw_offset。
        """
        norm = np.linalg.norm(avoid_vec)
        if norm < 1e-6:
            return 0.0

        avoid_angle = np.arctan2(avoid_vec[1], avoid_vec[0])
        avoid_error = self._normalize_angle(avoid_angle - robot_forward_yaw)
        strength = np.clip(norm, 0.0, 1.0)
        return float(avoid_error * strength)

    def get_next_action(self, perception_data):
        """
        根据当前目标距离、角度和避障向量返回速度、转向。
        到达第二个小球后，直接返回 0, 0，使机器人进入停止状态。
        """
        if self.mission_complete:
            return 0.0, 0.0

        dist_to_target = perception_data["dist_target"]
        angle_to_target = perception_data["angle_target"]
        avoid_vec = perception_data["avoid_vector"]
        danger_level = perception_data["danger_level"]
        robot_forward_yaw = perception_data.get("robot_forward_yaw", 0.0)

        # 判断是否“接触”当前动态小球。这里采用平面距离阈值判断。
        if dist_to_target < self.config.target_threshold:
            reached_name = self.target_names[self.current_target_idx]
            print(f"🎯 已接触目标: {reached_name}")

            if self.current_target_idx < len(self.patrol_targets) - 1:
                self.current_target_idx += 1
                next_name = self.target_names[self.current_target_idx]
                print(f"➡️  切换目标，开始追踪: {next_name}")
            else:
                self.mission_complete = True
                print("✅ 已接触远处动态小球，任务完成，机器人停止行动。")
                return 0.0, 0.0

        # 正常情况下追踪小球；靠近墙体时降低目标吸引权重，让避障优先。
        # danger_level 越高，越优先避障；但仍保留目标吸引，避免绕墙后跑偏。
        target_weight = 1.0 if danger_level == 0 else 0.45
        steer = angle_to_target * target_weight

        # 避障向量使用机器人当前 yaw 转为转向指令，适配 y 方向追踪和 x 方向绕墙。
        avoid_steer = self._avoid_vector_to_steer(avoid_vec, robot_forward_yaw)
        steer += avoid_steer * self.config.turn_force

        steer = float(np.clip(steer, -self.config.max_steer_angle, self.config.max_steer_angle))

        speed = self.base_speed

        if danger_level > 0:
            speed *= 0.58

        # 大角度绕墙或追踪横向移动小球时减速，但保留前进速度。
        if abs(steer) > 1.0:
            speed *= 0.55

        return speed, steer

    def check_stuck_status(self, current_pos):
        dist_moved = np.linalg.norm(current_pos[:2] - self.last_pos[:2])
        if dist_moved < 0.001:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.last_pos = current_pos.copy()
        return self.stuck_counter > 100

    def get_current_target(self):
        """
        返回当前追踪目标。
        任务完成后仍返回远球坐标，便于感知模块继续正常运行。
        """
        return self.patrol_targets[self.current_target_idx]

    def get_status_text(self):
        if self.mission_complete:
            return "任务完成：机器人已停止"
        return f"当前追踪目标：{self.target_names[self.current_target_idx]}"
