# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np


class AntConfig:
    """
    Ant 模型配置。

    当前任务设置：
    1. 机器人实际运动方向主要沿 y 轴，因此两个目标小球和墙体都布置在 y 轴方向；
    2. 两个小球沿 x 轴小范围往复移动，机器人需要追踪动态目标；
    3. 墙体位于近球和远球之间，机器人需要绕墙后继续追踪远球。
    """

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.model_path = str(self.base_dir / "ant.xml")

        # 对 motor ctrlrange=[-1,1]、gear=30 的模型，不宜用过大的 PD 增益。
        self.balance_kp = 3.8
        self.balance_kd = 0.9

        # ant.xml 中 torso 初始高度是 0.75，软启动时保持一致。
        self.initial_z = 0.75
        self.stabilization_time = 0.5

        self.root_joint_name = "freejoint"
        self.qpos_start_idx = 7
        self.qvel_start_idx = 6

        # 开环步态参数
        self.gait_frequency = 20#9.5        # 相位速度：提高步频，让机器人走得更快
        self.hip_swing_amp = 0.32        # 髋关节前后摆动幅度，单位 rad
        self.ankle_lift_amp = 0.25       # 踝关节抬腿幅度，单位 rad
        self.joint_limit_margin = 0.04   # 避免目标角贴近关节限位
        self.turn_gait_gain = 0.55       # 转向差速强度，增大后转向更明显
        self.hip_turn_bias_amp = 0.14    # 转向时给左右腿增加少量髋关节偏置，增强偏航力矩
        self.max_steer_angle = 1.2       # Planner 中认为的大角度阈值
        self.max_forward_speed = 1.45    # 允许高层速度指令超过 1.0，提高巡航速度
        self.stop_speed_epsilon = 0.03   # 速度接近 0 时进入站立保持状态

        # 当前开环步态的自然前进方向更接近机体局部 +Y，而不是局部 +X。
        # 因此感知中的朝向误差应使用 yaw + pi/2 作为“机器人前进方向”。
        self.robot_forward_yaw_offset = np.pi / 2.0

        # 偏航辅助控制：让 Planner 输出的 steer 真正作用到自由关节的 z 轴角速度。
        # 该项用于解决纯开环腿部差速难以明显改变朝向的问题。
        self.enable_yaw_assist = True
        self.yaw_assist_gain = 1.8
        self.max_yaw_rate = 1.6
        self.yaw_rate_smoothing = 0.35

        self.max_ctrl_amplitude = 0.45

        # ========= 动态小球追踪参数 =========
        # 小球基础位置：沿 y 轴放置，近球在 y=3，远球在 y=6。
        # x 坐标会在仿真过程中动态更新，因此这里是“中心位置”。
        self.ball_base_targets = [
            (0.0, 5.0, 0.0),   # 距离较近的小球中心轨迹
            (0.0, 15.0, 0.0),   # 距离较远的小球中心轨迹
        ]
        # 兼容 planner.py 中原有变量名，初始化时先使用中心位置；运行时由 env_manager.py 动态刷新。
        self.ball_targets = list(self.ball_base_targets)
        self.ball_target_names = ["near_ball", "far_ball"]
        self.ball_geom_names = ["target_ball_near", "target_ball_far"]

        # 小球沿 x 轴往复运动：x = base_x + A * sin(2πft + phase)
        self.ball_motion_axis = 0              # 0 表示 x 轴
        self.ball_forward_axis = 1             # 1 表示任务主方向为 y 轴
        self.ball_visual_z = 0.18              # ant.xml 中小球的显示高度
        self.ball_move_amplitude = 8       # 小范围移动幅度，单位 m
        self.ball_move_frequency = 0.01      # 往复移动频率，单位 Hz
        self.ball_move_phases = [0.0, np.pi / 2.0]

        # 用平面距离判断是否接触小球。动态目标下略放宽阈值，避免小球横向移动导致难以判定接触。
        self.target_threshold = 0.62

        # ========= 运动与避障参数 =========
        self.forward_force = 1.35
        self.turn_force = 1.05

        # 墙体参数：墙位于 y=4.5，横跨 x 方向，阻挡机器人沿 y 方向前进。
        # 坐标和尺寸需要与 ant.xml 中 obstacle_1 的 pos/size 保持一致。
        self.wall_center = (0.0, 10, 0.45)
        self.wall_half_size = (1.00, 0.10, 0.45)

        # 绕墙方向。任务主方向是 +Y，因此 1 表示从 +X 侧绕墙，-1 表示从 -X 侧绕墙。
        self.wall_bypass_side = 1.0
        self.wall_bypass_clearance = 0.70  # 绕墙时希望离墙侧边至少保留的距离
        self.wall_side_force = 1.45        # 墙体阻挡时侧向绕行引导强度
        self.wall_detect_margin = 1.45     # 距离墙体主方向多远开始明显绕行
        self.wall_exit_margin = 0.35       # 越过墙体后逐步取消绕行动作

        # obstacle_margin 越大，机器人越早感知障碍；safe_distance 控制减速距离。
        self.obstacle_margin = 1.80
        self.safe_distance = 1.15

    def __repr__(self):
        return f"<AntConfig: {self.model_path} | Ready>"


default_ant_config = AntConfig()
