import numpy as np
import time
import sys
from enum import Enum

# 解决Windows中文显示
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from planners.waypoint_planner import WaypointPlanner, Waypoint, WaypointNavigator
from controllers.pid_controller import PIDController


class LandingPhase(Enum):
    """降落阶段状态机"""
    APPROACH = "减速悬停"
    DESCEND = "慢速下降"
    FINAL_DESCENT = "垂直降落"
    TOUCHDOWN = "触地完成"


class LandingController:
    """定点降落控制器（仅负责降落过程）"""

    def __init__(self, waypoint: Waypoint, client):
        self.waypoint = waypoint
        self.client = client
        self.phase = LandingPhase.APPROACH
        self.phase_start_time = time.time()
        self.hover_duration = waypoint.hover_time
        self.descend_speed = waypoint.descend_speed
        self.touchdown_altitude = 0.3
        self.touchdown_vel_threshold = 0.15

    def _get_altitude_and_vel(self):
        state = self.client.get_state()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        return -pos.z_val, -vel.z_val

    def _set_vel(self, vx, vy, vz):
        self.client.client.moveByVelocityAsync(vx, vy, -vz, 0.1)

    def step(self):
        altitude, vert_vel = self._get_altitude_and_vel()
        elapsed = time.time() - self.phase_start_time

        if self.phase == LandingPhase.APPROACH:
            self._set_vel(0, 0, 0)
            if elapsed >= self.hover_duration:
                print(f"\n  悬停完成，开始下降 (速度={self.descend_speed}m/s)")
                self.phase = LandingPhase.DESCEND
                self.phase_start_time = time.time()

        elif self.phase == LandingPhase.DESCEND:
            if altitude <= self.touchdown_altitude * 3:
                self.phase = LandingPhase.FINAL_DESCENT
                self.phase_start_time = time.time()
                print(f"\n  进入垂直降落阶段")
            else:
                self._set_vel(0, 0, self.descend_speed)

        elif self.phase == LandingPhase.FINAL_DESCENT:
            self._set_vel(0, 0, self.descend_speed * 0.5)
            if altitude <= self.touchdown_altitude:
                self.phase = LandingPhase.TOUCHDOWN
                self._set_vel(0, 0, 0)
                print(f"\n  ✓ 检测到触地！高度={altitude:.2f}m")
                return 'done'

        elif self.phase == LandingPhase.TOUCHDOWN:
            self._set_vel(0, 0, 0)
            return 'done'

        if altitude < self.touchdown_altitude and abs(vert_vel) < self.touchdown_vel_threshold:
            if self.phase != LandingPhase.TOUCHDOWN:
                self.phase = LandingPhase.TOUCHDOWN
                self._set_vel(0, 0, 0)
                print(f"\n  ✓ 触地检测触发！")
                return 'done'

        return 'continue'


class WaypointAgent:
    """航点跟踪Agent（PID控制 + 实时可视化）"""

    def __init__(self, client, waypoints=None, reach_threshold=0.5,
                 kp=2.0, ki=0.05, kd=1.0, max_vel=2.0, update_interval=0.5):
        self.client = client
        self.reach_threshold = reach_threshold
        self.pid = PIDController(kp=kp, ki=ki, kd=kd, max_vel=max_vel)
        self.planner = WaypointPlanner(waypoints)
        self.navigator = WaypointNavigator(waypoints=waypoints, update_interval=update_interval)
        self.current_pos = None
        self.start_time = None
        self.landing_ctrl = None
        self.home_target = None    # 返航目标（不为None表示正在返航）
        self._last_arrived_wp = -1  # 防止同一航点重复触发
        self._last_printed_status = None  # 用于避免重复输出位置信息
        self._status_line = ""  # 用于存储当前状态行

    def _get_position(self):
        state = self.client.get_state()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, -pos.z_val])

    def _get_altitude_and_vel(self):
        state = self.client.get_state()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        return -pos.z_val, -vel.z_val

    def _move(self, vx, vy, vz):
        self.client.client.moveByVelocityAsync(vx, vy, -vz, 0.1)

    def _print_status(self, target, dist, wp_idx, total_wp, vel_cmd):
        last = getattr(self, '_last_printed_status', None)
        if last is None or abs(float(last) - dist) > 0.5:
            label = f"航点 {wp_idx+1}/{total_wp}"
            self._status_line = f"[{label}] 目标: ({target[0]:.1f},{target[1]:.1f},{target[2]:.1f}) | 距离: {dist:.2f}m | 位置: ({self.current_pos[0]:.1f},{self.current_pos[1]:.1f},{self.current_pos[2]:.1f}) | 速度: ({vel_cmd[0]:.2f},{vel_cmd[1]:.2f},{vel_cmd[2]:.2f})"
            print(f"\r{self._status_line}", end='', flush=True)
            self._last_printed_status = f"{dist:.2f}"

    def navigate_once(self, dt=0.1):
        self.current_pos = self._get_position()
        if self.navigator.start_pos is None:
            self.navigator.set_start(self.current_pos)

        # 正在执行降落状态机
        if self.landing_ctrl is not None:
            result = self.landing_ctrl.step()
            alt, vvel = self._get_altitude_and_vel()
            phase = self.landing_ctrl.phase.value
            print(f"\r[降落中] {phase} | 高度: {alt:.2f}m | 垂直速度: {vvel:.2f}m/s", end='', flush=True)
            if result == 'done':
                self.landing_ctrl = None
                if self.planner.current_idx < len(self.planner.waypoints):
                    self.planner.advance()
                if self.planner.current_idx >= len(self.planner.waypoints):
                    print("\n=== 所有航点已完成! ===")
                    return True
                return False
            return False

        # 航点已全部飞完（由 run() 处理返航，navigate_once 只发送飞回家的命令）
        target_wp = self.planner.get_current_target()
        if target_wp is None:
            return False  # 交给 run() 的返航逻辑处理

        # 防止到达同一航点重复触发
        wp_idx, total_wp = self.planner.get_progress()
        if wp_idx - 1 == self._last_arrived_wp:
            # 还在同一航点范围内飞行，忽略
            pass
        else:
            self._last_arrived_wp = -1  # 重置

        dist = self.planner.distance_to_target(self.current_pos)
        wp_obj = self.planner.waypoints[self.planner.current_idx]
        self.pid.max_vel = wp_obj.fly_speed

        vx, vy, vz = self.pid.compute(self.current_pos, target_wp, dt)
        vel_cmd = (vx, vy, vz)
        self._print_status(target_wp, dist, wp_idx - 1, total_wp, vel_cmd)
        self.navigator.update(self.current_pos)

        # 只有距离足够近才触发到达逻辑
        if dist > self.reach_threshold:
            self._move(vx, vy, vz)
            self._last_arrived_wp = self.planner.current_idx
            return False

        # 以下为到达后的处理
        print(f"\n✓ 到达航点 {wp_idx}/{total_wp}: {target_wp}")
        self.pid.reset()
        self._move(0, 0, 0)

        wp_obj = self.planner.waypoints[self.planner.current_idx]

        # 返航航点（只返航，不降落）
        if wp_obj.is_return_home and not wp_obj.is_landing:
            home = self.navigator.start_pos
            print(f"  → 返航，飞回初始位置 ({home[0]:.1f},{home[1]:.1f},{home[2]:.1f})")
            self._fly_home(home)
            self.planner.advance()
            return True  # 通知 run() 进入返航阶段

        # 定点降落（可带返航）
        if wp_obj.is_landing:
            print(f"  触发定点降落: 悬停{wp_obj.hover_time}s → 下降{wp_obj.descend_speed}m/s")
            self.landing_ctrl = LandingController(wp_obj, self.client)
            self.planner.advance()
            if wp_obj.is_return_home:
                self._fly_home(self.navigator.start_pos)
                return True  # 通知 run() 进入返航阶段
            return False

        finished = self.planner.advance()
        if finished:
            print("\n=== 所有航点已完成! ===")
            return True
        return False

    def _fly_home(self, home_pos):
        """让无人机飞回初始位置（非阻塞，启用API后立即返回）"""
        self.client.client.enableApiControl(True)
        self.client.client.armDisarm(True)
        self.home_target = home_pos
        self.home_arm_time = time.time()
        print(f"  → 飞回初始位置命令已发送")

    def run(self, loop=False):
        self.planner.loop = loop
        self.start_time = time.time()
        self.home_target = None
        self.home_arm_time = None

        print("=== 航点导航开始 ===")
        print(f"航点数量: {len(self.planner.waypoints)}")
        print(f"到达阈值: {self.reach_threshold}m")
        print("按 Ctrl+C 停止\n")

        try:
            # --- 阶段一：航点导航 ---
            while True:
                finished = self.navigate_once(dt=0.05)
                if finished and not loop:
                    break
                if finished and self.home_target is not None:
                    break  # 航点完成，进入返航阶段
                time.sleep(0.05)

            # --- 阶段二：返航控制 ---
            if self.home_target is not None:
                print(f"\n  → 返航中，目标: ({self.home_target[0]:.1f},{self.home_target[1]:.1f},{self.home_target[2]:.1f})")
                while True:
                    self.current_pos = self._get_position()
                    dist = np.linalg.norm(self.current_pos - self.home_target)
                    self.pid.max_vel = 1.5
                    vx, vy, vz = self.pid.compute(self.current_pos, self.home_target, dt=0.05)
                    self._move(vx, vy, vz)
                    print(f"\r  [飞回家] 距离: {dist:.2f}m | 位置: ({self.current_pos[0]:.1f},{self.current_pos[1]:.1f},{self.current_pos[2]:.1f})", end='', flush=True)
                    if dist < 2.0:
                        self._move(0, 0, 0)
                        print(f"\n  ✓ 已返回初始点! 距离: {dist:.2f}m")
                        self.pid.reset()
                        print("\n=== 所有航点已完成! ===")
                        break
                    time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n\n用户中断导航")
            self._move(0, 0, 0)
        finally:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"\n总飞行时间: {elapsed:.1f}秒")
            self.navigator.show()

    def reset(self):
        self.planner.reset()
        self.pid.reset()
        self.navigator.trajectory = []
        self.landing_ctrl = None
        self.home_target = None
        self.home_arm_time = None
        self._last_arrived_wp = -1

    def add_waypoint_interactive(self, show_help=True):
        """交互式添加航点"""
        if show_help:
            print("\n=== 交互式添加航点 ===")
            print("输入格式: x y z [landing] [descend_speed] [hover_time] [return_home] [fly_speed]")
            print("  landing: 0=普通航点, 1=触发降落 (可选，默认0)")
            print("  descend_speed: 下降速度m/s (可选，默认1.0)")
            print("  hover_time: 悬停时间s (可选，默认3.0)")
            print("  return_home: 0=降落后停在原地, 1=降落后起飞返回初始点 (可选，默认0)")
            print("  fly_speed: 飞向该航点的速度m/s (可选，默认1.0)")
            print("  示例: 5 5 10   → 普通航点")
            print("  示例: 0 0 0 1 0.3 3 → 定点降落(速度0.3m/s,悬停3s)")
            print("  示例: 0 0 0 1 0.3 3 1 → 定点降落并返回初始点")
            print("  示例: 5 5 10 0 0 0 0 2 → 普通航点，飞行速度2m/s")
            print("输入 'done' 完成")

        while True:
            inp = input("航点> ").strip()
            if inp.lower() == 'done':
                break
            parts = inp.split()
            try:
                x, y, z = map(float, parts[:3])
                is_landing = bool(int(parts[3])) if len(parts) > 3 else False
                descend_speed = float(parts[4]) if len(parts) > 4 else 1.0
                hover_time = float(parts[5]) if len(parts) > 5 else 3.0
                is_return_home = bool(int(parts[6])) if len(parts) > 6 else False
                fly_speed = float(parts[7]) if len(parts) > 7 else 1.0
                self.planner.add_waypoint(x, y, z, is_landing, descend_speed, hover_time, is_return_home, fly_speed)
                self.navigator.planner = self.planner
                self.navigator.trajectory = []
                tag = " [降落]" if is_landing else ""
                ret_tag = " [返航]" if is_return_home else ""
                print(f"  已添加{tag}{ret_tag}: ({x}, {y}, {z}) 飞行速度={fly_speed}m/s")
            except (ValueError, IndexError):
                print("格式错误，请输入: x y z [landing] [descend_speed] [hover_time] [return_home] [fly_speed]")
