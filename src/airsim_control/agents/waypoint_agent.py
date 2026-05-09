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
    NONE = "无降落"
    APPROACH = "减速悬停"      # 到达航点，悬停减速
    DESCEND = "慢速下降"       # 以 descend_speed 下降
    FINAL_DESCENT = "垂直降落" # 低速垂直落地
    TOUCHDOWN = "触地完成"     # 检测到触地


class LandingController:
    """定点降落控制器"""

    def __init__(self, waypoint: Waypoint, client):
        self.waypoint = waypoint
        self.client = client
        self.phase = LandingPhase.APPROACH
        self.phase_start_time = time.time()
        self.hover_duration = waypoint.hover_time   # 悬停时间
        self.descend_speed = waypoint.descend_speed  # 下降速度
        self.touchdown_altitude = 0.3               # 触地高度阈值(m)
        self.touchdown_vel_threshold = 0.15          # 触地速度阈值(m/s)
        self._hover_done = False
        self._descend_done = False

    def _get_altitude_and_vel(self):
        """获取当前高度(向上为正)和垂直速度(向上为正)"""
        state = self.client.get_state()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        return -pos.z_val, -vel.z_val

    def _set_vel(self, vx, vy, vz):
        # _move 里会对 vz 取反一次，_set_vel 直接透传，不重复取反
        self.client.move('velocity', vx, vy, vz)

    def step(self):
        """执行一步降落控制，返回是否完成触地"""
        altitude, vert_vel = self._get_altitude_and_vel()
        elapsed = time.time() - self.phase_start_time

        if self.phase == LandingPhase.APPROACH:
            # 减速悬停
            self._set_vel(0, 0, 0)
            if elapsed >= self.hover_duration:
                print(f"\n  悬停完成，开始下降 (速度={self.descend_speed}m/s)")
                self.phase = LandingPhase.DESCEND
                self.phase_start_time = time.time()

        elif self.phase == LandingPhase.DESCEND:
            # 慢速下降
            if altitude > self.touchdown_altitude * 3:
                self._set_vel(0, 0, self.descend_speed)
            else:
                self.phase = LandingPhase.FINAL_DESCENT
                self.phase_start_time = time.time()
                print(f"\n  进入垂直降落阶段")

        elif self.phase == LandingPhase.FINAL_DESCENT:
            # 垂直低速降落
            self._set_vel(0, 0, self.descend_speed * 0.5)
            if altitude <= self.touchdown_altitude:
                self.phase = LandingPhase.TOUCHDOWN
                self._set_vel(0, 0, 0)
                print(f"\n  ✓ 检测到触地！高度={altitude:.2f}m")

        elif self.phase == LandingPhase.TOUCHDOWN:
            self._set_vel(0, 0, 0)
            return True  # 触地完成

        # 触地检测：高度很低且速度很小时也认为触地
        if altitude < self.touchdown_altitude and abs(vert_vel) < self.touchdown_vel_threshold:
            if self.phase != LandingPhase.TOUCHDOWN:
                self.phase = LandingPhase.TOUCHDOWN
                self._set_vel(0, 0, 0)
                print(f"\n  ✓ 触地检测触发！")
                return True

        return False


class WaypointAgent:
    """航点跟踪Agent（PID控制 + 实时可视化）"""

    def __init__(self, client, waypoints=None, reach_threshold=2.0,
                 kp=1.0, ki=0.01, kd=0.5, max_vel=3.0, update_interval=0.5):
        self.client = client
        self.reach_threshold = reach_threshold

        # PID控制器
        self.pid = PIDController(kp=kp, ki=ki, kd=kd, max_vel=max_vel)

        # 航点规划器
        self.planner = WaypointPlanner(waypoints)

        # 3D可视化
        self.navigator = WaypointNavigator(
            waypoints=waypoints,
            update_interval=update_interval
        )

        self.current_pos = None
        self.start_time = None
        self.landing_ctrl = None  # 降落控制器

    def _get_position(self):
        state = self.client.get_state()
        pos = state.kinematics_estimated.position
        # AirSim NED坐标系: z向下为正，转换为向上为正的习惯
        return np.array([pos.x_val, pos.y_val, -pos.z_val])

    def _to_ned(self, pos):
        """将习惯坐标(上为正)转换为NED坐标(下为正)"""
        return np.array([pos[0], pos[1], -pos[2]])

    def _move(self, vx, vy, vz):
        # AirSim NED坐标系: z向下为正，需要对vz取反
        # 使用速度控制
        self.client.move('velocity', vx, vy, -vz)

        # 或者使用位置控制作为备选（更可靠）
        # 计算目标位置：当前位置 + 速度 * 时间步长
        # target_ned = self._to_ned(self.current_pos + np.array([vx, vy, vz]) * 0.5)
        # self.client.move('position', target_ned[0], target_ned[1], target_ned[2], 2.0)

    def _print_status(self, target, dist, wp_idx, total_wp, vel_cmd):
        print(f"\r[航点 {wp_idx+1}/{total_wp}] "
              f"目标: ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) | "
              f"距离: {dist:.2f}m | "
              f"位置: ({self.current_pos[0]:.1f}, {self.current_pos[1]:.1f}, {self.current_pos[2]:.1f}) | "
              f"速度命令: ({vel_cmd[0]:.2f}, {vel_cmd[1]:.2f}, {vel_cmd[2]:.2f})",
              end='', flush=True)

    def navigate_once(self, dt=0.1):
        """执行一步导航，返回是否完成"""
        self.current_pos = self._get_position()
        # 首次记录起始位置
        if self.navigator.start_pos is None:
            self.navigator.set_start(self.current_pos)
        target_wp = self.planner.get_current_target()

        if target_wp is None:
            print("\n无航点，等待添加...")
            time.sleep(1)
            return False

        # 如果正在执行降落状态机
        if self.landing_ctrl is not None:
            finished = self.landing_ctrl.step()
            alt, vvel = self.landing_ctrl._get_altitude_and_vel()
            print(f"\r[降落中] {self.landing_ctrl.phase.value} | "
                  f"高度: {alt:.2f}m | 垂直速度: {vvel:.2f}m/s", end='', flush=True)
            if finished:
                self.landing_ctrl = None
                return True
            return False

        dist = self.planner.distance_to_target(self.current_pos)
        wp_idx, total_wp = self.planner.get_progress()

        # PID计算速度
        vx, vy, vz = self.pid.compute(self.current_pos, target_wp, dt)
        vel_cmd = (vx, vy, vz)

        self._print_status(target_wp, dist, wp_idx-1, total_wp, vel_cmd)
        self.navigator.update(self.current_pos)

        # 检查是否到达当前航点
        if dist < self.reach_threshold:
            print(f"\n✓ 到达航点 {wp_idx}/{total_wp}: {target_wp}")
            self.pid.reset()
            self._move(0, 0, 0)

            # 如果当前航点需要降落
            wp_obj = self.planner.waypoints[self.planner.current_idx]
            if wp_obj.is_landing:
                print(f"  触发定点降落: 悬停{wp_obj.hover_time}s → 下降{wp_obj.descend_speed}m/s")
                self.landing_ctrl = LandingController(wp_obj, self.client)
                self.planner.advance()
                return False

            finished = self.planner.advance()
            if finished:
                print("\n=== 所有航点已完成! ===")
                return True
            return False

        # 执行移动
        if self.pid.prev_error is not None and np.linalg.norm(self.pid.prev_error) > 0.1:
            self._move(vx, vy, vz)
        return False

    def run(self, loop=False):
        """运行航点导航（阻塞式）"""
        self.planner.loop = loop
        self.start_time = time.time()

        print("=== 航点导航开始 ===")
        print(f"航点数量: {len(self.planner.waypoints)}")
        print(f"到达阈值: {self.reach_threshold}m")
        print("按 Ctrl+C 停止\n")

        try:
            while True:
                finished = self.navigate_once(dt=0.05)  # 提高控制频率
                if finished and not loop:
                    break
                time.sleep(0.05)  # 50ms控制周期
        except KeyboardInterrupt:
            print("\n\n用户中断导航")
            self._move(0, 0, 0)
        finally:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"\n总飞行时间: {elapsed:.1f}秒")
            self.navigator.show()

    def add_waypoint_interactive(self):
        """交互式添加航点"""
        print("\n=== 交互式添加航点 ===")
        print("输入格式: x y z [landing] [descend_speed] [hover_time]")
        print("  landing: 0=普通航点, 1=触发降落 (可选，默认0)")
        print("  descend_speed: 下降速度m/s (可选，默认0.5)")
        print("  hover_time: 悬停时间s (可选，默认2.0)")
        print("  示例: 5 5 10   → 普通航点")
        print("  示例: 0 0 0 1 0.3 3 → 定点降落(速度0.3m/s,悬停3s)")
        print("输入 'done' 完成")

        while True:
            inp = input("航点> ").strip()
            if inp.lower() == 'done':
                break
            parts = inp.split()
            try:
                x, y, z = map(float, parts[:3])
                is_landing = bool(int(parts[3])) if len(parts) > 3 else False
                descend_speed = float(parts[4]) if len(parts) > 4 else 0.5
                hover_time = float(parts[5]) if len(parts) > 5 else 2.0
                self.planner.add_waypoint(x, y, z, is_landing, descend_speed, hover_time)
                self.navigator.planner = self.planner
                self.navigator.trajectory = []
                tag = " [降落]" if is_landing else ""
                print(f"  已添加{tag}: ({x}, {y}, {z})")
            except (ValueError, IndexError):
                print("格式错误，请输入: x y z [landing] [descend_speed] [hover_time]")

        return self.planner.waypoints

    def reset(self):
        """重置导航状态"""
        self.planner.reset()
        self.pid.reset()
        self.navigator.trajectory = []
        self.landing_ctrl = None
