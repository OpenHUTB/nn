import numpy as np
import time
import sys

# 解决Windows中文显示
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from planners.waypoint_planner import WaypointPlanner, WaypointNavigator
from controllers.pid_controller import PIDController


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
        target = self.planner.get_current_target()

        if target is None:
            print("\n无航点，等待添加...")
            time.sleep(1)
            return False

        dist = self.planner.distance_to_target(self.current_pos)
        wp_idx, total_wp = self.planner.get_progress()

        # PID计算速度
        vx, vy, vz = self.pid.compute(self.current_pos, target, dt)
        vel_cmd = (vx, vy, vz)

        self._print_status(target, dist, wp_idx-1, total_wp, vel_cmd)
        self.navigator.update(self.current_pos)

        # 检查是否到达当前航点
        if dist < self.reach_threshold:
            print(f"\n✓ 到达航点 {wp_idx}/{total_wp}: {target}")
            self.pid.reset()
            self._move(0, 0, 0)  # 停止
            finished = self.planner.advance()
            if finished:
                print("\n=== 所有航点已完成! ===")
                return True
            return False

        # 执行移动（添加调试）
        if self.pid.prev_error is not None and np.linalg.norm(self.pid.prev_error) > 0.1:
            # 只有当误差足够大时才发送移动命令
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
        print("输入格式: x y z (空格分隔)，输入 'done' 完成")
        print("注意: z 为高度，向上为正（例如: 5 5 10 表示飞到高度10米）")

        while True:
            inp = input("航点坐标> ").strip()
            if inp.lower() == 'done':
                break
            try:
                x, y, z = map(float, inp.split())
                self.planner.add_waypoint(x, y, z)
                self.navigator.planner = self.planner
                self.navigator.trajectory = []
                print(f"已添加航点: ({x}, {y}, {z})")
                print(f"当前航点列表: {len(self.planner.waypoints)} 个")
            except ValueError:
                print("格式错误，请输入: x y z")

        return self.planner.waypoints

    def reset(self):
        """重置导航状态"""
        self.planner.reset()
        self.pid.reset()
        self.navigator.trajectory = []
