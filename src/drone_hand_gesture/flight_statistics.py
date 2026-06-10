"""
飞行统计数据收集与分析模块
"""
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional


class FlightStatistics:
    """飞行统计数据跟踪器"""

    def __init__(self):
        # 时间统计
        self.start_time: float = time.time()
        self.flight_start_time: Optional[float] = None  # 首次起飞时间
        self.total_flight_time: float = 0.0  # 累计飞行时长（秒）
        self.session_start: float = time.time()  # 本次会话开始时间

        # 位置统计
        self.position_history: List[np.ndarray] = []  # 位置历史
        self.total_distance: float = 0.0  # 累计飞行距离
        self.max_altitude: float = 0.0  # 最高高度
        self.max_distance_from_home: float = 0.0  # 离起飞点最远距离
        self.home_position: Optional[np.ndarray] = None  # 起飞点

        # 速度统计
        self.max_speed: float = 0.0  # 最大速度
        self.avg_speed: float = 0.0  # 平均速度（仅飞行时）
        self.speed_samples: List[float] = []  # 速度采样

        # 手势统计
        self.gesture_counts: Dict[str, int] = defaultdict(int)  # 每种手势的触发次数
        self.gesture_total_confidence: Dict[str, float] = defaultdict(float)  # 手势总置信度
        self.last_gesture: Optional[str] = None  # 上一次手势
        self.last_gesture_start: float = 0.0  # 手势开始时间

        # 命令统计
        self.command_counts: Dict[str, int] = defaultdict(int)  # 每种命令的执行次数

        # 起飞/降落统计
        self.takeoff_count: int = 0  # 起飞次数
        self.landing_count: int = 0  # 降落次数
        self.was_armed: bool = False  # 上一次解锁状态

        # 电池统计
        self.initial_battery: float = 100.0  # 初始电量
        self.min_battery: float = 100.0  # 最低电量
        self.battery_drain_total: float = 0.0  # 总耗电量
        self.battery_samples: List[float] = []  # 电量采样

        # 模式统计
        self.mode_time: Dict[str, float] = defaultdict(float)  # 每种模式的时间
        self.last_mode: Optional[str] = None  # 上一次模式
        self.last_mode_time: float = 0.0  # 模式切换时间

    def update(self, drone_state: dict, current_gesture: Optional[str],
               current_command: Optional[str], dt: float):
        """
        更新飞行统计数据

        Args:
            drone_state: 无人机状态字典
            current_gesture: 当前检测到的手势
            current_command: 当前执行的命令
            dt: 时间增量（秒）
        """
        now = time.time()
        position = np.array(drone_state['position'])
        battery = drone_state['battery']
        armed = drone_state['armed']
        mode = drone_state['mode']

        # ========== 飞行时间统计 ==========
        if armed:
            if self.flight_start_time is None:
                self.flight_start_time = now
            self.total_flight_time += dt

        # ========== 位置统计 ==========
        if armed and len(self.position_history) > 0:
            dist = np.linalg.norm(position - self.position_history[-1])
            self.total_distance += dist

        self.position_history.append(position.copy())

        altitude = abs(position[1])
        if altitude > self.max_altitude:
            self.max_altitude = altitude

        # 离起飞点最远距离
        if self.home_position is not None:
            dist_home = np.linalg.norm(position - self.home_position)
            if dist_home > self.max_distance_from_home:
                self.max_distance_from_home = dist_home

        # ========== 速度统计 ==========
        if armed and len(self.position_history) >= 2:
            vel = drone_state['velocity']
            speed = np.linalg.norm(vel)
            self.speed_samples.append(speed)
            if speed > self.max_speed:
                self.max_speed = speed

        # ========== 手势统计 ==========
        if current_gesture and current_gesture not in ["no_hand", "hand_detected", "none", "unknown"]:
            self.gesture_counts[current_gesture] += 1

        # ========== 命令统计 ==========
        if current_command and current_command != "none":
            self.command_counts[current_command] += 1

        # ========== 起飞/降落统计 ==========
        if armed and not self.was_armed:
            # 刚刚起飞
            self.takeoff_count += 1
            self.home_position = position.copy()  # 记录起飞点
        elif not armed and self.was_armed:
            # 刚刚降落
            self.landing_count += 1
        self.was_armed = armed

        # ========== 电池统计 ==========
        if battery < self.min_battery:
            self.min_battery = battery
        self.battery_samples.append(battery)

        # ========== 模式统计 ==========
        if mode and mode != self.last_mode:
            if self.last_mode and self.last_mode_time > 0:
                self.mode_time[self.last_mode] += now - self.last_mode_time
            self.last_mode = mode
            self.last_mode_time = now

    def finalize(self):
        """统计收尾（在会话结束时调用）"""
        now = time.time()
        if self.last_mode and self.last_mode_time > 0:
            self.mode_time[self.last_mode] += now - self.last_mode_time

        # 计算平均速度
        if self.speed_samples:
            self.avg_speed = np.mean(self.speed_samples)

        # 计算总电池消耗
        self.battery_drain_total = self.initial_battery - self.min_battery

    def get_report(self) -> dict:
        """获取飞行统计报告"""
        session_duration = time.time() - self.session_start

        # 计算平均速度
        avg_speed = self.avg_speed
        if not avg_speed and self.speed_samples:
            avg_speed = np.mean(self.speed_samples)

        # 计算电池消耗率
        battery_drain_per_minute = 0.0
        flight_mins = self.total_flight_time / 60.0
        if flight_mins > 0 and self.battery_samples:
            battery_used = self.initial_battery - min(self.battery_samples)
            battery_drain_per_minute = battery_used / flight_mins

        # 电量还能支撑的时间
        remaining_time = 0.0
        if battery_drain_per_minute > 0.01:
            current_battery = self.battery_samples[-1] if self.battery_samples else 100.0
            remaining_time = current_battery / battery_drain_per_minute

        # 手势排名
        sorted_gestures = sorted(self.gesture_counts.items(), key=lambda x: x[1], reverse=True)
        top_gestures = sorted_gestures[:5]

        # 命令排名
        sorted_commands = sorted(self.command_counts.items(), key=lambda x: x[1], reverse=True)
        top_commands = sorted_commands[:5]

        # 模式时间排名
        sorted_modes = sorted(self.mode_time.items(), key=lambda x: x[1], reverse=True)

        return {
            'session_duration': session_duration,
            'total_flight_time': self.total_flight_time,
            'total_distance': self.total_distance,
            'max_altitude': self.max_altitude,
            'max_distance_from_home': self.max_distance_from_home,
            'max_speed': self.max_speed,
            'avg_speed': avg_speed,
            'takeoff_count': self.takeoff_count,
            'landing_count': self.landing_count,
            'total_gestures': sum(self.gesture_counts.values()),
            'total_commands': sum(self.command_counts.values()),
            'top_gestures': top_gestures,
            'top_commands': top_commands,
            'gesture_counts': dict(self.gesture_counts),
            'command_counts': dict(self.command_counts),
            'battery': {
                'initial': self.initial_battery,
                'current': self.battery_samples[-1] if self.battery_samples else 100.0,
                'min': self.min_battery,
                'drain_per_minute': battery_drain_per_minute,
                'remaining_time': remaining_time
            },
            'mode_time': dict(self.mode_time),
            'sorted_modes': sorted_modes,
        }

    def reset(self):
        """重置统计数据"""
        self.__init__()

    def format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m{s}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h{m}m"

    def print_report(self):
        """打印统计报告到控制台"""
        report = self.get_report()

        print("\n" + "=" * 60)
        print("              飞 行 统 计 报 告")
        print("=" * 60)
        print(f" 会话时长:       {self.format_time(report['session_duration'])}")
        print(f" 累计飞行时长:   {self.format_time(report['total_flight_time'])}")
        print(f" 起飞次数:       {report['takeoff_count']}")
        print(f" 降落次数:       {report['landing_count']}")
        print("-" * 60)
        print(f" 累计飞行距离:   {report['total_distance']:.2f} m")
        print(f" 最高高度:       {report['max_altitude']:.2f} m")
        print(f" 最远距离(起飞点): {report['max_distance_from_home']:.2f} m")
        print(f" 最大速度:       {report['max_speed']:.2f} m/s")
        print(f" 平均速度:       {report['avg_speed']:.2f} m/s")
        print("-" * 60)

        battery = report['battery']
        print(f" 电池:           {battery['current']:.1f}% (初始{battery['initial']:.1f}%)")
        print(f" 耗电率:         {battery['drain_per_minute']:.2f} %/min")
        if battery['remaining_time'] > 0:
            print(f" 预估剩余飞行:   {self.format_time(battery['remaining_time'] * 60)}")
        print("-" * 60)

        print(f" 手势总触发:     {report['total_gestures']} 次")
        if report['top_gestures']:
            print("  热门手势:")
            for gesture, count in report['top_gestures']:
                pct = (count / report['total_gestures'] * 100) if report['total_gestures'] > 0 else 0
                print(f"    {gesture:20s} {count:4d} 次 ({pct:.0f}%)")

        print(f" 命令总执行:     {report['total_commands']} 次")
        if report['top_commands']:
            print("  热门命令:")
            for cmd, count in report['top_commands']:
                pct = (count / report['total_commands'] * 100) if report['total_commands'] > 0 else 0
                print(f"    {cmd:20s} {count:4d} 次 ({pct:.0f}%)")
        print("=" * 60)
