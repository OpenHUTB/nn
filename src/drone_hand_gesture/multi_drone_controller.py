"""
多无人机控制器
支持同时控制多架无人机，通过手势分别控制
"""

import numpy as np
import time
import logging
from drone_controller import SimulationDroneController


class MultiDroneController:
    """
    多无人机控制器
    管理多架无人机的状态和控制
    """

    # 无人机颜色定义（用于3D渲染）
    DRONE_COLORS = [
        (0.0, 0.6, 1.0),   # 青色 - Drone 1
        (1.0, 0.4, 0.0),   # 橙色 - Drone 2
        (0.6, 0.0, 1.0),   # 紫色 - Drone 3
        (0.0, 1.0, 0.4),   # 绿色 - Drone 4
    ]

    def __init__(self, num_drones=2, config=None, simulation_mode=True):
        """
        初始化多无人机控制器

        Args:
            num_drones: 无人机数量（默认2架）
            config: 配置管理器
            simulation_mode: 是否为仿真模式
        """
        self.num_drones = min(max(num_drones, 1), 4)  # 限制1-4架
        self.simulation_mode = simulation_mode
        self.config = config

        # 日志
        self.logger = logging.getLogger(__name__)

        # 初始化无人机列表
        self.drones = []
        self._init_drones()

        # 初始位置偏移（让无人机初始位置分开）
        self.initial_offsets = [
            np.array([0.0, 0.0, 0.0]),      # Drone 1: 原点
            np.array([3.0, 0.0, 3.0]),     # Drone 2: 偏移
            np.array([-3.0, 0.0, -3.0]),    # Drone 3: 偏移
            np.array([-3.0, 0.0, 3.0]),     # Drone 4: 偏移
        ]

        # 应用初始位置偏移
        self._apply_initial_offsets()

        # 当前选中的无人机（用于单无人机控制模式）
        self.selected_drone = 0

        # 多无人机控制模式
        # - "dual": 左手控制Drone1，右手控制Drone2（默认）
        # - "single": 只控制选中的无人机
        # - "sync": 双手控制所有无人机（同步模式）
        self.control_mode = "dual"

        print(f"[OK] 多无人机控制器初始化完成: {self.num_drones} 架无人机")
        self._log_drone_positions()

    def _init_drones(self):
        """初始化所有无人机"""
        for i in range(self.num_drones):
            drone = SimulationDroneController(
                config=self.config,
                simulation_mode=self.simulation_mode
            )
            drone.drone_id = i + 1
            self.drones.append(drone)

    def _apply_initial_offsets(self):
        """应用初始位置偏移"""
        for i, drone in enumerate(self.drones):
            if i < len(self.initial_offsets):
                drone.state['position'] = self.initial_offsets[i].copy()

    def _log_drone_positions(self):
        """记录所有无人机的初始位置"""
        for i, drone in enumerate(self.drones):
            pos = drone.state['position']
            print(f"  Drone {i+1}: 位置({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    def get_drone(self, index):
        """获取指定索引的无人机"""
        if 0 <= index < len(self.drones):
            return self.drones[index]
        return None

    def get_all_states(self):
        """获取所有无人机的状态"""
        states = []
        for i, drone in enumerate(self.drones):
            state = drone.get_state().copy()
            state['drone_id'] = i + 1
            state['color'] = self.DRONE_COLORS[i]
            states.append(state)
        return states

    def get_all_trajectories(self):
        """获取所有无人机的轨迹"""
        trajectories = []
        for drone in self.drones:
            trajectories.append(drone.get_trajectory())
        return trajectories

    def set_control_mode(self, mode):
        """设置控制模式"""
        valid_modes = ["dual", "single", "sync"]
        if mode in valid_modes:
            self.control_mode = mode
            print(f"[多无人机] 控制模式: {mode}")
            return True
        return False

    def cycle_control_mode(self):
        """循环切换控制模式"""
        modes = ["dual", "single", "sync"]
        current_idx = modes.index(self.control_mode) if self.control_mode in modes else 0
        next_idx = (current_idx + 1) % len(modes)
        self.set_control_mode(modes[next_idx])

    def select_drone(self, index):
        """选择要控制的无人机（单无人机模式）"""
        if 0 <= index < len(self.drones):
            self.selected_drone = index
            print(f"[多无人机] 选中 Drone {index + 1}")
            return True
        return False

    def send_command(self, command, intensity=0.5, drone_index=None):
        """
        发送控制命令

        Args:
            command: 命令类型
            intensity: 命令强度
            drone_index: 指定无人机索引（可选，默认控制选中的无人机）
        """
        if drone_index is not None:
            drone = self.get_drone(drone_index)
            if drone:
                drone.send_command(command, intensity)
                return True
            return False

        # 根据控制模式发送命令
        if self.control_mode == "single":
            # 只控制选中的无人机
            drone = self.get_drone(self.selected_drone)
            if drone:
                drone.send_command(command, intensity)
        elif self.control_mode == "sync":
            # 同步控制所有无人机
            for drone in self.drones:
                drone.send_command(command, intensity)
        # "dual" 模式由 handle_dual_hand_command 处理

        return True

    def handle_dual_hand_command(self, left_commands, right_commands):
        """
        处理双手控制命令（多无人机模式）

        Args:
            left_commands: 左手命令字典
            right_commands: 右手命令字典
        """
        results = {
            'drone1_command': None,
            'drone2_command': None,
            'special_command': None
        }

        # 处理左手 - 控制 Drone 1
        if left_commands:
            if left_commands.get('special_command'):
                results['special_command'] = left_commands['special_command']
            elif left_commands.get('direction_command'):
                results['drone1_command'] = left_commands['direction_command']
                drone1 = self.get_drone(0)
                if drone1:
                    drone1.send_command(
                        left_commands['direction_command'],
                        left_commands.get('direction_intensity', 0.5)
                    )

        # 处理右手 - 控制 Drone 2
        if right_commands:
            if right_commands.get('special_command') and not results['special_command']:
                # 特殊命令优先（如果左手没发送）
                results['special_command'] = right_commands['special_command']
                # 执行特殊命令到所有无人机
                self.send_command(right_commands['special_command'], 1.0)
            elif right_commands.get('altitude_command'):
                results['drone2_command'] = right_commands['altitude_command']
                drone2 = self.get_drone(1)
                if drone2:
                    drone2.send_command(
                        right_commands['altitude_command'],
                        right_commands.get('altitude_intensity', 0.5)
                    )

        return results

    def takeoff_all(self, altitude=None):
        """所有无人机起飞"""
        print(f"[多无人机] 全体起飞")
        for i, drone in enumerate(self.drones):
            drone.takeoff(altitude)
            print(f"  Drone {i+1}: 起飞")

    def land_all(self):
        """所有无人机降落"""
        print(f"[多无人机] 全体降落")
        for i, drone in enumerate(self.drones):
            drone.send_command("land", 0.5)
            print(f"  Drone {i+1}: 降落")

    def hover_all(self):
        """所有无人机悬停"""
        for drone in self.drones:
            drone.send_command("hover")

    def stop_all(self):
        """所有无人机停止"""
        for drone in self.drones:
            drone.send_command("stop")

    def reset_all(self):
        """重置所有无人机到初始位置"""
        self._apply_initial_offsets()
        for drone in self.drones:
            drone.state['velocity'] = np.array([0.0, 0.0, 0.0])
            drone.state['armed'] = False
            drone.state['mode'] = 'IDLE'
        print("[多无人机] 所有无人机已重置")

    def update_all_physics(self, dt):
        """更新所有无人机的物理状态"""
        for drone in self.drones:
            drone.update_physics(dt)

    def disconnect_all(self):
        """断开所有无人机的连接"""
        for drone in self.drones:
            drone.disconnect()

    def get_status_string(self):
        """获取所有无人机的状态字符串"""
        lines = [f"=== 多无人机状态 ({self.num_drones}架) ==="]
        for i, drone in enumerate(self.drones):
            state = drone.state
            pos = state['position']
            status = f"D{i+1}: {state['mode'][:6]:6} | " \
                    f"Pos({pos[0]:5.1f},{pos[1]:5.1f},{pos[2]:5.1f}) | " \
                    f"Batt:{state['battery']:5.1f}%"
            lines.append(status)
        return "\n".join(lines)

    def get_control_mode_description(self):
        """获取控制模式描述"""
        descriptions = {
            "dual": "双手分离控制（左手D1，右手D2）",
            "single": "单无人机控制（选中D1）",
            "sync": "同步控制（双手控制所有机）"
        }
        return descriptions.get(self.control_mode, "未知模式")


class DroneFormation:
    """
    无人机编队控制器
    支持无人机编队飞行
    """

    FORMATIONS = {
        "line": [(0, 0, 0), (3, 0, 0), (6, 0, 0), (9, 0, 0)],           # 直线
        "v_shape": [(0, 0, 0), (2, 0, 2), (4, 0, 0), (2, 0, -2)],       # V字形
        "square": [(0, 0, 0), (3, 0, 0), (3, 0, 3), (0, 0, 3)],          # 方形
        "triangle": [(0, 0, 3), (2, 0, 0), (-2, 0, 0), (0, 0, 0)],       # 三角形
    }

    def __init__(self, multi_controller):
        self.controller = multi_controller
        self.current_formation = "v_shape"
        self.formation_offset = np.array([0.0, 0.0, 0.0])
        self.leader_follows_gesture = True  # 队长跟随手势

    def set_formation(self, formation_name):
        """设置编队形状"""
        if formation_name in self.FORMATIONS:
            self.current_formation = formation_name
            self._apply_formation()
            print(f"[编队] 已设置为: {formation_name}")
            return True
        return False

    def _apply_formation(self):
        """应用编队位置"""
        formation = self.FORMATIONS.get(self.current_formation, [])
        drones = self.controller.drones

        for i, drone in enumerate(drones):
            if i < len(formation):
                target_pos = np.array(formation[i]) + self.formation_offset
                drone.state['target_formation_pos'] = target_pos

    def update_formation(self, leader_position):
        """更新编队位置（跟随队长）"""
        if not self.leader_follows_gesture:
            return

        formation = self.FORMATIONS.get(self.current_formation, [])
        drones = self.controller.drones

        for i, drone in enumerate(drones):
            if i < len(formation):
                target_pos = np.array(formation[i]) + leader_position
                drone.state['target_formation_pos'] = target_pos
                # 简单跟随：直接设置位置
                drone.state['position'] = target_pos.copy()

    def move_formation(self, direction, intensity=0.5):
        """移动编队"""
        direction_vectors = {
            "forward": np.array([0.0, 0.0, -intensity]),
            "backward": np.array([0.0, 0.0, intensity]),
            "left": np.array([-intensity, 0.0, 0.0]),
            "right": np.array([intensity, 0.0, 0.0]),
            "up": np.array([0.0, intensity, 0.0]),
            "down": np.array([0.0, -intensity, 0.0]),
        }

        if direction in direction_vectors:
            offset = direction_vectors[direction]
            self.formation_offset += offset
            self._apply_formation()
