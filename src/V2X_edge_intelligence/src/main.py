"""
MuJoCo自动巡航小车 - 优化版
核心功能：
1. 恒定速度巡航（可配置）
2. 前方障碍物检测（0.5米阈值）
3. 90度精准转向避障（左/右随机）
4. R键复位
5. 状态机管理（巡航/停止/转向/恢复）
"""
import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard
import math
import random
import time
from typing import Tuple, Optional, Dict


# ====================== 全局配置（集中管理，便于调整） ======================
class Config:
    # 速度配置
    CRUISE_SPEED = 0.0015  # 巡航速度（原两倍速）
    TURN_SPEED = 0.00075  # 转向速度（巡航速度的50%）
    # 距离配置
    OBSTACLE_THRESHOLD = 0.5  # 障碍物检测阈值（米）
    SAFE_DISTANCE = 0.2  # 紧急停止距离（米）
    # 转向配置
    TURN_ANGLE = math.pi / 2  # 90度转向（弧度）
    STEER_LIMIT = 0.3  # 转向角度限制（匹配XML中的ctrlrange）
    # 模型配置
    MODEL_PATH = "wheeled_car.xml"
    CHASSIS_NAME = "chassis"
    # 障碍物名称列表
    OBSTACLE_NAMES = [
        'obs_box1', 'obs_box2', 'obs_box3', 'obs_box4',
        'obs_ball1', 'obs_ball2', 'obs_ball3',
        'wall1', 'wall2', 'front_dark_box'
    ]
    # 视图配置
    CAM_DISTANCE = 2.5
    CAM_ELEVATION = -25


# ====================== 状态枚举（清晰定义小车状态） ======================
class CarState:
    CRUISING = "CRUISING"  # 正常巡航
    STOPPED = "STOPPED"  # 检测到障碍停止
    TURNING = "TURNING"  # 执行转向避障
    RESUMING = "RESUMING"  # 转向后恢复巡航


# ====================== 键盘管理器（封装键盘监听逻辑） ======================
class KeyManager:
    def __init__(self):
        self.keys = {keyboard.KeyCode.from_char('r'): False}
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, k):
        if k in self.keys:
            self.keys[k] = True

    def _on_release(self, k):
        if k in self.keys:
            self.keys[k] = False

    def is_pressed(self, key_char: str) -> bool:
        key = keyboard.KeyCode.from_char(key_char)
        return self.keys.get(key, False)

    def reset_key(self, key_char: str):
        key = keyboard.KeyCode.from_char(key_char)
        if key in self.keys:
            self.keys[key] = False


# ====================== 小车控制器（核心逻辑封装） ======================
class CruiseCarController:
    def __init__(self, config: Config):
        self.config = config
        self.key_manager = KeyManager()

        # 加载模型
        try:
            self.model = mujoco.MjModel.from_xml_path(config.MODEL_PATH)
            self.data = mujoco.MjData(self.model)
            print(f"✅ 成功加载模型: {config.MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"❌ 加载模型失败: {e}")

        # 初始化ID缓存（避免重复计算）
        self._init_ids()

        # 初始化状态
        self.car_state = CarState.CRUISING
        self.target_steer_angle = 0.0  # 目标转向角度
        self.turn_direction = 1  # 转向方向（1=右，-1=左）
        self.turn_progress = 0.0  # 转向进度（0-1）

        # 复位小车
        self.reset_car()

    def _init_ids(self):
        """初始化body ID缓存"""
        # 小车底盘ID
        self.chassis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.CHASSIS_NAME
        )
        if self.chassis_id == -1:
            raise RuntimeError(f"❌ 未找到底盘: {self.config.CHASSIS_NAME}")

        # 障碍物ID
        self.obstacle_ids = {}
        for name in self.config.OBSTACLE_NAMES:
            obs_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if obs_id != -1:
                self.obstacle_ids[name] = obs_id
        print(f"✅ 加载{len(self.obstacle_ids)}个障碍物ID")

    def reset_car(self):
        """复位小车到初始状态"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.03  # 设置初始高度
        self.car_state = CarState.CRUISING
        self.target_steer_angle = 0.0
        self.turn_progress = 0.0
        print("\n🔄 小车已复位")

    def detect_obstacle(self) -> Tuple[int, float, Optional[str]]:
        """
        检测前方障碍物
        返回值：(状态码, 最近距离, 障碍物名称)
        状态码：0=无障碍, 1=检测到障碍, 2=紧急停止
        """
        # 获取小车位置
        chassis_pos = self.data.body(self.chassis_id).xpos
        min_distance = float('inf')
        closest_obs = None

        # 遍历所有障碍物
        for obs_name, obs_id in self.obstacle_ids.items():
            try:
                obs_pos = self.data.body(obs_id).xpos
                # 计算相对距离
                dx = obs_pos[0] - chassis_pos[0]  # 前向距离
                dy = obs_pos[1] - chassis_pos[1]  # 横向距离
                distance = math.hypot(dx, dy)

                # 只检测前方且横向偏移小于小车宽度的障碍
                if dx > 0 and abs(dy) < 0.3 and distance < self.config.OBSTACLE_THRESHOLD:
                    if distance < min_distance:
                        min_distance = distance
                        closest_obs = obs_name
            except Exception:
                continue

        # 判断状态
        if closest_obs is None:
            return 0, 0.0, None
        elif min_distance < self.config.SAFE_DISTANCE:
            return 2, min_distance, closest_obs
        else:
            return 1, min_distance, closest_obs

    def _set_steer(self, angle: float):
        """设置转向角度（限制在合法范围内）"""
        clamped_angle = np.clip(angle, -self.config.STEER_LIMIT, self.config.STEER_LIMIT)
        self.data.ctrl[0] = clamped_angle  # 前左转向
        self.data.ctrl[1] = clamped_angle  # 前右转向

    def _set_speed(self, speed: float):
        """设置驱动速度"""
        self.data.ctrl[2] = speed  # 前左驱动
        self.data.ctrl[3] = speed  # 前右驱动
        self.data.ctrl[4] = speed  # 后左驱动
        self.data.ctrl[5] = speed  # 后右驱动

    def update(self):
        """更新小车状态和控制指令"""
        # 检查复位键
        if self.key_manager.is_pressed('r'):
            self.reset_car()
            self.key_manager.reset_key('r')
            return

        # 检测障碍物
        obs_status, obs_dist, obs_name = self.detect_obstacle()

        # 状态机逻辑
        if self.car_state == CarState.CRUISING:
            # 巡航状态：检测到障碍则停止
            if obs_status == 2:
                self.car_state = CarState.STOPPED
                self._set_speed(0.0)
                print(f"\n🛑 紧急停止！障碍物：{obs_name} (距离: {obs_dist:.2f}m)")
            elif obs_status == 1:
                self.car_state = CarState.STOPPED
                self._set_speed(0.0)
                print(f"\n⚠️  检测到障碍物：{obs_name} (距离: {obs_dist:.2f}m)，正在停止")
            else:
                # 正常巡航
                self._set_steer(0.0)
                self._set_speed(self.config.CRUISE_SPEED)

        elif self.car_state == CarState.STOPPED:
            # 停止状态：延迟后开始转向
            self.turn_progress += 1
            self._set_speed(0.0)

            if self.turn_progress > 10:  # 等待10帧确保停止
                # 随机选择转向方向
                self.turn_direction = 1 if random.random() > 0.5 else -1
                self.target_steer_angle = self.turn_direction * self.config.TURN_ANGLE
                self.car_state = CarState.TURNING
                self.turn_progress = 0.0
                dir_text = "右" if self.turn_direction > 0 else "左"
                print(f"\n🔄 开始{dir_text}转90度避障")

        elif self.car_state == CarState.TURNING:
            # 转向状态：精准控制转向角度
            self.turn_progress += 0.01
            # 计算当前转向角度（渐进式）
            current_steer = self.target_steer_angle * self.turn_progress
            self._set_steer(current_steer)
            self._set_speed(self.config.TURN_SPEED)

            # 转向完成判断（角度到位或进度完成）
            if self.turn_progress >= 1.0 or abs(current_steer) >= self.config.STEER_LIMIT:
                self.car_state = CarState.RESUMING
                self.turn_progress = 0.0
                print("\n✅ 转向完成，开始恢复巡航")

        elif self.car_state == CarState.RESUMING:
            # 恢复状态：逐渐回正转向并加速
            self.turn_progress += 0.01
            # 渐进回正转向
            current_steer = self.target_steer_angle * (1 - self.turn_progress)
            self._set_steer(current_steer)
            # 渐进恢复速度
            current_speed = self.config.TURN_SPEED + (
                        self.config.CRUISING - self.config.TURN_SPEED) * self.turn_progress
            self._set_speed(current_speed)

            # 恢复完成判断
            if self.turn_progress >= 1.0:
                # 检查前方是否安全
                if obs_status == 0:
                    self.car_state = CarState.CRUISING
                    print("\n🚗 前方安全，恢复正常巡航")
                else:
                    self.car_state = CarState.STOPPED
                    print("\n⚠️  转向后仍有障碍，重新停止")
                self.turn_progress = 0.0

    def get_status_info(self) -> str:
        """获取状态信息（用于打印）"""
        # 计算当前速度
        vel = np.linalg.norm(self.data.qvel[:3])
        # 当前转向角度（度）
        current_steer = math.degrees((self.data.ctrl[0] + self.data.ctrl[1]) / 2)

        # 基础状态信息
        info = f"状态: {self.car_state:<8} | 速度: {vel:.5f} m/s"
        if abs(current_steer) > 0.1:
            info += f" | 转向: {current_steer:.1f}°"

        # 巡航状态显示障碍物信息
        if self.car_state == CarState.CRUISING:
            obs_status, obs_dist, obs_name = self.detect_obstacle()
            if obs_status > 0 and obs_name:
                info += f" | 前方障碍: {obs_name} ({obs_dist:.2f}m)"

        return info


# ====================== 主程序入口 ======================
def main():
    # 初始化配置和控制器
    config = Config()
    try:
        controller = CruiseCarController(config)
    except RuntimeError as e:
        print(e)
        return

    # 启动可视化界面
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        # 设置相机参数
        viewer.cam.distance = config.CAM_DISTANCE
        viewer.cam.elevation = config.CAM_ELEVATION

        # 主循环
        print("\n🚀 自动巡航小车启动（按R键复位）")
        print("=====================================")
        while viewer.is_running():
            # 更新小车控制
            controller.update()
            # 执行仿真步骤
            mujoco.mj_step(controller.model, controller.data)
            # 打印状态信息
            status_info = controller.get_status_info()
            print(f"\r{status_info}", end='', flush=True)
            # 同步视图
            viewer.sync()

    print("\n\n👋 程序结束")


if __name__ == "__main__":
    main()