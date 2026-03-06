import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import sys
import keyboard  # 将导入提到顶部，避免重复导入
from typing import Dict, List, Tuple  # 增加类型注解提升可读性

# ===================== 配置常量（集中管理） =====================
# 全局控制变量
KEY_STATE: Dict[str, bool] = {
    'w': False, 's': False, 'a': False, 'd': False,
    'q': False, 'e': False, 'space': False,
    'auto': True  # 默认自动模式
}

# 控制参数（集中配置，便于调整）
CONFIG = {
    'action_speed': 0.08,  # 动作速度（越大越快）
    'grab_threshold': 0.12,  # 抓取判定阈值
    'ball_drop_threshold': 0.05,  # 小球脱落判定阈值
    'joint_limits': {  # 关节限位（集中管理）
        'shoulder': (-1.57, 1.57),
        'elbow': (-2.0, 2.0),
        'finger': (0.0, 0.8)
    },
    'auto_phase_durations': [2.0, 2.0, 1.0, 1.0, 3.0, 2.0, 2.0],  # 各阶段时长
    'initial_pose': np.array([0.2, -0.6, 0.0, 0.0]),  # 初始姿态
    'camera_params': {  # 相机参数
        'azimuth': 60,
        'elevation': -25,
        'distance': 1.8,
        'lookat': [0.2, 0, 0.3]
    }
}


# ===================== 工具函数 =====================
def key_listener() -> None:
    """键盘监听线程（非阻塞）- 优化异常处理和命名"""

    def on_press(key: keyboard.KeyboardEvent) -> None:
        try:
            key_name = key.name.lower()  # 统一小写，避免大小写问题
            if key_name in KEY_STATE:
                KEY_STATE[key_name] = True
            elif key_name == 'esc':
                sys.exit(0)
        except Exception as e:
            print(f"键盘监听异常: {e}", file=sys.stderr)

    def on_release(key: keyboard.KeyboardEvent) -> None:
        try:
            key_name = key.name.lower()
            if key_name in KEY_STATE:
                KEY_STATE[key_name] = False
        except Exception as e:
            print(f"键盘释放异常: {e}", file=sys.stderr)

    # 注册回调（增加防重复注册）
    keyboard.unhook_all()
    keyboard.on_press(on_press)
    keyboard.on_release(on_release)

    # 守护线程循环（优化睡眠）
    while True:
        time.sleep(0.001)  # 更精细的睡眠，提升响应速度


def get_ball_position(data: mujoco.MjData, model: mujoco.MjModel) -> np.ndarray:
    """获取目标小球位置 - 增加异常处理"""
    try:
        ball_body_id = model.body('target').id
        return data.xpos[ball_body_id].copy()
    except Exception as e:
        print(f"获取小球位置失败: {e}", file=sys.stderr)
        return np.zeros(3)


def is_ball_grabbed(data: mujoco.MjData, model: mujoco.MjModel) -> bool:
    """检测是否成功抓取小球 - 优化逻辑"""
    ball_pos = get_ball_position(data, model)
    gripper_pos = data.xpos[model.body('wrist').id]
    distance = np.linalg.norm(ball_pos - gripper_pos)
    return distance < CONFIG['grab_threshold']


def clamp_joint_values(target: np.ndarray) -> np.ndarray:
    """关节值限位 - 抽离为独立函数，提升复用性"""
    target[0] = np.clip(target[0], *CONFIG['joint_limits']['shoulder'])  # 肩部
    target[1] = np.clip(target[1], *CONFIG['joint_limits']['elbow'])  # 肘部
    target[2] = np.clip(target[2], *CONFIG['joint_limits']['finger'])  # 左指
    target[3] = np.clip(target[3], *CONFIG['joint_limits']['finger'])  # 右指
    return target


# ===================== 主函数 =====================
def main() -> None:
    # 加载模型（增加异常处理）
    try:
        model = mujoco.MjModel.from_xml_path("arm_model.xml")
        data = mujoco.MjData(model)
    except FileNotFoundError:
        print("错误：找不到arm_model.xml文件，请检查路径！", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"加载模型失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 启动键盘监听线程（优化线程命名）
    listener_thread = threading.Thread(target=key_listener, daemon=True, name="KeyListener")
    listener_thread.start()

    # 初始化变量（使用配置常量）
    phase = 0
    phase_time = 0.0
    target = CONFIG['initial_pose'].copy()
    last_ball_pos = get_ball_position(data, model)

    # 启动可视化
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 相机参数设置（使用配置常量）
        cam = viewer.cam
        cam.azimuth = CONFIG['camera_params']['azimuth']
        cam.elevation = CONFIG['camera_params']['elevation']
        cam.distance = CONFIG['camera_params']['distance']
        cam.lookat = CONFIG['camera_params']['lookat']

        # 打印控制说明（优化格式）
        print("=" * 60)
        print("MuJoCo 两指机械臂控制程序 | 优化版")
        print("├─ 控制说明：")
        print("│  W/S：控制肩部关节（上下） | A/D：控制肘部关节（前后）")
        print("│  Q/E：控制手指开合（抓取/释放） | 空格：重置初始姿态")
        print("│  ESC：退出程序 | 自动模式：默认开启，按任意键切换为手动")
        print("=" * 60)

        # 主循环（优化时间计算）
        last_frame_time = time.time()
        while viewer.is_running():
            # 控制帧率（更精准的控制）
            current_time = time.time()
            delta_time = current_time - last_frame_time
            if delta_time < 1 / 120:  # 限制最大帧率120fps
                time.sleep(1 / 120 - delta_time)
            last_frame_time = time.time()

            phase_time += model.opt.timestep
            manual_control = np.zeros(4)

            # 手动控制逻辑（优化判断逻辑）
            manual_triggered = any(KEY_STATE[k] for k in ['w', 's', 'a', 'd', 'q', 'e', 'space'])
            if manual_triggered:
                KEY_STATE['auto'] = False

                # 肩部控制（W=上，S=下）
                manual_control[0] -= CONFIG['action_speed'] if KEY_STATE['w'] else 0
                manual_control[0] += CONFIG['action_speed'] if KEY_STATE['s'] else 0

                # 肘部控制（A=前，D=后）
                manual_control[1] -= CONFIG['action_speed'] if KEY_STATE['a'] else 0
                manual_control[1] += CONFIG['action_speed'] if KEY_STATE['d'] else 0

                # 手指控制（Q=闭合，E=张开）
                finger_ctrl = -CONFIG['action_speed'] if KEY_STATE['q'] else (
                    CONFIG['action_speed'] if KEY_STATE['e'] else 0)
                manual_control[2] += finger_ctrl
                manual_control[3] += finger_ctrl

                # 重置姿态
                if KEY_STATE['space']:
                    target = CONFIG['initial_pose'].copy()
                    phase = 0
                    phase_time = 0
                    KEY_STATE['auto'] = True  # 重置后恢复自动模式

                # 应用手动控制并限位
                target += manual_control
                target = clamp_joint_values(target)

            # 自动循环抓取逻辑
            elif KEY_STATE['auto']:
                phase_durations = CONFIG['auto_phase_durations']
                phase_actions = [
                    np.array([0.2, -0.6, 0.0, 0.0]),  # 阶段0：初始准备
                    np.array([0.0, -1.2, 0.0, 0.0]),  # 阶段1：向前伸臂
                    np.array([0.0, -1.2, 0.7, 0.7]),  # 阶段2：张开手指
                    np.array([0.0, -1.2, 0.1, 0.1]),  # 阶段3：闭合抓取
                    np.array([-0.7, 0.6, 0.1, 0.1]),  # 阶段4：举升小球
                    np.array([0.0, -1.2, 0.1, 0.1]),  # 阶段5：放下小球
                    np.array([0.0, -1.2, 0.7, 0.7])  # 阶段6：张开手指复位
                ]
                phase_names = [
                    "阶段0：复位初始姿态", "阶段1：向前伸臂", "阶段2：张开手指",
                    "阶段3：闭合抓取", "阶段4：举升小球", "阶段5：放下小球",
                    "阶段6：张开手指复位"
                ]

                # 设置当前阶段目标
                target = phase_actions[phase]

                # 阶段切换逻辑
                if phase == 3:  # 抓取阶段：检测抓取状态
                    if is_ball_grabbed(data, model) or phase_time > phase_durations[phase] * 2:
                        phase = (phase + 1) % len(phase_actions)
                        phase_time = 0
                        print(f"\r[自动模式] {phase_names[phase]}", end="", flush=True)
                elif phase == 4:  # 举升阶段：检测小球脱落
                    current_ball_pos = get_ball_position(data, model)
                    if np.linalg.norm(current_ball_pos - last_ball_pos) > CONFIG['ball_drop_threshold']:
                        print("\r[警告] 小球脱落，重新抓取！", end="", flush=True)
                        phase = 1
                        phase_time = 0
                    elif phase_time > phase_durations[phase]:
                        phase = (phase + 1) % len(phase_actions)
                        phase_time = 0
                        print(f"\r[自动模式] {phase_names[phase]}", end="", flush=True)
                    last_ball_pos = current_ball_pos
                else:  # 普通阶段：按时长切换
                    if phase_time > phase_durations[phase]:
                        phase = (phase + 1) % len(phase_actions)
                        phase_time = 0
                        print(f"\r[自动模式] {phase_names[phase]}", end="", flush=True)

            # 平滑插值控制（避免动作突变）
            data.ctrl[:] = 0.95 * data.ctrl + 0.05 * target

            # 执行物理仿真步
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断，正常退出")
    except Exception as e:
        print(f"\n程序异常退出：{e}", file=sys.stderr)
        sys.exit(1)