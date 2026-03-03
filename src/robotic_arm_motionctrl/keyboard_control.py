import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
from pynput import keyboard
from typing import Dict, List
import os
import warnings

warnings.filterwarnings('ignore')  # 屏蔽无关警告


class KeyboardControlledArm:
    def __init__(self, model_path: str):
        """初始化键盘控制机械臂（优化版）"""
        # 1. 模型校验
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 2. 自动识别执行器（兼容多种命名规则，无需手动改）
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.actuator_ids, self.valid_actuators = self._auto_detect_actuators()
        self._print_actuator_info()

        # 3. 关节配置（适配6自由度机械臂安全范围）
        self.joint_limits = {
            "joint1": (-np.pi, np.pi),
            "joint2": (-1.8, 1.8),
            "joint3": (-2.0, 0.8),
            "joint4": (-2.0, 2.0),
            "joint5": (-1.8, 1.8),
            "joint6": (-2.0, 2.0)
        }

        # 4. 控制参数（优化手感）
        self.base_speed = 0.8  # 基础速度（降低避免动作过猛）
        self.speed_multiplier = 1.0  # 速度倍率（可按Shift加速）
        self.control_state = {name: 0.0 for name in self.joint_names}
        self.running = True
        self.last_key_time = 0.0  # 防抖

        # 5. 键盘监听（独立线程，提升响应）
        self.key_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release,
            suppress=False  # 不屏蔽系统按键
        )
        self.key_listener.start()
        self.print_controls()

    def _auto_detect_actuators(self) -> (Dict[str, int], List[str]):
        """自动检测执行器ID（兼容motor_jointX/jointX两种命名）"""
        actuator_ids = {}
        valid_actuators = []
        for name in self.joint_names:
            # 尝试两种命名规则
            act_names = [f"motor_{name}", name]
            for act_name in act_names:
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                if act_id != -1:
                    actuator_ids[name] = act_id
                    valid_actuators.append(name)
                    break
            else:
                print(f"⚠️  未找到{name}对应的执行器，请检查XML！")
        return actuator_ids, valid_actuators

    def _print_actuator_info(self):
        """打印执行器检测结果"""
        print("\n🔍 执行器检测结果：")
        for joint, aid in self.actuator_ids.items():
            print(f"   ✅ {joint} → 执行器ID: {aid}")
        if len(self.valid_actuators) < 6:
            print(f"⚠️  仅检测到{len(self.valid_actuators)}个执行器，部分关节无法控制！")

    def print_controls(self):
        """优化版控制说明"""
        print("\n" + "=" * 60)
        print("🎮 机械臂键盘控制说明（先点击MuJoCo窗口获取焦点！）")
        print("├─ 基础控制 ───────────────────────────")
        print("│  ← → ：joint1（基座旋转） | ↑ ↓ ：joint2（大臂）")
        print("│  PageUp/PageDown：joint3（小臂） | A/D：joint4（腕部旋转）")
        print("│  W/S：joint5（腕部俯仰） | Q/E：joint6（末端旋转）")
        print("├─ 功能键 ───────────────────────────")
        print("│  空格：紧急停止 | Shift：加速（1.5倍） | Ctrl：减速（0.5倍）")
        print("│  ESC：退出程序  | R：重置机械臂到零位")
        print("=" * 60 + "\n")

    def on_key_press(self, key):
        """按键按下处理（防抖+速度倍率）"""
        current_time = time.time()
        if current_time - self.last_key_time < 0.05:  # 防抖
            return
        self.last_key_time = current_time

        try:
            # 速度倍率控制
            if key == keyboard.Key.shift:
                self.speed_multiplier = 1.5
                return
            elif key == keyboard.Key.ctrl:
                self.speed_multiplier = 0.5
                return

            # 功能键
            if key == keyboard.Key.space:
                self.control_state = {name: 0.0 for name in self.joint_names}
                print("⏹️  所有关节已停止！")
                return
            elif key == keyboard.Key.esc:
                self.running = False
                print("🛑 准备退出程序...")
                return
            elif key == keyboard.KeyCode.from_char('r'):
                self._reset_arm()
                print("🔄 机械臂已重置到零位！")
                return

            # 关节控制（带速度倍率）
            speed = self.base_speed * self.speed_multiplier
            key_map = {
                keyboard.Key.left: ("joint1", -speed),
                keyboard.Key.right: ("joint1", speed),
                keyboard.Key.up: ("joint2", speed),
                keyboard.Key.down: ("joint2", -speed),
                keyboard.Key.page_up: ("joint3", speed),
                keyboard.Key.page_down: ("joint3", -speed),
                keyboard.KeyCode.from_char('a'): ("joint4", -speed),
                keyboard.KeyCode.from_char('d'): ("joint4", speed),
                keyboard.KeyCode.from_char('w'): ("joint5", speed),
                keyboard.KeyCode.from_char('s'): ("joint5", -speed),
                keyboard.KeyCode.from_char('q'): ("joint6", speed),
                keyboard.KeyCode.from_char('e'): ("joint6", -speed),
            }
            if key in key_map:
                joint, val = key_map[key]
                self.control_state[joint] = val

        except AttributeError:
            pass

    def on_key_release(self, key):
        """按键松开处理（精准停止对应关节）"""
        try:
            # 恢复速度倍率
            if key in [keyboard.Key.shift, keyboard.Key.ctrl]:
                self.speed_multiplier = 1.0
                return

            # 停止对应关节
            key_joint_map = {
                keyboard.Key.left: "joint1",
                keyboard.Key.right: "joint1",
                keyboard.Key.up: "joint2",
                keyboard.Key.down: "joint2",
                keyboard.Key.page_up: "joint3",
                keyboard.Key.page_down: "joint3",
                keyboard.KeyCode.from_char('a'): "joint4",
                keyboard.KeyCode.from_char('d'): "joint4",
                keyboard.KeyCode.from_char('w'): "joint5",
                keyboard.KeyCode.from_char('s'): "joint5",
                keyboard.KeyCode.from_char('q'): "joint6",
                keyboard.KeyCode.from_char('e'): "joint6",
            }
            if key in key_joint_map:
                joint = key_joint_map[key]
                self.control_state[joint] = 0.0

        except AttributeError:
            pass

    def _reset_arm(self):
        """重置机械臂到零位"""
        for joint in self.joint_names:
            if joint in self.actuator_ids:
                self.data.ctrl[self.actuator_ids[joint]] = 0.0
        self.data.qpos[:6] = 0.0  # 直接重置关节角
        mujoco.mj_forward(self.model, self.data)

    def _clamp_joint_angle(self, joint_name: str, angle: float) -> float:
        """关节角度限位保护"""
        min_angle, max_angle = self.joint_limits[joint_name]
        return np.clip(angle, min_angle, max_angle)

    def update_control(self):
        """优化版控制更新（加限位+平滑）"""
        for joint_name in self.joint_names:
            if joint_name not in self.actuator_ids:
                continue

            act_id = self.actuator_ids[joint_name]
            # 计算目标角度增量
            target_delta = self.control_state[joint_name] * self.model.opt.timestep
            # 当前关节角
            current_angle = self.data.qpos[self.actuator_ids[joint_name]]
            # 限位后更新
            new_angle = self._clamp_joint_angle(joint_name, current_angle + target_delta)
            # 速度控制 → 位置控制（更稳定）
            self.data.ctrl[act_id] = (new_angle - current_angle) / self.model.opt.timestep

    def run(self):
        """主循环（优化帧率+稳定性）"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 初始化相机视角（适配你的机械臂）
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 55
            viewer.cam.elevation = -20
            viewer.cam.lookat = [0.15, 0.0, 0.3]

            print("✅ 仿真启动成功！请点击MuJoCo窗口后开始控制...")
            sim_start = time.time()

            while self.running and viewer.is_running():
                # 控制更新
                self.update_control()
                # 步进仿真
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                # 稳定帧率（60FPS）
                time.sleep(max(0, 1 / 60 - (time.time() - sim_start)))

        # 资源清理
        self.key_listener.stop()
        print("\n👋 程序已安全退出！")


def main():
    """主函数（加异常捕获）"""
    model_path = "arm6dof_final.xml"
    try:
        arm = KeyboardControlledArm(model_path)
        arm.run()
    except FileNotFoundError as e:
        print(f"❌ 错误：{e}")
        print(f"   请确认文件路径：{os.path.abspath(model_path)}")
    except Exception as e:
        print(f"❌ 运行错误：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()