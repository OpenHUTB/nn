#!/usr/bin/env python3
"""完整五指手掌仿真"""
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer


def setup_environment():
    """设置环境"""
    os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
    return mujoco, mujoco.viewer


class FullHandController:
    """五指手掌控制器"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.mode = "idle"
        self.nu = model.nu

        # 设置初始位置
        self.set_default_positions()

        # 运动模式参数
        self.wave_phase = 0.0
        self.grasp_strength = 0.0

    def set_default_positions(self):
        """设置手的默认姿态（放松状态）"""
        # 手臂位置
        if self.nu >= 3:
            self.data.ctrl[0] = 0.1  # 肩部屈曲
            self.data.ctrl[1] = 0.05  # 肩部外展
            self.data.ctrl[2] = 0.0  # 肩部旋转

        if self.nu >= 4:
            self.data.ctrl[3] = -0.3  # 肘部弯曲

        if self.nu >= 6:
            self.data.ctrl[4] = 0.0  # 腕部屈曲
            self.data.ctrl[5] = 0.0  # 腕部外展

        # 手指位置（放松状态）
        finger_indices = list(range(6, min(self.nu, 21)))
        for i in finger_indices:
            self.data.ctrl[i] = 0.1  # 轻微弯曲

    def update_wave(self, t):
        """波浪形手指运动"""
        if self.nu >= 20:
            # 拇指
            self.data.ctrl[6] = 0.3 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
            self.data.ctrl[7] = 0.2 + 0.1 * np.sin(2 * np.pi * 0.5 * t + 0.5)
            self.data.ctrl[8] = 0.1 + 0.1 * np.sin(2 * np.pi * 0.5 * t + 1.0)

            # 食指
            self.data.ctrl[9] = 0.2 + 0.3 * np.sin(2 * np.pi * 0.8 * t)
            self.data.ctrl[10] = 0.1 + 0.2 * np.sin(2 * np.pi * 0.8 * t + 0.3)
            self.data.ctrl[11] = 0.1 + 0.1 * np.sin(2 * np.pi * 0.8 * t + 0.6)

            # 中指
            self.data.ctrl[12] = 0.2 + 0.3 * np.sin(2 * np.pi * 1.0 * t)
            self.data.ctrl[13] = 0.1 + 0.2 * np.sin(2 * np.pi * 1.0 * t + 0.3)
            self.data.ctrl[14] = 0.1 + 0.1 * np.sin(2 * np.pi * 1.0 * t + 0.6)

            # 无名指
            self.data.ctrl[15] = 0.15 + 0.25 * np.sin(2 * np.pi * 1.2 * t)
            self.data.ctrl[16] = 0.1 + 0.15 * np.sin(2 * np.pi * 1.2 * t + 0.3)
            self.data.ctrl[17] = 0.05 + 0.1 * np.sin(2 * np.pi * 1.2 * t + 0.6)

            # 小指
            self.data.ctrl[18] = 0.1 + 0.2 * np.sin(2 * np.pi * 1.5 * t)
            self.data.ctrl[19] = 0.05 + 0.1 * np.sin(2 * np.pi * 1.5 * t + 0.3)

    def update_grasp(self, strength):
        """抓握动作"""
        # 拇指
        if self.nu >= 9:
            self.data.ctrl[6] = 0.4 * strength  # 拇指CMC
            self.data.ctrl[7] = 0.6 * strength  # 拇指MCP
            self.data.ctrl[8] = 0.4 * strength  # 拇指IP

        # 其他手指（协同弯曲）
        finger_groups = [(9, 12), (10, 13), (11, 14)]  # 食指
        for start, end in finger_groups:
            if end < self.nu:
                for i in range(start, end + 1):
                    self.data.ctrl[i] = 0.7 * strength

    def update_idle(self, t):
        """空闲状态轻微运动"""
        if self.nu >= 6:
            # 手臂轻微摆动
            self.data.ctrl[0] = 0.05 * np.sin(2 * np.pi * 0.1 * t)
            self.data.ctrl[3] = -0.3 + 0.05 * np.sin(2 * np.pi * 0.15 * t + 0.5)

        # 手指轻微抖动
        for i in range(6, min(self.nu, 21)):
            self.data.ctrl[i] = 0.1 + 0.02 * np.sin(2 * np.pi * 0.2 * t + i * 0.1)

    def update(self, t, mode="idle", strength=0.0):
        """根据模式更新控制"""
        self.mode = mode

        if mode == "wave":
            self.update_wave(t)
        elif mode == "grasp":
            self.update_grasp(strength)
        else:  # idle
            self.update_idle(t)


class FullHandSimulation:
    """完整五指仿真"""

    def __init__(self, model_path="arm_model_full_hand.xml"):
        self.mujoco, self.viewer = setup_environment()

        print(f"加载完整五指模型: {model_path}")
        self.model = self.mujoco.MjModel.from_xml_path(model_path)
        self.data = self.mujoco.MjData(self.model)

        self.controller = FullHandController(self.model, self.data)
        self.sim_time = 0.0
        self.paused = False
        self.mode = "idle"
        self.grasp_strength = 0.0

        print(f"模型信息: {self.model.nu} 个执行器, {self.model.nv} 个自由度")

    def step(self):
        """执行仿真步"""
        if self.paused:
            return

        # 更新控制器
        if self.sim_time < 5.0:
            self.mode = "idle"
        elif self.sim_time < 10.0:
            self.mode = "wave"
        else:
            self.mode = "grasp"
            self.grasp_strength = 0.5 + 0.3 * np.sin(2 * np.pi * 0.3 * self.sim_time)

        self.controller.update(self.sim_time, self.mode, self.grasp_strength)

        # 物理步进
        self.mujoco.mj_step(self.model, self.data)
        self.sim_time = self.data.time

    def print_status(self):
        """打印状态信息"""
        if int(self.sim_time * 10) % 20 == 0:  # 每2秒打印一次
            print(f"\n时间: {self.sim_time:.1f}s | 模式: {self.mode}")
            print(f"拇指位置: {self.data.qpos[10]:.3f}, 食指: {self.data.qpos[13]:.3f}")

    def run_interactive(self):
        """交互式运行"""
        print("\n" + "=" * 60)
        print("完整五指手掌仿真 - 交互模式")
        print("=" * 60)
        print("控制指令:")
        print("  [空格] 暂停/继续")
        print("  [1] 空闲模式")
        print("  [2] 波浪模式")
        print("  [3] 抓握模式")
        print("  [ESC] 退出")
        print("=" * 60)

        with self.viewer.launch(self.model, self.data) as viewer:
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 120

            last_print_time = 0

            while viewer.is_running():
                # 处理按键
                if viewer.is_key_down(self.mujoco.mjtKey.mjKEY_SPACE):
                    self.paused = not self.paused
                    time.sleep(0.2)

                if viewer.is_key_down(ord('1')):
                    self.mode = "idle"
                    print("切换到空闲模式")

                if viewer.is_key_down(ord('2')):
                    self.mode = "wave"
                    print("切换到波浪模式")

                if viewer.is_key_down(ord('3')):
                    self.mode = "grasp"
                    print("切换到抓握模式")

                # 仿真步进
                self.step()

                # 打印状态
                if self.sim_time - last_print_time > 2.0:
                    self.print_status()
                    last_print_time = self.sim_time

                # 同步查看器
                viewer.sync()

                # 控制帧率
                time.sleep(0.001)

        print("\n仿真完成")


def main():
    """主函数"""
    print("🚀 启动完整五指手掌仿真")

    try:
        sim = FullHandSimulation()
        sim.run_interactive()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()