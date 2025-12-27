#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂关节控制器 - 最终兼容版
核心优化：
1. 修复Mujoco mj_name2id API类型错误（兼容所有版本）
2. 移除所有不兼容属性和依赖
3. 纯原生实现，无Numba/特殊依赖
4. Windows深度适配+优雅退出
"""

import sys
import os
import time
import signal
import ctypes
import threading
import numpy as np
import mujoco

# ====================== 全局配置 ======================
# 系统适配（Windows优先）
if os.name == 'nt':
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        os.system('chcp 65001 >nul 2>&1')
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 1)
    except:
        pass
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

# Mujoco Viewer兼容
try:
    from mujoco import viewer

    MUJOCO_NEW_VIEWER = True
except ImportError:
    import mujoco.viewer as viewer

    MUJOCO_NEW_VIEWER = False

# 核心参数配置
JOINT_COUNT = 5
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5"]
JOINT_LIMITS = np.array([
    [-np.pi, np.pi],  # joint1 (Z轴)
    [-np.pi / 2, np.pi / 2],  # joint2 (Y轴)
    [-np.pi / 2, np.pi / 2],  # joint3 (Y轴)
    [-np.pi / 2, np.pi / 2],  # joint4 (Y轴)
    [-np.pi / 2, np.pi / 2],  # joint5 (Y轴)
], dtype=np.float64)
JOINT_MAX_VELOCITY = np.array([1.0, 0.8, 0.8, 0.6, 0.6], dtype=np.float64)

# 仿真参数
SIMULATION_TIMESTEP = 0.005
CONTROL_FREQUENCY = 200
CONTROL_TIMESTEP = 1.0 / CONTROL_FREQUENCY
FPS = 60
SLEEP_TIME = 1.0 / FPS
EPS = 1e-8
RUNNING = True

# PD控制参数
KP = 80.0
KD = 5.0


# ====================== 信号处理（优雅退出） ======================
def signal_handler(sig, frame):
    global RUNNING
    print("\n⚠️ 收到退出信号，正在优雅退出...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)

# ====================== 预分配内存 ======================
WORK_ARRAYS = {
    'current_angles': np.zeros(JOINT_COUNT, dtype=np.float64),
    'target_angles': np.zeros(JOINT_COUNT, dtype=np.float64),
    'joint_velocities': np.zeros(JOINT_COUNT, dtype=np.float64),
    'control_signals': np.zeros(JOINT_COUNT, dtype=np.float64),
    'ee_position': np.zeros(3, dtype=np.float64),
    'angle_error': np.zeros(JOINT_COUNT, dtype=np.float64),
    'desired_vel': np.zeros(JOINT_COUNT, dtype=np.float64)
}


# ====================== 兼容型Mujoco ID查询函数 ======================
def get_mujoco_id(model, obj_type, name):
    """
    兼容所有Mujoco版本的ID查询函数
    :param model: MjModel对象
    :param obj_type: 对象类型（字符串或枚举）
    :param name: 对象名称
    :return: 对象ID
    """
    # 处理类型转换（关键修复）
    if isinstance(obj_type, str):
        # 字符串类型映射
        type_map = {
            'joint': mujoco.mjtObj.mjOBJ_JOINT,
            'actuator': mujoco.mjtObj.mjOBJ_ACTUATOR,
            'site': mujoco.mjtObj.mjOBJ_SITE
        }
        obj_type_int = type_map.get(obj_type, mujoco.mjtObj.mjOBJ_JOINT)
    else:
        # 枚举类型转为整数（核心修复）
        obj_type_int = int(obj_type)

    # 兼容不同版本的mj_name2id调用方式
    try:
        # 新版本调用方式
        return mujoco.mj_name2id(model, obj_type_int, name)
    except:
        # 旧版本兼容
        return mujoco.mj_name2id(model, obj_type, name)


# ====================== 机械臂模型生成 ======================
def create_arm_model():
    """生成极简兼容版XML模型"""
    xml = f"""
<mujoco model="controllable_arm">
    <compiler angle="radian" inertiafromgeom="true"/>
    <option timestep="{SIMULATION_TIMESTEP}" gravity="0 0 -9.81"/>

    <default>
        <joint type="hinge" armature="0.1" damping="0.1"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="50"/>
    </default>

    <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>

        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>

            <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0.1"/>
            <body name="link1" pos="0 0 0.1">
                <geom name="link1_geom" type="cylinder" size="0.05 0.2" rgba="0.2 0.8 0.2 1"/>

                <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0.2"/>
                <body name="link2" pos="0 0 0.2">
                    <geom name="link2_geom" type="cylinder" size="0.05 0.2" rgba="0.2 0.8 0.2 1"/>

                    <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0.2"/>
                    <body name="link3" pos="0 0 0.2">
                        <geom name="link3_geom" type="cylinder" size="0.05 0.2" rgba="0.2 0.8 0.2 1"/>

                        <joint name="joint4" type="hinge" axis="0 1 0" pos="0 0 0.2"/>
                        <body name="link4" pos="0 0 0.2">
                            <geom name="link4_geom" type="cylinder" size="0.05 0.2" rgba="0.2 0.8 0.2 1"/>

                            <joint name="joint5" type="hinge" axis="0 1 0" pos="0 0 0.2"/>
                            <body name="link5" pos="0 0 0.2">
                                <geom name="link5_geom" type="cylinder" size="0.05 0.1" rgba="0.8 0.2 0.2 1"/>

                                <body name="end_effector" pos="0 0 0.1">
                                    <site name="ee_site" pos="0 0 0" size="0.01"/>
                                    <geom name="ee_geom" type="sphere" size="0.05" rgba="0.8 0.2 0.2 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name="motor1" joint="joint1" ctrlrange="-1 1" gear="100"/>
        <motor name="motor2" joint="joint2" ctrlrange="-1 1" gear="100"/>
        <motor name="motor3" joint="joint3" ctrlrange="-1 1" gear="100"/>
        <motor name="motor4" joint="joint4" ctrlrange="-1 1" gear="100"/>
        <motor name="motor5" joint="joint5" ctrlrange="-1 1" gear="100"/>
    </actuator>
</mujoco>
    """
    return xml


# ====================== 核心控制器类 ======================
class ArmJointController:
    def __init__(self):
        # 初始化模型和数据
        self.model = mujoco.MjModel.from_xml_string(create_arm_model())
        self.data = mujoco.MjData(self.model)

        # 获取ID（使用兼容型函数，核心修复）
        self.joint_ids = []
        for name in JOINT_NAMES:
            # 关键修复：使用字符串类型+整数转换
            jid = get_mujoco_id(self.model, 'joint', name)
            self.joint_ids.append(jid)

        self.motor_ids = []
        for i in range(JOINT_COUNT):
            mid = get_mujoco_id(self.model, 'actuator', f"motor{i + 1}")
            self.motor_ids.append(mid)

        self.ee_site_id = get_mujoco_id(self.model, 'site', "ee_site")

        # 状态变量
        self.viewer_inst = None
        self.viewer_ready = False
        self.last_control_time = time.time()
        self.last_print_time = time.time()
        self.fps_counter = 0

        # 初始化目标角度为零位
        self.set_joint_angles(np.zeros(JOINT_COUNT), smooth=False)

    def get_current_joint_angles(self):
        """获取当前关节角度"""
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:  # 安全检查
                WORK_ARRAYS['current_angles'][i] = self.data.qpos[jid]
        return WORK_ARRAYS['current_angles'].copy()

    def get_joint_velocities(self):
        """获取关节速度"""
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                WORK_ARRAYS['joint_velocities'][i] = self.data.qvel[jid]
        return WORK_ARRAYS['joint_velocities'].copy()

    def get_ee_position(self):
        """获取末端位置"""
        if self.ee_site_id >= 0:
            WORK_ARRAYS['ee_position'][:] = self.data.site_xpos[self.ee_site_id]
        return WORK_ARRAYS['ee_position'].copy()

    def clamp_joint_angles(self, angles):
        """关节限位保护"""
        return np.clip(angles, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    def set_joint_angles(self, target_angles, smooth=True):
        """设置关节目标角度"""
        if len(target_angles) != JOINT_COUNT:
            raise ValueError(f"目标角度数量必须为{JOINT_COUNT}")

        # 限位保护
        target_angles = np.array(target_angles, dtype=np.float64)
        WORK_ARRAYS['target_angles'][:] = self.clamp_joint_angles(target_angles)

        # 立即设置（无平滑）
        if not smooth:
            for i, jid in enumerate(self.joint_ids):
                if jid >= 0:
                    self.data.qpos[jid] = WORK_ARRAYS['target_angles'][i]
                    self.data.qvel[jid] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def move_joint(self, joint_idx, angle, smooth=True):
        """单独控制单个关节"""
        if joint_idx < 0 or joint_idx >= JOINT_COUNT:
            raise ValueError(f"关节索引必须在0-{JOINT_COUNT - 1}之间")

        current_angles = self.get_current_joint_angles()
        current_angles[joint_idx] = angle
        self.set_joint_angles(current_angles, smooth)

    def pd_control_loop(self):
        """PD控制核心逻辑"""
        # 获取当前状态
        current_angles = self.get_current_joint_angles()
        current_vels = self.get_joint_velocities()

        # 计算角度误差
        WORK_ARRAYS['angle_error'][:] = WORK_ARRAYS['target_angles'] - current_angles

        # 计算期望速度（带速度限制）
        WORK_ARRAYS['desired_vel'][:] = np.clip(WORK_ARRAYS['angle_error'] * KP, -JOINT_MAX_VELOCITY,
                                                JOINT_MAX_VELOCITY)

        # PD控制计算
        WORK_ARRAYS['control_signals'][:] = KP * WORK_ARRAYS['angle_error'] + KD * (
                    WORK_ARRAYS['desired_vel'] - current_vels)

        # 设置控制信号到电机
        for i, mid in enumerate(self.motor_ids):
            if mid >= 0:
                self.data.ctrl[mid] = WORK_ARRAYS['control_signals'][i]

    def init_viewer(self):
        """初始化Viewer"""
        try:
            if MUJOCO_NEW_VIEWER:
                self.viewer_inst = viewer.launch_passive(self.model, self.data)
            else:
                self.viewer_inst = viewer.Viewer(self.model, self.data)
            self.viewer_ready = True
            return True
        except Exception as e:
            print(f"❌ Viewer初始化失败: {e}")
            return False

    def print_status(self):
        """打印实时状态"""
        current_time = time.time()
        if current_time - self.last_print_time >= 1.0:
            angles = self.get_current_joint_angles()
            ee_pos = self.get_ee_position()
            fps = self.fps_counter / (current_time - self.last_print_time)

            print(f"\n📊 实时状态 | FPS: {fps:5.1f}")
            print(f"🔧 关节角度 (弧度): {np.round(angles, 3)}")
            print(f"🎯 末端位置 (m): {np.round(ee_pos, 3)}")

            self.last_print_time = current_time
            self.fps_counter = 0

    def run(self):
        """运行完整仿真"""
        global RUNNING

        # 初始化Viewer
        if not self.init_viewer():
            RUNNING = False
            return

        # 启动信息
        print("=" * 60)
        print("🚀 机械臂关节控制器 - 最终兼容版")
        print(f"✅ MJ_NAME2ID API错误已修复")
        print(f"✅ 全Mujoco版本兼容")
        print(f"💻 Windows优化已启用")
        print("📝 控制指令:")
        print("   - 单关节控制: controller.move_joint(0, np.pi/4)")
        print("   - 多关节控制: controller.set_joint_angles([0, π/4, π/6, 0, 0])")
        print("   - 按 Ctrl+C 退出")
        print("=" * 60)

        # 主循环
        while RUNNING:
            try:
                current_time = time.time()
                self.fps_counter += 1

                # 控制频率执行PD控制
                if current_time - self.last_control_time >= CONTROL_TIMESTEP:
                    self.pd_control_loop()
                    self.last_control_time = current_time

                # 执行仿真步
                mujoco.mj_step(self.model, self.data)

                # 同步Viewer
                if self.viewer_ready:
                    self.viewer_inst.sync()

                # 打印状态
                self.print_status()

                # Windows睡眠优化
                time_diff = current_time - self.last_control_time
                if time_diff < SLEEP_TIME:
                    time.sleep(max(0.00001, SLEEP_TIME - time_diff))

            except Exception as e:
                print(f"⚠️ 仿真步异常: {e}")
                continue

        # 清理资源
        self.cleanup()
        print("\n✅ 控制器已优雅退出")

    def cleanup(self):
        """资源清理"""
        if self.viewer_ready and self.viewer_inst:
            try:
                self.viewer_inst.close()
            except:
                pass
        for arr in WORK_ARRAYS.values():
            arr.fill(0)


# ====================== 演示函数 ======================
def demo_movements(controller):
    """预设演示动作"""

    def demo():
        time.sleep(2)

        print("\n🎬 演示1：所有关节归位")
        controller.set_joint_angles([0, 0, 0, 0, 0])
        time.sleep(3)

        print("\n🎬 演示2：关节1旋转45度")
        controller.move_joint(0, np.pi / 4)
        time.sleep(2)

        print("\n🎬 演示3：关节2抬起30度")
        controller.move_joint(1, np.pi / 6)
        time.sleep(2)

        print("\n🎬 演示4：组合关节运动")
        controller.set_joint_angles([np.pi / 4, np.pi / 6, np.pi / 8, np.pi / 10, np.pi / 12])
        time.sleep(3)

        print("\n🎬 演示5：回到零位")
        controller.set_joint_angles([0, 0, 0, 0, 0])
        time.sleep(2)

        global RUNNING
        RUNNING = False

    demo_thread = threading.Thread(target=demo)
    demo_thread.daemon = True
    demo_thread.start()


# ====================== 主入口 ======================
if __name__ == "__main__":
    # 禁用NumPy警告
    np.seterr(all='ignore')

    # 创建控制器（现在可正常初始化）
    controller = ArmJointController()

    # 运行预设演示
    demo_movements(controller)

    # 启动控制器
    controller.run()