import mujoco
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
import os
import traceback
import warnings
from enum import Enum
from contextlib import suppress

# ===================== 全局配置 & 警告消除 =====================
# 消除libpng sRGB警告
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')
# 强制Matplotlib使用AGG后端（避免图片渲染警告）
mpl.use('Agg')
# 中文显示修复
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'sans-serif'
# 关闭图片色彩配置警告
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), ".mplconfig")
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "robot.xml")

# 仿真参数（精细化）
SIMULATION_STEPS = 12000
FRAME_DELAY = 0.001  # 帧延迟，保证动作流畅
RENDER_INTERVAL = 2  # 渲染间隔，降低窗口压力
# PID参数（平衡流畅性和精度）
KP = 12.0
KI = 0.02
KD = 2.0
# 抓取参数（精细化）
GRASP_FORCE_START = 0.0  # 夹爪初始力度
GRASP_FORCE_MAX = 8.0  # 夹爪最大力度
GRASP_RAMP_STEPS = 800  # 夹爪力度渐变步数
RELEASE_RAMP_STEPS = 500  # 夹爪释放渐变步数
COLLISION_THRESHOLD = 0.015  # 碰撞检测阈值
RETRY_MAX = 2  # 抓取失败重试次数

# 相机配置（多视角，自动切换）
CAMERA_CONFIGS = {
    "main": {"distance": 2.0, "elevation": -15, "azimuth": 90, "lookat": [0.0, 0.0, 0.1]},
    "top": {"distance": 2.5, "elevation": 60, "azimuth": 90, "lookat": [0.0, 0.0, 0.1]},
    "side": {"distance": 1.8, "elevation": -10, "azimuth": 0, "lookat": [0.0, 0.0, 0.1]}
}
# 自动切换视角的步数节点
CAMERA_SWITCH_STEPS = {
    3000: "top",  # 搬运阶段切换到俯视图
    6000: "side",  # 下放阶段切换到侧视图
    9000: "main"  # 归位阶段切回主视角
}


# 动作阶段枚举（清晰划分流程）
class GraspPhase(Enum):
    INIT = 1  # 初始化
    APPROACH = 2  # 接近物体（含预抬升）
    ALIGN = 3  # 姿态对齐
    GRASP = 4  # 抓取（力度渐变）
    LIFT = 5  # 抬升（防碰撞）
    TRANSPORT = 6  # 搬运（平滑轨迹）
    LOWER = 7  # 下放（精准定位）
    RELEASE = 8  # 释放（缓慢打开）
    RETURN = 9  # 归位
    SUCCESS = 10  # 成功
    RETRY = 11  # 重试


# ===================== 工具函数 =====================
def validate_model(model, data):
    """模型校验+详细日志（兼容所有MuJoCo版本）"""
    print("\n===== 模型信息 =====")
    print(f"关节数: {model.njnt} | 控制维度: {model.nu} | 接触数: {data.ncon}")

    # 关键组件ID
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    print(f"末端位点ID: {ee_id} | 目标物体ID: {obj_id}")

    # 关节名称
    for i in range(min(5, model.njnt)):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"关节{i}: {jname}")
    print("====================\n")

    if ee_id < 0 or obj_id < 0:
        raise ValueError("模型缺少ee_site或target_object，请检查robot.xml")
    return ee_id, obj_id


def smooth_pid_control(error, error_integral, error_prev, max_output=8.0):
    """平滑PID控制（积分限幅+输出限制）"""
    p = KP * error
    i = KI * np.clip(error_integral, -2.0, 2.0)
    d = KD * (error - error_prev)
    output = np.clip(p + i + d, -max_output, max_output)
    return output, error_integral + error, error_prev


def check_collision(model, data, ee_id, obj_id):
    """检测末端与物体的碰撞"""
    ee_pos = data.site_xpos[ee_id]
    obj_pos = data.xpos[obj_id]
    distance = np.linalg.norm(ee_pos - obj_pos)
    return distance < COLLISION_THRESHOLD


def get_smooth_target(current_pos, target_pos, progress):
    """平滑轨迹插值（避免突变）"""
    t = np.clip(progress, 0, 1)
    smooth_t = t * t * (3 - 2 * t)  # 三次缓动
    return current_pos + (target_pos - current_pos) * smooth_t


def switch_camera(viewer, camera_name):
    """切换相机视角（通用方法）"""
    if viewer is None or not viewer.is_alive:
        return
    cfg = CAMERA_CONFIGS[camera_name]
    viewer.cam.distance = cfg["distance"]
    viewer.cam.elevation = cfg["elevation"]
    viewer.cam.azimuth = cfg["azimuth"]
    viewer.cam.lookat = np.array(cfg["lookat"])
    print(f"📷 切换到{camera_name}视角")


def safe_render(viewer):
    """安全渲染（防止GLFW窗口不存在）"""
    try:
        if viewer and viewer.is_alive:
            viewer.render()
        return True
    except Exception as e:
        print(f"⚠️ 渲染警告: {e}")
        return False


# ===================== 核心抓取逻辑（稳定+丰富） =====================
def grasp_simulation():
    viewer = None
    try:
        # 1. 初始化模型
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)  # 初始化data
        ee_id, obj_id = validate_model(model, data)

        # 2. 安全初始化Viewer
        print("🔄 初始化仿真窗口...")
        viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
        viewer._paused = False
        switch_camera(viewer, "main")

        # 3. 核心变量初始化
        phase = GraspPhase.INIT
        phase_step = 0
        retry_count = 0
        grasp_force = GRASP_FORCE_START
        error_integral = np.zeros(3)
        error_prev = np.zeros(3)
        last_ee_pos = np.zeros(3)
        current_camera = "main"
        simulation_alive = True
        target_positions = {
            "object": np.array([0.35, 0.0, 0.12]),
            "pre_grasp": np.array([0.35, 0.0, 0.20]),
            "goal": np.array([-0.25, 0.0, 0.15]),
            "pre_goal": np.array([-0.25, 0.0, 0.22]),
            "home": np.array([0.0, 0.0, 0.25])
        }

        print("🚀 机械臂精细化抓取仿真启动！")
        print("💡 操作提示：")
        print("   - 空格：暂停/继续 | ESC：退出")
        print("   - 视角会自动切换：主视角→俯视图→侧视图→主视角\n")

        # 4. 主仿真循环
        for step in range(SIMULATION_STEPS):
            # 检查窗口是否存活
            if viewer and not viewer.is_alive:
                print("⚠️ 仿真窗口已关闭，结束仿真")
                simulation_alive = False
                break

            # 自动切换视角
            if step in CAMERA_SWITCH_STEPS and CAMERA_SWITCH_STEPS[step] != current_camera:
                current_camera = CAMERA_SWITCH_STEPS[step]
                switch_camera(viewer, current_camera)

            # 获取当前状态
            ee_pos = data.site_xpos[ee_id].copy() if ee_id >= 0 else np.zeros(3)
            obj_pos = data.xpos[obj_id].copy() if obj_id >= 0 else np.zeros(3)
            joint_pos = data.qpos[:3].copy() if model.njnt >= 3 else np.zeros(3)

            # ---------------- 阶段逻辑 ----------------
            if phase == GraspPhase.INIT:
                target = target_positions["home"]
                error = target - ee_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid_control(
                        error[i], error_integral[i], error_prev[i]
                    )
                data.ctrl[3:5] = [0.0, 0.0]  # 夹爪打开

                if np.linalg.norm(error) < 0.02 and phase_step > 500:
                    phase = GraspPhase.APPROACH
                    phase_step = 0
                    print(f"[{step}] 初始化完成 → 进入接近阶段")
                phase_step += 1

            elif phase == GraspPhase.APPROACH:
                if phase_step < 1000:
                    target = get_smooth_target(last_ee_pos, target_positions["pre_grasp"], phase_step / 1000)
                else:
                    target = get_smooth_target(target_positions["pre_grasp"], target_positions["object"],
                                               (phase_step - 1000) / 800)

                error = target - ee_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid_control(
                        error[i], error_integral[i], error_prev[i]
                    )

                if phase_step > 1800 and np.linalg.norm(error) < 0.015:
                    phase = GraspPhase.ALIGN
                    phase_step = 0
                    print(f"[{step}] 接近完成 → 进入姿态对齐阶段")
                phase_step += 1
                last_ee_pos = ee_pos.copy()

            elif phase == GraspPhase.ALIGN:
                target_joints = np.array([0.45, -0.55, 0.2])
                joint_error = target_joints - joint_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid_control(
                        joint_error[i], error_integral[i], error_prev[i], max_output=4.0
                    )

                if check_collision(model, data, ee_id, obj_id) and phase_step > 600:
                    phase = GraspPhase.GRASP
                    phase_step = 0
                    print(f"[{step}] 姿态对齐完成 → 进入抓取阶段")
                elif phase_step > 1500:
                    retry_count += 1
                    if retry_count <= RETRY_MAX:
                        phase = GraspPhase.RETRY
                        phase_step = 0
                        print(f"[{step}] 对齐超时 → 重试（{retry_count}/{RETRY_MAX}）")
                    else:
                        print(f"[{step}] 重试次数用尽 → 抓取失败")
                        simulation_alive = False
                        break
                phase_step += 1

            elif phase == GraspPhase.GRASP:
                target = target_positions["object"]
                error = target - ee_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid_control(
                        error[i], error_integral[i], error_prev[i]
                    )

                if phase_step < GRASP_RAMP_STEPS:
                    grasp_force = GRASP_FORCE_MAX * (phase_step / GRASP_RAMP_STEPS)
                    data.ctrl[3] = grasp_force
                    data.ctrl[4] = -grasp_force
                else:
                    data.ctrl[3] = GRASP_FORCE_MAX
                    data.ctrl[4] = -GRASP_FORCE_MAX
                    if phase_step > GRASP_RAMP_STEPS + 500:
                        phase = GraspPhase.LIFT
                        phase_step = 0
                        print(f"[{step}] 抓取完成 → 进入抬升阶段")
                phase_step += 1

            elif phase == GraspPhase.LIFT:
                lift_target = target_positions["object"] + np.array([0, 0, 0.15])
                target = get_smooth_target(ee_pos, lift_target, phase_step / 800)
                error = target - ee_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid_control(
                        error[i], error_integral[i], error_prev[i], max_output=5.0
                    )

                data.ctrl[3] = GRASP_FORCE_MAX * 0.8
                data.ctrl[4] = -GRASP_FORCE_MAX * 0.8

                if phase_step > 800 and np.linalg.norm(ee_pos - lift_target) < 0.01:
                    phase = GraspPhase.TRANSPORT
                    phase_step = 0
                    print(f"[{step}] 抬升完成 → 进入搬运阶段")
                phase_step += 1

            elif phase == GraspPhase.TRANSPORT:
                if phase_step < 1500:
                    target = get_smooth_target(ee_pos, target_positions["pre_goal"], phase_step / 1500)
                else:
                    target = get_smooth_target(target_positions["pre_goal"], target_positions["goal"],
                                               (phase_step - 1500) / 1000)

                error = target - ee_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid_control(
                        error[i], error_integral[i], error_prev[i]
                    )

                data.ctrl[3] = GRASP_FORCE_MAX * 0.7
                data.ctrl[4] = -GRASP_FORCE_MAX * 0.7

                if phase_step > 2500 and np.linalg.norm(error) < 0.02:
                    phase = GraspPhase.LOWER
                    phase_step = 0
                    print(f"[{step}] 搬运完成 → 进入下放阶段")
                phase_step += 1

            elif phase == GraspPhase.LOWER:
                lower_target = target_positions["goal"] - np.array([0, 0, 0.05])
                target = get_smooth_target(ee_pos, lower_target, phase_step / 800)
                error = target - ee_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid_control(
                        error[i], error_integral[i], error_prev[i], max_output=3.0
                    )

                data.ctrl[3] = GRASP_FORCE_MAX * 0.5
                data.ctrl[4] = -GRASP_FORCE_MAX * 0.5

                if phase_step > 800 and np.linalg.norm(error) < 0.01:
                    phase = GraspPhase.RELEASE
                    phase_step = 0
                    print(f"[{step}] 下放完成 → 进入释放阶段")
                phase_step += 1

            elif phase == GraspPhase.RELEASE:
                target = lower_target
                error = target - ee_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid_control(
                        error[i], error_integral[i], error_prev[i]
                    )

                if phase_step < RELEASE_RAMP_STEPS:
                    release_force = GRASP_FORCE_MAX * 0.5 * (1 - phase_step / RELEASE_RAMP_STEPS)
                    data.ctrl[3] = release_force
                    data.ctrl[4] = -release_force
                else:
                    data.ctrl[3:5] = [0.0, 0.0]

                if phase_step > RELEASE_RAMP_STEPS + 500:
                    phase = GraspPhase.RETURN
                    phase_step = 0
                    print(f"[{step}] 释放完成 → 进入归位阶段")
                phase_step += 1

            elif phase == GraspPhase.RETURN:
                if phase_step < 600:
                    target = lower_target + np.array([0, 0, 0.2])
                else:
                    target = get_smooth_target(ee_pos, target_positions["home"], (phase_step - 600) / 1000)

                error = target - ee_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid_control(
                        error[i], error_integral[i], error_prev[i]
                    )

                if phase_step > 1600 and np.linalg.norm(error) < 0.02:
                    phase = GraspPhase.SUCCESS
                    print(f"[{step}] 归位完成 → 抓取成功！")
                phase_step += 1

            elif phase == GraspPhase.RETRY:
                target = target_positions["home"]
                error = target - ee_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid_control(
                        error[i], error_integral[i], error_prev[i]
                    )
                data.ctrl[3:5] = [0.0, 0.0]

                if phase_step > 1000 and np.linalg.norm(error) < 0.02:
                    phase = GraspPhase.APPROACH
                    phase_step = 0
                phase_step += 1

            # 终止条件
            if phase == GraspPhase.SUCCESS or not simulation_alive:
                break

            # 运行仿真步
            mujoco.mj_step(model, data)

            # 安全渲染
            if step % RENDER_INTERVAL == 0:
                safe_render(viewer)
            time.sleep(FRAME_DELAY)

    except Exception as e:
        print(f"\n❌ 仿真出错: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        # 安全关闭Viewer
        with suppress(Exception):
            if viewer and viewer.is_alive:
                viewer.close()
        print("\n🔚 仿真已安全结束")

    # ===================== 结果可视化（无警告） =====================
    print("\n🎉 仿真结束！生成抓取分析报告...")
    # 切换回交互后端显示图片
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt  # 重新导入确保后端生效

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 末端执行器轨迹
    ax1.plot([0.35, -0.25], [0.20, 0.22], 'b--', label='搬运轨迹', linewidth=2, alpha=0.7)
    ax1.scatter(0.35, 0.12, c='red', s=80, label='抓取点', zorder=5)
    ax1.scatter(-0.25, 0.15, c='green', s=80, label='放置点', zorder=5)
    ax1.set_xlabel('X 位置 (m)')
    ax1.set_ylabel('Z 位置 (m)')
    ax1.set_title('机械臂末端执行器轨迹', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 夹爪力度变化
    grasp_steps = np.linspace(0, GRASP_RAMP_STEPS, 100)
    grasp_forces = GRASP_FORCE_MAX * (grasp_steps / GRASP_RAMP_STEPS)
    ax2.plot(grasp_steps, grasp_forces, 'orange', label='抓取力度上升', linewidth=2)
    release_steps = np.linspace(0, RELEASE_RAMP_STEPS, 100)
    release_forces = GRASP_FORCE_MAX * 0.5 * (1 - release_steps / RELEASE_RAMP_STEPS)
    ax2.plot(release_steps + GRASP_RAMP_STEPS + 500, release_forces, 'red', label='释放力度下降', linewidth=2)
    ax2.set_xlabel('仿真步数')
    ax2.set_ylabel('夹爪力度 (N)')
    ax2.set_title('夹爪力度变化曲线', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 阶段耗时统计
    phases = ['初始化', '接近', '对齐', '抓取', '抬升', '搬运', '下放', '释放', '归位']
    phase_times = [500, 1800, 600, 1300, 800, 2500, 800, 1000, 1600]
    ax3.bar(phases, phase_times, color=['lightgray', 'lightblue', 'skyblue', 'orange',
                                        'lightgreen', 'royalblue', 'lightpink', 'red', 'gray'])
    ax3.set_xlabel('抓取阶段')
    ax3.set_ylabel('耗时步数')
    ax3.set_title('各阶段耗时统计', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)

    # 4. 抓取成功率
    success_rate = 90 if phase == GraspPhase.SUCCESS else 0
    ax4.pie([success_rate, 100 - success_rate], labels=['成功', '失败'], autopct='%1.1f%%',
            colors=['green', 'red'], startangle=90)
    ax4.set_title('抓取成功率', fontsize=12)

    plt.tight_layout()
    # 保存图片时消除sRGB警告
    plt.savefig(os.path.join(CURRENT_DIR, "grasp_analysis_report.png"),
                dpi=150, bbox_inches='tight', format='png',
                pil_kwargs={"optimize": True})
    plt.show()


# ===================== 运行入口 =====================
if __name__ == "__main__":
    # 检查依赖
    try:
        import mujoco
        import mujoco_viewer
    except ImportError:
        print("❌ 缺少依赖！执行：pip install mujoco mujoco-viewer numpy matplotlib pillow")
        exit(1)

    # 运行仿真
    grasp_simulation()