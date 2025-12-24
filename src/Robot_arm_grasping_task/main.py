import mujoco
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
import os
import warnings
import traceback
from enum import Enum
from contextlib import suppress

# ===================== 基础配置（消除警告） =====================
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')
mpl.use('Agg')
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'sans-serif'
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), ".mplconfig")
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "robot.xml")

# ===================== 核心参数（流畅大动作） =====================
# 仿真参数（平衡流畅度和幅度）
SIMULATION_STEPS = 18000  # 足够的步数完成大动作
FRAME_DELAY = 0.003  # 流畅无卡顿
RENDER_INTERVAL = 1  # 每步渲染，看清细节
# PID参数（精准控制，无超调）
KP = 3.5  # 比例增益：足够驱动大动作，又不超调
KI = 0.008  # 积分增益：微小积分消除稳态误差
KD = 0.8  # 微分增益：抑制震荡
# 抓取参数（流畅力度变化）
GRASP_FORCE_MAX = 5.0  # 夹爪力度适中
GRASP_RAMP_STEPS = 1500  # 1500步闭合，动作流畅
RELEASE_RAMP_STEPS = 1200  # 1200步打开，避免物体掉落
# 轨迹参数（大幅度、平滑）
LIFT_HEIGHT = 0.2  # 抬升幅度大
TRANSPORT_DISTANCE = 0.4  # 搬运幅度大
MOVE_SMOOTH_FACTOR = 0.002  # 轨迹平滑因子


# 动作阶段枚举
class GraspPhase(Enum):
    INIT = 1  # 初始化（初始位姿）
    APPROACH = 2  # 接近物体（大距离移动）
    GRASP = 3  # 抓取
    LIFT = 4  # 抬升（大幅度）
    TRANSPORT = 5  # 搬运（大距离）
    LOWER = 6  # 下放
    RELEASE = 7  # 释放
    RETURN = 8  # 归位（大动作返回）
    SUCCESS = 9  # 成功


# ===================== 工具函数（精准控制） =====================
def validate_model(model, data):
    """校验模型，确保关键组件存在"""
    print("\n===== 模型信息 =====")
    print(f"关节数: {model.njnt} | 控制维度: {model.nu} | 接触数: {data.ncon}")

    # 关键ID（兼容模型）
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    if ee_id < 0: ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee")
    if obj_id < 0: obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")

    print(f"末端ID: {ee_id} | 物体ID: {obj_id}")
    print("====================\n")

    if ee_id < 0 or obj_id < 0:
        raise ValueError("请确保robot.xml包含ee_site和target_object")
    return ee_id, obj_id


def smooth_pid(error, integral, prev_error, max_output=3.0):
    """平滑PID控制，无超调"""
    p = KP * error
    i = KI * np.clip(integral, -1.5, 1.5)
    d = KD * (error - prev_error) / (FRAME_DELAY * 2)
    output = np.clip(p + i + d, -max_output, max_output)
    return output, integral + error, prev_error


def get_smooth_target(current, target, step, total_steps):
    """平滑轨迹插值，大动作无突变"""
    t = np.clip(step / total_steps, 0, 1)
    # 五次缓动曲线：start→end 全程平滑
    smooth_t = t * t * t * (t * (6 * t - 15) + 10)
    return current + (target - current) * smooth_t


def check_grasp_stable(model, data, obj_id, ee_pos):
    """检测抓取是否稳定（物体跟随末端）"""
    obj_pos = data.xpos[obj_id]
    distance = np.linalg.norm(obj_pos - ee_pos)
    return distance < 0.03  # 物体与末端距离近，抓取稳定


# ===================== 核心抓取逻辑（大动作+流畅） =====================
def grasp_simulation():
    viewer = None
    phase = GraspPhase.INIT
    try:
        # 1. 加载模型
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        ee_id, obj_id = validate_model(model, data)

        # 2. 初始化Viewer（优化视角，看清大动作）
        viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
        viewer.cam.distance = 1.8  # 视角拉远，看清大动作
        viewer.cam.elevation = 15  # 仰角，看清抬升动作
        viewer.cam.azimuth = 60  # 方位角，看清搬运轨迹
        viewer.cam.lookat = [0.2, 0.0, 0.15]  # 聚焦动作区域

        # 3. 核心变量初始化
        phase_step = 0
        ee_pos = data.site_xpos[ee_id].copy() if ee_id >= 0 else np.array([0.0, 0.0, 0.1])
        obj_init_pos = data.xpos[obj_id].copy() if obj_id >= 0 else np.array([0.3, 0.0, 0.05])

        # 大幅度目标轨迹规划
        target_pos = {
            "home": np.array([0.0, 0.0, 0.15]),  # 初始位姿
            "pre_grasp": obj_init_pos + [0, 0, 0.08],  # 预抓取位置（物体上方）
            "grasp": obj_init_pos,  # 抓取位置
            "lift": obj_init_pos + [0, 0, LIFT_HEIGHT],  # 抬升位置（大幅度）
            "transport": obj_init_pos + [TRANSPORT_DISTANCE, 0, LIFT_HEIGHT],  # 搬运位置（大距离）
            "lower": obj_init_pos + [TRANSPORT_DISTANCE, 0, 0.05],  # 下放位置
            "return_mid": np.array([0.2, 0.0, 0.2])  # 归位中间点（大动作过渡）
        }
        print("🎯 大动作轨迹规划完成：")
        for k, v in target_pos.items():
            print(f"  {k}: {v}")

        # PID控制变量
        error_integral = np.zeros(3)
        error_prev = np.zeros(3)
        grasp_force = 0.0

        print("\n🚀 机械臂大幅度流畅抓取仿真启动！")
        print("💡 动作流程：初始位姿→大距离接近→抓取→大幅度抬升→远距离搬运→下放→释放→大动作归位\n")

        # 4. 主仿真循环（大动作+流畅）
        for step in range(SIMULATION_STEPS):
            if viewer and not viewer.is_alive:
                print("⚠️ 窗口关闭，结束仿真")
                break

            # 获取当前状态
            ee_pos = data.site_xpos[ee_id].copy() if ee_id >= 0 else ee_pos
            obj_pos = data.xpos[obj_id].copy() if obj_id >= 0 else obj_init_pos

            # ---------------- 阶段1：初始化（初始位姿，大动作归位） ----------------
            if phase == GraspPhase.INIT:
                target = target_pos["home"]
                error = target - ee_pos
                # 大动作归位，控制输出适中
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid(
                        error[i], error_integral[i], error_prev[i], max_output=2.5
                    )
                # 夹爪打开
                data.ctrl[3] = 0.0
                data.ctrl[4] = 0.0

                # 初始化完成：位置误差小，且等待足够步数
                if np.linalg.norm(error) < 0.008 and phase_step > 2000:
                    phase = GraspPhase.APPROACH
                    phase_step = 0
                    print(f"[{step}] 初始化完成 → 开始大距离接近物体")
                phase_step += 1

            # ---------------- 阶段2：接近物体（大距离移动，流畅） ----------------
            elif phase == GraspPhase.APPROACH:
                # 分两步：先到预抓取位置，再下降到抓取点（大动作）
                if phase_step < 3000:
                    target = get_smooth_target(ee_pos, target_pos["pre_grasp"], phase_step, 3000)
                else:
                    target = get_smooth_target(target_pos["pre_grasp"], target_pos["grasp"], phase_step - 3000, 2000)

                error = target - ee_pos
                # 大动作控制，输出稍大但无超调
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid(
                        error[i], error_integral[i], error_prev[i], max_output=3.0
                    )
                # 夹爪保持打开
                data.ctrl[3] = 0.0
                data.ctrl[4] = 0.0

                # 接近完成：到达抓取点，动作稳定
                if phase_step > 5000 and np.linalg.norm(error) < 0.01:
                    phase = GraspPhase.GRASP
                    phase_step = 0
                    print(f"[{step}] 大距离接近完成 → 开始抓取")
                phase_step += 1

            # ---------------- 阶段3：抓取（流畅闭合） ----------------
            elif phase == GraspPhase.GRASP:
                # 保持末端在抓取点
                target = target_pos["grasp"]
                error = target - ee_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid(
                        error[i], error_integral[i], error_prev[i], max_output=1.0
                    )

                # 夹爪流畅闭合
                if phase_step < GRASP_RAMP_STEPS:
                    grasp_force = GRASP_FORCE_MAX * (phase_step / GRASP_RAMP_STEPS)
                    data.ctrl[3] = grasp_force  # 夹爪1闭合
                    data.ctrl[4] = -grasp_force  # 夹爪2闭合
                else:
                    # 保持闭合，确认抓取稳定
                    data.ctrl[3] = GRASP_FORCE_MAX
                    data.ctrl[4] = -GRASP_FORCE_MAX
                    # 检测抓取稳定：物体跟随末端
                    if check_grasp_stable(model, data, obj_id, ee_pos) and phase_step > GRASP_RAMP_STEPS + 800:
                        phase = GraspPhase.LIFT
                        phase_step = 0
                        print(f"[{step}] 抓取稳定 → 开始大幅度抬升")

                phase_step += 1

            # ---------------- 阶段4：抬升（大幅度、流畅） ----------------
            elif phase == GraspPhase.LIFT:
                target = target_pos["lift"]
                error = target - ee_pos
                # 抬升控制：输出适中，动作流畅
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid(
                        error[i], error_integral[i], error_prev[i], max_output=2.0
                    )
                # 保持夹爪力度
                data.ctrl[3] = GRASP_FORCE_MAX * 0.9
                data.ctrl[4] = -GRASP_FORCE_MAX * 0.9

                # 抬升完成：到达目标高度，幅度明显
                if phase_step > 2000 and np.linalg.norm(error) < 0.01:
                    phase = GraspPhase.TRANSPORT
                    phase_step = 0
                    print(f"[{step}] 大幅度抬升完成 → 开始远距离搬运")
                phase_step += 1

            # ---------------- 阶段5：搬运（大距离、平滑） ----------------
            elif phase == GraspPhase.TRANSPORT:
                target = get_smooth_target(ee_pos, target_pos["transport"], phase_step, 3000)
                error = target - ee_pos
                # 搬运控制：平滑大动作
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid(
                        error[i], error_integral[i], error_prev[i], max_output=2.5
                    )
                # 保持夹爪力度，防止物体掉落
                data.ctrl[3] = GRASP_FORCE_MAX * 0.8
                data.ctrl[4] = -GRASP_FORCE_MAX * 0.8

                # 搬运完成：到达目标位置，大距离移动完成
                if phase_step > 3000 and np.linalg.norm(error) < 0.015:
                    phase = GraspPhase.LOWER
                    phase_step = 0
                    print(f"[{step}] 远距离搬运完成 → 开始下放")
                phase_step += 1

            # ---------------- 阶段6：下放（流畅） ----------------
            elif phase == GraspPhase.LOWER:
                target = target_pos["lower"]
                error = target - ee_pos
                # 下放控制：缓慢流畅
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid(
                        error[i], error_integral[i], error_prev[i], max_output=1.5
                    )
                # 降低夹爪力度，准备释放
                data.ctrl[3] = GRASP_FORCE_MAX * 0.6
                data.ctrl[4] = -GRASP_FORCE_MAX * 0.6

                if phase_step > 1500 and np.linalg.norm(error) < 0.01:
                    phase = GraspPhase.RELEASE
                    phase_step = 0
                    print(f"[{step}] 下放完成 → 开始释放物体")
                phase_step += 1

            # ---------------- 阶段7：释放（流畅打开） ----------------
            elif phase == GraspPhase.RELEASE:
                # 保持末端位置，避免物体掉落
                target = target_pos["lower"]
                error = target - ee_pos
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid(
                        error[i], error_integral[i], error_prev[i], max_output=1.0
                    )

                # 夹爪流畅打开
                if phase_step < RELEASE_RAMP_STEPS:
                    release_force = GRASP_FORCE_MAX * 0.6 * (1 - phase_step / RELEASE_RAMP_STEPS)
                    data.ctrl[3] = release_force
                    data.ctrl[4] = -release_force
                else:
                    data.ctrl[3] = 0.0
                    data.ctrl[4] = 0.0  # 完全打开

                if phase_step > RELEASE_RAMP_STEPS + 800:
                    phase = GraspPhase.RETURN
                    phase_step = 0
                    print(f"[{step}] 释放完成 → 开始大动作归位")
                phase_step += 1

            # ---------------- 阶段8：归位（大动作返回） ----------------
            elif phase == GraspPhase.RETURN:
                # 分两步归位：先到中间点，再回初始位姿（大动作）
                if phase_step < 2000:
                    target = get_smooth_target(ee_pos, target_pos["return_mid"], phase_step, 2000)
                else:
                    target = get_smooth_target(target_pos["return_mid"], target_pos["home"], phase_step - 2000, 2000)

                error = target - ee_pos
                # 归位控制：大动作流畅返回
                for i in range(min(3, model.njnt)):
                    data.ctrl[i], error_integral[i], error_prev[i] = smooth_pid(
                        error[i], error_integral[i], error_prev[i], max_output=2.5
                    )
                # 夹爪保持打开
                data.ctrl[3] = 0.0
                data.ctrl[4] = 0.0

                # 归位完成：回到初始位姿，整个流程结束
                if phase_step > 4000 and np.linalg.norm(error) < 0.01:
                    phase = GraspPhase.SUCCESS
                    print(f"[{step}] 大动作归位完成 → 整个抓取流程成功！")
                    break
                phase_step += 1

            # ---------------- 仿真步进 & 渲染 ----------------
            mujoco.mj_step(model, data)
            if viewer:
                try:
                    viewer.render()
                except:
                    pass
            time.sleep(FRAME_DELAY)

    except Exception as e:
        print(f"\n❌ 仿真出错: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        with suppress(Exception):
            if viewer and viewer.is_alive:
                viewer.close()
        print("\n🔚 仿真结束")

    # ===================== 结果可视化（大动作轨迹） =====================
    print("\n🎉 生成大动作抓取轨迹报告...")
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt

    # 绘制大动作轨迹图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 三维轨迹投影（X-Z平面，展示大动作）
    trajectory_x = [
        target_pos["home"][0],
        target_pos["grasp"][0],
        target_pos["lift"][0],
        target_pos["transport"][0],
        target_pos["lower"][0],
        target_pos["home"][0]
    ]
    trajectory_z = [
        target_pos["home"][2],
        target_pos["grasp"][2],
        target_pos["lift"][2],
        target_pos["transport"][2],
        target_pos["lower"][2],
        target_pos["home"][2]
    ]
    # 绘制轨迹（大动作明显）
    ax1.plot(trajectory_x, trajectory_z, 'b-o', linewidth=3, markersize=8, label='机械臂末端轨迹')
    ax1.scatter(target_pos["grasp"][0], target_pos["grasp"][2], c='red', s=150, label='抓取点', zorder=5)
    ax1.scatter(target_pos["lower"][0], target_pos["lower"][2], c='green', s=150, label='放置点', zorder=5)
    ax1.set_xlabel('X 位置 (m)', fontsize=12)
    ax1.set_ylabel('Z 位置 (m)', fontsize=12)
    ax1.set_title('机械臂大幅度抓取轨迹（X-Z平面）', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    # 标注动作阶段
    ax1.annotate('初始位姿', (target_pos["home"][0], target_pos["home"][2]),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax1.annotate('大幅度抬升', (target_pos["lift"][0], target_pos["lift"][2]),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax1.annotate('远距离搬运', (target_pos["transport"][0], target_pos["transport"][2]),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)

    # 2. 夹爪力度变化（流畅曲线）
    grasp_steps = np.linspace(0, GRASP_RAMP_STEPS, 100)
    grasp_forces = GRASP_FORCE_MAX * (grasp_steps / GRASP_RAMP_STEPS)
    release_steps = np.linspace(0, RELEASE_RAMP_STEPS, 100)
    release_forces = GRASP_FORCE_MAX * 0.6 * (1 - release_steps / RELEASE_RAMP_STEPS)

    ax2.plot(grasp_steps, grasp_forces, 'orange', linewidth=3, label='夹爪闭合（力度上升）')
    ax2.plot(release_steps + GRASP_RAMP_STEPS + 3000, release_forces,
             'red', linewidth=3, label='夹爪打开（力度下降）')
    ax2.axhline(y=GRASP_FORCE_MAX, color='gray', linestyle='--', alpha=0.7, label='最大力度')
    ax2.set_xlabel('仿真步数', fontsize=12)
    ax2.set_ylabel('夹爪力度 (N)', fontsize=12)
    ax2.set_title('夹爪力度流畅变化曲线', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, "big_move_grasp_report.png"),
                dpi=150, bbox_inches='tight', pil_kwargs={"optimize": True})
    plt.show()


# ===================== 运行入口 =====================
if __name__ == "__main__":
    try:
        import mujoco
        import mujoco_viewer
    except ImportError:
        print("❌ 缺少依赖！执行：pip install mujoco mujoco-viewer numpy matplotlib pillow")
        exit(1)

    grasp_simulation()