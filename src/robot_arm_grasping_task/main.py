import mujoco
import mujoco_viewer
import numpy as np
import os
import warnings
import time
from contextlib import suppress

# ===================== 配置 =====================
warnings.filterwarnings('ignore')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "robot.xml")

# --- 任务清单 ---
TASK_QUEUE = [
    ["target_object", [-0.3, 0, 0.05]],
]

# --- 核心控制参数 (精细化) ---
IK_GAIN = 1.2
CLEARANCE_HEIGHT = 0.25
STEP_PER_MOVE = 1500

# --- 抓取力控制参数 (关键) ---
GRIPPER_OPEN_FORCE = 8.0  # 张开夹爪的力
GRIPPER_GRASP_FORCE = -12.0  # 抓取时尝试闭合的力
GRIPPER_HOLD_FORCE = -4.0  # 抓住物体后保持的力
CONTACT_THRESHOLD = 1.0  # 接触力阈值 (N)，超过此值认为已接触

# ===================== 全局状态机 =====================
viewer = None
current_task_index = 0
task_step = 0
grasp_force_applied = 0.0  # 记录当前施加的抓取力


class TaskState:
    MOVE_TO_OBJECT_ABOVE = 1
    PRE_GRASP_OPEN = 2
    MOVE_DOWN_TO_GRASP = 3
    GRASP_OBJECT = 4
    HOLD_OBJECT = 5
    MOVE_UP_AFTER_GRASP = 6
    MOVE_TO_TARGET_ABOVE = 7
    MOVE_DOWN_TO_PLACE = 8
    RELEASE_OBJECT = 9
    MOVE_UP_AFTER_RELEASE = 10
    FINISHED_ALL = 11


current_state = TaskState.MOVE_TO_OBJECT_ABOVE


# ===================== 辅助函数 =====================
def get_contact_force(model, data, geom1_name, geom2_name):
    """获取两个geom之间的接触力大小"""
    geom1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_name)
    geom2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_name)

    for i in range(data.ncon):
        con = data.contact[i]
        if (con.geom1 == geom1_id and con.geom2 == geom2_id) or \
                (con.geom1 == geom2_id and con.geom2 == geom1_id):
            # contact force is in data.cfrc_ext, but we need to compute it
            # For simplicity, we can use the normal force magnitude
            return abs(con.force[0])
    return 0.0


def simple_ik_control(model, data, ee_id, target_pos):
    """逆运动学控制"""
    current_pos = data.site_xpos[ee_id]
    error = target_pos - current_pos
    error = np.clip(error, -0.05, 0.05)

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, None, current_pos, ee_id)
    jnt_vel = np.dot(jacp[:, :3].T, error * IK_GAIN)
    jnt_vel = np.clip(jnt_vel, -0.5, 0.5)

    for i in range(min(3, model.nu)):
        data.ctrl[i] = jnt_vel[i] * 100


# ===================== 复杂抓取任务逻辑 =====================
def run_complex_grasp_task(model, data, ee_id):
    global current_task_index, task_step, current_state, grasp_force_applied

    if current_task_index >= len(TASK_QUEUE):
        if current_state != TaskState.FINISHED_ALL:
            print("\n🎉🎉🎉 所有复杂抓取任务已成功完成！🎉🎉🎉")
            current_state = TaskState.FINISHED_ALL
        return False

    obj_name, target_place_pos = TASK_QUEUE[current_task_index]
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)

    if obj_id == -1:
        print(f"❌ 错误：未在模型中找到物体 '{obj_name}'。")
        current_task_index += 1
        return True

    # --- 状态机逻辑 ---
    if current_state == TaskState.MOVE_TO_OBJECT_ABOVE:
        if task_step == 0: print("\n-> 状态: 移动到物体上方安全高度...")
        target_pos = data.xpos[obj_id].copy()
        target_pos[2] = CLEARANCE_HEIGHT
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.01:
            task_step = 0
            current_state = TaskState.PRE_GRASP_OPEN

    elif current_state == TaskState.PRE_GRASP_OPEN:
        if task_step == 0: print("-> 状态: 预抓取 - 张开夹爪...")
        # 张开夹爪
        data.ctrl[3] = GRIPPER_OPEN_FORCE
        data.ctrl[4] = -GRIPPER_OPEN_FORCE
        if task_step > 500:  # 等待夹爪完全张开
            task_step = 0
            current_state = TaskState.MOVE_DOWN_TO_GRASP

    elif current_state == TaskState.MOVE_DOWN_TO_GRASP:
        if task_step == 0: print("-> 状态: 下降接近物体...")
        target_pos = data.xpos[obj_id].copy()
        target_pos[2] += 0.04  # 非常接近物体表面
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.005:
            task_step = 0
            current_state = TaskState.GRASP_OBJECT

    elif current_state == TaskState.GRASP_OBJECT:
        if task_step == 0: print("-> 状态: 抓取 - 快速闭合夹爪...")
        # 快速闭合夹爪
        data.ctrl[3] = GRIPPER_GRASP_FORCE
        data.ctrl[4] = -GRIPPER_GRASP_FORCE
        # 检测是否接触到物体
        left_contact_force = get_contact_force(model, data, "gripper_left_geom", "target_geom")
        right_contact_force = get_contact_force(model, data, "gripper_right_geom", "target_geom")
        if left_contact_force > CONTACT_THRESHOLD or right_contact_force > CONTACT_THRESHOLD:
            print(f"   [接触检测] 检测到接触力！切换到保持模式。")
            grasp_force_applied = GRIPPER_HOLD_FORCE
            task_step = 0
            current_state = TaskState.HOLD_OBJECT

    elif current_state == TaskState.HOLD_OBJECT:
        if task_step == 0: print("-> 状态: 保持 - 施加稳定夹持力...")
        # 施加稳定的保持力
        data.ctrl[3] = grasp_force_applied
        data.ctrl[4] = -grasp_force_applied
        # 等待一小段时间确保抓稳
        if task_step > 500:
            task_step = 0
            current_state = TaskState.MOVE_UP_AFTER_GRASP

    elif current_state == TaskState.MOVE_UP_AFTER_GRASP:
        if task_step == 0: print("-> 状态: 抓取成功，上升...")
        target_pos = data.site_xpos[ee_id].copy()
        target_pos[2] = CLEARANCE_HEIGHT
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.01:
            task_step = 0
            current_state = TaskState.MOVE_TO_TARGET_ABOVE

    # ... (MOVE_TO_TARGET_ABOVE, MOVE_DOWN_TO_PLACE 状态与之前类似) ...
    elif current_state == TaskState.MOVE_TO_TARGET_ABOVE:
        if task_step == 0: print(f"-> 状态: 移动到目标放置区上方 {target_place_pos[:2]}...")
        target_pos = np.array(target_place_pos)
        target_pos[2] = CLEARANCE_HEIGHT
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.01:
            task_step = 0
            current_state = TaskState.MOVE_DOWN_TO_PLACE

    elif current_state == TaskState.MOVE_DOWN_TO_PLACE:
        if task_step == 0: print("-> 状态: 下降到放置位置...")
        target_pos = np.array(target_place_pos)
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.005:
            task_step = 0
            current_state = TaskState.RELEASE_OBJECT

    elif current_state == TaskState.RELEASE_OBJECT:
        if task_step == 0: print("-> 状态: 释放 - 打开夹爪...")
        # 完全松开夹爪
        data.ctrl[3] = GRIPPER_OPEN_FORCE
        data.ctrl[4] = -GRIPPER_OPEN_FORCE
        if task_step > 800:  # 等待夹爪完全打开
            task_step = 0
            current_state = TaskState.MOVE_UP_AFTER_RELEASE

    elif current_state == TaskState.MOVE_UP_AFTER_RELEASE:
        if task_step == 0: print("-> 状态: 释放成功，上升...")
        target_pos = data.site_xpos[ee_id].copy()
        target_pos[2] = CLEARANCE_HEIGHT
        simple_ik_control(model, data, ee_id, target_pos)
        if np.linalg.norm(data.site_xpos[ee_id] - target_pos) < 0.01:
            current_task_index += 1
            task_step = 0
            current_state = TaskState.MOVE_TO_OBJECT_ABOVE

    task_step += 1
    return True


# ===================== 主程序 (与之前版本类似) =====================
def init():
    global viewer
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"请确保 'robot.xml' 文件在当前目录: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90
    viewer.cam.lookat = [0.2, 0.0, 0.1]

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    if ee_id == -1:
        raise ValueError("模型中必须包含一个名为 'ee_site' 的site。")

    print("=" * 60)
    print("🚀 全自动复杂柔顺抓取系统启动！")
    print(f"📋 任务清单: 共 {len(TASK_QUEUE)} 个物体需要处理。")
    print("💡 系统将使用接触感知进行智能抓取。")
    print("=" * 60)
    return model, data, ee_id


def main():
    global viewer
    try:
        model, data, ee_id = init()

        while viewer.is_alive:
            if not run_complex_grasp_task(model, data, ee_id):
                break

            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.005)

        print("\n⏳ 所有任务已完成，窗口将在5秒后自动关闭。")
        for _ in range(5):
            viewer.render()
            time.sleep(1)

    except Exception as e:
        print(f"\n❌ 程序发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with suppress(Exception):
            viewer.close()
        print("🔚 程序已退出。")


if __name__ == "__main__":
    try:
        import mujoco, mujoco_viewer
    except ImportError:
        print("❌ 缺少依赖！请运行: pip install mujoco mujoco-viewer numpy")
        exit(1)
    main()