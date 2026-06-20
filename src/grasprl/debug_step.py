"""端到端抓取 Demo — 使用硬编码关节空间运动（绕过有bug的OSC控制器）。

流程：打开夹爪 → IK解算移动到物体上方 → 下降 → 闭合夹爪 → 抬起 → 移到放置区 → 打开夹爪
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import mujoco

from grasprl.envs.grasp import GraspRobot, _left_finger_name, _right_finger_name


# ==================== 初始化 ====================
env = GraspRobot(render_mode="human", frame_skip=5)
env.reset()

LEFT_ACT = env.left_finger_act
RIGHT_ACT = env.right_finger_act
print(f"actuator ID: left={LEFT_ACT}, right={RIGHT_ACT}, total nu={env.model.nu}")


# ==================== 辅助函数 ====================
def finger_mid():
    """手指中心位置"""
    return (env.get_body_com(_left_finger_name) + env.get_body_com(_right_finger_name)) / 2


def wait(msg, sec=1.0):
    print(f"  >> {msg} (等待{sec}s)", flush=True)
    t0 = time.time()
    # 保存当前关节位置
    hold_qpos = env.data.qpos[:6].copy()
    while time.time() - t0 < sec:
        # 强制保持当前关节位置，防止重力漂移
        env.data.qpos[:6] = hold_qpos
        # 清零臂关节速度
        for jn in env.arm_joints:
            vaddr = env.model.jnt_dofadr[jn]
            env.data.qvel[vaddr] = 0.0
        mujoco.mj_step(env.model, env.data)
        env._try_render()


# ==================== 放置一个物体 ====================
best_name = "ball_1"

# 策略：先把物体放在手指当前位置正下方
# 1. 先测量手指当前位置
env.reset()
finger_init = finger_mid().copy()
print(f"\n初始手指位置: {finger_init.round(4)}")

# 2. 把物体放在手指正下方（保持 X,Y，Z 放在桌面上）
PLACE_X = finger_init[0]
PLACE_Y = finger_init[1]
print(f"将物体放在: ({PLACE_X:.4f}, {PLACE_Y:.4f})")

print(f"\n--- 将 {best_name} 放在桌上 ({PLACE_X}, {PLACE_Y}) ---")
jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, best_name + '_x')
if jnt_id >= 0:
    qaddr = env.model.jnt_qposadr[jnt_id]
    # qpos 是相对于初始位置 (0.1, 0.1, 1.2) 的偏移
    env.data.qpos[qaddr]     = PLACE_X - 0.1   # x 偏移
    env.data.qpos[qaddr + 1] = PLACE_Y - 0.1   # y 偏移
    # z: 放在桌面上 (TABLE_HEIGHT + ball_radius)
    body_z = env.model.body_pos[
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, best_name)
    ][2]
    # ball_1 radius=0.035
    env.data.qpos[qaddr + 2] = env.TABLE_HEIGHT + 0.035 - body_z + 0.005

# 给物体增加阻尼防止被推走
for i in range(3):
    dof_adr = env.model.jnt_dofadr[jnt_id] + i
    env.model.dof_damping[dof_adr] = 10.0  # 增加阻尼到 10.0

mujoco.mj_forward(env.model, env.data)
for _ in range(20):
    env._sim_step()
env.data.qvel[:] = 0

target = env.get_body_com(best_name).copy()
target[2] = max(target[2], env.TABLE_HEIGHT + 0.04)
print(f"  {best_name} 位置: {target.round(4)}")

ee_start = env.get_ee_pos()
print(f"\n{'='*60}")
print(f"目标: {best_name} @ {target.round(4)}")
print(f"EE起始: {ee_start.round(4)}")
print(f"{'='*60}")


# ==================== Phase 1: 打开夹爪 ====================
print("\n--- Phase 1: 打开夹爪 ---")
env.open_gripper(steps=80)
print(f"  手指距: {env.get_finger_dist():.4f}")
wait("夹爪已打开", 1.0)


# ==================== Phase 2: 垂直下降到物体上方 ====================
print("\n--- Phase 2: 垂直下降到物体上方 ---")
# 策略：将物体直接移到手指下方，然后垂直下降

# 当前手指位置
finger_now = finger_mid().copy()
print(f"  当前手指: {finger_now.round(4)}")
print(f"  物体位置: {target.round(4)}")
print(f"  XY 误差: {np.linalg.norm(finger_now[:2] - target[:2]):.4f}")

# 关键：把物体移到手指正下方，确保手指垂直下降就能碰到
print(f"  将物体移到手指正下方...")
env.data.qpos[qaddr]     = finger_now[0] - 0.1
env.data.qpos[qaddr + 1] = finger_now[1] - 0.1
mujoco.mj_forward(env.model, env.data)
target = env.get_body_com(best_name).copy()
print(f"  新物体位置: {target.round(4)}")
print(f"  新 XY 误差: {np.linalg.norm(finger_now[:2] - target[:2]):.4f}")

# 直接垂直下降 - 使用当前 qpos，只调整 shoulder_lift
qpos_before = env.data.qpos[:6].copy()
print(f"  当前 qpos: {qpos_before.round(4)}")

qpos_grasp = qpos_before.copy()
qpos_grasp[1] -= 0.30  # shoulder_lift_joint 更向下
qpos_grasp[2] += 0.30  # elbow_joint 配合

# 验证
env.data.qpos[:6] = qpos_grasp
mujoco.mj_forward(env.model, env.data)
finger_check = finger_mid()
print(f"  验证手指: {finger_check.round(4)}, XY误差: {np.linalg.norm(finger_check[:2] - target[:2]):.4f}")
print(f"  手指 Z: {finger_check[2]:.4f}, 物体 Z: {target[2]:.4f}")

# 平滑下降
print("  平滑下降...")
env.move_joints_smooth(qpos_grasp, steps=100)

fm_final = finger_mid()
obj_now = env.get_body_com(best_name)
delta = fm_final - target
print(f"  最终 finger={fm_final.round(4)}, obj={obj_now.round(4)}")
print(f"  误差: xy={np.linalg.norm(delta[:2]):.4f}, z={delta[2]:.4f}")
print(f"  手指距: {env.get_finger_dist():.4f}")
wait("手指就位", 0.5)


# ==================== Phase 4: 闭合夹爪 ====================
print("\n--- Phase 4: 闭合夹爪 ---")
obj_before = env.get_body_com(best_name).copy()
print(f"  物体位置: {obj_before.round(4)}")

# 使用 close_gripper 方法（渐进闭合 + 保压）
env.close_gripper(target_val=0.95, steps=200)

obj_after = env.get_body_com(best_name).copy()
finger_dist = env.get_finger_dist()
print(f"  闭合后: obj={obj_after.round(4)}, 手指距={finger_dist:.4f}")

# 检测接触
contacts, left_contacts, right_contacts = env.get_finger_contacts()
print(f"  接触力: {contacts}")
print(f"  左手指接触: {left_contacts}, 右手指接触: {right_contacts}")

has_contact = (len(left_contacts) > 0 or len(right_contacts) > 0)
print(f"  有接触: {has_contact}")
wait("夹爪已闭合", 1.0)


# ==================== Phase 5: 抬起物体 ====================
print("\n--- Phase 5: 抬起物体 ---")
# 保持当前 X,Y，只抬升 Z
obj_before_lift = env.get_body_com(best_name).copy()
print(f"  物体位置: {obj_before_lift.round(4)}")

# 获取当前 EE 位置
ee_before = env.get_ee_pos().copy()
# 只抬升 Z 20cm
lift_eef = np.array([ee_before[0], ee_before[1], ee_before[2] + 0.20])
print(f"  EE {ee_before.round(4)} → {lift_eef.round(4)}")

# 求解 IK
print("  求解 IK...")
qpos_lift = env.solve_ik_numerical(lift_eef, max_iter=1000, tol=1e-5)

# 验证
env.data.qpos[:6] = qpos_lift
mujoco.mj_forward(env.model, env.data)
ee_check = env.get_ee_pos()
print(f"  验证 EE: {ee_check.round(4)}, 误差: {np.linalg.norm(ee_check - lift_eef):.4f}")

# 平滑抬起
print("  平滑抬起...")
env.move_joints_smooth(qpos_lift, steps=200)

obj_after_lift = env.get_body_com(best_name)
height_change = obj_after_lift[2] - obj_before_lift[2]
print(f"  抬升后: {obj_after_lift.round(4)}, 升高: {height_change*1000:.1f}mm")

if height_change > 0.02:  # 升高超过 2cm 算成功
    print(f"  ✅ 抓取成功! (物体升高 {height_change*1000:.1f}mm)")
else:
    print(f"  --- 抓取失败 (仅升高 {height_change*1000:.1f}mm) ---")

wait("完成抬起", 1.0)


# ==================== Phase 6: 移动到放置区 ====================
print("\n--- Phase 6: 移动到放置区 ---")
place_pos = np.array([-0.15, 0.15, target[2] + 0.25])  # 放置区
print(f"  目标位置: {place_pos.round(4)}")

# 求解 IK
print("  求解 IK...")
qpos_place = env.solve_ik_numerical(place_pos, max_iter=1000, tol=1e-5)

# 平滑移动
print("  平滑移动...")
env.move_joints_smooth(qpos_place, steps=300)
print(f"  到达放置区")
wait("已到达放置区", 1.0)


# ==================== Phase 7: 打开夹爪丢下物体 ====================
print("\n--- Phase 7: 打开夹爪丢下物体 ---")
obj_before_drop = env.get_body_com(best_name).copy()
print(f"  物体位置: {obj_before_drop.round(4)}")

# 完全打开夹爪
print("  打开夹爪...")
env.open_gripper(steps=100)
print(f"  手指距: {env.get_finger_dist():.4f}")

# 等待物体掉落
wait("等待物体掉落", 2.0)

obj_after_drop = env.get_body_com(best_name).copy()
drop_dist = np.linalg.norm(obj_after_drop[:2] - obj_before_drop[:2])
print(f"  物体移动: {drop_dist*1000:.1f}mm")

if drop_dist > 0.02:  # 物体移动超过 2cm
    print("  物体已释放!")
else:
    print("  物体可能还在夹爪中")
    
# 额外等待观察
wait("完成放置", 2.0)


# ==================== 完成 ====================
print(f"\n{'='*60}")
print("  完成! 按 Ctrl+C 退出...")
print(f"{'='*60}")

# 保持渲染窗口
try:
    while True:
        mujoco.mj_step(env.model, env.data)
        env._try_render()
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\n退出...")
