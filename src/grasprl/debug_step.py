"""单次抓取 demo — 动态测量 finger-to-EE offset，一次到位抓取。
关键：不用硬编码 FINGER_DX/DY/DZ，而是实时测量 ee 与 finger 的世界坐标差值，
然后用这个 offset 反推 EE 应该去哪里才能让手指对准物体。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import mujoco

from grasprl.envs.grasp import GraspRobot, _left_finger_name, _right_finger_name

env = GraspRobot(render_mode="human", frame_skip=5)
env.reset()

LEFT_ACT = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_finger_act')
RIGHT_ACT = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_finger_act')
print(f"\nactuator ID: left={LEFT_ACT}, right={RIGHT_ACT}, total={env.model.nu}")

# ── helpers ──
def _finger_mid():
    return (env.get_body_com(_left_finger_name) + env.get_body_com(_right_finger_name)) / 2

def _measure_offset():
    """测量当前姿态下 EE 到手指中心的世界坐标系偏移量。"""
    ee = env.get_ee_pos()
    fm = _finger_mid()
    return ee - fm   # offset = ee - finger，所以 ee_target = finger_target + offset

def _arm_move(target_ee):
    """IK 移动到目标，清零速度防止 NaN。"""
    env._move_eef_ik(target_ee)
    for jn in env.arm_joints_names:
        jid = env.find('joint', jn)
        env.data.qvel[env.model.jnt_dofadr[jid]] = 0.0
    env._sync_arm_ctrl()
    mujoco.mj_forward(env.model, env.data)

def _wait(msg, sec=1.0):
    print(f"    >> {msg} (等待{sec}s)...", flush=True)
    t0 = time.time()
    while time.time() - t0 < sec:
        env._try_render()
        time.sleep(0.02)

# ── 打开夹爪 ──
_gripper_joints = [
    'left_inner_knuckle_joint', 'left_outer_knuckle_joint', 'left_finger_joint',
    'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'right_finger_joint',
]
def _open_gripper():
    for jn in _gripper_joints:
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            env.data.qpos[env.model.jnt_qposadr[jid]] = 0.0
    if LEFT_ACT >= 0:  env.data.ctrl[LEFT_ACT] = 0.0
    if RIGHT_ACT >= 0: env.data.ctrl[RIGHT_ACT] = 0.0
    for _ in range(20):
        env._sim_step()

# ── 放置物体 ──
best_name = "ball_1"
PLACE_X, PLACE_Y = 0.05, 0.25
print(f"\n--- 将 {best_name} 放在桌上 ({PLACE_X}, {PLACE_Y}) ---")
jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, best_name + '_x')
if jnt_id >= 0:
    qaddr = env.model.jnt_qposadr[jnt_id]
    env.data.qpos[qaddr]     = PLACE_X - 0.1
    env.data.qpos[qaddr + 1] = PLACE_Y - 0.1
    for i in range(2):
        env.model.dof_damping[env.model.jnt_dofadr[jnt_id] + i] = 1000.0
mujoco.mj_forward(env.model, env.data)

target = env.get_body_com(best_name).copy()
target[2] = max(target[2], env.TABLE_HEIGHT + 0.02)
print(f"  {best_name} 位置: {target.round(4)}")

ee_start = env.get_ee_pos()
print(f"\n{'='*60}")
print(f"目标: {best_name} @ {target.round(4)}")
print(f"EE起始: {ee_start.round(4)}")
print(f"{'='*60}")

# ═══════════════════ Phase 1: 打开夹爪 ═══════════════════
print("\n--- Phase 1: 打开夹爪 ---")
_open_gripper()
print(f"  手指距: {env.get_finger_dist():.4f}")
_wait("夹爪已打开", 1.0)

# ═══════════════════ Phase 2: 移到物体正上方 (粗定位) ═══════════════════
# 用一个大概的估计先过去，然后测量真实 offset
print("\n--- Phase 2: 移到物体上方 ---")
rough_eef = np.array([target[0], target[1], target[2] + 0.20])  # 先假设 EE≈手指
rough_eef[2] = max(rough_eef[2], env.TABLE_HEIGHT + 0.22)
print(f"  粗略 EE 目标: {rough_eef.round(4)}")
_arm_move(rough_eef)
env._try_render()

# ★ 关键：在上方位置测量真实的 finger-to-EE 偏移量
real_offset = _measure_offset()
print(f"  ★ 实测 offset (EE-finger): [{real_offset[0]:.4f}, {real_offset[1]:.4f}, {real_offset[2]:.4f}]")
fm_now = _finger_mid()
print(f"  当前 EE={env.get_ee_pos().round(4)}, finger={fm_now.round(4)}, Δxy={np.linalg.norm(fm_now[:2]-target[:2]):.4f}")
_wait("已到达上方", 0.5)

# ═══════════════════ Phase 3: 用实测 offset 精准下降 ═══════════════════
print("\n--- Phase 3: 用实测 offset 下降 ---")
# 目标：手指在物体正上方 2cm 处（避免碰撞）
finger_above_target = target.copy()
finger_above_target[2] += 0.02  # 手指在物体表面上方 2cm

# 用实测 offset 计算 EE 需要去的位置
descent_eef = finger_above_target + real_offset
descent_eef[2] = max(descent_eef[2], env.TABLE_HEIGHT + 0.08)
print(f"  手指目标: {finger_above_target.round(4)}")
print(f"  EE目标 (finger+offset): {descent_eef.round(4)}")

# 分 3 段缓慢下降（减少推球风险）
start_descent = env.get_ee_pos().copy()
n_desc = 3
for seg_i in range(1, n_desc + 1):
    alpha = seg_i / n_desc
    wp = start_descent + alpha * (descent_eef - start_descent)
    _arm_move(wp)
    env._try_render()
    time.sleep(0.02)

# 检查手指位置
fm_final = _finger_mid()
obj_now = env.get_body_com(best_name)
delta = fm_final - target
print(f"\n  最终 finger={fm_final.round(4)} obj={obj_now.round(4)}")
print(f"  误差: xy={np.linalg.norm(delta[:2]):.4f} z={delta[2]:.4f}")
_wait("手指就位", 1.0)

# ═══════════════════ Phase 4: 合拢夹爪 ═══════════════════
print("\n--- Phase 4: 合拢夹爪 ---")
obj_before = env.get_body_com(best_name).copy()
print(f"  物体位置: {obj_before.round(4)}")

# 清零臂速度
for jn in env.arm_joints_names:
    jid = env.find('joint', jn)
    env.data.qvel[env.model.jnt_dofadr[jid]] = 0.0
env._sync_arm_ctrl()
mujoco.mj_forward(env.model, env.data)

hold_ee = env.get_ee_pos().copy()
ee_quat = np.zeros(4)
mujoco.mju_mat2Quat(ee_quat, env.data.site_xmat[env.eef_site].copy().flatten())
hold_pose = hold_ee.tolist() + ee_quat.tolist()

GRASP_STEPS = 80
for step_i in range(GRASP_STEPS):
    env.controller.run(hold_pose)
    progress = min(1.0, step_i / max(1, GRASP_STEPS // 2))
    # 直接改 qpos
    for jn in ['left_outer_knuckle_joint', 'right_outer_knuckle_joint']:
        jid = env.find('joint', jn)
        if jid >= 0:
            qaddr = env.model.jnt_qposadr[jid]
            current = env.data.qpos[qaddr]
            env.data.qpos[qaddr] = current + (0.943 - current) * min(0.2, progress * 0.2)
    ctrl_val = 1.0 * progress
    if LEFT_ACT >= 0:  env.data.ctrl[LEFT_ACT] = ctrl_val
    if RIGHT_ACT >= 0: env.data.ctrl[RIGHT_ACT] = ctrl_val
    env._sanitize_physics_data()
    env._sim_step()

    if step_i % 20 == 0:
        fo = env.get_body_com(best_name)
        fd = env.get_finger_dist()
        print(f"  close{step_i}: fd={fd:.4f} obj=({fo[0]:.3f},{fo[1]:.3f},{fo[2]:.4f}) ctrl={ctrl_val:.3f}")

for _ in range(30):
    env.controller.run(hold_pose)
    if LEFT_ACT >= 0:  env.data.ctrl[LEFT_ACT] = 1.0
    if RIGHT_ACT >= 0: env.data.ctrl[RIGHT_ACT] = 1.0
    env._sanitize_physics_data()
    env._sim_step()

env._try_render()
obj_after = env.get_body_com(best_name)
z_change = obj_after[2] - obj_before[2]
print(f"  抓取后: {obj_after.round(4)}, z变化: {z_change:.4f}, 手指距: {env.get_finger_dist():.4f}")
_wait("夹爪已合拢", 1.0)

# ═══════════════════ Phase 5: 抬起 ═══════════════════
print("\n--- Phase 5: 抬起物体 ---")
obj_before_lift = env.get_body_com(best_name).copy()
ee_now = env.get_ee_pos().copy()
ee_lift = ee_now.copy()
ee_lift[2] += env.LIFT_HEIGHT
print(f"  EE {ee_now.round(4)} → {ee_lift.round(4)}")

n_seg = 5
for seg in range(1, n_seg + 1):
    alpha = seg / n_seg
    wp = ee_now + alpha * (ee_lift - ee_now)
    _arm_move(wp)
    env._try_render()
    time.sleep(0.03)

obj_lifted = env.get_body_com(best_name)
z_rise = obj_lifted[2] - obj_before_lift[2]
print(f"  抬升后: {obj_lifted.round(4)}, 升高: {z_rise*1000:.1f}mm")

if z_rise > 0.003:
    print(f"\n  *** 抓取成功! (升高 {z_rise*1000:.1f}mm) ***\n")
    _wait("抬起成功", 2.0)

    print("\n--- Phase 6: 放下 ---")
    drop = np.array(env.drop_area)
    drop[2] = max(drop[2], env.TABLE_HEIGHT + 0.3)
    start_d = env.get_ee_pos().copy()
    for seg in range(1, 5):
        alpha = seg / 4
        wp = start_d + alpha * (drop - start_d)
        _arm_move(wp)
        env._try_render()
        time.sleep(0.02)

    _open_gripper()
    for _ in range(40):
        env._sim_step()
    env._try_render()
    _wait("已丢下", 2.0)
else:
    print(f"\n  --- 抓取失败 (仅升高 {z_rise*1000:.1f}mm) ---\n")
    _wait("失败", 2.0)

print(f"\n{'='*60}")
print("  完成! 按 Ctrl+C 退出...")
print(f"{'='*60}")

try:
    while True:
        env._try_render()
        time.sleep(0.02)
except KeyboardInterrupt:
    pass
env.close()
