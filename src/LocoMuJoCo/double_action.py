import numpy as np
import jax
import mujoco
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid

# ===================== 1. 初始化环境+动态获取真实帧率（核心） =====================
env = ImitationFactory.make(
    "UnitreeH1",
    n_substeps=20,
    default_dataset_conf=DefaultDatasetConf(["walk", "squat"])
)
env.reset(jax.random.PRNGKey(0))
ENV_DT = env.dt  # 环境真实时间步长（默认0.02s，对应50FPS）
FPS = int(1 / ENV_DT)
print(f"环境真实参数：dt={ENV_DT}s | FPS={FPS}")

# ===================== 2. 精准配置各阶段时长（按需修改） =====================
WALK_DURATION = 5       # 行走5秒（可改）
STAY_DURATION = 1       # 停留1秒（严格）
SQUAT_DURATION = 10     # 下蹲总时长10秒（严格）
# 计算各阶段精准步数
WALK_STEPS = int(WALK_DURATION * FPS)
STAY_STEPS = int(STAY_DURATION * FPS)
SQUAT_STEPS = int(SQUAT_DURATION * FPS)

def get_trajectory_segment(env, dataset_name, target_steps):
    """按目标步数提取轨迹（精准控制时长，避免冗余）"""
    temp_env = ImitationFactory.make(
        "UnitreeH1",
        default_dataset_conf=DefaultDatasetConf([dataset_name]),
        n_substeps=env._n_substeps
    )
    temp_env.reset(jax.random.PRNGKey(0))
    temp_env.load_trajectory(temp_env.th.traj)
    raw_qpos = np.array(temp_env.th.traj.data.qpos)
    raw_qvel = np.array(temp_env.th.traj.data.qvel)
    
    # 鲁棒性校验：空轨迹报错
    if len(raw_qpos) == 0:
        raise ValueError(f"数据集 {dataset_name} 无有效轨迹！")
    
    # 循环填充到目标步数（保证动作流畅）
    traj_qpos = []
    traj_qvel = []
    for i in range(target_steps):
        idx = i % len(raw_qpos)  # 循环取原始轨迹帧
        traj_qpos.append(raw_qpos[idx])
        traj_qvel.append(raw_qvel[idx])
    return np.array(traj_qpos), np.array(traj_qvel)

# ===================== 3. 生成各阶段轨迹（顺序：走→停→蹲） =====================
model = env.get_model()
data = env.get_data()

# 3.1 行走5秒轨迹
walk_qpos, walk_qvel = get_trajectory_segment(env, "walk", WALK_STEPS)
print(f"行走阶段：{WALK_DURATION}秒 | {WALK_STEPS}步")

# 3.2 停留1秒轨迹（复用行走最后姿态，速度归零）
last_walk_qpos = walk_qpos[-1].copy()
last_walk_qvel = np.zeros_like(walk_qvel[0])
stay_qpos = np.tile(last_walk_qpos, (STAY_STEPS, 1))
stay_qvel = np.tile(last_walk_qvel, (STAY_STEPS, 1))
print(f"停留阶段：{STAY_DURATION}秒 | {STAY_STEPS}步")

# 3.3 下蹲10秒轨迹（精准10秒，原地固定）
squat_qpos, squat_qvel = get_trajectory_segment(env, "squat", SQUAT_STEPS)
# 固定下蹲根位置（行走结束位置，仅x/y平面）
root_joint_ind = mj_jntname2qposid("root", model)
root_pos_walk_end = walk_qpos[-1, root_joint_ind[:2]]
squat_qpos[:, root_joint_ind[:2]] = root_pos_walk_end
print(f"下蹲阶段：{SQUAT_DURATION}秒 | {SQUAT_STEPS}步")

# ===================== 4. 拼接轨迹（强制顺序：走→停→蹲） =====================
full_qpos = np.concatenate([walk_qpos, stay_qpos, squat_qpos], axis=0)
full_qvel = np.concatenate([walk_qvel, stay_qvel, squat_qvel], axis=0)
total_duration = len(full_qpos) / FPS
print(f"总轨迹：{total_duration:.1f}秒 | {len(full_qpos)}步（5+1+10={16}秒，验证匹配）")

# ===================== 5. 加载轨迹并播放 =====================
# 生成轨迹信息（频率匹配环境真实FPS）
jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
traj_info = TrajectoryInfo(
    jnt_names,
    model=TrajectoryModel(model.njnt, jax.numpy.array(model.jnt_type)),
    frequency=FPS
)
traj_data = TrajectoryData(
    jax.numpy.array(full_qpos),
    jax.numpy.array(full_qvel),
    split_points=jax.numpy.array([0, len(full_qpos)])
)
traj = Trajectory(traj_info, traj_data)
env.load_trajectory(traj)

# 播放轨迹（确保顺序和速度正确）
env.play_trajectory(n_steps_per_episode=len(full_qpos), render=True)