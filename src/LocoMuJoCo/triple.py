import numpy as np
import jax
import mujoco
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf, LAFAN1DatasetConf
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid

# ===================== 1. 初始化环境+获取帧率参数 =====================
env = ImitationFactory.make(
    "UnitreeH1",
    n_substeps=20,
    default_dataset_conf=DefaultDatasetConf(["walk", "squat"]),  # 行走和下蹲属于默认数据集
    lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"])   # 舞蹈属于LAFAN1数据集
)
env.reset(jax.random.PRNGKey(0))
ENV_DT = env.dt  # 环境时间步长（默认0.02s）
FPS = int(1 / ENV_DT)
print(f"环境参数：dt={ENV_DT}s | FPS={FPS}")

# ===================== 2. 配置各阶段时长及循环参数 =====================
WALK_DURATION = 8        # 行走8秒
STAY_DURATION = 1        # 停留1秒
SQUAT_DURATION = 10      # 下蹲10秒
DANCE_DURATION = 10      # 跳舞10秒
CYCLE_TIMES = 3          # 循环次数（可调整）

# 计算各阶段步数
WALK_STEPS = int(WALK_DURATION * FPS)
STAY_STEPS = int(STAY_DURATION * FPS)
SQUAT_STEPS = int(SQUAT_DURATION * FPS)
DANCE_STEPS = int(DANCE_DURATION * FPS)
SINGLE_CYCLE_STEPS = WALK_STEPS + STAY_STEPS + SQUAT_STEPS + DANCE_STEPS
TOTAL_STEPS = SINGLE_CYCLE_STEPS * CYCLE_TIMES

# ===================== 3. 轨迹片段生成函数 =====================
def get_trajectory_segment(env, dataset_type, dataset_name, target_steps):
    """获取指定数据集的轨迹片段并调整到目标步数"""
    if dataset_type == "default":
        temp_env = ImitationFactory.make(
            "UnitreeH1",
            default_dataset_conf=DefaultDatasetConf([dataset_name]),
            n_substeps=env._n_substeps
        )
    elif dataset_type == "lafan1":
        temp_env = ImitationFactory.make(
            "UnitreeH1",
            lafan1_dataset_conf=LAFAN1DatasetConf([dataset_name]),
            n_substeps=env._n_substeps
        )
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")

    temp_env.reset(jax.random.PRNGKey(0))
    raw_qpos = np.array(temp_env.th.traj.data.qpos)
    raw_qvel = np.array(temp_env.th.traj.data.qvel)
    
    if len(raw_qpos) == 0:
        raise ValueError(f"数据集 {dataset_name} 无有效轨迹数据！")
    
    # 循环填充轨迹至目标步数
    traj_qpos = [raw_qpos[i % len(raw_qpos)] for i in range(target_steps)]
    traj_qvel = [raw_qvel[i % len(raw_qvel)] for i in range(target_steps)]
    return np.array(traj_qpos), np.array(traj_qvel)

# ===================== 4. 预生成单周期轨迹片段 =====================
model = env.get_model()
root_joint_ind = mj_jntname2qposid("root", model)  # 获取根关节索引

# 4.1 行走阶段
walk_qpos, walk_qvel = get_trajectory_segment(
    env, "default", "walk", WALK_STEPS
)
print(f"行走阶段：{WALK_DURATION}秒 | {WALK_STEPS}步")

# 4.2 停留阶段（复用行走结束姿态，速度归零）
last_walk_qpos = walk_qpos[-1].copy()
last_walk_qvel = np.zeros_like(walk_qvel[0])
stay_qpos = np.tile(last_walk_qpos, (STAY_STEPS, 1))
stay_qvel = np.tile(last_walk_qvel, (STAY_STEPS, 1))
print(f"停留阶段：{STAY_DURATION}秒 | {STAY_STEPS}步")

# 4.3 下蹲阶段（固定根位置为行走结束位置）
squat_qpos, squat_qvel = get_trajectory_segment(
    env, "default", "squat", SQUAT_STEPS
)
root_pos_walk_end = walk_qpos[-1, root_joint_ind[:2]]  # 获取行走结束时的x/y位置
squat_qpos[:, root_joint_ind[:2]] = root_pos_walk_end   # 固定下蹲时的根位置
print(f"下蹲阶段：{SQUAT_DURATION}秒 | {SQUAT_STEPS}步")

# 4.4 跳舞阶段（固定根位置为行走结束位置）
dance_qpos, dance_qvel = get_trajectory_segment(
    env, "lafan1", "dance2_subject4", DANCE_STEPS
)
dance_qpos[:, root_joint_ind[:2]] = root_pos_walk_end    # 固定跳舞时的根位置
print(f"跳舞阶段：{DANCE_DURATION}秒 | {DANCE_STEPS}步")

# ===================== 5. 生成多循环完整轨迹 =====================
full_qpos = []
full_qvel = []

for cycle in range(CYCLE_TIMES):
    print(f"生成第 {cycle+1}/{CYCLE_TIMES} 个循环轨迹...")
    # 按顺序拼接单循环的四个阶段
    full_qpos.extend([walk_qpos, stay_qpos, squat_qpos, dance_qpos])
    full_qvel.extend([walk_qvel, stay_qvel, squat_qvel, dance_qvel])

# 合并为numpy数组
full_qpos = np.concatenate(full_qpos, axis=0)
full_qvel = np.concatenate(full_qvel, axis=0)

# 验证总时长
total_duration = len(full_qpos) / FPS
print(f"总轨迹：{CYCLE_TIMES}次循环 | {total_duration:.1f}秒 | {len(full_qpos)}步")

# ===================== 6. 加载并播放轨迹 =====================
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

# 播放完整轨迹
env.play_trajectory(n_steps_per_episode=len(full_qpos), render=True)