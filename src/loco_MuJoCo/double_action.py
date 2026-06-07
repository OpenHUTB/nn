import numpy as np
import jax
import mujoco
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid

# 配置参数
FPS = int(1 / 0.002)  # 根据环境实际dt计算（假设dt=0.002）
WALK_DURATION = 5
STAY_DURATION = 2
SQUAT_REPEATS = 5
WALK_STEPS = WALK_DURATION * FPS
STAY_STEPS = STAY_DURATION * FPS

def get_trajectory_segment(env, dataset_name, num_steps):
    """从数据集提取指定长度的轨迹片段"""
    temp_env = ImitationFactory.make(
        "UnitreeH1",
        default_dataset_conf=DefaultDatasetConf([dataset_name]),
        n_substeps=env._n_substeps
    )
    temp_env.reset(jax.random.PRNGKey(0))
    # 加载轨迹（使用 th.traj 访问轨迹数据）
    temp_env.load_trajectory(temp_env.th.traj)
    # 从轨迹数据中获取关节位置和速度
    full_qpos = temp_env.th.traj.data.qpos
    full_qvel = temp_env.th.traj.data.qvel
    
    qpos = []
    qvel = []
    for i in range(num_steps):
        idx = i % len(full_qpos)
        qpos.append(full_qpos[idx])
        qvel.append(full_qvel[idx])
    return np.array(qpos), np.array(qvel)

# 创建主环境
env = ImitationFactory.make(
    "UnitreeH1",
    n_substeps=20,
    default_dataset_conf=DefaultDatasetConf(["walk", "squat"])
)
env.reset(jax.random.PRNGKey(0))
model = env.get_model()
data = env.get_data()

# 提取各阶段轨迹
walk_qpos, walk_qvel = get_trajectory_segment(env, "walk", WALK_STEPS)

# 停留阶段轨迹
last_walk_qpos = walk_qpos[-1]
last_walk_qvel = np.zeros_like(walk_qvel[0])
stay_qpos = np.tile(last_walk_qpos, (STAY_STEPS, 1))
stay_qvel = np.tile(last_walk_qvel, (STAY_STEPS, 1))

# 下蹲阶段轨迹
squat_temp_env = ImitationFactory.make(
    "UnitreeH1",
    default_dataset_conf=DefaultDatasetConf(["squat"]),
    n_substeps=env._n_substeps
)
squat_temp_env.reset(jax.random.PRNGKey(0))
squat_cycle_steps = len(squat_temp_env.th.traj.data.qpos)  # 修正：使用 th.traj 访问
squat_qpos, squat_qvel = get_trajectory_segment(
    env, "squat", squat_cycle_steps * SQUAT_REPEATS
)

# 固定下蹲时的根位置
root_joint_ind = mj_jntname2qposid("root", model)
root_pos_walk_end = walk_qpos[-1, root_joint_ind[:2]]
squat_qpos[:, root_joint_ind[:2]] = root_pos_walk_end

# 拼接轨迹
full_qpos = np.concatenate([walk_qpos, stay_qpos, squat_qpos], axis=0)
full_qvel = np.concatenate([walk_qvel, stay_qvel, squat_qvel], axis=0)

# 创建并加载轨迹
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

# 播放轨迹
env.play_trajectory(n_steps_per_episode=len(full_qpos), render=True)