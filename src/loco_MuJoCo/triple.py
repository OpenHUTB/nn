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
    default_dataset_conf=DefaultDatasetConf(["walk", "squat"]),
    lafan1_dataset_conf=LAFAN1DatasetConf(["dance1_subject3"])  # 替换为dance1_subject3
)
env.reset(jax.random.PRNGKey(0))
ENV_DT = env.dt
FPS = int(1 / ENV_DT)
print(f"环境参数：dt={ENV_DT}s | FPS={FPS}")

# ===================== 2. 配置各阶段时长及循环参数 =====================
WALK_DURATION = 7        
STAY_DURATION = 1        
SQUAT_DURATION = 8      
DANCE_DURATION = 15      
CYCLE_TIMES = 2          

# 计算各阶段步数
WALK_STEPS = int(WALK_DURATION * FPS)
STAY_STEPS = int(STAY_DURATION * FPS)
SQUAT_STEPS = int(SQUAT_DURATION * FPS)
DANCE_STEPS = int(DANCE_DURATION * FPS)
SINGLE_CYCLE_STEPS = WALK_STEPS + STAY_STEPS + SQUAT_STEPS + DANCE_STEPS
TOTAL_STEPS = SINGLE_CYCLE_STEPS * CYCLE_TIMES

# ===================== 3. 轨迹片段生成与增强函数 =====================
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

def enhance_dance_movement(q_data, model, is_velocity=False, intensity=1.2, speed_factor=1.0):
    """
    增强舞蹈动作：放大关节运动幅度并调整速度
    :param q_data: qpos（位置）或 qvel（速度）数组
    :param model: mujoco模型
    :param is_velocity: 是否是速度数据（qvel）
    :param intensity: 动作幅度增强系数
    :param speed_factor: 速度调整系数
    :return: 增强后的轨迹数据
    """
    # 1. 先获取有效维度，避免索引越界
    max_dim = q_data.shape[1]  # 获取q_data的第二维大小（比如25）
    print(f"当前数据维度：{q_data.shape}，最大有效索引：{max_dim-1}")

    # 2. 识别关键运动关节（只保留模型中存在的关节）
    motion_joints = [
        "left_shoulder", "right_shoulder", 
        "left_elbow", "right_elbow",
        "left_hip", "right_hip",
        "left_knee", "right_knee"
    ]
    
    # 3. 安全获取关节索引（过滤掉不存在/越界的索引）
    joint_indices = []
    for jnt_name in motion_joints:
        try:
            # 获取关节索引（处理返回列表/单个值的情况）
            idx = mj_jntname2qposid(jnt_name, model)
            if isinstance(idx, list):
                joint_indices.extend(idx)
            else:
                joint_indices.append(idx)
        except:
            continue  # 忽略不存在的关节
    
    # 4. 过滤掉超出维度范围的索引（核心修复点）
    joint_indices = [idx for idx in joint_indices if idx < max_dim]
    print(f"有效关节索引：{joint_indices}")

    # 5. 增强关节运动幅度（只处理角度关节，跳过前7个根关节）
    enhanced_data = q_data.copy()
    for idx in joint_indices:
        if idx >= 7:  # 跳过根关节（位置/旋转），只增强肢体关节
            enhanced_data[:, idx] = q_data[:, idx] * intensity

    # 6. 速度调整（仅对位置数据qpos生效，避免qvel重复调整）
    if not is_velocity and speed_factor != 1.0:
        orig_length = len(enhanced_data)
        new_length = int(orig_length * speed_factor)
        # 线性插值重采样
        time_orig = np.linspace(0, 1, orig_length)
        time_new = np.linspace(0, 1, new_length)
        enhanced_data = np.array([
            np.interp(time_new, time_orig, enhanced_data[:, i]) 
            for i in range(enhanced_data.shape[1])
        ]).T

    return enhanced_data

def blend_trajectories(traj1, traj2, blend_ratio=0.5):
    """混合两个轨迹，实现平滑过渡"""
    min_len = min(len(traj1), len(traj2))
    return traj1[:min_len] * (1 - blend_ratio) + traj2[:min_len] * blend_ratio

# ===================== 4. 预生成单周期轨迹片段 =====================
model = env.get_model()
root_joint_ind = mj_jntname2qposid("root", model)

# 4.1 行走阶段
walk_qpos, walk_qvel = get_trajectory_segment(
    env, "default", "walk", WALK_STEPS
)
print(f"行走阶段：{WALK_DURATION}秒 | {WALK_STEPS}步")

# 4.2 停留阶段
last_walk_qpos = walk_qpos[-1].copy()
last_walk_qvel = np.zeros_like(walk_qvel[0])
stay_qpos = np.tile(last_walk_qpos, (STAY_STEPS, 1))
stay_qvel = np.tile(last_walk_qvel, (STAY_STEPS, 1))
print(f"停留阶段：{STAY_DURATION}秒 | {STAY_STEPS}步")

# 4.3 下蹲阶段
squat_qpos, squat_qvel = get_trajectory_segment(
    env, "default", "squat", SQUAT_STEPS
)
root_pos_walk_end = walk_qpos[-1, root_joint_ind[:2]]
squat_qpos[:, root_joint_ind[:2]] = root_pos_walk_end
print(f"下蹲阶段：{SQUAT_DURATION}秒 | {SQUAT_STEPS}步")

# 4.4 增强舞蹈阶段（使用dance1_subject3数据集）
# 获取dance1_subject3的轨迹（可使用同一数据集的不同片段或重复使用）
dance1_qpos, dance1_qvel = get_trajectory_segment(
    env, "lafan1", "dance1_subject2", DANCE_STEPS  # 替换为dance1_subject2
)
dance2_qpos, dance2_qvel = get_trajectory_segment(
    env, "lafan1", "dance1_subject2", DANCE_STEPS  # 替换为dance1_subject2（用于混合）
)

# 混合同一数据集的轨迹（可增加多样性）
blended_qpos = blend_trajectories(dance1_qpos, dance2_qpos, blend_ratio=0.6)
blended_qvel = blend_trajectories(dance1_qvel, dance2_qvel, blend_ratio=0.6)

# 分阶段增强
mid_point = DANCE_STEPS // 2
# 处理qpos（位置数据）
dance_qpos = np.vstack([
    enhance_dance_movement(blended_qpos[:mid_point], model, is_velocity=False, intensity=1.3, speed_factor=1.0),
    enhance_dance_movement(blended_qpos[mid_point:], model, is_velocity=False, intensity=1.5, speed_factor=1.2)
])
# 处理qvel（速度数据）
dance_qvel = np.vstack([
    enhance_dance_movement(blended_qvel[:mid_point], model, is_velocity=True, intensity=1.0, speed_factor=1.0),
    enhance_dance_movement(blended_qvel[mid_point:], model, is_velocity=True, intensity=1.0, speed_factor=1.2)
])

# 确保长度正确并固定根位置
dance_qpos = dance_qpos[:DANCE_STEPS]
dance_qvel = dance_qvel[:DANCE_STEPS]
dance_qpos[:, root_joint_ind[:2]] = root_pos_walk_end
print(f"增强舞蹈阶段：{DANCE_DURATION}秒 | {DANCE_STEPS}步")

# ===================== 5. 生成多循环完整轨迹 =====================
full_qpos = []
full_qvel = []

for cycle in range(CYCLE_TIMES):
    print(f"生成第 {cycle+1}/{CYCLE_TIMES} 个循环轨迹...")
    # 每个循环使用不同的混合比例增加多样性
    cycle_blend = 0.5 + 0.3 * np.sin(cycle * np.pi / 2)
    cycle_dance_qpos = blend_trajectories(dance1_qpos, dance2_qpos, cycle_blend)[:DANCE_STEPS]
    cycle_dance_qpos[:, root_joint_ind[:2]] = root_pos_walk_end
    
    full_qpos.extend([walk_qpos, stay_qpos, squat_qpos, cycle_dance_qpos])
    full_qvel.extend([walk_qvel, stay_qvel, squat_qvel, dance_qvel])

full_qpos = np.concatenate(full_qpos, axis=0)
full_qvel = np.concatenate(full_qvel, axis=0)

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