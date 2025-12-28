import numpy as np
import jax
import mujoco
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid

# ===================== 1. 初始化环境（仅使用LAFAN1数据集） =====================
env = ImitationFactory.make(
    "UnitreeH1",
    n_substeps=20,
    lafan1_dataset_conf=LAFAN1DatasetConf(["fight1_subject3", "fightAndSports1_subject4"])  # 仅使用LAFAN1数据集
)
env.reset(jax.random.PRNGKey(0))
ENV_DT = env.dt  # 环境时间步长（通常为0.02s）
FPS = int(1 / ENV_DT)  # 计算帧率（通常为50FPS）
print(f"环境参数：dt={ENV_DT}s | FPS={FPS}")

# ===================== 2. 配置各阶段时长与步数 =====================
MOVE_DURATION = 15      # 移动时长15秒
STAY_DURATION = 1       # 停留时长1秒
JUMP_DURATION = 15      # 跳跃时长15秒
CYCLE_TIMES = 3         # 循环次数

# 计算各阶段步数
MOVE_STEPS = int(MOVE_DURATION * FPS)
STAY_STEPS = int(STAY_DURATION * FPS)
JUMP_STEPS = int(JUMP_DURATION * FPS)
SINGLE_CYCLE_STEPS = MOVE_STEPS + STAY_STEPS + JUMP_STEPS
TOTAL_STEPS = SINGLE_CYCLE_STEPS * CYCLE_TIMES

# 修正打印信息与实际配置一致
print(f"单循环结构：移动{7}s({MOVE_STEPS}步) + 停留{1}s({STAY_STEPS}步) + 跳跃{12}s({JUMP_STEPS}步)")
print(f"总循环次数：{CYCLE_TIMES}次 | 总时长：{CYCLE_TIMES*(7+1+12)}s")

# ===================== 3. 轨迹片段生成函数 =====================
def get_lafan_trajectory(env, dataset_name, target_steps):
    """从LAFAN1数据集提取轨迹并调整到目标步数"""
    # 创建临时环境加载指定LAFAN1数据集
    temp_env = ImitationFactory.make(
        "UnitreeH1",
        lafan1_dataset_conf=LAFAN1DatasetConf([dataset_name]),
        n_substeps=env._n_substeps
    )
    
    temp_env.reset(jax.random.PRNGKey(0))
    # 直接从轨迹处理器获取完整轨迹数据（与参考代码保持一致的访问方式）
    raw_qpos = np.array(temp_env.th.traj.data.qpos)
    raw_qvel = np.array(temp_env.th.traj.data.qvel)
    
    # 校验轨迹有效性
    if len(raw_qpos) == 0:
        raise ValueError(f"LAFAN1数据集 {dataset_name} 无有效轨迹数据！")
    
    # 循环填充轨迹至目标步数（保持动作连续性）
    traj_qpos = []
    traj_qvel = []
    for i in range(target_steps):
        idx = i % len(raw_qpos)  # 循环索引避免越界
        traj_qpos.append(raw_qpos[idx])
        traj_qvel.append(raw_qvel[idx])
    return np.array(traj_qpos), np.array(traj_qvel)

# ===================== 4. 生成各阶段轨迹（全部使用LAFAN1数据集） =====================
model = env.get_model()

# 4.1 移动轨迹（LAFAN1的跑步动作）
move_qpos, move_qvel = get_lafan_trajectory(
    env, "fight1_subject3", MOVE_STEPS
)
print(f"移动轨迹生成完成：{MOVE_STEPS}步")

# 4.2 停留轨迹（优化版：完全继承移动终点状态，确保连贯性）
# 复用移动阶段最后一帧的完整姿态（包括所有关节位置）
last_move_qpos = move_qpos[-1].copy()
# 速度严格归零（避免微小抖动）
last_move_qvel = np.zeros_like(move_qvel[0])
# 生成停留阶段轨迹（与参考代码的实现方式一致）
stay_qpos = np.tile(last_move_qpos, (STAY_STEPS, 1))
stay_qvel = np.tile(last_move_qvel, (STAY_STEPS, 1))
# 验证停留阶段的根位置与移动终点一致
root_joint_ind = mj_jntname2qposid("root", model)
print(f"停留阶段根位置：{stay_qpos[0, root_joint_ind[:2]]}（与移动终点一致）")
print(f"停留轨迹生成完成：{STAY_STEPS}步")

# 4.3 跳跃轨迹（LAFAN1的跳跃动作）
jump_qpos, jump_qvel = get_lafan_trajectory(
    env, "fightAndSports1_subject4", JUMP_STEPS
)
# 固定根位置（保持在移动结束时的位置，实现原地跳跃）
root_pos_move_end = move_qpos[-1, root_joint_ind[:2]]  # 获取移动结束时的x/y位置
jump_qpos[:, root_joint_ind[:2]] = root_pos_move_end  # 固定跳跃时的根位置
print(f"跳跃轨迹生成完成：{JUMP_STEPS}步")

# ===================== 5. 拼接多循环完整轨迹 =====================
full_qpos = []
full_qvel = []

for cycle in range(CYCLE_TIMES):
    print(f"生成第 {cycle+1}/{CYCLE_TIMES} 个循环...")
    # 循环内拼接：移动 -> 停留 -> 跳跃
    full_qpos.extend([move_qpos, stay_qpos, jump_qpos])
    full_qvel.extend([move_qvel, stay_qvel, jump_qvel])

# 合并为numpy数组
full_qpos = np.concatenate(full_qpos, axis=0)
full_qvel = np.concatenate(full_qvel, axis=0)

# 验证总轨迹长度
print(f"完整轨迹生成完成：{len(full_qpos)}步 | 实际总时长：{len(full_qpos)/FPS:.1f}s")

# ===================== 6. 加载轨迹并播放 =====================
# 创建轨迹元信息
jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
traj_info = TrajectoryInfo(
    jnt_names,
    model=TrajectoryModel(model.njnt, jax.numpy.array(model.jnt_type)),
    frequency=FPS
)

# 创建轨迹数据对象
traj_data = TrajectoryData(
    jax.numpy.array(full_qpos),
    jax.numpy.array(full_qvel),
    split_points=jax.numpy.array([0, len(full_qpos)])
)

# 组合轨迹并加载到环境
traj = Trajectory(traj_info, traj_data)
env.load_trajectory(traj)

# 播放轨迹（render=True显示可视化）
env.play_trajectory(n_steps_per_episode=len(full_qpos), render=True)