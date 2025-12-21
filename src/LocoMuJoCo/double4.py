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
    lafan1_dataset_conf=LAFAN1DatasetConf(["fallAndGetUp1_subject1", "sprint1_subject4"])  # 摔倒起身和冲刺数据集
)
env.reset(jax.random.PRNGKey(0))
ENV_DT = env.dt  # 环境时间步长（通常为0.02s）
FPS = int(1 / ENV_DT)  # 计算帧率（通常为50FPS）
print(f"环境参数：dt={ENV_DT}s | FPS={FPS}")

# ===================== 2. 配置各阶段时长与步数（修改为：摔倒起身+冲刺 循环） =====================
FALL_GETUP_DURATION = 14   # 摔倒与起身时长14秒
SPRINT_DURATION = 17       # 冲刺时长17秒
CYCLE_TIMES = 3            # 循环次数

# 计算各阶段步数
FALL_GETUP_STEPS = int(FALL_GETUP_DURATION * FPS)
SPRINT_STEPS = int(SPRINT_DURATION * FPS)
SINGLE_CYCLE_STEPS = FALL_GETUP_STEPS + SPRINT_STEPS  # 单循环：摔倒起身 + 冲刺
TOTAL_STEPS = SINGLE_CYCLE_STEPS * CYCLE_TIMES

# 打印配置信息
print(f"单循环结构：摔倒与起身{FALL_GETUP_DURATION}s({FALL_GETUP_STEPS}步) + 冲刺{SPRINT_DURATION}s({SPRINT_STEPS}步)")
print(f"总循环次数：{CYCLE_TIMES}次 | 总时长：{CYCLE_TIMES*(FALL_GETUP_DURATION + SPRINT_DURATION)}s")

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
    # 从轨迹处理器获取完整轨迹数据
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

# ===================== 4. 生成各阶段轨迹（摔倒起身+冲刺） =====================
model = env.get_model()
root_joint_ind = mj_jntname2qposid("root", model)  # 获取根关节索引

# 4.1 摔倒与起身轨迹（LAFAN1的摔倒与起身动作）
fall_getup_qpos, fall_getup_qvel = get_lafan_trajectory(
    env, "fallAndGetUp1_subject1", FALL_GETUP_STEPS
)
# 固定初始根位置（可选：设置为原点附近）
fall_getup_qpos[:, root_joint_ind[:2]] = np.array([0.0, 0.0])  # 原地摔倒起身
print(f"摔倒与起身轨迹生成完成：{FALL_GETUP_STEPS}步")

# 4.2 冲刺轨迹（与摔倒起身保持连贯）
sprint_qpos, sprint_qvel = get_lafan_trajectory(
    env, "sprint1_subject4", SPRINT_STEPS
)
# 确保从摔倒起身的终点位置开始冲刺
root_pos_getup_end = fall_getup_qpos[-1, root_joint_ind[:2]]
# 计算相对位移，保持冲刺动作的连贯性
sprint_qpos[:, root_joint_ind[:2]] = root_pos_getup_end + (
    sprint_qpos[:, root_joint_ind[:2]] - sprint_qpos[0, root_joint_ind[:2]]
)
print(f"冲刺轨迹生成完成：{SPRINT_STEPS}步")

# ===================== 5. 拼接多循环完整轨迹（摔倒起身+冲刺 循环） =====================
full_qpos = []
full_qvel = []

for cycle in range(CYCLE_TIMES):
    print(f"生成第 {cycle+1}/{CYCLE_TIMES} 个循环...")
    # 循环内拼接：摔倒起身 -> 冲刺
    full_qpos.extend([fall_getup_qpos, sprint_qpos])
    full_qvel.extend([fall_getup_qvel, sprint_qvel])

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