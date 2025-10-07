# 导入dm_control的核心模块
from dm_control import composer  # 用于组合环境、实体和任务的框架
from dm_control.composer.variation import distributions  # 用于生成随机分布的工具（如随机墙壁宽度）
from dm_control.locomotion.arenas import corridors as corr_arenas  # 走廊类竞技场定义
from dm_control.locomotion.tasks import corridors as corr_tasks  # 走廊相关任务定义
from dm_control.locomotion.walkers import cmu_humanoid  # CMU人形机器人模型

# 导入其他辅助模块
from absl import app  # Google的命令行参数处理库
from dm_control import viewer  # 用于可视化仿真的交互窗口


def cmu_humanoid_run_walls(random_state=None):
    """创建一个CMU人形机器人在带墙壁的走廊中奔跑的环境
    
    Args:
        random_state: 随机数生成器状态，用于保证实验可复现性
    Returns:
        composer.Environment: 构建好的强化学习环境
    """

    # 创建一个位置控制的CMU人形机器人
    # observable_options配置观测项：启用第一视角相机（egocentric_camera）
    walker = cmu_humanoid.CMUHumanoidPositionControlled(
        observable_options={'egocentric_camera': dict(enabled=True)})

    # 建造带墙壁的走廊竞技场
    arena = corr_arenas.WallsCorridor(
        wall_gap=4.,  # 墙壁之间的间隙（机器人需要穿过的宽度）
        wall_width=distributions.Uniform(1, 7),  # 墙壁宽度为1到7之间的随机值（增加环境多样性）
        wall_height=3.0,  # 墙壁高度（确保机器人无法翻越）
        corridor_width=10,  # 走廊总宽度
        corridor_length=100,  # 走廊总长度
        include_initial_padding=False)  # 不添加初始空白区域（机器人直接从起点开始）

    # 定义"穿过走廊"任务：奖励机器人快速通过走廊
    task = corr_tasks.RunThroughCorridor(
        walker=walker,  # 绑定前面创建的机器人
        arena=arena,  # 绑定前面创建的走廊环境
        walker_spawn_position=(0.5, 0, 0),  # 机器人初始 spawn 位置（走廊起点附近）
        target_velocity=3.0,  # 目标速度（机器人需要达到的速度以获得高奖励）
        physics_timestep=0.005,  # 物理仿真时间步长（越小精度越高，计算量越大）
        control_timestep=0.03)  # 控制信号时间步长（机器人决策的频率，0.03秒约33Hz）

    # 创建强化学习环境
    return composer.Environment(
        time_limit=30,  # 单次仿真最大时长30秒
        task=task,  # 绑定任务
        random_state=random_state,  # 传入随机状态（保证可复现）
        strip_singleton_obs_buffer_dim=True)  # 去除观测数据中多余的单元素维度（简化数据结构）


def main(unused_argv):
    """主函数：启动可视化窗口运行环境"""
    # 启动dm_control的交互 viewer，加载上面定义的环境
    viewer.launch(environment_loader=cmu_humanoid_run_walls)


if __name__ == '__main__':
    # 使用absl的app.run启动程序（处理命令行参数并调用main函数）
    app.run(main)