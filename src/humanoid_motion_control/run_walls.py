# 标准库
import numpy as np
# 第三方库
from absl import app
# dm_control 模块
from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control import viewer

# 全局环境变量，供策略函数访问
global_env = None

def create_environment():
    """创建带墙壁走廊的人形机器人环境"""
    walker = cmu_humanoid.CMUHumanoidPositionControlled(
        observable_options={'egocentric_camera': dict(enabled=True)})

    arena = corr_arenas.WallsCorridor(
        wall_gap=4.,
        wall_width=distributions.Uniform(1, 7),
        wall_height=3.0,
        corridor_width=10,
        corridor_length=100,
        include_initial_padding=False)
    
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(0.5, 0, 0),
        target_velocity=3.0,
        physics_timestep=0.005,
        control_timestep=0.03)
    
    return composer.Environment(
        time_limit=30,
        task=task,
        strip_singleton_obs_buffer_dim=True)

def random_policy(timestep):
    """随机动作策略：使用全局环境变量获取动作空间"""
    global global_env
    action_spec = global_env.action_spec()
    return np.random.uniform(
        low=action_spec.minimum,
        high=action_spec.maximum,
        size=action_spec.shape
    )

def main(unused_argv):
    """主函数：初始化全局环境并启动viewer"""
    global global_env
    # 初始化环境并赋值给全局变量
    global_env = create_environment()
    
    # 启动viewer
    viewer.launch(
        environment_loader=create_environment,
        policy=random_policy
    )


if __name__ == "__main__":
    app.run(main)