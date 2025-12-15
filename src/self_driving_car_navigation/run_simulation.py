import torch
import time
import sys
import pygame
import carla

# 修正导入路径：适配_agent子目录下的carla_environment.py
from _agent.carla_environment import CarlaEnvironment

print("="*60)
print(f"[启动时间] {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[Python解释器] {sys.executable}")
print("="*60)
sys.stdout.flush()

def set_spectator_smooth(world, vehicle, last_transform=None):
    """平滑跟随视角（移植自参考代码的核心逻辑）"""
    spectator = world.get_spectator()
    vehicle_tf = vehicle.get_transform()
    # 目标视角：车辆后上方，提供良好视野
    target_tf = carla.Transform(
        vehicle_tf.transform(carla.Location(x=-8, z=3, y=0.5)),
        vehicle_tf.rotation
    )
    
    if last_transform is None:
        spectator.set_transform(target_tf)
        return target_tf
    
    # 线性插值平滑过渡
    def lerp(a, b, t):
        return a + t * (b - a)
    
    smooth_loc = carla.Location(
        x=lerp(last_transform.location.x, target_tf.location.x, 0.1),
        y=lerp(last_transform.location.y, target_tf.location.y, 0.1),
        z=lerp(last_transform.location.z, target_tf.location.z, 0.1)
    )
    smooth_rot = carla.Rotation(
        pitch=lerp(last_transform.rotation.pitch, target_tf.rotation.pitch, 0.1),
        yaw=lerp(last_transform.rotation.yaw, target_tf.rotation.yaw, 0.1),
        roll=lerp(last_transform.rotation.roll, target_tf.rotation.roll, 0.1)
    )
    smooth_tf = carla.Transform(smooth_loc, smooth_rot)
    spectator.set_transform(smooth_tf)
    return smooth_tf

def run_simulation():
    env = None
    try:
        print("\n[CARLA连接] 创建环境...")
        env = CarlaEnvironment()
        
        print("\n[环境重置] 生成车辆和传感器...")
        env.reset()
        
        if not env.vehicle or not env.vehicle.is_alive:
            raise RuntimeError("车辆生成失败，请检查CARLA是否正常运行")
        print(f"[车辆状态] 生成成功（ID: {env.vehicle.id}），已启用自动驾驶")

        # 初始化平滑视角
        last_spectator_tf = set_spectator_smooth(env.world, env.vehicle)
        print("视角已切换至车辆后上方（平滑跟随模式）")

        # 初始化时钟控制帧率
        clock = pygame.time.Clock()

        print("\n[仿真开始] 车辆将沿车道行驶，按Ctrl+C退出...")
        sys.stdout.flush()
        
        # 持续运行仿真（不限制步数）
        step = 0
        while True:
            # 同步CARLA帧（关键优化：保证控制时序稳定）
            env.world.tick()
            
            # 获取观测和障碍物信息
            observation = env.get_observation()
            obstacle_distances = env.get_obstacle_directions(observation['lidar_distances'])
            
            # 打印状态信息
            if step % 10 == 0:  # 每10步打印一次
                print(f"\n[步骤 {step}] 障碍物距离 - 前{obstacle_distances['front']:.1f}m | 后{obstacle_distances['rear']:.1f}m | "
                      f"左{obstacle_distances['left']:.1f}m | 右{obstacle_distances['right']:.1f}m")
                sys.stdout.flush()
            
            # 更新平滑视角
            last_spectator_tf = set_spectator_smooth(env.world, env.vehicle, last_spectator_tf)
            
            # 控制帧率为30FPS
            clock.tick(30)
            step += 1

    except KeyboardInterrupt:
        print("\n[用户终止] 收到退出信号")
    except Exception as e:
        print(f"\n[仿真错误] {str(e)}")
        sys.stdout.flush()
    finally:
        if env is not None:
            print("\n[资源清理] 销毁资源...")
            env.close()
        print("\n[程序退出]")

if __name__ == "__main__":
    run_simulation()