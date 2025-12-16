"""
主程序入口 - 协调所有模块
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
from collections import deque

from car_env import CarEnv
from route_visualizer import RouteVisualizer
from vehicle_tracker import VehicleTracker
from model_manager import ModelManager
from trajectory_manager import TrajectoryManager
from traffic_manager import TrafficManager
from config_manager import ConfigManager
import config as cfg

def setup_environment():
    """设置整体环境"""
    print("=" * 60)
    print("CARLA自动驾驶系统启动")
    print("=" * 60)
    
    # 获取配置
    trajectory = cfg.get_current_trajectory()
    if trajectory is None:
        print("❌ 无法获取轨迹配置")
        return None, None, None, None, None, None, None
    
    print(f"📌 使用轨迹: {trajectory['description']}")
    
    # 创建CARLA环境
    try:
        env = CarEnv(trajectory['start'], trajectory['end'])
        print("✅ CARLA环境创建成功")
    except Exception as e:
        print(f"❌ 创建CARLA环境失败: {e}")
        return None, None, None, None, None, None, None
    
    # 创建配置管理器
    config_mgr = ConfigManager(client=env.client)
    
    # 应用默认设置
    config_mgr.apply_default_settings()
    
    # 设置仿真参数（提高性能）
    settings = env.world.get_settings()
    settings.fixed_delta_seconds = cfg.FIXED_DELTA_SECONDS
    settings.synchronous_mode = cfg.SYNCHRONOUS_MODE
    settings.no_rendering_mode = cfg.NO_RENDERING_MODE
    env.world.apply_settings(settings)
    
    print(f"📊 设置时间步长: {cfg.FIXED_DELTA_SECONDS}s ({1/cfg.FIXED_DELTA_SECONDS:.1f} FPS)")
    
    # 检查模拟器状态
    config_mgr.inspect_simulation()
    
    # 创建交通管理器
    traffic_mgr = TrafficManager(client=env.client)
    
    # 生成交通流
    if cfg.ENABLE_TRAFFIC:
        print("\n🚦 生成交通流...")
        traffic_mgr.generate_traffic(
            num_vehicles=cfg.TRAFFIC_VEHICLES,
            num_walkers=cfg.TRAFFIC_WALKERS,
            safe_mode=cfg.TRAFFIC_SAFE_MODE,
            hybrid_mode=cfg.TRAFFIC_HYBRID_MODE,
            sync_mode=cfg.TRAFFIC_SYNC_MODE,
            respawn_vehicles=cfg.TRAFFIC_RESPAWN
        )
    
    # 创建路线可视化器
    visualizer = RouteVisualizer(env.world)
    
    # 创建车辆跟踪器（控制视角）
    tracker = VehicleTracker(env.world)
    
    # 创建模型管理器
    model_mgr = ModelManager()
    
    # 创建轨迹管理器
    traj_mgr = TrajectoryManager(env)
    
    return env, config_mgr, traffic_mgr, visualizer, tracker, model_mgr, traj_mgr

def run_episode(env, config_mgr, traffic_mgr, visualizer, tracker, model_mgr, traj_mgr, episode_num):
    """运行单个episode"""
    print(f"\n{'='*60}")
    print(f"Episode {episode_num}")
    print(f"{'='*60}")
    
    # 重置环境
    try:
        current_state = env.reset()
        print(f"✅ 环境重置成功")
    except Exception as e:
        print(f"❌ 环境重置失败: {e}")
        return False
    
    # 获取车辆
    ego_vehicle = env.vehicle
    if ego_vehicle is None:
        print("❌ 未找到车辆")
        return False
    
    # 重置跟踪器
    tracker.reset()
    
    # 设置初始视角（俯视）
    tracker.set_top_down_view(ego_vehicle, height=cfg.TOP_DOWN_HEIGHT)
    
    # 获取并绘制规划路线
    route_points = traj_mgr.get_route_points()
    visualizer.draw_planned_route(route_points)
    
    # 重置可视化器历史
    visualizer.reset_history()
    
    done = False
    step_count = 0
    frame_skip_count = 0
    fps_counter = deque(maxlen=120)  # 增加历史长度
    last_frame_time = time.time()
    
    while not done and step_count < cfg.MAX_STEPS_PER_EPISODE:
        step_count += 1
        step_start = time.time()
        current_time = time.time()
        
        # 计算帧间隔
        frame_interval = current_time - last_frame_time
        last_frame_time = current_time
        
        # 自适应跳帧逻辑
        should_skip_frame = False
        if cfg.MAX_FRAME_SKIP > 0 and frame_interval > cfg.FIXED_DELTA_SECONDS * 1.5:
            frame_skip_count += 1
            if frame_skip_count <= cfg.MAX_FRAME_SKIP:
                should_skip_frame = True
                if cfg.DEBUG_MODE and step_count % 50 == 0:
                    print(f"[跳过] 帧间隔过大: {frame_interval:.3f}s，跳过更新")
        else:
            frame_skip_count = 0
        
        # 更新交通管理器（如果是同步模式）
        if traffic_mgr and cfg.TRAFFIC_SYNC_MODE and not should_skip_frame:
            traffic_mgr.update()
        
        # 获取车辆状态
        vehicle_state = tracker.get_vehicle_state(ego_vehicle)
        
        # 更新视角（跳过某些帧时也更新，但减少频率）
        if not should_skip_frame or step_count % 2 == 0:
            tracker.smooth_follow_vehicle(ego_vehicle, height=cfg.TOP_DOWN_HEIGHT)
        
        # 更新车辆可视化（可以适当降低频率）
        if vehicle_state and (step_count % 2 == 0 or not should_skip_frame):
            visualizer.update_vehicle_display(
                vehicle_state['x'],
                vehicle_state['y'],
                vehicle_state['heading']
            )
        
        # 模型预测动作
        action = model_mgr.predict_action(current_state, vehicle_state)
        
        # 执行动作
        try:
            new_state, reward, done, _ = env.step(action, current_state)
            current_state = new_state
        except Exception as e:
            print(f"❌ 执行动作失败: {e}")
            done = True
        
        # 显示进度
        if step_count % 100 == 0:
            progress_info = tracker.calculate_progress(
                vehicle_state['x'] if vehicle_state else 0,
                vehicle_state['y'] if vehicle_state else 0,
                route_points
            )
            print(f"步骤 {step_count}, 奖励: {reward:.2f}, {progress_info}")
        
        # 计算FPS
        frame_time = time.time() - step_start
        fps_counter.append(frame_time)
        
        # 计算平滑FPS
        if len(fps_counter) >= 30:
            avg_frame_time = np.mean(list(fps_counter)[-30:])
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            current_fps = len(fps_counter) / sum(fps_counter) if fps_counter else 0
        
        # 显示调试信息
        if cfg.DEBUG_MODE and step_count % 100 == 0:
            if vehicle_state:
                print(f"[{step_count:4d}] FPS: {current_fps:5.1f} | "
                      f"动作: {cfg.ACTION_NAMES[action]} | "
                      f"位置: ({vehicle_state['x']:.1f}, {vehicle_state['y']:.1f}) | "
                      f"速度: {vehicle_state['speed_2d']:.1f}m/s")
        
        if done:
            print(f"✅ Episode {episode_num} 完成，步数: {step_count}")
            break
        
        # 限制帧率（如果需要）
        if cfg.FPS_LIMIT > 0:
            target_frame_time = 1.0 / cfg.FPS_LIMIT
            actual_frame_time = time.time() - step_start
            if actual_frame_time < target_frame_time:
                time.sleep(target_frame_time - actual_frame_time)
    
    if step_count >= cfg.MAX_STEPS_PER_EPISODE:
        print(f"⏰ Episode {episode_num} 达到最大步数限制")
    
    # 显示统计信息
    path_length = visualizer.calculate_path_length()
    avg_fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0
    print(f"📊 行驶距离: {path_length:.1f}m, 平均FPS: {avg_fps:.1f}")
    
    return True

def main():
    """主函数"""
    # 初始化各模块
    result = setup_environment()
    if result[0] is None:
        return
    
    env, config_mgr, traffic_mgr, visualizer, tracker, model_mgr, traj_mgr = result
    
    # 加载模型
    if not model_mgr.load_models():
        print("❌ 模型加载失败")
        return
    
    print("\n🚗 开始自动驾驶...")
    
    # 运行多个episode
    for episode in range(cfg.TOTAL_EPISODES):
        success = run_episode(
            env, config_mgr, traffic_mgr, visualizer, tracker, model_mgr, traj_mgr, episode + 1
        )
        
        if not success:
            print(f"❌ Episode {episode + 1} 运行失败")
        
        # 等待片刻再开始下一个episode
        if episode < cfg.TOTAL_EPISODES - 1:
            print(f"\n等待 {cfg.EPISODE_INTERVAL} 秒开始下一个episode...")
            time.sleep(cfg.EPISODE_INTERVAL)
    
    # 清理
    print("\n" + "=" * 60)
    print("所有episode完成！")
    print("=" * 60)
    
    # 清理交通流
    if traffic_mgr:
        traffic_mgr.cleanup()
    
    # 显示最终统计
    print("\n📈 最终统计:")
    print(f"总episodes: {cfg.TOTAL_EPISODES}")
    print(f"每episode最大步数: {cfg.MAX_STEPS_PER_EPISODE}")
    print(f"交通模式: {'启用' if cfg.ENABLE_TRAFFIC else '禁用'}")
    
    if cfg.ENABLE_TRAFFIC and traffic_mgr:
        traffic_info = traffic_mgr.get_traffic_info()
        print(f"交通车辆数: {traffic_info['num_vehicles']}")
        print(f"交通行人数: {traffic_info['num_walkers']}")
    
    print("程序结束")

if __name__ == '__main__':
    main()
