#!/usr/bin/env python3
"""
主程序 - 简化版
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import sys
import glob

# 导入CARLA
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
from collections import deque

from car_env import CarEnv
from route_visualizer import RouteVisualizer
from vehicle_tracker import VehicleTracker
from traffic_manager import TrafficManager
import config as cfg

def setup_environment():
    """设置环境"""
    print("启动CARLA自动驾驶系统")
    
    # 获取轨迹
    trajectory = cfg.get_current_trajectory()
    if not trajectory:
        print("轨迹配置错误")
        return None, None, None, None, None
    
    print(f"使用轨迹: {trajectory['description']}")
    
    # 创建环境
    try:
        env = CarEnv(trajectory['start'], trajectory['end'])
        print("CARLA环境创建成功")
    except Exception as e:
        print(f"创建环境失败: {e}")
        return None, None, None, None, None
    
    # 设置仿真参数
    settings = env.world.get_settings()
    settings.fixed_delta_seconds = cfg.FIXED_DELTA_SECONDS
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    env.world.apply_settings(settings)
    
    # 设置天气
    env.world.set_weather(carla.WeatherParameters.ClearNoon)
    
    # 创建交通管理器
    traffic_mgr = TrafficManager(env.client, env.world)
    
    # 生成交通
    if cfg.ENABLE_TRAFFIC:
        print("生成交通流...")
        traffic_mgr.generate_traffic(
            num_vehicles=cfg.TRAFFIC_VEHICLES,
            num_walkers=cfg.TRAFFIC_WALKERS,
            safe_mode=cfg.TRAFFIC_SAFE_MODE
        )
    
    # 创建其他管理器
    visualizer = RouteVisualizer(env.world)
    tracker = VehicleTracker(env.world)
    
    return env, traffic_mgr, visualizer, tracker, trajectory

def load_models():
    """加载模型"""
    try:
        from tensorflow.keras.models import load_model
        import tensorflow as tf
        
        braking_model = load_model(cfg.MODEL_PATHS['braking'])
        driving_model = load_model(cfg.MODEL_PATHS['driving'])
        
        print("模型加载成功")
        return braking_model, driving_model
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None

def predict_action(braking_model, driving_model, current_state, vehicle_state=None):
    """预测动作"""
    if not braking_model or not driving_model:
        return 1  # 默认直行而不是刹车
    
    # 刹车模型预测
    braking_state = np.array(current_state[:2]).reshape(1, -1)
    braking_qs = braking_model.predict(braking_state, verbose=0)[0]
    braking_action = np.argmax(braking_qs)
    
    # 只有当距离障碍物很近时才刹车，否则继续前进
    if braking_action == 0 and current_state[0] > 0.3:  # 距离障碍物较远
        braking_action = 1  # 改为直行
    
    # 如果安全，使用驾驶模型
    if braking_action == 1:
        driving_state = np.array(current_state[2:]).reshape(1, -1)
        driving_qs = driving_model.predict(driving_state, verbose=0)[0]
        # 避免总是选择同一个动作，增加一些探索性
        if np.random.random() > 0.7:  # 30%的概率选择次优动作
            # 选择第二好的动作
            sorted_indices = np.argsort(driving_qs)[::-1]
            return sorted_indices[1] + 1
        return np.argmax(driving_qs) + 1
    
    return 0

def run_episode(env, traffic_mgr, visualizer, tracker, episode_num):
    """运行一个episode"""
    print(f"\nEpisode {episode_num}")
    
    # 加载模型
    braking_model, driving_model = load_models()
    if not braking_model or not driving_model:
        print("无法加载模型，退出")
        return False
    
    # 重置环境
    try:
        current_state = env.reset()
        print("环境重置成功")
    except Exception as e:
        print(f"环境重置失败: {e}")
        return False
    
    # 获取车辆
    ego_vehicle = env.vehicle
    if not ego_vehicle or not ego_vehicle.is_alive:
        print("未找到车辆或车辆已被销毁")
        return False
    
    # 设置后方跟随视角（启动独立线程）
    tracker.set_follow_view(ego_vehicle)
    
    # 绘制路线
    if hasattr(env, 'path') and env.path:
        route_points = []
        for waypoint in env.path:
            location = waypoint.transform.location
            route_points.append((location.x, location.y, location.z))
        visualizer.draw_planned_route(route_points)
    
    # 运行循环
    step_count = 0
    done = False
    fps_counter = deque(maxlen=30)
    stuck_counter = 0
    max_stuck_steps = 10  # 减少卡住检测步数
    
    # 性能监控
    total_control_time = 0
    control_count = 0
    
    while not done and step_count < cfg.MAX_STEPS_PER_EPISODE:
        step_count += 1
        step_start = time.time()
        
        # 检查车辆是否被销毁
        if not ego_vehicle or not ego_vehicle.is_alive:
            print("车辆已被销毁，终止episode")
            break
        
        # 检查车辆是否卡住
        velocity = ego_vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        if speed < 1.0:  # 速度小于1 km/h
            stuck_counter += 1
            if stuck_counter >= max_stuck_steps:
                print("车辆卡住，尝试全油门恢复...")
                # 尝试全油门摆脱卡住状态
                ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0, steer=0.0))
                time.sleep(0.1)  # 减少恢复时间
                stuck_counter = 0
        else:
            stuck_counter = 0
        
        # 更新车辆显示（降低频率）
        if step_count % 10 == 0:  # 每10步更新一次显示
            vehicle_state = tracker.get_vehicle_state(ego_vehicle)
            if vehicle_state:
                visualizer.update_vehicle_display(
                    vehicle_state['x'],
                    vehicle_state['y'],
                    vehicle_state['heading']
                )
        
        # 预测并执行动作
        action_start = time.time()
        action = predict_action(braking_model, driving_model, current_state, vehicle_state)
        
        try:
            new_state, reward, done, _ = env.step(action, current_state)
            current_state = new_state
            
            # 计算控制延迟
            control_time = time.time() - action_start
            total_control_time += control_time
            control_count += 1
            
            # 每50步报告一次性能
            if step_count % 50 == 0:
                avg_control_time = total_control_time / control_count
                print(f"平均控制延迟: {avg_control_time*1000:.1f} ms")
                total_control_time = 0
                control_count = 0
                
        except Exception as e:
            print(f"执行动作失败: {e}")
            done = True
        
        # 计算FPS
        frame_time = time.time() - step_start
        fps_counter.append(frame_time)
        
        if cfg.DEBUG_MODE and step_count % 100 == 0:
            fps = len(fps_counter) / sum(fps_counter) if fps_counter else 0
            vehicle_speed = vehicle_state.get('speed_2d', 0) if vehicle_state else 0
            print(f"步骤 {step_count}, FPS: {fps:.1f}, 速度: {vehicle_speed:.1f}m/s ({vehicle_speed*3.6:.1f}km/h)")
        
        if done:
            print(f"Episode {episode_num} 完成，步数: {step_count}")
            break
    
    # 清理跟踪器资源
    tracker.cleanup()
    
    return True

def main():
    """主函数"""
    # 设置环境
    result = setup_environment()
    if not result[0]:
        return
    
    env, traffic_mgr, visualizer, tracker, trajectory = result
    
    try:
        # 运行episode
        for episode in range(cfg.TOTAL_EPISODES):
            success = run_episode(env, traffic_mgr, visualizer, tracker, episode + 1)
            
            if episode < cfg.TOTAL_EPISODES - 1:
                print(f"等待 {cfg.EPISODE_INTERVAL} 秒...")
                time.sleep(cfg.EPISODE_INTERVAL)
    finally:
        # 确保清理
        if tracker:
            tracker.cleanup()
        if traffic_mgr:
            traffic_mgr.cleanup()
    
    print("\n程序结束")
    
if __name__ == '__main__':
    main()