# main.py
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as backend
from threading import Thread

from tqdm import tqdm
import pickle

# 导入本地模块
from Environment import CarEnv
from Model import DQNAgent
from TrainingStrategies import CurriculumManager, MultiObjectiveOptimizer, ImitationLearningManager
import Hyperparameters

# 从Hyperparameters导入所有参数
from Hyperparameters import *

def ensure_models_directory():
    """确保models目录存在"""
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✅ 已创建 models 目录")
    return 'models'

def save_model_with_retry(model, filepath, max_retries=3):
    """带重试机制的模型保存"""
    for attempt in range(max_retries):
        try:
            model.save(filepath)
            print(f"✅ 模型保存成功: {os.path.basename(filepath)}")
            return True
        except Exception as e:
            print(f"⚠️ 保存失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    
    print(f"❌ 无法保存模型: {filepath}")
    return False

def create_dummy_state(env):
    """创建虚拟状态用于测试"""
    return {
        'image': np.ones((env.im_height, env.im_width, 3)),
        'location': np.array([-81.0, -195.0]),  # 2维
        'speed': np.array([0.0]),
        'heading': np.array([0.0]),
        'last_action': np.array([1])
    }

def extended_reward_calculation(env, action, reward, done, step_info):
    """扩展的奖励计算函数"""
    # 获取车辆状态
    vehicle_location = env.vehicle.get_location()
    velocity = env.vehicle.get_velocity()
    speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
    
    # 计算多目标指标
    metrics = {}
    
    # 1. 反应时间指标
    reaction_time = 0
    if hasattr(env, 'obstacle_detected_time') and env.obstacle_detected_time is not None:
        if hasattr(env, 'reaction_start_time') and env.reaction_start_time is not None:
            reaction_time = time.time() - env.reaction_start_time
    
    metrics['reaction_time'] = reaction_time
    
    # 2. 主动避障指标
    proactive_action = False
    if hasattr(env, 'suggested_action') and env.suggested_action is not None:
        if action == env.suggested_action:
            proactive_action = True
    
    metrics['proactive_action'] = proactive_action
    
    # 3. 安全性指标
    min_ped_distance = getattr(env, 'last_ped_distance', float('inf'))
    safety_score = 0
    if min_ped_distance < 100:
        if min_ped_distance > 12:
            safety_score = 10
        elif min_ped_distance > 8:
            safety_score = 7
        elif min_ped_distance > 5:
            safety_score = 3
        elif min_ped_distance > 3:
            safety_score = 1
        else:
            safety_score = 0
    
    metrics['safety'] = safety_score
    
    # 4. 静态障碍物指标
    if hasattr(env, 'check_static_obstacles'):
        static_distance, _ = env.check_static_obstacles(vehicle_location)
        metrics['static_distance'] = static_distance
        
        if static_distance == 0:
            metrics['static_collision'] = True
        else:
            metrics['static_collision'] = False
    
    # 5. 道路边界指标
    if hasattr(env, 'check_road_boundary'):
        boundary_distance, out_of_boundary = env.check_road_boundary(vehicle_location)
        metrics['off_road'] = out_of_boundary
    
    # 6. 效率指标
    progress = (vehicle_location.x + 81) / 236.0
    efficiency_score = progress * 100
    metrics['efficiency'] = efficiency_score
    
    # 7. 舒适度指标
    comfort_score = 5
    
    if hasattr(env, 'last_action') and env.last_action in [3, 4]:
        if getattr(env, 'same_steer_counter', 0) > 2:
            comfort_score = 2
        elif getattr(env, 'same_steer_counter', 0) > 1:
            comfort_score = 3
        else:
            comfort_score = 4
    else:
        comfort_score = 5
    
    metrics['comfort'] = comfort_score
    
    # 8. 规则遵循指标
    rule_score = 0.3
    
    if 20 <= speed_kmh <= 35:
        rule_score = 1.0
    elif 15 <= speed_kmh < 20 or 35 < speed_kmh <= 40:
        rule_score = 0.7
    
    metrics['rule_following'] = rule_score
    
    # 9. 碰撞检测
    metrics['collision'] = len(getattr(env, 'collision_history', [])) > 0
    
    # 10. 危险动作检测
    if speed_kmh > 40 and action in [3, 4]:
        metrics['dangerous_action'] = True
    else:
        metrics['dangerous_action'] = False
    
    return metrics

if __name__ == '__main__':
    FPS = 60
    ep_rewards = [-200]

    print("自动驾驶模型训练开始...")
    print("=" * 60)
    
    # 确保models目录存在
    models_dir = ensure_models_directory()
    
    # GPU内存配置
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # 创建智能体和环境
    print("创建智能体和环境...")
    agent = DQNAgent(
        use_dueling=True, 
        use_per=True,
        use_curriculum=True,
        use_multi_objective=True
    )
    
    env = CarEnv()
    
    # 设置训练策略
    agent.setup_training_strategies(env)

    # 预热模型
    print("预热模型...")
    dummy_state = create_dummy_state(env)
    
    try:
        qs = agent.get_qs(dummy_state)
        print(f"✅ 模型预热成功，Q值形状: {qs.shape}")
    except Exception as e:
        print(f"⚠️ 模型预热失败: {e}")
        # 尝试直接调用predict进行预热
        dummy_image = np.ones((1, env.im_height, env.im_width, 3)) / 255
        dummy_vector = np.zeros((1, 10))
        try:
            qs = agent.model.predict([dummy_image, dummy_vector], verbose=0)
            print(f"✅ 使用直接预测方法预热成功，Q值形状: {qs.shape}")
        except Exception as e2:
            print(f"❌ 直接预测也失败: {e2}")
            print("检查模型输入维度...")
            print(f"模型输入: {agent.model.input}")
            print(f"模型输出: {agent.model.output}")
            sys.exit(1)

    # 启动训练线程
    print("启动训练线程...")
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    
    # 等待训练初始化完成
    print("等待训练初始化完成...")
    start_time = time.time()
    while not agent.training_initialized:
        time.sleep(0.01)
        if time.time() - start_time > 30:
            print("⚠️ 训练初始化超时，继续执行...")
            break
    
    print("✅ 训练初始化完成")

    # 训练统计变量
    best_score = -float('inf')
    success_count = 0
    scores = []
    avg_scores = []
    
    # 其他统计变量
    per_stats = {'buffer_size': []}
    multi_obj_stats = {
        'reaction_time': [], 'safety': [], 'efficiency': [], 
        'comfort': [], 'static_avoidance': []
    }
    curriculum_stages = []
    reaction_time_stats = []
    static_collision_stats = []

    # 迭代训练轮次
    print(f"\n开始训练，共 {EPISODES} 轮...")
    print("=" * 60)
    
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []
        agent.tensorboard.step = episode

        # 应用课程学习配置
        if agent.curriculum_manager:
            config = agent.curriculum_manager.get_current_config()
            if episode % 50 == 0:
                print(f"课程学习 - 阶段 {agent.curriculum_manager.current_stage}({config['difficulty_name']})")
            curriculum_stages.append(agent.curriculum_manager.current_stage)
        
        # 重置每轮统计
        score = 0
        step = 1
        episode_metrics = {
            'reaction_time': [], 'safety': [], 'efficiency': [], 
            'comfort': [], 'static_avoidance': []
        }

        # 重置环境
        try:
            current_state = env.reset(episode)
        except Exception as e:
            print(f"❌ 重置环境失败: {e}")
            continue

        done = False
        episode_start = time.time()
        static_collision_occurred = False

        # 最大步数
        max_steps_per_episode = SECONDS_PER_EPISODE * FPS
        if agent.curriculum_manager:
            config = agent.curriculum_manager.get_current_config()
            max_steps_per_episode = config['max_episode_steps']

        # 运行episode
        while not done and step < max_steps_per_episode:
            # 选择动作
            if np.random.random() > Hyperparameters.EPSILON:
                try:
                    qs = agent.get_qs(current_state)
                    action = np.argmax(qs)
                    
                    # 安全检查：如果接近静态障碍物，调整动作
                    if hasattr(env, 'check_static_obstacles'):
                        vehicle_location = env.vehicle.get_location()
                        static_distance, _ = env.check_static_obstacles(vehicle_location)
                        
                        if static_distance < 5.0:
                            if action in [3, 4] and qs[0] > qs[action] * 0.7:
                                action = 0
                
                except Exception as e:
                    print(f"⚠️ 获取Q值失败: {e}")
                    action = np.random.randint(0, 5)
            else:
                action = np.random.randint(0, 5)
                
                # 探索时的安全检查
                if hasattr(env, 'check_static_obstacles'):
                    vehicle_location = env.vehicle.get_location()
                    static_distance, _ = env.check_static_obstacles(vehicle_location)
                    
                    if static_distance < 3.0:
                        safe_actions = [0, 3, 4]
                        action = np.random.choice(safe_actions)
                
                time.sleep(1 / FPS)

            # 执行动作
            try:
                new_state, reward, done, _ = env.step(action)
            except Exception as e:
                print(f"❌ 执行动作失败: {e}")
                break

            # 检测静态碰撞
            static_collision = False
            if hasattr(env, 'check_static_obstacles'):
                vehicle_location = env.vehicle.get_location()
                static_distance, _ = env.check_static_obstacles(vehicle_location)
                if static_distance == 0:
                    static_collision = True
                    static_collision_occurred = True

            # 计算多目标指标
            if agent.multi_objective_optimizer:
                try:
                    step_info = {'step': step, 'action': action}
                    metrics = extended_reward_calculation(env, action, reward, done, step_info)
                    
                    for key in episode_metrics:
                        if key in metrics:
                            episode_metrics[key].append(metrics[key])
                    
                    composite_reward = agent.multi_objective_optimizer.compute_composite_reward(metrics)
                    reward = composite_reward
                except Exception as e:
                    print(f"⚠️ 计算多目标奖励失败: {e}")

            score += reward
            
            # 更新经验回放
            try:
                agent.update_replay_memory((current_state, action, reward, new_state, done))
            except Exception as e:
                print(f"⚠️ 更新经验回放失败: {e}")

            current_state = new_state
            step += 1

            if done:
                break

        # 清理环境
        try:
            env.cleanup_actors()
        except:
            pass

        # 记录统计
        scores.append(score)
        static_collision_stats.append(1 if static_collision_occurred else 0)
        
        # 更新成功计数
        if score > 5:
            success_count += 1

        # 保存模型
        if episode % 10 == 0:
            model_path = f'{models_dir}/{MODEL_NAME}_ep{episode}_score{score:.1f}.model'
            save_model_with_retry(agent.model, model_path)
        
        if score > best_score:
            best_score = score
            model_path = f'{models_dir}/{MODEL_NAME}_best_ep{episode}_score{score:.1f}.model'
            save_model_with_retry(agent.model, model_path)
            print(f"🏆 新的最佳模型: Episode {episode}, 得分: {score:.2f}")

        # 打印训练信息
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            print(f'轮次: {episode:3d}, 得分: {score:6.2f}, 最近10轮平均: {avg_score:6.2f}, 成功: {success_count:3d}')

        # 衰减探索率
        if Hyperparameters.EPSILON > Hyperparameters.MIN_EPSILON:
            Hyperparameters.EPSILON *= Hyperparameters.EPSILON_DECAY
            Hyperparameters.EPSILON = max(Hyperparameters.MIN_EPSILON, Hyperparameters.EPSILON)

    # 结束训练
    agent.terminate = True
    trainer_thread.join()
    
    # 保存最终模型
    final_model_path = f'{models_dir}/{MODEL_NAME}_final_ep{EPISODES}_avg{np.mean(scores):.1f}.model'
    if save_model_with_retry(agent.model, final_model_path):
        print(f"✅ 最终模型已保存: {final_model_path}")
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"最终统计:")
    print(f"  总轮次: {EPISODES}")
    print(f"  最佳得分: {max(scores) if scores else 0:.2f}")
    print(f"  平均得分: {np.mean(scores) if scores else 0:.2f}")
    print(f"  成功率: {(success_count/EPISODES)*100:.1f}%")
    print(f"  静态碰撞率: {np.mean(static_collision_stats) if static_collision_stats else 0:.2%}")
    print(f"  最终探索率: {Hyperparameters.EPSILON:.4f}")
    
    # 显示保存的模型文件
    print(f"\n已保存的模型文件:")
    model_files = glob.glob(f'{models_dir}/*.model')
    if model_files:
        for model_file in sorted(model_files, key=os.path.getmtime)[-10:]:
            file_size = os.path.getsize(model_file) / (1024 * 1024)
            print(f"  📁 {os.path.basename(model_file)} ({file_size:.1f} MB)")
    else:
        print("  ⚠️ 没有找到模型文件")