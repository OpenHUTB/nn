# main_fixed.py - 修复模型保存问题的版本
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

def extended_reward_calculation(env, action, reward, done, step_info):
    """
    扩展的奖励计算函数，用于多目标优化
    """
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
    
    # 3. 安全性指标 - 基于最近行人距离
    min_ped_distance = getattr(env, 'last_ped_distance', float('inf'))
    safety_score = 0
    if min_ped_distance < 100:
        if min_ped_distance > 12:
            safety_score = 10  # 非常安全
        elif min_ped_distance > 8:
            safety_score = 7   # 安全
        elif min_ped_distance > 5:
            safety_score = 3   # 警告
        elif min_ped_distance > 3:
            safety_score = 1   # 危险
        else:
            safety_score = 0   # 极危险
    
    metrics['safety'] = safety_score
    
    # 4. 效率指标 - 基于进度
    progress = (vehicle_location.x + 81) / 236.0  # 从-81到155
    efficiency_score = progress * 100  # 进度百分比
    metrics['efficiency'] = efficiency_score
    
    # 5. 舒适度指标 - 基于转向平滑性
    comfort_score = 5  # 默认舒适
    
    if hasattr(env, 'last_action') and env.last_action in [3, 4]:
        if getattr(env, 'same_steer_counter', 0) > 2:  # 连续同向转向
            comfort_score = 2   # 稍不舒适
        elif getattr(env, 'same_steer_counter', 0) > 1:
            comfort_score = 3   # 一般
        else:
            comfort_score = 4   # 舒适
    else:
        comfort_score = 5  # 直行，最舒适
    
    metrics['comfort'] = comfort_score
    
    # 6. 规则遵循指标 - 基于速度
    rule_score = 0.3  # 默认较低分数
    
    if 20 <= speed_kmh <= 35:  # 理想速度范围
        rule_score = 1.0
    elif 15 <= speed_kmh < 20 or 35 < speed_kmh <= 40:
        rule_score = 0.7
    elif 10 <= speed_kmh < 15 or 40 < speed_kmh <= 45:
        rule_score = 0.5
    elif 5 <= speed_kmh < 10:
        rule_score = 0.4
    
    metrics['rule_following'] = rule_score
    
    # 7. 特殊事件
    metrics['collision'] = len(getattr(env, 'collision_history', [])) > 0
    metrics['off_road'] = vehicle_location.x < -90 or abs(vehicle_location.y + 195) > 30
    
    # 8. 危险动作检测
    if speed_kmh > 40 and action in [3, 4]:  # 高速急转
        metrics['dangerous_action'] = True
    else:
        metrics['dangerous_action'] = False
    
    return metrics

if __name__ == '__main__':
    FPS = 60  # 帧率
    ep_rewards = [-200]  # 存储每轮奖励

    # 确保models目录存在
    models_dir = ensure_models_directory()
    
    # GPU内存配置
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # 创建智能体和环境
    agent = DQNAgent(
        use_dueling=True, 
        use_per=True,
        use_curriculum=True,
        use_multi_objective=True
    )
    
    env = CarEnv()
    
    # 设置训练策略
    agent.setup_training_strategies(env)

    # 启动训练线程并等待训练初始化完成
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # 预热Q网络
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # 训练统计变量
    best_score = -float('inf')  # 最佳得分
    success_count = 0  # 成功次数计数
    scores = []  # 存储每轮得分
    avg_scores = []  # 存储平均得分
    
    # 记录PER相关统计
    per_stats = {
        'avg_td_error': [],
        'buffer_size': []
    }
    
    # 多目标统计
    multi_obj_stats = {
        'reaction_time': [],
        'safety': [],
        'efficiency': [],
        'comfort': [],
        'rule_following': []
    }
    
    # 课程学习阶段记录
    curriculum_stages = []
    
    # 反应时间统计
    reaction_time_stats = []
    
    # 迭代训练轮次
    epds = []
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []  # 重置碰撞历史
        agent.tensorboard.step = episode  # 设置TensorBoard步数

        # 应用课程学习配置
        if agent.curriculum_manager:
            config = agent.curriculum_manager.get_current_config()
            if episode % 50 == 0:  # 每50轮打印一次
                print(f"课程学习 - 阶段 {agent.curriculum_manager.current_stage}({config['difficulty_name']}): "
                      f"行人(十字路口={config['pedestrian_cross']}, 普通={config['pedestrian_normal']})")
            curriculum_stages.append(agent.curriculum_manager.current_stage)
        
        # 重置每轮统计
        score = 0
        step = 1
        
        # 多目标指标记录
        episode_metrics = {
            'reaction_time': [],
            'safety': [],
            'efficiency': [],
            'comfort': [],
            'rule_following': []
        }

        # 重置环境并获取初始状态
        current_state = env.reset(episode)

        # 重置完成标志
        done = False
        episode_start = time.time()

        # 应用课程学习的最大步数限制
        if agent.curriculum_manager:
            config = agent.curriculum_manager.get_current_config()
            max_steps_per_episode = config['max_episode_steps']
        else:
            max_steps_per_episode = SECONDS_PER_EPISODE * FPS

        # 仅在给定秒数内运行
        while not done and step < max_steps_per_episode:
            # 选择动作策略
            if np.random.random() > Hyperparameters.EPSILON:
                # 从Q网络获取动作（利用）
                qs = agent.get_qs(current_state)
                action = np.argmax(qs)
                
                # 如果有建议的避让动作，考虑采纳
                if hasattr(env, 'suggested_action') and env.suggested_action is not None:
                    # 检查建议动作的Q值
                    suggested_q = qs[env.suggested_action]
                    current_best_q = qs[action]
                    
                    # 如果建议动作的Q值接近最佳动作，采纳建议
                    if suggested_q > current_best_q * 0.8:
                        action = env.suggested_action
                
                if episode % 100 == 0 and step % 100 == 0:  # 减少打印频率
                    print(f'Ep {episode} Step {step}: Q值 [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}, {qs[3]:>5.2f}, {qs[4]:>5.2f}] 动作: {action}')
            else:
                # 随机选择动作（探索）
                action = np.random.randint(0, 5)
                # 添加延迟以匹配60FPS
                time.sleep(1 / FPS)

            # 执行动作并获取结果
            new_state, reward, done, _ = env.step(action)
            
            # 计算反应时间
            reaction_time = 0
            if hasattr(env, 'obstacle_detected_time') and env.obstacle_detected_time is not None:
                if hasattr(env, 'reaction_start_time') and env.reaction_start_time is not None:
                    reaction_time = time.time() - env.reaction_start_time
            
            # 计算多目标指标
            if agent.multi_objective_optimizer:
                step_info = {'step': step, 'action': action}
                metrics = extended_reward_calculation(env, action, reward, done, step_info)
                
                # 记录指标
                for key in episode_metrics:
                    if key in metrics:
                        episode_metrics[key].append(metrics[key])
                
                # 使用多目标优化器计算综合奖励
                composite_reward = agent.multi_objective_optimizer.compute_composite_reward(metrics)
                reward = composite_reward  # 使用综合奖励
            
            score += reward  # 累加奖励
            
            # 更新经验回放（带反应时间信息）
            agent.update_replay_memory((current_state, action, reward, new_state, done), 
                                      reaction_time=reaction_time)
            current_state = new_state  # 更新当前状态

            step += 1

            if done:
                break

        # 本轮结束 - 销毁所有actor
        env.cleanup_actors()
        
        # 计算平均反应时间
        if episode_metrics['reaction_time']:
            avg_reaction_time = np.mean([rt for rt in episode_metrics['reaction_time'] if rt > 0])
            reaction_time_stats.append(avg_reaction_time)
        
        # 计算本轮平均指标
        avg_metrics = {}
        for key, values in episode_metrics.items():
            if values:
                # 过滤掉零值（无反应时）
                if key == 'reaction_time':
                    filtered_values = [v for v in values if v > 0]
                    avg_metrics[key] = np.mean(filtered_values) if filtered_values else 0
                else:
                    avg_metrics[key] = np.mean(values)
                # 记录到统计中
                if key in multi_obj_stats:
                    multi_obj_stats[key].append(avg_metrics[key])
        
        # 更新课程学习（带反应时间）
        success = score > 5  # 成功完成的阈值
        avg_rt = avg_metrics.get('reaction_time', 0)
        if agent.curriculum_manager:
            stage_changed = agent.curriculum_manager.update_stage(success, score, avg_rt)
            if stage_changed:
                print(f"课程学习阶段已更新: {agent.curriculum_manager.current_stage}")
                
                # 阶段变化时保存模型
                model_path = f'{models_dir}/{MODEL_NAME}_stage_{agent.curriculum_manager.current_stage}_ep_{episode}.model'
                save_model_with_retry(agent.model, model_path)
        
        # 更新多目标优化器权重
        if agent.multi_objective_optimizer and episode % 20 == 0:
            agent.multi_objective_optimizer.adjust_weights(avg_metrics)
        
        # 更新成功计数
        if success:
            success_count += 1
        
        # ============================================
        # 修复：简化模型保存条件
        # ============================================
        
        # 1. 定期保存模型（每10轮）
        if episode % 10 == 0:
            model_path = f'{models_dir}/{MODEL_NAME}_ep{episode}_score{score:.1f}.model'
            save_model_with_retry(agent.model, model_path)
        
        # 2. 保存最佳模型（如果比之前好）
        if score > best_score:
            best_score = score
            model_path = f'{models_dir}/{MODEL_NAME}_best_ep{episode}_score{score:.1f}.model'
            save_model_with_retry(agent.model, model_path)
            print(f"🏆 新的最佳模型: Episode {episode}, 得分: {score:.2f}")
        
        # 3. 保存里程碑模型（每50轮）
        if episode % 50 == 0:
            model_path = f'{models_dir}/{MODEL_NAME}_milestone_ep{episode}.model'
            save_model_with_retry(agent.model, model_path)
            print(f"🎯 里程碑模型: Episode {episode}")
        
        # 记录得分统计
        scores.append(score)
        avg_scores.append(np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores))

        # 记录PER缓冲区信息
        if hasattr(agent, 'replay_buffer'):
            per_stats['buffer_size'].append(len(agent.replay_buffer))

        # 定期聚合统计信息
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = np.mean(scores[-AGGREGATE_STATS_EVERY:]) if len(scores) >= AGGREGATE_STATS_EVERY else np.mean(scores)
            min_reward = min(scores[-AGGREGATE_STATS_EVERY:]) if len(scores) >= AGGREGATE_STATS_EVERY else min(scores)
            max_reward = max(scores[-AGGREGATE_STATS_EVERY:]) if len(scores) >= AGGREGATE_STATS_EVERY else max(scores)
            
            # 添加PER统计到TensorBoard
            stats_dict = {
                'reward_avg': average_reward, 
                'reward_min': min_reward, 
                'reward_max': max_reward,
                'epsilon': Hyperparameters.EPSILON
            }
            
            if hasattr(agent, 'replay_buffer'):
                avg_buffer = np.mean(per_stats['buffer_size'][-AGGREGATE_STATS_EVERY:]) if per_stats['buffer_size'] else 0
                stats_dict['buffer_size'] = avg_buffer
            
            # 添加多目标指标
            if agent.multi_objective_optimizer:
                for obj in ['reaction_time', 'safety', 'efficiency', 'comfort', 'rule_following']:
                    if multi_obj_stats[obj]:
                        recent_avg = np.mean(multi_obj_stats[obj][-AGGREGATE_STATS_EVERY:]) if len(multi_obj_stats[obj]) >= AGGREGATE_STATS_EVERY else np.mean(multi_obj_stats[obj])
                        stats_dict[f'{obj}_score'] = recent_avg
            
            # 添加反应时间统计
            if reaction_time_stats:
                avg_rt = np.mean(reaction_time_stats[-AGGREGATE_STATS_EVERY:]) if len(reaction_time_stats) >= AGGREGATE_STATS_EVERY else np.mean(reaction_time_stats)
                stats_dict['reaction_time'] = avg_rt
            
            agent.tensorboard.update_stats(**stats_dict)

        epds.append(episode)
        
        # 打印训练信息
        if episode % 10 == 0:  # 每10轮打印一次
            avg_rt = np.mean(reaction_time_stats[-10:]) if len(reaction_time_stats) >= 10 else 0
            info_str = f'轮次: {episode:3d}, 得分: {score:6.2f}, 成功: {success_count:3d}, 反应时间: {avg_rt:.2f}s'
            if agent.curriculum_manager:
                info_str += f', 阶段: {agent.curriculum_manager.current_stage}'
            print(info_str)
        
        # 衰减探索率
        if Hyperparameters.EPSILON > Hyperparameters.MIN_EPSILON:
            Hyperparameters.EPSILON *= Hyperparameters.EPSILON_DECAY
            Hyperparameters.EPSILON = max(Hyperparameters.MIN_EPSILON, Hyperparameters.EPSILON)

    # 设置训练线程终止标志并等待其结束
    agent.terminate = True
    trainer_thread.join()
    
    # ============================================
    # 修复：始终保存最终模型
    # ============================================
    
    # 保存最终模型
    final_model_path = f'{models_dir}/{MODEL_NAME}_final_ep{EPISODES}_avg{np.mean(scores):.1f}.model'
    save_model_with_retry(agent.model, final_model_path)
    print(f"✅ 最终模型已保存: {final_model_path}")
    
    # 同时保存目标网络
    final_target_path = f'{models_dir}/{MODEL_NAME}_target_final.model'
    save_model_with_retry(agent.target_model, final_target_path)
    print(f"✅ 目标网络已保存: {final_target_path}")
    
    # 保存训练统计数据
    training_stats = {
        'scores': scores,
        'avg_scores': avg_scores,
        'multi_obj_stats': multi_obj_stats,
        'reaction_time_stats': reaction_time_stats,
        'curriculum_stages': curriculum_stages,
        'final_scores': {
            'max': max(scores) if scores else 0,
            'avg': np.mean(scores) if scores else 0,
            'min': min(scores) if scores else 0,
        }
    }
    
    stats_file = f'training_stats_{int(time.time())}.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump(training_stats, f)
    print(f"📊 训练统计数据已保存到: {stats_file}")
    
    # 打印最终统计
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"最终统计:")
    print(f"  总轮次: {EPISODES}")
    print(f"  最佳得分: {max(scores) if scores else 0:.2f}")
    print(f"  平均得分: {np.mean(scores) if scores else 0:.2f}")
    print(f"  成功率: {(success_count/EPISODES)*100:.1f}%")
    print(f"  平均反应时间: {np.mean(reaction_time_stats) if reaction_time_stats else 0:.2f}秒")
    print(f"  最终探索率: {Hyperparameters.EPSILON:.4f}")
    
    # 显示保存的模型文件
    print(f"\n已保存的模型文件:")
    model_files = glob.glob(f'{models_dir}/*.model')
    if model_files:
        for model_file in sorted(model_files, key=os.path.getmtime):
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"  📁 {os.path.basename(model_file)} ({file_size:.1f} MB)")
    else:
        print("  ⚠️ 没有找到模型文件，请检查保存路径和权限")