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
    
    # 1. 安全性指标 - 基于最近行人距离
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
    
    # 2. 效率指标 - 基于进度
    progress = (vehicle_location.x + 81) / 236.0  # 从-81到155
    efficiency_score = progress * 100  # 进度百分比
    metrics['efficiency'] = efficiency_score
    
    # 3. 舒适度指标 - 基于转向平滑性
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
    
    # 4. 规则遵循指标 - 基于速度
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
    
    # 5. 特殊事件
    metrics['collision'] = len(getattr(env, 'collision_history', [])) > 0
    metrics['off_road'] = vehicle_location.x < -90 or abs(vehicle_location.y + 195) > 30
    
    # 6. 危险动作检测
    if speed_kmh > 40 and action in [3, 4]:  # 高速急转
        metrics['dangerous_action'] = True
    else:
        metrics['dangerous_action'] = False
    
    return metrics

if __name__ == '__main__':
    FPS = 60  # 帧率
    ep_rewards = [-200]  # 存储每轮奖励

    # GPU内存配置
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # 创建模型保存目录
    if not os.path.isdir('models'):
        os.makedirs('models')
    
    # 创建专家数据目录
    if not os.path.isdir('expert_data'):
        os.makedirs('expert_data')

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

    # 可选：使用模仿学习进行预训练
    use_imitation_pretraining = False  # 设置为True启用模仿学习预训练
    
    if use_imitation_pretraining:
        print("=" * 60)
        print("开始模仿学习预训练阶段")
        print("=" * 60)
        
        # 检查是否有现有的专家数据
        expert_files = glob.glob("expert_data/*.pkl")
        if expert_files:
            # 使用最新的专家数据
            latest_expert = max(expert_files, key=os.path.getctime)
            agent.imitation_manager.load_expert_data(latest_expert)
        else:
            # 收集新的专家数据
            print("未找到专家数据，开始收集...")
            agent.imitation_manager.collect_expert_demonstration(env, num_episodes=5)
        
        # 使用行为克隆进行预训练
        agent.model = agent.imitation_manager.pretrain_with_behavioral_cloning(agent.model, epochs=15)
        agent.target_model.set_weights(agent.model.get_weights())
        print("模仿学习预训练完成!")

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
        'safety': [],
        'efficiency': [],
        'comfort': [],
        'rule_following': []
    }
    
    # 课程学习阶段记录
    curriculum_stages = []
    
    # 迭代训练轮次
    epds = []
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []  # 重置碰撞历史
        agent.tensorboard.step = episode  # 设置TensorBoard步数

        # 应用课程学习配置
        if agent.curriculum_manager:
            config = agent.curriculum_manager.get_current_config()
            if episode % 50 == 0:  # 每50轮打印一次
                print(f"课程学习 - 阶段 {agent.curriculum_manager.current_stage}: "
                      f"行人(十字路口={config['pedestrian_cross']}, 普通={config['pedestrian_normal']})")
            curriculum_stages.append(agent.curriculum_manager.current_stage)
        
        # 重置每轮统计
        score = 0
        step = 1
        
        # 多目标指标记录
        episode_metrics = {
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
                if episode % 50 == 0 and step % 30 == 0:  # 减少打印频率
                    print(f'Ep {episode} Step {step}: Q值 [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}, {qs[3]:>5.2f}, {qs[4]:>5.2f}] 动作: {action}')
            else:
                # 随机选择动作（探索）
                action = np.random.randint(0, 5)
                # 添加延迟以匹配60FPS
                time.sleep(1 / FPS)

            # 执行动作并获取结果
            new_state, reward, done, _ = env.step(action)
            
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
            
            # 更新经验回放
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            current_state = new_state  # 更新当前状态

            step += 1

            if done:
                break

        # 本轮结束 - 销毁所有actor
        env.cleanup_actors()

        # 计算本轮平均指标
        avg_metrics = {}
        for key, values in episode_metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
                # 记录到统计中
                if key in multi_obj_stats:
                    multi_obj_stats[key].append(avg_metrics[key])
        
        # 更新课程学习
        success = score > 5  # 成功完成的阈值
        if agent.curriculum_manager:
            stage_changed = agent.curriculum_manager.update_stage(success, score)
            if stage_changed:
                print(f"课程学习阶段已更新: {agent.curriculum_manager.current_stage}")
        
        # 更新多目标优化器权重
        if agent.multi_objective_optimizer and episode % 20 == 0:
            agent.multi_objective_optimizer.adjust_weights(avg_metrics)
            if episode % 100 == 0:
                print(agent.multi_objective_optimizer.get_performance_report())
        
        # 更新成功计数
        if success:
            success_count += 1
        
        # 动态保存最佳模型
        if score > best_score:
            best_score = score
            model_suffix = f"advanced_best_{score:.2f}"
            agent.model.save(f'models/{MODEL_NAME}_{model_suffix}.model')
            print(f"新的最佳模型已保存: 得分={score:.2f}")

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
                for obj in ['safety', 'efficiency', 'comfort', 'rule_following']:
                    if multi_obj_stats[obj]:
                        recent_avg = np.mean(multi_obj_stats[obj][-AGGREGATE_STATS_EVERY:]) if len(multi_obj_stats[obj]) >= AGGREGATE_STATS_EVERY else np.mean(multi_obj_stats[obj])
                        stats_dict[f'{obj}_score'] = recent_avg
            
            agent.tensorboard.update_stats(**stats_dict)

            # 保存模型，仅当最小奖励达到设定值时
            if min_reward >= MIN_REWARD and (episode not in epds):
                model_suffix = f"advanced_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min"
                agent.model.save(f'models/{MODEL_NAME}_{model_suffix}__{int(time.time())}.model')

        epds.append(episode)
        
        # 打印训练信息
        if episode % 10 == 0:  # 每10轮打印一次
            info_str = f'轮次: {episode:3d}, 得分: {score:6.2f}, 成功: {success_count:3d}'
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
    
    # 保存最终模型
    if len(scores) > 0:
        final_max_reward = max(scores[-AGGREGATE_STATS_EVERY:] if len(scores) >= AGGREGATE_STATS_EVERY else scores)
        final_avg_reward = np.mean(scores[-AGGREGATE_STATS_EVERY:] if len(scores) >= AGGREGATE_STATS_EVERY else scores)
        final_min_reward = min(scores[-AGGREGATE_STATS_EVERY:] if len(scores) >= AGGREGATE_STATS_EVERY else scores)
        
        model_suffix = "advanced_final"
        agent.model.save(
            f'models/{MODEL_NAME}_{model_suffix}__{final_max_reward:_>7.2f}max_{final_avg_reward:_>7.2f}avg_{final_min_reward:_>7.2f}min__{int(time.time())}.model')
        
        # 保存训练统计数据
        training_stats = {
            'scores': scores,
            'avg_scores': avg_scores,
            'multi_obj_stats': multi_obj_stats,
            'curriculum_stages': curriculum_stages,
            'final_scores': {
                'max': final_max_reward,
                'avg': final_avg_reward,
                'min': final_min_reward
            }
        }
        
        stats_file = f'training_stats_{int(time.time())}.pkl'
        with open(stats_file, 'wb') as f:
            pickle.dump(training_stats, f)
        print(f"训练统计数据已保存到: {stats_file}")

    # 绘制训练曲线
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    # 1. 得分曲线
    axes[0, 0].plot(scores, label='每轮得分', alpha=0.6, linewidth=1)
    axes[0, 0].plot(avg_scores, label='平均得分(最近10轮)', linewidth=2, color='red')
    axes[0, 0].set_ylabel('得分')
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_title('训练进度 - 得分曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 探索率衰减曲线
    eps_values = [max(MIN_EPSILON, 1.0 * (EPSILON_DECAY ** i)) for i in range(len(scores))]
    axes[0, 1].plot(eps_values, color='red', linewidth=2)
    axes[0, 1].set_ylabel('探索率 (ε)')
    axes[0, 1].set_xlabel('训练轮次')
    axes[0, 1].set_title('探索率衰减曲线')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. PER缓冲区大小
    if per_stats['buffer_size']:
        axes[1, 0].plot(per_stats['buffer_size'], color='green', linewidth=2)
        axes[1, 0].axhline(y=REPLAY_MEMORY_SIZE, color='r', linestyle='--', alpha=0.5, label='最大容量')
        axes[1, 0].set_ylabel('缓冲区大小')
        axes[1, 0].set_xlabel('训练轮次')
        axes[1, 0].set_title('PER缓冲区使用情况')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 课程学习阶段变化
    if curriculum_stages:
        axes[1, 1].plot(curriculum_stages, color='purple', linewidth=2, drawstyle='steps-post')
        axes[1, 1].set_ylabel('课程学习阶段')
        axes[1, 1].set_xlabel('训练轮次')
        axes[1, 1].set_title('课程学习阶段变化')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 5. 多目标指标
    if multi_obj_stats['safety']:
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (key, values) in enumerate(multi_obj_stats.items()):
            if values:
                # 计算滑动平均
                window = 10
                if len(values) >= window:
                    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                    axes[2, 0].plot(range(len(smoothed)), smoothed, label=key, color=colors[i], alpha=0.7)
                else:
                    axes[2, 0].plot(values, label=key, color=colors[i], alpha=0.7)
        
        axes[2, 0].set_ylabel('分数')
        axes[2, 0].set_xlabel('训练轮次')
        axes[2, 0].set_title('多目标优化指标')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    # 6. 成功率统计
    success_rates = []
    for i in range(len(scores)):
        window = scores[max(0, i-9):i+1]
        success_rate = sum(1 for s in window if s > 5) / len(window) * 100
        success_rates.append(success_rate)
    
    axes[2, 1].plot(success_rates, color='darkred', linewidth=2)
    axes[2, 1].axhline(y=80, color='g', linestyle='--', alpha=0.5, label='目标成功率80%')
    axes[2, 1].set_ylabel('成功率 (%)')
    axes[2, 1].set_xlabel('训练轮次')
    axes[2, 1].set_title('最近10轮成功率')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.suptitle('高级训练策略 - 综合训练报告', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"最终统计:")
    print(f"  总轮次: {EPISODES}")
    print(f"  最佳得分: {best_score:.2f}")
    print(f"  平均得分: {np.mean(scores):.2f}")
    print(f"  成功率: {(success_count/EPISODES)*100:.1f}%")
    print(f"  最终探索率: {Hyperparameters.EPSILON:.4f}")