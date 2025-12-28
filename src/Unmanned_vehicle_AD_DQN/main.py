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

import Hyperparameters
from Environment import *
from Model import *
from Hyperparameters import *


def manage_top_models(agent, score, top_models, max_models=10):
    """
    管理最佳模型，只保留前10个最佳模型
    当有更好的模型时，替换最差的
    """
    import os
    
    # 创建模型文件名
    model_suffix = f"enhanced_top_{score:.2f}"
    model_filename = f'{MODEL_NAME}_{model_suffix}.model'
    model_path = f'models/{model_filename}'
    
    # 检查是否应该保存这个模型
    should_save = False
    model_to_remove = None
    
    if len(top_models) < max_models:
        # 还没有10个模型，直接保存
        should_save = True
    elif score > min(score for score, _ in top_models):
        # 比当前最差的模型更好，替换它
        should_save = True
        # 找到最差的模型
        min_score_idx = min(range(len(top_models)), key=lambda i: top_models[i][0])
        _, model_to_remove = top_models[min_score_idx]
        # 从列表中移除
        top_models.pop(min_score_idx)
    
    if should_save:
        # 保存新模型
        try:
            agent.save_model(model_path)
            top_models.append((score, model_filename))
            print(f"保存最佳模型: 得分={score:.2f}, 排名={len(top_models)}")
        except Exception as e:
            print(f"保存模型失败: {e}")
            print("继续训练，不保存此模型...")
        
        # 删除被替换的模型
        if model_to_remove:
            remove_path = f'models/{model_to_remove}'
            if os.path.exists(remove_path):
                try:
                    # 删除模型目录
                    import shutil
                    if os.path.isdir(remove_path):
                        shutil.rmtree(remove_path)
                    else:
                        os.remove(remove_path)
                    print(f"删除较差模型: {model_to_remove}")
                except Exception as e:
                    print(f"删除模型失败 {model_to_remove}: {e}")
    
    # 按得分排序（降序）
    top_models.sort(key=lambda x: x[0], reverse=True)


def extended_reward_calculation(env, action, reward, done, step_info, obstacle_info=None):
    """
    扩展的奖励计算函数，用于多目标优化
    """
    # 获取车辆状态
    vehicle_location = env.vehicle.get_location()
    velocity = env.vehicle.get_velocity()
    speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
    
    # 计算多目标指标
    metrics = {}
    
    # 1. 安全性指标 - 增强
    min_ped_distance = env.last_ped_distance if hasattr(env, 'last_ped_distance') else float('inf')
    safety_score = 0
    
    if min_ped_distance < 100:
        if min_ped_distance > 15:
            safety_score = 10  # 非常安全
        elif min_ped_distance > 10:
            safety_score = 8   # 安全
        elif min_ped_distance > 6:
            safety_score = 5   # 注意
        elif min_ped_distance > 3:
            safety_score = 2   # 危险
        else:
            safety_score = 0   # 极危险
    
    metrics['safety'] = safety_score
    
    # 2. 效率指标
    progress = (vehicle_location.x + 81) / 236.0  # 从-81到155
    efficiency_score = progress * 100  # 进度百分比
    metrics['efficiency'] = efficiency_score
    
    # 3. 舒适度指标 - 增强
    comfort_score = 5  # 基础分数
    
    # 转向平滑性
    if hasattr(env, 'last_action') and env.last_action in [3, 4]:
        if env.same_steer_counter > 2:  # 连续同向转向
            comfort_score -= 3  # 不舒适
        else:
            comfort_score += 1  # 舒适转向
    
    # 加减速平滑性
    if action in [0, 2]:  # 减速或加速
        if hasattr(env, 'last_action') and env.last_action in [0, 2]:
            comfort_score -= 1  # 连续加减速不舒适
    
    metrics['comfort'] = comfort_score
    
    # 4. 障碍物避让指标（新增）
    if obstacle_info:
        avoidance_score = 0
        min_distance = obstacle_info.get('min_distance', float('inf'))
        warning_level = obstacle_info.get('warning_level', 0)
        avoidance_success = obstacle_info.get('avoidance_success', False)
        
        if avoidance_success:
            avoidance_score = 8  # 成功避让
        elif min_distance > 12:
            avoidance_score = 6  # 保持安全距离
        elif min_distance > 8:
            avoidance_score = 4  # 中等距离
        elif min_distance > 5:
            avoidance_score = 2  # 近距离但安全
        else:
            avoidance_score = 0  # 危险
        
        metrics['obstacle_avoidance'] = avoidance_score
    
    # 5. 规则遵循指标
    if 18 <= speed_kmh <= 38:
        rule_score = 1.0
    elif 15 <= speed_kmh < 18 or 38 < speed_kmh <= 42:
        rule_score = 0.7
    else:
        rule_score = 0.3
    
    metrics['rule_following'] = rule_score
    
    # 6. 特殊事件
    metrics['collision'] = len(env.collision_history) > 0
    metrics['off_road'] = vehicle_location.x < -90 or abs(vehicle_location.y + 195) > 30
    metrics['dangerous_action'] = speed_kmh > 45 and action in [3, 4]  # 高速急转
    metrics['near_miss'] = min_ped_distance < 3.0 and min_ped_distance > 1.0  # 惊险避让
    
    return metrics


def train_enhanced_agent():
    """增强版训练函数"""
    FPS = 60
    ep_rewards = [-200]
    
    # GPU内存配置
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))
    
    # 创建目录
    for dir_name in ['models', 'expert_data', 'logs', 'training_stats']:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
    
    # 创建智能体和环境 - 启用所有增强功能
    print("=" * 60)
    print("初始化增强版智能体和环境")
    print("=" * 60)
    
    agent = EnhancedDQNAgent(
        use_dueling=True, 
        use_per=True,
        use_curriculum=True,
        use_multi_objective=True,
        use_attention=True,
        use_enhanced_model=True  # 使用增强版模型
    )
    
    env = CarEnv(obstacle_detection_mode='advanced')
    
    # 设置训练策略
    agent.setup_training_strategies(env)
    
    # 模仿学习预训练（可选）
    use_imitation_pretraining = True
    
    if use_imitation_pretraining:
        print("=" * 60)
        print("开始模仿学习预训练阶段")
        print("=" * 60)
        
        # 检查是否有现有的专家数据
        expert_files = glob.glob("expert_data/*enhanced*.pkl")
        if expert_files:
            latest_expert = max(expert_files, key=os.path.getctime)
            agent.imitation_manager.load_expert_data(latest_expert)
        else:
            # 收集新的专家数据，特别关注避障
            print("未找到专家数据，开始收集...")
            agent.imitation_manager.collect_expert_demonstration(
                env, 
                num_episodes=3,
                focus_avoidance=True
            )
        
        # 使用行为克隆进行预训练，专注避障
        agent.model = agent.imitation_manager.pretrain_with_behavioral_cloning(
            agent.model, 
            epochs=20,
            focus_avoidance=True
        )
        agent.target_model.set_weights(agent.model.get_weights())
        print("模仿学习预训练完成!")
    
    # 启动训练线程
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
    
    # 预热Q网络
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))
    
    # 训练统计变量
    best_score = -float('inf')
    success_count = 0
    scores = []
    avg_scores = []
    
    # 最佳模型管理器 - 只保留前10个最佳模型
    top_models = []  # 存储 (score, filename) 元组的列表
    
    # 避障统计
    avoidance_stats = {
        'success_rate': [],
        'near_misses': [],
        'collisions': [],
        'encounters': []
    }
    
    # 多目标统计
    multi_obj_stats = {
        'safety': [],
        'efficiency': [],
        'comfort': [],
        'obstacle_avoidance': []
    }
    
    # 课程学习阶段记录
    curriculum_stages = []
    curriculum_progress = []
    
    # 迭代训练轮次
    print("=" * 60)
    print("开始主要训练阶段")
    print("=" * 60)
    
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []
        agent.tensorboard.step = episode
        
        # 应用课程学习配置
        if agent.curriculum_manager:
            config = agent.curriculum_manager.get_current_config()
            stage_info = agent.curriculum_manager.get_stage_info()
            
            # 重置环境时应用课程配置
            current_state = env.reset(episode, curriculum_config=config)
            
            print(f"课程学习 - 阶段 {stage_info['stage']}: {stage_info['name']}")
            print(f"  进度: {stage_info['progress']:.2%}, 行人数量: {stage_info['pedestrian_total']}")
            
            curriculum_stages.append(stage_info['stage'])
            curriculum_progress.append(stage_info['progress'])
        else:
            current_state = env.reset(episode)
        
        # 重置每轮统计
        score = 0
        step = 1
        
        # 多目标指标记录
        episode_metrics = {
            'safety': [],
            'efficiency': [],
            'comfort': [],
            'obstacle_avoidance': []
        }
        
        # 重置完成标志
        done = False
        episode_start = time.time()
        
        # 应用课程学习的最大步数限制
        if agent.curriculum_manager:
            config = agent.curriculum_manager.get_current_config()
            max_steps_per_episode = config['max_episode_steps']
        else:
            max_steps_per_episode = SECONDS_PER_EPISODE * FPS
        
        # 步进循环
        while not done and step < max_steps_per_episode:
            # 选择动作策略
            if np.random.random() > Hyperparameters.EPSILON:
                # 从Q网络获取动作
                qs = agent.get_qs(current_state)
                action = np.argmax(qs)
                
                # 减少打印频率，仅在调试时启用
                if episode % 50 == 0 and step % 30 == 0:
                    print(f'Step {step}: Q值 [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}, {qs[3]:>5.2f}, {qs[4]:>5.2f}] 选择动作 {action}')
            else:
                # 随机选择动作（探索）
                action = np.random.randint(0, 5)
                time.sleep(1 / FPS)  # 匹配帧率
            
            # 执行动作并获取结果
            new_state, reward, done, extra_info = env.step(action)
            
            # 提取障碍物信息
            obstacle_info = extra_info.get('obstacle_info', {}) if extra_info else {}
            
            # 计算多目标指标
            if agent.multi_objective_optimizer:
                step_info = {'step': step, 'action': action}
                metrics = extended_reward_calculation(env, action, reward, done, step_info, obstacle_info)
                
                # 记录指标
                for key in episode_metrics:
                    if key in metrics:
                        episode_metrics[key].append(metrics[key])
                
                # 使用多目标优化器计算综合奖励
                composite_reward = agent.multi_objective_optimizer.compute_composite_reward(
                    metrics, 
                    obstacle_info
                )
                reward = composite_reward  # 使用综合奖励
            
            score += reward
            
            # 判断是否为避障相关经验
            is_obstacle_experience = (
                obstacle_info.get('min_distance', float('inf')) < 10.0 or
                obstacle_info.get('warning_level', 0) > 0
            )
            
            # 判断是否为成功经验
            is_success_experience = (reward > 2.0 and not done)
            
            # 更新经验回放（特别标记重要经验）
            agent.update_replay_memory(
                (current_state, action, reward, new_state, done),
                is_obstacle=is_obstacle_experience,
                is_success=is_success_experience
            )
            
            current_state = new_state
            step += 1
            
            if done:
                break
        
        # 本轮结束
        
        # 更新避障统计 - 在清理之前获取信息
        env_info = env.get_environment_info()
        if env_info['obstacle_warning_level'] > 0:
            avoidance_stats['encounters'].append(1)
        else:
            avoidance_stats['encounters'].append(0)
        
        avoidance_stats['success_rate'].append(env_info['successful_avoidance'])
        avoidance_stats['collisions'].append(1 if env_info['collision_count'] > 0 else 0)
        
        # 更新课程学习
        success = score > 8  # 提高成功阈值
        obstacle_avoidance_score = env_info['successful_avoidance'] / max(1, env_info['pedestrian_count'])
        
        if agent.curriculum_manager:
            stage_changed, change_type = agent.curriculum_manager.update_stage(
                success, 
                score, 
                obstacle_avoidance_score
            )
            if stage_changed:
                stage_info = agent.curriculum_manager.get_stage_info()
                print(f"课程学习阶段已更新: {stage_info['name']} (变更类型: {change_type})")
        
        # 计算本轮平均指标
        avg_metrics = {}
        for key, values in episode_metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
                if key in multi_obj_stats:
                    multi_obj_stats[key].append(avg_metrics[key])
        
        env.cleanup_actors()
        
        # 更新多目标优化器权重
        if agent.multi_objective_optimizer and episode % 10 == 0:
            training_stage = agent.curriculum_manager.current_stage if agent.curriculum_manager else 0
            agent.multi_objective_optimizer.adjust_weights(avg_metrics, training_stage)
            
            if episode % 50 == 0:
                print(agent.multi_objective_optimizer.get_performance_report())
        
        # 更新成功计数
        if success:
            success_count += 1
        
        # 管理最佳模型（只保留前10个）
        manage_top_models(agent, score, top_models)
        
        # 定期保存检查点
        if episode % 50 == 0:
            model_suffix = f"enhanced_checkpoint_ep{episode}"
            agent.save_model(f'models/{MODEL_NAME}_{model_suffix}.model')
        
        # 记录得分统计
        scores.append(score)
        avg_scores.append(np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores))
        
        # 定期聚合统计信息
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = np.mean(scores[-AGGREGATE_STATS_EVERY:]) if len(scores) >= AGGREGATE_STATS_EVERY else np.mean(scores)
            min_reward = min(scores[-AGGREGATE_STATS_EVERY:]) if len(scores) >= AGGREGATE_STATS_EVERY else min(scores)
            max_reward = max(scores[-AGGREGATE_STATS_EVERY:]) if len(scores) >= AGGREGATE_STATS_EVERY else max(scores)
            
            # 准备TensorBoard统计
            stats_dict = {
                'reward_avg': average_reward, 
                'reward_min': min_reward, 
                'reward_max': max_reward,
                'epsilon': Hyperparameters.EPSILON,
                'success_rate': success_count / episode if episode > 0 else 0
            }
            
            # 添加避障统计
            if avoidance_stats['encounters']:
                encounter_rate = np.mean(avoidance_stats['encounters'][-AGGREGATE_STATS_EVERY:]) if len(avoidance_stats['encounters']) >= AGGREGATE_STATS_EVERY else np.mean(avoidance_stats['encounters'])
                stats_dict['obstacle_encounter_rate'] = encounter_rate
            
            # 添加多目标指标
            if agent.multi_objective_optimizer:
                for obj in ['safety', 'efficiency', 'comfort', 'obstacle_avoidance']:
                    if multi_obj_stats[obj]:
                        recent_avg = np.mean(multi_obj_stats[obj][-AGGREGATE_STATS_EVERY:]) if len(multi_obj_stats[obj]) >= AGGREGATE_STATS_EVERY else np.mean(multi_obj_stats[obj])
                        stats_dict[f'{obj}_score'] = recent_avg
            
            agent.tensorboard.update_stats(**stats_dict)
        
        # 打印训练信息
        info_str = f'轮次: {episode:3d}, 得分: {score:6.2f}, 成功: {success_count:3d}, 避障成功: {env_info["successful_avoidance"]}'
        if agent.curriculum_manager:
            stage_info = agent.curriculum_manager.get_stage_info()
            info_str += f', 阶段: {stage_info["stage"]}({stage_info["name"]})'
        print(info_str)
        
        # 衰减探索率
        if Hyperparameters.EPSILON > Hyperparameters.MIN_EPSILON:
            Hyperparameters.EPSILON *= Hyperparameters.EPSILON_DECAY
            Hyperparameters.EPSILON = max(Hyperparameters.MIN_EPSILON, Hyperparameters.EPSILON)
    
    # 训练结束
    agent.terminate = True
    trainer_thread.join()
    
    # 保存最终模型
    if len(scores) > 0:
        final_max_reward = max(scores[-AGGREGATE_STATS_EVERY:] if len(scores) >= AGGREGATE_STATS_EVERY else scores)
        final_avg_reward = np.mean(scores[-AGGREGATE_STATS_EVERY:] if len(scores) >= AGGREGATE_STATS_EVERY else scores)
        final_min_reward = min(scores[-AGGREGATE_STATS_EVERY:] if len(scores) >= AGGREGATE_STATS_EVERY else scores)
        
        model_suffix = "enhanced_final"
        agent.save_model(
            f'models/{MODEL_NAME}_{model_suffix}__{final_max_reward:_>7.2f}max_{final_avg_reward:_>7.2f}avg_{final_min_reward:_>7.2f}min__{int(time.time())}.model'
        )
        
        # 保存训练统计数据
        training_stats = {
            'scores': scores,
            'avg_scores': avg_scores,
            'multi_obj_stats': multi_obj_stats,
            'avoidance_stats': avoidance_stats,
            'curriculum_stages': curriculum_stages,
            'curriculum_progress': curriculum_progress,
            'top_models': top_models,  # 添加最佳模型信息
            'final_scores': {
                'max': final_max_reward,
                'avg': final_avg_reward,
                'min': final_min_reward
            },
            'training_config': {
                'episodes': EPISODES,
                'epsilon_decay': EPSILON_DECAY,
                'model_name': MODEL_NAME
            }
        }
        
        timestamp = int(time.time())
        stats_file = f'training_stats/training_stats_enhanced_{timestamp}.pkl'
        
        with open(stats_file, 'wb') as f:
            pickle.dump(training_stats, f)
        
        print(f"训练统计数据已保存到: {stats_file}")
        
        # 生成训练报告
        generate_training_report(training_stats)
    
    print("\n" + "="*60)
    print("增强版训练完成!")
    print("="*60)
    print(f"最终统计:")
    print(f"  总轮次: {EPISODES}")
    print(f"  最佳得分: {best_score:.2f}")
    print(f"  平均得分: {np.mean(scores):.2f}")
    print(f"  成功率: {(success_count/EPISODES)*100:.1f}%")
    print(f"  最终探索率: {Hyperparameters.EPSILON:.4f}")
    print(f"  总避障成功次数: {sum(avoidance_stats['success_rate'])}")
    
    return agent


def generate_training_report(stats):
    """生成训练报告图表"""
    try:
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # 1. 得分曲线
        axes[0, 0].plot(stats['scores'], label='每轮得分', alpha=0.6, linewidth=1)
        axes[0, 0].plot(stats['avg_scores'], label='平均得分(最近10轮)', linewidth=2, color='red')
        axes[0, 0].set_ylabel('得分')
        axes[0, 0].set_xlabel('训练轮次')
        axes[0, 0].set_title('训练进度 - 得分曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 探索率衰减曲线
        eps_values = [max(MIN_EPSILON, 1.0 * (EPSILON_DECAY ** i)) for i in range(len(stats['scores']))]
        axes[0, 1].plot(eps_values, color='red', linewidth=2)
        axes[0, 1].set_ylabel('探索率 (ε)')
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_title('探索率衰减曲线')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 多目标指标
        colors = ['blue', 'green', 'orange', 'purple']
        obj_keys = ['safety', 'efficiency', 'comfort', 'obstacle_avoidance']
        
        for i, key in enumerate(obj_keys):
            if key in stats['multi_obj_stats'] and stats['multi_obj_stats'][key]:
                values = stats['multi_obj_stats'][key]
                window = 10
                if len(values) >= window:
                    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                    axes[1, 0].plot(range(len(smoothed)), smoothed, label=key, color=colors[i], alpha=0.7)
                else:
                    axes[1, 0].plot(values, label=key, color=colors[i], alpha=0.7)
        
        axes[1, 0].set_ylabel('分数')
        axes[1, 0].set_xlabel('训练轮次')
        axes[1, 0].set_title('多目标优化指标')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 课程学习阶段变化
        if 'curriculum_stages' in stats and stats['curriculum_stages']:
            axes[1, 1].plot(stats['curriculum_stages'], color='purple', linewidth=2, drawstyle='steps-post')
            axes[1, 1].set_ylabel('课程学习阶段')
            axes[1, 1].set_xlabel('训练轮次')
            axes[1, 1].set_title('课程学习阶段变化')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 避障成功率
        if 'avoidance_stats' in stats and 'success_rate' in stats['avoidance_stats']:
            success_rates = []
            success_values = stats['avoidance_stats']['success_rate']
            
            for i in range(len(success_values)):
                window = success_values[max(0, i-9):i+1]
                if window:
                    success_rates.append(np.mean(window))
            
            axes[2, 0].plot(success_rates, color='darkgreen', linewidth=2)
            axes[2, 0].axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='目标成功率70%')
            axes[2, 0].set_ylabel('平均避障成功次数')
            axes[2, 0].set_xlabel('训练轮次')
            axes[2, 0].set_title('避障性能')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 成功率统计
        success_rates = []
        scores = stats['scores']
        
        for i in range(len(scores)):
            window = scores[max(0, i-9):i+1]
            success_rate = sum(1 for s in window if s > 8) / len(window) * 100
            success_rates.append(success_rate)
        
        axes[2, 1].plot(success_rates, color='darkred', linewidth=2)
        axes[2, 1].axhline(y=70, color='g', linestyle='--', alpha=0.5, label='目标成功率70%')
        axes[2, 1].set_ylabel('成功率 (%)')
        axes[2, 1].set_xlabel('训练轮次')
        axes[2, 1].set_title('最近10轮成功率')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.suptitle('增强版训练策略 - 综合训练报告', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        timestamp = int(time.time())
        chart_file = f'training_stats/training_chart_enhanced_{timestamp}.png'
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"训练图表已保存到: {chart_file}")
        
        # 显示最佳模型总结
        print("\n" + "="*60)
        print("最佳模型总结 (前10个)")
        print("="*60)
        top_models = stats.get('top_models', [])
        if top_models:
            for i, (score, filename) in enumerate(top_models, 1):
                print(f"{i:2d}. 得分: {score:8.2f} - {filename}")
        else:
            print("没有保存任何模型")
        print("="*60)
        
    except Exception as e:
        print(f"生成训练报告时出错: {e}")


if __name__ == '__main__':
    train_enhanced_agent()