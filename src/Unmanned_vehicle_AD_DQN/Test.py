# Test.py
import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import tensorflow.keras.backend as backend
from tensorflow.keras.models import load_model
from Environment import CarEnv, MEMORY_FRACTION
from Hyperparameters import *


MODEL_PATH = r'D:\Work\T_Unmanned_vehicle_AD_DQN\models\YY_Optimized_best_541.48.model'  # 新模型路径

def get_safe_action_improved(model, state, env, previous_action, uncertainty_threshold=1.0):
    """
    改进的安全动作选择，结合模型预测、安全规则和不确定性估计
    """
    # 模型预测
    state_normalized = np.array(state).reshape(-1, *state.shape) / 255
    qs = model.predict(state_normalized, verbose=0)[0]
    
    # 计算不确定性（如果模型有多个输出头）
    uncertainty = 0.0
    if hasattr(model, 'output'):
        if isinstance(model.output, list):
            # 如果是Dueling DQN，可以分别获取价值和优势
            predictions = model.predict(state_normalized, verbose=0)
            if isinstance(predictions, list):
                value = predictions[0][0] if len(predictions) > 0 else 0
                advantage = predictions[1][0] if len(predictions) > 1 else np.zeros(5)
                # 计算不确定性作为价值和优势的差异
                uncertainty = np.std(advantage)
    
    # 安全规则：高不确定性时更加保守
    if uncertainty > uncertainty_threshold:
        # 降低激进动作的Q值
        qs[2] *= 0.5  # 降低加速倾向
        qs[3] *= 0.7  # 降低左转倾向
        qs[4] *= 0.7  # 降低右转倾向
    
    # 如果有建议的避让动作（来自环境），优先考虑
    if hasattr(env, 'suggested_action') and env.suggested_action is not None:
        suggested_q = qs[env.suggested_action]
        qs[env.suggested_action] += 2.0  # 大幅提高建议动作的Q值
        print(f"安全建议: 执行动作 {env.suggested_action} 以避让行人")
        env.suggested_action = None  # 重置
    
    # 避免频繁切换动作（平滑性）
    if previous_action in [3, 4]:  # 如果是转向动作
        qs[previous_action] += 0.5  # 提高继续当前转向的倾向
    
    # 防止过度转向
    if env.same_steer_counter > 3:  # 连续同向转向超过3次
        qs[previous_action] -= 1.0  # 降低当前转向动作的Q值
    
    # 速度相关的动作调整
    velocity = env.vehicle.get_velocity()
    speed_kmh = 3.6 * np.linalg.norm([velocity.x, velocity.y, velocity.z])
    
    if speed_kmh > 40:  # 高速时更加谨慎
        qs[2] *= 0.8  # 降低加速倾向
    elif speed_kmh < 10:  # 低速时鼓励加速
        qs[0] *= 0.7  # 降低减速倾向
        qs[2] *= 1.2  # 提高加速倾向
    
    # 选择动作
    action = np.argmax(qs)
    
    # 安全检查：避免危险动作
    if speed_kmh > 35 and action in [3, 4]:  # 高速时避免急转
        # 检查是否有更安全的替代动作
        alternative_actions = [1, 0, 2]  # 保持、减速、加速
        safe_qs = [qs[a] for a in alternative_actions]
        if max(safe_qs) > qs[action] * 0.8:  # 如果安全动作的Q值接近
            action = alternative_actions[np.argmax(safe_qs)]
            print(f"安全调整: 高速时避免急转，选择动作 {action}")
    
    return action, qs, uncertainty


if __name__ == '__main__':
    # GPU内存配置
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # 加载训练好的模型
    print(f"加载模型: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH, custom_objects={'Add': tf.keras.layers.Add, 
                                                      'Subtract': tf.keras.layers.Subtract,
                                                      'Lambda': tf.keras.layers.Lambda})
    except:
        model = load_model(MODEL_PATH)
    
    print("模型加载成功!")
    print(f"模型架构: {model.layers[-1].name}")
    
    # 检查是否是Dueling DQN
    is_dueling = any('value' in layer.name or 'advantage' in layer.name for layer in model.layers)
    print(f"模型类型: {'Dueling DQN' if is_dueling else 'Standard DQN'}")

    # 创建测试环境
    env = CarEnv()
    env.SHOW_CAM = False  # 关闭小窗口预览

    # 性能统计
    fps_counter = deque(maxlen=60)
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    total_episodes = 0

    # 初始化预测
    model.predict(np.ones((1, env.im_height, env.im_width, 3)), verbose=0)

    print("\n" + "="*60)
    print("开始测试改进的DQN模型!")
    print("="*60)
    print("请查看CARLA窗口观看智能体运行...")
    print("按Ctrl+C停止测试")
    print(f"模型类型: {'Dueling DQN with PER' if is_dueling else 'Standard DQN'}")

    # 循环测试多个episode
    episode_count = 0
    previous_action = 1  # 初始动作为保持
    
    try:
        while True:
            episode_count += 1
            total_episodes += 1
            print(f'\n{"="*40}')
            print(f'开始第 {episode_count} 个测试轮次')
            print(f'{"="*40}')

            # 重置环境并获取初始状态
            # 测试时使用正常难度（相当于训练的第3阶段）
            current_state = env.reset(401)  # 401表示使用正常难度
            env.collision_hist = []  # 重置碰撞历史

            done = False
            total_reward = 0
            step_count = 0
            max_steps = SECONDS_PER_EPISODE * 60  # 最大步数限制

            # 单次episode内的循环
            while not done and step_count < max_steps:
                # FPS计数开始
                step_start = time.time()

                # 基于当前观察空间预测动作（使用改进的安全版本）
                action, qs, uncertainty = get_safe_action_improved(
                    model, current_state, env, previous_action
                )
                previous_action = action

                # 执行环境步进
                new_state, reward, done, _ = env.step(action)

                # 更新当前状态
                current_state = new_state
                total_reward += reward
                step_count += 1

                # 计算帧时间，更新FPS计数器
                frame_time = time.time() - step_start
                fps_counter.append(frame_time)
                
                # 每20步打印一次详细信息
                if step_count % 20 == 0:
                    fps = len(fps_counter)/sum(fps_counter) if fps_counter else 0
                    speed_vector = env.vehicle.get_velocity()
                    speed_kmh = 3.6 * np.linalg.norm([speed_vector.x, speed_vector.y, speed_vector.z])
                    
                    print(f'轮次 {episode_count} | 步数: {step_count:3d} | FPS: {fps:4.1f} | '
                          f'速度: {speed_kmh:4.1f} km/h | '
                          f'动作: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}, {qs[3]:>5.2f}, {qs[4]:>5.2f}] {action} | '
                          f'奖励: {reward:5.2f} | 累计: {total_reward:6.2f}')

                # 如果完成（碰撞等），结束当前episode
                if done:
                    break

            # episode结束时显示结果
            result = "成功到达终点!" if reward > 5 else "发生碰撞或失败"
            success = reward > 5
            
            if success:
                success_count += 1
                print(f"✓ 第 {episode_count} 轮: {result}")
            else:
                print(f"✗ 第 {episode_count} 轮: {result}")
                
            print(f'总步数: {step_count} | 总奖励: {total_reward:.2f}')
            
            # 记录统计
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            
            # 显示统计摘要
            if len(episode_rewards) >= 5:
                avg_reward = np.mean(episode_rewards[-5:])
                avg_steps = np.mean(episode_lengths[-5:])
                success_rate = (success_count / 5) * 100 if len(episode_rewards) >= 5 else 0
                
                print(f"\n最近5轮统计:")
                print(f"  平均奖励: {avg_reward:.2f}")
                print(f"  平均步数: {avg_steps:.1f}")
                print(f"  成功率: {success_rate:.1f}%")
            
            env.cleanup_actors()
                
            # 短暂暂停后开始下一轮
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        # 显示最终统计
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
        
        if total_episodes > 0:
            success_rate = (success_count / total_episodes) * 100
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_steps = np.mean(episode_lengths) if episode_lengths else 0
            
            print(f"总测试轮次: {total_episodes}")
            print(f"成功次数: {success_count}")
            print(f"成功率: {success_rate:.1f}%")
            print(f"平均奖励: {avg_reward:.2f}")
            print(f"平均步数: {avg_steps:.1f}")
            print(f"模型类型: {'Dueling DQN with PER' if is_dueling else 'Standard DQN'}")
        
        # 清理环境
        print("\n清理环境...")
        env.cleanup_actors()