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


MODEL_PATH = r'D:\Work\T_Unmanned_vehicle_AD_DQN\models\YY_Optimized___290.14max___97.16avg___13.42min__1764553908.model'  # 请替换为实际的最佳模型路径

if __name__ == '__main__':

    # GPU内存配置
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # 加载训练好的模型
    model = load_model(MODEL_PATH)

    # 创建测试环境 - 禁用摄像头预览，让CARLA主窗口显示
    env = CarEnv()
    env.SHOW_CAM = False  # 关闭小窗口预览

    # FPS计数器 - 保存最近60帧的时间
    fps_counter = deque(maxlen=60)

    # 初始化预测 - 第一次预测需要较长时间进行初始化
    model.predict(np.ones((1, env.im_height, env.im_width, 3)))

    print("开始测试！请查看CARLA窗口观看智能体运行...")
    print("按Ctrl+C停止测试")

    # 循环测试多个episode
    episode_count = 0
    try:
        while True:
            episode_count += 1
            print(f'\n开始第 {episode_count} 个测试轮次')

            # 重置环境并获取初始状态
            current_state = env.reset()
            env.collision_hist = []  # 重置碰撞历史

            done = False
            total_reward = 0
            step_count = 0

            # 单次episode内的循环
            while True:

                # FPS计数开始
                step_start = time.time()

                # 基于当前观察空间预测动作
                qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
                action = np.argmax(qs)  # 选择Q值最大的动作

                # 执行环境步进
                new_state, reward, done, _ = env.step(action)

                # 更新当前状态
                current_state = new_state
                total_reward += reward
                step_count += 1

                # 如果完成（碰撞等），结束当前episode
                if done:
                    break

                # 计算帧时间，更新FPS计数器，打印统计信息
                frame_time = time.time() - step_start
                fps_counter.append(frame_time)
                if step_count % 10 == 0:  # 每10步打印一次信息
                    print(f'轮次 {episode_count} | 步数: {step_count} | FPS: {len(fps_counter)/sum(fps_counter):>4.1f} | 动作: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}, {qs[3]:>5.2f}, {qs[4]:>5.2f}] {action} | 奖励: {reward:.2f} | 累计奖励: {total_reward:.2f}')

            # episode结束时显示结果并销毁所有actor
            result = "成功到达终点!" if reward > 5 else "发生碰撞或失败"
            print(f'第 {episode_count} 轮结束: {result} | 总步数: {step_count} | 总奖励: {total_reward:.2f}')
            
            env.cleanup_actors()
                
            # 短暂暂停后开始下一轮
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        # 清理环境
        print("清理环境...")
        env.cleanup_actors()