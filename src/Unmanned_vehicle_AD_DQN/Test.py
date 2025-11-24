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


MODEL_PATH = r'D:\Work\T_Unmanned_vehicle_AD_DQN\models\YY_best_74.00.model'  # 请替换为实际的最佳模型路径

if __name__ == '__main__':

    # GPU内存配置
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # 加载训练好的模型
    model = load_model(MODEL_PATH)

    # 创建测试环境
    env = CarEnv()

    # FPS计数器 - 保存最近60帧的时间
    fps_counter = deque(maxlen=60)

    # 初始化预测 - 第一次预测需要较长时间进行初始化
    model.predict(np.ones((1, env.im_height, env.im_width, 3)))

    # 循环测试多个episode
    while True:

        print('开始新的测试轮次')

        # 重置环境并获取初始状态
        current_state = env.reset()
        env.collision_hist = []  # 重置碰撞历史

        done = False

        # 单次episode内的循环
        while True:

            # FPS计数开始
            step_start = time.time()

            # 显示当前帧
            cv2.imshow(f'智能体预览', current_state)
            cv2.waitKey(1)

            # 基于当前观察空间预测动作
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            action = np.argmax(qs)  # 选择Q值最大的动作

            # 执行环境步进
            new_state, reward, done, _ = env.step(action)

            # 更新当前状态
            current_state = new_state

            # 如果完成（碰撞等），结束当前episode
            if done:
                break

            # 计算帧时间，更新FPS计数器，打印统计信息
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'智能体: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | 动作: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action} | 奖励: {reward}')

        # episode结束时销毁所有actor
        for actor in env.actor_list:
            actor.destroy()