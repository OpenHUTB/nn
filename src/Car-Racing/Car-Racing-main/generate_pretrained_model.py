import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import numpy as np
import os

# 创建与原模型结构相同的模型
def create_model():
    # 定义动作空间，与main.py中的配置一致
    action_size_for_steer = [-1.0, -0.5, 0.00, 0.5, 1.0]
    action_size_for_acceleration = [1.0]
    action_size_for_brake = [0.75]
    
    # 计算总动作数
    action_space = []
    for action_space_acc in action_size_for_acceleration:
        for action_space_brake in action_size_for_brake:
            for action_space_steer in action_size_for_steer:
                action_space.append([action_space_steer, action_space_acc, action_space_brake])
    
    action_size = len(action_space)
    
    # 创建与Agent.py中相同的CNN模型结构
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    
    # 编译模型
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.001)
    )
    
    # 为了模拟预训练效果，我们可以进行一些简单的初始化
    # 创建一些模拟数据进行前向传播，帮助初始化权重
    dummy_input = np.random.random((1, 96, 96, 3))
    _ = model.predict(dummy_input)
    
    return model, action_size

# 创建并保存模型
if __name__ == "__main__":
    print("创建预训练模型...")
    model, action_size = create_model()
    
    # 确保data目录存在
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 保存模型权重
    model.save_weights('data/pretrained_model.weights.h5')
    print(f"预训练模型已保存到: data/pretrained_model.weights.h5")
    print(f"模型输出动作数: {action_size}")
    print("模型结构:")
    model.summary()
    print("\n注意: 这是一个初始化的模型，需要进一步训练才能表现良好。")