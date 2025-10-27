import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np

# 定义与原模型相同的动作空间
def get_action_space():
    action_size_for_steer = [-1.0, -0.5, 0.00, 0.5, 1.0]
    action_size_for_acceleration = [1.0]
    action_size_for_brake = [0.75]
    
    # 计算总动作数
    action_space = []
    for action_space_acc in action_size_for_acceleration:
        for action_space_brake in action_size_for_brake:
            for action_space_steer in action_size_for_steer:
                action_space.append([action_space_steer, action_space_acc, action_space_brake])
    
    return action_space

# 创建与原模型相同的结构
def create_model():
    action_space = get_action_space()
    action_size = len(action_space)
    
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
    
    return model, action_space

# 加载预训练模型并进行测试
def load_and_test_model():
    print("加载预训练模型...")
    model, action_space = create_model()
    
    # 加载预训练权重
    try:
        model.load_weights('data/pretrained_model.weights.h5')
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 模拟使用模型
    print("\n测试模型推理:")
    # 创建一个随机的图像输入(模拟游戏画面)
    dummy_image = np.random.random((1, 96, 96, 3))
    
    # 使用模型预测动作
    q_values = model.predict(dummy_image)
    action_index = np.argmax(q_values[0])
    chosen_action = action_space[action_index]
    
    print(f"预测的Q值: {q_values[0]}")
    print(f"选择的动作索引: {action_index}")
    print(f"选择的动作: [转向={chosen_action[0]}, 加速={chosen_action[1]}, 刹车={chosen_action[2]}]")
    
    print("\n如何在训练中使用此模型:")
    print("- 使用命令: python main.py --load-model=data/pretrained_model.weights.h5 --start-episode=1")
    print("- 注意：这是一个初始化的模型，需要进一步训练才能获得良好表现")

if __name__ == "__main__":
    load_and_test_model()