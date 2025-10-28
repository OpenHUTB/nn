import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息
import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from braking_dqn import CarEnv, MEMORY_FRACTION
import carla
from carla import Transform 
from carla import Location
from carla import Rotation

town2 = {1: [80, 306.6, 5, 0], 2:[150,306.6]}
curves = [0, town2]

epsilon = 0
MODEL_PATH = "models/Braking___282.model"

def setup_tensorflow():
    """设置 TensorFlow 2.x 配置"""
    # 设置日志级别减少输出
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print(f"TensorFlow 版本: {tf.__version__}")
    
    # GPU 配置
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置GPU内存按需增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ 找到 {len(gpus)} 个GPU，已启用内存增长")
        except RuntimeError as e:
            print(f"⚠️ GPU设置错误: {e}")
            # 如果GPU设置失败，回退到CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("使用CPU运行")
    else:
        print("ℹ️ 未找到GPU，使用CPU运行")

def safe_load_model(model_path):
    """安全加载模型，处理可能的兼容性问题"""
    try:
        print(f"尝试加载模型: {model_path}")
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None
            
        # 尝试不同的加载方式
        try:
            # 方式1: 直接加载
            model = load_model(model_path)
            print(f"✅ 成功加载模型: {model_path}")
            return model
        except Exception as e:
            print(f"标准加载失败，尝试自定义对象加载: {e}")
            try:
                # 方式2: 使用 compile=False
                model = load_model(model_path, compile=False)
                # 重新编译模型
                model.compile(optimizer='adam', loss='mse')
                print(f"✅ 使用 compile=False 成功加载模型: {model_path}")
                return model
            except Exception as e2:
                print(f"❌ 所有加载方式都失败: {e2}")
                return None
                
    except Exception as e:
        print(f"❌ 加载模型时发生错误 {model_path}: {e}")
        return None

def preprocess_state_for_prediction(state_data):
    """预处理状态数据用于模型预测"""
    try:
        if isinstance(state_data, list):
            state_array = np.array(state_data)
        else:
            state_array = state_data
        
        # 确保正确的形状
        if len(state_array.shape) == 1:
            state_array = state_array.reshape(1, -1)
        
        return state_array
    except Exception as e:
        print(f"状态预处理错误: {e}")
        # 返回默认状态
        return np.array([[0, 0]])

if __name__ == '__main__':
    
    FPS = 60
    EPISODES = 1

    # 设置 TensorFlow
    setup_tensorflow()

    # 加载模型
    print("\n" + "="*50)
    print("加载刹车模型")
    print("="*50)
    
    model = safe_load_model(MODEL_PATH)
    
    # 如果模型加载失败，创建新的模型
    if model is None:
        print("❌ 无法加载模型，程序退出")
        exit(1)
    
    print(f"✅ 模型加载完成:")
    print(f"   - 输入形状: {model.input_shape}")
    print(f"   - 输出形状: {model.output_shape}")

    # 创建环境
    print("\n初始化CARLA环境...")
    env = CarEnv()

    # 用于FPS计算 - 保持最近60帧的时间
    fps_counter = deque(maxlen=60)

    # 初始化预测 - 第一次预测需要初始化时间
    print("预热模型...")
    try:
        # 使用正确的预处理
        dummy_state = preprocess_state_for_prediction([0, 0])
        model.predict(dummy_state, verbose=0)
        print("✅ 模型预热完成")
    except Exception as e:
        print(f"⚠️ 模型预热警告: {e}")

    # 循环 episodes
    for episode in range(EPISODES):
        print(f'\n{"="*50}')
        print(f'开始 Episode {episode + 1}/{EPISODES}')
        print(f'{"="*50}')

        # 设置路径点
        try:
            env.waypoint = env.client.get_world().get_map().get_waypoint(
                Location(x=curves[env.curves][1][0], y=curves[env.curves][1][1], z=curves[env.curves][1][2]), 
                project_to_road=True
            )
        except Exception as e:
            print(f"⚠️ 设置路径点错误: {e}")

        # 重置环境并获取初始状态
        current_state = env.reset()
        if hasattr(env, 'collision_hist'):
            env.collision_hist = []
        
        # 生成轨迹
        if hasattr(env, 'trajectory'):
            env.trajectory()
        
        done = False
        step_count = 0

        # 循环步骤
        while not done:
            step_count += 1
            
            # FPS 计数器
            step_start = time.time()

            # 显示当前帧（可选）
            if len(current_state) > 0 and isinstance(current_state[0], np.ndarray):
                print("当前帧")
                cv2.imshow(f'Agent - preview', current_state[0])
                cv2.waitKey(1)

            # 预测基于当前观察空间的动作
            action = None
            qs = None
            
            try:
                if np.random.random() > epsilon:
                    # 从 Q 表获取动作
                    state_for_prediction = preprocess_state_for_prediction(current_state)
                    qs = model.predict(state_for_prediction, verbose=0)[0]
                    action = np.argmax(qs)
                else:
                    # 获取随机动作
                    action = np.random.randint(0, 2)
                    # 这不需要时间，所以我们添加匹配 60 FPS 的延迟（上面的预测需要更长时间）
                    if len(fps_counter) > 0:
                        time.sleep(sum(fps_counter) / len(fps_counter))
                    else:
                        time.sleep(1/FPS)

            except Exception as e:
                print(f"❌ 预测错误: {e}")
                action = 0  # 默认安全动作
                qs = np.zeros(2)  # 默认 Q 值
            
            
            # 环境步骤（额外的标志通知环境不要因时间限制而中断episode）
            try:
        
                new_state, reward, done, waypoint = env.step(action, current_state)

                # 设置下一步的当前状态
                current_state = new_state
                
                # 保存路径点
                if hasattr(env, 'waypoint'):
                    env.waypoint = waypoint
                
            except Exception as e:
                print(f"❌ 环境步骤错误: {e}")
                done = True
            
            # 如果完成 - 代理崩溃，中断episode
            if done:
                print(f"Episode {episode + 1} 完成，步数: {step_count}")
                break
            
            # 测量步骤时间，添加到deque，然后打印最近60帧的平均FPS、q值和采取的动作
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            
            if len(fps_counter) > 0:
                current_fps = len(fps_counter) / sum(fps_counter)
            else:
                current_fps = 0
            
            # 安全地打印Q值
            try:
                if qs is not None:
                    qs_display = f"[{qs[0]:>5.2f}, {qs[1]:>5.2f}]"
                else:
                    qs_display = "[N/A, N/A]"
                    
                print(f'Step: {step_count:>3d} | FPS: {current_fps:>4.1f} | Q-values: {qs_display} | Action: {action}')
            except Exception as e:
                print(f'Step: {step_count:>3d} | FPS: {current_fps:>4.1f} | Action: {action}')

        # 在episode结束时销毁actor
        print(f"清理 Episode {episode + 1} 的actor...")
        try:
            if hasattr(env, 'actor_list'):
                for actor in env.actor_list:
                    try:
                        actor.destroy()
                    except Exception as e:
                        print(f"销毁actor错误: {e}")
        except Exception as e:
            print(f"清理错误: {e}")

    print("\n" + "="*50)
    print("所有episodes完成!")
    print("="*50)
    
    # 清理资源
    try:
        cv2.destroyAllWindows()
    except:
        pass
        
    print("程序正常退出")