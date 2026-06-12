import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
from collections import deque
import numpy as np
import cv2
import time
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from driving_dqn import CarEnv

epsilon = 0.05
MODEL_PATH = "models/Driving__6030.model"

# ==================== 路径配置 ====================
# 测试场景配置

LEFT_TURN_SCENARIO = {
    "name": "左转测试",
    "start": [53.12553405761719,137.06280517578125,1.3652913570404053, 0],  # [x, y, z, yaw]
    "end": [105.81783294677734,97.80741882324219] 
}

STRAIGHT_SCENARIO = {
    "name": "直行测试", 
    "start": [-47.66019058227539,137.0165252685547,0.8818629384040833,0],
    "end": [60.39826965332031,137.57113647460938]
}

CUSTOM_SCENARIO = {
    "name": "自定义测试",
    "start": [200.0, 250.0, 5, 225],
    "end": [150.0, 200.0]
}

# 选择要测试的场景
SELECTED_SCENARIO = LEFT_TURN_SCENARIO

def setup_tensorflow():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print(f"TensorFlow 版本: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ 找到 {len(gpus)} 个GPU，已启用内存增长")
        except RuntimeError as e:
            print(f"⚠️ GPU设置错误: {e}")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("使用CPU运行")
    else:
        print("ℹ️ 未找到GPU，使用CPU运行")

def safe_load_model(model_path):
    try:
        print(f"尝试加载模型: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None
            
        try:
            model = load_model(model_path)
            print(f"✅ 成功加载模型: {model_path}")
            return model
        except Exception as e:
            print(f"标准加载失败: {e}")
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='mse')
                print(f"✅ 使用 compile=False 成功加载模型: {model_path}")
                return model
            except Exception as e2:
                print(f"❌ 所有加载方式都失败: {e2}")
                return None
                
    except Exception as e:
        print(f"❌ 加载模型时发生错误: {e}")
        return None

def create_compatible_model(input_shape=(2,), output_units=5):
    print("创建新的驾驶模型...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(output_units, activation='linear')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def preprocess_state_for_prediction(state_data):
    try:
        if isinstance(state_data, list):
            state_array = np.array(state_data)
        else:
            state_array = state_data
        
        if len(state_array.shape) == 1:
            state_array = state_array.reshape(1, -1)
        
        if state_array.shape[1] != 2:
            if state_array.shape[1] > 2:
                state_array = state_array[:, :2]
            elif state_array.shape[1] < 2:
                padding = np.zeros((1, 2 - state_array.shape[1]))
                state_array = np.concatenate([state_array, padding], axis=1)
        
        return state_array
    except Exception as e:
        print(f"状态预处理错误: {e}")
        return np.array([[0.0, 0.0]])

def print_scenario_info(scenario):
    print("\n" + "="*60)
    print(f"测试场景: {scenario['name']}")
    print("="*60)
    print(f"起点坐标: ({scenario['start'][0]:.2f}, {scenario['start'][1]:.2f})")
    print(f"起点高度: {scenario['start'][2]:.1f}m")
    print(f"初始航向: {scenario['start'][3]}°")
    print(f"终点坐标: ({scenario['end'][0]:.2f}, {scenario['end'][1]:.2f})")
    
    start_x, start_y = scenario['start'][0], scenario['start'][1]
    end_x, end_y = scenario['end'][0], scenario['end'][1]
    distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    print(f"起点到终点距离: {distance:.2f} 单位")
    print("="*60)

if __name__ == '__main__':
    
    FPS = 60
    EPISODES = 3
    
    # 选择测试场景
    print("请选择测试场景:")
    print("1. 左转测试 (默认)")
    print("2. 直行测试")
    choice = input("请输入选择 (1-2): ") or "1"
    
    if choice == "2":
        SELECTED_SCENARIO = STRAIGHT_SCENARIO
    
    # 设置 TensorFlow
    setup_tensorflow()

    # 加载模型
    model = safe_load_model(MODEL_PATH)
    
    if model is None:
        print("创建新的驾驶模型...")
        model = create_compatible_model(input_shape=(2,), output_units=5)
    
    print(f"✅ 模型加载完成:")
    print(f"   - 输入形状: {model.input_shape}")
    print(f"   - 输出形状: {model.output_shape}")

    # 创建环境并传入起点终点
    print("\n初始化CARLA环境...")
    env = CarEnv(
        start_point=SELECTED_SCENARIO["start"],
        end_point=SELECTED_SCENARIO["end"]
    )
    
    # 打印场景信息
    print_scenario_info(SELECTED_SCENARIO)

    # 预热模型
    print("预热模型...")
    try:
        dummy_state = preprocess_state_for_prediction([0.0, 0.0])
        model.predict(dummy_state, verbose=0)
        print("✅ 模型预热完成")
    except Exception as e:
        print(f"⚠️ 模型预热警告: {e}")

    # 循环 episodes
    for episode in range(EPISODES):
        print(f'\n{"="*50}')
        print(f'开始 Episode {episode + 1}/{EPISODES}')
        print(f'测试场景: {SELECTED_SCENARIO["name"]}')
        print(f'{"="*50}')

        try:
            # 重置环境
            print("正在重置环境...")
            current_state = env.reset()
            print(f"初始状态: phi={current_state[0]:.1f}°, d={current_state[1]:.1f}")
            
            if len(current_state) != 2:
                print(f"⚠️ 警告: 状态维度为 {len(current_state)}，期望 2")
                current_state = [0.0, 0.0]
            
            done = False
            step_count = 0
            total_reward = 0

            # 循环步骤
            while not done:
                step_count += 1
                env.step_count = step_count  # 用于检查移动距离
                
                # 显示进度
                if step_count % 20 == 0:
                    print(f"\nEpisode {episode+1}, Step {step_count}")
                    print(f"累计奖励: {total_reward:.1f}")
                    if hasattr(env, 'current_waypoint_index'):
                        print(f"当前路径点: {env.current_waypoint_index}/{len(env.path)-1}")
                
                # 预测动作
                action = None
                qs = None
                
                try:
                    if np.random.random() > epsilon or step_count <= 1:
                        state_for_prediction = preprocess_state_for_prediction(current_state)
                        qs = model.predict(state_for_prediction, verbose=0)[0]
                        action = np.argmax(qs)
                        if step_count <= 5:
                            print(f"Step {step_count}: 预测动作 {action}")
                    else:
                        action = np.random.randint(0, 5)
                        if step_count <= 5:
                            print(f"Step {step_count}: 随机动作 {action}")

                except Exception as e:
                    print(f"❌ 预测错误: {e}")
                    action = 0
                    qs = np.zeros(5)

                # 环境步骤
                try:
                    new_state, reward, done, waypoint = env.step(action, current_state)
                    total_reward += reward
                    
                    # 显示重要信息
                    if step_count <= 10 or done or abs(reward) > 10:
                        print(f"Step {step_count}: 动作={action}, 奖励={reward:.1f}, "
                              f"phi={new_state[0]:.1f}°, d={new_state[1]:.1f}, 完成={done}")

                    # 更新状态
                    current_state = new_state

                except Exception as e:
                    print(f"❌ 环境步骤错误: {e}")
                    import traceback
                    traceback.print_exc()
                    done = True

                # 如果完成
                if done:
                    print(f"\nEpisode {episode + 1} 完成!")
                    print(f"总步数: {step_count}")
                    print(f"总奖励: {total_reward:.1f}")
                    if hasattr(env, 'reached') and env.reached == 1:
                        print("✅ 成功到达目的地!")
                    else:
                        print("❌ 未到达目的地")
                    break

        except KeyboardInterrupt:
            print("\n用户中断，跳过当前episode...")
        except Exception as e:
            print(f"❌ Episode {episode + 1} 发生错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 保存数据
            print(f"保存 Episode {episode + 1} 的数据...")
            try:
                os.makedirs(f"data/traj_test", exist_ok=True)
                
                scenario_name = SELECTED_SCENARIO["name"].replace(" ", "_")
                
                if hasattr(env, 'phi') and env.phi:
                    np.savetxt(f"data/traj_test/ep{episode+1}_{scenario_name}_phi.txt", env.phi)
                if hasattr(env, 'dc') and env.dc:
                    np.savetxt(f"data/traj_test/ep{episode+1}_{scenario_name}_d.txt", env.dc)
                if hasattr(env, 'vel') and env.vel:
                    np.savetxt(f"data/traj_test/ep{episode+1}_{scenario_name}_vel.txt", env.vel)
                if hasattr(env, 'time') and env.time:
                    np.savetxt(f"data/traj_test/ep{episode+1}_{scenario_name}_time.txt", env.time)
                    
                # 保存场景信息
                with open(f"data/traj_test/ep{episode+1}_{scenario_name}_info.txt", "w") as f:
                    f.write(f"场景名称: {SELECTED_SCENARIO['name']}\n")
                    f.write(f"起点: {SELECTED_SCENARIO['start']}\n")
                    f.write(f"终点: {SELECTED_SCENARIO['end']}\n")
                    f.write(f"总步数: {step_count}\n")
                    f.write(f"总奖励: {total_reward}\n")
                    f.write(f"是否到达: {getattr(env, 'reached', 0)}\n")
                    
                print(f"✅ Episode {episode + 1} 数据保存完成")
            except Exception as e:
                print(f"❌ 保存数据错误: {e}")

            # 清理actor
            print(f"清理 Episode {episode + 1} 的actor...")
            try:
                if hasattr(env, 'actor_list'):
                    for actor in env.actor_list:
                        try:
                            actor.destroy()
                        except Exception as e:
                            pass
                    env.actor_list = []
            except Exception as e:
                print(f"清理错误: {e}")

    print("\n" + "="*60)
    print("所有测试完成!")
    print(f"测试了 {EPISODES} 个episodes")
    print(f"测试场景: {SELECTED_SCENARIO['name']}")
    print("="*60)
    
    try:
        cv2.destroyAllWindows()
    except:
        pass
        
    print("程序正常退出")