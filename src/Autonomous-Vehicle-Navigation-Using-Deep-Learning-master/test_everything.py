import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from car_env import CarEnv, MEMORY_FRACTION
import carla
from carla import Transform, Location, Rotation

# 轨迹定义
trajectories = {
    "custom_trajectory": {
        "start": [-8.77956485748291,140.2951202392578,2.0014660358428955, 0], 
        "end": [74.17852020263672,-56.52183151245117,0.18172569572925568],
        "description": "自定义轨迹"
    }
}

SELECTED_TRAJECTORY = "custom_trajectory"

def get_selected_trajectory():
    """获取选定的轨迹"""
    if SELECTED_TRAJECTORY in trajectories:
        trajectory = trajectories[SELECTED_TRAJECTORY]
        print(f"✅ 使用轨迹: {SELECTED_TRAJECTORY}")
        print(f"  描述: {trajectory['description']}")
        print(f"  起点: {trajectory['start']}")
        print(f"  终点: {trajectory['end']}")
        return trajectory
    else:
        print(f"❌ 轨迹 '{SELECTED_TRAJECTORY}' 不存在")
        return None

def safe_load_model(model_path):
    """安全加载模型"""
    try:
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None
            
        model = load_model(model_path)
        print(f"✅ 成功加载模型: {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None

def setup_tensorflow():
    """设置 TensorFlow 配置"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print(f"TensorFlow 版本: {tf.__version__}")
    
    # GPU 配置
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

def set_spectator_to_vehicle(world, vehicle):
    """设置观察者视角"""
    try:
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        
        # 更安全的视角
        spectator.set_transform(Transform(
            transform.location + Location(z=15, x=-15),
            Rotation(pitch=-30)
        ))
        print("✅ 观察者视角已设置")
        
    except Exception as e:
        print(f"⚠️ 设置视角时出错: {e}")

def preprocess_state_for_prediction(state_data, model_type="braking"):
    """预处理状态数据用于模型预测"""
    try:
        if model_type == "braking":
            state_array = np.array(state_data[:2])
        else:
            state_array = np.array(state_data[2:])
        
        if len(state_array.shape) == 1:
            state_array = state_array.reshape(1, -1)
        
        return state_array
    except Exception as e:
        print(f"状态预处理错误: {e}")
        return np.array([[0, 0]])

def debug_vehicle_state(vehicle):
    """调试车辆状态"""
    if vehicle is None:
        print("❌ 车辆为 None")
        return
    
    try:
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        print(f"📍 车辆位置: ({transform.location.x:.2f}, {transform.location.y:.2f}, {transform.location.z:.2f})")
        print(f"🧭 车辆朝向: {transform.rotation.yaw:.2f}°")
        print(f"🚀 车辆速度: {np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2):.2f} m/s")
    except Exception as e:
        print(f"❌ 获取车辆状态失败: {e}")

def main():
    # 设置 TensorFlow
    setup_tensorflow()
    
    # 获取选定的轨迹
    trajectory = get_selected_trajectory()
    if trajectory is None:
        print("❌ 无法获取轨迹，退出程序")
        return
        
    start_location = trajectory["start"]
    end_location = trajectory["end"]
    
    # 加载模型
    print("\n" + "="*50)
    print("加载自动驾驶模型")
    print("="*50)
    
    MODEL_PATH = "models/Braking___282.model"
    MODEL_PATH2 = "models/Driving__6030.model"
    
    model = safe_load_model(MODEL_PATH)
    model2 = safe_load_model(MODEL_PATH2)
    
    if model is None or model2 is None:
        print("❌ 模型加载失败，退出程序")
        return
    
    # 创建环境
    print("\n初始化CARLA环境...")
    try:
        env = CarEnv(start_location, end_location)
        world = env.client.get_world()
        
        # 设置仿真设置
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
    except Exception as e:
        print(f"❌ 初始化环境失败: {e}")
        return
    
    # 主循环
    fps_counter = deque(maxlen=60)
    EPISODES = 2
    
    for episode in range(EPISODES):
        print(f'\n{"="*50}')
        print(f'开始 Episode {episode + 1}/{EPISODES}')
        print(f'{"="*50}')
        
        # 重置环境 - 这会生成车辆
        try:
            print("重置环境...")
            current_state = env.reset()
            print(f"初始状态: {current_state}")
        except Exception as e:
            print(f"❌ 环境重置失败: {e}")
            continue
        
        # 从环境中获取车辆
        ego_vehicle = env.vehicle
        
        if ego_vehicle is None:
            print("❌ 环境中没有车辆，跳过此episode")
            continue
        
        # 调试车辆状态
        debug_vehicle_state(ego_vehicle)
        
        # 设置观察者视角
        set_spectator_to_vehicle(world, ego_vehicle)
        
        done = False
        step_count = 0
        max_steps = 1000
        
        while not done and step_count < max_steps:
            step_count += 1
            step_start = time.time()
            
            # 定期更新视角
            if step_count % 20 == 0:
                set_spectator_to_vehicle(world, ego_vehicle)
            
            # 动作预测
            action = 0
            try:
                # 检查交通灯
                if hasattr(env, 'vehicle') and env.vehicle and env.vehicle.is_at_traffic_light():
                    traffic_light = env.vehicle.get_traffic_light()
                    if traffic_light and traffic_light.get_state() == carla.TrafficLightState.Red:
                        print("🚦 红灯 - 停车")
                        action = 0
                    else:
                        # 使用模型预测
                        state_array = preprocess_state_for_prediction(current_state, "braking")
                        qs = model.predict(state_array, verbose=0)[0]
                        action = np.argmax(qs)
                        
                        if action == 1:  # 安全时才使用驾驶模型
                            state_array2 = preprocess_state_for_prediction(current_state, "driving")
                            qs2 = model2.predict(state_array2, verbose=0)[0]
                            action = np.argmax(qs2) + 1
                else:
                    # 正常情况下的决策
                    state_array = preprocess_state_for_prediction(current_state, "braking")
                    qs = model.predict(state_array, verbose=0)[0]
                    action = np.argmax(qs)
                    
                    if action == 1:
                        state_array2 = preprocess_state_for_prediction(current_state, "driving")
                        qs2 = model2.predict(state_array2, verbose=0)[0]
                        action = np.argmax(qs2) + 1
                        
            except Exception as e:
                print(f"❌ 预测错误: {e}")
                action = 0
            
            # 执行动作
            try:
                new_state, reward, done, waypoint = env.step(action, current_state)
                current_state = new_state
                
                # 显示额外信息
                if step_count % 10 == 0:
                    print(f"步骤 {step_count}, 奖励: {reward}, 完成: {done}")
                
            except Exception as e:
                print(f"❌ 环境步骤错误: {e}")
                done = True
            
            # 计算FPS
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            current_fps = len(fps_counter) / sum(fps_counter) if fps_counter else 0
            
            # 显示动作名称
            action_names = ["刹车", "直行", "左转", "右转", "微左", "微右"]
            action_name = action_names[action] if action < len(action_names) else str(action)
            
            print(f'Step: {step_count:>3d} | FPS: {current_fps:>4.1f} | Action: {action_name}')
            
            if done:
                print(f"Episode {episode + 1} 完成，步数: {step_count}")
                break
        
        if step_count >= max_steps:
            print(f"Episode {episode + 1} 达到最大步数限制")
        
        # 等待一段时间再开始下一个episode
        print(f"等待下一个episode...")
        time.sleep(2.0)
    
    # 最终清理
    print("\n" + "="*50)
    print("所有episodes完成!")
    print("="*50)
    
    print("清理资源...")
    try:
        # 环境会在重置时自动清理车辆
        cv2.destroyAllWindows()
    except:
        pass
    
    print("程序结束")

if __name__ == '__main__':
    main()