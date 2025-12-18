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
from TrainingStrategies import *
import os
import json
import glob


def get_script_directory():
    """获取Test.py脚本所在的目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return script_dir


def find_model_files(model_dir="models", pattern="*.model"):
    """
    自动查找模型文件（只在Test.py所在目录及其子目录中查找）
    """
    script_dir = get_script_directory()
    
    # 只在Test.py所在目录及其子目录中查找
    possible_paths = [
        os.path.join(script_dir, model_dir),  # 脚本目录下的models文件夹
        os.path.join(script_dir, "models"),  # 脚本目录下的models
        os.path.join(script_dir, "saved_models"),  # 脚本目录下的saved_models
        os.path.join(script_dir, "model"),  # 脚本目录下的model
        script_dir,  # 脚本目录本身（可能模型文件直接放在这里）
    ]
    
    model_files = []
    
    for path in possible_paths:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, pattern))
            if files:
                # 显示相对路径（相对于脚本目录）
                rel_path = os.path.relpath(path, script_dir)
                if rel_path == ".":
                    rel_path = "当前目录"
                print(f"在目录 '{rel_path}' 中找到 {len(files)} 个模型文件")
                model_files.extend(files)
    
    # 去重
    model_files = list(set(model_files))
    
    # 按修改时间排序（最新的在前面）
    model_files.sort(key=os.path.getmtime, reverse=True)
    
    return model_files


def select_best_model(model_files, preferred_keywords=None, excluded_keywords=None):
    """
    从模型文件列表中选择最佳模型
    """
    if not model_files:
        return None
    
    if preferred_keywords is None:
        preferred_keywords = ["best", "advanced", "dueling_per", "v2"]
    
    if excluded_keywords is None:
        excluded_keywords = ["min", "avg", "final"]  # 排除统计文件
    
    # 评分系统：根据关键词和文件属性给模型打分
    scored_models = []
    
    for file_path in model_files:
        filename = os.path.basename(file_path)
        score = 0
        
        # 基于文件名关键词打分
        for keyword in preferred_keywords:
            if keyword.lower() in filename.lower():
                score += 10
        
        # 排除包含特定关键词的文件
        exclude = False
        for keyword in excluded_keywords:
            if keyword.lower() in filename.lower() and not filename.lower().endswith(".model"):
                exclude = True
                break
        
        if exclude:
            continue
        
        # 基于文件大小和修改时间打分
        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            if file_size > 100:  # 大于100MB的模型可能更复杂
                score += 5
            
            # 文件修改时间（越新越好）
            days_old = (time.time() - os.path.getmtime(file_path)) / (24 * 3600)
            if days_old < 7:  # 一周内的文件
                score += 10
            elif days_old < 30:  # 一个月内的文件
                score += 5
        except:
            pass
        
        scored_models.append((file_path, score, filename))
    
    if not scored_models:
        return None
    
    # 按分数排序
    scored_models.sort(key=lambda x: x[1], reverse=True)
    
    print("\n找到的模型文件（按优先级排序）:")
    for i, (path, score, name) in enumerate(scored_models[:5]):  # 显示前5个
        # 显示相对路径
        script_dir = get_script_directory()
        rel_path = os.path.relpath(path, script_dir)
        print(f"  {i+1}. [{score:3d}分] {name}")
        print(f"      路径: {rel_path}")
    
    return scored_models[0][0]  # 返回最佳模型的路径


def get_safe_action_advanced(model, state, env, previous_action, uncertainty_threshold=1.0):
    """
    高级安全动作选择，结合模型预测、安全规则、不确定性估计和多目标优化
    """
    # 模型预测
    state_normalized = np.array(state).reshape(-1, *state.shape) / 255
    qs = model.predict(state_normalized, verbose=0)[0]
    
    # 获取车辆速度
    velocity = env.vehicle.get_velocity()
    speed_kmh = 3.6 * np.linalg.norm([velocity.x, velocity.y, velocity.z])
    
    # 1. 速度自适应调整
    speed_factor = max(0.3, min(1.0, 30.0 / max(1.0, speed_kmh)))
    
    if speed_kmh > 40:  # 高速时更加保守
        qs[2] *= 0.6  # 降低加速倾向
        qs[3] *= 0.5  # 大幅降低左转倾向
        qs[4] *= 0.5  # 大幅降低右转倾向
    elif speed_kmh < 10:  # 低速时鼓励前进
        qs[0] *= 0.5  # 降低减速倾向
        qs[1] *= 1.2  # 提高保持倾向
        qs[2] *= 1.3  # 提高加速倾向
    
    # 2. 行人避障优先级 - 增强版
    if hasattr(env, 'suggested_action') and env.suggested_action is not None:
        # 根据距离调整优先级
        min_ped_distance = getattr(env, 'last_ped_distance', float('inf'))
        if min_ped_distance < 8.0:  # 危险距离
            qs[env.suggested_action] += 5.0  # 大幅提高建议动作的Q值
            print(f"🚨 紧急避让: 执行动作 {env.suggested_action}, 距离: {min_ped_distance:.1f}m")
        elif min_ped_distance < 12.0:  # 预警距离
            qs[env.suggested_action] += 2.0
            print(f"⚠️ 安全避让: 执行动作 {env.suggested_action}, 距离: {min_ped_distance:.1f}m")
        env.suggested_action = None
    
    # 3. 防止过度转向
    if hasattr(env, 'same_steer_counter') and env.same_steer_counter > 2:
        if previous_action in [3, 4]:
            qs[previous_action] -= 2.0  # 降低连续同向转向的倾向
    
    # 4. 动作平滑性
    if previous_action in [3, 4]:  # 转向动作
        qs[previous_action] += 0.8 * speed_factor  # 速度相关的平滑性
    elif previous_action in [0, 2]:  # 加减速动作
        qs[previous_action] += 0.3  # 轻微的惯性保持
    
    # 5. 道路保持倾向
    # 如果车辆方向偏差小，鼓励保持直行
    if hasattr(env, 'vehicle'):
        vehicle_rotation = env.vehicle.get_transform().rotation.yaw
        if abs(vehicle_rotation) < 10:  # 方向良好
            qs[1] += 0.5  # 鼓励保持
        elif abs(vehicle_rotation) > 30:  # 方向偏差大
            # 鼓励向相反方向转向以回正
            if vehicle_rotation > 0:  # 偏左，鼓励右转
                qs[4] += 1.5
            else:  # 偏右，鼓励左转
                qs[3] += 1.5
    
    # 6. 紧急情况处理 - 增强版
    min_ped_distance = getattr(env, 'last_ped_distance', float('inf'))
    if min_ped_distance < 5.0:  # 紧急避让距离
        # 大幅调整Q值以确保安全
        qs[0] += 3.0  # 紧急制动
        if min_ped_distance < 3.0:  # 极危险
            qs[2] = -float('inf')  # 禁止加速
            qs[1] = -float('inf')  # 禁止保持
            print("⚠️ 紧急制动! 危险距离!")
    
    # 7. 考虑运动检测结果
    if hasattr(env, 'motion_detected') and env.motion_detected:
        # 如果检测到运动，提高警惕
        qs[0] += 0.5  # 鼓励减速观察
    
    # 选择动作
    action = np.argmax(qs)
    
    # 最终安全检查
    if speed_kmh > 35 and action in [3, 4]:
        # 高速急转检查
        steer_magnitude = abs(qs[3]) if action == 3 else abs(qs[4])
        if steer_magnitude > 2.0:  # 急转倾向强
            # 考虑更安全的替代动作
            safe_alternatives = [1, 0]  # 保持或减速
            safe_qs = [qs[a] for a in safe_alternatives]
            if max(safe_qs) > qs[action] * 0.7:
                action = safe_alternatives[np.argmax(safe_qs)]
                print(f"安全调整: 高速时避免急转，选择动作 {action}")
    
    return action, qs


def run_test_episode(model, env, episode_num, use_advanced_safety=True):
    """运行单个测试episode"""
    print(f"\n{'='*50}")
    print(f"测试 Episode {episode_num}")
    print(f"{'='*50}")
    
    # 重置环境
    current_state = env.reset(401)  # 正常难度
    env.collision_hist = []
    
    # 初始化统计
    total_reward = 0
    step_count = 0
    done = False
    previous_action = 1
    fps_counter = deque(maxlen=30)
    
    # 反应时间统计
    reaction_times = []
    
    # 运行episode
    max_steps = SECONDS_PER_EPISODE * 60
    
    while not done and step_count < max_steps:
        step_start = time.time()
        
        # 选择动作
        if use_advanced_safety:
            action, qs = get_safe_action_advanced(model, current_state, env, previous_action)
        else:
            # 基础动作选择
            state_normalized = np.array(current_state).reshape(-1, *current_state.shape) / 255
            qs = model.predict(state_normalized, verbose=0)[0]
            action = np.argmax(qs)
        
        previous_action = action
        
        # 记录反应开始时间（如果有障碍物）
        if hasattr(env, 'obstacle_detected_time') and env.obstacle_detected_time is not None:
            if hasattr(env, 'reaction_start_time') and env.reaction_start_time is not None:
                reaction_time = time.time() - env.reaction_start_time
                reaction_times.append(reaction_time)
        
        # 执行动作
        new_state, reward, done, _ = env.step(action)
        
        # 更新状态
        current_state = new_state
        total_reward += reward
        step_count += 1
        
        # 计算FPS
        frame_time = time.time() - step_start
        fps_counter.append(frame_time)
        
        # 每30步显示一次状态
        if step_count % 30 == 0:
            fps = len(fps_counter)/sum(fps_counter) if fps_counter else 0
            velocity = env.vehicle.get_velocity()
            speed_kmh = 3.6 * np.linalg.norm([velocity.x, velocity.y, velocity.z])
            
            # 获取最近行人距离
            min_ped_distance = getattr(env, 'last_ped_distance', float('inf'))
            
            status = "✅" if reward > 0 else "⚠️" if reward < -1 else "➡️"
            
            print(f"{status} 步数: {step_count:4d} | FPS: {fps:4.1f} | "
                  f"速度: {speed_kmh:5.1f} km/h | 行人距离: {min_ped_distance:5.1f}m | "
                  f"奖励: {reward:6.2f} | 累计: {total_reward:7.2f}")
        
        if done:
            break
    
    # 清理环境
    env.cleanup_actors()
    
    # 判断结果
    success = total_reward > 5
    result = "成功" if success else "失败"
    
    # 计算平均反应时间
    avg_reaction_time = np.mean(reaction_times) if reaction_times else 0
    
    print(f"\nEpisode {episode_num} 结果: {result}")
    print(f"总步数: {step_count}, 总奖励: {total_reward:.2f}")
    print(f"平均反应时间: {avg_reaction_time:.2f}秒")
    
    return success, total_reward, step_count, avg_reaction_time


def load_model_with_fallback(model_path):
    """加载模型，支持多种格式和回退机制"""
    print(f"尝试加载模型: {model_path}")
    
    # 如果是相对路径，尝试转换为绝对路径（相对于脚本目录）
    if not os.path.isabs(model_path):
        script_dir = get_script_directory()
        model_path = os.path.join(script_dir, model_path)
    
    if not os.path.exists(model_path):
        # 尝试在当前目录下查找
        model_name = os.path.basename(model_path)
        script_dir = get_script_directory()
        possible_paths = [
            os.path.join(script_dir, model_name),
            os.path.join(script_dir, "models", model_name),
            os.path.join(script_dir, "saved_models", model_name),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"找到模型文件: {model_path}")
                break
        else:
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    # 定义自定义层
    custom_objects = {
        'Add': tf.keras.layers.Add, 
        'Subtract': tf.keras.layers.Subtract,
        'Lambda': tf.keras.layers.Lambda,
        'Multiply': tf.keras.layers.Multiply
    }
    
    try:
        # 尝试加载完整模型
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"✅ 模型加载成功 (使用自定义层)")
        return model
    except Exception as e1:
        print(f"使用自定义层加载失败: {e1}")
        try:
            # 尝试不加载自定义层
            model = load_model(model_path)
            print(f"✅ 模型加载成功 (基础加载)")
            return model
        except Exception as e2:
            print(f"基础加载失败: {e2}")
            
            # 尝试使用 tf.keras.models.load_model 的不同参数
            try:
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,
                    custom_objects=custom_objects
                )
                print(f"✅ 模型加载成功 (不编译)")
                return model
            except Exception as e3:
                print(f"所有加载尝试失败: {e3}")
                raise ValueError(f"无法加载模型: {model_path}")


def comprehensive_model_evaluation(model_path, num_episodes=5):
    """综合模型评估"""
    print(f"\n{'='*60}")
    print(f"开始综合模型评估")
    print(f"模型路径: {model_path}")
    print(f"测试轮次: {num_episodes}")
    print(f"{'='*60}")
    
    # GPU配置
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))
    
    # 加载模型
    model = load_model_with_fallback(model_path)
    
    # 创建环境
    env = CarEnv()
    env.SHOW_CAM = False
    
    # 预热模型
    print("预热模型...")
    model.predict(np.ones((1, env.im_height, env.im_width, 3)), verbose=0)
    
    # 运行测试
    results = {
        'successes': 0,
        'total_rewards': [],
        'episode_lengths': [],
        'reaction_times': [],
        'start_time': time.time()
    }
    
    try:
        for episode in range(1, num_episodes + 1):
            success, reward, length, avg_rt = run_test_episode(model, env, episode, use_advanced_safety=True)
            
            if success:
                results['successes'] += 1
            results['total_rewards'].append(reward)
            results['episode_lengths'].append(length)
            results['reaction_times'].append(avg_rt)
            
            # 短暂暂停
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        # 清理环境
        env.cleanup_actors()
        
        # 计算统计
        results['end_time'] = time.time()
        results['total_time'] = results['end_time'] - results['start_time']
        
        if results['total_rewards']:
            results['success_rate'] = results['successes'] / len(results['total_rewards']) * 100
            results['avg_reward'] = np.mean(results['total_rewards'])
            results['avg_length'] = np.mean(results['episode_lengths'])
            results['max_reward'] = max(results['total_rewards'])
            results['min_reward'] = min(results['total_rewards'])
            results['avg_reaction_time'] = np.mean(results['reaction_times']) if results['reaction_times'] else 0
        
        # 显示评估报告
        print(f"\n{'='*60}")
        print("综合评估报告")
        print(f"{'='*60}")
        print(f"测试轮次: {num_episodes}")
        print(f"成功次数: {results['successes']}")
        print(f"成功率: {results.get('success_rate', 0):.1f}%")
        print(f"平均奖励: {results.get('avg_reward', 0):.2f}")
        print(f"平均步数: {results.get('avg_length', 0):.1f}")
        print(f"平均反应时间: {results.get('avg_reaction_time', 0):.2f}秒")
        print(f"最佳表现: {results.get('max_reward', 0):.2f}")
        print(f"最差表现: {results.get('min_reward', 0):.2f}")
        print(f"总测试时间: {results.get('total_time', 0):.1f}秒")
        print(f"模型路径: {model_path}")
        
        # 保存评估结果到脚本所在目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path).replace('.model', '')
        script_dir = get_script_directory()
        eval_file = os.path.join(script_dir, f"model_evaluation_{model_name}_{timestamp}.json")
        
        # 转换numpy类型为Python原生类型
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.generic):
                serializable_results[key] = value.item()
            else:
                serializable_results[key] = value
        
        serializable_results['model_path'] = model_path
        serializable_results['model_name'] = model_name
        serializable_results['evaluation_date'] = timestamp
        serializable_results['num_episodes'] = num_episodes
        
        with open(eval_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n评估结果已保存到: {eval_file}")
        
        return results


def interactive_model_selection(model_files):
    """交互式模型选择"""
    if not model_files:
        print("❌ 未找到任何模型文件")
        return None
    
    script_dir = get_script_directory()
    
    print(f"\n找到 {len(model_files)} 个模型文件:")
    for i, file_path in enumerate(model_files):
        # 显示相对路径
        rel_path = os.path.relpath(file_path, script_dir)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        mod_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(file_path)))
        print(f"  {i+1}. {filename} ({file_size:.1f} MB, 修改于: {mod_time})")
        print(f"      路径: {rel_path}")
    
    while True:
        try:
            choice = input(f"\n请选择模型 (1-{len(model_files)}) 或按回车选择最新模型: ").strip()
            
            if choice == "":
                # 选择最新的模型
                selected = model_files[0]
                rel_path = os.path.relpath(selected, script_dir)
                print(f"选择最新的模型: {os.path.basename(selected)}")
                print(f"路径: {rel_path}")
                return selected
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_files):
                selected = model_files[choice_idx]
                rel_path = os.path.relpath(selected, script_dir)
                print(f"选择模型: {os.path.basename(selected)}")
                print(f"路径: {rel_path}")
                return selected
            else:
                print(f"请输入 1 到 {len(model_files)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n选择被用户中断")
            return None


def main():
    """主函数 - 自动查找和测试模型"""
    print(f"\n{'='*60}")
    print("自动驾驶模型测试系统 - 优化版")
    print(f"{'='*60}")
    
    # 显示当前脚本所在目录
    script_dir = get_script_directory()
    print(f"脚本所在目录: {script_dir}")
    
    # 自动查找模型文件（只在脚本目录及其子目录中查找）
    print("\n正在搜索模型文件（仅在当前项目目录中）...")
    model_files = find_model_files()
    
    if not model_files:
        print("❌ 未找到任何模型文件 (.model)")
        print("请确保:")
        print("  1. 已经训练过模型")
        print("  2. 模型文件保存在当前目录或 'models' 子目录中")
        print("  3. 模型文件扩展名为 .model")
        
        # 尝试搜索其他可能的扩展名
        for ext in [".h5", ".keras", ".tf"]:
            alt_files = find_model_files(pattern=f"*{ext}")
            if alt_files:
                print(f"\n找到 {len(alt_files)} 个 {ext} 格式的模型文件")
                model_files = alt_files
                break
        
        if not model_files:
            return
    
    # 交互式选择模型
    selected_model = interactive_model_selection(model_files)
    
    if not selected_model:
        print("未选择模型，退出测试")
        return
    
    # 开始测试
    comprehensive_model_evaluation(selected_model, num_episodes=3)


def quick_test():
    """快速测试 - 自动选择最佳模型并运行少量测试"""
    print("\n正在执行快速测试...")
    
    # 显示当前脚本所在目录
    script_dir = get_script_directory()
    print(f"脚本所在目录: {script_dir}")
    
    # 查找模型（只在脚本目录及其子目录中查找）
    model_files = find_model_files()
    
    if not model_files:
        print("❌ 未找到模型文件")
        return
    
    # 自动选择最佳模型
    selected_model = select_best_model(model_files)
    
    if not selected_model:
        print("❌ 无法选择模型")
        return
    
    rel_path = os.path.relpath(selected_model, script_dir)
    print(f"自动选择模型: {os.path.basename(selected_model)}")
    print(f"路径: {rel_path}")
    
    # 运行1个episode进行快速测试
    comprehensive_model_evaluation(selected_model, num_episodes=1)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='自动驾驶模型测试')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    parser.add_argument('--model', type=str, help='指定模型文件路径')
    parser.add_argument('--episodes', type=int, default=3, help='测试轮次数量')
    
    args = parser.parse_args()
    
    if args.model:
        # 使用指定的模型文件
        script_dir = get_script_directory()
        
        # 如果指定的是相对路径，转换为绝对路径
        if not os.path.isabs(args.model):
            args.model = os.path.join(script_dir, args.model)
        
        if os.path.exists(args.model):
            print(f"使用指定模型: {args.model}")
            comprehensive_model_evaluation(args.model, num_episodes=args.episodes)
        else:
            print(f"❌ 指定的模型文件不存在: {args.model}")
            # 尝试在脚本目录下查找
            model_name = os.path.basename(args.model)
            possible_paths = [
                os.path.join(script_dir, model_name),
                os.path.join(script_dir, "models", model_name),
                os.path.join(script_dir, "saved_models", model_name),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"找到模型文件: {path}")
                    comprehensive_model_evaluation(path, num_episodes=args.episodes)
                    break
            else:
                print("无法找到指定的模型文件")
    elif args.quick:
        # 快速测试模式
        quick_test()
    else:
        # 交互式测试模式
        main()