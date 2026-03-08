import glob
import os
import time
import numpy as np
import cv2
import csv
from datetime import datetime
from stable_baselines3 import PPO
from custom_env import AirSimMazeEnv
import math

# ==============================================================================
# 配置区域
# ==============================================================================

# 项目名称缩写：ASMN (AirSim Maze Navigation)
PROJECT_ABBR = "ASMN"

# 获取当前脚本所在目录作为项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径
MODELS_DIR = os.path.join(BASE_DIR, f"{PROJECT_ABBR}_models")
LOG_PATH = os.path.join(BASE_DIR, f"{PROJECT_ABBR}_inference_logs")  # 轨迹保存路径
os.makedirs(LOG_PATH, exist_ok=True)

# 可视化配置
SHOW_DASHBOARD = True  # 是否显示 OpenCV 仪表盘
DASHBOARD_SIZE = (800, 400)  # 宽, 高


# ==============================================================================
# 辅助函数
# ==============================================================================

def get_all_models(path_dir):
    """获取所有模型并按时间排序"""
    files = glob.glob(os.path.join(path_dir, '*.zip'))
    # 按修改时间倒序排列 (最新的在前)
    files.sort(key=os.path.getctime, reverse=True)
    return files


def draw_dashboard(obs, action, reward, step_count, last_info):
    """绘制实时仪表盘"""
    # 1. 创建黑色背景画布
    canvas = np.zeros((DASHBOARD_SIZE[1], DASHBOARD_SIZE[0], 3), dtype=np.uint8)

    # --- 左侧: 深度摄像头画面 ---
    if 'image' in obs:
        depth_img = obs['image']
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
        depth_img = cv2.resize(depth_img, (350, 350), interpolation=cv2.INTER_NEAREST)
        h, w, _ = depth_img.shape
        canvas[25:25 + h, 25:25 + w] = depth_img
        cv2.putText(canvas, "Depth Camera", (25, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- 右侧: 数据与 Lidar ---
    x_start = 400
    y_start = 50
    line_height = 30

    infos = [
        f"Step: {step_count}",
        f"Fwd Vel: {action[0]:.2f} (x5.0 m/s)",
        f"Yaw Rate: {action[1]:.2f} (x60 deg/s)",
        f"Reward: {reward:.3f}",
        f"Status: {last_info}"
    ]

    for i, text in enumerate(infos):
        color = (0, 255, 0)
        if "Reward" in text and reward < 0:
            color = (0, 0, 255)
        if "Status" in text and "撞墙" in text:
            color = (0, 0, 255)

        cv2.putText(canvas, text, (x_start, y_start + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- 右下角: Lidar 雷达图 ---
    lidar_center = (600, 300)
    lidar_radius = 80
    cv2.circle(canvas, lidar_center, 2, (0, 255, 255), -1)
    cv2.circle(canvas, lidar_center, lidar_radius, (50, 50, 50), 1)

    if 'lidar' in obs:
        lidar_data = obs['lidar']
        for i in range(0, 180, 2):
            dist = lidar_data[i]
            if dist < 20:
                angle_deg = -90 + i
                angle_rad = math.radians(angle_deg - 90)
                r_pixel = (dist / 20.0) * lidar_radius
                pt_x = int(lidar_center[0] + r_pixel * math.cos(angle_rad))
                pt_y = int(lidar_center[1] + r_pixel * math.sin(angle_rad))
                pt_color = (0, 255, 0) if dist > 5 else (0, 0, 255)
                cv2.circle(canvas, (pt_x, pt_y), 2, pt_color, -1)

    return canvas


# ==============================================================================
# 主逻辑
# ==============================================================================

def main():
    print("==================================================")
    print(f"       {PROJECT_ABBR} - AirSim UAV Maze Navigation       ")
    print("==================================================")

    # 1. 模型选择
    models = get_all_models(MODELS_DIR)
    if not models:
        print("错误: 未找到任何模型文件！请先运行 train.py。")
        return

    print(f"发现 {len(models)} 个模型存档。")
    print(f"默认加载最新的: {os.path.basename(models[0])}")
    model_path = models[0]

    # 2. 加载环境与模型
    print("正在初始化环境...")
    env = AirSimMazeEnv()

    print(f"正在加载神经网络: {model_path} ...")
    model = PPO.load(model_path)

    # 3. 初始化统计数据
    stats = {
        "episodes": 0,
        "success": 0,
        "collision": 0,
        "out_of_bounds": 0,
        "timeout": 0
    }

    # 4. 创建轨迹日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_PATH, f"{PROJECT_ABBR}_trajectory_{timestamp}.csv")

    # 使用 utf-8-sig 编码支持中文
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Step", "X", "Y", "Z", "Reward", "DoneType"])

    print(f"\n>>> 开始推理 (按 'q' 退出 OpenCV 窗口或 Ctrl+C 停止) <<<\n")

    try:
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        current_traj = []
        done_reason = "Running"

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            pos = env.client.getMultirotorState().kinematics_estimated.position
            current_traj.append([stats['episodes'], step_count, pos.x_val, pos.y_val, pos.z_val, reward])

            if done:
                stats['episodes'] += 1
                if reward >= 50:
                    stats['success'] += 1
                    done_reason = "✅ SUCCESS"
                elif reward <= -50:
                    stats['collision'] += 1
                    done_reason = "❌ COLLISION"
                elif reward == -20:
                    stats['out_of_bounds'] += 1
                    done_reason = "⚠️ OUT OF BOUNDS"
                else:
                    stats['timeout'] += 1
                    done_reason = "⏳ TIMEOUT/OTHER"

                print(
                    f"Episode {stats['episodes']} 结束 | 原因: {done_reason} | 总分: {episode_reward:.2f} | 步数: {step_count}")

                # 追加数据到 CSV 文件
                with open(csv_filename, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    for row in current_traj:
                        writer.writerow(row + [done_reason])

                obs, _ = env.reset()
                episode_reward = 0
                step_count = 0
                current_traj = []
                if SHOW_DASHBOARD:
                    time.sleep(0.5)
            else:
                done_reason = "Flying..."

            if SHOW_DASHBOARD:
                dashboard = draw_dashboard(obs, action, reward, step_count, done_reason)
                cv2.imshow(f"{PROJECT_ABBR} Dashboard", dashboard)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户手动停止测试。")
                    break

    except KeyboardInterrupt:
        print("\n检测到键盘中断，停止测试。")
    finally:
        print("\n" + "=" * 50)
        print(f"              {PROJECT_ABBR} 测试总结报告              ")
        print("=" * 50)
        total = stats['episodes']
        if total > 0:
            print(f"总回合数: {total}")
            print(f"成功次数: {stats['success']} ({stats['success'] / total * 100:.1f}%)")
            print(f"撞墙次数: {stats['collision']} ({stats['collision'] / total * 100:.1f}%)")
            print(f"越界次数: {stats['out_of_bounds']} ({stats['out_of_bounds'] / total * 100:.1f}%)")
            print(f"轨迹数据已保存至: {csv_filename}")
        else:
            print("未完成任何完整回合。")
        print("=" * 50)

        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()