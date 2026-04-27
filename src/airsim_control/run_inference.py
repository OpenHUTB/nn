import time
from pathlib import Path
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
MODELS_DIR = Path(r"D:\Others\MyAirsimprojects\models")
LOG_PATH = Path(r"D:\Others\MyAirsimprojects\inference_logs")  # 轨迹保存路径
LOG_PATH.mkdir(parents=True, exist_ok=True)

# 可视化配置
SHOW_DASHBOARD = True  # 是否显示 OpenCV 仪表盘
DASHBOARD_SIZE = (800, 400)  # 宽, 高
DEPTH_PANEL_SIZE = (350, 350)
LIDAR_MAX_DISTANCE = 20.0
LIDAR_DANGER_DISTANCE = 5.0
TRAJECTORY_CSV_HEADERS = ["Episode", "Step", "X", "Y", "Z", "Reward", "DoneType"]
CSV_ENCODING = "utf-8-sig"


# ==============================================================================
# 辅助函数
# ==============================================================================

def get_all_models(path_dir):
    """获取所有模型并按时间排序"""
    path_dir = Path(path_dir)
    return sorted(path_dir.glob("*.zip"), key=lambda file_path: file_path.stat().st_ctime, reverse=True)


def get_episode_outcome(reward):
    """根据回合奖励推断结束类型。"""
    if reward >= 50:
        return "success", "✅ SUCCESS"
    if reward <= -50:
        return "collision", "❌ COLLISION"
    if reward == -20:
        return "out_of_bounds", "⚠️ OUT OF BOUNDS"
    return "timeout", "⏳ TIMEOUT/OTHER"


def create_trajectory_log(log_dir):
    """创建轨迹日志并写入表头。"""
    csv_path = Path(log_dir) / f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with csv_path.open("w", newline="", encoding=CSV_ENCODING) as file_obj:
        csv.writer(file_obj).writerow(TRAJECTORY_CSV_HEADERS)
    return csv_path


def append_trajectory_rows(csv_path, rows, done_reason):
    """将单回合轨迹批量写入 CSV。"""
    with Path(csv_path).open("a", newline="", encoding=CSV_ENCODING) as file_obj:
        writer = csv.writer(file_obj)
        for row in rows:
            writer.writerow(row + [done_reason])


def _get_obs_value(obs, key, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else default


def _prepare_depth_image(image):
    if image is None:
        return None
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) != 3 or image.shape[2] != 3:
        return None
    return cv2.resize(image, DEPTH_PANEL_SIZE, interpolation=cv2.INTER_NEAREST)


def _dashboard_status_text(text):
    ascii_text = text.encode("ascii", errors="ignore").decode().strip()
    return ascii_text or text


def draw_dashboard(obs, action, reward, step_count, last_info):
    """绘制实时仪表盘"""
    # 1. 创建黑色背景画布
    canvas = np.zeros((DASHBOARD_SIZE[1], DASHBOARD_SIZE[0], 3), dtype=np.uint8)
    action = np.asarray(action).flatten() if action is not None else np.zeros(2, dtype=float)
    forward_velocity = action[0] if action.size > 0 else 0.0
    yaw_rate = action[1] if action.size > 1 else 0.0

    # --- 左侧: 深度摄像头画面 ---
    depth_img = _prepare_depth_image(_get_obs_value(obs, "image"))
    if depth_img is not None:
        h, w, _ = depth_img.shape
        canvas[25:25 + h, 25:25 + w] = depth_img
        cv2.putText(canvas, "Depth Camera", (25, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- 右侧: 数据与 Lidar ---
    x_start = 400
    y_start = 50
    line_height = 30

    infos = [
        f"Step: {step_count}",
        f"Fwd Vel: {forward_velocity:.2f} (x5.0 m/s)",
        f"Yaw Rate: {yaw_rate:.2f} (x60 deg/s)",
        f"Reward: {reward:.3f}",
        f"Status: {_dashboard_status_text(last_info)}"
    ]

    for i, text in enumerate(infos):
        color = (0, 255, 0)
        if "Reward" in text and reward < 0:
            color = (0, 0, 255)
        if "Status" in text and ("COLLISION" in text or "OUT OF BOUNDS" in text):
            color = (0, 0, 255)

        cv2.putText(canvas, text, (x_start, y_start + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- 右下角: Lidar 雷达图 ---
    lidar_center = (600, 300)
    lidar_radius = 80
    cv2.circle(canvas, lidar_center, 2, (0, 255, 255), -1)
    cv2.circle(canvas, lidar_center, lidar_radius, (50, 50, 50), 1)

    lidar_data = _get_obs_value(obs, "lidar")
    if lidar_data is not None:
        lidar_data = np.asarray(lidar_data).flatten()
        for i in range(0, min(180, lidar_data.size), 2):
            dist = lidar_data[i]
            if dist < LIDAR_MAX_DISTANCE:
                angle_deg = -90 + i
                angle_rad = math.radians(angle_deg - 90)
                r_pixel = (dist / LIDAR_MAX_DISTANCE) * lidar_radius
                pt_x = int(lidar_center[0] + r_pixel * math.cos(angle_rad))
                pt_y = int(lidar_center[1] + r_pixel * math.sin(angle_rad))
                pt_color = (0, 255, 0) if dist > LIDAR_DANGER_DISTANCE else (0, 0, 255)
                cv2.circle(canvas, (pt_x, pt_y), 2, pt_color, -1)

    return canvas


# ==============================================================================
# 主逻辑
# ==============================================================================

def main():
    print("==================================================")
    print("       AirSim 无人机智能导航 - 推理与评估系统       ")
    print("==================================================")

    # 1. 模型选择
    models = get_all_models(MODELS_DIR)
    if not models:
        print("错误: 未找到任何模型文件！请先运行 train.py。")
        return

    print(f"发现 {len(models)} 个模型存档。")
    print(f"默认加载最新的: {models[0].name}")
    model_path = models[0]

    env = None
    csv_filename = None

    # 2. 初始化统计数据
    stats = {
        "episodes": 0,
        "success": 0,
        "collision": 0,
        "out_of_bounds": 0,
        "timeout": 0
    }

    try:
        # 3. 加载环境与模型
        print("正在初始化环境...")
        env = AirSimMazeEnv()

        print(f"正在加载神经网络: {model_path} ...")
        model = PPO.load(str(model_path))

        # 4. 创建轨迹日志文件
        csv_filename = create_trajectory_log(LOG_PATH)

        print("\n>>> 开始推理 (按 'q' 退出 OpenCV 窗口或 Ctrl+C 停止) <<<\n")

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
                outcome_key, done_reason = get_episode_outcome(reward)
                stats[outcome_key] += 1

                print(
                    f"Episode {stats['episodes']} 结束 | 原因: {done_reason} | 总分: {episode_reward:.2f} | 步数: {step_count}")

                append_trajectory_rows(csv_filename, current_traj, done_reason)

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
                cv2.imshow("AirSim AI Dashboard", dashboard)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户手动停止测试。")
                    break

    except KeyboardInterrupt:
        print("\n检测到键盘中断，停止测试。")
    finally:
        print("\n" + "=" * 50)
        print("              测试总结报告              ")
        print("=" * 50)
        total = stats['episodes']
        if total > 0:
            print(f"总回合数: {total}")
            print(f"成功次数: {stats['success']} ({stats['success'] / total * 100:.1f}%)")
            print(f"撞墙次数: {stats['collision']} ({stats['collision'] / total * 100:.1f}%)")
            print(f"越界次数: {stats['out_of_bounds']} ({stats['out_of_bounds'] / total * 100:.1f}%)")
            if csv_filename is not None:
                print(f"轨迹数据已保存至: {csv_filename}")
        else:
            print("未完成任何完整回合。")
        print("=" * 50)

        if env is not None:
            env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
