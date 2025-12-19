import cv2
import argparse
import numpy as np
import os  # 新增：用于创建截图目录
from drone_control import VirtualDrone
from detection_module import DroneDetection


def parse_args():
    parser = argparse.ArgumentParser(description="AI无人机面部识别与人体追踪（虚拟版）")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--track-thres", type=float, default=0.4, help="追踪IOU阈值")
    parser.add_argument("--map-alpha", type=float, default=0.3, help="地图透明度")
    return parser.parse_args()


def draw_clean_text(img, text, pos, color=(0, 255, 0), font_size=0.6):
    """过滤乱码字符，保证文本正常显示"""
    valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :.-()%")
    clean_text = ''.join([c for c in text if c in valid_chars])
    cv2.putText(
        img, clean_text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, font_size,
        color, 1, lineType=cv2.LINE_AA
    )


def init_screenshot_dir():
    """初始化截图保存目录，避免保存失败"""
    if not os.path.exists("drone_screenshots"):
        os.makedirs("drone_screenshots")
    return "drone_screenshots"


def main():
    args = parse_args()
    drone = VirtualDrone()
    detector = DroneDetection(drone=drone)
    screenshot_dir = init_screenshot_dir()  # 初始化截图目录

    # 打印初始化信息（明确截图键为Z）
    print("=" * 60)
    print("✅ 虚拟无人机系统初始化完成")
    print(f"初始状态 | 电量：{drone.get_battery():.1f}% | 状态：{drone.state.value} | 位置：{drone.position}")
    print("=" * 60)
    print("🎮 操作说明：")
    print("  ESC → 退出程序 | T → 起飞 | L → 降落 | Z → 保存截图（保存至drone_screenshots目录）")
    print("  W/A/S/D → 前/后/左/右 | ↑/↓ → 上升/下降 | Q/E → 左转/右转")
    print("=" * 60)

    # 创建可视化窗口
    cv2.namedWindow("AI Drone Control System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Drone Control System", 800, 600)

    # 检测项解释映射
    detection_explain = {
        "状态检测": "State Detection",
        "电量检测": "Battery Detection",
        "位置检测": "Position Detection",
        "障碍物检测": "Obstacle Detection",
        "碰撞预警": "Collision Warning"
    }

    try:
        while True:
            # 创建黑色背景帧
            frame = np.zeros((600, 800, 3), dtype=np.uint8)

            # 绘制无人机基础状态
            status_y = 30
            draw_clean_text(frame, f"Battery: {drone.get_battery():.1f}%", (10, status_y), (0, 255, 0))
            draw_clean_text(frame, f"Position: {drone.position.round(1)}", (10, status_y + 30), (0, 255, 0))
            draw_clean_text(frame, f"State: {drone.state.value}", (10, status_y + 60), (0, 255, 0))
            draw_clean_text(frame, f"Yaw Angle: {drone.yaw:.0f}°", (10, status_y + 90), (0, 255, 0))

            # 绘制检测结果（带解释）
            detection_y = 150
            draw_clean_text(frame, "=== Detection Results (检测结果解释) ===", (10, detection_y), (255, 255, 0))
            draw_clean_text(frame, "【State:状态 | Battery:电量 | Position:位置 | Obstacle:障碍物 | Collision:碰撞】",
                            (10, detection_y + 20), (255, 255, 255), 0.4)
            detection_y += 40

            detection_results = detector.full_detection()
            for idx, res in enumerate(detection_results):
                detection_y += 30
                color = (0, 0, 255) if res.get("warning") or res.get("risk") else (0, 255, 0)
                core_msg = res['message'].split("|")[0].strip()
                explain = detection_explain.get(res['type'], "Unknown")
                display_text = f"{explain}: {core_msg}"
                draw_clean_text(frame, display_text, (10, detection_y), color, 0.5)

            # 绘制操作提示（明确截图键）
            draw_clean_text(
                frame,
                "Operation: ESC(Exit) | T(Takeoff) | L(Land) | Z(Save) | Q/E(Rotate) | W/A/S/D(Move)",
                (10, 550), (255, 255, 255), 0.45
            )

            # 显示画面
            cv2.imshow("AI Drone Control System", frame)

            # 键盘控制逻辑（核心：截图键为Z，S仅后退）
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC退出
                print("\n👋 程序退出中...")
                break
            elif key == ord('t'):  # T起飞
                drone.takeoff()
            elif key == ord('l'):  # L降落
                drone.land()
            elif key == ord('z'):  # Z截图（独立键，无冲突）
                # 生成唯一截图文件名（时间戳+状态）
                screenshot_name = f"drone_{drone.state.value}_{cv2.getTickCount()}.jpg"
                screenshot_path = os.path.join(screenshot_dir, screenshot_name)
                # 保存截图
                cv2.imwrite(screenshot_path, frame)
                print(f"✅ 截图已保存：{screenshot_path}")
            elif key == ord('w'):  # W前进
                drone.move("forward")
            elif key == ord('s'):  # S后退（仅后退，无截图）
                drone.move("back")
            elif key == ord('a'):  # A左移
                drone.move("left")
            elif key == ord('d'):  # D右移
                drone.move("right")
            elif key == 2490368:  # 上方向键上升
                drone.move("up")
            elif key == 2621440:  # 下方向键下降
                drone.move("down")
            elif key == ord('q'):  # Q左转
                drone.rotate("left")
            elif key == ord('e'):  # E右转
                drone.rotate("right")

    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")
        print("💡 请检查依赖：pip install opencv-python numpy")
    finally:
        # 退出前收尾
        if drone.state.value == "Flying":
            drone.land()
            print("✅ 无人机已自动降落")
        cv2.destroyAllWindows()
        print("✅ 程序正常退出")


if __name__ == "__main__":
    main()