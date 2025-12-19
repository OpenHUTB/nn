import cv2
import argparse
import numpy as np
import os
from drone_control import VirtualDrone
from detection_module import DroneDetection


def parse_args():
    parser = argparse.ArgumentParser(description="AI无人机+真实摄像头版")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--track-thres", type=float, default=0.4, help="追踪IOU阈值")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头ID（0为默认摄像头，1为外接）")
    return parser.parse_args()


def draw_clean_text(img, text, pos, color=(0, 255, 0), font_size=0.6):
    valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :.-()%")
    clean_text = ''.join([c for c in text if c in valid_chars])
    cv2.putText(
        img, clean_text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, font_size,
        color, 1, lineType=cv2.LINE_AA
    )


def init_screenshot_dir():
    if not os.path.exists("drone_screenshots"):
        os.makedirs("drone_screenshots")
    return "drone_screenshots"


def main():
    args = parse_args()
    drone = VirtualDrone()
    detector = DroneDetection(drone=drone)
    screenshot_dir = init_screenshot_dir()

    # ===================== 新增：初始化摄像头 =====================
    cap = cv2.VideoCapture(args.camera_id)  # 打开摄像头（0=默认，1=外接）
    if not cap.isOpened():  # 检查摄像头是否打开成功
        print("❌ 无法打开摄像头！请检查：")
        print("  1. 摄像头是否被其他程序占用（如微信、浏览器）")
        print("  2. Python是否有摄像头访问权限（系统设置→隐私）")
        print("  3. 摄像头ID是否正确（尝试修改--camera-id 1）")
        return  # 打开失败则退出程序

    # 打印初始化信息
    print("=" * 60)
    print("✅ 虚拟无人机+摄像头系统初始化完成")
    print(f"初始状态 | 电量：{drone.get_battery():.1f}% | 状态：{drone.state.value}")
    print(f"摄像头状态 | ID：{args.camera_id} | 已成功打开")
    print("=" * 60)
    print("🎮 操作说明：")
    print("  ESC → 退出 | T → 起飞 | L → 降落 | Z → 保存截图")
    print("  W/A/S/D → 前后左右 | ↑/↓ → 上升/下降 | Q/E → 左转/右转")
    print("=" * 60)

    # 创建可视化窗口
    cv2.namedWindow("AI Drone + Camera System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Drone + Camera System", 1280, 720)  # 适配摄像头分辨率

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
            # ===================== 读取摄像头画面（核心新增） =====================
            ret, frame = cap.read()  # 读取摄像头一帧画面
            if not ret:  # 摄像头读取失败（如断开）
                print("❌ 摄像头画面读取失败！")
                break
            # 调整画面尺寸适配窗口
            frame = cv2.resize(frame, (1280, 720))

            # ===================== 绘制无人机状态（叠加在摄像头画面上） =====================
            # 1. 绘制基础状态（顶部左侧）
            status_y = 30
            draw_clean_text(frame, f"Battery: {drone.get_battery():.1f}%", (10, status_y), (0, 255, 0), 0.7)
            draw_clean_text(frame, f"Position: {drone.position.round(1)}", (10, status_y + 40), (0, 255, 0), 0.7)
            draw_clean_text(frame, f"State: {drone.state.value}", (10, status_y + 80), (0, 255, 0), 0.7)
            draw_clean_text(frame, f"Yaw Angle: {drone.yaw:.0f}°", (10, status_y + 120), (0, 255, 0), 0.7)

            # 2. 绘制检测结果（顶部右侧，避免遮挡摄像头画面）
            detection_y = 30
            draw_clean_text(frame, "=== Detection Results ===", (800, detection_y), (255, 255, 0), 0.7)
            draw_clean_text(frame, "【State:状态 | Battery:电量 | Position:位置】", (800, detection_y + 40),
                            (255, 255, 255), 0.5)
            detection_y += 80

            detection_results = detector.full_detection()
            for idx, res in enumerate(detection_results):
                detection_y += 40
                color = (0, 0, 255) if res.get("warning") else (0, 255, 0)
                core_msg = res['message'].split("|")[0].strip()
                explain = detection_explain.get(res['type'], "Unknown")
                display_text = f"{explain}: {core_msg}"
                draw_clean_text(frame, display_text, (800, detection_y), color, 0.6)

            # 3. 绘制操作提示（底部）
            draw_clean_text(
                frame,
                "Operation: ESC(Exit) | T(Takeoff) | L(Land) | Z(Save) | Q/E(Rotate) | W/A/S/D(Move)",
                (10, 680), (255, 255, 255), 0.6
            )

            # 显示摄像头+无人机状态叠加画面
            cv2.imshow("AI Drone + Camera System", frame)

            # ===================== 键盘控制逻辑 =====================
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC退出
                print("\n👋 程序退出中...")
                break
            elif key == ord('t'):
                drone.takeoff()
            elif key == ord('l'):
                drone.land()
            elif key == ord('z'):  # Z保存截图（摄像头画面+无人机状态）
                screenshot_name = f"drone_camera_{drone.state.value}_{cv2.getTickCount()}.jpg"
                screenshot_path = os.path.join(screenshot_dir, screenshot_name)
                cv2.imwrite(screenshot_path, frame)
                print(f"✅ 摄像头截图已保存：{screenshot_path}")
            elif key == ord('w'):
                drone.move("forward")
            elif key == ord('s'):
                drone.move("back")
            elif key == ord('a'):
                drone.move("left")
            elif key == ord('d'):
                drone.move("right")
            elif key == 2490368:
                drone.move("up")
            elif key == 2621440:
                drone.move("down")
            elif key == ord('q'):
                drone.rotate("left")
            elif key == ord('e'):
                drone.rotate("right")

    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")
        print("💡 建议检查：摄像头是否可用 | OpenCV版本（pip install opencv-python --upgrade）")
    finally:
        # ===================== 资源释放（核心：关闭摄像头） =====================
        cap.release()  # 关闭摄像头
        if drone.state.value == "Flying":
            drone.land()
            print("✅ 无人机已自动降落")
        cv2.destroyAllWindows()
        print("✅ 摄像头已关闭，程序正常退出")


if __name__ == "__main__":
    main()