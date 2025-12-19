import cv2
import argparse
import numpy as np
from drone_control import VirtualDrone  # 导入正确的无人机类
from detection_module import DroneDetection  # 导入正确的检测类


# 注：MapOverlay/FaceDatabase为自定义模块，若未实现先注释，避免运行报错
# from map_overlay import MapOverlay
# from face_database import FaceDatabase


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AI无人机面部识别与人体追踪（虚拟版）")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--track-thres", type=float, default=0.4, help="追踪IOU阈值")
    parser.add_argument("--map-alpha", type=float, default=0.3, help="地图透明度")
    return parser.parse_args()


def main():
    args = parse_args()

    # ===================== 初始化核心模块（适配虚拟无人机） =====================
    # 1. 初始化虚拟无人机（替换原TelloDrone）
    drone = VirtualDrone()
    # 2. 初始化检测模块（关联无人机，替换原DetectionEngine）
    detector = DroneDetection(drone=drone)
    # 3. 人脸数据库/地图叠加（未实现则注释，后续可补充）
    # face_db = FaceDatabase(db_path="face_database/")
    # map_overlay = MapOverlay(map_path=args.map_path, alpha=args.map_alpha)

    # ===================== 模拟初始化逻辑 =====================
    # 加载人脸库（注释，待实现FaceDatabase后启用）
    # face_db.load_all_faces()
    # print(f"人脸库加载完成，共{len(face_db.get_face_names())}个人脸")

    # 虚拟无人机无需真实连接，模拟启动视频流
    print("✅ 虚拟无人机初始化完成")
    print(f"初始电量：{drone.get_battery()}% | 初始状态：{drone.state.value}")

    # ===================== 主循环（适配虚拟无人机逻辑） =====================
    try:
        # 创建虚拟视频窗口（模拟无人机视频流）
        cv2.namedWindow("AI Drone Face & Human Tracking", cv2.WINDOW_NORMAL)

        while True:
            # 1. 生成虚拟帧（替代真实无人机视频帧）
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # 在虚拟帧上绘制无人机状态
            cv2.putText(frame, f"Battery: {drone.get_battery():.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Position: {drone.position.round(1)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"State: {drone.state.value}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 2. 执行无人机状态检测（替代真实人脸/人体检测）
            detection_results = detector.full_detection()
            # 在帧上绘制检测预警信息
            y_offset = 150
            for res in detection_results:
                color = (0, 0, 255) if res.get("warning") or res.get("risk") else (0, 255, 0)
                cv2.putText(frame, res["message"], (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30

            # 3. 地图叠加（注释，待实现MapOverlay后启用）
            # frame = map_overlay.overlay(frame)

            # 4. 显示画面
            cv2.imshow("AI Drone Face & Human Tracking", frame)

            # ===================== 按键控制（适配虚拟无人机） =====================
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 退出
                break
            elif key == ord('t'):  # 起飞
                drone.takeoff()
            elif key == ord('l'):  # 降落
                drone.land()
            elif key == ord('s'):  # 保存画面
                cv2.imwrite(f"drone_capture_{cv2.getTickCount()}.jpg", frame)
                print("✅ 画面已保存")
            # 无人机移动控制
            elif key == ord('w'):
                drone.move("forward")
            elif key == ord('s'):
                drone.move("back")
            elif key == ord('a'):
                drone.move("left")
            elif key == ord('d'):
                drone.move("right")
            elif key == 2490368:  # 上方向键：上升
                drone.move("up")
            elif key == 2621440:  # 下方向键：下降
                drone.move("down")
            elif key == ord('q'):  # 左转
                drone.rotate("left")
            elif key == ord('e'):  # 右转
                drone.rotate("right")

    except Exception as e:
        print(f"❌ 程序异常：{str(e)}")
    finally:
        # ===================== 资源释放（适配虚拟无人机） =====================
        print("\n🔄 程序退出，释放资源...")
        if drone.state.value == "FLYING":
            drone.land()  # 紧急降落
        cv2.destroyAllWindows()
        print("✅ 程序正常退出")


if __name__ == "__main__":
    main()