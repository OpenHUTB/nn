import cv2
import argparse
from drone_control import TelloDrone
from detection_module import DetectionEngine
from map_overlay import MapOverlay
from face_database import FaceDatabase


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AI无人机面部识别与人体追踪")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO模型路径")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--track-thres", type=float, default=0.4, help="追踪IOU阈值")
    parser.add_argument("--map-path", default="map.png", help="地图图片路径")
    parser.add_argument("--map-alpha", type=float, default=0.3, help="地图透明度")
    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化核心模块
    drone = TelloDrone()  # 无人机控制
    detector = DetectionEngine(
        model_path=args.yolo_model,
        conf_thres=args.conf_thres,
        track_thres=args.track_thres
    )  # 检测引擎
    face_db = FaceDatabase(db_path="face_database/")  # 人脸数据库
    map_overlay = MapOverlay(map_path=args.map_path, alpha=args.map_alpha)  # 地图叠加

    # 加载人脸库
    face_db.load_all_faces()
    print(f"人脸库加载完成，共{len(face_db.get_face_names())}个人脸")

    # 无人机连接与视频流启动
    if not drone.connect():
        print("无人机连接失败！")
        return
    drone.start_video_stream()
    print(f"无人机电量：{drone.get_battery()}%")

    try:
        while True:
            # 获取无人机视频帧
            frame = drone.get_frame()
            if frame is None:
                continue

            # 1. 检测人脸/人体
            results = detector.detect(frame)
            # 2. 人脸匹配
            frame = detector.match_faces(frame, results, face_db)
            # 3. 人体追踪（可选：追踪最大的人体目标）
            track_target = detector.get_largest_human(results)
            if track_target:
                # 无人机追踪逻辑（简化版：根据目标位置调整云台/飞行）
                drone.track_target(frame, track_target)

            # 4. 叠加地图
            frame = map_overlay.overlay(frame)

            # 显示画面
            cv2.imshow("AI Drone Face & Human Tracking", frame)

            # 按键控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 退出
                break
            elif key == ord('t'):  # 起飞
                drone.takeoff()
            elif key == ord('l'):  # 降落
                drone.land()
            elif key == ord('s'):  # 保存画面
                cv2.imwrite(f"drone_capture_{cv2.getTickCount()}.jpg", frame)
                print("画面已保存")

    except Exception as e:
        print(f"程序异常：{str(e)}")
    finally:
        # 资源释放
        drone.stop_video_stream()
        drone.land()  # 紧急降落
        cv2.destroyAllWindows()
        print("程序正常退出")


if __name__ == "__main__":
    main()