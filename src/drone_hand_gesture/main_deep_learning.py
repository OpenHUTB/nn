import cv2
import os
import argparse
from deep_gesture_detector import DeepGestureDetector
from gesture_visualizer import GestureVisualizer

def main():
    parser = argparse.ArgumentParser(description='深度学习手势控制无人机演示')
    parser.add_argument('--model_path', type=str, default=None,
                        help='预训练深度学习模型路径')
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'transformer', 'mlp'],
                        help='深度学习模型类型')
    parser.add_argument('--camera', type=int, default=0,
                        help='摄像头索引')
    parser.add_argument('--show_charts', action='store_true', default=True,
                        help='显示统计图表')
    parser.add_argument('--save_video', action='store_true', default=False,
                        help='保存视频')
    
    args = parser.parse_args()
    
    # 初始化手势检测器
    detector = DeepGestureDetector(
        model_path=args.model_path,
        model_type=args.model_type,
        use_deep_learning=True if args.model_path else False
    )
    
    # 初始化可视化器
    visualizer = GestureVisualizer()
    
    # 初始化摄像头
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 视频保存
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter('gesture_recognition.avi', fourcc, 30, (640, 480))
    
    print("=== 深度学习手势识别演示 ===")
    print("按 'q' 或 'ESC' 退出")
    print("按 'r' 重置统计")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测手势
        frame, gestures, final_gesture = detector.detect_gestures(frame)
        
        # 获取置信度
        confidence = gestures[0]['confidence'] if gestures else 0.0
        
        # 获取命令
        command = detector.get_command(final_gesture)
        command_text = visualizer.gesture_commands.get(final_gesture, "无命令")
        
        # 更新可视化器
        visualizer.update_history(final_gesture, confidence)
        
        # 绘制信息面板
        frame = visualizer.draw_info_panel(frame, final_gesture, confidence, command_text)
        
        # 绘制手势图标
        frame = visualizer.draw_gesture_icon(frame, final_gesture)
        
        # 绘制手势信息
        frame = detector.draw_gesture_info(frame, gestures)
        
        # 显示统计图表
        if args.show_charts:
            chart_img = visualizer.update_charts()
            chart_img = cv2.resize(chart_img, (320, 160))
            frame[300:460, 320:640] = chart_img
        
        # 显示画面
        cv2.imshow('Deep Learning Gesture Recognition', frame)
        
        # 保存视频
        if video_writer:
            video_writer.write(frame)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # ESC
            break
        elif key == ord('r'):
            visualizer.reset()
            print("统计已重置")
    
    # 释放资源
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    detector.release()
    
    # 显示统计信息
    visualizer.show_statistics()

if __name__ == "__main__":
    main()