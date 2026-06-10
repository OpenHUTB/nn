import cv2
import time
from detection_engine import DetectionEngine

class VideoDetector:
    def __init__(self):
        """初始化检测引擎"""
        self.engine = DetectionEngine()

    def process_video(self, video_path, output_path=None):
        """
        处理视频文件
        
        参数:
            video_path (str): 输入视频文件路径
            output_path (str, optional): 输出视频文件路径（如果为 None 则不保存）
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 准备视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()
        print(f"开始处理视频，总帧数: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 调用检测引擎，得到标注后的图像
            annotated_frame, _ = self.engine.detect(frame)

            # 显示
            cv2.imshow("Video Detection", annotated_frame)

            # 保存（如果指定了输出路径）
            if writer:
                writer.write(annotated_frame)

            frame_count += 1

            # 按 'q' 键提前退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 每 30 帧打印一次进度
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_estimate = frame_count / elapsed if elapsed > 0 else 0
                print(f"进度: {frame_count}/{total_frames} 帧, 实时 FPS: {fps_estimate:.2f}")

        # 释放资源
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start_time
        print(f"处理完成！共处理 {frame_count} 帧，总耗时 {total_time:.2f} 秒，平均 FPS: {frame_count/total_time:.2f}")

    def process_video_file(self, video_path, output_path=None):
        """便捷方法，直接调用 process_video"""
        self.process_video(video_path, output_path)