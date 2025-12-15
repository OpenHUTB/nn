# image_detector.py
import cv2

class ImageDetector:
    def __init__(self, detection_engine):
        self.engine = detection_engine
    
    def detect_static_image(self, image_path):
        """检测静态图像"""
        print(f"正在加载图像: {image_path}")
        
        if not self._check_image_exists(image_path):
            return
        
        frame = self._load_image(image_path)
        if frame is None:
            print("错误: 无法读取图像文件")
            return
        
        print("正在检测图像...")
        results = self._perform_detection(frame)
        
        # 显示检测结果
        self._display_results(results)
        
        # 显示图像
        self._show_image(results[1][0].plot(), 'YOLO 检测结果 - 静态图像')
    
    def _check_image_exists(self, image_path):
        """检查图像文件是否存在"""
        if not cv2.os.path.exists(image_path):
            print(f"错误: 图像文件不存在 - {image_path}")
            return False
        return True
    
    def _load_image(self, image_path):
        """加载图像"""
        return cv2.imread(image_path)
    
    def _perform_detection(self, frame):
        """执行检测"""
        annotated_frame, results = self.engine.detect(frame)
        return results
    
    def _display_results(self, results):
        """显示检测结果"""
        detected_count = len(results[0].boxes)
        print("检测完成，正在显示结果...")
        print(f"检测到 {detected_count} 个对象:")
        
        for i, box in enumerate(results[0].boxes):
            cls_index = int(box.cls)
            cls_name = self.engine.model.names[cls_index]
            confidence = box.conf.item()
            print(f"  {i+1}. {cls_name} (置信度: {confidence:.2f})")
    
    def _show_image(self, annotated_frame, window_name):
        """显示图像"""
        print("\n图像已显示。按任意键关闭窗口...")
        cv2.imshow(window_name, annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("窗口已关闭。")