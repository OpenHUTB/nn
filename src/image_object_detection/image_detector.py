# image_detector.py
import cv2
import numpy as np
import traceback

class ImageDetector:
    def __init__(self, detection_engine):
        self.engine = detection_engine
    
    def detect_static_image(self, image_path):
        """检测静态图像"""
        print(f"正在加载图像: {image_path}")
        
        if not self._check_image_exists(image_path):
            return
        
        frame = self._load_image(image_path)
        if frame is None or frame.size == 0:
            print("错误: 无法读取图像文件或图像为空")
            return
        
        print("正在检测图像...")
        results, annotated_frame = self._perform_detection(frame)
        
        if annotated_frame is None:
            print("错误: 检测未返回有效图像")
            return
        
        # 显示检测结果文本
        self._display_results(results)
        
        # 显示图像窗口（带固定初始大小）
        self._show_image(annotated_frame, 'YOLO 检测结果 - 静态图像')
    
    def _check_image_exists(self, image_path):
        import os
        if not os.path.exists(image_path):
            print(f"错误: 图像文件不存在 - {image_path}")
            return False
        return True
    
    def _load_image(self, image_path):
        try:
            with open(image_path, "rb") as f:
                bytes_data = bytearray(f.read())
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"加载图像时出错: {e}")
            return None
    
    def _perform_detection(self, frame):
        try:
            annotated_frame, results = self.engine.detect(frame)
            return results, annotated_frame
        except Exception as e:
            print(f"检测过程中发生错误: {e}")
            traceback.print_exc()
            return [], None
    
    def _display_results(self, results):
        if not results:
            print("未检测到任何对象（results 为空）")
            return
        
        result = results[0]
        boxes = result.boxes
        if len(boxes) == 0:
            print("未检测到任何对象")
            return
        
        print("检测完成，正在显示结果...")
        print(f"检测到 {len(boxes)} 个对象:")
        
        names_list = self.engine.model.names
        
        for i, box in enumerate(boxes):
            try:
                cls_index = int(box.cls.item())
                confidence = box.conf.item()
                
                if 0 <= cls_index < len(names_list):
                    cls_name = names_list[cls_index]
                else:
                    cls_name = f"unknown_class_{cls_index}"
                    print(f"  ⚠️ 警告：类别索引 {cls_index} 超出范围（共 {len(names_list)} 类）")
                
                print(f"  {i+1}. {cls_name} (置信度: {confidence:.2f})")
            except Exception as e:
                print(f"  ⚠️ 解析第 {i+1} 个检测框时出错: {e}")
    
    def _show_image(self, annotated_frame, window_name):
        """显示图像，并设置窗口为可调整的小尺寸"""
        if annotated_frame is None or annotated_frame.size == 0:
            print("错误: 标注帧无效，无法显示")
            return
        
        print("\n图像已显示。按任意键关闭窗口...")
        try:
            # 创建可调整大小的窗口
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # 设置初始窗口大小为 800x600（你可以按需修改）
            cv2.resizeWindow(window_name, 800, 600)
            # 显示图像（OpenCV 会自动适应窗口）
            cv2.imshow(window_name, annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("窗口已关闭。")
        except Exception as e:
            print(f"显示图像时发生错误: {e}")
            traceback.print_exc()
