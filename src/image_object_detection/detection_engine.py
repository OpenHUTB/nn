# detection_engine.py
from ultralytics import YOLO
import io
import sys

class DetectionEngine:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = self._load_model()
    
    def _load_model(self):
        """加载YOLO模型并抑制输出"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            model = YOLO(self.model_path)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return model
    
    def detect(self, frame):
        """对输入帧进行检测"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        annotated_frame = results[0].plot()
        return annotated_frame, results
    
    def find_objects(self, frame):
        """检测并分类图像中的对象"""
        _, results = self.detect(frame)
        objects = []
        for box in results[0].boxes:
            cls_index = int(box.cls)
            cls_name = self.model.names[cls_index]
            coords = box.xyxy.tolist()[0]
            confidence = box.conf.item()
            objects.append({
                "type": cls_name,
                "bbox": coords,
                "confidence": confidence
            })
        return objects