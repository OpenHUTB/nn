from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt'):
        self.model = YOLO(model_name)
        # 目标: 2=Car, 3=Motor, 5=Bus, 7=Truck
        self.target_classes = [2, 3, 5, 7]

    def detect(self, frame):
        """
        返回: (original_frame, detections_list)
        """
        results = self.model(frame, stream=True, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf < 0.4: continue

                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'class': self.model.names[cls_id],
                        'conf': conf,
                        'width': x2 - x1,
                        'center': ((x1+x2)//2, (y1+y2)//2)
                    })
        return frame, detections