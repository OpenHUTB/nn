import numpy as np
import cv2
import os

class YOLOInference:
    def __init__(self, model_path=None, conf_threshold=0.5, iou_threshold=0.45, use_cuda=False):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_cuda = use_cuda
        self.model = None
        self.class_names = []
        self.input_size = (640, 640)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            try:
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt" if not model_path else model_path)
                if use_cuda:
                    self.model.to('cuda')
                self.class_names = list(self.model.names.values())
            except ImportError:
                print("Warning: ultralytics not installed, using OpenCV DNN")
                self._load_opencv_dnn()

    def _load_opencv_dnn(self):
        if self.model_path and os.path.exists(self.model_path):
            self.net = cv2.dnn.readNet(self.model_path)
            if self.use_cuda:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.class_names = [f"class_{i}" for i in range(80)]

    def detect(self, image):
        if image is None or image.size == 0:
            return []

        if self.model is not None:
            try:
                results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                        })
                return detections
            except Exception as e:
                print(f"YOLO detection error: {e}")
                return []
        elif hasattr(self, 'net'):
            return self._detect_opencv(image)
        else:
            return []

    def _detect_opencv(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)

        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        detections = []

        if len(outputs) > 0:
            outputs = outputs[0]

        if len(outputs) > 0:
            for detection in outputs:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    width = int(detection[2] * image.shape[1])
                    height = int(detection[3] * image.shape[0])

                    x1 = center_x - width // 2
                    y1 = center_y - height // 2
                    x2 = x1 + width
                    y2 = y1 + height

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    })

        return detections

    def annotate_image(self, image, detections):
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return annotated

    def get_center_offset(self, image_shape, detection):
        h, w = image_shape[:2]
        x1, y1, x2, y2 = detection['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        offset_x = (center_x - w / 2) / (w / 2)
        offset_y = (center_y - h / 2) / (h / 2)
        return offset_x, offset_y

    def get_detection_info(self, image_shape, detections):
        if not detections:
            return {
                'has_target': False,
                'target_center_offset': (0.0, 0.0),
                'target_distance': 1.0,
                'target_size': 0.0,
                'num_detections': 0
            }

        closest = max(detections, key=lambda d: d['confidence'])
        offset_x, offset_y = self.get_center_offset(image_shape, closest)

        x1, y1, x2, y2 = closest['bbox']
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = image_shape[0] * image_shape[1]
        relative_size = bbox_area / image_area

        return {
            'has_target': True,
            'target_center_offset': (offset_x, offset_y),
            'target_distance': 1.0 - relative_size,
            'target_size': relative_size,
            'num_detections': len(detections),
            'best_detection': closest
        }
