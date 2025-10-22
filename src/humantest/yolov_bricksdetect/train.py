import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

model = YOLO("E:/yolov8/ultralytics-main/ultralytics/yolov8n.pt")

results = model.train(data="E:/yolov8/ultralytics-main/ultralytics/datasets/bricks.yaml", imgsz=640, epochs=100, batch=16, device='cpu', workers=4)