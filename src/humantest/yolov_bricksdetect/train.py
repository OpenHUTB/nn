import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.train(data="bricks.yaml", imgsz=640, epochs=100, batch=16, device='cpu', workers=4)