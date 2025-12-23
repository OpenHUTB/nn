# monocular_3d_detector.py（修正版）
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 YOLO
yolo_model = YOLO("yolov8n.pt")#pip install ultralytics 即可下载YOLOv8n.pt

# 正确加载 MiDaS（无需 pip install midas！）
print("Loading MiDaS model...")
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
midas_model.to(device).eval()

def pixel_to_3d(x, y, depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = depth[y, x]
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return np.array([X, Y, Z])

def detect_monocular_3d(image_path, output_path="output_3d.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # 相机内参（示例值）
    K = np.array([[721.5, 0, W / 2],
                  [0, 721.5, H / 2],
                  [0, 0, 1]])

    # 2D detection
    results = yolo_model(img_rgb)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names

    # Depth estimation
    input_batch = midas_transform(img_rgb).to(device)
    with torch.no_grad():
        depth_pred = midas_model(input_batch)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = depth_pred.cpu().numpy()

    # Visualize depth
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
    plt.imsave("depth_map.png", depth_vis, cmap='plasma')

    # Estimate 3D centers
    print("Detected objects with approximate 3D centers:")
    for i, (box, cls_id) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
        d = depth[cy, cx]
        p3d = pixel_to_3d(cx, cy, depth, K)
        label = f"{names[int(cls_id)]}: ({p3d[0]:.2f}, {p3d[1]:.2f}, {p3d[2]:.2f})"
        print(f"  {label}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, names[int(cls_id)], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"\nResult saved to {output_path}")

if __name__ == "__main__":
    detect_monocular_3d("data/sample_image.jpg")#https://699pic.com/tupian-501301794.html  图片下载链接
