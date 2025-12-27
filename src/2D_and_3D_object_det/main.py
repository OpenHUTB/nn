# monocular_3d_detector.py（带详细注释的修正版）

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 自动选择设备：如果有 CUDA GPU 则使用 GPU，否则使用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 YOLOv8 模型用于 2D 目标检测（使用轻量级模型 yolov8n）
yolo_model = YOLO("yolov8n.pt")

# 加载 MiDaS 深度估计模型（使用小型模型以加快推理速度）
print("Loading MiDaS model...")
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# 获取对应的图像预处理变换函数（适用于 small 模型）
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
# 将 MiDaS 模型移动到指定设备并设为评估模式
midas_model.to(device).eval()

def pixel_to_3d(x, y, depth, K):
    """
    将像素坐标 (x, y) 和深度图 depth 转换为相机坐标系下的 3D 点。
    
    参数:
        x, y: 像素坐标（整数）
        depth: 深度图（H×W 的 NumPy 数组）
        K: 相机内参矩阵（3×3）
    
    返回:
        3D 点 [X, Y, Z]（单位与深度图一致，通常为任意尺度）
    """
    fx, fy = K[0, 0], K[1, 1]  # 焦距（像素单位）
    cx, cy = K[0, 2], K[1, 2]  # 主点（图像中心）

    Z = depth[y, x]            # 获取该像素处的深度值
    X = (x - cx) * Z / fx      # 根据针孔相机模型反投影
    Y = (y - cy) * Z / fy
    return np.array([X, Y, Z])

def detect_monocular_3d(image_path, output_path="output_3d.jpg"):
    """
    对单张图像执行 2D 目标检测 + 单目深度估计，并估算每个检测框中心的 3D 坐标。
    
    参数:
        image_path: 输入图像路径
        output_path: 输出结果图像保存路径（带检测框和类别标签）
    """
    # 读取输入图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # 转换为 RGB（YOLO 和 MiDaS 都期望 RGB 输入）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]  # 获取图像高宽

    # 定义相机内参矩阵 K（这里使用 KITTI 数据集常用近似值，fx=fy=721.5）
    # 实际应用中应使用真实标定参数；若未知，可假设主点在图像中心
    K = np.array([
        [721.5,     0,   W / 2],
        [    0, 721.5,   H / 2],
        [    0,     0,       1]
    ])

    # Step 1: 使用 YOLO 进行 2D 目标检测
    results = yolo_model(img_rgb)
    boxes = results[0].boxes.xyxy.cpu().numpy()      # 检测框 [x1, y1, x2, y2]
    classes = results[0].boxes.cls.cpu().numpy()     # 类别 ID
    names = results[0].names                         # 类别名称字典 {0: 'person', ...}

    # Step 2: 使用 MiDaS 估计单目深度图
    input_batch = midas_transform(img_rgb).to(device)  # 应用预处理并移至设备
    with torch.no_grad():
        depth_pred = midas_model(input_batch)          # 前向推理得到深度图（低分辨率）
        # 上采样到原始图像尺寸（使用双三次插值）
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = depth_pred.cpu().numpy()  # 转为 NumPy 数组

    # 可视化深度图并保存
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min())  # 归一化到 [0,1]
    plt.imsave("depth_map.png", depth_vis, cmap='plasma')
    print("Depth map saved as depth_map.png")

    # Step 3: 对每个检测目标，估算其 3D 中心点
    print("Detected objects with approximate 3D centers:")
    for i, (box, cls_id) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)  # 检测框中心像素坐标

        d = depth[cy, cx]  # 获取中心点深度（注意：MiDaS 输出是逆深度的相对值，此处当作正深度使用）
        p3d = pixel_to_3d(cx, cy, depth, K)  # 转换为 3D 点

        # 构造显示标签
        label = f"{names[int(cls_id)]}: ({p3d[0]:.2f}, {p3d[1]:.2f}, {p3d[2]:.2f})"
        print(f"  {label}")

        # 在原图上绘制检测框和类别名
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, names[int(cls_id)], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 保存带标注的结果图像
    cv2.imwrite(output_path, img)
    print(f"\nResult saved to {output_path}")

if __name__ == "__main__":
    # 默认使用 sample_image.jpg 进行测试
    detect_monocular_3d("data/sample_image.jpg")

# 图片下载链接（供参考）：
# https://699pic.com/tupian-501301794.html
