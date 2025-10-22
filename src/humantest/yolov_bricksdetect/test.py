from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# 1. 加载模型
model = YOLO("E:/yolov8/ultralytics-main/runs/detect/train3/weights/best.pt")

# 2. 设置测试图像文件夹和结果保存文件夹
test_img_dir = "E:/yolov8/ultralytics-main/ultralytics/datasets/bricks/test/images"  # 存放待测试图像的文件夹
save_dir = "E:/yolov8/ultralytics-main/runs/pre"  # 保存检测结果的文件夹
os.makedirs(save_dir, exist_ok=True)  # 创建保存目录

# 3. 遍历文件夹中的所有图像
for img_file in os.listdir(test_img_dir):
    # 过滤非图像文件
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue
    
    img_path = os.path.join(test_img_dir, img_file)
    print(f"正在检测：{img_path}")
    
    # 4. 执行检测
    results = model(img_path, conf=0.5, device='cpu')
    
    # 5. 保存结果
    for result in results:
        annotated_img = result.plot()
        save_path = os.path.join(save_dir, img_file)
        cv2.imwrite(save_path, annotated_img)
        print(f"已保存：{save_path}")

print("批量检测完成！")