import cv2
import os

os.makedirs("outputs", exist_ok=True)
img_path = "outputs/prediction.png"
img = cv2.imread(img_path)

if img is None:
    print(f"错误：找不到图片 {img_path}，请先运行 data_collect.py 并保持运行！")
else:
    cv2.imshow('Temporal Collage Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()