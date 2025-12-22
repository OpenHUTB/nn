# 创建 test_compat.py
import numpy as np
import cv2

print(f"NumPy 版本: {np.__version__}")
print(f"OpenCV 版本: {cv2.__version__}")

# 测试关键功能
img = np.zeros((100, 100, 3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("OpenCV 与 NumPy 2.x 兼容性测试通过！")