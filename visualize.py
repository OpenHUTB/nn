import cv2
import numpy as np

# 读取预测结果
img = cv2.imread('outputs/prediction.png')
cv2.imshow('Temporal Collage Result', img)
cv2.waitKey(0)
cv2.imwrite('outputs/result_vis.png', img)