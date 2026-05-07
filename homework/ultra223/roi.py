import cv2
import numpy as np
def roi(img):
    h = img.shape[0]
    poly = np.array([[(200,h),(1100,h),(550,250)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(img, mask)
