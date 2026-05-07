import cv2
import numpy as np
def detect_lines(img):
    return cv2.HoughLinesP(img,2,np.pi/180,100,minLineLength=40,maxLineGap=5)
