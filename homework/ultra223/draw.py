import cv2
import numpy as np
def draw_lines(img, lines):
    res = np.zeros_like(img)
    if lines:
        for l in lines:
            x1,y1,x2,y2 = l.reshape(4)
            cv2.line(res,(x1,y1),(x2,y2),(0,255,0),5)
