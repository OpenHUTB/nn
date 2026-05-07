import cv2
def draw_path(img, x):
    cv2.circle(img,(x,350),15,(0,0,255),-1)
    return img
