import cv2
def blur(img):
    return cv2.GaussianBlur(img, (5,5), 0)
