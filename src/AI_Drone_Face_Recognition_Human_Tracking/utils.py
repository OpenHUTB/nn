'''工具模块'''
import cv2
import numpy as np

def preprocess_face(face_img):
    """人脸预处理：灰度化+均衡化+归一化"""
    # 灰度化
    if len(face_img.shape) == 3:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_img
    # 直方图均衡化
    face_eq = cv2.equalizeHist(face_gray)
    # 归一化尺寸
    face_resized = cv2.resize(face_eq, (100, 100))
    # 归一化像素值
    face_normalized = face_resized / 255.0
    return face_normalized

def calculate_similarity(face1, face2):
    """计算人脸相似度（余弦相似度）"""
    try:
        # 展平为一维向量
        vec1 = face1.flatten()
        vec2 = face2.flatten()
        # 余弦相似度
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        sim = dot / (norm1 * norm2) if (norm1 * norm2) != 0 else 0.0
        return sim
    except Exception as e:
        print(f"计算相似度失败：{e}")
        return 0.0