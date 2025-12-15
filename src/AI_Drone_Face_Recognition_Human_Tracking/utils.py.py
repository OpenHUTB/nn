import os
import cv2
from deepface import DeepFace
import numpy as np


class FaceDB:
    """人脸数据库管理类"""

    def __init__(self, face_dir="faces"):
        self.face_dir = face_dir
        self.face_data = {}  # 存储{人名: 人脸特征}
        self._load_face_database()

    def _load_face_database(self):
        """加载人脸库中的所有人脸特征"""
        if not os.path.exists(self.face_dir):
            os.makedirs(self.face_dir)
            print(f"创建人脸库目录: {self.face_dir}")

        self.face_data.clear()  # 清空旧数据
        for img_name in os.listdir(self.face_dir):
            if img_name.endswith((".jpg", ".png")):
                person_name = os.path.splitext(img_name)[0]
                img_path = os.path.join(self.face_dir, img_name)
                try:
                    # 提取人脸特征（Facenet模型）
                    embedding = DeepFace.represent(
                        img_path,
                        model_name="Facenet",
                        enforce_detection=True
                    )[0]["embedding"]
                    self.face_data[person_name] = embedding
                    print(f"✅ 成功加载人脸: {person_name}")
                except Exception as e:
                    print(f"❌ 加载{img_name}失败: {str(e)[:50]}...")

    def recognize_face(self, frame, threshold=0.6):
        """识别人脸，返回匹配的人名或None"""
        try:
            # 提取当前帧人脸特征
            frame_embedding = DeepFace.represent(
                frame,
                model_name="Facenet",
                enforce_detection=True
            )[0]["embedding"]

            # 对比人脸库（计算欧式距离）
            min_distance = float("inf")
            matched_name = None
            for name, embedding in self.face_data.items():
                distance = np.linalg.norm(np.array(frame_embedding) - np.array(embedding))
                if distance < min_distance and distance < threshold:
                    min_distance = distance
                    matched_name = name
            return matched_name, min_distance  # 新增返回距离，方便调试
        except Exception as e:
            print(f"❌ 人脸识别失败: {str(e)[:50]}...")
            return None, float("inf")

    def add_face(self, frame, person_name):
        """添加人脸到数据库"""
        # 先检测人脸，确保有效
        try:
            # 验证帧中有人脸
            DeepFace.extract_faces(frame, enforce_detection=True)
            # 保存人脸图片
            img_path = os.path.join(self.face_dir, f"{person_name}.jpg")
            cv2.imwrite(img_path, frame)
            # 重新加载数据库
            self._load_face_database()
            print(f"✅ 已添加[{person_name}]到人脸库，路径: {img_path}")
            return True
        except Exception as e:
            print(f"❌ 添加人脸失败: {str(e)[:50]}...")
            return False

    def list_faces(self):
        """列出人脸库中所有已注册的人名"""
        return list(self.face_data.keys())


def draw_detection_box(frame, bbox, label, color=(0, 255, 0)):
    """绘制检测框和标签"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, label, (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )
    return frame


# ===================== 独立运行测试逻辑 =====================
if __name__ == "__main__":
    # 初始化人脸库
    face_db = FaceDB(face_dir="faces")
    print(f"\n当前人脸库已注册: {face_db.list_faces()}\n")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查设备！")
        exit(1)

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("=" * 50)
    print("🎯 FaceDB 测试工具（独立运行模式）")
    print("按键说明：")
    print("  a → 添加当前帧人脸到数据库")
    print("  r → 识别当前帧人脸")
    print("  l → 列出已注册人脸")
    print("  q → 退出程序")
    print("=" * 50)

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面！")
            break

        # 显示实时画面
        cv2.imshow("FaceDB Test (utils.py)", frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🔚 退出测试程序...")
            break

        elif key == ord('a'):
            # 添加人脸
            person_name = input("\n请输入要添加的人名: ").strip()
            if not person_name:
                print("❌ 人名不能为空！")
                continue
            face_db.add_face(frame, person_name)

        elif key == ord('r'):
            # 识别人脸
            print("\n🔍 正在识别人脸...")
            name, distance = face_db.recognize_face(frame)
            if name:
                print(f"✅ 识别成功！匹配到: {name} (距离: {distance:.4f})")
            else:
                print(f"❌ 未识别到匹配人脸（最小距离: {distance:.4f}）")

        elif key == ord('l'):
            # 列出已注册人脸
            faces = face_db.list_faces()
            print(f"\n📋 人脸库已注册名单: {faces if faces else '空'}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ 资源已释放，程序结束")