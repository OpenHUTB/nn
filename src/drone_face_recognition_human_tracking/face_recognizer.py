# 人脸识别模块（DeepFace）
import os
import cv2
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier


class FaceRecognizer:
    def __init__(self, database_path='faces/', model_name='Facenet'):
        """初始化人脸识别器 - 兼容新版DeepFace"""
        print(f"🔄 正在初始化人脸识别系统 ({model_name})...")

        self.database_path = database_path
        self.model_name = model_name
        self.embeddings = {}
        self.labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        self.knn_classifier = None

        # 创建数据库文件夹
        os.makedirs(database_path, exist_ok=True)

        # 加载或训练模型
        self.load_or_train()

        print(f"✅ 人脸识别系统初始化完成，已加载 {len(set(self.labels))} 个人")

    def load_or_train(self):
        """加载已有的人脸数据库或重新训练"""
        database_file = os.path.join(self.database_path, 'face_database.pkl')

        if os.path.exists(database_file):
            # 加载已有的数据库
            try:
                with open(database_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings']
                    self.labels = data['labels']
                    self.label_to_name = data['label_to_name']

                # 重建name_to_label映射
                self.name_to_label = {name: label for label, name in self.label_to_name.items()}

                # 训练KNN分类器
                self.train_knn_classifier()
                print(f"✅ 已加载人脸数据库，包含 {len(set(self.labels))} 个人")

            except Exception as e:
                print(f"❌ 加载数据库失败: {e}")
                self.build_database_from_folders()
        else:
            # 从文件夹构建数据库
            self.build_database_from_folders()

    def build_database_from_folders(self):
        """从文件夹结构构建人脸数据库"""
        print("📂 正在从文件夹构建人脸数据库...")

        if not os.path.exists(self.database_path):
            print("⚠️  人脸数据库文件夹不存在")
            return

        # 遍历数据库文件夹
        person_count = 0
        for person_name in os.listdir(self.database_path):
            person_path = os.path.join(self.database_path, person_name)

            if os.path.isdir(person_path):
                print(f"👤 处理: {person_name}")

                # 为每个人分配标签
                if person_name not in self.name_to_label:
                    label = len(self.name_to_label)
                    self.name_to_label[person_name] = label
                    self.label_to_name[label] = person_name

                label = self.name_to_label[person_name]

                # 处理该人的所有图片
                image_count = 0
                for img_file in os.listdir(person_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(person_path, img_file)

                        try:
                            # 读取图像
                            img = cv2.imread(img_path)
                            if img is None:
                                continue

                            # 提取人脸嵌入
                            embedding = self.extract_embedding(img)
                            if embedding is not None:
                                if person_name not in self.embeddings:
                                    self.embeddings[person_name] = []
                                self.embeddings[person_name].append(embedding)
                                self.labels.append(label)
                                image_count += 1

                        except Exception as e:
                            print(f"❌ 处理图像 {img_file} 失败: {e}")

                if image_count > 0:
                    person_count += 1
                    print(f"   ✅ 添加了 {image_count} 张图片")

        if self.labels:
            # 训练KNN分类器
            self.train_knn_classifier()

            # 保存数据库
            self.save_database()

            print(f"✅ 人脸数据库构建完成，包含 {person_count} 个人，总共 {len(self.labels)} 张图片")
        else:
            print("⚠️  未找到任何人脸图像")

    def extract_embedding(self, face_img):
        """从人脸图像中提取特征嵌入 - 兼容新版DeepFace"""
        try:
            # 方法1：使用DeepFace（如果可用）
            try:
                from deepface import DeepFace

                # 新版DeepFace的调用方式
                embedding_obj = DeepFace.represent(
                    face_img,
                    model_name=self.model_name,
                    enforce_detection=False,
                    detector_backend='opencv'
                )

                if embedding_obj:
                    # 转换为numpy数组
                    embedding = np.array(embedding_obj[0]['embedding'])
                    return embedding

            except ImportError:
                print("⚠️  DeepFace未安装")
                return self.extract_embedding_simple(face_img)
            except Exception as e:
                print(f"⚠️  DeepFace提取失败: {e}")
                return self.extract_embedding_simple(face_img)

        except Exception as e:
            print(f"❌ 提取嵌入失败: {e}")
            return self.extract_embedding_simple(face_img)

    def extract_embedding_simple(self, face_img):
        """简单的人脸特征提取（备用方法）"""
        try:
            # 转换为灰度图
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img

            # 调整大小
            gray_resized = cv2.resize(gray, (128, 128))

            # 直方图均衡化
            gray_eq = cv2.equalizeHist(gray_resized)

            # 展平并归一化
            features = gray_eq.flatten().astype(np.float32) / 255.0

            # 使用PCA降维到128维（模拟Facenet）
            from sklearn.decomposition import PCA

            # 如果数据足够，训练PCA
            if len(self.labels) > 10:
                if not hasattr(self, 'pca'):
                    self.pca = PCA(n_components=128)
                    # 收集一些样本训练PCA
                    sample_data = []
                    for _ in range(min(100, len(self.labels))):
                        random_img = np.random.rand(128, 128) * 255
                        random_img = random_img.astype(np.uint8)
                        sample_data.append(random_img.flatten())

                    if sample_data:
                        self.pca.fit(np.array(sample_data))

                features = self.pca.transform(features.reshape(1, -1)).flatten()
            else:
                # 使用简单的特征（前128个像素）
                features = features[:128]

            return features

        except Exception as e:
            print(f"❌ 简单特征提取失败: {e}")
            # 返回随机特征（仅用于测试）
            return np.random.randn(128).astype(np.float32)

    def train_knn_classifier(self):
        """训练KNN分类器"""
        if not self.labels:
            return

        # 准备训练数据
        X = []
        y = []

        for person_name, embeddings_list in self.embeddings.items():
            for embedding in embeddings_list:
                X.append(embedding)
                label = self.name_to_label[person_name]
                y.append(label)

        if X and y:
            X = np.array(X)
            y = np.array(y)

            # 创建并训练KNN分类器
            n_neighbors = min(3, len(set(y)))
            self.knn_classifier = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric='euclidean'
            )
            self.knn_classifier.fit(X, y)
            print(f"✅ KNN分类器训练完成，使用 {n_neighbors} 个邻居")

    def recognize(self, face_img):
        """识别人脸"""
        if self.knn_classifier is None or not self.label_to_name:
            return "Unknown", 0.0

        try:
            # 提取特征嵌入
            embedding = self.extract_embedding(face_img)
            if embedding is None:
                return "Unknown", 0.0

            # 使用KNN进行识别
            embedding_reshaped = embedding.reshape(1, -1)

            # 获取最近邻居
            distances, indices = self.knn_classifier.kneighbors(
                embedding_reshaped,
                n_neighbors=min(3, len(self.knn.classes_))
            )

            # 获取预测标签和概率
            predicted_label = self.knn_classifier.predict(embedding_reshaped)[0]
            predicted_proba = self.knn_classifier.predict_proba(embedding_reshaped)

            # 获取置信度
            if len(self.knn.classes_) > 0:
                label_index = list(self.knn.classes_).index(predicted_label)
                confidence = predicted_proba[0][label_index]
            else:
                confidence = 0.0

            # 计算平均距离
            avg_distance = np.mean(distances[0])

            # 设置阈值（可以根据实际情况调整）
            if avg_distance < 0.8 and confidence > 0.6:  # 调整阈值
                person_name = self.label_to_name.get(predicted_label, "Unknown")
                return person_name, confidence
            else:
                return "Unknown", confidence

        except Exception as e:
            print(f"❌ 人脸识别失败: {e}")
            return "Unknown", 0.0

    def add_face(self, face_img, person_name):
        """添加新的人脸到数据库"""
        try:
            # 提取嵌入
            embedding = self.extract_embedding(face_img)
            if embedding is None:
                print("❌ 无法提取人脸特征")
                return False

            # 添加到数据库
            if person_name not in self.embeddings:
                self.embeddings[person_name] = []

                # 为新的人分配标签
                if person_name not in self.name_to_label:
                    new_label = len(self.name_to_label)
                    self.name_to_label[person_name] = new_label
                    self.label_to_name[new_label] = person_name

            # 找到对应的标签
            label = self.name_to_label[person_name]

            # 添加嵌入和标签
            self.embeddings[person_name].append(embedding)
            self.labels.append(label)

            # 重新训练分类器
            self.train_knn_classifier()

            # 保存数据库
            self.save_database()

            # 保存人脸图像到文件夹
            person_folder = os.path.join(self.database_path, person_name)
            os.makedirs(person_folder, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_path = os.path.join(person_folder, f"{timestamp}.jpg")
            cv2.imwrite(face_path, face_img)

            print(f"✅ 已添加 {person_name} 的人脸到数据库")
            return True

        except Exception as e:
            print(f"❌ 添加人脸失败: {e}")
            return False

    def save_database(self):
        """保存人脸数据库"""
        try:
            database_file = os.path.join(self.database_path, 'face_database.pkl')

            data = {
                'embeddings': self.embeddings,
                'labels': self.labels,
                'label_to_name': self.label_to_name
            }

            with open(database_file, 'wb') as f:
                pickle.dump(data, f)

            print("✅ 人脸数据库已保存")

        except Exception as e:
            print(f"❌ 保存数据库失败: {e}")

    def list_registered_persons(self):
        """列出所有已注册的人员"""
        if not self.label_to_name:
            print("📋 数据库为空")
            return []

        persons = []
        print("📋 已注册的人员列表:")
        for label, name in sorted(self.label_to_name.items()):
            image_count = len(self.embeddings.get(name, []))
            print(f"  {label}: {name} ({image_count} 张图片)")
            persons.append((name, image_count))

        return persons

    def verify_face(self, face_img, person_name):
        """验证人脸是否属于指定人员"""
        if person_name not in self.name_to_label:
            return False, 0.0

        predicted_name, confidence = self.recognize(face_img)

        if predicted_name == person_name and confidence > 0.7:
            return True, confidence
        else:
            return False, confidence


# 测试函数
def test_face_recognizer():
    """测试人脸识别器"""
    print("🧪 测试人脸识别模块")
    print("-" * 40)

    # 创建识别器
    print("🔄 初始化人脸识别器...")
    recognizer = FaceRecognizer(database_path='faces_test')

    # 创建测试人脸图像
    print("📸 创建测试图像...")

    # 创建第一个人脸
    face1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(face1, (50, 50), 40, (200, 200, 200), -1)
    cv2.circle(face1, (35, 40), 8, (0, 0, 0), -1)
    cv2.circle(face1, (65, 40), 8, (0, 0, 0), -1)
    cv2.ellipse(face1, (50, 65), (25, 15), 0, 0, 180, (0, 0, 0), 3)

    # 创建第二个人脸
    face2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(face2, (50, 50), 40, (150, 150, 150), -1)
    cv2.circle(face2, (35, 40), 8, (0, 0, 0), -1)
    cv2.circle(face2, (65, 40), 8, (0, 0, 0), -1)
    cv2.ellipse(face2, (50, 70), (20, 10), 0, 0, 180, (0, 0, 0), 3)

    # 添加第一个人脸
    print("\n👤 添加第一个人脸: Alice")
    if recognizer.add_face(face1, "Alice"):
        print("✅ Alice添加成功")

    # 添加第二个人脸
    print("\n👤 添加第二个人脸: Bob")
    if recognizer.add_face(face2, "Bob"):
        print("✅ Bob添加成功")

    # 列出已注册人员
    print("\n📋 已注册人员:")
    recognizer.list_registered_persons()

    # 测试识别
    print("\n🔍 测试人脸识别...")

    # 测试识别Alice
    name1, conf1 = recognizer.recognize(face1)
    print(f"   识别Alice: {name1} (置信度: {conf1:.2f})")

    # 测试识别Bob
    name2, conf2 = recognizer.recognize(face2)
    print(f"   识别Bob: {name2} (置信度: {conf2:.2f})")

    # 创建新人脸测试
    print("\n👤 测试未知人脸...")
    unknown_face = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(unknown_face, (50, 50), 40, (100, 100, 100), -1)
    cv2.circle(unknown_face, (30, 45), 8, (0, 0, 0), -1)
    cv2.circle(unknown_face, (70, 45), 8, (0, 0, 0), -1)

    name3, conf3 = recognizer.recognize(unknown_face)
    print(f"   识别未知人脸: {name3} (置信度: {conf3:.2f})")

    # 验证测试
    print("\n🔐 人脸验证测试...")
    is_alice, alice_conf = recognizer.verify_face(face1, "Alice")
    print(f"   验证Alice: {'通过' if is_alice else '失败'} (置信度: {alice_conf:.2f})")

    is_bob, bob_conf = recognizer.verify_face(face2, "Bob")
    print(f"   验证Bob: {'通过' if is_bob else '失败'} (置信度: {bob_conf:.2f})")

    # 显示图像
    print("\n🖼️ 显示测试图像...")
    cv2.imshow('Alice', face1)
    cv2.imshow('Bob', face2)
    cv2.imshow('Unknown', unknown_face)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\n✅ 人脸识别测试完成!")
    return recognizer


if __name__ == "__main__":
    # 检查是否需要安装依赖
    try:
        import deepface

        print(f"✅ DeepFace版本: {deepface.__version__}")
    except ImportError:
        print("⚠️  DeepFace未安装，使用简单模式")
        print("💡 如需完整功能，请运行: pip install deepface")

    try:
        import sklearn

        print(f"✅ scikit-learn版本: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn未安装")
        print("💡 请运行: pip install scikit-learn")
        exit(1)

    # 运行测试
    test_face_recognizer()