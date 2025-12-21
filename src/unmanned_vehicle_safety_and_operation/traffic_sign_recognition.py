import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# ===================== 核心配置（适配PyCharm工作目录） =====================
# 获取当前脚本所在目录（PyCharm中确保路径正确）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "traffic_signs")  # 数据集路径
TEST_IMG_PATH = os.path.join(BASE_DIR, "test_sign.png")  # 测试图片路径
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "confusion_matrix.png")  # 混淆矩阵保存路径


# ===================== 1. 自动生成模拟数据集（PyCharm友好） =====================
def create_simulated_dataset():
    """
    自动生成模拟交通标志数据集
    路径：当前脚本目录/traffic_signs/
    """
    # 创建数据集根目录
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"创建数据集目录：{DATA_PATH}")

    # 定义3类交通标志
    categories = ["stop_sign", "speed_limit_50", "yield_sign"]
    n_samples_per_class = 50  # 每类生成50张图片

    # 为每个类别生成图片
    for cat in categories:
        cat_dir = os.path.join(DATA_PATH, cat)
        if not os.path.exists(cat_dir):
            os.makedirs(cat_dir)

        # 生成不同特征的模拟灰度图（区分不同类别）
        for i in range(n_samples_per_class):
            # 不同类别设置不同的像素分布（便于模型区分）
            if cat == "stop_sign":
                img_arr = np.random.normal(0.8, 0.08, (64, 64))  # 偏亮
            elif cat == "speed_limit_50":
                img_arr = np.random.normal(0.4, 0.08, (64, 64))  # 中等亮度
            else:  # yield_sign
                img_arr = np.random.normal(0.2, 0.08, (64, 64))  # 偏暗

            # 归一化到0-255并转为uint8格式
            img_arr = np.clip(img_arr * 255, 0, 255).astype(np.uint8)
            # 保存图片（PNG格式，兼容PIL）
            img = Image.fromarray(img_arr, mode='L')
            img.save(os.path.join(cat_dir, f"{cat}_{i}.png"))

    print(f"\n✅ 模拟数据集生成完成！")
    print(f"📂 数据集路径：{DATA_PATH}")
    print(f"📊 包含类别：{categories}（每类{n_samples_per_class}张）")


# ===================== 2. 纯Numpy实现HOG特征提取 =====================
def hog_feature_extract(img):
    """
    输入：64x64归一化灰度图（0-1）
    输出：HOG特征向量
    """
    # 1. 计算x/y方向梯度
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]

    # 2. 计算梯度幅值和方向（0-180度）
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

    # 3. 分Cell计算梯度直方图（8x8像素/Cell，9个方向）
    cell_size = 8
    orientations = 9
    orient_bin = 180 / orientations
    n_cells = 64 // cell_size  # 8个Cell（64/8）

    cell_hist = np.zeros((n_cells, n_cells, orientations), dtype=np.float32)
    for y in range(n_cells):
        for x in range(n_cells):
            # 提取当前Cell的梯度
            cell_mag = magnitude[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size]
            cell_orient = orientation[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size]

            # 统计每个方向的梯度和
            for bin_idx in range(orientations):
                bin_min = bin_idx * orient_bin
                bin_max = (bin_idx + 1) * orient_bin
                mask = (cell_orient >= bin_min) & (cell_orient < bin_max)
                cell_hist[y, x, bin_idx] = np.sum(cell_mag[mask])

    # 4. 分Block归一化（2x2 Cell/Block，L2-Hys归一化）
    block_size = 2
    n_blocks = n_cells - block_size + 1  # 7个Block
    hog_feat = []

    for y in range(n_blocks):
        for x in range(n_blocks):
            block = cell_hist[y:y + block_size, x:x + block_size, :].flatten()
            # L2归一化（加小值避免除零）
            norm = np.sqrt(np.sum(block ** 2) + 1e-6)
            block = block / norm
            # Hys截断（限制最大值0.2）
            block = np.clip(block, 0, 0.2)
            # 再次归一化
            norm = np.sqrt(np.sum(block ** 2) + 1e-6)
            block = block / norm
            hog_feat.extend(block)

    return np.array(hog_feat, dtype=np.float32)


# ===================== 3. 简化版SVM分类器（多分类） =====================
class SimpleSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=800):
        self.lr = lr  # 学习率
        self.lambda_param = lambda_param  # 正则化系数
        self.n_iters = n_iters  # 迭代次数
        self.weights = None  # 类别权重
        self.biases = None  # 类别偏置
        self.classes = None  # 类别列表

    def fit(self, X, y):
        """训练多分类SVM（一对其余策略）"""
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples, n_feat = X.shape

        # 初始化权重和偏置
        self.weights = np.zeros((n_classes, n_feat))
        self.biases = np.zeros(n_classes)

        print("\n🚀 开始训练SVM模型...")
        # 为每个类别训练二分类SVM
        for idx, c in enumerate(self.classes):
            # 构建二分类标签（当前类=1，其他类=-1）
            y_bin = np.where(y == c, 1, -1)
            w = np.zeros(n_feat)
            b = 0

            # 梯度下降优化
            for iter in range(self.n_iters):
                if iter % 200 == 0:
                    print(f"  类别{c}：迭代{iter}/{self.n_iters}")

                for i in range(n_samples):
                    z = np.dot(X[i], w) + b
                    if y_bin[i] * z < 1:
                        # 误分类样本：更新权重和偏置
                        w -= self.lr * (2 * self.lambda_param * w - y_bin[i] * X[i])
                        b -= self.lr * (-y_bin[i])
                    else:
                        # 正确分类样本：仅正则化
                        w -= self.lr * 2 * self.lambda_param * w

            self.weights[idx] = w
            self.biases[idx] = b

        print("✅ SVM模型训练完成！")

    def predict(self, X):
        """预测类别（返回类别索引）"""
        pred = []
        for x in X:
            # 计算每个类别的得分
            scores = [np.dot(x, self.weights[idx]) + self.biases[idx] for idx in range(len(self.classes))]
            pred.append(self.classes[np.argmax(scores)])
        return np.array(pred)

    def predict_proba(self, X):
        """预测概率（Softmax转换）"""
        probs = []
        for x in X:
            scores = [np.dot(x, self.weights[idx]) + self.biases[idx] for idx in range(len(self.classes))]
            # Softmax归一化（防止数值溢出）
            exp_scores = np.exp(scores - np.max(scores))
            prob = exp_scores / np.sum(exp_scores)
            probs.append(prob)
        return np.array(probs)


# ===================== 4. 模型评估指标（纯手动实现） =====================
class Metrics:
    @staticmethod
    def train_test_split(X, y, test_size=0.2, seed=42):
        """划分训练集/测试集"""
        np.random.seed(seed)
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]

        split_idx = int(len(X) * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    @staticmethod
    def accuracy(y_true, y_pred):
        """计算准确率"""
        return np.sum(y_true == y_pred) / len(y_true)

    @staticmethod
    def confusion_matrix(y_true, y_pred, classes):
        """计算混淆矩阵"""
        n_classes = len(classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        cat2idx = {c: i for i, c in enumerate(classes)}

        for t, p in zip(y_true, y_pred):
            cm[cat2idx[t], cat2idx[p]] += 1
        return cm

    @staticmethod
    def classification_report(y_true, y_pred, classes):
        """生成分类报告（精确率/召回率/F1）"""
        report = []
        report.append("=" * 50)
        report.append("            分类报告（精确率/召回率/F1）")
        report.append("=" * 50)
        report.append(f"{'类别':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'样本数':<8}")
        report.append("-" * 50)

        total_support = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0

        for c in classes:
            # 计算TP/FN/FP
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            fp = np.sum((y_true != c) & (y_pred == c))

            support = tp + fn
            total_support += support

            # 计算指标（避免除零）
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            avg_precision += precision * support
            avg_recall += recall * support
            avg_f1 += f1 * support

            report.append(f"{c:<12} {precision:.2f}      {recall:.2f}      {f1:.2f}      {support}")

        # 加权平均
        avg_precision /= total_support
        avg_recall /= total_support
        avg_f1 /= total_support

        report.append("-" * 50)
        report.append(
            f"{'加权平均':<12} {avg_precision:.2f}      {avg_recall:.2f}      {avg_f1:.2f}      {total_support}")
        report.append("=" * 50)

        return "\n".join(report)


# ===================== 5. 交通标志识别主系统 =====================
class TrafficSignDetector:
    def __init__(self):
        self.X = []  # 特征集
        self.y = []  # 标签集
        self.categories = []  # 类别列表
        self.svm = SimpleSVM()
        self.mean = None  # 特征均值（标准化）
        self.std = None  # 特征标准差（标准化）

    def load_data(self):
        """加载数据集并提取HOG特征"""
        print("\n📥 开始加载数据集...")
        # 获取类别列表
        self.categories = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]

        # 遍历所有图片
        for label, cat in enumerate(self.categories):
            cat_dir = os.path.join(DATA_PATH, cat)
            img_files = [f for f in os.listdir(cat_dir) if f.endswith(('.png', '.jpg'))]

            for img_file in img_files:
                img_path = os.path.join(cat_dir, img_file)
                # 预处理图片
                img = self._preprocess(img_path)
                if img is None:
                    continue
                # 提取HOG特征
                hog_feat = hog_feature_extract(img)
                self.X.append(hog_feat)
                self.y.append(cat)  # 直接存储类别名称（更直观）

        # 转换为Numpy数组
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        # 划分训练/测试集
        X_train, X_test, y_train, y_test = Metrics.train_test_split(self.X, self.y)

        # 特征标准化
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0) + 1e-6
        X_train = (X_train - self.mean) / self.std
        X_test = (X_test - self.mean) / self.std

        print(f"✅ 数据集加载完成！")
        print(f"📊 总样本数：{len(self.X)} | 训练集：{len(X_train)} | 测试集：{len(X_test)}")
        return X_train, X_test, y_train, y_test

    def _preprocess(self, img_path):
        """图片预处理：灰度化→缩放→归一化"""
        try:
            with Image.open(img_path) as img:
                # 灰度化+缩放至64x64
                img_gray = img.convert('L')
                img_resized = img_gray.resize((64, 64), Image.Resampling.LANCZOS)
                # 归一化到0-1
                img_arr = np.array(img_resized, dtype=np.float32) / 255.0
                return img_arr
        except Exception as e:
            print(f"⚠️ 图片预处理失败 {img_path}：{e}")
            return None

    def evaluate(self, X_test, y_test):
        """评估模型并生成检测报告"""
        # 预测测试集
        y_pred = self.svm.predict(X_test)
        y_pred_proba = self.svm.predict_proba(X_test)

        # 1. 整体准确率
        acc = Metrics.accuracy(y_test, y_pred)
        print(f"\n📈 整体识别准确率：{acc:.4f} ({acc * 100:.2f}%)")

        # 2. 分类报告
        report = Metrics.classification_report(y_test, y_pred, self.categories)
        print(report)

        # 3. 混淆矩阵（可视化+保存）
        cm = Metrics.confusion_matrix(y_test, y_pred, self.categories)
        self._plot_confusion_matrix(cm)

        # 4. 各类别准确率
        print("\n📋 各类别识别准确率：")
        for i, cat in enumerate(self.categories):
            total = cm[i].sum()
            correct = cm[i, i]
            acc_cat = correct / total if total > 0 else 0
            print(f"  {cat}：{acc_cat:.4f} ({correct}/{total})")

        # 5. 低置信度检测
        low_conf_idx = np.where(np.max(y_pred_proba, axis=1) < 0.8)[0]
        print(f"\n⚠️ 低置信度预测（概率<0.8）：{len(low_conf_idx)} 个样本")
        if len(low_conf_idx) > 0:
            for idx in low_conf_idx[:3]:  # 展示前3个
                true_cat = y_test[idx]
                pred_cat = y_pred[idx]
                conf = np.max(y_pred_proba[idx])
                print(f"  样本{idx}：真实={true_cat} | 预测={pred_cat} | 置信度={conf:.4f}")

    def _plot_confusion_matrix(self, cm):
        """绘制并保存混淆矩阵"""
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
        plt.figure(figsize=(8, 6))
        # 绘制混淆矩阵
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('交通标志识别 - 混淆矩阵', fontsize=12)
        plt.colorbar()
        # 设置坐标轴
        tick_marks = np.arange(len(self.categories))
        plt.xticks(tick_marks, self.categories, rotation=45)
        plt.yticks(tick_marks, self.categories)
        # 标注数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        # 标签
        plt.ylabel('真实类别', fontsize=10)
        plt.xlabel('预测类别', fontsize=10)
        plt.tight_layout()
        # 保存图片（PyCharm中可直接查看）
        plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches='tight')
        print(f"\n🖼️ 混淆矩阵已保存：{CONFUSION_MATRIX_PATH}")
        # 显示图片（PyCharm会弹出窗口）
        plt.show()

    def predict_single(self):
        """预测单张测试图片"""
        # 生成测试图片（模拟stop_sign）
        test_img_arr = np.random.normal(0.8, 0.08, (64, 64))
        test_img_arr = np.clip(test_img_arr * 255, 0, 255).astype(np.uint8)
        test_img = Image.fromarray(test_img_arr, mode='L')
        test_img.save(TEST_IMG_PATH)
        print(f"\n📸 生成测试图片：{TEST_IMG_PATH}")

        # 预处理+提取特征
        img = self._preprocess(TEST_IMG_PATH)
        hog_feat = hog_feature_extract(img)
        hog_feat = (hog_feat - self.mean) / self.std

        # 预测
        pred_cat = self.svm.predict(np.array([hog_feat]))[0]
        pred_proba = self.svm.predict_proba(np.array([hog_feat]))[0]
        conf = pred_proba[self.svm.classes.tolist().index(pred_cat)]

        # 输出结果
        print("\n🎯 单张图片识别结果：")
        print(f"  预测类别：{pred_cat}")
        print(f"  置信度：{conf:.4f}")
        print(f"  是否高置信度：{'是' if conf >= 0.8 else '否'}")
        print("  各类别概率：")
        for i, cat in enumerate(self.categories):
            print(f"    {cat}：{pred_proba[i]:.4f}")


# ===================== 6. 主运行函数（PyCharm一键运行） =====================
def main():
    """PyCharm主运行入口"""
    # 步骤1：生成模拟数据集
    create_simulated_dataset()

    # 步骤2：初始化检测器
    detector = TrafficSignDetector()

    # 步骤3：加载数据
    X_train, X_test, y_train, y_test = detector.load_data()

    # 步骤4：训练模型
    detector.svm.fit(X_train, y_train)

    # 步骤5：评估模型（生成混淆矩阵图片）
    detector.evaluate(X_test, y_test)

    # 步骤6：单张图片预测（生成测试图片）
    detector.predict_single()

    print("\n🎉 所有任务完成！生成的文件：")
    print(f"  1. 混淆矩阵：{CONFUSION_MATRIX_PATH}")
    print(f"  2. 测试图片：{TEST_IMG_PATH}")
    print(f"  3. 数据集：{DATA_PATH}")


# ===================== 运行入口 =====================
if __name__ == "__main__":
    # PyCharm中直接运行此脚本
    main()