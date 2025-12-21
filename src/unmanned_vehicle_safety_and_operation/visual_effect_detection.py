import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from PIL import Image, ImageFilter, ImageStat

warnings.filterwarnings('ignore')

# 设置中文字体（解决PyCharm中matplotlib中文显示乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ===================== 1. 特征提取函数（保留，兼容真实图像） =====================
class RainFogFeatureExtractor:
    def __init__(self):
        pass

    def calculate_fog_density(self, img):
        gray = img.convert('L')
        min_filtered = gray.filter(ImageFilter.MinFilter(size=15))
        gray_array = np.array(min_filtered)
        fog_density = np.mean(gray_array) / 255.0
        return fog_density

    def calculate_contrast(self, img):
        gray = img.convert('L')
        stat = ImageStat.Stat(gray)
        contrast = stat.stddev[0]
        return contrast

    def calculate_edge_density(self, img):
        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges)
        edge_density = np.sum(edge_array > 0) / (img.width * img.height)
        return edge_density

    def calculate_color_saturation(self, img):
        hsv = img.convert('HSV')
        sat_band = hsv.getchannel(1)
        sat_array = np.array(sat_band)
        sat_mean = np.mean(sat_array) / 255.0
        return sat_mean

    def extract_all_features(self, img_path):
        try:
            with Image.open(img_path) as img:
                img = img.resize((640, 480))
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                features = [
                    self.calculate_fog_density(img),
                    self.calculate_contrast(img),
                    self.calculate_edge_density(img),
                    self.calculate_color_saturation(img)
                ]
                features = np.array(features)
                features[1] = features[1] / 255.0
                return features
        except Exception as e:
            return None


# ===================== 2. 生成模拟数据集 =====================
def build_dataset():
    """生成模拟数据集（无需真实图像）"""
    np.random.seed(42)
    n_samples_per_class = 100  # 每类100个样本

    # 清晰样本（标签0）
    clear_fog = np.random.normal(0.1, 0.05, n_samples_per_class)
    clear_contrast = np.random.normal(0.8, 0.1, n_samples_per_class)
    clear_edge = np.random.normal(0.2, 0.05, n_samples_per_class)
    clear_sat = np.random.normal(0.8, 0.1, n_samples_per_class)
    clear_features = np.column_stack([clear_fog, clear_contrast, clear_edge, clear_sat])
    clear_labels = np.zeros(n_samples_per_class)

    # 轻度雨雾（标签1）
    light_fog = np.random.normal(0.3, 0.05, n_samples_per_class)
    light_contrast = np.random.normal(0.6, 0.1, n_samples_per_class)
    light_edge = np.random.normal(0.15, 0.05, n_samples_per_class)
    light_sat = np.random.normal(0.6, 0.1, n_samples_per_class)
    light_features = np.column_stack([light_fog, light_contrast, light_edge, light_sat])
    light_labels = np.ones(n_samples_per_class)

    # 中度雨雾（标签2）
    medium_fog = np.random.normal(0.5, 0.05, n_samples_per_class)
    medium_contrast = np.random.normal(0.4, 0.1, n_samples_per_class)
    medium_edge = np.random.normal(0.1, 0.05, n_samples_per_class)
    medium_sat = np.random.normal(0.4, 0.1, n_samples_per_class)
    medium_features = np.column_stack([medium_fog, medium_contrast, medium_edge, medium_sat])
    medium_labels = np.ones(n_samples_per_class) * 2

    # 重度雨雾（标签3）
    heavy_fog = np.random.normal(0.8, 0.05, n_samples_per_class)
    heavy_contrast = np.random.normal(0.2, 0.1, n_samples_per_class)
    heavy_edge = np.random.normal(0.05, 0.05, n_samples_per_class)
    heavy_sat = np.random.normal(0.2, 0.1, n_samples_per_class)
    heavy_features = np.column_stack([heavy_fog, heavy_contrast, heavy_edge, heavy_sat])
    heavy_labels = np.ones(n_samples_per_class) * 3

    # 合并数据并限制范围
    X = np.vstack([clear_features, light_features, medium_features, heavy_features])
    y = np.hstack([clear_labels, light_labels, medium_labels, heavy_labels])
    X = np.clip(X, 0, 1)
    feature_names = ['雾度', '对比度', '边缘密度', '饱和度']

    print(f"模拟数据集生成完成：总样本数={len(X)}, 特征数={X.shape[1]}")
    return X, y, feature_names


# ===================== 3. 简易K近邻分类器 =====================
class SimpleKNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            pred_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(pred_label)
        return np.array(predictions)

    def predict_proba(self, X):
        probas = []
        for x in X:
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for label in k_nearest_labels:
                label_counts[label] += 1
            total = sum(label_counts.values())
            proba = [label_counts[i] / total for i in range(4)]
            probas.append(proba)
        return np.array(probas)


# ===================== 4. 模型评估与可视化（核心：生成图片） =====================
def evaluate_and_visualize(model, X_train, X_test, y_train, y_test, feature_names):
    """评估模型并生成可视化图片"""
    # 1. 计算训练/测试准确率
    train_pred = model.predict(X_train)
    train_acc = np.sum(train_pred == y_train) / len(y_train)
    test_pred = model.predict(X_test)
    test_acc = np.sum(test_pred == y_test) / len(y_test)

    # 2. 生成图1：训练/测试准确率对比图
    plt.figure(figsize=(8, 5))
    plt.bar(['训练集准确率', '测试集准确率'], [train_acc, test_acc], color=['#2E86AB', '#A23B72'])
    plt.ylim(0, 1.1)
    plt.title('无人车雨雾检测模型准确率', fontsize=14)
    plt.ylabel('准确率', fontsize=12)
    # 添加数值标签
    plt.text(0, train_acc + 0.02, f'{train_acc:.4f}', ha='center', fontsize=12)
    plt.text(1, test_acc + 0.02, f'{test_acc:.4f}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('./rainfog_acc.png', dpi=150, bbox_inches='tight')  # 保存图片
    print("✅ 准确率对比图已保存：rainfog_acc.png")

    # 3. 生成图2：特征分布散点图（雾度 vs 对比度）
    plt.figure(figsize=(10, 8))
    colors = ['#F18F01', '#C73E1D', '#8B0000', '#000000']  # 清晰/轻度/中度/重度颜色
    labels = ['清晰', '轻度雨雾', '中度雨雾', '重度雨雾']
    for i in range(4):
        mask = y_train == i
        plt.scatter(X_train[mask, 0], X_train[mask, 1], c=colors[i], label=labels[i], alpha=0.7)
    plt.xlabel('雾度', fontsize=12)
    plt.ylabel('对比度', fontsize=12)
    plt.title('雨雾天图像特征分布（雾度 vs 对比度）', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('./rainfog_feature_dist.png', dpi=150, bbox_inches='tight')
    print("✅ 特征分布图已保存：rainfog_feature_dist.png")

    # 4. 生成图3：各分类准确率详情
    plt.figure(figsize=(10, 6))
    class_acc = []
    for label in range(4):
        mask = y_test == label
        if np.sum(mask) == 0:
            class_acc.append(0)
            continue
        acc = np.sum((test_pred == label) & mask) / np.sum(mask)
        class_acc.append(acc)

    plt.bar(labels, class_acc, color=['#F18F01', '#C73E1D', '#8B0000', '#000000'])
    plt.ylim(0, 1.1)
    plt.title('各雨雾等级检测准确率', fontsize=14)
    plt.ylabel('准确率', fontsize=12)
    # 添加数值标签
    for i, acc in enumerate(class_acc):
        plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('./rainfog_class_acc.png', dpi=150, bbox_inches='tight')
    print("✅ 分类准确率图已保存：rainfog_class_acc.png")

    # 输出评估结果
    print("\n===== 模型评估结果 =====")
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print("\n各等级检测准确率：")
    for i, label in enumerate(labels):
        print(f"  {label}: {class_acc[i]:.4f}")

    return train_acc, test_acc


# ===================== 5. 模拟单张图像预测 =====================
def predict_simulated_image(feature_type, model):
    """模拟单张图像预测"""
    np.random.seed(42)
    feature_map = {
        'clear': np.array([0.12, 0.78, 0.21, 0.82]),
        'light': np.array([0.28, 0.62, 0.16, 0.61]),
        'medium': np.array([0.52, 0.41, 0.09, 0.39]),
        'heavy': np.array([0.79, 0.22, 0.04, 0.18])
    }
    features = feature_map.get(feature_type, feature_map['heavy'])

    X = features.reshape(1, -1)
    pred_label = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]

    label_map = {0: '清晰', 1: '轻度雨雾', 2: '中度雨雾', 3: '重度雨雾'}
    result = {
        '预测等级': label_map[pred_label],
        '置信度': {
            '清晰': f"{pred_proba[0]:.4f}",
            '轻度雨雾': f"{pred_proba[1]:.4f}",
            '中度雨雾': f"{pred_proba[2]:.4f}",
            '重度雨雾': f"{pred_proba[3]:.4f}"
        },
        '原始特征': {
            '雾度': features[0],
            '对比度': features[1],
            '边缘密度': features[2],
            '饱和度': features[3]
        }
    }
    return result


# ===================== 主函数（PyCharm入口） =====================
if __name__ == "__main__":
    # 创建保存图片的目录（如果不存在）
    if not os.path.exists('./'):
        os.makedirs('./')

    try:
        print("===== 无人车雨雾天视觉效果检测程序（PyCharm版） =====")

        # 1. 生成模拟数据集
        print("\n【1/4】生成模拟数据集...")
        X, y, feature_names = build_dataset()

        # 2. 划分训练/测试集
        print("\n【2/4】划分训练/测试集...")
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        print(f"训练集：{len(X_train)} 样本，测试集：{len(X_test)} 样本")

        # 3. 训练KNN模型11
        print("\n【3/4】训练雨雾检测模型...")
        model = SimpleKNNClassifier(k=5)
        model.fit(X_train, y_train)

        # 4. 评估模型并生成可视化图片1112
        print("\n【4/4】评估模型并生成图片...")
        train_acc, test_acc = evaluate_and_visualize(model, X_train, X_test, y_train, y_test, feature_names)

        # 5. 模拟预测（重度雨雾）11
        print("\n===== 单张图像预测示例（重度雨雾） =====")
        result = predict_simulated_image('heavy', model)
        for k, v in result.items():
            print(f"  {k}: {v}")

        # 额外测试：预测轻度雨雾
        print("\n===== 单张图像预测示例（轻度雨雾） =====")
        result_light = predict_simulated_image('light', model)
        for k, v in result_light.items():
            print(f"  {k}: {v}")

        print("\n🎉 程序运行完成！生成的图片：")
        print("  - rainfog_acc.png（准确率对比图）")
        print("  - rainfog_feature_dist.png（特征分布图）")
        print("  - rainfog_class_acc.png（分类准确率图）")

    except Exception as e:
        print(f"\n❌ 程序执行出错：{e}")
        # 打印详细错误栈（方便调试）
        import traceback

        traceback.print_exc()