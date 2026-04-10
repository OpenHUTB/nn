import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
# 导入仓库内的手势分类器
from gesture_classifier import GestureClassifier

# 数据集路径：优先用仓库已有数据，若无则生成演示数据
DATASET_PATH = os.path.join(os.path.dirname(__file__), "gesture_dataset.pkl")
TEST_SIZE = 0.2  # 测试集比例
ALGORITHMS = ['svm', 'random_forest', 'mlp', 'knn', 'decision_tree', 'ada_boost']  # 对比算法
# 结果保存路径（保存在当前文件夹）
SAVE_PLOT_PATH = os.path.join(os.path.dirname(__file__), "algorithm_comparison.png")
SAVE_RESULT_PATH = os.path.join(os.path.dirname(__file__), "comparison_results.csv")

def generate_demo_dataset():
    """生成测试用手势数据集（6类，600样本，63维特征）"""
    np.random.seed(42)
    num_classes = 6
    samples_per_class = 100
    feature_dim = 63  # 21个关键点×3坐标，匹配仓库规范

    X = []
    y = []
    gesture_classes = ["open_palm", "closed_fist", "victory", "thumb_up", "pointing_up", "ok_sign"]

    for class_idx in range(num_classes):
        base_feature = np.ones(feature_dim) * class_idx * 2
        noise = np.random.randn(samples_per_class, feature_dim) * 0.5
        class_features = base_feature + noise
        X.extend(class_features)
        y.extend([class_idx] * samples_per_class)

    X = np.array(X)
    y = np.array(y)
    dataset = {
        "features": X,
        "labels": y,
        "gesture_classes": gesture_classes
    }

    with open(DATASET_PATH, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"✅ 演示数据集已生成：{DATASET_PATH}")
    print(f"数据集信息：{len(X)}样本，{feature_dim}维特征，{num_classes}手势类别")

# -------------------------- 运行对比实验 --------------------------
def main():
    # 1. 检查/生成数据集
    if not os.path.exists(DATASET_PATH):
        print("⚠️ 未找到真实数据集，生成演示数据集...")
        generate_demo_dataset()

    # 2. 初始化结果存储
    results = {
        "algorithm": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "train_time": [],
        "infer_time_per_sample": []
    }

    print("\n" + "="*60)
    print("开始手势分类算法对比实验（作业）")
    print(f"当前路径：{os.path.dirname(__file__)}")
    print(f"对比算法：{ALGORITHMS}")
    print("="*60)

    # 3. 遍历算法训练+评估
    for algo in ALGORITHMS:
        print(f"\n--- 训练 {algo.upper()} 模型 ---")
        start_time = time.time()

        # 创建分类器并训练
        classifier = GestureClassifier(model_type=algo)
        test_acc = classifier.train(DATASET_PATH, test_size=TEST_SIZE)

        # 计算耗时和训练集准确率
        train_time = time.time() - start_time
        with open(DATASET_PATH, 'rb') as f:
            dataset = pickle.load(f)
        X, y = dataset['features'], dataset['labels']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42, stratify=y
        )
        X_train_scaled = classifier.scaler.transform(X_train)
        train_acc = classifier.model.score(X_train_scaled, y_train)

        # 计算单样本推理耗时
        X_test_scaled = classifier.scaler.transform(X_test)
        infer_start = time.time()
        classifier.model.predict(X_test_scaled)
        infer_per_sample = (time.time() - infer_start) / len(X_test_scaled)

        # 保存结果
        results["algorithm"].append(algo)
        results["train_accuracy"].append(round(train_acc, 4))
        results["test_accuracy"].append(round(test_acc, 4))
        results["train_time"].append(round(train_time, 4))
        results["infer_time_per_sample"].append(round(infer_per_sample, 6))

        # 打印当前结果
        print(f"  训练准确率：{train_acc:.4f} | 测试准确率：{test_acc:.4f}")
        print(f"  训练耗时：{train_time:.4f}s | 单样本推理耗时：{infer_per_sample:.6f}s")

    # 4. 保存+可视化结果
    df = pd.DataFrame(results)
    df.to_csv(SAVE_RESULT_PATH, index=False)
    print(f"\n✅ 结果表格已保存：{SAVE_RESULT_PATH}")

    # 绘制对比图（支持中文）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1：测试准确率
    ax1.bar(df['algorithm'], df['test_accuracy'], color='skyblue')
    ax1.set_title('各算法测试准确率', fontsize=14)
    ax1.set_ylabel('准确率')
    ax1.set_ylim(0, 1.1)
    for i, v in enumerate(df['test_accuracy']):
        ax1.text(i, v+0.02, f"{v:.4f}", ha='center')

    # 子图2：训练耗时
    ax2.bar(df['algorithm'], df['train_time'], color='orange')
    ax2.set_title('各算法训练耗时', fontsize=14)
    ax2.set_ylabel('耗时（秒）')
    for i, v in enumerate(df['train_time']):
        ax2.text(i, v+0.02, f"{v:.4f}", ha='center')

    # 子图3：推理耗时
    ax3.bar(df['algorithm'], df['infer_time_per_sample'], color='green')
    ax3.set_title('单样本推理耗时', fontsize=14)
    ax3.set_ylabel('耗时（秒）')
    for i, v in enumerate(df['infer_time_per_sample']):
        ax3.text(i, v+0.00001, f"{v:.6f}", ha='center')

    # 子图4：训练vs测试准确率
    x = np.arange(len(df['algorithm']))
    ax4.bar(x-0.2, df['train_accuracy'], 0.4, label='训练准确率', color='lightcoral')
    ax4.bar(x+0.2, df['test_accuracy'], 0.4, label='测试准确率', color='lightgreen')
    ax4.set_title('训练/测试准确率对比', fontsize=14)
    ax4.set_ylabel('准确率')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['algorithm'])
    ax4.legend()

    plt.tight_layout()
    plt.savefig(SAVE_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存：{SAVE_PLOT_PATH}")
    print("\n🎉 实验完成！所有结果已保存到当前文件夹。")

if __name__ == "__main__":
    main()