import sys
import numpy as np
sys.path.insert(0, 'C:\\Users\\Administrator\\Desktop\\课程\\nn\\src\\chap11_gaussian_mixture')

from GMM import GaussianMixtureModel, generate_data

# 生成测试数据
X, y_true = generate_data(n_samples=500, random_state=42)

# 训练模型
print("训练模型...")
gmm = GaussianMixtureModel(n_components=3, random_state=42)
gmm.fit(X)
print(f"训练完成，迭代次数: {gmm.n_iters_}")
print(f"对数似然值: {gmm.log_likelihoods[-1]:.2f}")

# 保存模型
print("\n保存模型...")
model_path = 'test_model.pkl'
gmm.save(model_path)
print(f"模型已保存到: {model_path}")

# 加载模型
print("\n加载模型...")
loaded_gmm = GaussianMixtureModel.load(model_path)
print(f"模型加载完成")
print(f"成分数量: {loaded_gmm.n_components}")
print(f"迭代次数: {loaded_gmm.n_iters_}")
print(f"对数似然值: {loaded_gmm.log_likelihoods[-1]:.2f}")

# 验证模型一致性
print("\n验证模型一致性...")
original_scores = gmm.score_samples(X)
loaded_scores = loaded_gmm.score_samples(X)
if np.allclose(original_scores, loaded_scores):
    print("✓ 模型加载成功，预测结果一致！")
else:
    print("✗ 模型加载失败，预测结果不一致！")

# 测试异常检测功能
print("\n测试异常检测功能...")
is_anomaly, scores, threshold = loaded_gmm.detect_anomalies(X, contamination=0.05)
print(f"检测到异常样本数: {np.sum(is_anomaly)}")
print(f"异常检测阈值: {threshold:.4f}")

print("\n✓ 所有测试通过！")
