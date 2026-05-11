# PCA 图像压缩与重建误差分析

本项目实现了一个机器学习课程中的 PCA 降维与图像压缩实验。项目使用合成 16x16 图案图像，不依赖 CARLA 或任何模拟器数据，通过主成分数量变化观察重建误差、解释方差和压缩率之间的关系。

## 功能内容

- 构建圆形、斜线、条纹和区域块四类合成图案图像。
- 使用 SVD 实现 PCA 主成分分解与重建。
- 比较 2、4、8、16、32 个主成分下的重建质量。
- 统计 MSE、解释方差和存储压缩率。
- 生成重建网格图、误差曲线和解释方差/压缩率曲线。
- 输出 `pca_metrics.csv` 和 `metrics.json`。

## 运行方法

```bash
python src/pca_image_compression_analysis/pca_demo.py --output docs/pr_assets/pca_image_compression_analysis
```

运行后会在 `docs/pr_assets/pca_image_compression_analysis` 中生成运行效果图和指标文件，用于 PR 说明或课程文档展示，不放入 `src` 目录。

## 项目意义

PCA 是神经网络课程前置的经典表示学习方法。它展示了如何用低维潜变量保留图像主要结构，也能作为自编码器等深度表示学习方法的传统对照实验。
