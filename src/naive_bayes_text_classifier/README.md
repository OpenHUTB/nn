# 朴素贝叶斯文本分类实验

本项目实现了一个多项式朴素贝叶斯文本分类实验，用于演示概率模型在短文本分类任务中的训练、预测和可视化分析。项目使用程序内部生成的课程主题短文本数据，不使用 CARLA 或任何模拟器数据。

## 功能内容

- 构建算法、机器人、视觉三类合成短文本数据集。
- 实现分词、词表构建和词频向量化。
- 使用拉普拉斯平滑训练多项式朴素贝叶斯分类器。
- 输出测试集准确率、macro F1 和混淆矩阵。
- 生成混淆矩阵图、类别高概率特征词图和指标汇总图。
- 将运行效果图和指标文件输出到 `docs/pr_assets/naive_bayes_text_classifier`，不放入 `src` 目录。

## 运行方法

```bash
python src/naive_bayes_text_classifier/nb_text_demo.py --output docs/pr_assets/naive_bayes_text_classifier
```

## 验证方法

```bash
python src/naive_bayes_text_classifier/tests/test_nb_text_demo.py
python -m py_compile src/naive_bayes_text_classifier/nb_text_demo.py src/naive_bayes_text_classifier/tests/test_nb_text_demo.py
```

## 项目意义

朴素贝叶斯是机器学习课程中经典的概率分类模型。它能够清楚展示先验概率、条件概率、拉普拉斯平滑、词频特征和分类决策之间的关系，也能作为后续神经网络文本分类模型的传统基线。
