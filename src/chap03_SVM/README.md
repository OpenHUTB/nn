

# 支持向量机(SVM)



## 问题描述：

本次作业分为三个部分：

1. 使用基于某种核函数（线性，多项式或高斯核函数）的SVM 解决非线性可分的二分类问题，数
  据集为train_kernel.txt 及test_kernel.txt。

2. 分别使用线性分类器（squared error）、logistic 回归（cross entropy error）以及SVM（hinge error) 解
  决线性二分类问题，并比较三种模型的效果。数据集为train_linear.txt 及test_linear.txt。
  三种误差函数定义如下（Bishop P327）：
![image](http://latex.codecogs.com/gif.latex?E_%7Blinear%7D%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%28y_%7Bn%7D%20-t_%7Bn%7D%29%5E%7B2%7D&plus;%5Clambda%20%5Cleft%20%5C%7C%20%5Cmathbf%7Bw%7D%20%5Cright%20%5C%7C%5E%7B2%7D)  

![image](http://latex.codecogs.com/gif.latex?E_%7Blogistic%7D%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7Dlog%281&plus;exp%28-y_%7Bn%7Dt_%7Bn%7D%29%29%20&plus;%20%5Clambda%5Cleft%20%5C%7C%20%5Cmathbf%7Bw%7D%20%5Cright%20%5C%7C%5E%7B2%7D) 

![image](http://latex.codecogs.com/gif.latex?E_%7BSVM%7D%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5B1-y_%7Bn%7Dt_%7Bn%7D%5D&plus;%5Clambda%20%5Cleft%20%5C%7C%20%5Cmathbf%7Bw%7D%20%5Cright%20%5C%7C%5E%7B2%7D)


  ​
  其中![image](http://latex.codecogs.com/gif.latex?y_%7Bn%7D%3D%5Cmathbf%7Bw%7D%5E%7BT%7Dx_%7Bn%7D&plus;b),![image](http://latex.codecogs.com/gif.latex?t_%7Bn%7D) 为类别标签。

3. 使用多分类SVM 解决三分类问题。数据集为train_multi.txt 及test_multi.txt。（5%）





## 数据集: 

 	MNIST数据集包括60000张训练图片和10000张测试图片。图片样本的数量已经足够训练一个很复杂的模型（例如 CNN的深层神经网络）。它经常被用来作为一个新 的模式识别模型的测试用例。而且它也是一个方便学生和研究者们执行用例的数据集。除此之外，MNIST数据集是一个相对较小的数据集，可以在你的笔记本CPUs上面直接执行





## 题目要求： 

- [ ] 请使用代码模板rbm.py，补全缺失部分，尽量不改动主体部分。
- [ ] 推荐使用python 及numpy 编写主要逻辑代码，适度使用框架。

SVM 线性分类器实现

## 项目概述

本项目实现了一个基于梯度下降法的线性支持向量机(SVM)分类器，用于二分类任务。该实现包含数据加载、模型训练、预测和评估功能。

## 文件结构

├── svm.py                # SVM实现主文件

├── data/
├── train_linear.txt  # 训练数据

└── test_linear.txt   # 测试数据

└── README.md             # 本说明文件

## 功能说明

数据加载
load_data(fname) 函数从文本文件加载数据

数据格式：每行包含两个特征(x1, x2)和一个标签(0或1)

SVM模型
实现了线性SVM分类器

使用梯度下降法优化目标函数

包含L2正则化项控制模型复杂度

## 主要功能
数据加载和预处理

模型训练

预测功能

准确率评估

## 使用方法
准备训练和测试数据文件(格式见示例)

运行主程序：

      python svm.py
   
程序将输出训练集和测试集的准确率

## 参数配置

可在代码中调整以下超参数：
学习率(learning_rate)

正则化系数(lambda_)

训练轮数(epochs)

## 示例输出

train accuracy: 98.3%
test accuracy: 97.5%

## 依赖环境
Python 3.5.2+

NumPy

## 注意事项
当前实现为线性SVM，适用于线性可分数据

对于非线性问题，可考虑添加核函数扩展

训练数据应标准化以获得更好效果
