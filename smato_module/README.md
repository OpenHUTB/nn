<<<<<<< HEAD
# SMATO - Image-based Pedestrian Smartphone Usage Classification

In today's digital age, it is becoming increasingly common to witness pedestrians engrossed in their smartphones while navigating through bustling traffic or its vicinity [[1]](#1). Although regulations surrounding smartphone usage by drivers are prevalent in many countries, the impact of smartphone distraction on pedestrians should not be overlooked. Studies have revealed that smartphone use among pedestrians significantly hampers their situational awareness and attentiveness, consequently increasing the likelihood of accidents and injuries [[2]](#2). With the world gradually shifting towards autonomous driving, it is imperative to incorporate advanced safety measures that can identify smartphone usage among pedestrians. Leveraging widely available sensors like cameras, it becomes feasible to detect instances of smartphone engagement and take additional precautions. To facilitate this endeavor, this repository presents a carefully curated dataset and transfer-learning-based classifier.

![alt text](https://github.com/saadejazz/smato/blob/main/images/example_predictions.png)

The prediction scheme for the pictures above is **<predicted_label: (true|false)>, <true_label: (1|0)>**


This module is part of the following [paper](https://ieeexplore.ieee.org/abstract/document/10737454)
```
@article{ejaz2024trust,
  title={Trust-aware safe control for autonomous navigation: Estimation of system-to-human trust for trust-adaptive control barrier functions},
  author={Ejaz, Saad and Inoue, Masaki},
  journal={IEEE Transactions on Control Systems Technology},
  year={2024},
  publisher={IEEE}
}
```

## Sourcing the Dataset

The dataset utilized for this project has been compiled from various publicly available pedestrian datasets that encompass a wide range of images depicting pedestrians in diverse ambient environments, orientations, and engagements. These datasets include prominent sources such as PETA [[3]](#3).

To ensure the dataset's comprehensiveness, an additional collection of images was obtained through Open Source Intelligence (OSINT) techniques, specifically by leveraging image search functionalities provided by platforms like Google and Bing (example scraper: [icrawler](https://icrawler.readthedocs.io/en/latest/). This process facilitated the inclusion of images captured under varying lighting conditions and camera angles. In order to augment the dataset, most images were flipped horizontally, which effectively doubled the dataset's size while maintaining its diversity and correctness.

All images underwent a preliminary step of pedestrian isolation. This was achieved by employing a person detector, specifically [OpenPifPaf](https://openpifpaf.github.io/intro.html). By accurately detecting and segmenting pedestrians within the images, the subsequent annotation process was applied. The annotation process itself followed an iterative approach to gradually improve the dataset's quality and inclusiveness. It commenced with the curation of a small initial dataset, which was used to train an initial classifier. With this classifier in place, additional images were sourced from a variety of different sources. Subsequently, the previously trained classifier was utilized to automatically annotate these newly acquired images. However, to ensure accuracy, manual inspection of the annotations was conducted to identify any misclassifications (as the dataset grew in size, it was expected that the number of misclassifications would decrease). This manual inspection of misclasification also presented an opportunity to analyze the reasons behind misclassifications - knowledge that influenced future data sourcing strategies or inform adjustments to the model building process. The iterative nature of this approach, involving constant refinement through training, data sourcing, and manual inspection, ensured the dataset's continuous improvement, ultimately leading to a more reliable and comprehensive resource for smartphone usage detection in pedestrian images.

The dataset consist of a total of 13866 images of pedestrian (single pedestrian per image), from which 3770 are engaged with a smartphone while 10096 of them are not. This imbalance is kept to demonstrate real-world composition of smartphone users amongst pedestrians, while avoiding severe imbalances that might hinder training. The dataset can be downloaded from [here](https://drive.google.com/file/d/1cI6OcMlKPXcWCLZtmScGpkwnprvdVOLo/view?usp=sharing).

## Training the Classifier

The notebook ```train.ipynb``` contains comprehensive information regarding the steps followed to train the classifier. Transfer learning was employed to capitalize on the capabilities of pre-trained deep learning models. Specifically, MobileNet V2 was chosen as the feature extractor, enabling the extraction of a 1280-dimensional embedding vector from input images resized to 224x224 pixels. This embedding vector effectively captured the essential features and patterns related to smartphone usage in pedestrian images. The classifier's architecture consists of three fully connected layers following the feature extractor layer, with batch normalization and dropout applied after each layer. Dropout layers play a crucial role in mitigating overfitting issues by randomly disabling a fraction of neurons during training, thereby promoting model generalization.

Within the notebook, the train-validation-test split is detailed, highlighting how the dataset was divided to facilitate model training, validation, and final evaluation. Preprocessing steps, such as resizing the images and preparing the data for training, are also outlined. During training, the model was compiled, and the F1 score was chosen as the evaluation metric. The F1 score is particularly suited for imbalanced datasets, providing a comprehensive assessment of the model's performance by considering both precision and recall. After training, the model achieved an impressive F1 score of 87.68% and an accuracy of 92.71% on the test dataset. These results indicate that the classifier has learned to accurately detect instances of smartphone usage in pedestrian images, contributing to the advancement of safety considerations in autonomous driving and pedestrian environments.

The notebook ```train.ipynb``` provides further insights into the code implementation, training parameters, hyperparameter tuning, and any additional details necessary to reproduce the results or further improve the classifier's performance.

## Inference
The notebook ```infer.ipynb``` provides details on how to classify your images using this classifier. A saved model is already included in the repository in the folder ```saved_moels```.

## EfficientNet V2 as another model
A model using EfficientNet V2M as the feature extractor (has better benchmark performance) was also trained. This trained model performed slightly better than the one based on MobileNet V2, and can be downloaded from [here](https://drive.google.com/file/d/1IEBlPKuedAusiFGQOx-udnTLAt3-Aj2c/view?usp=sharing). However, it must be noted that the preprocessing steps for this feature extractor are included in the model and should be avoided in the inference code. More details on the training of this model are in the notebook: ```misc/train-efficientnet.ipynb```. Moreover, it should also be noted that this model is heavier and hence would require a greater computational burden and hence inference time.

## References
<a id="1">[1]</a> Nasar, J.L. and Troyer, D., 2013. Pedestrian injuries due to mobile phone use in public places. Accident Analysis & Prevention, 57, pp.91-95.

<a id="2">[2]</a>  Frej, D., Jaśkiewicz, M., Poliak, M. and Zwierzewicz, Z., 2022. Smartphone Use in Traffic: A Pilot Study on Pedestrian Behavior. Applied Sciences, 12(24), p.12676.

<a id="3">[3]</a> Y. Deng, P. Luo, C. C. Loy, X. Tang, "Pedestrian attribute recognition at far distance," in Proceedings of ACM Multimedia (ACM MM), 2014

## 网盘资源下载（夸克云盘）

### 1. 模型权重smato_mobilenet_v2m
链接：https://pan.quark.cn/s/20a4c9ebadb2
提取码：MhpM

### 2. 训练数据集smato_images
链接：https://pan.quark.cn/s/e0310dc2237b
提取码：tVBE
=======
# 神经网络实现代理

利用神经网络/ROS 实现 Carla（车辆、行人的感知、规划、控制）、AirSim、Mujoco 中人和载具的代理。

## 环境配置

* 平台：Windows 10/11，Ubuntu 20.04/22.04
* 软件：Python 3.7-3.12（需支持3.7）、Pytorch（尽量不使用Tensorflow）
* 相关软件下载 [链接](https://pan.baidu.com/s/1IFhCd8X9lI24oeYQm5-Edw?pwd=hutb)

## 功能模块表
| 模块类别                | 模块名 | 链接                                                                            | 其他                                                                  |
|-------------------|------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------|
| 感知             | 车辆检测以及跟踪 | [Yolov4_Vehicle_inspection](https://github.com/OpenHUTB/nn/tree/main/src/Yolov4_Vehicle_inspection)                                                                          | -                                          |
| 规划             | 车辆全局路径规划 | [carla_slam_gmapping](https://github.com/OpenHUTB/nn/tree/main/src/carla_slam_gmapping)                                                                          | -                                          |
| 控制             | 手势控制无人机 | [autonomus_drone_hand_gesture_project](https://github.com/OpenHUTB/nn/tree/main/src/autonomus_drone_hand_gesture_project)                                                                          | -                                          |
| 控制             | 倒车入库 | [autonomus_drone_hand_gesture_project](https://github.com/OpenHUTB/nn/tree/main/src/autonomus_drone_hand_gesture_project)                                                                          | [效果](https://github.com/OpenHUTB/nn/pull/4399)                                          |


## 贡献指南

准备提交代码之前，请阅读 [贡献指南](https://github.com/OpenHUTB/.github/blob/master/CONTRIBUTING.md) 。
代码的优化包括：注释、[PEP 8 风格调整](https://peps.pythonlang.cn/pep-0008/) 、将神经网络应用到Carla模拟器中、撰写对应 [文档](https://openhutb.github.io/nn/) 、添加 [源代码对应的自动化测试](https://docs.github.com/zh/actions/use-cases-and-examples/building-and-testing/building-and-testing-python) 等（从Carla场景中获取神经网络所需数据或将神经网络的结果输出到场景中）。

### 约定

* 每个模块位于`src/{模块名}`目录下，`模块名`需要用2-3个单词表示，首字母不需要大写，下划线`_`分隔，不能宽泛，越具体越好
* 每个模块的入口须为`main.`开头，比如：main.py、main.cpp、main.bat、main.sh等，提供的ROS功能以`main.launch`文件作为启动配置文件
* 每次pull request都需要保证能够通过main脚本直接运行整个模块，在提交信息中提供运行动图或截图；Pull Request的标题不能随意，需要概括具体的修改内容；README.md文档中提供运行环境和运行步骤的说明
* 仓库尽量保存文本文件，二进制文件需要慎重，如运行需要示例数据，可以保存少量数据，大量数据可以通过提供网盘链接并说明下载链接和运行说明


### 文档生成

测试生成的文档：
1. 使用以下命令安装`mkdocs`和相关依赖：
```shell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
（可选）安装完成后使用`mkdocs --version`查看是否安装成功。

2. 在命令行中进入`nn`目录下，运行：
```shell
mkdocs build
mkdocs serve
```
然后使用浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)，查看文档页面能否正常显示。

## 参考

* [代理模拟器文档](https://openhutb.github.io)
* 已有相关 [无人车](https://openhutb.github.io/doc/used_by/) 、[无人机](https://openhutb.github.io/air_doc/third/used_by/) 、[具身人](https://openhutb.github.io/doc/pedestrian/humanoid/) 的实现
* [神经网络原理](https://github.com/OpenHUTB/neuro)
>>>>>>> b87deba5e3e1477199105a0de3a1097a7b009e9f
