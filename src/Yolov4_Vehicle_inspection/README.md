## Yolov4: 基于Pytorch的Yolov4算法实现车辆检测以及跟踪（深度学习）

---

本项目是基于**[yolov4-pytorch](https://github.com/bubbliiiing/yolov4-pytorch)**实现的

## 文件下载

### 预训练权重下载
链接：https://pan.baidu.com/s/1Wt7_yn6m8B9Uud80ScLkPQ 提取码：6666
其中yolov4_voc_weights.pth是voc数据集的预训练权重，yolov4_coco_weights.pth是coco数据集的预训练权重

### UA-DATRAC数据集下载

链接：https://pan.baidu.com/s/14G_LLbilhe7adrLDVq1k-w 提取码：6666

### 训练完成模型文件下载
链接：https://pan.baidu.com/s/1Ipw7KXqdUIjmUTQ6i0clhw 提取码：6666
该文件为以voc数据集预训练权重使用UA-DATRAC数据集训练了23个epoch后的模型文件

## 数据处理

其中`transfer.py`和`move_pic.py`是针对UA-DATRAC数据集进行处理的两个文件。

## 训练步骤

1. 将VOC格式文件放至data的文件夹
2. JPEGImages文件夹中存放数据图片
3. Annotations文件夹中存放与图片一一对应的标签文件
4. 调整voc2yolo4.py文件中的`trainval_percent` 、`train_percent`的值，控制训练集、测试集、验证集的比例；调整完成后运行该文件，在`data/ImageSets/Main`文件夹中生成对应数据集的图片文件名
5. 运行`voc_annotation.py`文件，运行前将文件中的`classes`列表替换为自己数据集的class的名字
6. 运行后自动生成`yolo_train.txt`文件，文件中每一行的格式为图片路径和图片中标注框的位置以及分类，如下所示
   `data/JPEGImages/MVI_20011__img00001.jpg 592,378,752,540,3 557,120,604,163,2 545,88,580,118,2 508,67,536,92,3 553,70,582,89,7 731,114,783,153,2 902,250,960,357,3`
7. 修改`train.py`中的路径参数与自己电脑上的路径相匹配
   + classes_path：`classes.txt`文件路径，文件格式如下：
        ```
        Truck-Flatbed
        MiniVan
        Sedan
        Suv
        Truck-Pickup
        Police
        Truck-Box-Med
        Taxi
        ...
        ```
   + model_path：预训练权重文件路径或者中断后继续训练时前一个保存的模型文件的路径
   + annotation_path：第6步生成的`yolo_train.txt`文件
8. 调整`train.py`中的参数后运行文件进行训练


## 预测
1. 修改`model_path`的参数，选择合适的模型用于预测，并根据自己电脑情况选择是否启用cuda
2. 修改`predict.py`中的`mode`参数，进入不同的预测模式：`predict`表示单张图片预测，`video`表示视频检测
3. 在predict模式下运行`predict.py`文件，输入预测图片的路径即可进行前向推理，最后通过PIL库输出处理后的图片与处理时间
4. 在video模式下，需要预先修改video_name参数的值，将路径指向需要检测的视频，修改video_fps参数可以控制输出视频的fps；调整好后运行`predict.py`即可对视频进行预测

## 效果

![１](data/%EF%BC%91.jpg)

![２](data/%EF%BC%92.jpg)