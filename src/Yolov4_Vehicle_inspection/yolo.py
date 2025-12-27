# -------------------------------------#
#       创建YOLO类
# -------------------------------------#
import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.yolo4 import YoloBody
from utils.utils import (DecodeBox, letterbox_image, non_max_suppression,
                         yolo_correct_boxes)


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
# --------------------------------------------#
class YOLO(object):
    # YOLO类的默认配置字典，包含模型路径、先验框路径等核心参数
    _defaults = {
        # "model_path"        : 'model_data/Epoch23-Total_Loss0.7718-Val_Loss0.6001.pth',
        "model_path": 'model_data/yolo4_coco_weights.pth',  # 预训练/自定义训练模型权重文件路径
        # "model_path"          : 'model_data/Epoch25-Total_Loss0.7681-Val_Loss0.5913.pth',
        "anchors_path": 'model_data/yolo_anchors.txt',  # YOLOv4先验框配置文件路径
        "classes_path": 'model_data/coco_classes.txt',  # 目标类别名称配置文件路径
        "model_image_size": (416, 416, 3),  # 模型输入图像尺寸（宽，高，通道数）
        "confidence": 0.7,  # 置信度阈值，过滤低置信度预测框
        "iou": 0.3,  # NMS IoU阈值，去除重叠冗余预测框
        "cuda": True,  # 是否使用GPU加速推理
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": False,  # 是否开启不失真缩放（添加灰条）
    }

    # 类方法，用于获取默认配置字典中的指定参数值
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    # YOLO类构造函数，接收自定义参数并初始化核心属性
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # 更新实例属性为默认配置
        self.__dict__.update(kwargs)  # 用自定义参数覆盖默认配置
        self.class_names = self._get_class()  # 加载目标类别名称
        self.anchors = self._get_anchors()  # 加载先验框数据
        self.generate()  # 构建YOLOv4模型并加载权重

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    # 私有方法，读取类别配置文件并返回处理后的类别名称列表
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)  # 解析用户主目录路径
        with open(classes_path) as f:
            class_names = f.readlines()  # 按行读取类别名称
        class_names = [c.strip() for c in class_names]  # 去除每行首尾的空格和换行符
        return class_names

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    # 私有方法，读取先验框配置文件并返回重塑后的先验框数组
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)  # 解析用户主目录路径
        with open(anchors_path) as f:
            anchors = f.readline()  # 读取先验框一行数据
        anchors = [float(x) for x in anchors.split(',')]  # 分割字符串并转换为浮点型
        # 重塑为[3,3,2]数组（3个特征层，每层3个先验框，每个先验框宽高2个维度），[::-1]反转顺序匹配特征层
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    # 私有方法，构建YOLOv4网络、加载权重、初始化解码工具和可视化颜色
    def generate(self):
        # ---------------------------------------------------#
        #   建立yolov4模型
        # ---------------------------------------------------#
        # 构建YOLOv4网络结构，设置为评估模式（关闭Dropout/BatchNorm训练模式）
        self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        # ---------------------------------------------------#
        #   载入yolov4模型的权重
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        # 自动判断使用GPU还是CPU设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载模型权重文件，映射到指定设备
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)  # 将权重加载到网络中
        print('Finished!')

        # 若开启CUDA且设备支持，启用多GPU并行推理
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        # ---------------------------------------------------#
        #   建立三个特征层解码用的工具
        # ---------------------------------------------------#
        # 初始化三个特征层对应的解码工具列表，用于将网络输出转换为真实坐标
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], len(self.class_names), (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        # 基于HSV色彩空间生成不同类别的独特颜色，保证类别间颜色区分度
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    # 核心方法，输入图像进行目标检测+简易追踪，返回绘制了检测框/轨迹的图像
    # cars：已追踪目标的坐标列表（每个元素为目标的历史中心坐标）
    # car_not_vis：每个追踪目标的连续不可见帧数
    def detect_image(self, image, cars=[], car_not_vis=[]):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        # ---------------------------------------------------------#
        image = image.convert('RGB')  # 将图像转换为RGB格式（兼容灰度图输入）

        image_shape = np.array(np.shape(image)[0:2])  # 获取原图尺寸（高，宽）
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        # 图像预处理：按配置进行不失真缩放或直接缩放
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img, dtype=np.float32) / 255.0  # 图像像素值归一化（0-1）
        photo = np.transpose(photo, (2, 0, 1))  # 通道顺序转换（HWC→CHW），适配PyTorch输入
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        images = [photo]  # 添加batch维度（批量大小为1）

        # 关闭梯度计算，提升推理速度并节省显存
        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))  # 将numpy数组转换为Tensor
            if self.cuda:
                images = images.cuda()  # 将Tensor移至GPU

            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)  # 网络前向传播，获取三个特征层的输出
            output_list = []  # 存储各特征层解码后的预测框
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))  # 对每个特征层输出进行解码

            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            output = torch.cat(output_list, 1)  # 拼接三个特征层的预测框
            # 非极大值抑制（NMS），去除重叠冗余预测框
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.iou)

            # ---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            # ---------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()  # 将Tensor转换为numpy数组并移至CPU
            except:
                return image  # 无检测结果时直接返回原图

            # ---------------------------------------------------------#
            #   对预测框进行得分筛选
            # ---------------------------------------------------------#
            # 筛选置信度（目标置信度×类别置信度）大于阈值的预测框
            top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]  # 筛选后的置信度
            top_label = np.array(batch_detections[top_index, -1], np.int32)  # 筛选后的目标类别索引
            top_bboxes = np.array(batch_detections[top_index, :4])  # 筛选后的预测框坐标（xmin,ymin,xmax,ymax）
            # 拆分预测框坐标为单独的数组，便于后续处理
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

            # -----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            # -----------------------------------------------------------------#
            # 将预测框坐标转换为原图尺寸
            if self.letterbox_image:
                boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                           np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)
            else:
                # 直接按比例缩放预测框坐标至原图尺寸
                top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)  # 拼接为完整的坐标数组

        # 设置可视化字体（黑体），字体大小根据原图宽度自适应
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        # 设置检测框线宽，根据原图尺寸自适应
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        # print(cars)
        tmp_vis = [0] * len(cars)  # 临时标记数组，记录每个追踪目标当前帧是否可见
        # 遍历所有检测到的目标，进行可视化和追踪更新
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]  # 获取目标类别名称
            score = top_conf[i]  # 获取目标置信度

            top, left, bottom, right = boxes[i]  # 获取目标框坐标（上，左，下，右）
            top = top - 2  # 轻微调整框坐标，优化可视化效果
            left = left - 2
            bottom = bottom + 2
            right = right + 2

            # 限制坐标在图像范围内，避免越界
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            # label = '{} {:.2f}'.format(predicted_class, score)
            label = '{}{:.2f}'.format(predicted_class, score)  # 拼接类别和置信度标签
            draw = ImageDraw.Draw(image)  # 创建图像绘制对象
            label_size = draw.textsize(label, font)  # 获取标签文字尺寸
            label = label.encode('utf-8')  # 编码标签文字为UTF-8
            # print(label, top, left, bottom, right)

            # 确定标签文字的绘制起始位置
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 计算检测目标的中心坐标，用于追踪匹配
            car = ((left + right) // 2, (top + bottom) // 2)
            if cars != None:
                # 定义距离计算函数（欧氏距离平方，避免开方提升效率）
                def distance(a, b):
                    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])

                id = None  # 匹配到的追踪目标ID
                mindis = int(1e9)  # 最小匹配距离，初始化为极大值
                # 遍历已有追踪目标，寻找最近的匹配目标
                for carid, tmpcar in enumerate(cars):
                    dis = distance(car, tmpcar[-1])
                    if dis <= min(170, mindis):
                        id = carid
                        mindis = dis
                # 无匹配目标则新增追踪，有匹配则更新追踪轨迹
                if id == None or id >= len(tmp_vis):
                    cars.append([car])
                else:
                    tmp_vis[id] += 1
                    cars[id].append(car)
                    # 绘制目标的追踪轨迹线
                    for i in range(len(cars[id]) - 1):
                        draw.line((cars[id][i], cars[id][i + 1]),
                                  fill=self.colors[self.class_names.index(predicted_class)],
                                  width=1)
            else:
                cars.append([car])
            # print(cars)
            # 绘制检测框（多厚度叠加，增强视觉效果）
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            # 绘制标签文字背景框
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            # 绘制标签文字
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)

            del draw  # 释放绘制对象
        # print(car_not_vis)
        # 更新追踪目标的不可见帧数
        for id in range(len(tmp_vis)):
            if tmp_vis[id] == 0:
                car_not_vis[id] += 1
            else:
                car_not_vis[id] = 0
        # 为新增的追踪目标初始化不可见帧数
        for i in range(len(cars) - len(car_not_vis)):
            car_not_vis.append(0)
        fail_frame = 5  # 最大连续不可见帧数，超过则删除追踪目标
        # 倒序遍历，删除超过最大不可见帧数的追踪目标
        for id in reversed(range(len(car_not_vis))):
            if car_not_vis[id] == fail_frame:
                cars.pop(id)
                car_not_vis.pop(id)
        return image  # 返回绘制完成的图像

    # 用于测试模型推理FPS（每秒处理帧数）的方法
    def get_FPS(self, image, test_interval):
        # 调整图片使其符合输入要求
        image_shape = np.array(np.shape(image)[0:2])  # 获取原图尺寸（高，宽）

        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        # 图像预处理，与detect_image方法保持一致
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img, dtype=np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        images = [photo]

        # 首次推理（预热模型，排除初始化耗时影响）
        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.iou)
            try:
                batch_detections = batch_detections[0].cpu().numpy()
                top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
                top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                top_label = np.array(batch_detections[top_index, -1], np.int32)
                top_bboxes = np.array(batch_detections[top_index, :4])
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                    top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

                if self.letterbox_image:
                    boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                               np.array([self.model_image_size[0], self.model_image_size[1]]),
                                               image_shape)
                else:
                    top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                    top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                    top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                    top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                    boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

            except:
                pass

        # 记录多次推理的起始时间
        t1 = time.time()
        # 重复推理test_interval次，计算平均耗时
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                output_list = []
                for i in range(3):
                    output_list.append(self.yolo_decodes[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, len(self.class_names),
                                                       conf_thres=self.confidence,
                                                       nms_thres=self.iou)
                try:
                    batch_detections = batch_detections[0].cpu().numpy()
                    top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
                    top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                    top_label = np.array(batch_detections[top_index, -1], np.int32)
                    top_bboxes = np.array(batch_detections[top_index, :4])
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                        top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3],
                                                                                                    -1)

                    if self.letterbox_image:
                        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                                   np.array([self.model_image_size[0], self.model_image_size[1]]),
                                                   image_shape)
                    else:
                        top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                        top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                        top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                        top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                        boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

                except:
                    pass

        # 记录多次推理的结束时间
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval  # 计算单张图像的平均推理时间
        return tact_time  # 返回平均推理时间（用于计算FPS=1/tact_time）