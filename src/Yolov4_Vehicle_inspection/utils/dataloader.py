import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import merge_bboxes


class YoloDataset(Dataset):
    # 新增：YOLO模型专用数据集类，实现数据加载、实时数据增强（含Mosaic）
    def __init__(self, train_lines, image_size, mosaic=True, is_train=True):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines  # 数据集标注行列表，每行格式：图片路径 框1(x1,y1,x2,y2,cls) 框2(...)
        self.train_batches = len(train_lines)  # 数据集总样本数
        self.image_size = image_size  # 模型输入图片尺寸 (h, w)
        self.mosaic = mosaic  # 是否启用Mosaic数据增强（提升小目标检测效果）
        self.flag = True  # Mosaic交替标志，控制是否每次取4张图拼接
        self.is_train = is_train  # 是否为训练模式（训练模式开启数据增强，验证模式关闭）

    def __len__(self):
        # 新增：返回数据集总长度，PyTorch Dataset必需实现的方法
        return self.train_batches

    def rand(self, a=0, b=1):
        # 新增：生成[a, b)区间的随机浮点数，用于数据增强的随机参数
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        """实时数据增强的随机预处理"""
        # 新增：解析单条标注行，分割图片路径和目标框信息
        line = annotation_line.split()
        image = Image.open(line[0])  # 打开图片（PIL格式）
        iw, ih = image.size  # 原始图片宽、高
        h, w = input_shape  # 模型输入尺寸（高、宽）
        # 新增：解析目标框，格式转换为numpy数组，每个框：[x1, y1, x2, y2, 类别]
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 新增：验证模式（不做随机增强，仅缩放+居中）
        if not random:
            # 计算等比例缩放系数（保证图片完整放入输入尺寸）
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)  # 缩放后图片宽
            nh = int(ih*scale)  # 缩放后图片高
            dx = (w-nw)//2  # x方向偏移（居中）
            dy = (h-nh)//2  # y方向偏移（居中）

            # 缩放图片（BICUBIC插值，保证缩放质量）
            image = image.resize((nw,nh), Image.BICUBIC)
            # 新增：创建灰色背景画布（128,128,128），将缩放后的图片粘贴到中心
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)  # 转换为numpy数组

            # 调整目标框坐标
            box_data = np.zeros((len(box), 5))
            if len(box) > 0:
                np.random.shuffle(box)  # 随机打乱框顺序（无实际作用，保持与训练模式一致）
                # 新增：框坐标缩放 + 偏移（匹配图片的缩放和居中）
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                # 新增：限制框坐标在图片范围内（防止越界）
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                # 新增：计算框的宽、高，过滤掉无效框（宽/高<1像素）
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            return image_data, box_data

        # 新增：训练模式（随机数据增强）
        # 调整图片大小：随机宽高比抖动
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)  # 随机缩放系数（0.25~2倍）
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)  # 缩放图片

        # 放置图片：随机偏移（增加位置多样性）
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        # 新增：创建随机颜色背景画布（替代固定灰色，增强多样性）
        new_image = Image.new('RGB', (w, h),
                              (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 是否翻转图片：50%概率水平翻转
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域变换：HSV空间随机调整色调、饱和度、明度
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        # 新增：PIL转OpenCV格式，RGB转HSV，归一化到0~1
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        # 调整色调（hue范围：-0.1~0.1，对应HSV的0~360）
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        # 调整饱和度、明度
        x[..., 1] *= sat
        x[..., 2] *= val
        # 新增：限制HSV值在有效范围（0~360 for H, 0~1 for S/V）
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        # 新增：HSV转回RGB，恢复到0~255范围
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # 调整目标框坐标
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)  # 随机打乱框顺序
            # 新增：框坐标缩放 + 偏移（匹配图片的缩放和随机放置）
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            # 新增：如果图片翻转，框的x坐标需要对称调整
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            # 新增：限制框坐标在图片范围内
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            # 新增：过滤无效框（宽/高<1像素）
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        return image_data, box_data

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, hue=.1, sat=1.5, val=1.5):
        # 新增：Mosaic数据增强核心函数，将4张图片拼接成1张，提升小目标和多尺度检测能力
        h, w = input_shape
        min_offset_x = 0.3  # x方向最小偏移比例（避免图片重叠过多）
        min_offset_y = 0.3  # y方向最小偏移比例
        scale_low = 1 - min(min_offset_x, min_offset_y)  # 缩放系数下限
        scale_high = scale_low + 0.2  # 缩放系数上限

        image_datas = []  # 存储4张图片的预处理结果
        box_datas = []    # 存储4张图片的框数据
        index = 0

        # 新增：4张图片的初始放置位置（左上、左下、右下、右上）
        place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
        place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]
        for line in annotation_line:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0])
            image = image.convert("RGB")  # 确保图片为RGB格式（避免灰度图）
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # 是否翻转图片
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 对输入进来的图片进行缩放
            new_ar = w / h
            scale = self.rand(scale_low, scale_high)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 进行色域变换（同单图增强逻辑）
            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue*360
            x[..., 0][x[..., 0]>1] -= 1
            x[..., 0][x[..., 0]<0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:,:, 0]>360, 0] = 360
            x[:, :, 1:][x[:, :, 1:]>1] = 1
            x[x<0] = 0
            image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) # numpy array, 0 to 1

            # 新增：将归一化的图片转回0~255的PIL格式
            image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w, h),
                                  (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # 对box进行重新处理（同单图增强逻辑）
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # 新增：随机生成切割点（将4张图片拼接成1张的核心步骤）
        cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
        cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

        # 新增：拼接4张图片，按切割点组合成最终的Mosaic图片
        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]  # 左上区域
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]  # 左下区域
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]  # 右下区域
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]  # 右上区域

        # 对框进行进一步的处理：合并4张图片的框，处理跨切割点的框
        new_boxes = np.array(merge_bboxes(box_datas, cutx, cuty))

        return new_image, new_boxes

    def __getitem__(self, index):
        # 新增：PyTorch Dataset必需实现的方法，返回单样本的预处理数据
        lines = self.train_lines
        n = self.train_batches
        index = index % n  # 新增：防止索引越界（支持多epoch循环）
        if self.mosaic:
            # 新增：Mosaic模式，每次取4张连续的图片拼接（需保证索引不越界）
            if self.flag and (index + 4) < n:
                img, y = self.get_random_data_with_Mosaic(lines[index:index + 4], self.image_size[0:2])
            else:
                # 新增：不足4张时，退化为单图增强
                img, y = self.get_random_data(lines[index], self.image_size[0:2], random=self.is_train)
            self.flag = bool(1-self.flag)  # 新增：交替开关，避免连续使用Mosaic导致显存波动
        else:
            # 新增：禁用Mosaic时，直接使用单图增强
            img, y = self.get_random_data(lines[index], self.image_size[0:2], random=self.is_train)

        if len(y) != 0:
            # 从坐标转换成0~1的百分比（归一化）
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]  # x1/w
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]  # y1/h
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]  # x2/w
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]  # y2/h

            boxes = np.maximum(np.minimum(boxes, 1), 0)  # 新增：限制归一化后的值在0~1之间
            # 新增：将x1y1x2y2格式转换为中心坐标+宽高格式（YOLO模型输入要求）
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2  # cx = x1 + w/2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2  # cy = y1 + h/2
            # 新增：合并中心坐标+宽高+类别标签
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        img = np.array(img, dtype=np.float32)

        # 新增：调整图片维度顺序 (h, w, c) → (c, h, w)（适配PyTorch张量格式）
        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_targets = np.array(y, dtype=np.float32)  # 框数据转为浮点型
        return tmp_inp, tmp_targets


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    # 新增：自定义DataLoader批次处理函数，解决不同样本框数量不一致的问题
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)  # 新增：图片列表转为numpy数组（形状：[batch, c, h, w]）
    # 新增：框数据保持列表格式（每个元素是不同长度的框数组）
    return images, bboxes