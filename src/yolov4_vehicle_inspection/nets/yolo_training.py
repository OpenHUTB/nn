import os

import math
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


def jaccard(_box_a, _box_b):
    # 转换xywh格式为x1y1x2y2格式（左上角+右下角）
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    # 构造转换后的框张量
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    # 获取两个框集合的数量
    A = box_a.size(0)
    B = box_b.size(0)
    # 广播计算所有框对的交集右下角（取最小值）
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    # 广播计算所有框对的交集左上角（取最大值）
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    # 计算交集宽高，裁剪负值（无交集时为0）
    inter = torch.clamp((max_xy - min_xy), min=0)

    # 计算交集面积
    inter = inter[:, :, 0] * inter[:, :, 1]
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 求IOU
    union = area_a + area_b - inter
    return inter / union  # [A,B]  返回所有框对的交并比矩阵


# ---------------------------------------------------#
#   平滑标签
# ---------------------------------------------------#
def smooth_labels(y_true, label_smoothing, num_classes):
    # 标签平滑公式：缓解过拟合，提升模型泛化能力
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    # 防止分母为0，添加极小值裁剪
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    # 计算中心的欧氏距离
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算包裹框的对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    # 计算CIOU的中心距离惩罚项
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

    # 计算长宽比惩罚项v和权重系数alpha
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
        b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
    # 最终CIOU：包含IOU、中心距离、长宽比三项惩罚
    ciou = ciou - alpha * v
    return ciou


def clip_by_tensor(t, t_min, t_max):
    # 将张量t的值裁剪到[t_min, t_max]区间，避免后续对数计算出现无穷大
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def MSELoss(pred, target):
    # 均方误差损失：用于回归任务（此处为预留接口）
    return (pred - target) ** 2


def BCELoss(pred, target):
    # 二分类交叉熵损失：用于置信度和类别概率的预测
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, label_smooth=0, cuda=True, normalize=True):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors  # 原始先验框尺寸（图像尺度）
        self.num_anchors = len(anchors)  # 先验框总数
        self.num_classes = num_classes  # 目标类别数
        self.bbox_attrs = 5 + num_classes  # 每个先验框的预测属性数（xywh+置信度+类别）
        self.img_size = img_size  # 输入图像尺寸 (w, h)
        self.feature_length = [img_size[0] // 32, img_size[0] // 16, img_size[0] // 8]  # 三个特征层的尺寸
        self.label_smooth = label_smooth  # 标签平滑系数

        self.ignore_threshold = 0.5  # 负样本忽略阈值
        self.lambda_conf = 1.0  # 置信度损失权重
        self.lambda_cls = 1.0  # 类别损失权重
        self.lambda_loc = 1.0  # 定位损失权重
        self.cuda = cuda  # 是否使用GPU加速
        self.normalize = normalize  # 是否按正样本数量归一化损失

    def forward(self, input, targets=None):
        # ----------------------------------------------------#
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        # ----------------------------------------------------#

        # -----------------------#
        #   一共多少张图片
        # -----------------------#
        bs = input.size(0)
        # -----------------------#
        #   特征层的高
        # -----------------------#
        in_h = input.size(2)
        # -----------------------#
        #   特征层的宽
        # -----------------------#
        in_w = input.size(3)

        # -----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        # -----------------------------------------------------------------------#
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w

        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        # -------------------------------------------------#
        # 将原始先验框缩放到当前特征层尺度
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # -----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        # -----------------------------------------------#
        # 调整输入张量形状，便于逐先验框计算损失
        prediction = input.view(bs, int(self.num_anchors / 3),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # 获得置信度，是否有物体（sigmoid激活转为0-1区间）
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度（sigmoid激活转为0-1区间）
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # ---------------------------------------------------------------#
        #   找到哪些先验框内部包含物体
        #   利用真实框和先验框计算交并比
        #   mask        batch_size, 3, in_h, in_w   有目标的特征点
        #   noobj_mask  batch_size, 3, in_h, in_w   无目标的特征点
        #   t_box       batch_size, 3, in_h, in_w, 4   中心宽高的真实值
        #   tconf       batch_size, 3, in_h, in_w   置信度真实值
        #   tcls        batch_size, 3, in_h, in_w, num_classes  种类真实值
        # ----------------------------------------------------------------#
        # 生成损失计算所需的真实标签和掩码
        mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y = self.get_target(targets,
                                                                                                   scaled_anchors, in_w,
                                                                                                   in_h,
                                                                                                   self.ignore_threshold)

        # ---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        # ----------------------------------------------------------------#
        # 更新无目标掩码，忽略IOU过高的负样本
        noobj_mask, pred_boxes_for_ciou = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask)

        # 将所有张量移至GPU（若启用）
        if self.cuda:
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            box_loss_scale_x, box_loss_scale_y = box_loss_scale_x.cuda(), box_loss_scale_y.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            pred_boxes_for_ciou = pred_boxes_for_ciou.cuda()
            t_box = t_box.cuda()

        # 计算框损失缩放系数：小目标权重更大，大目标权重更小
        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y
        # ---------------------------------------------------------------#
        #   计算预测结果和真实结果的CIOU
        # ----------------------------------------------------------------#
        # 仅对有目标的先验框计算定位损失，并乘以缩放系数
        ciou = (1 - box_ciou(pred_boxes_for_ciou[mask.bool()], t_box[mask.bool()])) * box_loss_scale[mask.bool()]
        loss_loc = torch.sum(ciou)

        # 计算置信度的loss：分有目标和无目标两部分
        loss_conf = torch.sum(BCELoss(conf, mask) * mask) + \
                    torch.sum(BCELoss(conf, mask) * noobj_mask)

        # 计算类别损失：仅对有目标的先验框计算，使用平滑标签
        loss_cls = torch.sum(
            BCELoss(pred_cls[mask == 1], smooth_labels(tcls[mask == 1], self.label_smooth, self.num_classes)))

        # 总损失：加权求和三个部分的损失
        loss = loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc

        # 损失归一化：避免批次正样本数量差异影响训练
        if self.normalize:
            num_pos = torch.sum(mask)
            num_pos = torch.max(num_pos, torch.ones_like(num_pos))  # 防止除以0
        else:
            num_pos = bs / 3

        return loss, num_pos  # 返回总损失和归一化系数

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        bs = len(target)
        # -------------------------------------------------------#
        #   获得当前特征层先验框所属的编号，方便后面对先验框筛选
        # -------------------------------------------------------#
        # 匹配当前特征层对应的先验框组（3个一组）
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        # 先验框组的偏移量，用于转换为组内索引
        subtract_index = [0, 3, 6][self.feature_length.index(in_w)]
        # -------------------------------------------------------#
        #   创建全是0或者全是1的阵列
        # -------------------------------------------------------#
        mask = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)  # 有目标掩码
        noobj_mask = torch.ones(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)  # 无目标掩码

        # 真实框中心x/y、宽高w/h的标签（特征层尺度）
        tx = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        t_box = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, 4, requires_grad=False)  # 整合后的真实框
        tconf = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)  # 置信度真实标签
        tcls = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, self.num_classes, requires_grad=False)  # 类别真实标签

        # 框损失缩放系数（原始图像尺度的宽高比）
        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        # 遍历每个批次的图像
        for b in range(bs):
            if len(target[b]) == 0:  # 该图像无目标，跳过
                continue
            # -------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            # -------------------------------------------------------#
            gxs = target[b][:, 0:1] * in_w  # 真实框x中心（特征层尺度）
            gys = target[b][:, 1:2] * in_h  # 真实框y中心（特征层尺度）

            # -------------------------------------------------------#
            #   计算出正样本相对于特征层的宽高
            # -------------------------------------------------------#
            gws = target[b][:, 2:3] * in_w  # 真实框宽度（特征层尺度）
            ghs = target[b][:, 3:4] * in_h  # 真实框高度（特征层尺度）

            # -------------------------------------------------------#
            #   计算出正样本属于特征层的哪个特征点
            # -------------------------------------------------------#
            gis = torch.floor(gxs)  # 真实框中心对应的特征点x坐标
            gjs = torch.floor(gys)  # 真实框中心对应的特征点y坐标

            # -------------------------------------------------------#
            #   将真实框转换一个形式
            #   num_true_box, 4
            # -------------------------------------------------------#
            # 构造(0,0,gw,gh)格式的真实框，用于计算与先验框的IOU
            gt_box = torch.FloatTensor(torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs], 1))

            # -------------------------------------------------------#
            #   将先验框转换一个形式
            #   9, 4
            # -------------------------------------------------------#
            # 构造(0,0,aw,ah)格式的先验框，用于计算与真实框的IOU
            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros((self.num_anchors, 2)), torch.FloatTensor(anchors)), 1))
            # -------------------------------------------------------#
            #   计算交并比
            #   num_true_box, 9
            # -------------------------------------------------------#
            anch_ious = jaccard(gt_box, anchor_shapes)  # 每个真实框与9个先验框的IOU

            # -------------------------------------------------------#
            #   计算重合度最大的先验框是哪个
            #   num_true_box,
            # -------------------------------------------------------#
            best_ns = torch.argmax(anch_ious, dim=-1)  # 每个真实框匹配的最佳先验框索引
            # 遍历每个真实框
            for i, best_n in enumerate(best_ns):
                if best_n not in anchor_index:  # 最佳先验框不属于当前特征层，跳过
                    continue
                # -------------------------------------------------------------#
                #   取出各类坐标：
                #   gi和gj代表的是真实框对应的特征点的x轴y轴坐标
                #   gx和gy代表真实框的x轴和y轴坐标
                #   gw和gh代表真实框的宽和高
                # -------------------------------------------------------------#
                gi = gis[i].long()
                gj = gjs[i].long()
                gx = gxs[i]
                gy = gys[i]
                gw = gws[i]
                gh = ghs[i]
                if (gj < in_h) and (gi < in_w):  # 确保特征点坐标在特征层范围内
                    best_n = best_n - subtract_index  # 转换为组内索引（0-2）
                    # ----------------------------------------#
                    #   noobj_mask代表无目标的特征点
                    # ----------------------------------------#
                    noobj_mask[b, best_n, gj, gi] = 0  # 取消无目标标记
                    # ----------------------------------------#
                    #   mask代表有目标的特征点
                    # ----------------------------------------#
                    mask[b, best_n, gj, gi] = 1  # 标记为有目标
                    # ----------------------------------------#
                    #   tx、ty代表中心的真实值
                    # ----------------------------------------#
                    tx[b, best_n, gj, gi] = gx
                    ty[b, best_n, gj, gi] = gy
                    # ----------------------------------------#
                    #   tw、th代表宽高的真实值
                    # ----------------------------------------#
                    tw[b, best_n, gj, gi] = gw
                    th[b, best_n, gj, gi] = gh
                    # ----------------------------------------#
                    #   用于获得xywh的比例
                    #   大目标loss权重小，小目标loss权重大
                    # ----------------------------------------#
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][i, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][i, 3]
                    # ----------------------------------------#
                    #   tconf代表物体置信度
                    # ----------------------------------------#
                    tconf[b, best_n, gj, gi] = 1  # 置信度真实值为1
                    # ----------------------------------------#
                    #   tcls代表种类置信度
                    # ----------------------------------------#
                    tcls[b, best_n, gj, gi, target[b][i, 4].long()] = 1  # 对应类别真实值为1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue
        # 整合真实框的xywh
        t_box[..., 0] = tx
        t_box[..., 1] = ty
        t_box[..., 2] = tw
        t_box[..., 3] = th
        return mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self, prediction, target, scaled_anchors, in_w, in_h, noobj_mask):
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        bs = len(target)
        # -------------------------------------------------------#
        #   获得当前特征层先验框所属的编号，方便后面对先验框筛选
        # -------------------------------------------------------#
        # 获取当前特征层对应的先验框组
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]  # 当前特征层的先验框

        # 先验框的中心位置的调整参数（sigmoid激活限制在0-1之间）
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数（未激活，后续用exp转换为正数）
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        # 根据设备类型选择张量类型
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        # 生成特征层网格的x坐标
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * self.num_anchors / 3), 1, 1).view(x.shape).type(FloatTensor)
        # 生成特征层网格的y坐标
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * self.num_anchors / 3), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

        # 调整先验框宽高的形状，匹配预测结果
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # -------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        # -------------------------------------------------------#
        # 解码预测框，得到特征层尺度的xywh
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x + grid_x  # 预测框x中心
        pred_boxes[..., 1] = y + grid_y  # 预测框y中心
        pred_boxes[..., 2] = torch.exp(w) * anchor_w  # 预测框宽度
        pred_boxes[..., 3] = torch.exp(h) * anchor_h  # 预测框高度
        # 遍历每个批次的图像
        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            # -------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            # -------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)  # 展平为先验框数量x4
            # -------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            # -------------------------------------------------------#
            if len(target[i]) > 0:  # 该图像有目标
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh], -1)).type(FloatTensor)

                # -------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                # -------------------------------------------------------#
                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)
                # -------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                # -------------------------------------------------------#
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[i].size()[:3])
                # 忽略IOU大于阈值的负样本
                noobj_mask[i][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes


def weights_init(net, init_type='normal', init_gain=0.02):
    # 网络权重初始化函数：初始化卷积层和批归一化层
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)  # 递归应用初始化函数到网络所有层


class LossHistory():
    def __init__(self, log_dir):
        # 损失历史记录类：用于保存和可视化训练/验证损失
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses = []  # 训练损失列表
        self.val_loss = []  # 验证损失列表

        os.makedirs(self.save_path)  # 创建损失保存目录

    def append_loss(self, loss, val_loss):
        # 添加单次epoch的损失，并保存到文件和绘制图像
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        # 保存训练损失
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        # 保存验证损失
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()  # 绘制损失曲线

    def loss_plot(self):
        # 绘制训练/验证损失曲线
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            # 绘制平滑后的损失曲线，便于观察趋势
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        # 保存损失图像
        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))