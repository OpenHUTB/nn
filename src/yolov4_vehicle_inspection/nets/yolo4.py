# 导入有序字典，用于按顺序构建神经网络层（保证层执行顺序）
from collections import OrderedDict

# 导入PyTorch核心库，用于构建神经网络
import torch
# 导入PyTorch神经网络模块，包含各类层定义
import torch.nn as nn

# 导入自定义的CSPDarknet53主干网络（特征提取基础）
from nets.CSPdarknet import darknet53


# 定义基础卷积模块：Conv2d + BatchNorm2d + LeakyReLU
# filter_in: 输入通道数 | filter_out: 输出通道数 | kernel_size: 卷积核尺寸 | stride: 卷积步长
def conv2d(filter_in, filter_out, kernel_size, stride=1):
    # 计算padding值，实现same padding（卷积后特征图尺寸不变）
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    # 使用OrderedDict构建有序的卷积+BN+激活组合层
    return nn.Sequential(OrderedDict([
        # 卷积层：无偏置（BN层已包含偏置参数，避免冗余）
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        # 批量归一化层：加速训练收敛，降低过拟合风险
        ("bn", nn.BatchNorm2d(filter_out)),
        # LeakyReLU激活：缓解梯度消失，负区间保留小梯度（斜率0.1）
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        # 创建多个最大池化层，池化核5/9/13，步长1，padding=池化核//2（保证池化后尺寸不变）
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        # 逆序遍历池化层，对输入特征图做多尺度池化（13→9→5）
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        # 将多尺度池化结果与原始特征图在通道维度拼接（dim=1），融合多尺度特征
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        # 上采样模块：先1x1卷积调整通道数，再最近邻上采样放大2倍（保持特征清晰）
        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),  # 1x1卷积降维，减少计算量
            nn.Upsample(scale_factor=2, mode='nearest')  # 最近邻上采样，尺寸放大2倍
        )

    def forward(self, x,):
        # 执行上采样操作
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
# filters_list: [中间通道数, 3x3卷积通道数] | in_filters: 输入通道数
def make_three_conv(filters_list, in_filters):
    # 三次卷积结构：1x1→3x3→1x1，提取特征并调整通道数
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),    # 1x1卷积降维
        conv2d(filters_list[0], filters_list[1], 3),# 3x3卷积扩大感受野，提取特征
        conv2d(filters_list[1], filters_list[0], 1),# 1x1卷积再次降维
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
# filters_list: [中间通道数, 3x3卷积通道数] | in_filters: 输入通道数
def make_five_conv(filters_list, in_filters):
    # 五次卷积结构：1x1→3x3→1x1→3x3→1x1，强化特征提取能力
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
# filters_list: [3x3卷积通道数, 最终输出通道数] | in_filters: 输入通道数
def yolo_head(filters_list, in_filters):
    # YOLO检测头：3x3卷积强化特征 + 1x1卷积输出预测结果
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),  # 3x3卷积提升特征表达
        nn.Conv2d(filters_list[0], filters_list[1], 1),  # 1x1卷积输出最终检测结果（锚框数*(5+类别数)）
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        # 初始化CSPDarknet53主干网络，用于提取基础特征
        self.backbone = darknet53(None)

        # 处理13x13x1024特征层：三次卷积（1024→512→1024→512）
        self.conv1 = make_three_conv([512,1024],1024)
        # 接入SPP模块，融合多尺度特征
        self.SPP = SpatialPyramidPooling()
        # 再次三次卷积处理SPP输出（2048→512→1024→512）
        self.conv2 = make_three_conv([512,1024],2048)

        # 上采样模块1：512→256，尺寸从13x13→26x26
        self.upsample1 = Upsample(512,256)
        # 调整26x26x512特征层通道数：512→256
        self.conv_for_P4 = conv2d(512,256,1)
        # 五次卷积融合26x26特征（512→256→512→256→512→256）
        self.make_five_conv1 = make_five_conv([256, 512],512)

        # 上采样模块2：256→128，尺寸从26x26→52x52
        self.upsample2 = Upsample(256,128)
        # 调整52x52x256特征层通道数：256→128
        self.conv_for_P3 = conv2d(256,128,1)
        # 五次卷积融合52x52特征（256→128→256→128→256→128）
        self.make_five_conv2 = make_five_conv([128, 256],256)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        # 计算52x52特征层输出通道数：锚框数*(坐标4+置信度1+类别数)
        final_out_filter2 = num_anchors * (5 + num_classes)
        # 52x52特征层的YOLO检测头（检测小目标）
        self.yolo_head3 = yolo_head([256, final_out_filter2],128)

        # 下采样模块1：128→256，步长2，尺寸从52x52→26x26
        self.down_sample1 = conv2d(128,256,3,stride=2)
        # 五次卷积再次融合26x26特征（512→256→512→256→512→256）
        self.make_five_conv3 = make_five_conv([256, 512],512)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        # 计算26x26特征层输出通道数
        final_out_filter1 =  num_anchors * (5 + num_classes)
        # 26x26特征层的YOLO检测头（检测中目标）
        self.yolo_head2 = yolo_head([512, final_out_filter1],256)

        # 下采样模块2：256→512，步长2，尺寸从26x26→13x13
        self.down_sample2 = conv2d(256,512,3,stride=2)
        # 五次卷积再次融合13x13特征（1024→512→1024→512→1024→512）
        self.make_five_conv4 = make_five_conv([512, 1024],1024)

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 计算13x13特征层输出通道数
        final_out_filter0 =  num_anchors * (5 + num_classes)
        # 13x13特征层的YOLO检测头（检测大目标）
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512)


    def forward(self, x):
        #  backbone
        # 前向传播通过主干网络，获取三个有效特征层
        # x2: 52x52x256 | x1:26x26x512 | x0:13x13x1024
        x2, x1, x0 = self.backbone(x)

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048
        # 处理13x13x1024特征层：三次卷积→SPP→三次卷积
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.conv2(P5)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        # 上采样P5并与x1（26x26x512）融合
        P5_upsample = self.upsample1(P5)
        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(x1)
        # 26,26,256 + 26,26,256 -> 26,26,512
        # P4 = torch.cat([P4,P5_upsample],axis=1)
        # 通道维度拼接上采样后的P5和调整后的x1
        P4 = torch.cat([P4, P5_upsample], dim=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        # 五次卷积融合特征
        P4 = self.make_five_conv1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        # 上采样P4并与x2（52x52x256）融合
        P4_upsample = self.upsample2(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(x2)
        # 52,52,128 + 52,52,128 -> 52,52,256
        # P3 = torch.cat([P3,P4_upsample],axis=1)
        # 通道维度拼接上采样后的P4和调整后的x2
        P3 = torch.cat([P3, P4_upsample], dim=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        # 五次卷积融合特征
        P3 = self.make_five_conv2(P3)

        # 52,52,128 -> 26,26,256
        # 下采样P3，与P4融合
        P3_downsample = self.down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        # P4 = torch.cat([P3_downsample,P4],axis=1)
        # 通道维度拼接下采样后的P3和原P4
        P4 = torch.cat([P3_downsample, P4], dim=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        # 五次卷积融合特征
        P4 = self.make_five_conv3(P4)

        # 26,26,256 -> 13,13,512
        # 下采样P4，与P5融合
        P4_downsample = self.down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        # P5 = torch.cat([P4_downsample,P5],axis=1)
        # 通道维度拼接下采样后的P4和原P5
        P5 = torch.cat([P4_downsample, P5], dim=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        # 五次卷积融合特征
        P5 = self.make_five_conv4(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        # 52x52特征层输出检测结果（小目标）
        out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        # 26x26特征层输出检测结果（中目标）
        out1 = self.yolo_head2(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        # 13x13特征层输出检测结果（大目标）
        out0 = self.yolo_head1(P5)

        # 返回三个尺度的检测结果（13x13、26x26、52x52）
        return out0, out1, out2