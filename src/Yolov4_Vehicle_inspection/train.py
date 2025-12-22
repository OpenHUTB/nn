# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # 用于显示训练进度条的工具库

from nets.yolo4 import YoloBody  # 导入YOLOv4的网络结构定义
from nets.yolo_training import LossHistory, YOLOLoss, weights_init  # 导入训练相关的损失函数、历史记录保存等工具
from utils.dataloader import YoloDataset, yolo_dataset_collate  # 导入自定义的数据加载器和数据拼接函数


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    # 读取类别名称文件，该文件每行对应一个类别名称
    with open(classes_path) as f:
        class_names = f.readlines()
    # 去除每个类别名称前后的空白字符（如换行符、空格）
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    # 读取先验框文件，该文件中一行存储所有先验框的宽高（格式为：w1,h1,w2,h2,...w9,h9）
    with open(anchors_path) as f:
        anchors = f.readline()
    # 将读取到的字符串转换为浮点型列表
    anchors = [float(x) for x in anchors.split(',')]
    # 重塑为三维数组：[3,3,2]，其中3个尺度层，每个尺度层3个先验框，每个先验框2个值（宽、高）
    # [::-1,:,:] 表示对尺度层进行逆序，适配YOLOv4的大、中、小尺度输出顺序
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def get_lr(optimizer):
    # 获取当前优化器的学习率（遍历参数组，返回第一个参数组的学习率）
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    # 全局变量声明，用于Tensorboard的步数计数（避免每次调用函数时重置）
    if Tensorboard:
        global train_tensorboard_step, val_tensorboard_step
    # 初始化训练总损失和验证总损失
    total_loss = 0
    val_loss = 0

    # 将网络设置为训练模式（启用批量归一化、dropout等训练专属层）
    net.train()
    print('Start Train')
    # 使用tqdm创建训练进度条，显示当前epoch信息、实时损失和学习率
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            # 若迭代次数超过当前epoch的总步数，提前终止训练循环
            if iteration >= epoch_size:
                break
            # 从数据加载器中获取当前批次的图像和标签
            images, targets = batch[0], batch[1]
            # torch.no_grad() 上下文管理器：禁用梯度计算，减少显存占用（此处仅用于数据转换，无需梯度）
            with torch.no_grad():
                if cuda:
                    # 将numpy数组转换为torch浮点张量，并移至GPU
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    # 将numpy数组转换为torch浮点张量，保留在CPU
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            # ----------------------#
            #   清零梯度
            # ----------------------#
            # 每次反向传播前必须清零梯度，避免梯度累积
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            # 将图像输入网络，得到3个尺度层的预测输出
            outputs = net(images)
            losses = []
            num_pos_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            # 遍历3个尺度层的输出，分别计算每个尺度的损失
            for i in range(3):
                loss_item, num_pos = yolo_loss(outputs[i], targets)
                losses.append(loss_item)  # 保存每个尺度的损失
                num_pos_all += num_pos  # 累计所有尺度的正样本数量

            # 总损失 = 所有尺度损失之和 / 正样本总数（归一化损失，避免批次大小影响）
            loss = sum(losses) / num_pos_all
            # 累加当前批次的损失到训练总损失
            total_loss += loss.item()

            # ----------------------#
            #   反向传播
            # ----------------------#
            # 根据损失计算梯度（反向传播）
            loss.backward()
            # 根据梯度更新网络参数（优化器步骤）
            optimizer.step()

            # 若启用Tensorboard，记录训练损失（按步骤记录）
            if Tensorboard:
                writer.add_scalar('Train_loss', loss, train_tensorboard_step)
                train_tensorboard_step += 1

            # 更新进度条的实时显示信息：平均训练损失和当前学习率
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            # 进度条前进1步
            pbar.update(1)

    # 将loss写入tensorboard，下面注释的是每个世代保存一次
    # if Tensorboard:
    #     writer.add_scalar('Train_loss', total_loss/(iteration+1), epoch)
    # 将网络设置为验证模式（关闭批量归一化、dropout等训练专属层，使用滑动平均参数）
    net.eval()
    print('Start Validation')
    # 使用tqdm创建验证进度条，显示当前epoch信息和实时验证损失
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            # 若迭代次数超过当前epoch的验证总步数，提前终止验证循环
            if iteration >= epoch_size_val:
                break
            # 从验证数据加载器中获取当前批次的图像和标签
            images_val, targets_val = batch[0], batch[1]

            # torch.no_grad() 上下文管理器：验证阶段无需计算梯度，大幅减少显存占用
            with torch.no_grad():
                if cuda:
                    images_val = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets_val]
                else:
                    images_val = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                # 验证阶段无需更新参数，此处清零梯度仅为代码一致性（无实际作用）
                optimizer.zero_grad()

                # 前向传播获取验证集预测输出
                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                # 遍历3个尺度层，计算验证损失
                for i in range(3):
                    loss_item, num_pos = yolo_loss(outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                # 计算验证总损失（归一化）
                loss = sum(losses) / num_pos_all
                # 累加当前批次验证损失到总验证损失
                val_loss += loss.item()

            # 将loss写入tensorboard, 下面注释的是每一步都写
            # if Tensorboard:
            #     writer.add_scalar('Val_loss', loss, val_tensorboard_step)
            #     val_tensorboard_step += 1
            # 更新验证进度条的实时显示信息：平均验证损失
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    # 若启用Tensorboard，按epoch记录验证损失（便于观察长期趋势）
    if Tensorboard:
        writer.add_scalar('Val_loss', val_loss / (epoch_size_val + 1), epoch)
    # 将当前epoch的训练损失和验证损失保存到损失历史记录中（用于绘制损失曲线）
    loss_history.append_loss(total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1))
    print('Finish Validation')
    # 打印当前epoch的训练信息
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    # 保存当前epoch的网络权重
    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(),
               'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % ((epoch + 1), total_loss / (epoch_size + 1),
                                                                 val_loss / (epoch_size_val + 1)))


# ----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Tensorboard
    # -------------------------------#
    Tensorboard = False  # 关闭Tensorboard可视化（如需启用，改为True）
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True  # 启用GPU加速训练（无GPU需改为False）
    # ------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    # ------------------------------------------------------#
    normalize = False  # 不使用批次大小归一化损失，使用正样本数量归一化
    # -------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    # -------------------------------#
    input_shape = (320, 320)  # 网络输入图像尺寸为320x320（宽x高，需为32的整数倍）
    # ----------------------------------------------------#
    #   classes和anchor的路径，非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    # ----------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'  # 先验框配置文件路径
    classes_path = 'model_data/voc_classes.txt'  # 类别名称配置文件路径
    # ------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    # ------------------------------------------------------#
    mosaic = False  # 关闭马赛克数据增强（训练不稳定，如需提升泛化能力可改为True）
    Cosine_lr = False  # 关闭余弦退火学习率调度，使用步长衰减
    smoooth_label = 0  # 关闭标签平滑（值大于0时启用，如0.01）

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names = get_classes(classes_path)  # 加载类别名称列表
    anchors = get_anchors(anchors_path)  # 加载先验框数组
    num_classes = len(class_names)  # 获取类别数量

    # ------------------------------------------------------#
    #   创建yolo模型
    #   训练前一定要修改classes_path和对应的txt文件
    # ------------------------------------------------------#
    # 初始化YOLOv4网络结构：参数1为每个尺度层的先验框数量，参数2为类别数量
    model = YoloBody(len(anchors[0]), num_classes)
    # 对网络权重进行初始化（使用自定义的权重初始化策略）
    weights_init(model)

    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    # ------------------------------------------------------#
    model_path = "model_data/yolo4_weights.pth"  # 预训练权重文件路径
    print('Loading weights into state dict...')
    # 自动检测设备（有GPU用GPU，无GPU用CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取当前模型的权重字典
    model_dict = model.state_dict()
    # 加载预训练权重，并映射到当前设备
    pretrained_dict = torch.load(model_path, map_location=device)
    # 筛选预训练权重中与当前模型形状匹配的层（避免类别数量不一致导致的报错）
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # 用预训练权重更新当前模型的权重字典
    model_dict.update(pretrained_dict)
    # 将更新后的权重字典加载到模型中
    model.load_state_dict(model_dict)
    print('Finished!')
    # torch.save(model.state_dict(),"logs/test.pth",_use_new_zipfile_serialization=False )

    # 将模型设置为训练模式（启用训练专属层）
    net = model.train()

    if Cuda:
        # net = torch.nn.DataParallel(model)  # 多GPU训练（注释掉，使用单GPU训练）
        cudnn.benchmark = True  # 启用cudnn基准测试，加速GPU训练
        net = net.cuda()  # 将模型移至GPU

    # 初始化YOLOv4损失函数：参数依次为先验框、类别数、输入尺寸、标签平滑值、是否使用GPU、是否归一化
    yolo_loss = YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, (input_shape[1], input_shape[0]), smoooth_label,
                         Cuda, normalize)
    # 初始化损失历史记录器：保存训练/验证损失到logs目录，并绘制损失曲线
    loss_history = LossHistory("logs/")

    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    annotation_path = '2007_train.txt'  # 训练数据集标注文件路径（每行对应一张图片的路径和标签）
    # ----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    # ----------------------------------------------------------------------#
    val_split = 0.1  # 验证集占总数据集的比例（10%）
    # 读取标注文件中的所有行（每张图片的标注信息）
    with open(annotation_path, encoding="utf-8") as f:
        lines = f.readlines()
    # 设置随机种子，保证数据集划分的可重复性
    np.random.seed(10101)
    # 打乱标注数据的顺序
    np.random.shuffle(lines)
    # 重置随机种子，恢复默认随机状态
    np.random.seed(None)
    # 计算验证集的样本数量
    num_val = int(len(lines) * val_split)
    # 计算训练集的样本数量
    num_train = len(lines) - num_val

    # 若启用Tensorboard，初始化SummaryWriter
    if Tensorboard:
        from tensorboardX import SummaryWriter

        # 创建日志写入器：日志保存到logs目录，每60秒刷新一次
        writer = SummaryWriter(log_dir='logs', flush_secs=60)
        if Cuda:
            # 创建随机张量作为网络输入，用于绘制网络计算图
            graph_inputs = torch.randn(1, 3, input_shape[0], input_shape[1]).type(torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.randn(1, 3, input_shape[0], input_shape[1]).type(torch.FloatTensor)
        # 将网络计算图写入Tensorboard
        writer.add_graph(model, graph_inputs)
        # 初始化训练和验证的步数计数器
        train_tensorboard_step = 1
        val_tensorboard_step = 1

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    # 第一阶段：冻结主干网络训练（仅训练头部网络，加快训练速度，防止预训练权重被破坏）
    if True:
        lr = 1e-3  # 冻结阶段学习率（1e-3）
        Batch_size = 4  # 冻结阶段批次大小（显存不足可减小）
        Init_Epoch = 0  # 起始训练世代
        Freeze_Epoch = 50  # 冻结训练的结束世代（共50个epoch）

        # ----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        # ----------------------------------------------------------------------------#
        # 初始化Adam优化器（仅优化当前可训练的网络参数）
        optimizer = optim.Adam(net.parameters(), lr)
        # 根据配置选择学习率调度器
        if Cosine_lr:
            # 余弦退火调度器：学习率随epoch呈余弦曲线下降
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            # 步长衰减调度器：每经过1个epoch，学习率乘以0.92
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        # 创建训练数据集对象：传入训练集标注、输入尺寸、是否使用mosaic增强、是否为训练模式
        train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
        # 创建验证数据集对象：验证集不使用mosaic增强
        val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
        # 创建训练数据加载器：打乱数据、指定批次大小、开启多线程加载、丢弃最后不完整批次
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        # 创建验证数据加载器：配置与训练集一致
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        # 计算每个epoch的训练步数（训练样本数 // 批次大小）
        epoch_size = num_train // Batch_size
        # 计算每个epoch的验证步数（验证样本数 // 批次大小）
        epoch_size_val = num_val // Batch_size

        # 检查数据集大小：若训练/验证步数为0，说明数据集过小，无法训练
        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        # 冻结主干网络的所有参数（设置requires_grad=False，禁止梯度更新）
        for param in model.backbone.parameters():
            param.requires_grad = False

        # 遍历冻结训练的每个epoch，执行训练和验证
        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            # 更新学习率（根据调度器调整）
            lr_scheduler.step()

    # 第二阶段：解冻主干网络训练（训练整个网络，微调权重，提升模型精度）
    if True:
        lr = 1e-4  # 解冻阶段学习率（比冻结阶段小10倍，避免权重震荡）
        Batch_size = 2  # 解冻阶段批次大小（显存占用更高，需减小）
        Freeze_Epoch = 50  # 解冻训练的起始世代
        Unfreeze_Epoch = 100  # 解冻训练的结束世代（总训练100个epoch）

        # ----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        # ----------------------------------------------------------------------------#
        # 初始化Adam优化器（优化所有网络参数）
        optimizer = optim.Adam(net.parameters(), lr)
        # 根据配置选择学习率调度器
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        # 重新创建训练/验证数据集（配置与冻结阶段一致）
        train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
        val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
        # 重新创建训练/验证数据加载器（批次大小减小）
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        # 重新计算每个epoch的训练/验证步数
        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        # 检查数据集大小
        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        # 解冻主干网络的所有参数（设置requires_grad=True，允许梯度更新）
        for param in model.backbone.parameters():
            param.requires_grad = True

        # 遍历解冻训练的每个epoch，执行训练和验证
        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            # 更新学习率
            lr_scheduler.step()
