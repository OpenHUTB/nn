# 导入相机内参变换相关模块：transform_img(图像变换函数)、eon_intrinsics(EON相机内参)
from common.transformations.camera import transform_img, eon_intrinsics
# 导入模型输入图像的内参配置（medmodel_intrinsics）
from common.transformations.model import medmodel_intrinsics
# 导入数值计算库
import numpy as np
# 导入进度条库（虽定义但未实际使用）
from tqdm import tqdm
# 导入matplotlib绘图库（用于可视化车道线/路径）
import matplotlib
import matplotlib.pyplot as plt
# 导入车道线坐标变换函数
from common.lanes_image_space import transform_points
# 导入系统路径模块
import os
# 导入keras模型加载函数
from tensorflow.keras.models import load_model
# 导入模型输出解析器（解析supercombo预测结果）
from common.tools.lib.parser import parser
# 导入OpenCV库（用于视频读取、图像格式转换）
import cv2
# 导入系统参数模块（用于读取命令行参数）
import sys

# matplotlib.use('Agg')  # 注释：非交互式后端（如无GUI环境使用），当前启用交互式可视化故注释

# ===================== 初始化参数 =====================
# 从命令行参数获取视频文件路径（运行命令：python xxx.py 视频文件路径）
camerafile = sys.argv[1]
# 加载预训练的supercombo模型（车道线/路径预测模型）
supercombo = load_model('models/supercombo.keras')

# 以下常量定义后未实际使用，保留原代码仅添加注释
MAX_DISTANCE = 140.  # 最大检测距离（米）
LANE_OFFSET = 1.8  # 车道偏移量（米）
MAX_REL_V = 10.  # 最大相对速度（m/s）
LEAD_X_SCALE = 10  # 前车X轴缩放系数
LEAD_Y_SCALE = 10  # 前车Y轴缩放系数


# ===================== 核心函数定义 =====================
def frames_to_tensor(frames):
    """
    将图像帧转换为模型输入张量（适配supercombo模型的输入格式）
    参数：
        frames: 输入图像帧数组，shape=(帧数, 高度, 宽度)，格式为YUV420
    返回：
        in_img1: 模型输入张量，shape=(帧数, 6, H//2, W//2)，6通道分别对应YUV拆解后的不同采样区域
    """
    # 计算图像有效高度（基于YUV420格式的尺寸规则）
    H = (frames.shape[1] * 2) // 3
    W = frames.shape[2]
    # 初始化模型输入张量（帧数, 6通道, 高度//2, 宽度//2）
    in_img1 = np.zeros((frames.shape[0], 6, H // 2, W // 2), dtype=np.uint8)

    # 拆分YUV420格式的Y通道（亮度）：4个不同采样区域
    in_img1[:, 0] = frames[
        :, 0:H:2, 0::2]  # Y通道 - 偶数行、偶数列
    in_img1[:, 1] = frames[
        :, 1:H:2, 0::2]  # Y通道 - 奇数行、偶数列
    in_img1[:, 2] = frames[
        :, 0:H:2, 1::2]  # Y通道 - 偶数行、奇数列
    in_img1[:, 3] = frames[
        :, 1:H:2, 1::2]  # Y通道 - 奇数行、奇数列
    # 拆分U通道（色度）：reshape适配模型输入尺寸
    in_img1[:, 4] = frames[:, H:H + H // 4].reshape((-1, H // 2, W // 2))
    # 拆分V通道（色度）：reshape适配模型输入尺寸
    in_img1[:, 5] = frames[:, H + H // 4:H + H // 2].reshape((-1, H // 2, W // 2))
    return in_img1


# ===================== 全局变量初始化 =====================
# 模型输入图像缓存：存储前一帧和当前帧（2帧），尺寸(2, 384, 512)，格式uint8
imgs_med_model = np.zeros((2, 384, 512), dtype=np.uint8)
# 模型状态反馈变量：shape=(1,512)，用于循环喂回模型（supercombo是时序模型）
state = np.zeros((1, 512))
# 驾驶意图变量：shape=(1,8)，默认无特定意图（全0）
desire = np.zeros((1, 8))

# 打开视频文件（从命令行参数传入）
cap = cv2.VideoCapture(camerafile)

# 初始化车道线/路径的X轴基准坐标：0到192的等间距数组（共192个点）
x_left = x_right = x_path = np.linspace(0, 192, 192)
# 读取视频第一帧（作为前一帧）
(ret, previous_frame) = cap.read()
if not ret:  # 若读取失败，直接退出程序
    exit()
else:
    # 将BGR格式（OpenCV默认）转换为YUV420格式（模型输入要求）
    img_yuv = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2YUV_I420)
    # 图像内参变换：从EON相机内参转换为模型要求的medmodel内参，输出尺寸(512,256)
    imgs_med_model[0] = transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                      output_size=(512, 256))

# ===================== 主循环：逐帧处理视频 =====================
while True:
    # 清空当前绘图窗口（避免帧叠加）
    plt.clf()
    # 设置绘图标题
    plt.title("lanes and path")
    # 设置X轴范围（适配视频帧尺寸）
    plt.xlim(0, 1200)
    # 设置Y轴范围（反转Y轴，适配图像坐标系：原点在左上角）
    plt.ylim(800, 0)

    # 读取当前帧
    (ret, current_frame) = cap.read()
    if not ret:  # 若读取失败（视频结束），退出循环
        break

    # 复制当前帧（避免原数据被修改）
    frame = current_frame.copy()
    # 当前帧转换为YUV420格式
    img_yuv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YUV_I420)
    # 当前帧执行内参变换，存入缓存的第2个位置
    imgs_med_model[1] = transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                      output_size=(512, 256))

    # 将缓存的2帧图像转换为模型输入张量，并归一化到[-1, 1]（模型训练时的输入范围）
    frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32) / 128.0 - 1.0

    # 构造模型输入：[图像张量, 驾驶意图, 状态反馈]
    inputs = [np.vstack(frame_tensors[0:2])[None], desire, state]
    # 模型预测：输出包含车道线、路径、状态等信息
    outs = supercombo.predict(inputs)
    # 解析模型输出：将原始输出转换为易读的字典格式（如"lll"左车道线、"rll"右车道线、"path"路径）
    parsed = parser(outs)

    # 关键：将模型输出的状态反馈喂回下一轮循环（时序模型的核心逻辑）
    state = outs[-1]
    # 6自由度标定参数（当前代码未使用，仅保留）
    pose = outs[-2]

    # 将OpenCV的BGR格式转换为matplotlib的RGB格式（用于绘图显示）
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 绘制原始视频帧作为背景
    plt.imshow(frame)

    # 转换左车道线坐标：从模型输出空间转换到图像像素空间
    new_x_left, new_y_left = transform_points(x_left, parsed["lll"][0])
    # 转换右车道线坐标
    new_x_right, new_y_right = transform_points(x_left, parsed["rll"][0])
    # 转换行驶路径坐标
    new_x_path, new_y_path = transform_points(x_left, parsed["path"][0])

    # 绘制左车道线（白色）
    plt.plot(new_x_left, new_y_left, label='transformed', color='w')
    # 绘制右车道线（白色）
    plt.plot(new_x_right, new_y_right, label='transformed', color='w')
    # 绘制行驶路径（绿色）
    plt.plot(new_x_path, new_y_path, label='transformed', color='green')

    # 缓存切换：当前帧变为下一轮的前一帧
    imgs_med_model[0] = imgs_med_model[1]

    # 暂停0.001秒，刷新绘图窗口（实现实时可视化）
    plt.pause(0.001)

    # 按键检测：按下q键退出循环
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# plt.show()  # 注释：主循环内已用plt.pause实时显示，无需最终show