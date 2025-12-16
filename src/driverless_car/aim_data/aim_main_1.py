from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw
import os
import numpy as np
import random
import time
from tqdm import tqdm  # 进度条库

# --------------------------
# 基础配置：提速+可视化优化
# --------------------------
# 解决Matplotlib中文显示问题
plt.rcParams["font.family"] = ["SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
# 切换后端并关闭交互式模式（提速）
plt.switch_backend('TkAgg')
plt.ioff()

# 加载YOLOv8n预训练模型（轻量化，提速）
model = YOLO('yolov8n.pt')

# 无人机场景常见类别（中文映射+颜色区分）
drone_common_classes = {
    0: {'name': '人', 'color': 'red'},
    2: {'name': '汽车', 'color': 'blue'},
    3: {'name': '摩托车', 'color': 'yellow'},
    5: {'name': '公交车', 'color': 'green'},
    4: {'name': '飞机', 'color': 'purple'}
}
cls_ids = list(drone_common_classes.keys())

# --------------------------
# 核心优化：快速生成模拟无人机图像（减少计算量）
# --------------------------
def generate_drone_image_fast(image_size=(640, 480), num_targets=4):
    """
    快速生成模拟无人机航拍图像（简化绘制逻辑，提速）
    :param image_size: 图像尺寸
    :param num_targets: 目标数量
    :return: PIL图像对象、目标框列表
    """
    w, h = image_size
    # 快速创建背景（浅灰色，模拟道路）
    img = Image.new('RGB', (w, h), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)

    # 简化道路标线（只画中心线，减少计算）
    draw.line([(w//2, 0), (w//2, h)], fill=(255, 255, 0), width=5)

    target_boxes = []
    for _ in range(num_targets):
        # 随机选择类别
        cls_id = random.choice(cls_ids)
        cls_info = drone_common_classes[cls_id]
        # 随机位置（避免边缘）
        x = random.randint(50, w - 50)
        y = random.randint(50, h - 50)

        # 简化目标绘制（只保留核心形状，提速）
        if cls_id == 0:  # 人：圆形+短矩形
            r = 10
            x1, y1 = x - r, y - r
            x2, y2 = x + r, y + r
            draw.ellipse([x1, y1, x2, y2], fill=cls_info['color'], outline=(0,0,0), width=1)
            draw.rectangle([x-3, y+r, x+3, y+r+20], fill=cls_info['color'], outline=(0,0,0))
            box = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'cls':cls_id, 'conf':random.uniform(0.8, 0.99)}

        elif cls_id in [2,5]:  # 汽车/公交车：矩形（公交车更大）
            w_t = 60 if cls_id == 2 else 100
            h_t = 30 if cls_id == 2 else 40
            x1, y1 = x - w_t//2, y - h_t//2
            x2, y2 = x + w_t//2, y + h_t//2
            draw.rectangle([x1, y1, x2, y2], fill=cls_info['color'], outline=(0,0,0), width=2)
            box = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'cls':cls_id, 'conf':random.uniform(0.8, 0.99)}

        else:  # 摩托车/飞机：小矩形
            w_t = 30 if cls_id == 3 else 50
            h_t = 20 if cls_id == 3 else 25
            x1, y1 = x - w_t//2, y - h_t//2
            x2, y2 = x + w_t//2, y + h_t//2
            draw.rectangle([x1, y1, x2, y2], fill=cls_info['color'], outline=(0,0,0), width=1)
            box = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'cls':cls_id, 'conf':random.uniform(0.8, 0.99)}

        target_boxes.append(box)

    return img, target_boxes

# --------------------------
# 核心增强：可视化函数（多图对比+动画+颜色区分）
# --------------------------
def visualize_detection_advanced(img, target_boxes, image_name):
    """
    增强版可视化：原图+检测图并列+动态标注+颜色区分
    :param img: PIL图像
    :param target_boxes: 目标框列表
    :param image_name: 图像名称
    """
    img_np = np.array(img)
    # 创建2列1行的子图（原图+检测图）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'无人机航拍图像目标检测 - {image_name}', fontsize=16, fontweight='bold')

    # 显示原图
    ax1.imshow(img_np)
    ax1.set_title('原始航拍图像', fontsize=14)
    ax1.axis('off')

    # 显示检测图（初始为空，用于动画）
    ax2.imshow(img_np)
    ax2.set_title('目标检测结果（动态标注）', fontsize=14)
    ax2.axis('off')

    # 存储绘制的元素（用于动画更新）
    drawn_elements = []

    def update(frame):
        """动画更新函数：逐帧绘制目标框"""
        if frame >= len(target_boxes):
            return drawn_elements
        box = target_boxes[frame]
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        cls_id = box['cls']
        conf = box['conf']
        cls_info = drone_common_classes[cls_id]

        # 绘制边界框（对应类别颜色）
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=cls_info['color'], facecolor='none', alpha=0.8
        )
        rect_obj = ax2.add_patch(rect)
        drawn_elements.append(rect_obj)

        # 绘制文本（带背景）
        text = f'{cls_info["name"]} {conf:.2f}'
        text_obj = ax2.text(
            x1, y1 - 10, text, fontsize=12, color='white',
            bbox=dict(facecolor=cls_info['color'], alpha=0.8, edgecolor='none'),
            fontweight='bold'
        )
        drawn_elements.append(text_obj)
        return drawn_elements

    # 创建动画（每200ms绘制一个目标，共len(target_boxes)帧）
    ani = FuncAnimation(
        fig, update, frames=len(target_boxes)+1, interval=200,
        blit=True, repeat=False
    )

    # 保存结果（提速：降低dpi，简化保存）
    os.makedirs('detect_results', exist_ok=True)
    save_path = os.path.join('detect_results', f'{image_name}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=80, format='png')
    print(f'\n检测结果已保存：{save_path}')

    # 显示图像（block=True，避免多窗口冲突）
    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)
    return fig

# --------------------------
# 新增：生成检测结果汇总图（缩略图展示）
# --------------------------
def generate_summary_plot(image_list, name_list):
    """生成所有检测结果的汇总缩略图"""
    num_images = len(image_list)
    rows = (num_images + 1) // 2  # 向上取整
    cols = 2 if num_images > 1 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
    fig.suptitle('无人机目标检测结果汇总', fontsize=18, fontweight='bold')
    axes = axes.flatten() if num_images > 1 else [axes]

    for idx, (img, name) in enumerate(zip(image_list, name_list)):
        axes[idx].imshow(np.array(img))
        axes[idx].set_title(name, fontsize=12)
        axes[idx].axis('off')
    # 隐藏多余的子图
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    # 保存汇总图
    save_path = os.path.join('detect_results', '检测结果汇总.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=80)
    print(f'\n检测结果汇总图已保存：{save_path}')

    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)

# --------------------------
# 主程序：批量处理+进度条+提速
# --------------------------
def batch_process_drone_images(num_images=3):
    """批量处理模拟无人机图像（带进度条，提速）"""
    print("=== 开始模拟无人机实时航拍目标检测 ===")
    image_list = []
    name_list = []

    # 使用tqdm添加进度条
    for i in tqdm(range(num_images), desc="处理图像", unit="张"):
        # 快速生成图像（减少目标数量，提速）
        img, target_boxes = generate_drone_image_fast(num_targets=random.randint(3, 5))
        image_name = f'航拍视角{i+1}'
        # 增强可视化
        visualize_detection_advanced(img, target_boxes, image_name)
        # 存储图像用于汇总
        image_list.append(img)
        name_list.append(image_name)
        time.sleep(0.1)  # 轻微延迟，避免界面卡顿

    # 生成汇总图
    generate_summary_plot(image_list, name_list)
    print("\n=== 所有无人机图像检测完成！===")

# --------------------------
# 可选：摄像头实时检测（优化版，更快）
# --------------------------
def camera_real_time_detection_fast():
    """摄像头实时检测（优化版，使用OpenCV直接显示，提速）"""
    try:
        import cv2
        print("\n=== 开启无人机摄像头实时检测（按q退出）===")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 降低分辨率，提速

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 快速推理（只检测指定类别，减少计算）
            results = model(frame, classes=cls_ids)
            annotated_frame = results[0].plot()  # YOLO内置绘制，速度快

            # 显示图像
            cv2.imshow('无人机摄像头实时目标检测（640x480）', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except ImportError:
        print("未安装opencv-python，跳过摄像头实时检测！")
    except Exception as e:
        print(f"摄像头启动失败：{e}")

# --------------------------
# 运行主程序
# --------------------------
if __name__ == '__main__':
    # 批量处理模拟图像（默认3张，可调整）
    batch_process_drone_images(num_images=3)

    # 可选：摄像头实时检测（优化版，更快）
    # camera_real_time_detection_fast()