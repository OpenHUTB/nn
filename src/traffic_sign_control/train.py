"""
YOLO模型训练脚本
基于采集的数据集训练自定义模型
"""
import os
import shutil
import argparse
import torch
from ultralytics import YOLO


def split_dataset(dataset_dir, val_ratio=0.2):
    """
    将数据集划分为训练集和验证集

    Args:
        dataset_dir: 数据集根目录
        val_ratio: 验证集比例 (默认: 0.2)
    """
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    train_images_dir = os.path.join(dataset_dir, 'train', 'images')
    train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
    val_images_dir = os.path.join(dataset_dir, 'val', 'images')
    val_labels_dir = os.path.join(dataset_dir, 'val', 'labels')

    # 如果已经划分过，跳过
    if os.path.exists(train_images_dir) and os.path.exists(val_images_dir):
        train_count = len([f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.png'))])
        val_count = len([f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png'))])
        print(f"数据集已划分: 训练集 {train_count} 张, 验证集 {val_count} 张")
        return True

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

    if len(image_files) == 0:
        print("错误: 没有找到图像文件")
        return False

    # 创建目录
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(d, exist_ok=True)

    # 随机划分
    import random
    random.seed(42)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - val_ratio))

    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # 移动文件
    for f in train_files:
        shutil.copy2(os.path.join(images_dir, f), os.path.join(train_images_dir, f))
        label_name = os.path.splitext(f)[0] + '.txt'
        label_src = os.path.join(labels_dir, label_name)
        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(train_labels_dir, label_name))

    for f in val_files:
        shutil.copy2(os.path.join(images_dir, f), os.path.join(val_images_dir, f))
        label_name = os.path.splitext(f)[0] + '.txt'
        label_src = os.path.join(labels_dir, label_name)
        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(val_labels_dir, label_name))

    print(f"数据集划分完成: 训练集 {len(train_files)} 张, 验证集 {len(val_files)} 张")
    return True


def train_model(data_yaml, epochs=100, batch_size=16, img_size=640, model_name='yolov8n.pt', project='runs/detect', name='traffic_sign'):
    """
    训练YOLO模型

    Args:
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像尺寸
        model_name: 基础模型名称
        project: 训练结果保存目录
        name: 训练实验名称
    """
    # 检查配置文件
    if not os.path.exists(data_yaml):
        print(f"错误: 配置文件不存在: {data_yaml}")
        print("提示: 请先运行 generate_dataset_yaml.py 生成配置文件")
        return None

    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"训练设备: {device}")
    if device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # 加载基础模型
    print(f"\n加载基础模型: {model_name}")
    model = YOLO(model_name)

    # 开始训练
    print(f"\n{'='*50}")
    print(f"开始训练...")
    print(f"  - 数据集配置: {data_yaml}")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 图像尺寸: {img_size}")
    print(f"  - 实验目录: {project}/{name}")
    print(f"{'='*50}\n")

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            plots=True,
            save=True,
            save_period=10,
            verbose=True
        )

        best_path = os.path.join(project, name, 'weights', 'best.pt')
        last_path = os.path.join(project, name, 'weights', 'last.pt')
        results_path = os.path.join(project, name, 'results.csv')

        print(f"\n{'='*50}")
        print(f"训练完成!")
        print(f"{'='*50}")
        print(f"最佳模型: {best_path}")
        print(f"最末模型: {last_path}")
        print(f"训练结果: {results_path}")
        print(f"{'='*50}\n")

        # 确保权重文件存在
        if os.path.exists(best_path):
            size_mb = os.path.getsize(best_path) / (1024*1024)
            print(f"✓ best.pt 已保存 ({size_mb:.1f} MB)")
        if os.path.exists(last_path):
            size_mb = os.path.getsize(last_path) / (1024*1024)
            print(f"✓ last.pt 已保存 ({size_mb:.1f} MB)")

        return best_path if os.path.exists(best_path) else last_path

    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO模型训练脚本")
    parser.add_argument('--data', type=str, default='dataset.yaml',
                       help='数据集配置文件 (默认: dataset.yaml)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--batch', type=int, default=16,
                       help='批次大小 (默认: 16)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='图像尺寸 (默认: 640)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='基础模型路径 (默认: yolov8n.pt)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='训练结果保存目录 (默认: runs/detect)')
    parser.add_argument('--name', type=str, default='traffic_sign',
                       help='实验名称 (默认: traffic_sign)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='验证集比例 (默认: 0.2)')

    args = parser.parse_args()

    # 从yaml中读取数据集目录并划分
    import yaml
    if os.path.exists(args.data):
        with open(args.data, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        dataset_dir = cfg.get('path', 'dataset')
        split_dataset(dataset_dir, args.val_ratio)

    # 开始训练
    best_weights = train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        model_name=args.model,
        project=args.project,
        name=args.name
    )

    if best_weights:
        print(f"\n训练成功! 模型路径: {best_weights}")
        print(f"运行推理: python main.py --model {best_weights}")
    else:
        print("\n训练失败!")
        exit(1)
