"""
Traffic Light专用训练脚本
只训练traffic light类别，提高检测精度
"""
import os
import shutil
import argparse
import torch
from ultralytics import YOLO


def create_single_class_dataset(dataset_dir, output_dir='traffic_light_dataset'):
    """
    创建只有traffic light的数据集

    Args:
        dataset_dir: 原始数据集目录
        output_dir: 输出目录
    """
    print(f"创建traffic light专用数据集: {output_dir}")

    # 创建目录结构
    for d in ['images', 'semantic', 'labels', 'train', 'val']:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    for d in ['images', 'labels']:
        os.makedirs(os.path.join(train_dir, d), exist_ok=True)
        os.makedirs(os.path.join(val_dir, d), exist_ok=True)

    train_images_dir = os.path.join(train_dir, 'images')
    val_images_dir = os.path.join(val_dir, 'images')

    # 统计数据
    total_count = 0
    traffic_light_count = 0

    # 处理原始数据集
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

    print(f"原始数据集图像数: {len(image_files)}")

    # 随机划分训练集和验证集（80% train, 20% val）
    import random
    random.seed(42)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)

    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")

    # 复制文件（只保留有traffic light标注的图像）
    for i, img_file in enumerate(train_files):
        label_name = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        # 检查是否有traffic light标注并转换类别ID
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read()
                if '1 ' in content:  # traffic light的类别ID是1
                    # 转换类别ID从1到0
                    new_content = content.replace('1 ', '0 ')
                    # 创建新的标注文件
                    new_label_path = os.path.join(train_dir, 'labels', label_name)
                    with open(new_label_path, 'w') as new_f:
                        new_f.write(new_content)
                    # 复制图像
                    shutil.copy2(os.path.join(images_dir, img_file),
                               os.path.join(train_dir, 'images', img_file))
                    traffic_light_count += 1
                    total_count += 1

    for i, img_file in enumerate(val_files):
        label_name = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read()
                if '1 ' in content:
                    # 复制图像
                    shutil.copy2(os.path.join(images_dir, img_file),
                               os.path.join(val_dir, 'images', img_file))
                    # 复制标注
                    shutil.copy2(label_path, os.path.join(val_dir, 'labels', label_name))

    print(f"筛选后数据集:")
    print(f"  - 总traffic light标注数: {traffic_light_count}")
    train_images_dir = os.path.join(train_dir, 'images')
    val_images_dir = os.path.join(val_dir, 'images')
    print(f"  - 训练集: {len([f for f in os.listdir(train_images_dir)])} 张")
    print(f"  - 验证集: {len([f for f in os.listdir(val_images_dir)])} 张")

    # 生成配置文件
    config = {
        'path': './' + output_dir,
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['traffic_light']
    }

    import yaml
    with open(os.path.join(output_dir, 'traffic_light.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n配置文件已生成: {output_dir}/traffic_light.yaml")
    print(f"内容:")
    print(yaml.dump(config, default_flow_style=False, allow_unicode=True))

    return traffic_light_count


def train_traffic_light(data_yaml, epochs=100, batch_size=16, img_size=640, model_name='yolov8n.pt', project='traffic_light_train'):
    """
    训练traffic light专用模型

    Args:
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像尺寸
        model_name: 基础模型名称
        project: 训练结果保存目录
    """
    # 检查配置文件
    if not os.path.exists(data_yaml):
        print(f"错误: 配置文件不存在: {data_yaml}")
        return False

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
    print(f"\n开始训练traffic light专用模型...")
    print(f"  - 数据集配置: {data_yaml}")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 图像尺寸: {img_size}")
    print(f"  - 结果保存: {project}")
    print(f"  - TensorBoard: tensorboard --logdir {project}\n")

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=project,
            name='traffic_light',
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True
        )

        print("\n训练完成!")
        print(f"最佳模型: {project}/traffic_light/weights/best.pt")
        print(f"最末模型: {project}/traffic_light/weights/last.pt")
        print(f"训练结果: {project}/traffic_light/results.csv")

        return True

    except Exception as e:
        print(f"\n训练出错: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Light专用训练脚本")
    parser.add_argument('--dataset', type=str, default='final_dataset',
                       help='原始数据集目录 (默认: final_dataset)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--batch', type=int, default=16,
                       help='批次大小 (默认: 16)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='图像尺寸 (默认: 640)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='基础模型路径 (默认: yolov8n.pt)')
    parser.add_argument('--project', type=str, default='traffic_light_train',
                       help='训练结果保存目录 (默认: traffic_light_train)')

    args = parser.parse_args()

    # 第一步：创建traffic light专用数据集
    traffic_light_count = create_single_class_dataset(args.dataset)

    if traffic_light_count < 10:
        print(f"\n警告: traffic light标注数太少 ({traffic_light_count})，训练效果可能不佳")
        print("建议至少需要50+个标注")

    # 第二步：开始训练
    success = train_traffic_light(
        data_yaml='traffic_light_dataset/traffic_light.yaml',
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        model_name=args.model,
        project=args.project
    )

    if not success:
        exit(1)