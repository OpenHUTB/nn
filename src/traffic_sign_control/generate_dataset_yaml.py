"""
生成YOLO训练配置文件 (dataset.yaml)
基于采集的数据集自动生成配置
"""
import os
import yaml

def generate_dataset_yaml(dataset_dir='dataset', output_file='dataset.yaml'):
    """
    生成YOLO训练配置文件

    Args:
        dataset_dir: 数据集目录路径
        output_file: 输出的yaml文件路径
    """
    # 检查数据集目录是否存在
    if not os.path.exists(dataset_dir):
        print(f"错误: 数据集目录不存在: {dataset_dir}")
        print("提示: 请先运行 main.py 采集数据")
        return False

    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"错误: 数据集目录结构不完整")
        print(f"需要: {images_dir} 和 {labels_dir}")
        return False

    # 统计数据集信息
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    print(f"数据集统计:")
    print(f"  - 图像数量: {len(image_files)}")
    print(f"  - 标注数量: {len(label_files)}")

    if len(image_files) == 0:
        print("警告: 数据集为空，请先采集数据")
        return False

    # 检查是否已经划分train/val
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        # 已经划分过，使用划分后的目录
        train_images = len([f for f in os.listdir(os.path.join(train_dir, 'images'))
                           if f.endswith(('.jpg', '.png'))])
        val_images = len([f for f in os.listdir(os.path.join(val_dir, 'images'))
                         if f.endswith(('.jpg', '.png'))])
        print(f"  - 训练集: {train_images} 张")
        print(f"  - 验证集: {val_images} 张")

        # YOLO配置内容
        dataset_config = {
            'path': './' + dataset_dir,           # 数据集根目录（相对路径）
            'train': 'train/images',              # 训练集图像目录
            'val': 'val/images',                  # 验证集图像目录
            'nc': 2,                              # 类别数量
            'names': [                            # 类别名称
                'stop_sign',
                'traffic_light'
            ]
        }
    else:
        # 未划分，使用原始数据集
        print("  - 未划分train/val集，训练时会自动划分")

        # YOLO配置内容
        dataset_config = {
            'path': './' + dataset_dir,           # 数据集根目录（相对路径）
            'train': 'images',                    # 训练集图像目录
            'val': 'images',                      # 验证集图像目录（简单起见，使用同一目录）
            'nc': 2,                              # 类别数量
            'names': [                            # 类别名称
                'stop_sign',
                'traffic_light'
            ]
        }

    # 写入YAML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n配置文件已生成: {output_file}")
    print(f"内容:")
    print(yaml.dump(dataset_config, default_flow_style=False, allow_unicode=True))

    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成YOLO训练配置文件")
    parser.add_argument('--dataset-dir', type=str, default='dataset',
                       help='数据集目录 (默认: dataset)')
    parser.add_argument('--output', type=str, default='dataset.yaml',
                       help='输出配置文件路径 (默认: dataset.yaml)')

    args = parser.parse_args()

    success = generate_dataset_yaml(args.dataset_dir, args.output)
    if not success:
        exit(1)
