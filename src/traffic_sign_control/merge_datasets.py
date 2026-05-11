"""
合并多个数据集
将多个地图的数据集合并为一个大的数据集
"""
import os
import shutil
import argparse
from tqdm import tqdm

def merge_datasets(dataset_dirs, output_dir='merged_dataset'):
    """
    合并多个数据集

    Args:
        dataset_dirs: 数据集目录列表
        output_dir: 输出目录
    """
    print(f"开始合并数据集: {dataset_dirs}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录结构
    for d in ['images', 'semantic', 'labels']:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    total_frames = 0

    # 合并数据集
    for i, dataset_dir in enumerate(dataset_dirs):
        if not os.path.exists(dataset_dir):
            print(f"警告: 数据集目录不存在: {dataset_dir}")
            continue

        # 检查数据集完整性
        images_dir = os.path.join(dataset_dir, 'images')
        semantic_dir = os.path.join(dataset_dir, 'semantic')
        labels_dir = os.path.join(dataset_dir, 'labels')

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"警告: 数据集结构不完整，跳过: {dataset_dir}")
            continue

        # 统计当前数据集
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

        print(f"\n数据集 {i+1}: {dataset_dir}")
        print(f"  - 图像数量: {len(image_files)}")
        print(f"  - 标注数量: {len(label_files)}")

        # 复制文件
        print("  合并中...")
        for img_file in tqdm(image_files):
            # 复制图像
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, 'images', f"frame_{total_frames+1:06d}.jpg")
            shutil.copy2(src_img, dst_img)

            # 复制语义分割（如果有）
            if os.path.exists(semantic_dir):
                src_seg = os.path.join(semantic_dir, f"frame_{total_frames+1:06d}.png")
                if os.path.exists(src_seg):
                    dst_seg = os.path.join(output_dir, 'semantic', f"frame_{total_frames+1:06d}.png")
                    shutil.copy2(src_seg, dst_seg)

            # 复制标注
            label_name = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(labels_dir, label_name)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_dir, 'labels', f"frame_{total_frames+1:06d}.txt")
                shutil.copy2(src_label, dst_label)

            total_frames += 1

    print(f"\n合并完成！")
    print(f"总帧数: {total_frames}")
    print(f"输出目录: {output_dir}")

    # 生成配置文件
    print("\n正在生成配置文件...")
    from generate_dataset_yaml import generate_dataset_yaml
    generate_dataset_yaml(output_dir, f"{output_dir}_merged.yaml")

    return total_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并多个数据集")
    parser.add_argument('--datasets', nargs='+', required=True,
                       help='要合并的数据集目录（多个）')
    parser.add_argument('--output', type=str, default='merged_dataset',
                       help='输出目录 (默认: merged_dataset)')

    args = parser.parse_args()

    success = merge_datasets(args.datasets, args.output)
    if success > 0:
        print(f"\n🎉 成功合并了 {success} 帧数据！")
    else:
        print("\n❌ 合并失败")
        exit(1)
