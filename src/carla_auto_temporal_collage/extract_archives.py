import os
import zipfile

def merge_split_zip(parts_dir, output_zip, base_name='videos.zip'):
    """合并分卷压缩文件"""
    parts = []
    i = 1
    while True:
        part_path = os.path.join(parts_dir, f'{base_name}.{i:03d}')
        if os.path.exists(part_path):
            parts.append(part_path)
            i += 1
        else:
            break
    
    if not parts:
        print(f"未找到分卷文件: {base_name}.001")
        return False
    
    print(f"找到 {len(parts)} 个分卷文件")
    
    # 合并文件
    with open(output_zip, 'wb') as outfile:
        for part in parts:
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
    
    print(f"合并完成: {output_zip}")
    return True

def extract_zip(zip_path, extract_to):
    """解压 ZIP 文件"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"解压完成: {zip_path} -> {extract_to}")
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False

def main():
    # 处理 data/videos.zip.001 等
    data_dir = 'data'
    videos_zip = os.path.join(data_dir, 'videos_merged.zip')
    
    if merge_split_zip(data_dir, videos_zip, 'videos.zip'):
        extract_zip(videos_zip, data_dir)
    
    # 处理 experiments/experiments.zip
    exp_dir = 'experiments'
    exp_zip = os.path.join(exp_dir, 'experiments.zip')
    if os.path.exists(exp_zip):
        extract_zip(exp_zip, exp_dir)

if __name__ == '__main__':
    main()
