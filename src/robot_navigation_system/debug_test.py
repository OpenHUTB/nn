import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=== DEBUG TEST START ===")

import os

# 检查当前目录
current_dir = os.getcwd()
print("当前目录:", current_dir)

# 列出当前目录内容
print("\n当前目录文件列表:")
for item in os.listdir(current_dir):
    print("  -", item)

# 创建results目录
results_path = os.path.join(current_dir, 'results')
print("\n尝试创建目录:", results_path)
os.makedirs(results_path, exist_ok=True)
print("目录创建完成")

# 再次检查
print("\n检查results目录是否存在:", os.path.exists(results_path))

# 写入一个测试文件
test_file = os.path.join(results_path, 'test.txt')
print("\n尝试写入文件:", test_file)
with open(test_file, 'w', encoding='utf-8') as f:
    f.write('测试内容')
print("文件写入完成")

# 检查文件是否存在
print("文件是否存在:", os.path.exists(test_file))

print("\n=== DEBUG TEST END ===")