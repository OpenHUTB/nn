import os
import numpy as np

data_dir = "grasprl/dataset/grasp_samples"

rgb_files = [f for f in os.listdir(data_dir) if f.startswith("rgb_") and f.endswith(".png")]
total_samples = len(rgb_files)

success_num = 0
fail_num = 0
for f in rgb_files:
    iter_num = int(f.split("_")[1].split(".")[0])
    label_path = os.path.join(data_dir, f"label_{iter_num}.npy")
    label = np.load(label_path, allow_pickle=True).item()
    if label["grasp_success"] == 1:
        success_num += 1
    else:
        fail_num += 1

print("="*30)
print("数据集统计结果")
print("="*30)
print(f"总样本数：{total_samples}")
print(f"成功抓取样本：{success_num}（{success_num/total_samples:.2%}）")
print(f"失败抓取样本：{fail_num}（{fail_num/total_samples:.2%}）")
print("="*30)