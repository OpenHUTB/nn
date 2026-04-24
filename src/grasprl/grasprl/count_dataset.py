import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "grasprl", "dataset", "grasp_samples")

if not os.path.exists(data_dir):
    print("数据集目录不存在")
    exit()

rgb_files = [f for f in os.listdir(data_dir) if f.startswith("rgb_") and f.endswith(".png")]
total = len(rgb_files)
success = 0
fail = 0

for f in rgb_files:
    try:
        idx = f.split("_")[1].split(".")[0]
        label = np.load(os.path.join(data_dir, f"label_{idx}.npy"), allow_pickle=True).item()
        if label["grasp_success"] == 1:
            success +=1
        else:
            fail +=1
    except:
        continue

print("==============================")
print("数据集统计结果")
print("==============================")
print(f"总样本数：{total}")
print(f"成功抓取：{success} ({success/total:.2%})" if total>0 else "成功抓取：0")
print(f"失败抓取：{fail}")
print("==============================")