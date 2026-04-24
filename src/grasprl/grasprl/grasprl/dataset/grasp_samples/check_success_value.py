import os
import numpy as np

data_path = r"D:\nn\src\grasprl\grasprl\grasprl\dataset\grasp_samples"
success_num = 0
total_num = 0
error_files = []

label_files = [f for f in os.listdir(data_path) if f.startswith("label_") and f.endswith(".npy")]

print("===== 各Label的grasp_success值 =====")
for f in label_files:
    total_num += 1
    try:
        label = np.load(os.path.join(data_path, f), allow_pickle=True).item()
        val = label.get("grasp_success", -1)
        print(f"{f} → {val}")
        if val == 1:
            success_num += 1
    except:
        error_files.append(f)
        print(f"{f} → 读取失败")

print("\n===== 统计结果 =====")
print(f"总数量：{total_num}")
print(f"成功数：{success_num}")
print(f"失败数：{total_num - success_num - len(error_files)}")
print(f"读取失败：{len(error_files)}")
if total_num > 0:
    print(f"成功率：{success_num/total_num*100:.2f}%")

if error_files:
    print("\n===== 失败文件 =====")
    for f in error_files:
        print(f"- {f}")