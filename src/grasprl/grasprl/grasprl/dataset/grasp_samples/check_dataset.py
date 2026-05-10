import os
import numpy as np

data_path = os.path.dirname(os.path.abspath(__file__))
need_fields = ["grasp_success", "grasp_force", "grasp_pose"]


total = 0
lack = 0
error_files = []
label_err = []

files = []
for f in os.listdir(data_path):
    if f.endswith(".npy"):
        files.append(f)

for f in files:
    total += 1
    p = os.path.join(data_path, f)
    try:
        d = np.load(p, allow_pickle=True)
        if isinstance(d, np.ndarray):
            d = d.item()
        for field in need_fields:
            if field not in d:
                lack += 1
                label_err.append(f"{f} 缺少 {field}")
        if "grasp_force" in d:
            v = d["grasp_force"]
            if not (0 <= v <= 100):
                label_err.append(f"{f} grasp_force 异常")
        if "grasp_pose" in d:
            pose = d["grasp_pose"]
            if not isinstance(pose, list) or len(pose) != 3:
                label_err.append(f"{f} grasp_pose 异常")
    except:
        error_files.append(f)

print("检查完成")
print("总文件", total)
print("缺失字段数", lack)
print("错误文件", len(error_files))
print("数据异常", len(label_err))