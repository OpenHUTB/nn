import os
import numpy as np

data_path = os.path.dirname(os.path.abspath(__file__))

clean_rules = {
    "grasp_success": {"valid_range": [0, 1], "default": 0},
    "grasp_force": {"valid_range": [0, 100], "default": 50},
    "grasp_pose": {"valid_type": list, "valid_length": 3, "default": [0, 0, 0]}
    "grasp_success": {"valid_range": [0,1], "default": 0},
    "grasp_force": {"valid_range": [0, 100], "default": 50},
    "grasp_pose": {"valid_type": list, "default": [0,0,0]}
}

clean_count = 0
total_count = 0
error_files = []

files = [f for f in os.listdir(data_path) if f.endswith(".npy")]

log = []

for f in files:
    total_count += 1
    p = os.path.join(data_path, f)
    try:
        data = np.load(p, allow_pickle=True)
        if isinstance(data, np.ndarray):
            data = data.item()
        modified = False
        for field, rule in clean_rules.items():
            if field not in data:
                data[field] = rule["default"]
                modified = True
                continue
            val = data[field]
            if "valid_range" in rule:
                minv, maxv = rule["valid_range"]
                if not (minv <= val <= maxv):
                    data[field] = rule["default"]
                    modified = True
            if "valid_type" in rule:
                if not isinstance(val, rule["valid_type"]):
                    data[field] = rule["default"]
                    modified = True
            if "valid_length" in rule:
                if len(val) != rule["valid_length"]:
                    data[field] = rule["default"]
                    modified = True
        if modified:
            np.save(p, data, allow_pickle=True)
            clean_count += 1
    except:
        error_files.append(f)


print("清洗完成")
print("总样本", total_count)
print("已修复", clean_count)
print("错误文件", len(error_files))
print("错误文件", len(error_files))
label_files = [f for f in os.listdir(data_path) if f.startswith("label_") and f.endswith(".npy")]

print("===== 机械抓取数据清洗 =====")
for f in label_files:
    total_count += 1
    label_path = os.path.join(data_path, f)
    
    try:
        label = np.load(label_path, allow_pickle=True).item()
        modified = False
        
        for field, rule in clean_rules.items():
            if field not in label:
                label[field] = rule["default"]
                modified = True
                print(f"{f} → 缺失 {field}，已填充")
                continue
            
            val = label[field]
            
            if "valid_range" in rule:
                vmin, vmax = rule["valid_range"]
                if not (vmin <= val <= vmax):
                    label[field] = rule["default"]
                    modified = True
                    print(f"{f} → {field} 异常，已修正")
            
            if "valid_type" in rule:
                if not isinstance(val, rule["valid_type"]):
                    label[field] = rule["default"]
                    modified = True
                    print(f"{f} → {field} 类型错误，已修正")
        
        if modified:
            np.save(label_path, label, allow_pickle=True)
            clean_count += 1
    
    except:
        error_files.append(f)
        print(f"{f} → 读取失败")

print("\n清洗完成")
print(f"总样本：{total_count}")
print(f"已修正：{clean_count}")
print(f"异常文件：{len(error_files)}")
