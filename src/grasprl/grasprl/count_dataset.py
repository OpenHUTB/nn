import os
import numpy as np

DATA_DIR_RELATIVE = ["dataset", "grasp_samples"]
SUCCESS_FIELD = "grasp_success"
SUCCESS_VALUE = 1

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, *DATA_DIR_RELATIVE)


if not os.path.exists(data_dir):
    print(f"❌ 数据集目录不存在：{data_dir}")
    exit(1)

rgb_files = [f for f in os.listdir(data_dir) if f.startswith("rgb_") and f.endswith(".png")]
total = len(rgb_files)
success = 0
fail = 0
error_files = [] 


for f in rgb_files:
    try:
        idx = f.replace("rgb_", "").replace(".png", "")
        label_path = os.path.join(data_dir, f"label_{idx}.npy")
        
        if not os.path.exists(label_path):
            error_files.append(f"缺少label文件：{label_path}")
            fail += 1  # 缺失label视为失败
            continue
        
        label = np.load(label_path, allow_pickle=True).item()
        
        if SUCCESS_FIELD not in label:
            error_files.append(f"{label_path} 缺少字段：{SUCCESS_FIELD}")
            fail += 1
            continue
    
        if label[SUCCESS_FIELD] == SUCCESS_VALUE:
            success += 1
        else:
            fail += 1
            
    except Exception as e:
        error_files.append(f"{f} 处理失败：{str(e)}")
        fail += 1

# 输出统计结果
print("==============================")
print("数据集统计结果")
print("==============================")
print(f"总样本数（rgb文件数）：{total}")
if total > 0:
    success_rate = success / total * 100
    print(f"成功抓取：{success} ({success_rate:.2f}%)")
else:
    print("成功抓取：0 (0.00%)")
print(f"失败抓取：{fail}")
print(f"处理异常文件数：{len(error_files)}")
print("==============================")

# 输出调试信息（关键：定位计数为0的原因）
if error_files:
    print("\n【异常文件列表（前10个）】")
    for err in error_files[:10]:
        print(f"- {err}")
    if len(error_files) > 10:
        print(f"- ... 还有 {len(error_files)-10} 个异常文件")

# 校验计数是否合理
if success + fail != total:
    print(f"\n⚠️  计数异常：成功({success}) + 失败({fail}) ≠ 总样本({total})")