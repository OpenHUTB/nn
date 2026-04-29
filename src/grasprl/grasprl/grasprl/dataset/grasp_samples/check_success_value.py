import os
import sys
import numpy as np

# 改成相对路径
data_path = sys.argv[1] if len(sys.argv)>1 else os.path.dirname(os.path.abspath(__file__))
success_num = 0
total_num = 0
error_files = []

# 按编号分段统计
segment1_success = 0  # 1-20号样本
segment2_success = 0  # 21-50号样本
segment1_total = 0
segment2_total = 0

label_files = [f for f in os.listdir(data_path) if f.startswith("label_") and f.endswith(".npy")]
if not label_files:
    print("未找到任何label_开头的npy文件！")
    exit()

print("===== 各Label的grasp_success值 =====")
for f in label_files:
    total_num += 1
    try:
        label = np.load(os.path.join(data_path, f), allow_pickle=True).item()
        val = label.get("grasp_success", -1)
        print(f"{f} → {val}")
        
        # 提取样本编号，分层统计
        idx = int(f.replace("label_", "").replace(".npy", ""))
        if 1 <= idx <= 20:
            segment1_total += 1
            if val == 1:
                segment1_success += 1
        elif 21 <= idx <= 50:
            segment2_total += 1
            if val == 2:
                segment2_success += 1
        
        if val == 1:
            success_num += 1
    except FileNotFoundError:
        error_files.append(f)
        print(f"{f} → 文件不存在")
    except ValueError:
        error_files.append(f)
        print(f"{f} → 格式错误，无法转字典")
    except Exception as e:
        error_files.append(f)
        print(f"{f} → 未知错误：{str(e)}")

# 输出分层统计结果
print("\n===== 分层成功率统计（机械抓取核心分析） =====")
print(f"整体：总数量={total_num} | 成功数={success_num} | 成功率={success_num/total_num*100:.2f}%")
print(f"1-20号样本（模拟抓取区域A）：总数量={segment1_total} | 成功数={segment1_success} | 成功率={segment1_success/segment1_total*100:.2f}%" if segment1_total>0 else "1-20号样本：无数据")
print(f"21-50号样本（模拟抓取区域B）：总数量={segment2_total} | 成功数={segment2_success} | 成功率={segment2_success/segment2_total*100:.2f}%" if segment2_total>0 else "21-50号样本：无数据")
print(f"读取失败：{len(error_files)}个")

if error_files:
    print("\n===== 失败文件 =====")
    for f in error_files:
        print(f"- {f}")