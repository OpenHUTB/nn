# 数据集校验工具
# 检查RGB与Label是否匹配、Label字段是否完整
import os
import numpy as np

data_path = r"D:\nn\src\grasprl\grasprl\grasprl\dataset\grasp_samples"
need_fields = ["grasp_success"]
report_name = "dataset_check.txt"

def check_files():
    # 统计变量
    total_rgb = 0
    total_label = 0
    no_label = []    
    no_rgb = []      
    label_err = []   
    lack_field = []  

    # 筛选RGB和Label文件
    rgb_list = []
    label_list = []
    for file in os.listdir(data_path):
        full_path = os.path.join(data_path, file)
        # RGB：rgb_*.png
        if file[:4] == "rgb_" and file[-4:] == ".png":
            rgb_list.append(file)
            total_rgb += 1
        # Label：label_*.npy
        elif file[:6] == "label_" and file[-4:] == ".npy":
            label_list.append(file)
            total_label += 1

    # 提取索引并检查格式
    rgb_idx = set()
    for f in rgb_list:
        try:
            idx = int(f.replace("rgb_", "").replace(".png", ""))
            rgb_idx.add(idx)
        except:
            label_err.append(f"{f}：文件名格式错误")

    label_idx = set()
    for f in label_list:
        try:
            idx = int(f.replace("label_", "").replace(".npy", ""))
            label_idx.add(idx)
        except:
            label_err.append(f"{f}：标签文件名格式错误")

    # 匹配检查：有图无标签 / 有标签无图
    for f in rgb_list:
        try:
            idx = int(f.replace("rgb_", "").replace(".png", ""))
            if idx not in label_idx:
                no_label.append(f)
        except:
            pass

    for f in label_list:
        try:
            idx = int(f.replace("label_", "").replace(".npy", ""))
            if idx not in rgb_idx:
                no_rgb.append(f)
        except:
            pass

    # 检查Label内容与字段
    for label_file in label_list:
        label_full = os.path.join(data_path, label_file)
        try:
            label_data = np.load(label_full, allow_pickle=True).item()
            # 必须是字典
            if not isinstance(label_data, dict):
                label_err.append(f"{label_file}：非字典格式")
                continue
            # 检查必填字段
            lack = [field for field in need_fields if field not in label_data]
            if lack:
                lack_field.append(f"{label_file}：缺少字段{lack}")
        except:
            label_err.append(f"{label_file}：加载失败或文件损坏")

    # 生成报告
    report = []
    report.append("数据集检查结果")
    report.append("------------")
    report.append(f"RGB文件总数：{total_rgb}")
    report.append(f"Label文件总数：{total_label}\n")

    report.append(f"有RGB但无Label({len(no_label)}个)：")
    for f in no_label[:10]:
        report.append(f"  - {f}")
    if len(no_label) > 10:
        report.append(f"  - 其余{len(no_label)-10}个未显示")
    report.append("")

    report.append(f"有Label但无RGB({len(no_rgb)}个)：")
    for f in no_rgb[:10]:
        report.append(f"  - {f}")
    if len(no_rgb) > 10:
        report.append(f"  - 其余{len(no_rgb)-10}个未显示")
    report.append("")

    report.append(f"Label格式错误({len(label_err)}个)：")
    for f in label_err[:10]:
        report.append(f"  - {f}")
    if len(label_err) > 10:
        report.append(f"  - 其余{len(label_err)-10}个未显示")
    report.append("")

    report.append(f"Label缺少字段({len(lack_field)}个)：")
    for f in lack_field[:10]:
        report.append(f"  - {f}")
    if len(lack_field) > 10:
        report.append(f"  - 其余{len(lack_field)-10}个未显示")
    report.append("")


    with open(os.path.join(data_path, report_name), "w", encoding="utf-8") as f:
        f.write('\n'.join(report))
    print('\n'.join(report))
    print(f"\n报告已保存至：{data_path}/{report_name}")

if __name__ == "__main__":
    check_files()