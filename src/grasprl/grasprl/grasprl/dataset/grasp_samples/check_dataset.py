import os
import numpy as np

# 改成相对路径 
data_path = os.path.dirname(os.path.abspath(__file__))
need_fields = ["grasp_success"]  
report_name = "dataset_check.txt"

def check_files():
    total_rgb = 0
    total_label = 0
    no_label = []
    no_rgb = []
    label_err = []
    lack_field = []

    rgb_list = []
    label_list = []
    for file in os.listdir(data_path):
        full_path = os.path.join(data_path, file)
        if file[:4] == "rgb_" and file[-4:] == ".png":
            rgb_list.append(file)
            total_rgb += 1
        elif file[:6] == "label_" and file[-4:] == ".npy":
            label_list.append(file)
            total_label += 1

    rgb_idx = set()
    for f in rgb_list:
        try:
            idx = int(f.replace("rgb_", "").replace(".png", ""))
            rgb_idx.add(idx)
        except:
            label_err.append(f"{f}：文件名不对，得是rgb_数字.png这种格式")

    label_idx = set()
    for f in label_list:
        try:
            idx = int(f.replace("label_", "").replace(".npy", ""))
            label_idx.add(idx)
        except:
            label_err.append(f"{f}：标签文件名不对，得是label_数字.npy")

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

    for label_file in label_list:
        label_full = os.path.join(data_path, label_file)
        try:
            label_data = np.load(label_full, allow_pickle=True).item()
            if type(label_data) != dict:
                label_err.append(f"{label_file}：不是字典格式，存的时候要转字典！")
                continue
            lack = []
            for field in need_fields:
                if field not in label_data:
                    lack.append(field)
            if lack:
                lack_field.append(f"{label_file}：少字段{lack}")
        except:
            label_err.append(f"{label_file}：文件坏了或者加载失败")

    report = []
    report.append("数据集检查结果")
    report.append("------------")
    report.append(f"RGB文件总数：{total_rgb}")
    report.append(f"Label文件总数：{total_label}\n")

    report.append(f"有RGB但没Label的文件({len(no_label)}个)：")
    for f in no_label[:10]:
        report.append(f"  - {f}")
    if len(no_label) > 10:
        report.append(f"  - 还有{len(no_label)-10}个没显示")
    report.append("")

    report.append(f"有Label但没RGB的文件({len(no_rgb)}个)：")
    for f in no_rgb[:10]:
        report.append(f"  - {f}")
    if len(no_rgb) > 10:
        report.append(f"  - 还有{len(no_rgb)-10}个没显示")
    report.append("")

    report.append(f"Label格式错误({len(label_err)}个)：")
    for f in label_err[:10]:
        report.append(f"  - {f}")
    if len(label_err) > 10:
        report.append(f"  - 还有{len(label_err)-10}个没显示")
    report.append("")

    report.append(f"Label缺字段({len(lack_field)}个)：")
    for f in lack_field[:10]:
        report.append(f"  - {f}")
    if len(lack_field) > 10:
        report.append(f"  - 还有{len(lack_field)-10}个没显示")
    report.append("")

    with open(os.path.join(data_path, report_name), "w", encoding="utf-8") as f:
        f.write('\n'.join(report))
    print('\n'.join(report))
    print(f"\n报告已经存到{data_path}里的{report_name}了")

if __name__ == "__main__":
    check_files()