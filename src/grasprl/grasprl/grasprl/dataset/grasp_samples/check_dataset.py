import os
import numpy as np


#改成相对路径
data_path = os.path.dirname(os.path.abspath(__file__))
need_fields = ["grasp_success", "grasp_pose"]
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
            label_err.append(f + "：文件名错误")

    label_idx = set()
    for f in label_list:
        try:
            idx = int(f.replace("label_", "").replace(".npy", ""))
            label_idx.add(idx)
        except:
            label_err.append(f + "：文件名错误")

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
                label_err.append(label_file + "：格式错误")
                continue

            lack = []
            for field in need_fields:
                if field not in label_data:
                    lack.append(field)
            if lack:
                lack_field.append(label_file + " 缺少字段：" + str(lack))
        except:
            label_err.append(label_file + "：读取失败")

    usable = total_label - len(lack_field) - len(label_err)

    report = []
    report.append("机械抓取数据集检查")
    report.append("-------------------")
    report.append("RGB总数：" + str(total_rgb))
    report.append("Label总数：" + str(total_label))
    report.append("可用样本：" + str(usable))
    report.append("")

    report.append("有RGB无Label：" + str(len(no_label)))
    for f in no_label[:5]:
        report.append("  " + f)
    report.append("")

    report.append("有Label无RGB：" + str(len(no_rgb)))
    for f in no_rgb[:5]:
        report.append("  " + f)
    report.append("")

    report.append("Label错误：" + str(len(label_err)))
    for f in label_err[:5]:
        report.append("  " + f)
    report.append("")

    report.append("缺少抓取关键字段：" + str(len(lack_field)))
    for f in lack_field[:5]:
        report.append("  " + f)
    report.append("")

    report.append("可用样本 = 格式正确 + 字段完整")
    report.append("可直接用于机械抓取训练与分析")

    with open(os.path.join(data_path, report_name), "w", encoding="utf-8") as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))
    print("\n报告已保存")

if __name__ == "__main__":
    check_files()