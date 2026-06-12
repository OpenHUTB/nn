"""
导出管理器 - 处理检测结果的各种导出功能
"""

import os
import json
import csv
from datetime import datetime
from tkinter import filedialog, messagebox
import cv2


class ExportManager:
    """检测结果导出管理器"""

    def __init__(self):
        self.export_history = []

    def export_image(self, result_image, root=None):
        """导出带标注的结果图片"""
        if result_image is None:
            if root:
                messagebox.showwarning("警告", "没有可导出的结果图片")
            return False

        # 选择保存路径
        default_name = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = filedialog.asksaveasfilename(
            title="保存结果图片",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG图片", "*.png"),
                ("JPEG图片", "*.jpg *.jpeg"),
                ("所有文件", "*.*")
            ]
        )

        if not file_path:
            return False

        try:
            # 保存图片
            success = cv2.imwrite(file_path, result_image)

            if success:
                # 记录导出历史
                self.export_history.append({
                    'type': 'image',
                    'path': file_path,
                    'time': datetime.now().isoformat()
                })

                if root:
                    messagebox.showinfo("成功", f"结果图片已保存到:\n{file_path}")
                return True
            else:
                if root:
                    messagebox.showerror("错误", "保存图片失败")
                return False

        except Exception as e:
            if root:
                messagebox.showerror("错误", f"导出图片时出错: {str(e)}")
            print(f"导出图片失败: {e}")
            return False

    def export_json_report(self, detection_result, current_image, current_image_path, root=None):
        """导出JSON格式的检测报告"""
        if detection_result is None:
            if root:
                messagebox.showwarning("警告", "没有可导出的检测结果")
            return False

        # 选择保存路径
        default_name = f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = filedialog.asksaveasfilename(
            title="保存JSON报告",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[
                ("JSON文件", "*.json"),
                ("所有文件", "*.*")
            ]
        )

        if not file_path:
            return False

        try:
            # 构建完整的报告数据
            report_data = {
                'report_info': {
                    'generated_at': datetime.now().isoformat(),
                    'software_version': '1.0',
                    'source_image': current_image_path
                },
                'detection_result': detection_result,
                'image_info': {
                    'width': int(current_image.shape[1]) if current_image is not None else None,
                    'height': int(current_image.shape[0]) if current_image is not None else None,
                    'channels': int(current_image.shape[2]) if current_image is not None and len(
                        current_image.shape) > 2 else None
                }
            }

            # 写入JSON文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

            # 记录导出历史
            self.export_history.append({
                'type': 'json',
                'path': file_path,
                'time': datetime.now().isoformat()
            })

            if root:
                messagebox.showinfo("成功", f"JSON报告已保存到:\n{file_path}")
            return True

        except Exception as e:
            if root:
                messagebox.showerror("错误", f"导出JSON报告时出错: {str(e)}")
            print(f"导出JSON失败: {e}")
            return False

    def export_csv_report(self, detection_result, current_image, current_image_path, root=None):
        """导出CSV格式的检测报告"""
        if detection_result is None:
            if root:
                messagebox.showwarning("警告", "没有可导出的检测结果")
            return False

        # 选择保存路径
        default_name = f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = filedialog.asksaveasfilename(
            title="保存CSV报告",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[
                ("CSV文件", "*.csv"),
                ("所有文件", "*.*")
            ]
        )

        if not file_path:
            return False

        try:
            # 准备CSV数据
            result = detection_result

            # 定义CSV表头和数据
            headers = [
                '字段', '值'
            ]

            rows = [
                ['生成时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['源图片', current_image_path or 'N/A'],
                ['', ''],
                ['检测结果', ''],
                ['方向', result.get('direction', 'N/A')],
                ['置信度', f"{result.get('confidence', 0):.2%}"],
                ['推理说明', result.get('reasoning', 'N/A')],
                ['', ''],
                ['概率分布', ''],
            ]

            # 添加各方向的概率
            probabilities = result.get('probabilities', {})
            for direction, prob in probabilities.items():
                rows.append([f'{direction}概率', f"{prob:.2%}"])

            # 添加图像信息
            rows.extend([
                ['', ''],
                ['图像信息', ''],
            ])

            if current_image is not None:
                rows.extend([
                    ['宽度', current_image.shape[1]],
                    ['高度', current_image.shape[0]],
                ])
                if len(current_image.shape) > 2:
                    rows.append(['通道数', current_image.shape[2]])

            # 写入CSV文件
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

            # 记录导出历史
            self.export_history.append({
                'type': 'csv',
                'path': file_path,
                'time': datetime.now().isoformat()
            })

            if root:
                messagebox.showinfo("成功", f"CSV报告已保存到:\n{file_path}")
            return True

        except Exception as e:
            if root:
                messagebox.showerror("错误", f"导出CSV报告时出错: {str(e)}")
            print(f"导出CSV失败: {e}")
            return False

    def get_export_history(self):
        """获取导出历史"""
        return self.export_history.copy()

    def clear_export_history(self):
        """清空导出历史"""
        self.export_history.clear()
