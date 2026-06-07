"""
批量处理器 - 处理文件夹中的多张图片
"""

import os
import csv
import json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2


class BatchProcessor:
    """批量图片处理器"""

    def __init__(self, detection_service):
        self.detection_service = detection_service

    def process_folder(self, root_window):
        """处理整个文件夹的图片"""
        # 选择输入文件夹
        input_folder = filedialog.askdirectory(
            title="选择包含道路图片的文件夹"
        )

        if not input_folder:
            return

        # 选择输出文件夹
        output_folder = filedialog.askdirectory(
            title="选择导出结果的保存文件夹"
        )

        if not output_folder:
            return

        # 获取所有图片文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith(image_extensions)
        ]

        if not image_files:
            messagebox.showwarning("警告", "所选文件夹中没有找到图片文件")
            return

        # 确认批量处理
        confirm = messagebox.askyesno(
            "确认批量处理",
            f"将处理 {len(image_files)} 张图片\n\n是否继续？"
        )

        if not confirm:
            return

        # 创建进度窗口
        progress_window = tk.Toplevel(root_window)
        progress_window.title("批量处理中...")
        progress_window.geometry("400x200")
        progress_window.transient(root_window)
        progress_window.grab_set()

        # 进度信息标签
        info_label = ttk.Label(
            progress_window,
            text=f"正在处理 0/{len(image_files)} 张图片...",
            font=("微软雅黑", 10)
        )
        info_label.pack(pady=20)

        # 进度条
        progress_bar = ttk.Progressbar(
            progress_window,
            mode='determinate',
            length=350
        )
        progress_bar.pack(pady=10)
        progress_bar['maximum'] = len(image_files)

        # 状态标签
        status_label = ttk.Label(
            progress_window,
            text="",
            font=("微软雅黑", 9),
            foreground="#7f8c8d"
        )
        status_label.pack(pady=10)

        # 在后台线程中处理
        import threading
        thread = threading.Thread(
            target=self._batch_process,
            args=(input_folder, output_folder, image_files,
                  progress_window, info_label, progress_bar, status_label),
            daemon=True
        )
        thread.start()

    def _batch_process(self, input_folder, output_folder, image_files,
                       progress_window, info_label, progress_bar, status_label):
        """执行批量处理"""
        success_count = 0
        fail_count = 0
        results = []

        for i, image_file in enumerate(image_files):
            try:
                # 更新进度
                progress_window.after(0, lambda idx=i + 1, total=len(image_files), name=image_file: (
                    info_label.config(text=f"正在处理 {idx}/{total}: {name}"),
                    progress_bar.step(1),
                    status_label.config(text=f"当前: {name}")
                ))

                # 读取图片
                image_path = os.path.join(input_folder, image_file)
                image = cv2.imread(image_path)

                if image is None:
                    fail_count += 1
                    continue

                # 检测方向
                direction_result = self._detect_single_image(image)

                # 保存结果图片
                result_filename = f"result_{os.path.splitext(image_file)[0]}.png"
                result_path = os.path.join(output_folder, result_filename)
                cv2.imwrite(result_path, direction_result['result_image'])

                # 保存JSON报告
                json_filename = f"report_{os.path.splitext(image_file)[0]}.json"
                json_path = os.path.join(output_folder, json_filename)

                report_data = {
                    'image_file': image_file,
                    'generated_at': datetime.now().isoformat(),
                    'direction': direction_result['direction'],
                    'confidence': direction_result['confidence'],
                    'reasoning': direction_result.get('reasoning', '')
                }

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False)

                results.append({
                    'file': image_file,
                    'direction': direction_result['direction'],
                    'confidence': direction_result['confidence']
                })

                success_count += 1

            except Exception as e:
                print(f"处理 {image_file} 失败: {e}")
                fail_count += 1

        # 处理完成
        progress_window.after(0, lambda: (
            info_label.config(text=f"处理完成！成功: {success_count}, 失败: {fail_count}"),
            status_label.config(text=f"结果已保存到: {output_folder}"),
            progress_window.after(2000, progress_window.destroy)
        ))

        # 保存汇总报告
        summary_path = os.path.join(output_folder, "batch_summary.csv")
        with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'direction', 'confidence'])
            writer.writeheader()
            writer.writerows(results)

        progress_window.after(0, lambda: messagebox.showinfo(
            "批量处理完成",
            f"成功: {success_count}\n失败: {fail_count}\n\n汇总报告: {summary_path}"
        ))

    def _detect_single_image(self, image):
        """检测单张图片"""
        try:
            # 使用检测服务进行处理
            result = self.detection_service.detect_image_from_array(image)

            if result is None:
                raise ValueError("检测失败")

            return {
                'direction': result['direction_info']['direction'],
                'confidence': result['direction_info']['confidence'],
                'reasoning': result['direction_info'].get('reasoning', ''),
                'result_image': result['visualization']
            }

        except Exception as e:
            print(f"检测失败: {e}")
            return {
                'direction': '未知',
                'confidence': 0.0,
                'reasoning': f'检测失败: {str(e)}',
                'result_image': image
            }
