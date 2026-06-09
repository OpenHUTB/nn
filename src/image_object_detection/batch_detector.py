# batch_detector.py
"""
批量图像检测器模块。

该模块提供一个 BatchDetector 类，用于对指定输入目录中的所有图像文件
进行批量目标检测，并将带有检测结果（如边界框、标签等）的图像保存到输出目录。
"""

import os
import cv2
from pathlib import Path
from detection_engine import DetectionEngine, ModelLoadError
from result_saver import ResultSaver


class BatchDetector:
    """
    批量图像检测器类。

    使用提供的 DetectionEngine 对象，对输入目录中的图像逐一进行目标检测，
    并将标注后的图像保存至输出目录。
    """

    def __init__(self, detection_engine, input_dir, output_dir):
        """
        初始化 BatchDetector 实例。

        参数:
            detection_engine (DetectionEngine): 已加载模型的检测引擎实例。
            input_dir (str 或 Path): 包含待检测图像的输入目录路径。
            output_dir (str 或 Path): 用于保存检测结果图像的输出目录路径。
        """
        self.engine = detection_engine                      # 检测引擎实例
        self.input_dir = Path(input_dir)                    # 输入目录（转换为 Path 对象）
        self.output_dir = Path(output_dir)                  # 输出目录（转换为 Path 对象）

        # 支持的图像文件扩展名集合（小写）
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # 自动创建输出目录（若不存在），包括中间父目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 检查输入目录是否存在
        if not self.input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")

        # 初始化结果保存器
        self.result_saver = ResultSaver(str(output_dir))

    def run(self, save_detections=True, save_csv=False):
        """
        执行批量检测流程。

        遍历输入目录中所有支持格式的图像文件，调用检测引擎进行推理，
        将带标注的图像保存到输出目录，并打印处理进度与结果统计。

        参数:
            save_detections: 是否保存检测数据JSON
            save_csv: 是否保存CSV报告
        """
        # 筛选出输入目录中所有符合支持扩展名的图像文件
        image_files = [
            f for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.image_extensions
        ]

        # 若未找到任何有效图像，提前退出
        if not image_files:
            print(f"⚠️ No valid image files found in {self.input_dir}")
            return

        print(f"🔍 Found {len(image_files)} images. Starting batch detection...")
        success_count = 0  # 成功处理的图像计数器

        # 按文件名排序以保证处理顺序可预测
        for img_path in sorted(image_files):
            try:
                # 使用 OpenCV 读取图像
                frame = cv2.imread(str(img_path))
                if frame is None:
                    # 若读取失败（如文件损坏或格式不支持），跳过该图像
                    print(f"❌ Failed to read image (corrupted or unsupported): {img_path.name}")
                    continue

                # 调用检测引擎进行目标检测
                # 返回值：annotated_frame（带标注的图像），results（检测结果）
                annotated_frame, results = self.engine.detect(frame)

                # 保存带标注的图像
                saved_path = self.result_saver.save_image(
                    annotated_frame, 
                    img_path.name,
                    suffix="detected"
                )
                print(f"✅ Saved: {os.path.basename(saved_path)}")

                # 保存检测数据JSON
                if save_detections:
                    self.result_saver.save_detection_data(
                        img_path.name,
                        results[0].boxes if results and results[0].boxes else []
                    )

                success_count += 1

            except Exception as e:
                # 捕获并报告处理单张图像时发生的任何异常
                print(f"💥 Error processing {img_path.name}: {e}")

        # 保存CSV报告
        if save_csv and success_count > 0:
            csv_path = self.result_saver.save_csv_report()
            print(f"📊 CSV report saved: {os.path.basename(csv_path)}")

        # 打印最终处理统计信息
        print(f"\n🎉 Batch detection completed. {success_count}/{len(image_files)} images processed successfully.")
        print(f"📁 Output directory: {self.output_dir}")
