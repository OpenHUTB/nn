import time
import cv2
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from pathlib import Path

from detection_engine import DetectionEngine, ModelLoadError


class ModelComparisonResult:
    """单模型测试结果容器"""

    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.detection_counts = []
        self.confidences = []
        self.class_counts = {}
        self.inference_times = []
        self.total_detections = 0
        self.avg_detection_count = 0.0
        self.avg_inference_time = 0.0
        self.success = False
        self.error_message = ""

    def finalize(self):
        """计算统计数据"""
        if self.detection_counts:
            self.total_detections = sum(self.detection_counts)
            self.avg_detection_count = self.total_detections / len(self.detection_counts)
        if self.inference_times:
            self.avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        if self.confidences:
            sorted_confs = sorted(self.confidences)
            self.confidence_stats = {
                'mean': sum(self.confidences) / len(self.confidences),
                'max': max(self.confidences),
                'min': min(self.confidences),
                'median': sorted_confs[len(sorted_confs) // 2]
            }


class ModelComparison:
    """
    多模型对比测试模块。

    支持对多个 YOLO 模型进行批量对比测试，
    提供检测数量、置信度分布、推理速度等多维度性能对比。
    """

    SUPPORTED_MODELS = {
        'yolov8n': 'yolov8n.pt',
        'yolov8s': 'yolov8s.pt',
        'yolov8m': 'yolov8m.pt',
        'yolov8l': 'yolov8l.pt',
        'yolov8x': 'yolov8x.pt',
    }

    def __init__(self, model_list: List[str], conf_threshold: float = 0.25):
        """
        初始化多模型对比器。

        Args:
            model_list: 模型路径或名称列表，如 ['yolov8n.pt', 'yolov8s.pt']
            conf_threshold: 置信度阈值
        """
        self.conf_threshold = conf_threshold
        self.model_results = {}
        self._engines = {}
        self._load_models(model_list)

    def _load_models(self, model_list: List[str]):
        """加载所有模型"""
        for model_path in model_list:
            model_name = self._get_model_display_name(model_path)
            try:
                engine = DetectionEngine(model_path=model_path, conf_threshold=self.conf_threshold)
                self._engines[model_name] = engine
                self.model_results[model_name] = ModelComparisonResult(model_name, model_path)
                self.model_results[model_name].success = True
                print(f"✅ Loaded model: {model_name}")
            except ModelLoadError as e:
                self.model_results[model_name] = ModelComparisonResult(model_name, model_path)
                self.model_results[model_name].success = False
                self.model_results[model_name].error_message = str(e)
                print(f"❌ Failed to load model {model_name}: {e}")

    def _get_model_display_name(self, model_path: str) -> str:
        """从路径获取模型显示名称"""
        basename = os.path.basename(model_path)
        name_without_ext = os.path.splitext(basename)[0]
        return name_without_ext

    def compare_on_image(self, image_path: str) -> Dict[str, Any]:
        """
        在单张图像上对比所有模型的检测结果。

        Args:
            image_path: 图像文件路径

        Returns:
            包含各模型检测结果的字典
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Failed to read image: {image_path}")

        results = {}
        for model_name, engine in self._engines.items():
            start_time = time.time()
            annotated, raw_results = engine.detect(frame)
            inference_time = time.time() - start_time

            detection_count = 0
            confidences = []
            class_counts = {}

            if raw_results and len(raw_results) > 0:
                result = raw_results[0]
                if result.boxes is not None:
                    boxes = result.boxes
                    detection_count = len(boxes)

                    for box in boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = result.names.get(cls_id, f"class_{cls_id}")
                        confidences.append(conf)
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            results[model_name] = {
                'annotated': annotated,
                'detection_count': detection_count,
                'confidences': confidences,
                'class_counts': class_counts,
                'inference_time': inference_time
            }

            self.model_results[model_name].detection_counts.append(detection_count)
            self.model_results[model_name].confidences.extend(confidences)
            self.model_results[model_name].inference_times.append(inference_time)

            for cls, count in class_counts.items():
                if cls not in self.model_results[model_name].class_counts:
                    self.model_results[model_name].class_counts[cls] = 0
                self.model_results[model_name].class_counts[cls] += count

        return results

    def compare_on_batch(self, image_dir: str, extensions: set = None) -> Dict[str, Any]:
        """
        在批量图像上对比所有模型。

        Args:
            image_dir: 包含图像文件的目录
            extensions: 支持的图像扩展名集合，默认为常见的图片格式

        Returns:
            批量测试汇总结果
        """
        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise ValueError(f"Invalid directory: {image_dir}")

        image_files = [
            f for f in image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]

        if not image_files:
            raise ValueError(f"No valid image files found in {image_dir}")

        print(f"🔍 Found {len(image_files)} images. Running model comparison...")
        batch_results = {}

        for img_path in sorted(image_files):
            try:
                results = self.compare_on_image(str(img_path))
                batch_results[img_path.name] = results
                print(f"✅ Processed: {img_path.name}")
            except Exception as e:
                print(f"❌ Error processing {img_path.name}: {e}")

        for model_name in self.model_results:
            self.model_results[model_name].finalize()

        return batch_results

    def get_comparison_summary(self) -> str:
        """生成模型对比汇总报告"""
        report = ["=" * 70, "📊 多模型对比测试报告", "=" * 70]

        successful_models = [m for m in self.model_results if self.model_results[m].success]

        if not successful_models:
            report.append("\n❌ No models loaded successfully!")
            return "\n".join(report)

        report.append(f"\n📋 测试模型数量: {len(successful_models)}")
        report.append(f"📋 测试图像数量: {len(self.model_results[successful_models[0]].detection_counts)}")
        report.append(f"🎚️ 置信度阈值: {self.conf_threshold}")

        report.append("\n" + "-" * 70)
        report.append("📈 检测性能对比")
        report.append("-" * 70)

        header = f"{'模型':<15} {'检测总数':>10} {'每图平均':>10} {'总耗时(s)':>12} {'平均耗时(ms)':>14}"
        report.append(header)
        report.append("-" * 70)

        for model_name in successful_models:
            result = self.model_results[model_name]
            total_time = sum(result.inference_times)
            avg_time_ms = result.avg_inference_time * 1000
            report.append(
                f"{model_name:<15} {result.total_detections:>10} "
                f"{result.avg_detection_count:>10.2f} {total_time:>12.3f} {avg_time_ms:>14.2f}"
            )

        report.append("-" * 70)

        report.append("\n" + "-" * 70)
        report.append("🎯 类别检测分布")
        report.append("-" * 70)

        all_classes = set()
        for model_name in successful_models:
            all_classes.update(self.model_results[model_name].class_counts.keys())

        if all_classes:
            header = f"{'类别':<20}"
            for model_name in successful_models:
                header += f" {model_name:>10}"
            report.append(header)
            report.append("-" * 70)

            for cls in sorted(all_classes):
                row = f"{cls:<20}"
                for model_name in successful_models:
                    count = self.model_results[model_name].class_counts.get(cls, 0)
                    row += f" {count:>10}"
                report.append(row)

        report.append("-" * 70)

        report.append("\n" + "-" * 70)
        report.append("📉 置信度分布")
        report.append("-" * 70)

        header = f"{'模型':<15} {'平均':>10} {'最大':>10} {'最小':>10} {'中位数':>10}"
        report.append(header)
        report.append("-" * 70)

        for model_name in successful_models:
            result = self.model_results[model_name]
            if hasattr(result, 'confidence_stats') and result.confidence_stats:
                stats = result.confidence_stats
                report.append(
                    f"{model_name:<15} {stats['mean']:>10.3f} {stats['max']:>10.3f} "
                    f"{stats['min']:>10.3f} {stats['median']:>10.3f}"
                )

        report.append("-" * 70)

        failed_models = [m for m in self.model_results if not self.model_results[m].success]
        if failed_models:
            report.append("\n❌ 加载失败的模型:")
            for model_name in failed_models:
                report.append(f"  - {model_name}: {self.model_results[model_name].error_message}")

        report.append("\n" + "=" * 70)
        report.append(f"📝 报告生成时间: {self._get_current_time()}")
        report.append("=" * 70)

        return "\n".join(report)

    def save_comparison_images(self, output_dir: str, image_path: str):
        """
        保存各模型对同一图像的检测结果对比图。

        Args:
            output_dir: 输出目录
            image_path: 原始图像路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image_name = Path(image_path).stem

        results = self.compare_on_image(image_path)

        for model_name, result in results.items():
            output_path = output_dir / f"{image_name}_{model_name}_result.jpg"
            cv2.imwrite(str(output_path), result['annotated'])
            print(f"✅ Saved: {output_path}")

    def export_to_json(self, filename: str = "model_comparison_report.json"):
        """导出对比结果为JSON格式"""
        export_data = {
            'conf_threshold': self.conf_threshold,
            'models': {},
            'generated_at': self._get_current_time()
        }

        for model_name, result in self.model_results.items():
            model_data = {
                'model_path': result.model_path,
                'success': result.success,
                'error_message': result.error_message,
                'total_detections': result.total_detections,
                'avg_detection_count': result.avg_detection_count,
                'avg_inference_time': result.avg_inference_time,
                'class_counts': result.class_counts,
            }
            if hasattr(result, 'confidence_stats') and result.confidence_stats:
                model_data['confidence_stats'] = result.confidence_stats

            export_data['models'][model_name] = model_data

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Comparison report exported to: {filename}")

    def clear_results(self):
        """清空所有测试结果（保留已加载的模型）"""
        for model_name in self.model_results:
            result = self.model_results[model_name]
            result.detection_counts = []
            result.confidences = []
            result.class_counts = {}
            result.inference_times = []
            result.total_detections = 0
            result.avg_detection_count = 0.0
            result.avg_inference_time = 0.0
            if hasattr(result, 'confidence_stats'):
                result.confidence_stats = {}
        print("✅ All results cleared")

    @staticmethod
    def _get_current_time() -> str:
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_available_models(self) -> List[str]:
        """获取已成功加载的模型列表"""
        return [m for m in self.model_results if self.model_results[m].success]

    def benchmark_models(self, image_path: str, runs: int = 10) -> Dict[str, Dict]:
        """
        对模型进行基准测试（多次运行取平均）。

        Args:
            image_path: 测试图像路径
            runs: 运行次数

        Returns:
            各模型的基准测试结果
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Failed to read image: {image_path}")

        benchmark_results = {}

        for model_name, engine in self._engines.items():
            times = []
            for _ in range(runs):
                start = time.time()
                engine.detect(frame)
                times.append(time.time() - start)

            benchmark_results[model_name] = {
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': self._calculate_std(times),
                'runs': runs
            }

        return benchmark_results


def _calculate_std(values: List[float]) -> float:
    """计算标准差"""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 多模型对比测试演示")
    print("=" * 70)

    test_models = ['yolov8n.pt', 'yolov8s.pt']
    comparison = ModelComparison(test_models, conf_threshold=0.25)

    print("\n" + comparison.get_comparison_summary())

    comparison.export_to_json("model_comparison_report.json")
