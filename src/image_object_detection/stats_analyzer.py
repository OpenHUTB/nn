import json
import os
from collections import Counter
from typing import List, Dict, Any, Optional


class DetectionStatsAnalyzer:
    """检测结果统计分析器"""

    def __init__(self):
        self.detection_history = []

    def add_result(self, detections: List[Dict], image_name: str = ""):
        """添加检测结果

        Args:
            detections: 检测结果列表
            image_name: 图像名称（可选）
        """
        self.detection_history.append({
            'image_name': image_name,
            'detections': detections,
            'count': len(detections)
        })

    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布统计

        Returns:
            类别名称到数量的字典
        """
        all_classes = []
        for result in self.detection_history:
            for det in result['detections']:
                all_classes.append(det.get('name', 'unknown'))
        return dict(Counter(all_classes))

    def get_confidence_stats(self) -> Dict[str, float]:
        """获取置信度统计（均值、最大值、最小值）

        Returns:
            包含均值、最大值、最小值和总数的字典
        """
        confidences = []
        for result in self.detection_history:
            for det in result['detections']:
                confidences.append(det.get('confidence', 0))

        if not confidences:
            return {}

        return {
            'mean': sum(confidences) / len(confidences),
            'max': max(confidences),
            'min': min(confidences),
            'count': len(confidences)
        }

    def get_spatial_stats(self) -> Dict[str, float]:
        """获取目标空间分布统计

        Returns:
            包含目标平均大小和位置信息的字典
        """
        widths = []
        heights = []
        centers_x = []
        centers_y = []

        for result in self.detection_history:
            for det in result['detections']:
                box = det.get('box', [0, 0, 0, 0])
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    widths.append(x2 - x1)
                    heights.append(y2 - y1)
                    centers_x.append((x1 + x2) / 2)
                    centers_y.append((y1 + y2) / 2)

        if not widths:
            return {}

        return {
            'avg_width': sum(widths) / len(widths),
            'avg_height': sum(heights) / len(heights),
            'avg_center_x': sum(centers_x) / len(centers_x),
            'avg_center_y': sum(centers_y) / len(centers_y),
            'max_width': max(widths),
            'max_height': max(heights)
        }

    def generate_report(self) -> str:
        """生成统计报告

        Returns:
            格式化的统计报告字符串
        """
        report = ["=" * 60, "📊 YOLOv8 检测结果统计报告", "=" * 60]
        report.append(f"\n📷 处理图像数量: {len(self.detection_history)}")

        total_detections = sum(r['count'] for r in self.detection_history)
        report.append(f"🎯 检测目标总数: {total_detections}")

        if self.detection_history:
            avg_detections_per_image = total_detections / len(self.detection_history)
            report.append(f"📈 每图平均检测数: {avg_detections_per_image:.2f}")

        # 类别分布
        class_dist = self.get_class_distribution()
        if class_dist:
            report.append("\n📋 类别分布统计:")
            sorted_classes = sorted(class_dist.items(), key=lambda x: -x[1])
            for cls, count in sorted_classes:
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                report.append(f"  ├─ {cls}: {count}个 ({percentage:.1f}%)")

        # 置信度统计
        conf_stats = self.get_confidence_stats()
        if conf_stats:
            report.append("\n🎯 置信度统计:")
            report.append(f"  ├─ 平均置信度: {conf_stats['mean']:.2%}")
            report.append(f"  ├─ 最高置信度: {conf_stats['max']:.2%}")
            report.append(f"  └─ 最低置信度: {conf_stats['min']:.2%}")

        # 空间分布统计
        spatial_stats = self.get_spatial_stats()
        if spatial_stats:
            report.append("\n📐 目标空间分布:")
            report.append(f"  ├─ 平均宽度: {spatial_stats['avg_width']:.1f}px")
            report.append(f"  ├─ 平均高度: {spatial_stats['avg_height']:.1f}px")
            report.append(f"  └─ 最大目标: {spatial_stats['max_width']:.0f}x{spatial_stats['max_height']:.0f}px")

        report.append("\n" + "=" * 60)
        report.append("📝 报告生成时间: " + self._get_current_time())
        report.append("=" * 60)

        return "\n".join(report)

    def save_report(self, filename: str = "detection_report.txt"):
        """保存报告到文件

        Args:
            filename: 输出文件名，默认为 detection_report.txt
        """
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        print(f"✅ 统计报告已保存到: {filename}")

    def export_to_json(self, filename: str = "detection_stats.json"):
        """导出统计数据为JSON格式

        Args:
            filename: 输出文件名，默认为 detection_stats.json
        """
        stats = {
            'total_images': len(self.detection_history),
            'total_detections': sum(r['count'] for r in self.detection_history),
            'class_distribution': self.get_class_distribution(),
            'confidence_stats': self.get_confidence_stats(),
            'spatial_stats': self.get_spatial_stats(),
            'generated_at': self._get_current_time()
        }

        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ 统计数据已导出到: {filename}")

    def clear_history(self):
        """清空检测历史记录"""
        self.detection_history = []
        print("✅ 检测历史已清空")

    @staticmethod
    def _get_current_time() -> str:
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 测试示例
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = DetectionStatsAnalyzer()

    # 添加模拟检测结果
    sample_detections = [
        {
            'name': 'car',
            'confidence': 0.95,
            'box': [100, 200, 300, 400]
        },
        {
            'name': 'person',
            'confidence': 0.88,
            'box': [400, 150, 500, 350]
        },
        {
            'name': 'car',
            'confidence': 0.72,
            'box': [50, 300, 200, 450]
        }
    ]

    analyzer.add_result(sample_detections, "street_view.jpg")

    # 添加更多模拟数据
    analyzer.add_result([
        {'name': 'dog', 'confidence': 0.91, 'box': [50, 50, 150, 150]},
        {'name': 'cat', 'confidence': 0.85, 'box': [200, 80, 300, 180]}
    ], "park.jpg")

    analyzer.add_result([
        {'name': 'bicycle', 'confidence': 0.78, 'box': [100, 100, 250, 250]}
    ], "road.jpg")

    # 生成并打印报告
    print("\n" + analyzer.generate_report())

    # 保存报告
    analyzer.save_report()
    analyzer.export_to_json()