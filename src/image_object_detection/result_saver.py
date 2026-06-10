# result_saver.py
# 功能：结果保存管理器 - 支持多种格式的检测结果保存

import os
import json
import cv2
import csv
from pathlib import Path
from datetime import datetime


class ResultSaver:
    """
    结果保存管理器。
    
    支持多种格式的检测结果保存：
    - 图像文件（带检测框标注）
    - JSON格式检测数据
    - CSV格式检测报告
    - 统计分析报告
    """

    def __init__(self, output_dir="output", auto_create=True):
        """
        初始化结果保存器。
        
        参数:
            output_dir: 输出目录路径
            auto_create: 是否自动创建目录
        """
        self.output_dir = Path(output_dir)
        
        if auto_create:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.detection_records = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_image(self, image, filename, prefix="", suffix=""):
        """
        保存检测后的图像。
        
        参数:
            image: 图像数据（OpenCV格式）
            filename: 原始文件名
            prefix: 文件名前缀
            suffix: 文件名后缀
        
        返回:
            保存的文件路径
        """
        name, ext = os.path.splitext(filename)
        if prefix:
            name = f"{prefix}_{name}"
        if suffix:
            name = f"{name}_{suffix}"
        
        output_path = self.output_dir / f"{name}{ext}"
        
        if cv2.imwrite(str(output_path), image):
            return str(output_path)
        else:
            raise IOError(f"Failed to save image: {output_path}")

    def save_detection_data(self, filename, detections, frame_info=None):
        """
        保存检测数据为JSON格式。
        
        参数:
            filename: 原始文件名
            detections: 检测结果列表
            frame_info: 帧信息（可选）
        
        返回:
            保存的文件路径
        """
        data = {
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "detections": [],
        }
        
        if frame_info:
            data.update(frame_info)
        
        for det in detections:
            det_dict = {
                "class_id": int(det.cls),
                "class_name": det.name,
                "confidence": float(det.conf),
                "bbox": [float(x) for x in det.xyxy[0]],
            }
            
            if hasattr(det, 'distance'):
                det_dict["distance"] = float(det.distance)
            if hasattr(det, 'danger_level'):
                det_dict["danger_level"] = det.danger_level
            
            data["detections"].append(det_dict)
        
        name = os.path.splitext(filename)[0]
        output_path = self.output_dir / f"{name}_detections.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.detection_records.append(data)
        return str(output_path)

    def save_batch_results(self, results):
        """
        保存批量检测结果。
        
        参数:
            results: 批量检测结果列表
        
        返回:
            汇总报告路径
        """
        batch_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_images": len(results),
            "total_detections": sum(len(r["detections"]) for r in results),
            "results": results,
        }
        
        output_path = self.output_dir / f"batch_results_{self.session_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)

    def save_statistics_report(self, stats, filename="statistics"):
        """
        保存统计分析报告。
        
        参数:
            stats: 统计数据字典
            filename: 报告文件名（不含扩展名）
        
        返回:
            保存的文件路径
        """
        report_path = self.output_dir / f"{filename}_{self.session_id}.json"
        
        report_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return str(report_path)

    def save_csv_report(self, filename="detection_report"):
        """
        保存CSV格式报告。
        
        参数:
            filename: 报告文件名（不含扩展名）
        
        返回:
            保存的文件路径
        """
        output_path = self.output_dir / f"{filename}_{self.session_id}.csv"
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                "文件名", "检测数量", "检测时间", 
                "类别分布", "平均置信度", "危险目标数"
            ])
            
            for record in self.detection_records:
                filename = record.get("filename", "")
                det_count = len(record.get("detections", []))
                timestamp = record.get("timestamp", "")
                
                class_dist = {}
                total_conf = 0.0
                danger_count = 0
                
                for det in record.get("detections", []):
                    class_name = det.get("class_name", "unknown")
                    class_dist[class_name] = class_dist.get(class_name, 0) + 1
                    total_conf += det.get("confidence", 0)
                    if det.get("danger_level") == "DANGER":
                        danger_count += 1
                
                avg_conf = total_conf / det_count if det_count > 0 else 0.0
                
                class_dist_str = "; ".join([f"{k}:{v}" for k, v in class_dist.items()])
                
                writer.writerow([
                    filename, det_count, timestamp,
                    class_dist_str, f"{avg_conf:.3f}", danger_count
                ])
        
        return str(output_path)

    def save_model_comparison(self, comparison_results, filename="model_comparison"):
        """
        保存多模型对比结果。
        
        参数:
            comparison_results: 对比结果字典
            filename: 报告文件名（不含扩展名）
        
        返回:
            保存的文件路径
        """
        output_path = self.output_dir / f"{filename}_{self.session_id}.json"
        
        report_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "comparison": comparison_results,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)

    def get_session_dir(self):
        """获取当前会话目录"""
        return str(self.output_dir)

    def get_session_id(self):
        """获取当前会话ID"""
        return self.session_id

    def clear_records(self):
        """清空检测记录"""
        self.detection_records = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
