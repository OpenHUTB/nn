"""
utils.py - 通用工具函数
包含：图像处理、几何计算、性能监控、文件操作等工具函数
"""

import cv2
import numpy as np
import time
import os
import sys
from numba import njit
from datetime import datetime

# 配置loguru logger
# 配置日志
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# 尝试导入yaml，如果失败提供友好的错误信息
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML未安装，配置文件功能将受限")

def valid_img(img):
    """
    检查图像是否有效
    
    Args:
        img: 输入图像
        
    Returns:
        bool: 图像是否有效
    """
    return img is not None and len(img.shape) == 3 and img.shape[2] == 3 and img.size > 0

def clip_box(bbox, img_shape):
    """
    裁剪边界框到图像范围内
    
    Args:
        bbox: [x1, y1, x2, y2] 边界框坐标
        img_shape: (height, width) 图像尺寸
        
    Returns:
        np.ndarray: 裁剪后的边界框
    """
    h, w = img_shape[:2]
    return np.array([
        max(0, min(bbox[0], w - 1)),
        max(0, min(bbox[1], h - 1)),
        max(bbox[0] + 1, min(bbox[2], w - 1)),
        max(bbox[1] + 1, min(bbox[3], h - 1))
    ], dtype=np.float32)

def make_div(x, d=32):
    """
    将数值调整为d的倍数（用于YOLO输入尺寸）
    
    Args:
        x: 原始数值
        d: 倍数（默认为32）
        
    Returns:
        int: 调整后的数值
    """
    return (x + d - 1) // d * d

def resize_with_padding(image, target_size, color=(114, 114, 114)):
    """
    保持长宽比的resize，用指定颜色填充
    
    Args:
        image: 输入图像
        target_size: (width, height) 目标尺寸
        color: 填充颜色
        
    Returns:
        tuple: (resized_image, scale, padding)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    if scale != 1:
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 创建填充图像
    padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    # 计算填充位置（居中）
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    
    # 放置图像
    padded[dy:dy + new_h, dx:dx + new_w] = image
    
    return padded, scale, (dx, dy)

@njit
def iou_numpy(box1, box2):
    """
    计算两个边界框的IoU（交并比）- 使用numpy数组版本
    
    Args:
        box1: np.array([x1, y1, x2, y2])
        box2: np.array([x1, y1, x2, y2])
        
    Returns:
        float: IoU值
    """
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    
    ia = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    ua = a1 + a2 - ia
    
    return ia / ua if ua > 0 else 0.0

def iou(box1, box2):
    """
    计算两个边界框的IoU（兼容list和numpy数组）
    
    Args:
        box1: [x1, y1, x2, y2] 或 np.array
        box2: [x1, y1, x2, y2] 或 np.array
        
    Returns:
        float: IoU值
    """
    # 转换为numpy数组
    box1_np = np.array(box1, dtype=np.float32)
    box2_np = np.array(box2, dtype=np.float32)
    return iou_numpy(box1_np, box2_np)

@njit
def iou_batch(boxes1, boxes2):
    """
    批量计算IoU矩阵
    
    Args:
        boxes1: (N, 4) 边界框数组
        boxes2: (M, 4) 边界框数组
        
    Returns:
        np.ndarray: (N, M) IoU矩阵
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    iou_matrix = np.zeros((N, M), dtype=np.float32)
    
    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = iou_numpy(boxes1[i], boxes2[j])
    
    return iou_matrix

def bbox_center(bbox):
    """
    计算边界框中心点
    
    Args:
        bbox: [x1, y1, x2, y2] 边界框
        
    Returns:
        tuple: (cx, cy) 中心点坐标
    """
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def bbox_area(bbox):
    """
    计算边界框面积
    
    Args:
        bbox: [x1, y1, x2, y2] 边界框
        
    Returns:
        float: 边界框面积
    """
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

def bbox_aspect_ratio(bbox):
    """
    计算边界框宽高比
    
    Args:
        bbox: [x1, y1, x2, y2] 边界框
        
    Returns:
        float: 宽高比（宽/高）
    """
    width = max(0.1, bbox[2] - bbox[0])
    height = max(0.1, bbox[3] - bbox[1])
    return width / height

class FPSCounter:
    """
    FPS计数器
    """
    
    def __init__(self, window_size=15):
        """
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.timestamps = []
        self.fps = 0.0
        self.avg_fps = 0.0
        self.fps_history = []
        
    def update(self):
        """
        更新FPS计数
        
        Returns:
            float: 当前FPS
        """
        self.timestamps.append(time.time())
        
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        if len(self.timestamps) >= 2:
            self.fps = (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])
            self.fps_history.append(self.fps)
            
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
            
            self.avg_fps = np.mean(self.fps_history) if self.fps_history else self.fps
        
        return self.fps
    
    def reset(self):
        """重置计数器"""
        self.timestamps = []
        self.fps = 0.0
        self.fps_history = []
        self.avg_fps = 0.0

class PerformanceMonitor:
    """
    性能监控器
    """
    
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_times = []
        self.detection_times = []
        self.tracking_times = []
        
    def start_frame(self):
        """开始新帧计时"""
        self.frame_start = time.time()
        
    def end_frame(self):
        """结束帧计时"""
        frame_time = time.time() - self.frame_start
        self.frame_times.append(frame_time)
        self.frame_count += 1
        
        # 保留最近100帧的计时
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        
    def record_detection_time(self, dt):
        """记录检测时间"""
        self.detection_times.append(dt)
        if len(self.detection_times) > 100:
            self.detection_times.pop(0)
    
    def record_tracking_time(self, dt):
        """记录跟踪时间"""
        self.tracking_times.append(dt)
        if len(self.tracking_times) > 100:
            self.tracking_times.pop(0)
    
    def get_stats(self):
        """获取性能统计"""
        stats = {
            'total_frames': self.frame_count,
            'total_time': time.time() - self.start_time,
            'avg_fps': len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0,
            'avg_frame_time': np.mean(self.frame_times) * 1000 if self.frame_times else 0,
            'avg_detection_time': np.mean(self.detection_times) * 1000 if self.detection_times else 0,
            'avg_tracking_time': np.mean(self.tracking_times) * 1000 if self.tracking_times else 0,
        }
        return stats
    
    def print_stats(self):
        """打印性能统计"""
        stats = self.get_stats()
        logger.info(f"总帧数: {stats['total_frames']}")
        logger.info(f"总时间: {stats['total_time']:.1f}s")
        logger.info(f"平均FPS: {stats['avg_fps']:.1f}")
        logger.info(f"平均帧时间: {stats['avg_frame_time']:.1f}ms")
        logger.info(f"平均检测时间: {stats['avg_detection_time']:.1f}ms")
        logger.info(f"平均跟踪时间: {stats['avg_tracking_time']:.1f}ms")

def create_output_dir(base_dir="outputs"):
    """
    创建输出目录
    
    Args:
        base_dir: 基础目录名
        
    Returns:
        str: 创建的目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "screenshots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    logger.info(f"创建输出目录: {output_dir}")
    return output_dir

def save_image(image, path, create_dir=True):
    """
    保存图像
    
    Args:
        image: 要保存的图像
        path: 保存路径
        create_dir: 是否创建目录
        
    Returns:
        bool: 是否保存成功
    """
    if not valid_img(image):
        logger.warning(f"无效图像，无法保存到 {path}")
        return False
    
    try:
        if create_dir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        cv2.imwrite(path, image)
        logger.debug(f"图像已保存: {path}")
        return True
        
    except Exception as e:
        logger.error(f"保存图像失败 {path}: {e}")
        return False

def load_yaml_config(path):
    """
    加载YAML配置文件
    
    Args:
        path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    if not YAML_AVAILABLE:
        logger.error("无法加载YAML配置: PyYAML未安装")
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {path}")
        return config if config else {}
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {path}")
        return {}
    except Exception as e:
        logger.error(f"加载配置文件失败 {path}: {e}")
        return {}

def save_yaml_config(config, path):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        path: 保存路径
    """
    if not YAML_AVAILABLE:
        logger.error("无法保存YAML配置: PyYAML未安装")
        return
    
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.debug(f"配置已保存: {path}")
    except Exception as e:
        logger.error(f"保存配置失败 {path}: {e}")

def draw_bbox(image, bbox, color=(255, 0, 0), thickness=2, label=None):
    """
    在图像上绘制单个边界框
    
    Args:
        image: 输入图像
        bbox: [x1, y1, x2, y2] 边界框
        color: 颜色 (B, G, R)
        thickness: 线宽
        label: 标签文本
        
    Returns:
        np.ndarray: 绘制后的图像
    """
    if not valid_img(image):
        return image
    
    x1, y1, x2, y2 = map(int, bbox)
    
    # 检查坐标有效性
    if x1 >= x2 or y1 >= y2:
        return image
    
    # 绘制边界框
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # 绘制标签
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        
        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 绘制标签背景
        cv2.rectangle(image, (x1, y1 - text_height - 5),
                     (x1 + text_width, y1), color, -1)
        
        # 绘制文本
        cv2.putText(image, label, (x1, y1 - 5),
                   font, font_scale, (255, 255, 255), thickness)
    
    return image

def draw_trajectory(image, points, color=(0, 255, 0), thickness=2, max_points=20):
    """
    在图像上绘制轨迹
    
    Args:
        image: 输入图像
        points: 轨迹点列表 [(x1, y1), (x2, y2), ...]
        color: 轨迹颜色
        thickness: 线宽
        max_points: 最大显示点数
        
    Returns:
        np.ndarray: 绘制后的图像
    """
    if not valid_img(image) or len(points) < 2:
        return image
    
    # 限制轨迹点数量
    points = points[-max_points:]
    
    # 绘制轨迹线
    for i in range(1, len(points)):
        pt1 = (int(points[i-1][0]), int(points[i-1][1]))
        pt2 = (int(points[i][0]), int(points[i][1]))
        
        # 检查点是否有效
        if 0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and \
           0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]:
            cv2.line(image, pt1, pt2, color, thickness)
    
    return image

def draw_info_panel(image, info_dict, position="top_left"):
    """
    在图像上绘制信息面板
    
    Args:
        image: 输入图像
        info_dict: 信息字典 {key: value}
        position: 位置 ("top_left", "top_right", "bottom_left", "bottom_right")
        
    Returns:
        np.ndarray: 绘制后的图像
    """
    if not valid_img(image):
        return image
    
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 25
    
    # 确定起始位置
    if position == "top_left":
        x, y = 10, 30
    elif position == "top_right":
        x, y = w - 200, 30
    elif position == "bottom_left":
        x, y = 10, h - 30 - len(info_dict) * line_height
    elif position == "bottom_right":
        x, y = w - 200, h - 30 - len(info_dict) * line_height
    else:
        x, y = 10, 30
    
    # 绘制信息背景
    bg_height = len(info_dict) * line_height + 10
    cv2.rectangle(image, (x - 5, y - 25), (x + 190, y + bg_height - 20), (0, 0, 0), -1)
    
    # 绘制标题
    cv2.putText(image, "SYSTEM INFO", (x, y - 5), font, 0.7, (0, 255, 0), thickness)
    
    # 绘制信息项
    for i, (key, value) in enumerate(info_dict.items()):
        text = f"{key}: {value}"
        cv2.putText(image, text, (x, y + (i + 1) * line_height), 
                   font, font_scale, (255, 255, 255), thickness)
    
    return image

def run_self_tests():
    """运行自测试"""
    print("=" * 50)
    print("运行 utils.py 自测试...")
    print("=" * 50)
    
    tests_passed = 0
    tests_failed = 0
    
    # 测试 1: valid_img
    try:
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        assert valid_img(test_img) == True, "valid_img应该返回True"
        assert valid_img(None) == False, "valid_img(None)应该返回False"
        assert valid_img(np.zeros((100, 100), dtype=np.uint8)) == False, "灰度图应该返回False"
        print("✅ valid_img测试通过")
        tests_passed += 1
    except AssertionError as e:
        print(f"❌ valid_img测试失败: {e}")
        tests_failed += 1
    
    # 测试 2: clip_box
    try:
        bbox = [10, 10, 200, 200]
        clipped = clip_box(bbox, (150, 150))
        expected = [10, 10, 149, 149]  # 索引从0开始，所以是149不是150
        assert np.allclose(clipped[:2], expected[:2]), f"clip_box坐标错误: {clipped[:2]} != {expected[:2]}"
        assert clipped[2] <= 149 and clipped[3] <= 149, "clip_box应该限制在图像范围内"
        print("✅ clip_box测试通过")
        tests_passed += 1
    except AssertionError as e:
        print(f"❌ clip_box测试失败: {e}")
        tests_failed += 1
    
    # 测试 3: iou (兼容性版本)
    try:
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou_val = iou(box1, box2)
        expected_iou = 25 / (100 + 100 - 25)  # (5x5)/(100+100-25) = 25/175 ≈ 0.1429
        assert abs(iou_val - expected_iou) < 0.001, f"iou计算错误: {iou_val} != {expected_iou}"
        
        # 测试numpy数组版本
        box1_np = np.array(box1, dtype=np.float32)
        box2_np = np.array(box2, dtype=np.float32)
        iou_val_np = iou_numpy(box1_np, box2_np)
        assert abs(iou_val_np - expected_iou) < 0.001, f"iou_numpy计算错误"
        
        print("✅ iou测试通过")
        tests_passed += 1
    except AssertionError as e:
        print(f"❌ iou测试失败: {e}")
        tests_failed += 1
    
    # 测试 4: make_div
    try:
        assert make_div(100) == 128, "make_div(100)应该返回128"
        assert make_div(128) == 128, "make_div(128)应该返回128"
        assert make_div(129) == 160, "make_div(129)应该返回160"
        assert make_div(0, 32) == 0, "make_div(0)应该返回0"
        print("✅ make_div测试通过")
        tests_passed += 1
    except AssertionError as e:
        print(f"❌ make_div测试失败: {e}")
        tests_failed += 1
    
    # 测试 5: FPSCounter (修复的测试)
    try:
        fps_counter = FPSCounter(window_size=3)
        
        # 第一次update会初始化但不会计算FPS（需要至少2个时间点）
        fps1 = fps_counter.update()
        time.sleep(0.05)  # 等待50ms
        
        # 第二次update才会计算FPS
        fps2 = fps_counter.update()
        time.sleep(0.05)
        
        fps3 = fps_counter.update()
        
        # 现在应该有FPS值了
        assert fps3 > 0, f"FPS应该大于0，当前: {fps3}"
        assert fps_counter.fps > 0, f"内部FPS应该大于0"
        
        print(f"✅ FPSCounter测试通过 (FPS: {fps3:.1f})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FPSCounter测试失败: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # 测试 6: 可视化函数
    try:
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # 测试draw_bbox
        result1 = draw_bbox(test_img.copy(), [10, 10, 50, 50], label="test")
        assert result1.shape == test_img.shape, "draw_bbox应该返回相同尺寸的图像"
        
        # 测试draw_trajectory
        points = [(20, 20), (30, 30), (40, 40)]
        result2 = draw_trajectory(test_img.copy(), points)
        assert result2.shape == test_img.shape, "draw_trajectory应该返回相同尺寸的图像"
        
        # 测试draw_info_panel
        info = {"FPS": "30.0", "Objects": "5"}
        result3 = draw_info_panel(test_img.copy(), info)
        assert result3.shape == test_img.shape, "draw_info_panel应该返回相同尺寸的图像"
        
        print("✅ 可视化函数测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 可视化函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # 测试 7: bbox工具函数
    try:
        bbox = [10, 20, 50, 80]
        center = bbox_center(bbox)
        area = bbox_area(bbox)
        aspect = bbox_aspect_ratio(bbox)
        
        assert center == (30.0, 50.0), f"中心点计算错误: {center}"
        assert area == 40 * 60, f"面积计算错误: {area}"
        assert abs(aspect - 40/60) < 0.001, f"宽高比计算错误: {aspect}"
        
        print("✅ bbox工具函数测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"❌ bbox工具函数测试失败: {e}")
        tests_failed += 1
    
    print("=" * 50)
    print(f"测试结果: {tests_passed}通过, {tests_failed}失败")
    
    if tests_failed == 0:
        print("🎉 所有测试通过!")
    else:
        print("⚠️  有测试失败，请检查")
    
    return tests_failed == 0

if __name__ == "__main__":
    # 运行自测试
    success = run_self_tests()
    
    if success:
        print("\nutils.py 可以安全使用")
        sys.exit(0)
    else:
        print("\n⚠️ utils.py 有测试失败，请修复")
        sys.exit(1)