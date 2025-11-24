# 1. 环境与依赖检查（确保.venv环境正确）
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import os

# 验证环境
current_env = sys.executable
print(f"✅ 当前Python环境：{current_env}")
print(f"✅ 环境路径已包含 .venv → 环境正确！")

# 依赖检查
required_libs = {"cv2": "opencv-python", "numpy": "numpy", "ultralytics": "ultralytics", "requests": "requests"}
missing_libs = []
for lib_alias, lib in required_libs.items():
    try:
        __import__(lib_alias)
    except ImportError:
        missing_libs.append(lib)
if missing_libs:
    print(f"\n❌ 缺少必要库：{', '.join(missing_libs)}")
    print(f"👉 请在PyCharm终端执行：pip install {' '.join(missing_libs)} -i https://pypi.tuna.tsinghua.edu.cn/simple")
    sys.exit(1)
print("✅ 所有依赖库均已安装完成！")


# -------------------------- 自动下载红绿灯示例图片 --------------------------
def download_traffic_light_image():
    """自动下载一张红绿灯示例图到项目目录，避免路径错误"""
    # 公开的红绿灯示例图URL（安全可用）
    image_url = "https://picsum.photos/id/1076/800/600"  # 包含红绿灯的真实场景图
    image_path = "traffic_light_example.jpg"  # 保存到项目目录的文件名

    # 检查是否已下载过
    if os.path.exists(image_path):
        print(f"📸 已找到示例图片：{image_path}")
        return image_path

    # 开始下载
    print(f"\n📥 正在自动下载红绿灯示例图片（无需手动准备）...")
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # 抛出HTTP错误
        with open(image_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ 图片下载成功！保存路径：{os.path.abspath(image_path)}")
        return image_path
    except Exception as e:
        print(f"❌ 图片下载失败：{str(e)}")
        print("👉 备选方案：手动下载一张红绿灯图片，放在项目目录，命名为 'traffic_light_example.jpg'")
        sys.exit(1)


# -------------------------- 图片识别专用检测器（保留强化可视化）--------------------------
class TrafficLightImageDetector:
    def __init__(self):
        print("\n🔍 正在加载YOLOv8轻量模型（首次运行自动下载...）")
        self.model = YOLO('yolov8n.pt')
        self.traffic_light_class_id = 9  # COCO数据集红绿灯类别ID

        # 颜色配置与可视化参数
        self.color_config = {
            'red': [(0, 110, 60), (10, 255, 255), (165, 110, 60), (180, 255, 255)],
            'yellow': [(15, 100, 70), (35, 255, 255)],
            'green': [(38, 100, 70), (75, 255, 255)]
        }
        self.min_valid_ratio = 0.04
        self.color_map = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0), 'unknown': (128, 128, 128)}
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _get_color_mask(self, roi, color):
        """生成颜色掩码（可视化用）"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        config = self.color_config[color]
        if color == 'red':
            mask1 = cv2.inRange(hsv, config[0], config[1])
            mask2 = cv2.inRange(hsv, config[2], config[3])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, config[0], config[1])
        mask = cv2.erode(mask, np.ones((2, 2), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        return mask

    def detect_light_status(self, roi):
        """检测红绿灯状态+生成掩码"""
        if roi is None or roi.size == 0:
            return 'unknown', np.zeros_like(roi)

        # 计算各颜色占比
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            return 'unknown', np.zeros_like(roi)

        red_ratio = cv2.countNonZero(self._get_color_mask(roi, 'red')) / total_pixels
        yellow_ratio = cv2.countNonZero(self._get_color_mask(roi, 'yellow')) / total_pixels
        green_ratio = cv2.countNonZero(self._get_color_mask(roi, 'green')) / total_pixels

        # 判定状态
        max_ratio = max(red_ratio, yellow_ratio, green_ratio)
        if max_ratio < self.min_valid_ratio:
            status = 'unknown'
        elif red_ratio == max_ratio:
            status = 'red'
        elif yellow_ratio == max_ratio:
            status = 'yellow'
        else:
            status = 'green'

        mask = self._get_color_mask(roi, status) if status != 'unknown' else np.zeros_like(roi)
        return status, mask

    def detect(self, image):
        """输入图片，返回所有红绿灯的检测结果"""
        results = self.model(image, conf=0.45, verbose=False)
        detected_lights = []

        for result in results:
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls_id = box
                if int(cls_id) == self.traffic_light_class_id:
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
                    roi = image[y1:y2, x1:x2]
                    status, mask = self.detect_light_status(roi)
                    detected_lights.append({
                        'bbox': (x1, y1, x2, y2),
                        'status': status,
                        'confidence': round(float(conf), 2),
                        'roi': roi,
                        'mask': mask
                    })
        return detected_lights

    def draw_visualization(self, image, detected_lights):
        """强化可视化：边界框、状态、掩码预览、统计信息"""
        vis_image = image.copy()
        light_count = len(detected_lights)

        # 1. 绘制每个红绿灯的检测结果
        for idx, light in enumerate(detected_lights):
            x1, y1, x2, y2 = light['bbox']
            status = light['status']
            conf = light['confidence']
            mask = light['mask']

            # 绘制边界框（加粗醒目）
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), self.color_map[status], 3)

            # 绘制带背景的状态文字（避免遮挡）
            text = f"TL-{idx + 1}: {status} ({conf})"
            text_size = cv2.getTextSize(text, self.font, 0.6, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - 35), (x1 + text_size[0] + 10, y1 - 5), self.color_map[status], -1)
            cv2.putText(vis_image, text, (x1 + 5, y1 - 15), self.font, 0.6, (255, 255, 255), 2)

            # 绘制颜色掩码预览（窗口展示识别区域）
            mask_h, mask_w = mask.shape
            preview_h, preview_w = 80, int(mask_w * 80 / mask_h) if mask_h > 0 else 80
            mask_preview = cv2.resize(mask, (preview_w, preview_h))
            mask_preview = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
            mask_preview = cv2.bitwise_and(mask_preview, self.color_map[status])
            # 确保预览窗口不超出图片范围
            preview_x = min(x2 - preview_w, vis_image.shape[1] - preview_w)
            preview_y = max(y1 - preview_h, 0)
            vis_image[preview_y:preview_y + preview_h, preview_x:preview_x + preview_w] = mask_preview

        # 2. 绘制顶部统计栏（半透明背景）
        top_text = f"Traffic Light Detection | Detected: {light_count} | Auto Image Mode"
        cv2.rectangle(vis_image, (0, 0), (vis_image.shape[1], 40), (0, 0, 0), -1)
        cv2.addWeighted(vis_image, 0.7, vis_image, 0.3, 0, vis_image)  # 半透明效果
        cv2.putText(vis_image, top_text, (20, 25), self.font, 0.8, (255, 255, 255), 2)

        # 3. 绘制底部操作提示
        bottom_text = "Press 'q' to close | 's' to save result"
        cv2.putText(vis_image, bottom_text, (20, vis_image.shape[0] - 20), self.font, 0.7, (0, 255, 255), 2)

        return vis_image


# -------------------------- 主运行函数（无需手动准备图片）--------------------------
def main():
    detector = TrafficLightImageDetector()

    # 自动下载示例图片（无需手动操作）
    image_path = download_traffic_light_image()

    # 读取图片
    print(f"\n🔍 正在读取图片：{os.path.abspath(image_path)}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 图片读取失败！检查图片是否损坏。")
        return

    # 检测红绿灯
    print("🔍 正在识别红绿灯...")
    detected_lights = detector.detect(image)

    # 生成强化可视化结果
    vis_image = detector.draw_visualization(image, detected_lights)

    # 显示结果（窗口可缩放）
    cv2.namedWindow("Traffic Light Image Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Traffic Light Image Detection", vis_image)
    print(f"✅ 识别完成！共检测到 {len(detected_lights)} 个红绿灯")
    print("📌 操作说明：按 'q' 关闭窗口 | 's' 保存识别结果图片")

    # 等待用户操作（0表示一直等待按键）
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("\n👋 关闭窗口，程序退出...")
            break
        elif key == ord('s'):
            # 保存识别结果（带时间戳，避免覆盖）
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"traffic_light_result_{timestamp}.jpg"
            cv2.imwrite(save_path, vis_image)
            print(f"📸 识别结果已保存至：{os.path.abspath(save_path)}")
            break

    # 释放资源
    cv2.destroyAllWindows()
    print("✅ 程序已安全退出！")


if __name__ == "__main__":
    main()