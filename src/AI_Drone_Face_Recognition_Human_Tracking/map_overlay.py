
import cv2
import numpy as np


class MapOverlay:
    def __init__(self, map_path="map.png", alpha=0.3):
        self.map_path = map_path
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.map_img = self.load_map()

    def load_map(self):
        """加载地图，失败则生成默认地图"""
        try:
            map_img = cv2.imread(self.map_path)
            if map_img is None:
                raise ValueError("地图图片读取失败")
            return map_img
        except Exception as e:
            print(f"加载地图失败：{e}，使用默认地图")
            # 生成默认地图（带网格的简易地图）
            default_map = np.ones((200, 300, 3), dtype=np.uint8) * 240
            # 绘制网格
            for x in range(0, 300, 30):
                cv2.line(default_map, (x, 0), (x, 200), (200, 200, 200), 1)
            for y in range(0, 200, 20):
                cv2.line(default_map, (0, y), (300, y), (200, 200, 200), 1)
            cv2.putText(default_map, "DEFAULT MAP", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
            return default_map

    def overlay(self, frame):
        """将地图叠加到帧的右上角"""
        if frame is None or self.map_img is None:
            return frame

        h, w = frame.shape[:2]
        map_h, map_w = self.map_img.shape[:2]

        # 缩放地图（占视频宽度的25%）
        target_w = int(w * 0.25)
        scale = target_w / map_w
        target_h = int(map_h * scale)
        resized_map = cv2.resize(self.map_img, (target_w, target_h))

        # 计算叠加位置（右上角，留10像素边距）
        x_start = w - target_w - 10
        y_start = 10
        x_end = x_start + target_w
        y_end = y_start + target_h

        # 边界校验（防止地图超出视频帧范围）
        x_end = min(x_end, w)
        y_end = min(y_end, h)
        resized_map = resized_map[:y_end - y_start, :x_end - x_start]

        # 透明度混合
        roi = frame[y_start:y_end, x_start:x_end]
        blended = cv2.addWeighted(resized_map, self.alpha, roi, 1 - self.alpha, 0)
        frame[y_start:y_end, x_start:x_end] = blended

        # 绘制无人机位置标记（地图中心）
        map_center_x = x_start + (x_end - x_start) // 2
        map_center_y = y_start + (y_end - y_start) // 2
        cv2.circle(frame, (map_center_x, map_center_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Drone", (map_center_x + 10, map_center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return frame

    def adjust_alpha(self, delta):
        """调整地图透明度"""
        old_alpha = self.alpha
        self.alpha = np.clip(self.alpha + delta, 0.0, 1.0)
        if old_alpha != self.alpha:  # 仅在透明度变化时打印
            print(f"✅ 地图透明度调整为：{self.alpha:.1f}")
        else:
            print(f"⚠️  透明度已达极限（{self.alpha:.1f}），无法继续调整")


def main():
    # 初始化地图叠加器
    map_overlay = MapOverlay(alpha=0.4)

    # 选择视频源：0=摄像头，也可以替换为视频文件路径（如"test.mp4"）
    video_source = 0
    cap = cv2.VideoCapture(video_source)

    # 检查视频源是否打开
    if not cap.isOpened():
        print(f"❌ 无法打开视频源：{video_source}")
        return

    # 设置窗口可调整大小（方便操作）
    cv2.namedWindow("Map Overlay Demo", cv2.WINDOW_NORMAL)

    print("=" * 60)
    print("📢 操作说明：")
    print("  W / 上方向键：增加地图透明度（+0.1）")
    print("  S / 下方向键：降低地图透明度（-0.1）")
    print("  Q / ESC键    ：退出程序")
    print("=" * 60)

    # 主循环
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("❌ 视频流已结束或读取失败")
            break

        # 叠加地图
        frame_with_map = map_overlay.overlay(frame)

        # 显示操作提示和当前透明度
        cv2.putText(frame_with_map, "W/S: Adjust Alpha | Q/ESC: Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_map, f"Alpha: {map_overlay.alpha:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 显示结果
        cv2.imshow("Map Overlay Demo", frame_with_map)

        # 键盘交互（优化跨平台按键检测，等待时间调整为30ms提高响应性）
        key = cv2.waitKey(30) & 0xFF

        # 退出逻辑（Q键 / ESC键）
        if key == ord('q') or key == ord('Q') or key == 27:
            print("📤 退出程序")
            break

        # 增加透明度（W键 / 上方向键）
        elif key == ord('w') or key == ord('W') or key == 82 or key == 104:
            map_overlay.adjust_alpha(0.1)

        # 降低透明度（S键 / 下方向键）
        elif key == ord('s') or key == ord('S') or key == 84 or key == 101:
            map_overlay.adjust_alpha(-0.1)

        # 调试：打印未知按键编码（方便排查问题）
        elif key != 255:
            print(f"🔍 检测到未映射按键，编码：{key}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()