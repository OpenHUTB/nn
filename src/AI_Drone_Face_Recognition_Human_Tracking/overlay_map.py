import cv2
import numpy as np
import os
import platform


class MapOverlay:
    """地图叠加类（增强容错+单独运行支持）"""

    def __init__(self, map_path="map.png", alpha=0.3):
        """
        :param map_path: 地图图片路径
        :param alpha: 地图透明度（0-1，越小越透明）
        """
        self.alpha = alpha
        self.map_img = None
        self._load_map(map_path)  # 加载地图（兼容无文件场景）

    def _load_map(self, map_path):
        """加载地图图片，无文件时生成默认地图"""
        try:
            if os.path.exists(map_path):
                self.map_img = cv2.imread(map_path)
                if self.map_img is None:
                    raise Exception("地图文件损坏或格式不支持")
                print(f"✅ 成功加载地图: {map_path}")
            else:
                # 生成默认空白地图（带文字提示）
                self.map_img = np.zeros((200, 300, 3), dtype=np.uint8)
                cv2.putText(
                    self.map_img, "DEFAULT MAP", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                cv2.putText(
                    self.map_img, f"({map_path} not found)", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )
                print(f"⚠️ 未找到地图文件: {map_path}，使用默认地图")
        except Exception as e:
            # 异常时仍生成默认地图
            self.map_img = np.zeros((200, 300, 3), dtype=np.uint8)
            cv2.putText(
                self.map_img, "MAP LOAD FAILED", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            print(f"❌ 加载地图失败: {str(e)}")

    def overlay(self, frame, drone_pos=(0, 0)):
        """
        将地图叠加到视频帧右上角
        :param frame: 原始视频帧
        :param drone_pos: 无人机位置（用于绘制标记）
        :return: 叠加后的帧
        """
        if self.map_img is None or frame is None:
            return frame  # 无地图/无帧时直接返回原帧

        h, w = frame.shape[:2]
        map_h, map_w = self.map_img.shape[:2]

        # 调整地图大小（适配帧尺寸，占帧的1/4）
        target_map_w = int(w / 4)
        target_map_h = int(target_map_w * (map_h / map_w))  # 保持宽高比
        map_resized = cv2.resize(self.map_img, (target_map_w, target_map_h))
        map_h, map_w = map_resized.shape[:2]

        # 叠加位置（右上角，留10px边距）
        x_offset = w - map_w - 10
        y_offset = 10

        # 边界校验（避免地图超出帧范围）
        if x_offset < 0: x_offset = 10
        if y_offset < 0: y_offset = 10
        if (y_offset + map_h) > h: map_h = h - y_offset - 10
        if (x_offset + map_w) > w: map_w = w - x_offset - 10

        # 透明度混合（仅叠加有效区域）
        roi = frame[y_offset:y_offset + map_h, x_offset:x_offset + map_w]
        map_cropped = map_resized[:map_h, :map_w]  # 裁剪地图适配ROI
        blended = cv2.addWeighted(roi, 1 - self.alpha, map_cropped, self.alpha, 0)
        frame[y_offset:y_offset + map_h, x_offset:x_offset + map_w] = blended

        # 绘制无人机位置标记（地图中心）
        drone_x = x_offset + map_w // 2
        drone_y = y_offset + map_h // 2
        cv2.circle(frame, (drone_x, drone_y), 5, (0, 0, 255), -1)  # 红色圆点
        cv2.putText(
            frame, "Drone", (drone_x - 20, drone_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )
        return frame

    def adjust_alpha(self, new_alpha):
        """调整地图透明度（调试用）"""
        if 0.0 <= new_alpha <= 1.0:
            self.alpha = new_alpha
            print(f"✅ 地图透明度已调整为: {new_alpha}")
        else:
            print("⚠️ 透明度范围需在0.0-1.0之间！")


# ===================== 独立运行测试逻辑 =====================
if __name__ == "__main__":
    # 初始化地图叠加器（可自定义地图路径）
    map_overlay = MapOverlay(map_path="map.png", alpha=0.3)

    # 打印系统信息
    print("\n" + "=" * 50)
    print("🎯 地图叠加模块测试工具（独立运行模式）")
    print(f"💻 当前系统: {platform.system()}")
    print(f"🗺️ 地图透明度: {map_overlay.alpha}")
    print("=" * 50)

    # 打开摄像头（默认0号设备）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        exit(1)

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 操作提示
    print("\n📢 操作说明：")
    print("  ↑ → 增加地图透明度")
    print("  ↓ → 降低地图透明度")
    print("  s → 保存当前叠加画面")
    print("  q → 退出程序")
    print("-" * 30)

    save_count = 0  # 保存图片计数
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面！")
            break

        # 叠加地图
        frame_overlay = map_overlay.overlay(frame)

        # 显示叠加后的画面
        cv2.imshow("Map Overlay Test (overlay_map.py)", frame_overlay)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🔚 退出地图叠加测试程序...")
            break

        elif key == ord('s'):
            # 保存当前画面
            save_path = f"overlay_test_{save_count}.jpg"
            cv2.imwrite(save_path, frame_overlay)
            print(f"✅ 已保存画面: {save_path}")
            save_count += 1

        elif key == 2490368:  # 上方向键（增加透明度）
            new_alpha = min(map_overlay.alpha + 0.1, 1.0)
            map_overlay.adjust_alpha(new_alpha)

        elif key == 2621440:  # 下方向键（降低透明度）
            new_alpha = max(map_overlay.alpha - 0.1, 0.0)
            map_overlay.adjust_alpha(new_alpha)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ 资源已释放，程序结束")