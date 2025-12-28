"""
无人机视觉导航系统 - 简化版
可以立即运行测试
"""

import os
import sys
import time
import cv2
import numpy as np

# 添加项目路径，让Python能找到src模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def ensure_directories():
    """确保所有目录都存在"""
    dirs = ['data/images', 'data/videos', 'data/logs', 'data/config']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ 目录已就绪: {d}")


class SimpleDroneCamera:
    """简单的无人机摄像头类"""

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None

    def open(self):
        """打开摄像头"""
        print(f"尝试打开摄像头 {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print("⚠️  无法打开物理摄像头，使用模拟模式")
            return False
        else:
            print("✓ 摄像头已连接")
            return True

    def read_frame(self):
        """读取一帧"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame

        # 如果没有摄像头或读取失败，返回模拟图像
        return self.simulate_frame()

    def simulate_frame(self):
        """生成模拟图像"""
        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 添加一些图形
        cv2.putText(frame, "无人机模拟视图", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
        cv2.circle(frame, (400, 200), 50, (0, 0, 255), -1)

        return frame

    def release(self):
        """释放摄像头"""
        if self.cap:
            self.cap.release()
            print("摄像头已释放")


def analyze_scene_simple(frame):
    """简单场景分析（基于颜色）"""
    if frame is None:
        return "未知", 0.5

    # 转换为HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 检测绿色（植被）
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_pct = np.sum(green_mask > 0) / green_mask.size

    # 检测蓝色（水域）
    blue_mask = cv2.inRange(hsv, (100, 40, 40), (140, 255, 255))
    blue_pct = np.sum(blue_mask > 0) / blue_mask.size

    # 判断场景
    if green_pct > 0.3:
        return "森林/草地", green_pct
    elif blue_pct > 0.2:
        return "水域", blue_pct
    else:
        return "城市/建筑", max(green_pct, blue_pct)


def get_decision(scene_type, confidence):
    """根据场景类型做出决策"""
    decisions = {
        "森林/草地": "✓ 安全区域，继续飞行",
        "水域": "⚠️  接近水域，提高飞行高度",
        "城市/建筑": "⚠️  城市区域，降低速度并避让",
        "未知": "? 无法识别，保持警戒"
    }
    return decisions.get(scene_type, "保持当前状态")


def main():
    """主函数"""
    print("=" * 50)
    print("无人机视觉导航系统")
    print("版本: 1.0.0")
    print("按 'q' 键退出，按 's' 键保存图像")
    print("=" * 50)

    # 确保目录存在
    ensure_directories()

    # 创建无人机摄像头
    drone_cam = SimpleDroneCamera(camera_id=0)
    drone_cam.open()

    # 初始化状态
    battery = 100
    flight_time = 0
    frame_count = 0
    start_time = time.time()

    print("\n开始飞行...")

    while True:
        # 读取帧
        frame = drone_cam.read_frame()
        frame_count += 1

        # 分析场景
        scene_type, confidence = analyze_scene_simple(frame)

        # 获取决策
        decision = get_decision(scene_type, confidence)

        # 更新状态
        flight_time = time.time() - start_time
        battery = max(0, battery - 0.05)  # 慢慢消耗电池

        # 在图像上显示信息
        info_lines = [
            f"场景: {scene_type} ({confidence:.1%})",
            f"决策: {decision}",
            f"飞行时间: {flight_time:.1f}s",
            f"电池: {battery:.1f}%",
            f"帧数: {frame_count}"
        ]

        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25

        # 显示图像
        cv2.imshow('无人机视觉导航', frame)

        # 检查按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 q 退出
            break
        elif key == ord('s'):  # 按 s 保存图像
            filename = f"data/images/capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ 保存图像: {filename}")

        # 检查电池
        if battery <= 0:
            print("\n⚠️  电池耗尽！紧急降落...")
            break

        # 限制运行时间（可选）
        if flight_time > 60:  # 60秒后自动停止
            print("\n⏰ 飞行时间到，安全降落...")
            break

    # 清理
    drone_cam.release()
    cv2.destroyAllWindows()

    # 显示统计信息
    print("\n" + "=" * 50)
    print("飞行统计:")
    print(f"- 总飞行时间: {flight_time:.1f} 秒")
    print(f"- 处理帧数: {frame_count}")
    print(f"- 平均帧率: {frame_count / flight_time:.1f} FPS" if flight_time > 0 else "- 平均帧率: N/A")
    print(f"- 最终电池: {battery:.1f}%")
    print("=" * 50)

    print("\n飞行结束！")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        exit_code = 0
    except Exception as e:
        print(f"\n程序出错: {e}")
        exit_code = 1

    input("\n按 Enter 键退出...")
    sys.exit(exit_code)