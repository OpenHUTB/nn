"""
无人机视觉导航系统 - 简化版
可以立即运行测试
"""

import os
import sys
import time
import cv2
import numpy as np

# 添加项目路径，让Python能找到src模块（如果需要）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def ensure_directories():
    """确保所有目录都存在（用于存储数据）"""
    # 定义需要创建的目录列表
    dirs = ['data/images', 'data/videos', 'data/logs', 'data/config']
    for d in dirs:
        # 如果目录不存在则创建
        os.makedirs(d, exist_ok=True)
        print(f"✓ 目录已就绪: {d}")


class SimpleDroneCamera:
    """简单的无人机摄像头类 - 模拟或真实摄像头"""

    def __init__(self, camera_id=0):
        """
        初始化摄像头
        Args:
            camera_id: 摄像头ID，0通常表示默认摄像头
        """
        self.camera_id = camera_id
        self.cap = None  # OpenCV视频捕获对象

    def open(self):
        """打开摄像头"""
        print(f"尝试打开摄像头 {self.camera_id}...")
        # 尝试打开指定ID的摄像头
        self.cap = cv2.VideoCapture(self.camera_id)

        # 检查摄像头是否成功打开
        if not self.cap.isOpened():
            print("⚠️  无法打开物理摄像头，使用模拟模式")
            return False  # 打开失败
        else:
            print("✓ 摄像头已连接")
            return True  # 打开成功

    def read_frame(self):
        """读取一帧图像"""
        # 如果有摄像头且已打开，尝试读取帧
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:  # 读取成功
                return frame

        # 如果没有摄像头或读取失败，返回模拟图像
        return self.simulate_frame()

    def simulate_frame(self):
        """生成模拟图像（在没有摄像头时使用）"""
        # 创建黑色背景图像
        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 在图像上添加一些图形作为模拟内容
        cv2.putText(frame, "无人机模拟视图", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
        cv2.circle(frame, (400, 200), 50, (0, 0, 255), -1)

        return frame

    def release(self):
        """释放摄像头资源"""
        if self.cap:
            self.cap.release()
            print("摄像头已释放")


def analyze_scene_simple(frame):
    """简单场景分析（基于颜色特征）"""
    # 检查输入帧是否有效
    if frame is None:
        return "未知", 0.5

    # 将BGR颜色空间转换为HSV（色调、饱和度、明度）便于颜色分析
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 检测绿色（植被）范围
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    # 计算绿色像素占比
    green_pct = np.sum(green_mask > 0) / green_mask.size

    # 检测蓝色（水域）范围
    blue_mask = cv2.inRange(hsv, (100, 40, 40), (140, 255, 255))
    # 计算蓝色像素占比
    blue_pct = np.sum(blue_mask > 0) / blue_mask.size

    # 根据颜色占比判断场景类型
    if green_pct > 0.3:  # 绿色占比超过30%认为是森林/草地
        return "森林/草地", green_pct
    elif blue_pct > 0.2:  # 蓝色占比超过20%认为是水域
        return "水域", blue_pct
    else:  # 其他情况认为是城市/建筑区域
        return "城市/建筑", max(green_pct, blue_pct)


def get_decision(scene_type, confidence):
    """根据场景类型做出飞行决策"""
    # 决策映射表：不同场景类型对应的飞行决策
    decisions = {
        "森林/草地": "✓ 安全区域，继续飞行",
        "水域": "⚠️  接近水域，提高飞行高度",
        "城市/建筑": "⚠️  城市区域，降低速度并避让",
        "未知": "? 无法识别，保持警戒"
    }
    # 获取对应决策，如果没有匹配则返回默认决策
    return decisions.get(scene_type, "保持当前状态")


def main():
    """主函数 - 无人机视觉导航系统主循环"""
    # 打印程序标题和说明
    print("=" * 50)
    print("无人机视觉导航系统")
    print("版本: 1.0.0")
    print("按 'q' 键退出，按 's' 键保存图像")
    print("=" * 50)

    # 确保所需目录存在
    ensure_directories()

    # 创建无人机摄像头对象
    drone_cam = SimpleDroneCamera(camera_id=0)
    drone_cam.open()  # 打开摄像头

    # 初始化飞行状态变量
    battery = 100  # 电池电量（百分比）
    flight_time = 0  # 飞行时间（秒）
    frame_count = 0  # 已处理帧数
    start_time = time.time()  # 飞行开始时间

    print("\n开始飞行...")

    # 主循环：处理每一帧图像
    while True:
        # 1. 读取当前帧
        frame = drone_cam.read_frame()
        frame_count += 1  # 帧数计数器递增

        # 2. 分析场景类型
        scene_type, confidence = analyze_scene_simple(frame)

        # 3. 根据场景类型做出决策
        decision = get_decision(scene_type, confidence)

        # 4. 更新飞行状态
        flight_time = time.time() - start_time  # 计算已飞行时间
        battery = max(0, battery - 0.05)  # 模拟电池消耗（每帧减少0.05%）

        # 5. 在图像上显示状态信息
        info_lines = [
            f"场景: {scene_type} ({confidence:.1%})",
            f"决策: {decision}",
            f"飞行时间: {flight_time:.1f}s",
            f"电池: {battery:.1f}%",
            f"帧数: {frame_count}"
        ]

        # 逐行绘制状态信息到图像上
        y_offset = 30  # 文字起始y坐标
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25  # 每行文字间隔25像素

        # 6. 显示处理后的图像
        cv2.imshow('无人机视觉导航', frame)

        # 7. 检查键盘输入
        key = cv2.waitKey(1) & 0xFF  # 等待1毫秒并获取按键
        if key == ord('q'):  # 按 'q' 键退出程序
            print("\n用户请求退出...")
            break
        elif key == ord('s'):  # 按 's' 键保存当前图像
            filename = f"data/images/capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ 保存图像: {filename}")

        # 8. 检查电池电量（安全条件）
        if battery <= 0:
            print("\n⚠️  电池耗尽！紧急降落...")
            break

        # 9. 检查飞行时间限制（可选安全限制）
        if flight_time > 60:  # 运行60秒后自动停止（演示用途）
            print("\n⏰ 飞行时间到，安全降落...")
            break

    # 10. 清理资源
    drone_cam.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

    # 11. 打印飞行统计信息
    print("\n" + "=" * 50)
    print("飞行统计:")
    print(f"- 总飞行时间: {flight_time:.1f} 秒")
    print(f"- 处理帧数: {frame_count}")
    # 计算平均帧率（FPS）
    if flight_time > 0:
        print(f"- 平均帧率: {frame_count / flight_time:.1f} FPS")
    else:
        print("- 平均帧率: N/A")
    print(f"- 最终电池: {battery:.1f}%")
    print("=" * 50)

    print("\n飞行结束！")
    return 0  # 返回成功状态码


# Python标准入口点
if __name__ == "__main__":
    try:
        exit_code = main()  # 运行主函数
    except KeyboardInterrupt:  # 处理Ctrl+C中断
        print("\n程序被用户中断")
        exit_code = 0
    except Exception as e:  # 处理其他异常
        print(f"\n程序出错: {e}")
        exit_code = 1

    # 等待用户确认退出（便于查看输出结果）
    input("\n按 Enter 键退出...")
    sys.exit(exit_code)  # 退出程序