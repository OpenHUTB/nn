# 人脸检测模块
import cv2
import sys
import os
import numpy as np

# ========== 设置路径 ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"📁 当前目录: {current_dir}")

# 添加modules目录
modules_dir = os.path.join(current_dir, 'modules')
if os.path.exists(modules_dir):
    sys.path.insert(0, modules_dir)
    print(f"✅ 添加modules路径: {modules_dir}")

# ========== 导入模块 ==========
print("\n🔄 导入人脸检测模块...")
try:
    from face_detector import FaceDetector

    print("✅ 成功导入 FaceDetector")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("💡 正在创建简单的FaceDetector类用于测试...")


    # 创建简单的FaceDetector类
    class FaceDetector:
        def __init__(self, model_path=None):
            print("🔄 初始化简单人脸检测器")
            # 使用OpenCV的Haar级联分类器
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if self.face_cascade.empty():
                print("⚠️  Haar级联分类器加载失败")

        def detect(self, frame, min_face_size=30):
            if frame is None:
                return []

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size)
            )

            return [(x, y, w, h) for (x, y, w, h) in faces]


# ========== 测试函数 ==========
def test_with_webcam():
    """使用摄像头测试人脸检测"""
    print("\n🎥 摄像头人脸检测测试")
    print("-" * 40)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        print("💡 尝试使用不同的摄像头索引...")
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ 找到摄像头: 索引 {i}")
                break

    if not cap.isOpened():
        print("❌ 所有摄像头都不可用，使用测试图像")
        return test_with_images()

    # 创建人脸检测器
    print("🔄 初始化人脸检测器...")
    detector = FaceDetector()

    print("✅ 开始检测，按 'q' 键退出")
    print("   's' 键: 保存当前帧")

    face_count_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头帧")
            break

        # 调整图像大小以提高性能
        frame_small = cv2.resize(frame, (640, 480))

        # 检测人脸
        faces = detector.detect(frame_small)

        # 更新历史记录
        face_count_history.append(len(faces))
        if len(face_count_history) > 30:  # 保留最近30帧
            face_count_history.pop(0)

        # 绘制人脸框
        for (x, y, w, h) in faces:
            # 由于图像被缩小，需要调整坐标
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            x_orig = int(x * scale_x)
            y_orig = int(y * scale_y)
            w_orig = int(w * scale_x)
            h_orig = int(h * scale_y)

            # 绘制矩形
            cv2.rectangle(frame, (x_orig, y_orig),
                          (x_orig + w_orig, y_orig + h_orig),
                          (0, 255, 0), 2)

            # 添加标签
            cv2.putText(frame, 'Face', (x_orig, y_orig - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示统计信息
        avg_faces = np.mean(face_count_history) if face_count_history else 0
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Avg: {avg_faces:.1f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 显示帧率
        cv2.putText(frame, f'Press "q" to quit', (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示图像
        cv2.imshow('Face Detection - Webcam', frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存图像
            timestamp = cv2.getTickCount()
            filename = f'face_detection_{timestamp}.jpg'
            cv2.imwrite(filename, frame)
            print(f"💾 保存图像: {filename}")
        elif key == ord('f'):
            # 切换全屏
            cv2.setWindowProperty('Face Detection - Webcam',
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

    # 清理
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 摄像头测试完成")


def test_with_images():
    """使用图片测试人脸检测"""
    print("\n🖼️ 图片人脸检测测试")
    print("-" * 40)

    # 创建测试图片
    test_images = []

    # 1. 生成带人脸的测试图片
    print("📸 创建测试图片...")

    # 图片1: 单人脸
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    # 绘制一个简单的"人脸"（椭圆）
    cv2.ellipse(img1, (150, 150), (80, 100), 0, 0, 360, (255, 255, 255), -1)
    # 眼睛
    cv2.circle(img1, (120, 120), 15, (0, 0, 0), -1)
    cv2.circle(img1, (180, 120), 15, (0, 0, 0), -1)
    # 嘴巴
    cv2.ellipse(img1, (150, 190), (40, 20), 0, 0, 180, (0, 0, 0), 3)
    test_images.append(("生成的人脸", img1))

    # 图片2: 多个人脸
    img2 = np.zeros((400, 600, 3), dtype=np.uint8)
    # 第一个脸
    cv2.ellipse(img2, (150, 200), (70, 90), 0, 0, 360, (200, 200, 200), -1)
    cv2.circle(img2, (130, 170), 12, (0, 0, 0), -1)
    cv2.circle(img2, (170, 170), 12, (0, 0, 0), -1)
    # 第二个脸
    cv2.ellipse(img2, (450, 200), (70, 90), 0, 0, 360, (200, 200, 200), -1)
    cv2.circle(img2, (430, 170), 12, (0, 0, 0), -1)
    cv2.circle(img2, (470, 170), 12, (0, 0, 0), -1)
    test_images.append(("双人脸", img2))

    # 尝试加载真实图片（如果有）
    test_files = ['test.jpg', 'face.jpg', 'person.jpg', 'example.jpg']
    for file in test_files:
        if os.path.exists(file):
            img = cv2.imread(file)
            if img is not None:
                test_images.append((f"文件: {file}", img))
                print(f"✅ 加载图片: {file}")

    if len(test_images) == 0:
        print("⚠️  没有找到测试图片")
        return

    # 创建人脸检测器
    print("🔄 初始化人脸检测器...")
    detector = FaceDetector()

    # 测试每张图片
    for name, img in test_images:
        print(f"\n🔍 测试: {name}")
        print(f"   图片大小: {img.shape}")

        # 检测人脸
        faces = detector.detect(img, min_face_size=20)

        print(f"   检测到人脸数: {len(faces)}")

        # 绘制结果
        result = img.copy()
        for i, (x, y, w, h) in enumerate(faces):
            print(f"     人脸{i + 1}: 位置({x}, {y}), 大小({w}x{h})")

            # 绘制边界框
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f'Face {i + 1}', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 绘制人脸中心点
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(result, (center_x, center_y), 3, (0, 0, 255), -1)

        # 显示原图和结果
        cv2.imshow(f'Original: {name}', img)
        cv2.imshow(f'Result: {name}', result)

        # 保存结果
        if len(faces) > 0:
            filename = f'face_result_{name}.jpg'.replace(':', '_').replace(' ', '_')
            cv2.imwrite(filename, result)
            print(f"💾 保存结果: {filename}")

        cv2.waitKey(2000)  # 显示2秒
        cv2.destroyAllWindows()

    print("\n✅ 图片测试完成")


def test_detection_performance():
    """测试检测性能"""
    print("\n📊 人脸检测性能测试")
    print("-" * 40)

    # 创建测试图像
    test_sizes = [(320, 240), (640, 480), (800, 600), (1024, 768)]

    detector = FaceDetector()

    for width, height in test_sizes:
        print(f"\n📏 测试图像大小: {width}x{height}")

        # 创建测试图像
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # 添加一些"人脸"
        num_faces = np.random.randint(1, 4)
        for _ in range(num_faces):
            x = np.random.randint(50, width - 50)
            y = np.random.randint(50, height - 50)
            w = np.random.randint(40, 100)
            h = np.random.randint(50, 120)
            cv2.ellipse(img, (x, y), (w // 2, h // 2), 0, 0, 360,
                        (np.random.randint(200, 255),
                         np.random.randint(200, 255),
                         np.random.randint(200, 255)), -1)

        # 测试检测时间
        import time
        start_time = time.time()

        faces = detector.detect(img)

        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0

        print(f"   检测时间: {elapsed * 1000:.2f} ms")
        print(f"   帧率: {fps:.1f} FPS")
        print(f"   检测到人脸: {len(faces)}")

    print("\n✅ 性能测试完成")


def simple_demo():
    """简单演示模式"""
    print("=" * 60)
    print("🧪 人脸检测模块测试")
    print("=" * 60)

    print("\n选择测试模式:")
    print("1. 🎥 摄像头实时检测")
    print("2. 🖼️ 图片文件检测")
    print("3. 📊 性能测试")
    print("4. 🚀 全部测试")
    print("5. ❌ 退出")

    try:
        choice = input("\n请输入选择 (1-5): ").strip()

        if choice == "1":
            test_with_webcam()
        elif choice == "2":
            test_with_images()
        elif choice == "3":
            test_detection_performance()
        elif choice == "4":
            print("🚀 运行全部测试...")
            test_with_images()
            test_detection_performance()
            if input("\n继续摄像头测试? (y/n): ").lower() == 'y':
                test_with_webcam()
        elif choice == "5":
            print("👋 退出")
        else:
            print("⚠️  无效选择，运行简单演示")
            test_with_webcam()

    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


def quick_test():
    """快速测试（无需交互）"""
    print("⚡ 快速测试人脸检测模块")

    # 基本检查
    print("1. ✅ 检查OpenCV...")
    print(f"   OpenCV版本: {cv2.__version__}")

    print("2. ✅ 检查人脸检测器...")
    detector = FaceDetector()
    print("   人脸检测器初始化成功")

    print("3. ✅ 测试简单图像...")
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    test_img[50:150, 50:150] = [255, 255, 255]  # 白色方块

    faces = detector.detect(test_img)
    print(f"   检测结果: {len(faces)} 个人脸")

    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            print(f"     人脸{i + 1}: ({x}, {y}, {w}, {h})")

    print("\n✅ 快速测试完成!")

    # 询问是否进行更多测试
    response = input("\n是否进行摄像头测试? (y/n): ").strip().lower()
    if response == 'y':
        test_with_webcam()


# ========== 主函数 ==========
if __name__ == "__main__":
    print("🚀 人脸检测模块测试脚本")
    print(f"工作目录: {os.getcwd()}")

    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == 'quick':
            quick_test()
        elif sys.argv[1] == 'cam':
            test_with_webcam()
        elif sys.argv[1] == 'img':
            test_with_images()
        else:
            simple_demo()
    else:
        # 默认运行交互式演示
        simple_demo()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)