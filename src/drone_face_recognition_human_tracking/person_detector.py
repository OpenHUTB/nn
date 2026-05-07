# 人物检测模块（YOLOv8）
import cv2
import sys
import os
import numpy as np

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

try:
    from person_detector import PersonDetector

    print("✅ 成功导入 PersonDetector")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("💡 请确保 modules/person_detector.py 存在")
    sys.exit(1)


def test_with_camera():
    """使用摄像头测试人物检测"""
    print("🎥 正在打开摄像头...")

    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 0 = 默认摄像头

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        print("💡 尝试使用不同的摄像头索引: cv2.VideoCapture(1)")
        return

    # 初始化人物检测器
    print("🔄 正在加载YOLOv8模型...")
    detector = PersonDetector()

    print("✅ 开始检测，按 'q' 键退出")
    print("   's' 键: 保存当前帧")
    print("   'p' 键: 暂停/继续")

    paused = False

    while True:
        if not paused:
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头帧")
                break

            # 检测人物
            persons, annotated_frame = detector.detect(frame)

            # 显示结果
            cv2.putText(annotated_frame, f'Persons: {len(persons)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示每个检测到的人的信息
            for i, person in enumerate(persons):
                bbox = person['bbox']
                confidence = person['confidence']
                cv2.putText(annotated_frame, f'Person {i + 1}: {confidence:.2f}',
                            (10, 70 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 在控制台输出详细信息（每10帧输出一次）
                if cv2.getTickCount() % 10 == 0:
                    print(f"👤 Person {i + 1}: bbox={bbox}, confidence={confidence:.2f}")

            # 显示图像
            cv2.imshow('Person Detection - YOLOv8', annotated_frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # 退出
            break
        elif key == ord('p'):  # 暂停/继续
            paused = not paused
            print(f"{'⏸️ 暂停' if paused else '▶️ 继续'}")
        elif key == ord('s'):  # 保存图像
            timestamp = cv2.getTickCount()
            filename = f'detection_{timestamp}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"💾 已保存到: {filename}")

    # 清理
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 测试完成")


def test_with_image(image_path=None):
    """使用图片测试人物检测"""
    if image_path is None or not os.path.exists(image_path):
        print("📸 未提供有效图片路径，使用测试图片...")
        # 创建一个简单的测试图片
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        # 画一个"人物"（简单的矩形）
        cv2.rectangle(image, (200, 100), (400, 300), (255, 255, 255), -1)
        cv2.putText(image, "Test Person", (220, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        image_path = "test_generated.jpg"
        cv2.imwrite(image_path, image)
        print(f"🖼️ 创建测试图片: {image_path}")

    print(f"🖼️ 正在测试图片: {image_path}")

    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    # 初始化人物检测器
    print("🔄 正在加载YOLOv8模型...")
    detector = PersonDetector()

    # 检测人物
    persons, annotated_frame = detector.detect(frame)

    print(f"✅ 检测到 {len(persons)} 个人物")

    # 显示每个检测到的人的信息
    for i, person in enumerate(persons):
        bbox = person['bbox']
        confidence = person['confidence']
        print(f"👤 Person {i + 1}:")
        print(f"   Bounding Box: {bbox}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Class: {person['class_name']}")

        # 在图像上标注
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Person {i + 1}: {confidence:.2f}',
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Person Detection Result', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    output_path = 'detection_result.jpg'
    cv2.imwrite(output_path, annotated_frame)
    print(f"💾 结果已保存到: {output_path}")

    return len(persons)


def test_yolov8_model():
    """测试YOLOv8模型加载和基本功能"""
    print("🧪 测试YOLOv8模型加载...")

    try:
        # 创建检测器（这会下载模型）
        print("📥 创建PersonDetector实例...")
        detector = PersonDetector()
        print("✅ PersonDetector创建成功")

        # 创建一个简单的测试图像
        print("📸 创建测试图像...")
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # 蓝色背景
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)  # 白色矩形作为"人物"

        # 测试检测
        print("🔍 进行人物检测...")
        persons, result = detector.detect(test_image)

        print(f"✅ 检测完成")
        print(f"   图像大小: {test_image.shape}")
        print(f"   检测到人物数量: {len(persons)}")

        if persons:
            for i, person in enumerate(persons):
                print(f"   Person {i + 1}: bbox={person['bbox']}, conf={person['confidence']:.2f}")
        else:
            print("   ℹ️  未检测到人物（正常，因为是简单测试图像）")

        # 显示结果
        cv2.imshow('Model Test Result', result)
        cv2.waitKey(1000)  # 显示1秒
        cv2.destroyAllWindows()

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_test():
    """快速测试（无需用户交互）"""
    print("⚡ 快速测试模式")
    print("1. 测试模型加载...")
    if test_yolov8_model():
        print("✅ 模型测试通过")

        print("\n2. 测试图片检测...")
        num_persons = test_with_image()
        print(f"✅ 图片检测完成，找到 {num_persons} 个人物")

        print("\n3. 是否测试摄像头? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            print("\n🎥 开始摄像头测试...")
            test_with_camera()
        else:
            print("📊 快速测试完成！")
    else:
        print("❌ 模型测试失败")


def main():
    """主函数"""
    print("=" * 50)
    print("🧪 YOLOv8 人物检测模块测试")
    print("=" * 50)

    print("\n选择测试模式:")
    print("1. 使用摄像头实时检测")
    print("2. 使用图片文件测试")
    print("3. 测试模型加载和推理")
    print("4. 快速测试（自动流程）")
    print("5. 退出")

    try:
        choice = input("\n请输入选择 (1-5): ").strip()

        if choice == "1":
            test_with_camera()
        elif choice == "2":
            image_path = input("请输入图片路径 (直接回车使用测试图片): ").strip()
            if not image_path:
                image_path = None
            test_with_image(image_path)
        elif choice == "3":
            test_yolov8_model()
        elif choice == "4":
            quick_test()
        elif choice == "5":
            print("👋 退出测试")
        else:
            print("❌ 无效选择，默认运行快速测试")
            quick_test()

    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    except Exception as e:
        print(f"❌ 运行出错: {e}")


if __name__ == "__main__":
    main()